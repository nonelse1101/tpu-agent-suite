#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The sharding sweep: enumerate the space, cost every sharding at ONE operating point.

    python3 search/sweep.py presets/<cell>.json --out <runs>/<run_id>/artifacts/sharding-strategy/<cell>

Where the work is done
======================
    ../roofline-analysis/roofline.py        shape -> per-op ms          the arithmetic
    ../roofline-analysis/serving.py         load  -> TTFT, TPOT, tok/s  the scheduler
    search/space.py                         what the code permits       legality
    this file                               enumerate x cost            the sweep
    search/report.py                        group, render, chart        the report

Nothing here computes a time or a byte. Every number in every CSV comes back from
`serving.operating_point`, which gets it from `roofline.py`. This file decides *which
shardings to ask about* and writes the answers down -- if arithmetic appears below then
the seam has moved and the report has two provenances again.

One operating point per sharding
================================
The cell pins `concurrency`, `max_num_batched_tokens` and `max_num_seqs` to one value
each, so every sharding is priced at the same offered load with the same knobs. That is
what makes the combined chart like-for-like: two points differ by the sharding and
nothing else. The concurrency and mnbt sweeps were removed 2026-09-24 and will come
back as their own report sections; until then this file refuses a list.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(
    0, os.path.normpath(os.path.join(_HERE, "..", "..", "roofline-analysis")))

import roofline as R  # noqa: E402
import serving as SV  # noqa: E402
import space as SP  # noqa: E402

# ============================================================ the cell


@dataclass(frozen=True)
class Spec:
    """A **cell**: the deployment question this sweep answers.

    The architecture pins are NOT here. They live in the roofline config this points
    at, and are read from it -- one checkpoint, one place, so a re-pin cannot leave a
    stale copy behind in a preset nobody diffs.
    """
    name: str
    roofline_config: str  # path, resolved relative to the preset
    topology: SP.Topology
    isl: int
    osl: int
    concurrency: int  # fleet-wide, what --max-concurrency takes
    max_num_batched_tokens: int  # per DP engine
    max_num_seqs: int  # per DP engine
    dispatch_bound: SP.DispatchBound
    # TODO(correctness): a non-unified role is not refused. `space.py` filters its
    # legality correctly (PCP is refused on decode workers), and then the serving model
    # costs it with a model that describes neither half of a P/D pair.
    role: str = "unified"
    source: str = ""

    @property
    def workload(self) -> SV.Workload:
        return SV.Workload(self.isl, self.osl)

    @property
    def knobs(self) -> SV.Knobs:
        return SV.Knobs(self.max_num_batched_tokens, self.max_num_seqs)


def _scalar(raw: dict, key: str) -> int:
    v = raw[key]
    if isinstance(v, list):
        raise SystemExit(
            f"{key} is a list ({v}); the sweep pins one operating point per cell. "
            f"Concurrency and mnbt sweeps are not implemented in this report yet."
        )
    return int(v)


def load_spec(path: str) -> Spec:
    raw = json.load(open(path))
    base = os.path.dirname(os.path.abspath(path))
    if "roofline_config" not in raw:
        raise SystemExit(
            f"{path}: no `roofline_config` -- this preset predates the "
            f"pins/cell split and restates architecture pins. Rewrite it "
            f"against presets/GLM-5.2-FP8-2x2x2-8k1k.json.")
    db = raw.get("dispatch_bound", {})
    return Spec(
        name=raw.get("name") or os.path.basename(path).removesuffix(".json"),
        roofline_config=os.path.normpath(
            os.path.join(base, raw["roofline_config"])),
        topology=SP.Topology.parse(raw["topology"], d2d=raw.get("d2d", True)),
        isl=int(raw["workload"]["isl"]),
        osl=int(raw["workload"]["osl"]),
        concurrency=_scalar(raw, "concurrency"),
        max_num_batched_tokens=_scalar(raw, "max_num_batched_tokens"),
        max_num_seqs=_scalar(raw, "max_num_seqs"),
        dispatch_bound=SP.DispatchBound(enabled=bool(db.get("enabled", False)),
                                        topk=int(db.get("topk", 0)),
                                        max_dispatch=int(
                                            db.get("max_dispatch", 0)),
                                        note=db.get("note", "")),
        role=raw.get("role", "unified"),
        source=path,
    )


def model_facts(cfg: R.Config) -> SP.ModelFacts:
    """Legality facts, DERIVED from the roofline config rather than restated."""
    m = cfg.model
    return SP.ModelFacts(
        name=m.name,
        is_moe=m.n_experts > 0 and m.moe_layers > 0,
        num_routed_experts=m.n_experts,
        top_k=m.topk,
        num_kv_heads=max(1, m.kv_heads),
        use_mla=m.use_mla,
        num_layers=m.layers,
        num_q_heads=m.heads,
    )


def as_roofline_sharding(s: SP.Sharding) -> R.Sharding:
    """`space.Sharding` -> `roofline.Sharding`. Two dataclasses, one concept: `space`
    owns what the code permits, `roofline` owns what it costs, and a blocked sharding
    has to survive the first to be priced by the second."""
    return R.Sharding(tp=s.tp, dp=s.dp, pp=s.pp, pcp=s.pcp, dcp=s.dcp, ep=s.ep)


# ============================================================ the sweep


@dataclass
class Row:
    """One sharding, costed at the cell's operating point."""
    sharding: SP.Sharding
    verdict: SP.Verdict
    over_cap: bool  # breaches the MoE dispatch cap at this mnbt
    point: SV.Point
    cap_reason: str = ""

    @property
    def cls(self) -> str:
        return self.verdict.blocking_class

    @property
    def shippable(self) -> bool:
        return self.cls == SP.SHIPPABLE and not self.over_cap


@dataclass
class Result:
    spec: Spec
    base: R.Config
    facts: SP.ModelFacts
    model: int
    enumerated: list[tuple[SP.Sharding, SP.Verdict]]
    rows: list[Row]
    funnel: list[dict[str, Any]]

    def config_for(self, s: SP.Sharding) -> R.Config:
        return replace(self.base, sharding=as_roofline_sharding(s))


def run(spec: Spec, *, model: int = SV.DEFAULT_MODEL) -> Result:
    """Every coherent sharding, blocked ones included, at one operating point.

    `hardware` and `infra` rows are priced so the report can say what lifting the rule
    would be worth; `arithmetic` rows never are -- there are no ranks to shard onto.
    """
    base = R.load(spec.roofline_config)
    # The CELL is the authority on the slice; an override is announced, because a
    # silent reshape is exactly the kind of thing found three reports later.
    was = base.hardware
    hw = replace(was,
                 slice_dims=spec.topology.slice_dims,
                 d2d=spec.topology.d2d)
    if (tuple(was.slice_dims), was.d2d) != (tuple(hw.slice_dims), hw.d2d):
        print(
            f"note: the cell's topology {spec.topology.name} "
            f"({spec.topology.devices} cores) overrides the roofline config's "
            f"{'x'.join(map(str, was.slice_dims))} ({was.world} cores).")
    base = replace(base, hardware=hw)
    if hw.world != spec.topology.devices:
        raise SystemExit(
            f"internal: hardware.world {hw.world} != topology.devices "
            f"{spec.topology.devices} for {spec.topology.name}")
    facts = model_facts(base)
    enumerated = SP.enumerate_shardings(spec.topology,
                                        facts,
                                        include_pp=True,
                                        role=spec.role)
    rows = []
    for s, v in enumerated:
        if not v.costable:
            continue
        cfg = SP.ServerConfig(
            s,
            max_num_seqs=spec.max_num_seqs,
            max_num_batched_tokens=spec.max_num_batched_tokens)
        cap = spec.dispatch_bound.violation(cfg) or ""
        # `force` because `Sharding.check` refuses `infra` rows too, and pricing them
        # is the point. Never forced for `arithmetic`: `costable` is False there.
        p = SV.operating_point(replace(base, sharding=as_roofline_sharding(s)),
                               spec.workload,
                               spec.knobs,
                               spec.concurrency,
                               model,
                               force=v.costable and not v.ok)
        rows.append(Row(s, v, bool(cap), p, cap))
    return Result(spec, base, facts, model, enumerated, rows,
                  _funnel(spec, enumerated, rows))


def _funnel(spec: Spec, enumerated: list[tuple[SP.Sharding, SP.Verdict]],
            rows: list[Row]) -> list[dict[str, Any]]:
    """One row per stage. Every enumerated sharding lands in exactly one bucket, and
    the assert says so -- a funnel whose arithmetic does not close is an accounting of
    exclusions that is itself hiding one."""
    cls = Counter(v.blocking_class for _, v in enumerated)
    fits = [r for r in rows if r.point.feasible]
    nofit = [r for r in rows if not r.point.feasible]
    ship = [r for r in fits if r.shippable]
    blocked = [r for r in fits if not r.shippable]
    assert cls[SP.ARITHMETIC] + len(nofit) + len(ship) + len(blocked) == len(
        enumerated)
    return [
        {
            "stage": "enumerated",
            "count": len(enumerated),
            "removed": 0,
            "class": "",
            "kind": "",
            "note": f"every (tp, dp, pcp, dcp, pp, ep) on {spec.topology.name}"
        },
        {
            "stage": "coherent",
            "count": len(rows),
            "removed": cls[SP.ARITHMETIC],
            "class": SP.ARITHMETIC,
            "kind": "not an exclusion",
            "note": "the partition does not exist -- no ranks to shard onto"
        },
        {
            "stage":
            "fits in HBM",
            "count":
            len(fits),
            "removed":
            len(nofit),
            "class":
            SP.MEMORY,
            "kind":
            "verdict",
            "note":
            "; ".join(sorted(r.sharding.label for r in nofit)[:4]) or "none"
        },
        {
            "stage": "shippable today",
            "count": len(ship),
            "removed": len(blocked),
            "class": f"{SP.HARDWARE} / {SP.INFRA} / {SP.DISPATCH}",
            "kind": "omission, priced",
            "note": f"{len(blocked)} priced but refused by the stack today"
        },
    ]


# ============================================================ output


def point_row(r: Row) -> dict[str, Any]:
    p = r.point
    s = r.sharding
    return {
        "sharding": s.label,
        "tp": s.tp,
        "dp": s.dp,
        "pp": s.pp,
        "pcp": s.pcp,
        "dcp": s.dcp,
        "ep": int(s.ep),
        "class": r.cls,
        "shippable": int(r.shippable and p.feasible),
        "over_cap": int(r.over_cap),
        "feasible": int(p.feasible),
        "reason": p.reason,
        "concurrency": p.concurrency,
        "running": round(p.running, 3),
        "gate": p.gate,
        "weights_gib": round(p.weights_gib, 3),
        "kv_pool_gib": round(p.kv_pool_gib, 3),
        "kv_cap_reqs": round(p.kv_cap_reqs, 2),
        "kv_used_gib": round(p.kv_used_gib, 3),
        "hbm_used_gib": round(p.hbm_used_gib, 3),
        "t_prefill_ms": round(p.ttft_ms - p.queue_ms, 3),
        "t_decode_ms": round(p.step_ms, 5),
        "tpot_ms": round(p.tpot_ms, 5),
        "queue_ms": round(p.queue_ms, 3),
        "ttft_ms": round(p.ttft_ms, 3),
        "e2e_ms": round(p.e2e_ms, 2),
        "output_tok_s": round(p.output_tok_s, 2),
        "total_tok_s": round(p.total_tok_s, 2),
        "req_s": round(p.req_s, 4),
        "n_chunks": p.n_chunks,
        "step_bound": p.step_bound,
        "ici_share": round(p.ici_share, 4),
        "pp_balance": round(p.pp_balance, 4),
        "model": p.model,
    }


COLUMNS = """# Raw data — `{out}`

Generated {when} · step model {model} · cell `{cell}` · C={conc}, mnbt={mnbt},
max-num-seqs={seqs}. Every number came from `serving.operating_point` over
`roofline_model{model}.py`. **Modelled.**

| file | rows | one row is |
| --- | ---: | --- |
| `points.csv` | {n_points} | one coherent sharding at the cell's operating point; infeasible rows kept with `reason` |
| `space.csv` | {n_space} | one enumerated sharding: every rule that refused it, its class |
| `funnel.csv` | {n_funnel} | one funnel stage |

## `points.csv` columns

| column | is |
| --- | --- |
| `class` / `shippable` | blocking class from `space.py`; 1 when runnable today and it fits |
| `over_cap` | breaches the MoE dispatch cap at this mnbt (only when the cell enables it) |
| `running` / `gate` | requests resident per engine, and which cap bound it: `offered` \\| `slots` \\| `kv` |
| `weights_gib` / `kv_pool_gib` / `kv_used_gib` / `hbm_used_gib` | per core; `used` is what the running set holds, not the reserved pool |
| `kv_cap_reqs` | requests the pool holds at full context — the `kv` gate |
| `t_prefill_ms` | one prompt's prefill, queueing excluded |
| `t_decode_ms` | one decode step. Under `pp>1` the **mean stage**; `tpot_ms` is the cycle |
| `ttft_ms` / `queue_ms` | `t_prefill + queue` (PP adds the last chunk's drain) |
| `output_tok_s` | fleet-wide, generated tokens only |
| `total_tok_s` | **total tokens throughput**, fleet-wide: `req_s × (ISL + OSL)`, input + output |
| `req_s` | fleet-wide completed requests per second |
| `step_bound` / `ici_share` | what bounds the decode step, and its collective share |
"""


def write_csvs(res: Result, out_dir: str) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    def dump(name: str, rows: list[dict[str, Any]]) -> None:
        p = os.path.join(out_dir, name)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        paths[name] = p

    pts = [point_row(r) for r in res.rows]
    dump("points.csv", pts)
    dump("space.csv", [{
        "sharding": s.label,
        "tp": s.tp,
        "dp": s.dp,
        "pcp": s.pcp,
        "dcp": s.dcp,
        "pp": s.pp,
        "ep": int(s.ep),
        "status": v.status,
        "class": v.blocking_class,
        "costable": int(v.costable),
        "rules": " | ".join(v.reasons),
    } for s, v in res.enumerated])
    dump("funnel.csv", res.funnel)
    sp = res.spec
    with open(os.path.join(out_dir, "columns.md"), "w") as f:
        f.write(
            COLUMNS.format(out=out_dir,
                           when=datetime.date.today().isoformat(),
                           model=res.model,
                           cell=sp.name,
                           conc=sp.concurrency,
                           mnbt=sp.max_num_batched_tokens,
                           seqs=sp.max_num_seqs,
                           n_points=len(pts),
                           n_space=len(res.enumerated),
                           n_funnel=len(res.funnel)))
    return paths


def summary(res: Result, top: int = 15) -> None:
    sp = res.spec
    print(
        f"\n{res.base.model.name}  {sp.topology.name}  ISL {sp.isl} / OSL {sp.osl}  "
        f"C={sp.concurrency}  mnbt={sp.max_num_batched_tokens}  "
        f"seqs={sp.max_num_seqs}  [model {res.model}]")
    for f in res.funnel:
        print(
            f"  {f['stage']:<18}{f['count']:>5}  (-{f['removed']} {f['class']})"
        )
    ok = sorted((r for r in res.rows if r.point.feasible),
                key=lambda r: -r.point.output_tok_s)
    print(f"\n  {'sharding':<26}{'class':<11}{'tok/s':>9}{'TTFT ms':>11}"
          f"{'T_dec ms':>10}{'gate':>9}")
    for r in ok[:top]:
        p = r.point
        print(f"  {r.sharding.label:<26}{r.cls:<11}{p.output_tok_s:>9.0f}"
              f"{p.ttft_ms:>11.1f}{p.step_ms:>10.2f}{p.gate:>9}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec")
    ap.add_argument("--out", help="directory for the CSVs")
    ap.add_argument("--roofline-model",
                    type=int,
                    choices=(1, 2),
                    default=SV.DEFAULT_MODEL)
    a = ap.parse_args()
    res = run(load_spec(a.spec), model=a.roofline_model)
    summary(res)
    if a.out:
        write_csvs(res, a.out)
        print(f"\nwrote {a.out}/{{points,space,funnel}}.csv, columns.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
