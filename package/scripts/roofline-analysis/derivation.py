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
"""One operating point, derived step by step -- the audit trail behind a serving number.

    python3 scripts/derivation.py <cfg>.json --concurrency 256 --mnbt 1024
    python3 scripts/derivation.py --selftest

`serving.operating_point` returns a number; this returns how it was reached: every
step as `formula`, `inputs`, `value`, recomputed here from the parts `roofline.py`
exposes, and checked against what `operating_point` itself returned. A step whose
recompute disagrees is a finding, and `ok` is False on it -- callers print it rather
than smooth it over.

What it hands back
==================
    weights       roofline.weight_terms   per tensor: whole-model, divisor, per core
    kv            roofline.kv_terms       per cache: bytes/token, divisor, per core
    decode        phase_detail            blocks, layer kinds, collectives
    prefill       phase_detail            the same, over every chunk
    steps         derive                  ranks -> memory -> admission -> T_decode,
                                          T_prefill -> QPS -> tok/s -> queue -> TTFT

Consumers (`sharding-strategy/search/report.py`) render this and do no arithmetic of
their own. **Model 1 only** -- it is the default step model and the only one whose
composition is short enough to write out; model 2's mixed step is an iteration.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roofline as R  # noqa: E402
import serving as SV  # noqa: E402

GiB = R.GiB

#: Op names -> the block a reader thinks in. README s3's rollup, in one place: a
#: 30-row op diff is not an explanation, and every consumer grouping on its own drifts.
BLOCKS = (
    ("collective", ("TP:", "MoE: Dispatch", "MoE: Collect",
                    "MoE: Router logits", "DCP:", "PCP:", "PP:")),
    ("MoE routed experts",
     ("MoE: FFi", "MoE: FFo", "MoE: SwiGLU", "MoE: permute", "MoE: unpermute",
      "MoE: combine", "MoE: Router (", "MoE: router")),
    ("MoE shared expert", ("MoE: Shared", )),
    ("attention", ("MLA:", "Attn:", "GDN:")),
    ("DSA indexer", ("DSA", )),
    ("dense FFN", ("Dense:", )),
    ("norms / residual", ("Norm:", )),
)


def block_of(name: str) -> str:
    for g, prefixes in BLOCKS:
        if any(name.startswith(p) for p in prefixes):
            return g
    return "other"


# ============================================================ one phase, opened up


@dataclass
class PhaseDetail:
    """One phase of one engine, per core. `total_ms` is the phase as model 1 prices it
    (mean PP stage), and every breakdown below sums to it."""
    name: str
    tokens: float
    kv_len: float
    passes: int
    total_ms: float
    blocks: dict[str, float]  # block -> ms, summed over layers and passes
    kinds: list[dict[str, Any]]  # layer kind -> count, ms/layer, all-layer ms
    ops: list[dict[str, Any]]  # op -> layers, ms, bound, block
    collectives: list[dict[str,
                           Any]]  # the ops with ICI time, with their comment
    bound_share: dict[str, float]  # HBM | MXU | VPU | ICI -> ms


def phase_detail(cfg: R.Config, ph: R.Phase) -> PhaseDetail:
    """`run_phase` and `layer_rows`, regrouped. No new arithmetic: every ms is an
    `Op.roofline_ms` times its multiplicity, exactly as `run_phase` sums it."""
    res = R.run_phase(cfg, ph)
    blocks: dict[str, float] = {}
    bound: dict[str, float] = {}
    ops = []
    for r in res.rows:
        if r["total_ms"] <= 0:
            continue
        b = block_of(r["op"])
        blocks[b] = blocks.get(b, 0.0) + r["total_ms"]
        bound[r["bound"]] = bound.get(r["bound"], 0.0) + r["total_ms"]
        ops.append({
            "op": r["op"],
            "block": b,
            "layers": r["layers"],
            "ms_per_layer": r["roofline_ms"],
            "total_ms": r["total_ms"],
            "bound": r["bound"],
            "mem_ms": r["mem_ms"],
            "comp_ms": r["comp_ms"],
            "ici_ms": r["ici_ms"]
        })
    lrows = R.layer_rows(cfg, ph)
    kinds = [{
        "kind": t["layer_kind"],
        "count": t["layer_count"],
        "ms_per_layer": t["roofline_ms"],
        "total_ms": t["layer_total_ms"]
    } for t in lrows if t["role"] == "total"]
    comments = {t["operator"]: t.get("comment", "") for t in lrows}
    colls = [
        dict(o, comment=comments.get(o["op"], "")) for o in ops
        if o["ici_ms"] > 0
    ]
    return PhaseDetail(ph.name, ph.tokens, ph.kv_len, len(R.chunks_of(ph)),
                       res.total_ms, blocks, kinds, ops, colls, bound)


# ============================================================ the derivation


@dataclass
class Step:
    key: str
    what: str
    formula: str
    inputs: str
    value: float
    unit: str
    check: float | None = None  # what operating_point returned, when it has it

    @property
    def ok(self) -> bool:
        if self.check is None:
            return True
        tol = 1e-6 * max(1.0, abs(self.check))
        return abs(self.value - self.check) <= tol


@dataclass
class Derivation:
    label: str
    point: SV.Point
    weights: list[R.WeightTerm]
    kv: list[R.KVTerm]
    decode: PhaseDetail | None
    prefill: PhaseDetail | None
    steps: list[Step] = field(default_factory=list)
    stage_weights_gib: list[float] = field(default_factory=list)  # pp > 1 only
    decode_stage_ms: list[float] = field(default_factory=list)  # pp > 1 only

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    def step(self, key: str) -> Step:
        return next(s for s in self.steps if s.key == key)


def derive(cfg: R.Config,
           w: SV.Workload,
           k: SV.Knobs,
           concurrency: int,
           *,
           force: bool = False) -> Derivation:
    """Model 1, written out. Mirrors `serving.operating_point` over
    `roofline_model1.solve` step for step, and checks each against it."""
    s, hw, m = cfg.sharding, cfg.hardware, cfg.model
    wcfg = replace(cfg, workload=R.Workload(isl=w.isl, osl=w.osl))
    p = SV.operating_point(cfg, w, k, concurrency, 1, force=force)
    d = Derivation(s.label, p, R.weight_terms(wcfg), R.kv_terms(wcfg), None,
                   None)
    if s.pp > 1:
        d.stage_weights_gib = [x / GiB for x in R.stage_weight_bytes(wcfg)]
    S = d.steps
    add = S.append

    add(
        Step("ranks", "cores per engine and engines",
             "world = TP * PCP * PP * DP",
             f"{s.tp} * {s.pcp} * {s.pp} * {s.dp}",
             float(s.tp * s.pcp * s.pp * s.dp), "cores", float(hw.world)))
    flat = sum(t.local_bytes for t in d.weights) / GiB
    if s.pp > 1:
        add(
            Step("weights", "weights per core, heaviest PP stage",
                 "max over stages of Σ its layers' terms + embedding ends",
                 " / ".join(f"{x:.2f}" for x in d.stage_weights_gib) + " GiB",
                 max(d.stage_weights_gib), "GiB", p.weights_gib or None))
    else:
        add(
            Step(
                "weights", "weights per core", "Σ_terms global / divisor",
                " + ".join(f"{t.local_bytes / GiB:.2f}"
                           for t in d.weights) + " GiB", flat, "GiB",
                p.weights_gib or None))
    cap = hw.hbm_capacity * hw.memory_utilization / GiB
    wt = S[-1].value
    add(
        Step(
            "kv_pool", "HBM left for the KV cache, per core",
            "HBM * mem_util - weights - resident overhead",
            f"{hw.hbm_capacity / GiB:.1f} * {hw.memory_utilization} - {wt:.2f} "
            f"- {hw.resident_overhead_gb:.1f}",
            cap - wt - hw.resident_overhead_gb, "GiB", p.kv_pool_gib or None))
    if not p.feasible and p.kv_pool_gib <= 0:
        return d
    kvt = R.kv_bytes_per_token(wcfg)
    kv_note = (" (heaviest stage)" if s.pp > 1 else "")
    add(
        Step(
            "kv_per_token", "KV bytes per token per core" + kv_note,
            "Σ_caches bytes/token/layer * layers / divisor"
            if s.pp <= 1 else "heaviest stage's layers only",
            " + ".join(f"{t.local_bytes_per_token / 1024:.2f}" for t in d.kv) +
            " KiB" if s.pp <= 1 else f"{kvt / 1024:.2f} KiB", kvt / 1024,
            "KiB"))
    per_req = kvt * (w.isl + w.osl) / GiB
    add(
        Step("kv_per_req", "KV per request at full context",
             "kv_per_token * (ISL + OSL)",
             f"{kvt / 1024:.2f} KiB * {w.isl + w.osl}", per_req, "GiB"))
    add(
        Step("kv_cap", "requests the pool holds (the kv gate)",
             "kv_pool / kv_per_req", f"{S[2].value:.3f} / {per_req:.5f}",
             S[2].value / per_req, "req", p.kv_cap_reqs or None))
    if not p.feasible:
        return d
    c = concurrency / s.dp
    gates = {"offered": c, "slots": float(k.max_num_seqs), "kv": S[-1].value}
    gate = min(gates, key=lambda g: gates[g])
    add(
        Step(
            "running", f"requests resident per engine (gate: {gate})",
            "min(C / DP, max_num_seqs, kv_cap)",
            f"min({concurrency}/{s.dp} = {c:g}, {k.max_num_seqs}, "
            f"{gates['kv']:.1f})", gates[gate], "req", p.running))

    running = gates[gate]
    full = w.isl + w.osl
    stages = max(1, s.pp)
    v = float(max(1, min(stages, int(max(1.0,
                                         running))))) if stages > 1 else 1.0
    db = max(1.0, running) / v
    dec_fetch = db * (min(full, m.indexer_topk) if m.indexer_topk else full)
    dec = R.Phase("decode",
                  tokens=db,
                  kv_len=full,
                  batch=max(1, int(round(db))),
                  kv_fetched=max(1.0, dec_fetch))
    pre = R.Phase("prefill",
                  tokens=w.isl,
                  kv_len=w.isl,
                  batch=1,
                  mnbt=k.max_num_batched_tokens)
    d.decode = phase_detail(wcfg, dec)
    d.prefill = phase_detail(wcfg, pre)

    if stages > 1:
        d.decode_stage_ms = R.stage_ms(wcfg,
                                       dec,
                                       bucket=k.max_num_batched_tokens or db)
        add(
            Step("t_decode", "decode, mean stage of one batch",
                 "Σ_ops op_ms * layers / PP",
                 f"{db:.1f} tokens, kv_len {full}", d.decode.total_ms, "ms",
                 p.step_ms))
        tpot = R.pp_cycle_ms(d.decode_stage_ms, v)
        add(
            Step(
                "tpot", "one decode cycle (TPOT)",
                "max(Σ stages, v * max stage)",
                f"stages {', '.join(f'{x:.2f}' for x in d.decode_stage_ms)}; v={v:g}",
                tpot, "ms", p.tpot_ms))
        chunk_st = [
            R.stage_ms(wcfg, ch, bucket=k.max_num_batched_tokens or ch.tokens)
            for ch in R.chunks_of(pre)
        ]
        t_pre = sum(max(x) for x in chunk_st)
        add(
            Step("t_prefill", "prefill, engine-busy time",
                 "Σ_chunks slowest stage", f"{len(chunk_st)} chunks", t_pre,
                 "ms"))
        ttft_core = R.pp_ttft_ms(chunk_st)
        add(
            Step("ttft_core", "prefill latency of one prompt",
                 "Σ_chunks slowest stage + drain of the last chunk",
                 f"{t_pre:.1f} + {sum(chunk_st[-1]) - max(chunk_st[-1]):.2f}",
                 ttft_core, "ms"))
    else:
        tpot = d.decode.total_ms
        add(
            Step("t_decode", "one decode step (= TPOT)",
                 "Σ_ops op_ms * layers", f"{db:.1f} tokens, kv_len {full}",
                 tpot, "ms", p.step_ms))
        t_pre = d.prefill.total_ms
        add(
            Step(
                "t_prefill", "one prompt, run alone, chunked at mnbt",
                "Σ_chunks Σ_ops op_ms * layers",
                f"{d.prefill.passes} chunks of {min(k.max_num_batched_tokens, w.isl)}",
                t_pre, "ms", p.ttft_ms - p.queue_ms))
        ttft_core = t_pre
    qp = 1e3 / t_pre
    qd = max(1.0, running) / (w.osl * tpot / 1e3)
    qc = 1.0 / (1.0 / qp + 1.0 / qd)
    add(
        Step("qps_prefill", "req/s if the engine only prefilled",
             "1000 / T_prefill", f"1000 / {t_pre:.2f}", qp, "req/s"))
    add(
        Step("qps_decode", "req/s if the engine only decoded",
             "running / (OSL * TPOT / 1000)",
             f"{max(1.0, running):.1f} / ({w.osl} * {tpot:.3f} / 1000)", qd,
             "req/s"))
    add(
        Step("qps", "req/s per engine -- it must do both, serially",
             "1 / (1/QPS_prefill + 1/QPS_decode)",
             f"1 / (1/{qp:.3f} + 1/{qd:.3f})", qc, "req/s", p.req_s / s.dp))
    add(
        Step("tok_s", "output tok/s, fleet", "QPS * OSL * DP",
             f"{qc:.4f} * {w.osl} * {s.dp}", qc * w.osl * s.dp, "tok/s",
             p.output_tok_s))
    add(
        Step("total_tok_s", "total tokens throughput (input + output), fleet",
             "QPS * (ISL + OSL) * DP", f"{qc:.4f} * {w.isl + w.osl} * {s.dp}",
             qc * (w.isl + w.osl) * s.dp, "tok/s", p.total_tok_s))
    queue = 1e3 * max(0.0, c - running) / qc
    add(
        Step("queue", "admission wait",
             "(C/DP - running) / QPS  (Little's Law)",
             f"max(0, {c:g} - {running:.1f}) / {qc:.4f}", queue, "ms",
             p.queue_ms))
    add(
        Step("ttft", "time to first token", "prefill latency + queue",
             f"{ttft_core:.1f} + {queue:.1f}", ttft_core + queue, "ms",
             p.ttft_ms))
    return d


# ============================================================ cli


def render(d: Derivation) -> None:
    print(f"\n=== {d.label}")
    for t in d.weights:
        print(
            f"  W {t.tensor:<40}{t.global_bytes / GiB:9.2f} GiB / {t.divisor_expr}"
            f"={t.divisor:<3}-> {t.local_bytes / GiB:8.3f} GiB")
    for t in d.kv:
        print(
            f"  KV {t.cache:<39}{t.global_bytes_per_token / 1024:9.2f} KiB/tok / "
            f"{t.divisor_expr}={t.divisor:<3}-> {t.local_bytes_per_token / 1024:8.3f}"
        )
    for s in d.steps:
        mark = "" if s.check is None else (
            "  ok" if s.ok else f"  MISMATCH {s.check}")
        print(
            f"  {s.key:<13}{s.value:14.4f} {s.unit:<6}{s.formula}   [{s.inputs}]{mark}"
        )


def selftest() -> int:
    """Every step that operating_point also returns must agree with it, on every
    legal sharding of the bundled 8K/1K cell -- including PP and EP rows."""
    here = os.path.dirname(os.path.abspath(__file__))
    base = R.load(os.path.join(here, "configs", "glm-5.2-fp8-8k1k-2x2x2.json"))
    w, k = SV.Workload(8192, 1024), SV.Knobs(1024, 256)
    fails, n, tried = [], 0, 0
    world = base.hardware.world
    for tp in (1, 2, 4, 8, 16):
        for pp in (1, 2, 16):
            for dcp in (1, 2):
                for ep in (False, True):
                    dp = world // (tp * pp)
                    if tp * pp * dp != world or tp % dcp:
                        continue
                    if pp > 1 and (tp > 1 or dp > 1):
                        continue
                    cfg = replace(base,
                                  sharding=R.Sharding(tp=tp,
                                                      dp=dp,
                                                      pp=pp,
                                                      dcp=dcp,
                                                      ep=ep))
                    tried += 1
                    d = derive(cfg, w, k, 256)
                    n += d.point.feasible
                    fails += [(d.label, s.key, s.value, s.check)
                              for s in d.steps if not s.ok]
                    # the weight terms ARE the aggregate
                    if cfg.sharding.pp == 1:
                        tot = sum(t.local_bytes for t in d.weights)
                        if abs(tot - R.weight_bytes(cfg)) > 1:
                            fails.append((d.label, "weight_terms", tot,
                                          R.weight_bytes(cfg)))
                        kvs = sum(t.local_bytes_per_token for t in d.kv)
                        if abs(kvs - R.kv_bytes_per_token(cfg)) > 1e-6:
                            fails.append((d.label, "kv_terms", kvs,
                                          R.kv_bytes_per_token(cfg)))
                    if d.decode:
                        bs = sum(d.decode.blocks.values())
                        ks = sum(x["total_ms"] for x in d.decode.kinds)
                        if abs(bs - d.decode.total_ms) > 1e-9 or \
                                abs(ks - d.decode.total_ms) > 1e-6:
                            fails.append((d.label, "decode rollup", bs, ks))
    for f in fails:
        print("  FAIL", f)
    print(f"  {tried} shardings, {n} feasible, {len(fails)} mismatches")
    if n == 0:
        print("  FAIL nothing feasible -- the test exercised nothing")
        return 1
    print("OK" if not fails else "FAILED")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", nargs="?")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--mnbt", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.config:
        ap.error("need a config path, or --selftest")
    cfg = R.load(a.config)
    wl = cfg.workload
    render(
        derive(cfg, SV.Workload(wl.isl, wl.osl),
               SV.Knobs(a.mnbt, a.max_num_seqs), a.concurrency))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
