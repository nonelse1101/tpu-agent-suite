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
"""The legal sharding space for a vllm-torchtpu deployment on a fixed TPU slice.

Enumeration is the cheap half of the search: the degrees have to multiply out to the
slice, and vLLM refuses most of what is left. This module encodes those refusals so a
config is *dropped with a reason* rather than silently costed.

The rules live in ``../CONSTRAINTS.md`` with a ``file:line`` for each. They are read out
of two checkouts that drift — re-verify before trusting a run of this. Nothing here
touches hardware; it is pure arithmetic over a config.

Zero dependencies beyond the standard library, deliberately: this has to run under a
bare ``python3`` on any box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# --------------------------------------------------------------------------- hardware

#: Chips per host on a v7x slice (check with `compute inspect`). Decides `hosts`, and
#: `hosts > 1` is what takes PCP off the table — so this constant is load-bearing for legality, not cosmetic.
CHIPS_PER_HOST = 4


@dataclass(frozen=True)
class Topology:
    """One TPU slice, for **legality only**.

    Deliberately carries no hardware pins. HBM capacity, bandwidth and the two MXU
    peaks live in `roofline-analysis`'s `Hardware` block and are read from the
    roofline config this search points at — a second copy here is a second authority,
    and the one that is stale is indistinguishable from the one that is not.

    ``devices`` counts *chiplets*: on v7x a chip is two chiplets sharing a package but
    presenting as two JAX devices. A slice is named in chips (`2x2x2`) and every degree
    below is in devices, which is where the factor of two goes.
    """

    slice_dims: tuple[int, ...]
    d2d: bool = True

    @property
    def chips(self) -> int:
        n = 1
        for d in self.slice_dims:
            n *= d
        return n

    @property
    def devices(self) -> int:
        return self.chips * (2 if self.d2d else 1)

    @property
    def hosts(self) -> int:
        return max(1, self.chips // CHIPS_PER_HOST)

    @property
    def name(self) -> str:
        return "x".join(str(d) for d in self.slice_dims)

    # TODO: cross-slice comparison is by hand. One slice per run is the design and the
    # report format is stable enough that two runs diff cleanly -- but nothing computes
    # tokens/s per CHIP, which is the only honest metric across slices (a bigger slice
    # always wins on absolute throughput), and nothing checks the "does it fit at all"
    # question that is usually the real reason to take one. GLM-5.2-FP8 does not fit on
    # 2x2x1 at all: 86.6 GiB of weights plus 4 GiB resident overhead against 88.3 usable,
    # EP or not -- which is the kind of answer a cross-slice section would lead with.
    @classmethod
    def parse(cls,
              spec: str | Sequence[int],
              *,
              d2d: bool = True) -> "Topology":
        """`"2x2x2"`, `"2x2x4"` or `[2, 2, 4]`. Any slice the estimator can build."""
        if isinstance(spec, str):
            dims = tuple(int(x) for x in spec.lower().split("x"))
        else:
            dims = tuple(int(x) for x in spec)
        if not dims or any(d < 1 for d in dims):
            raise ValueError(f"bad slice dims {spec!r}")
        return cls(dims, d2d=d2d)


#: Two predefined v7x shapes. Any other slice is constructible via
#: `Topology.parse` -- nothing below is specialised to these two.
V7X_1NODE = Topology((2, 2, 1))  # 4 chips, 8 devices, 1 host
V7X_2NODE = Topology((2, 2, 2))  # 8 chips, 16 devices, 2 hosts

TOPOLOGIES = {t.name: t for t in (V7X_1NODE, V7X_2NODE)}

# ------------------------------------------------------------------------ model facts


@dataclass(frozen=True)
class ModelFacts:
    """Only the facts that decide *legality*. Costing facts live in `model.py`."""

    name: str
    is_moe: bool = False
    num_routed_experts: int = 0
    top_k: int = 0
    num_kv_heads: int = 1
    use_mla: bool = False
    #: Layer count, for the one PP rule that needs it: a stage cannot hold zero layers.
    #: `0` means "not supplied" and the rule is skipped rather than guessed.
    num_layers: int = 0
    #: Query head count (optional; when > 0, allows GQA KV head replication for tp > num_kv_heads
    #: per vLLM QKVParallelLinear, and enforces GQA DCP bounds per vllm/config/model.py:1335-1362).
    num_q_heads: int = 0

    def __post_init__(self) -> None:
        if self.is_moe and self.num_routed_experts <= 0:
            raise ValueError(
                f"{self.name}: is_moe but num_routed_experts={self.num_routed_experts}"
            )


# ---------------------------------------------------------------------------- sharding


@dataclass(frozen=True)
class Sharding:
    """A point in the parallelism space.

    ``ep`` is a *boolean*, not a degree — vLLM's EP group spans TP x PCP x DP, so you
    cannot shard attention and the MoE with independent degrees. ``dcp`` subdivides TP
    rather than multiplying the world size.
    """

    tp: int = 1
    dp: int = 1
    pcp: int = 1
    dcp: int = 1
    pp: int = 1
    ep: bool = False

    @property
    def world(self) -> int:
        return self.tp * self.pcp * self.pp * self.dp

    @property
    def ep_group(self) -> int:
        """How many ranks the experts are spread over. 1 means every rank holds all."""
        return self.tp * self.pcp * self.dp if self.ep else 1

    @property
    def label(self) -> str:
        bits = [f"tp{self.tp}", f"dp{self.dp}"]
        if self.pcp > 1:
            bits.append(f"pcp{self.pcp}")
        if self.dcp > 1:
            bits.append(f"dcp{self.dcp}")
        if self.pp > 1:
            bits.append(f"pp{self.pp}")
        if self.ep:
            bits.append("ep")
        return "-".join(bits)

    @property
    def family(self) -> str:
        return family_of(self.tp, self.dp, self.pp, self.pcp)


#: The axes that consume ranks, in the order a family name lists them. EP and DCP are
#: not here: EP is a boolean over the ranks TP x PCP x DP already hold, and DCP
#: subdivides TP -- so they are variants *within* a family, not families of their own.
RANK_AXES = ("tp", "dp", "pp", "pcp")


def family_of(tp: int, dp: int, pp: int, pcp: int) -> str:
    """The report's grouping key: which rank-consuming axes are above 1, e.g. `TP×DP`."""
    deg = {"tp": tp, "dp": dp, "pp": pp, "pcp": pcp}
    on = [a.upper() for a in RANK_AXES if deg[a] > 1]
    return "×".join(on) or "single core"


LEGAL = "legal"
UNVERIFIED = "unverified"  # plumbed upstream, not established on TPU — cost it, flag it
ILLEGAL = "illegal"

_STATUS_ORDER = {LEGAL: 0, UNVERIFIED: 1, ILLEGAL: 2}

# ------------------------------------------------------- why a sharding is not shippable
#
# Four kinds of refusal, and they are not interchangeable: each has a different owner, a
# different fix, and a different answer to "so what would it be worth". The wide view of
# the search exists to keep them apart, so `class` is carried next to `reason` everywhere
# — a table that only says "refused" cannot distinguish arithmetic from a missing feature.

#: The partition does not exist. Degrees that do not multiply to the slice, a DCP that does
#: not divide TP, an EP group that does not divide the expert count. **Not costable, and
#: not a gap**: there is no such deployment to be fast or slow, so it gets no curve at any
#: width. This is the class that must never be reported as "excluded" — it was never in.
ARITHMETIC = "arithmetic"

#: The partition is coherent; vLLM or vllm-torchtpu refuses to *run* it today. Every rule
#: in this class is a `NotImplementedError` or a config assertion — a software gap, and
#: most of them say "yet" in the source. Costable, and worth costing: the number is what
#: closing the gap would buy, which is the one question the shippable view cannot answer.
INFRA = "infra"

#: Measured broken on this TPU generation, with job ids in `CONSTRAINTS.md`. Costable
#: for the same reason as INFRA, but the fix is not a flag.
#:
#: **Currently unpopulated.** Its only member was `pp > 1` on tpu7x, and that was
#: retired on 2026-09-23 — the stack was fixed under it (`j-78adee7b-af0e-43cc-9ee0`).
#: The class stays because the distinction is real and the next one will want it; an
#: empty `hardware` column in the wide view is the correct rendering, not a bug.
HARDWARE = "hardware"

#: Assigned downstream, in `run.py`, not here: the weights do not fit in HBM, or the KV
#: pool holds under one request of the cell's shape. `space.py` structurally cannot see it
#: — it needs the model's bytes and the workload — but the vocabulary lives in one place.
#: A memory-blocked sharding has **no operating point**, so it gets no curve in either
#: view; what it gets is the two numbers that say how far off it was.
MEMORY = "memory"

#: Also downstream: the per-checkpoint MoE dispatch cap. It rejects `(sharding x knobs)`
#: configs rather than shardings, which is why it is not one of the two views. See
#: `DispatchBound` below and SKILL.md 1.4.
DISPATCH = "dispatch"

#: Not a refusal — the absence of one. What survives every filter and may be benchmarked.
SHIPPABLE = "shippable"

#: Decisive-class order. A row refused by several rules is attributed to the hardest of
#: them: no feature flag rescues an arithmetic impossibility, and no code change rescues
#: silicon. Reporting the *first rule that fired* instead would make the funnel depend on
#: the order of the `if`s in `classify`.
_CLASS_ORDER = (ARITHMETIC, MEMORY, HARDWARE, INFRA)


@dataclass(frozen=True)
class Verdict:
    status: str
    reasons: tuple[str, ...] = ()
    #: One class per reason, positionally aligned with `reasons`.
    classes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status != ILLEGAL

    @property
    def costable(self) -> bool:
        """Can the cost model price this partition, whether or not anything will run it?

        True for a legal sharding, and true for one refused only by `INFRA` or `HARDWARE`
        rules — those describe a deployment that exists and would have a throughput if the
        refusal were lifted. False as soon as an `ARITHMETIC` rule fired: costing a DCP
        that does not divide TP means dividing by a shard count that is not a shard count,
        and the model would return a number rather than an error.
        """
        return all(c in (INFRA, HARDWARE) for c in self.classes)

    @property
    def blocking_class(self) -> str:
        """The class this row is attributed to in the funnel. `SHIPPABLE` if none."""
        for c in _CLASS_ORDER:
            if c in self.classes:
                return c
        return SHIPPABLE

    def reason_for(self, cls: str) -> str:
        """The first reason of a given class, for the funnel's `e.g.` column."""
        for c, r in zip(self.classes, self.reasons):
            if c == cls:
                return r
        return ""


#: What this deployment's workers are. `unified` is a normal server; the other two are
#: the halves of a P/D-disaggregated one, which vLLM distinguishes by `kv_role`.
ROLES = ("unified", "prefill", "decode")


def classify(s: Sharding,
             topo: Topology,
             model: ModelFacts,
             *,
             role: str = "unified") -> Verdict:
    """Why this sharding is or is not runnable. See `../CONSTRAINTS.md` for citations.

    Every refusal carries its class — `ARITHMETIC`, `INFRA` or `HARDWARE` — because the
    wide view of the search costs the last two and refuses the first. Adding a rule means
    deciding which of the three it is: is this partition impossible, unimplemented, or
    broken on the silicon?
    """
    hard: list[tuple[str, str]] = []
    soft: list[tuple[str, str]] = []

    if s.world != topo.devices:
        hard.append((
            ARITHMETIC,
            f"tp*pcp*pp*dp = {s.world} != {topo.devices} devices on {topo.name}"
        ))

    # PCP. Four separate refusals, three of them on the TPU side. The multihost one is
    # the wide net: it takes PCP off the 2-node recipe entirely, whatever else is set.
    # All four are `INFRA` — every one is a `NotImplementedError` or a config assertion,
    # and three of them say "yet" in the source line CONSTRAINTS.md cites.
    if s.pcp > 1:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        if s.dp > 1:
            hard.append((
                INFRA,
                "PCP does not support data parallelism (vllm config/parallel.py)"
            ))
        if s.pp > 1:
            hard.append(
                (INFRA, "PCP with pipeline parallelism is not implemented"))
        # Added upstream between 1dc0f5ce and d43dae9d (read 2026-09-17). Without EP,
        # PCP would shard the experts across the PCP ranks and the TPU MoE kernels do
        # not implement that. Matters to the *wide* view specifically: it means lifting
        # the multihost rule alone does not make a PCP row runnable on an MoE.
        if model.is_moe and not s.ep:
            hard.append(
                (INFRA, "PCP on an MoE model requires expert parallelism "
                 "(--enable-expert-parallel)"))
        if topo.hosts > 1:
            hard.append(
                (INFRA,
                 f"PCP is not supported on TPU multihost ({topo.name} is "
                 f"{topo.hosts} hosts)"))
        if role == "decode":
            hard.append(
                (INFRA, "PCP is refused on KV-consumer/decode workers"))

    # DCP subdivides TP; it does not add ranks. The divisibility rules are arithmetic —
    # there is no set of ranks to hand the shards to — while "PCP together with DCP" is a
    # missing implementation over a partition that does exist.
    if s.dcp < 1:
        hard.append((ARITHMETIC, f"dcp={s.dcp} must be >= 1"))
    elif s.pcp == 1:
        if s.tp % s.dcp:
            hard.append(
                (ARITHMETIC, f"dcp={s.dcp} must divide tp={s.tp} when pcp=1 "
                 f"(DCP reuses the TP ranks)"))
    else:
        if s.dcp not in (1, s.pcp, s.tp * s.pcp):
            hard.append((ARITHMETIC, f"dcp={s.dcp} must be in {{1, {s.pcp}, "
                         f"{s.tp * s.pcp}}} when pcp={s.pcp}"))
        if s.dcp > 1:
            hard.append(
                (INFRA, "PCP together with DCP is not implemented on TPU"))

    # Non-MLA models shard KV heads by TP when tp <= num_kv_heads, or replicate KV heads
    # across TP ranks when tp > num_kv_heads (vLLM QKVParallelLinear) provided num_q_heads
    # is supplied and divisible by tp.
    if not model.use_mla and s.tp > 1:
        if model.num_q_heads > 0 and s.tp > model.num_kv_heads:
            if s.tp % model.num_kv_heads or model.num_q_heads % s.tp:
                hard.append((
                    ARITHMETIC,
                    f"tp={s.tp} must be a multiple of num_kv_heads={model.num_kv_heads} "
                    f"and divide num_q_heads={model.num_q_heads} for GQA KV replication"
                ))
        elif model.num_kv_heads % s.tp:
            hard.append((
                ARITHMETIC,
                f"tp={s.tp} does not divide num_kv_heads={model.num_kv_heads}"
            ))
    if not model.use_mla and s.dcp > 1:
        # vllm/config/model.py:1335-1362 -- GQA DCP group shares one replicated KV head
        if s.tp <= model.num_kv_heads:
            hard.append((
                ARITHMETIC,
                f"GQA DCP needs tp={s.tp} > num_kv_heads={model.num_kv_heads}"
            ))
        elif s.dcp > s.tp // model.num_kv_heads:
            hard.append((
                ARITHMETIC,
                f"GQA DCP needs dcp={s.dcp} <= tp // num_kv_heads = {s.tp // model.num_kv_heads}"
            ))
        elif model.num_q_heads > 0 and (model.num_q_heads //
                                        model.num_kv_heads) % s.dcp:
            hard.append((
                ARITHMETIC,
                f"GQA DCP needs (H_q / H_kv)={model.num_q_heads // model.num_kv_heads} "
                f"divisible by dcp={s.dcp}"
            ))

    # EP. Both are arithmetic: nothing to shard, or a shard count that does not divide.
    if s.ep and not model.is_moe:
        hard.append((
            ARITHMETIC,
            f"{model.name} is dense; expert parallelism has nothing to shard"))
    if s.ep and model.is_moe and model.num_routed_experts % s.ep_group:
        hard.append((ARITHMETIC, f"EP group {s.ep_group} does not divide "
                     f"{model.num_routed_experts} routed experts"))

    # PP. Wired on TPU by `012e4f24` (2026-09-09) and `15085885` (2026-09-11) and now
    # validated by `platforms/pp_validation.py` — which is where these refusals are read
    # from, rather than inferred. The 2026-09-10 "broken on tpu7x" measurement predates
    # the wiring and is retired: `j-78adee7b-af0e-43cc-9ee0` served GLM-5.2-FP8 at
    # `pp16-tp1-dp1` across both hosts of a `2x2x2`, 32/32 requests. See CONSTRAINTS.md.
    if s.pp > 1:
        if s.tp > 1:
            # A stage's TP collectives stall on the first serving step, because the
            # other stages are not launching the same collectives. Warmup passes.
            hard.append(
                (INFRA,
                 "PP does not support tensor parallelism inside a stage "
                 "(pp_validation.py:48-53) — use --tensor-parallel-size=1"))
        if s.dp > 1:
            # Not a code refusal: `pp_validation.py` lets this through. It is the
            # SPMD/MPMD split — PP is SPMD, DP is MPMD — and it is *unmeasured*. Kept
            # hard so the search cannot spend a 2.5-4 h benchmark settling it by accident;
            # one job at `pp8-dp2` would retire the rule either way.
            hard.append(
                (INFRA, "only pure PP is supported today — PP is SPMD, DP is "
                 "MPMD (unmeasured, no code refusal; CONSTRAINTS.md)"))
        if model.num_layers and s.pp > model.num_layers:
            hard.append(
                (ARITHMETIC, f"pp={s.pp} exceeds {model.num_layers} layers — "
                 f"a stage would hold none"))

    if hard:
        return Verdict(ILLEGAL, tuple(r for _, r in hard),
                       tuple(c for c, _ in hard))
    if soft:
        return Verdict(UNVERIFIED, tuple(r for _, r in soft),
                       tuple(c for c, _ in soft))
    return Verdict(LEGAL)


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def enumerate_shardings(
    topo: Topology,
    model: ModelFacts,
    *,
    include_pp: bool = True,
    include_pcp: bool = True,
    include_dcp: bool = True,
    role: str = "unified",
) -> list[tuple[Sharding, Verdict]]:
    """Every sharding on this slice, each with its verdict. Illegal ones are kept.

    Keeping the illegal rows is the point: "why is there no TP=16 row" is a question
    the caller will ask, and the answer should be in the table rather than in a
    reconstruction of what this function skipped.
    """
    out: list[tuple[Sharding, Verdict]] = []
    seen: set[Sharding] = set()
    ds = _divisors(topo.devices)
    for tp in ds:
        for dp in ds:
            for pcp in (ds if include_pcp else [1]):
                for pp in (ds if include_pp else [1]):
                    if tp * dp * pcp * pp != topo.devices:
                        continue
                    dcps = _divisors(tp) if include_dcp else [1]
                    for dcp in dcps:
                        for ep in ((False, True) if model.is_moe else
                                   (False, )):
                            s = Sharding(tp=tp,
                                         dp=dp,
                                         pcp=pcp,
                                         dcp=dcp,
                                         pp=pp,
                                         ep=ep)
                            if s in seen:
                                continue
                            seen.add(s)
                            out.append((s, classify(s, topo, model,
                                                    role=role)))
    out.sort(key=lambda sv: (_STATUS_ORDER[sv[1].status], sv[0].label))
    return out


def legal_shardings(topo: Topology, model: ModelFacts, **kw) -> list[Sharding]:
    """The runnable subset, `unverified` included. Convenience over `enumerate_shardings`."""
    return [s for s, v in enumerate_shardings(topo, model, **kw) if v.ok]


# ----------------------------------------------------------------- the scheduler knobs


@dataclass(frozen=True)
class ServerConfig:
    """A sharding plus the three scheduler knobs that move TTFT.

    ``max_num_seqs`` and ``max_num_batched_tokens`` are **per DP engine**, exactly as
    vLLM takes them. Fleet-wide values are ``* dp``, and mixing the two up is the most
    common way to be wrong by 16x.
    """

    sharding: Sharding
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 1024
    prefill_schedule_interval: int = 1

    @property
    def label(self) -> str:
        base = f"{self.sharding.label}/seqs{self.max_num_seqs}/mnbt{self.max_num_batched_tokens}"
        if self.prefill_schedule_interval > 1:
            base += f"/psi{self.prefill_schedule_interval}"
        return base

    def flags(self) -> str:
        """The vLLM server flags this config is. What goes in `VLLM_SERVER_CMD`."""
        s = self.sharding
        out = [
            f"--tensor-parallel-size={s.tp}", f"--data-parallel-size={s.dp}"
        ]
        if s.pcp > 1:
            out.append(f"--prefill-context-parallel-size={s.pcp}")
        if s.dcp > 1:
            out.append(f"--decode-context-parallel-size={s.dcp}")
        if s.pp > 1:
            out.append(f"--pipeline-parallel-size={s.pp}")
            # `pp_validation.py:54-58` refuses async scheduling under PP: the stages
            # before the last never see the sampled tokens the async placeholder
            # substitution reads back from the device. Emitting the flag here is what
            # keeps a PP row from dying in `__post_init__` 20 minutes into a job.
            out.append("--no-async-scheduling")
        if s.ep:
            out.append("--enable-expert-parallel")
        out.append(f"--max-num-seqs={self.max_num_seqs}")
        out.append(f"--max-num-batched-tokens={self.max_num_batched_tokens}")
        if self.prefill_schedule_interval > 1:
            out.append(
                f"--prefill-schedule-interval={self.prefill_schedule_interval}"
            )
        return " ".join(out)


@dataclass(frozen=True)
class DispatchBound:
    """The MoE all-to-all dispatch dimension cap, when a model has one.

    Off by default. On GLM-5.2-FP8 the recorded bound is `topk x DP x mnbt <= 65536`,
    but the arithmetic in the source comment only closes at topk=4 while `config.json`
    says 8 — so the *factor* is unresolved and rebuilding it for another DP degree
    lands on the faulting shape. Set `topk` explicitly and only from something you have
    verified; leaving `enabled=False` means the search will happily propose a shape that
    faults a SparseCore, which is a cheaper failure than silently excluding the winner.
    """

    enabled: bool = False
    topk: int = 0
    max_dispatch: int = 0
    note: str = ""

    def violation(self, cfg: ServerConfig) -> str | None:
        if not self.enabled:
            return None
        dim = self.topk * cfg.sharding.dp * cfg.max_num_batched_tokens
        if dim > self.max_dispatch:
            return (
                f"MoE dispatch dim topk({self.topk}) x dp({cfg.sharding.dp}) x mnbt"
                f"({cfg.max_num_batched_tokens}) = {dim} > {self.max_dispatch}"
                + (f" — {self.note}" if self.note else ""))
        return None


def enumerate_configs(
    shardings: list[Sharding],
    *,
    max_num_seqs: list[int],
    max_num_batched_tokens: list[int],
    prefill_schedule_interval: list[int] = (1, ),
    dispatch_bound: DispatchBound = DispatchBound(),
) -> list[tuple[ServerConfig, str | None]]:
    """Cross the shardings with the scheduler knobs. Second element is a rejection reason."""
    out: list[tuple[ServerConfig, str | None]] = []
    for s in shardings:
        for seqs in max_num_seqs:
            for mnbt in max_num_batched_tokens:
                for psi in prefill_schedule_interval:
                    cfg = ServerConfig(s, seqs, mnbt, psi)
                    out.append((cfg, dispatch_bound.violation(cfg)))
    return out
