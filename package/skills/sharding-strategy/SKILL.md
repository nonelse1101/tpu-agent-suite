---
name: sharding-strategy
description: Pick how to shard a model on a TPU slice before a benchmark is spent — enumerate every TP/DP/EP/PCP/DCP/PP partition on any slice (2x2x1, 2x2x2, 2x2x4, …), price each at one operating point through roofline-analysis, and return an auditable report grouped by combination (TP, DP, TP×DP, PP, …, and the combinations blocked today as forward-looking projections) — how each combination divides weights and KV, every member, what EP and DCP change, one worked example whose derivation is checked step by step against the sweep — then every point on one TTFT-vs-throughput and one TTFT-vs-T_decode chart, with a separate table of rejections and their reasons. Use when asked "which sharding should we use", "how should I deploy <model> on 2x2x2 or 2x2x4", "TP or DP for this model", "is EP worth it here", "what would lifting the PCP block buy", or to pick the configuration an optimization effort starts from. Proposes and ranks; measures nothing.
scripts:
- scripts/sharding-strategy/search/space.py
- scripts/sharding-strategy/search/sweep.py
- scripts/sharding-strategy/search/report.py
- scripts/sharding-strategy/search/plot.py
- scripts/sharding-strategy/search/selftest.py
- scripts/sharding-strategy/presets/GLM-5.2-FP8-2x2x2-8k1k.json
- scripts/sharding-strategy/presets/GLM-5.2-FP8-2x2x2-128k.json
- scripts/sharding-strategy/presets/Qwen3.5-35B-A3B-BF16-2x2x1-8k1k.json
- scripts/sharding-strategy/presets/Qwen3.5-35B-A3B-FP8-1chip-2devices-8k1k.json
- scripts/sharding-strategy/presets/Qwen3.5-35B-A3B-FP8-2x2x1-8k1k.json
# The roofline modules the search imports, and the configs a preset points at.
- scripts/roofline-analysis/roofline.py
- scripts/roofline-analysis/models.py
- scripts/roofline-analysis/roofline_model1.py
- scripts/roofline-analysis/roofline_model2.py
- scripts/roofline-analysis/serving.py
- scripts/roofline-analysis/derivation.py
- scripts/roofline-analysis/step_api.py
- scripts/roofline-analysis/cost_estimator/__init__.py
- scripts/roofline-analysis/cost_estimator/collective.py
- scripts/roofline-analysis/configs/glm-5.2-fp8-8k1k-2x2x2.json
- scripts/roofline-analysis/configs/glm-5.2-fp8-256k1k-2x2x2.json
- scripts/roofline-analysis/configs/glm-5.2-fp8-128k-1chip-t1024.json
- scripts/roofline-analysis/configs/qwen3.5-35b-a3b-bf16-8k1k-2x2x1.json
- scripts/roofline-analysis/configs/qwen3.5-35b-a3b-fp8-8k1k-1chip.json
- scripts/roofline-analysis/configs/qwen3.5-35b-a3b-fp8-8k1k-2x2x1.json
references:
- references/sharding-strategy/CONSTRAINTS.md
- references/roofline-analysis/parallelism
host_dependencies:
- matplotlib
---

# Sharding strategy — enumerate, price, present

Choose **how to shard a model on a fixed TPU slice**, before a benchmark is spent.

> [!IMPORTANT]
> **Prediction, not measurement.** Nothing here runs on hardware, and nothing here is a result.
> Run in the **main thread**: in step 0, {{built-in:ask_user_question}} for any cell input that
> is missing.

## What this skill owns, and what it does not

```
{{script:sharding-strategy/search/space.py}}        what the code permits: topology, the six axes, legality, the classes
{{script:sharding-strategy/search/sweep.py}}        enumerate x price at ONE operating point -> points/space/funnel.csv
{{script:sharding-strategy/search/report.py}}       group by combination, render REPORT.md, call plot.py
{{script:sharding-strategy/search/plot.py}}         the two combined-view charts, from points.csv only
```

**This skill has no roofline logic.** Every time, byte and ratio of either comes from
{{skill:roofline-analysis}}:

```
{{script:roofline-analysis/serving.py}}      operating_point: admission, T_step, TTFT, tok/s
{{script:roofline-analysis/derivation.py}}   one point written out step by step, checked
{{script:roofline-analysis/roofline.py}}     weight_terms, kv_terms, the op tables
```

`selftest.py::NoRooflineHere` fails if a file here calls anything in the cost model beyond
`load`, the value types, `serving.operating_point` and `derivation.derive`. If the report needs
a number the roofline does not expose, **add it to {{skill:roofline-analysis}} behind its selftest**, then
render it here. The arithmetic done in this directory is presentation only — a ratio of two
printed numbers, printed next to both.

Why the line is drawn there: measured 2026-09-10, five agents given the same pins returned
prefill estimates spanning five orders of magnitude, while reproducing the *pins* to within 4%.
Transcription is reliable; derivation is not. So scripts derive and the report transcribes.

# Prerequisites

1. **A roofline config for the checkpoint** — a `<model>-<cell>.json` file such as
   {{script:roofline-analysis/configs/glm-5.2-fp8-8k1k-2x2x2.json}}.
   The preset names it and restates no architecture pin (`selftest.py` fails if it does). None
   exists → {{skill:roofline-analysis}} Path B step 1.
2. **The constraints, re-verified against your checkout.** {{reference:sharding-strategy/CONSTRAINTS.md}}
   cites every legality rule to `file:line` at recorded shas:

   ```bash
   grep -n "does not support data parallelism\|DCP reuses the TP ranks\|EP group spans" \
       <vllm>/vllm/config/parallel.py
   grep -n "NotImplementedError" -A2 <checkout>/src/vllm_torchtpu/platforms/pcp_validation.py
   ```

   Anything moved → fix {{script:sharding-strategy/search/space.py}} and `CONSTRAINTS.md` in the same edit and bump the shas.
   A rule silently *lifted* costs the winning config; one silently *added* costs a job that dies
   in `__post_init__` after the image build.

The rules that decide most sweeps: `world = TP × PCP × PP × DP`; **EP is a boolean** spanning
`TP × PCP × DP`; **DCP subdivides TP** and is the only axis that shards an MLA KV cache; **PCP is
refused on multihost**; **PP runs in the pure-PP shape only** (`TP = DP = PCP = 1`,
`--no-async-scheduling`, stage boundaries via `VLLM_PP_LAYER_PARTITION`) — legal since
2026-09-23, `CONSTRAINTS.md` § PP.

---

# Step 0 — the cell

A **cell** is the deployment question: slice, workload, one operating point, role.

The presets are named `<model-dir>-<topology>-<cell-tag>.json`:

- {{script:sharding-strategy/presets/GLM-5.2-FP8-2x2x2-8k1k.json}}
- {{script:sharding-strategy/presets/GLM-5.2-FP8-2x2x2-128k.json}} (older format without
  `roofline_config`; `sweep.py` refuses it until it is rewritten against the 8k1k preset)

If one matches, offer it — **read its `_comment`, not its filename**. Otherwise
{{built-in:ask_user_question}} for each input below; clarify, never guess:

| input | why it is not guessable |
| --- | --- |
| slice (`2x2x1`, `2x2x2`, `2x2x4`, …) | different legal sets — PCP exists on one host, not on two |
| ISL / OSL | the ranking moves with the ratio, not the lengths |
| concurrency (fleet-wide) | every sharding is priced at this one load; make it a multiple of the largest DP |
| `max-num-batched-tokens`, `max-num-seqs` | per DP engine, as vLLM takes them; one value each |
| role | only `unified` is in scope — ask so the refusal is explicit |

**One operating point per cell, deliberately.** Every sharding sees the same load and knobs, so two
points on the combined chart differ by the sharding and nothing else. Concurrency and `mnbt`
sweeps are not implemented yet; `sweep.py` refuses a list rather than silently taking one element.
To compare loads, write two cells.

**One slice per run.** Compare slices on tokens/s per chip, never absolute.

# Step 1 — the preset

Copy {{script:sharding-strategy/presets/GLM-5.2-FP8-2x2x2-8k1k.json}} to
`{{runs}}/<run_id>/artifacts/sharding-strategy/<cell>.json` and edit the copy. It holds a
pointer to the roofline config, the slice, the workload, the operating point, the dispatch bound
and the role — nothing about the model.

`sweep.py` resolves `roofline_config` relative to the preset's own directory. In the copy, set
`roofline_config` to an absolute path, for example the installed path of
{{script:roofline-analysis/configs/glm-5.2-fp8-8k1k-2x2x2.json}}.

**The dispatch bound** (`topk × DP × mnbt ≤ 65536` on some checkpoints) is off by default because
its factor is unresolved (`U-260909-17`). On, it marks over-cap shardings `‡cap` — priced,
never shippable. If you enable it, run the cell both ways and say whether the bound changed
membership or order.

# Step 2 — run

```bash
python3 {{script:roofline-analysis/roofline.py}}   --selftest     # the op arithmetic
python3 {{script:roofline-analysis/serving.py}}    --selftest     # gates, queueing, Little's Law
python3 {{script:roofline-analysis/derivation.py}} --selftest     # every worked-example step vs the sweep
python3 {{script:sharding-strategy/search/selftest.py}}           # enumeration, funnel, grouping, report, the seam

python3 {{script:sharding-strategy/search/report.py}} {{runs}}/<run_id>/artifacts/sharding-strategy/<cell>.json \
    --out {{runs}}/<run_id>/artifacts/sharding-strategy/<cell>
```

All four, every time; they guard different things. The run takes seconds. **Never truncate the
sweep** — every coherent sharding is priced, blocked ones included.

## The classes — every sharding is attributed to exactly one

A row is attributed to the **hardest** rule that fired, never the first in source order. Use this
vocabulary; do not invent adjacent words like "unsupported".

| class | means | in the report |
| --- | --- | --- |
| `arithmetic` | the partition does not exist | §2.1, **rejected** — never a config |
| `memory` | exists, does not fit (weights, or KV pool < 1 request) | §2.1, **rejected** — with its numbers |
| `infra` | vLLM / vllm-torchtpu refuses it today | its combination's section, marked **projection**; §2.2 gives the rule |
| `hardware` | measured broken on this silicon (currently no members) | as `infra` |
| `dispatch` | over the MoE dispatch cap at this `mnbt` | as `infra`, marked `‡cap` |
| `shippable` | runs today and fits | its combination's section |

**Projections are first-class.** They are priced by the same model at the same operating point, so
the report is forward-looking: it says what lifting each rule would buy. They are never a benchmark
candidate until the rule is lifted, and the report never lets one be read as shippable.

# Step 3 — the report

`report.py` writes `<out>/REPORT.md`, `points.csv`, `space.csv`, `funnel.csv`, `columns.md` and
`png/`. **Everything numeric is rendered; you write only the `[annotate]` blocks.**

| § | what | why it is there |
| --- | --- | --- |
| 0 | cell, operating point, pins, the reproduce command | the question, before any answer |
| 1 | funnel, then one index row per **combination** | "did you consider X" is answerable first |
| 2 | **rejections and blocks** — §2.1 arithmetic and memory with reasons; §2.2 each blocking rule, how many rows it blocks, which combinations | the reasons live in one table, not scattered |
| 3… | **one section per combination** with an operating point — shippable first, then projections | the explanation next to the numbers |
| n | **combined view** — TTFT × throughput, TTFT × TPOT, and every point ranked | the comparison across combinations |
| n+1 | what the model gets wrong; the self-check line; artifacts | how far to trust it |

A **combination** is the set of rank-consuming axes above 1 — `TP`, `DP`, `TP×DP`, `PP`,
`TP×PCP`, … EP and DCP consume no ranks, so they are **variants inside** each section.

Each combination section is the same five parts:

1. **How it works** — one line per axis present, plus EP and DCP, each citing
   `<axis>/SKILL.md` under {{reference:roofline-analysis/parallelism}}.
2. **Every member** — not a top-N: stack verdict, fits, weights, KV pool and cap, running and gate,
   T_prefill, TPOT, TTFT, tok/s, bound. Members that do not fit stay, with the reason.
3. **What EP and DCP change** — each variant against the sibling differing in that knob only,
   both operands printed.
4. **Worked example** on the best member (best shippable; best projection when none is):
   - how the weights divide — `weight_terms`, tensor by tensor: whole model, divisor as axes,
     per core, and why that divisor;
   - how the KV cache divides — `kv_terms`, the same way;
   - what it communicates — every op with ICI time, decode and prefill;
   - where the time goes — blocks, layer kinds, largest ops; each sums to the phase total;
   - **the derivation** — ranks → weights → KV pool → KV cap → running → T_decode → T_prefill →
     QPS → tok/s → queue → TTFT, each step with formula, inputs, value, and what
     `serving.operating_point` returned. A ✗ is a finding: report it, do not smooth it over.
5. `[annotate]` — why this combination lands where it does in the combined view.

## Filling the `[annotate]` blocks

- **Quote only numbers the report already prints**, and name the table they came from. A new
  number in prose is a derivation, which is what this skill exists not to do.
- Say **which step** a claim rests on — e.g. "wins on admission (step 7: 16 running against
  6.2), not on step time (step 8 is 4% slower)". A step win survives a change of load; an
  admission win is a claim about this ISL+OSL.
- For a projection, say what rule blocks it and what lifting it would buy against the best
  shippable row, and nothing about whether it will be lifted.
- A margin under 5% is two candidates, not a ranking.
- Do not add a recommendation, TL;DR or shortlist above §1.

Then read `REPORT.md` top to bottom once as a reviewer: every ✓ present, the self-check line
clean, every figure's legend legible.

## Where the output goes

| destination | what |
| --- | --- |
| `{{runs}}/<run_id>/artifacts/sharding-strategy/<cell>/` | the report, CSVs, PNGs. A modelled result is in-flight |
| `{{runs}}/<run_id>/artifacts/sharding-strategy/<M>-<topo>-<tag>.json` | the cell. Kept beside the report — the next step 0 |

**Do not record a sharding report as an experiment result.** It is a prediction.

---

# What the model gets wrong — read before quoting

`report.py` prints the rows that bear on a one-point comparison; this is the full list.

| defect | effect | what to do |
| --- | --- | --- |
| **Collectives fully exposed** — a profile puts real exposure at 21.9% of decode, 37.8% of prefill | every ICI-bound row is pessimistic | treat a comm-bound ordering as the weakest part of the ranking |
| **Collective fixed cost ~2.6× underpriced** (13.5 µs measured vs 5.2 µs modelled, `j-484cd4c0`) | lands on small decode collectives, so on TPOT | `collective.py`'s `launch_overhead` exists for it and defaults to 0 |
| **DCP's balanced ownership is assumed** | the sparse core's `K_idx / DCP` may be optimistic | read a DCP row's margin over its `dcp=1` sibling as unmeasured |
| **PCP folds into TP** for weights | a PCP projection is optimistic | a ceiling on what fixing multihost PCP would buy |
| **The indexer's layer positions are guessed** (count pinned, spread evenly) | small error in PP stage balance | `layer_costs`' docstring |
| **`kernel_efficiency = 1.0`**, peak FLOPs and bandwidth, no host path | every absolute number is a ceiling | quote ratios |
| **No layout or copy term** — 7.2% of prefill, 4.6% of decode measured | moves levels more than order | state it |

---

## Where this sits in the loop

```
{{skill:roofline-analysis}}   the config, and every number here     ← prerequisite, and the engine
sharding-strategy   this: the space, grouped and derived  ← proposes; measures nothing
hillclimbing        kernels, scheduling, layout           ← inside the shape chosen here
```

Sharding is chosen **once, before** hillclimbing: an optimization effort that re-opens it mid-flight throws away
every comparison made under the old shape. **The one backedge:** an exposed collective with no
compute left to hide behind is evidence the partitioning is wrong, and comes back here with the
profile.

## Files

| file | what it is |
| --- | --- |
| {{reference:sharding-strategy/CONSTRAINTS.md}} | what the code permits, cited to `file:line`, with the shas |
| {{script:sharding-strategy/presets/GLM-5.2-FP8-2x2x2-8k1k.json}}, {{script:sharding-strategy/presets/GLM-5.2-FP8-2x2x2-128k.json}} | one file per cell; a pointer to the pins, never a copy |
| {{script:sharding-strategy/search/space.py}} | topology, the six axes, legality, the classes, `family_of`, the dispatch bound |
| {{script:sharding-strategy/search/sweep.py}} | enumerate × price at one operating point → CSVs |
| {{script:sharding-strategy/search/report.py}} | grouping, the sections, the reasons tables, the combined view |
| {{script:sharding-strategy/search/plot.py}} | the two charts, from `points.csv` alone |
| {{script:sharding-strategy/search/selftest.py}} | enumeration, funnel, grouping, report completeness, and `NoRooflineHere` |
