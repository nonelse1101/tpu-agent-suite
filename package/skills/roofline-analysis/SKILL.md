---
name: roofline-analysis
description: Models the analytic roofline of an LLM on TPU — per-op FLOPs, HBM bytes, collective volume, and whether each op is memory, compute or communication bound. Use when asked "what is the hardware floor for this op", "is this memory or compute bound", "how much headroom is there", "model the roofline for this model", "how fast could this kernel possibly be", "estimate KV capacity or max concurrent requests", or to rank optimization candidates before spending hours on a benchmark.
scripts:
- scripts/roofline-analysis/roofline.py
- scripts/roofline-analysis/models.py
- scripts/roofline-analysis/roofline_model1.py
- scripts/roofline-analysis/roofline_model2.py
- scripts/roofline-analysis/serving.py
- scripts/roofline-analysis/derivation.py
- scripts/roofline-analysis/step_api.py
- scripts/roofline-analysis/cost_estimator/__init__.py
- scripts/roofline-analysis/cost_estimator/collective.py
- scripts/roofline-analysis/configs/glm-5.2-fp8-128k-1chip-t1024.json
- scripts/roofline-analysis/configs/glm-5.2-fp8-256k1k-2x2x2.json
- scripts/roofline-analysis/configs/glm-5.2-fp8-8k1k-2x2x2.json
- scripts/roofline-analysis/configs/qwen3.5-35b-a3b-bf16-8k1k-2x2x1.json
- scripts/roofline-analysis/configs/qwen3.5-35b-a3b-fp8-8k1k-1chip.json
- scripts/roofline-analysis/configs/qwen3.5-35b-a3b-fp8-8k1k-2x2x1.json
references:
- references/roofline-analysis
---

# Roofline Analysis

Work out how fast something *could* run on a TPU, from shapes and hardware limits alone.

Everything here uses one formula:

```
time = max(HBM time, compute time) + collective time
```

Compute and memory overlap inside an op; collectives are charged on top.

## Which of the three are you being asked for?

| the ask | go to |
| --- | --- |
| a block of code, a kernel, a fused region | Path A |
| one named op — "MLA QK", "the MoE down-projection" | Path A |
| a whole model on a slice | Path B |

If it is ambiguous, prefer A. It is cheap, and a model roofline needs inputs you may not have.

---

## Path A — a block of code, or one op

### 1. Get the shapes

Read the code. You need, per call: input and output tensor shapes, weight shapes, dtypes, and how
many times it runs per forward pass. If the dtypes are not in the code, say so in the output rather
than assuming bf16.

### 2. Read the matching spec in `model/common/`

**Do not skip this.** [`model/common/<op>/SKILL.md`]({{reference:roofline-analysis/model/common}}) lists the sub-operations a
component actually has, and the point of reading it is to not leave any out. The ones that get
forgotten:

- **Anything on the VPU** — normalisation, residual adds, RoPE, softmax, SwiGLU, top-k, permutes.
  The VPU runs at about **1/16 the MXU rate**, so a small FLOP count is not automatically free.
  Most of these do turn out memory-bound, but you have to price them to find that out.
- **Data movement that is not a collective** — the MoE token permute is 4.8% of prefill and is
  neither a GEMM nor a collective, which is exactly why it was missing for weeks.
- **Activation traffic.** A GEMM reads its input and writes its output. For `MLA: Query Down Proj`
  at 8K tokens that is 80 of its 92 MiB — the weights are the small part.

### 3. Compute it

```
FLOPs    = 2 × tokens × pad(contracting_dim, 256) × output_dim × n_matrices
compute  = FLOPs / peak(all operand dtypes)     # one wide operand → the slower rate
HBM      = weights + activations in + activations out (+ KV for attention)
memory   = HBM / hbm_bw
```

Hardware numbers from {{reference:roofline-analysis/tpu/V7X.md}} and {{reference:roofline-analysis/tpu/V7X_2x2x2.md}}.

#### MXU padding — the contracting dimension ceil-pads to 256

Normative in {{reference:roofline-analysis/model/common/REQUIREMENTs.md}} §MXU; restated here
because **Path A is the only path where a human does this by hand.** Path B can't get it wrong —
{{script:roofline-analysis/roofline.py}} pads every matmul — so this section is for the one-op case.

```
FLOPs = 2 × tokens × max(contracting_dim, 256) × output_dim × n_matrices
```

The rule is easy. **Finding the contracting dimension is not**, and that is where it actually goes
wrong. The trap is an op that is a matmul but is not *described* as one — anything called a
weighted sum, a score aggregation, a head reduction, a gather-and-combine. `model/common` writes
these as `2·B·S²·H_idx`-style formulas, which state the **useful** work; the MXU charges the
**padded** work, and no per-op formula in `model/common` shows the padding.

Every sub-256 contraction on GLM-5.2 at the 128K cell, and what it costs:

| Operator | contracts over | padded to | multiplier | charged |
| --- | --- | --- | --- | --- |
| `DSA Indexer: Weighted Sum` | `H_idx` **32** | 256 | **8×** | 68.72 GFLOP for 8.59 of work |
| `MLA: QK Rope` | `D_rope` **64** | 256 | **4×** | 68.72 GFLOP for 17.18 of work |
| `DSA Indexer: QK` | `D_idx` **128** | 256 | **2×** | 2,199 GFLOP for 1,100 of work |

That last row is 13 ms of prefill in pure zeros. **A reduction over heads is a matmul** — `H_idx`
is the worst ratio here precisely because nobody thinks of a weighted sum as needing a 256-wide
contraction. Quote both numbers when the multiplier is over 2×: "68.72 GFLOP, of which 51.54 is
padding" says a kernel fix exists; "68.72 GFLOP" says buy a bigger chip.

If there is a collective, price it with
{{script:roofline-analysis/cost_estimator/collective.py}} and get the payload from
[`parallelism/<scheme>/SKILL.md`]({{reference:roofline-analysis/parallelism}}). Never write your own ring formula — the estimator
knows the on-package link from the off-package one and yours will not.

Do the arithmetic in Python, not in your head.

---

## Path B — a whole model

Here the arithmetic lives in [`roofline.py`]({{script:roofline-analysis/roofline.py}}), which is fixed and
tested. Your job is to supply a config, not to compute anything.

### 1. Check you have everything

If a config already exists in {{script:roofline-analysis/configs}} for this model and slice, use it and skip ahead.

Otherwise, two of the five inputs you should **go and read**, and three you have to be given.

#### Read these yourself — do not ask

**Start with [`model/architecture/<family>/<size>.md`]({{reference:roofline-analysis/model/architecture}}).** If the model has one,
it already has the shapes, the layer structure and the per-op dimensions worked out in this skill's
own conventions — that is what it is for. GLM-5.2, Qwen3.5-397B, DeepSeek-V3, Kimi K3 and two Gemmas
are there today.

If there is no file for the model, build one from, in order:

| | |
| --- | --- |
| the checkpoint's `config.json` | layers, hidden, heads, MLA ranks or GQA head counts, FFN widths, experts and top-k, vocab |
| `vllm/model_executor/models/<arch>.py` | only what the one above does not state |

Then **write what you worked out to `<family>-<size>.md` in the output directory, in the format of
{{reference:roofline-analysis/model/architecture/SKILL.md}}**, so the next person reads it instead of
deriving it again. The output directory is the one the caller names, or
`{{runs}}/<run_id>/artifacts/roofline/` when the caller names none.

**Numerics**: weight dtype from `config.json`'s `quantization_config` — check
`modules_to_not_convert`, it can run to hundreds of entries. KV cache dtype from the server's
`--kv-cache-dtype`, which lands in `CacheConfig.cache_dtype` and defaults to `"auto"`, meaning the
model dtype.

Two things to watch, both worth several times the effort of looking them up:

- **How many layers actually run a given sub-block.** GLM-5.2 has 78 layers but only **21** run the
  indexer — the rest reuse the indices. `config.json` does not say 21; the architecture file gives
  the rule (`(max(layer_idx - 2, 0) % 4) == 0`, so layers 0,1,2,6,10,…,74) and that is where the
  number comes from. Assuming 78 overstates the indexer terms by nearly 4×.
- **Activation dtypes** are the genuinely hard ones — precision is set at the quantization method's
  call sites, not in one field, so there is no single place to look. Pin what you can, then **name
  the sites you could not** in the output. The script defaults them and reports which it defaulted.
  An unpinned activation dtype is worth 2× on every GEMM that uses it, so say it out loud.

#### Ask for these — they are choices, not facts

| | |
| --- | --- |
| **Hardware** | Which slice, and which of the two v7x specs in this repo — they disagree by 0.78× on FLOPs and 0.91× on bandwidth |
| **Sharding** | TP, DP, PP, PCP, DCP, and whether EP and SP are on |
| **Workload** | input length, output length, `max-num-batched-tokens`, and how many requests decode at once |

**If one of these three is missing, stop and say so.** Do not guess and do not pick a plausible
default — a wrong input here produces a confident wrong answer, which is worse than no answer. You
may be running as a subagent with nobody to ask, so **return the list and end there** rather than
trying to start a conversation:

```
Can't compute this yet. I need:
- <thing>: <why it matters>
- <thing>: <why it matters>

Already have: <short list, including what you read out of config.json and the model code>.
```

### 2. Write the config and run it

Copy the nearest file in {{script:roofline-analysis/configs}} into the output directory and edit the copy. Then:

```bash
python3 {{script:roofline-analysis/roofline.py}} --selftest              # first, every time
python3 {{script:roofline-analysis/roofline.py}} {{script:roofline-analysis/configs}}/<cfg>.json
python3 {{script:roofline-analysis/roofline.py}} <cfg>.json --layer-csv layers.csv    # for a spreadsheet
```

The selftest checks 23 numbers against a hand-built reference. If it fails, stop — the arithmetic
is broken and nothing downstream means anything.

### 3. Check nothing is missing from the op list

Walk `model/common/` for each component the model uses and confirm every op in the spec shows up in
the output. This found two missing ops the last time anyone did it. Mention anything that does not
line up.

### 4. Name the layer kinds

`roofline.py` prints, per phase, **one op table per kind of layer** — dataflow-ordered, already
summed, with a `TOTAL` row and a layer-sums table. It prints the kind's *key*. **You supply the
name**, because that is model vocabulary and the arithmetic is not.

The keys come from crossing two facts in the config, so they are the same four for every model:

| key | the layer | name it |
| --- | --- | --- |
| `dense_idx` | dense FFN, runs the indexer | Full Indexer Dense Layer |
| `dense` | dense FFN, no indexer | Dense Layer |
| `moe_idx` | MoE FFN, runs the indexer | Full Indexer MoE Layer |
| `moe` | MoE FFN, reuses another layer's indices | Index Share MoE Layer |

Empty kinds are dropped, so most models show fewer than four. GLM-5.2 shows three — `dense_idx` ×3,
`moe_idx` ×18, `moe` ×57 — and a model with no indexer at all shows `dense` and `moe`, where "Index
Share" would be meaningless and the plain names are right. **Rename freely for the model in front of
you**; keep the key beside the name once, so a reader can get back to the CSV.

### 5. Report the tables

Lead each phase with its layer sums (which kind owns the phase), then the op table for the kind that
owns it. Print the others only if the reader needs them — three near-identical 25-row tables is how
a report stops being read.

Columns, left to right: Operator, Memory Time, Compute Time, ICI Latency Time, ICI BW Time, Roofline
Time, Boundness, then Memory Occupancy as Activations / Weights / KV.

Four things about these rows are load-bearing and easy to get wrong:

- **A `sub` row is already inside its `fused` parent.** `MLA: Flash Attention QK+AV` is the three
  MLA attention ops and the softmax, which are one kernel on the device; its children are printed
  under it with `-->` for the shape, not to be added. Show the parent, indent the children, never
  sum both.
- **A merged row does not satisfy `max(mem, compute) + ici = roofline`.** Parents and `TOTAL`s sum
  their parts, which run one after another. Only leaf rows obey the formula.
- **Memory Occupancy is HBM residency on this core, not traffic** ({{reference:roofline-analysis/model/common/REQUIREMENTs.md}}
  §TRANSFER and OCCUPANCY) — the weights an op owns, the activation tensors it holds, the cache it
  attends over. It does **not** divide by `hbm_bw` to give Memory Time; under sparse attention the
  two are 64× apart. It is not fleet-wide. Say "per core" wherever you print it.
- **Occupancy peaks, it does not add, wherever things are not live together.** Across a phase's
  passes each column is the max over passes. A `TOTAL`'s activations are the max over its ops (a
  fused parent sums its children, whose buffers coexist); weights and KV sum. The layer-sums
  `TOTAL` takes the max activation across kinds and sums weights and KV.
- **Times are summed over every pass in the phase.** A chunked prefill's 8 passes are already in the
  number, so the prefill table is the whole prompt, not one chunk.

ICI split two ways because the halves respond to different fixes: bandwidth time falls when the
payload does, latency time does not — it is hop count plus the per-call barrier. A collective that
is mostly latency will not be fixed by quantizing what it carries.

---

## What to hand back

Lead with the numbers, keep the prose short.

**For one op or a block:**

```
<op name>, <shape summary>

  compute     X.XXX ms   <FLOPs> at <dtype>
  HBM         X.XXX ms   <bytes> = weights <w> + act in <i> + act out <o>
  collective  X.XXX ms   <kind> over <group>          (omit if there is none)
  ------------------------------------------------
  roofline    X.XXX ms   bound by <HBM | MXU | VPU | collective>
  × <n> per forward pass = X.XX ms
```

Then a sentence or two: what binds it, and what would have to change to move it.

**For a model:** a report a reader can **audit** — every headline number next to the formula and
inputs that produced it. Every Path B run returns all of the sections below, in this order. A section
you cannot fill (no workload given, say) is printed with the reason it is empty, not dropped.

Run all three and keep their output — the report is filtering and naming over them, never new
arithmetic:

```bash
python3 {{script:roofline-analysis/roofline.py}} --selftest
python3 {{script:roofline-analysis/roofline.py}} <cfg>.json --layer-csv <out>.layers.csv > <out>.raw.txt
python3 {{script:roofline-analysis/serving.py}}  <cfg>.json --concurrency <C> --both   # model 1 vs model 2
```

Write `<out>.md` beside the raw output and CSV, in the run's artifacts directory, `{{runs}}/<run_id>/artifacts/roofline/`, when the caller
names no directory. Give the reproduce commands at the top.

1. **Inputs** — one table: model shapes, numerics, hardware, sharding, workload (ISL, OSL,
   concurrency → requests per core, `mnbt` → number of prefill chunks), decode `kv_len`. Each row
   has a **source** column: file + pin, "user", or "assumed/carried from `<cell>`". The script's
   `numerics defaulted to …` line goes here verbatim.
2. **Headline, per core, with ×fleet and ×chip** — one table with a `formula` column:

   | metric | formula |
   | --- | --- |
   | T_prefill | Σ over `n_c` chunks of T_chunk(i) — also give the per-chunk average |
   | T_decode (= T_step = TPOT) | one decode step at `batch` — also give 1000/TPOT tok/s per user |
   | TTFT | = T_prefill (model 1, no queueing) |
   | E2E / request | TTFT + (OSL−1)·TPOT |
   | QPS_prefill, QPS_decode, QPS | 1/T_prefill · batch/(OSL·T_step) · reciprocal sum |
   | throughput | QPS·(ISL+OSL) per core, per chip, fleet; plus output tok/s fleet |
   | prefill share | which phase a win moves |
   | KV pool / gate | pool GiB, req/core admitted, % of the gate the configured batch uses |
   | HBM resident | weights + KV of capacity |

3. **Where the time goes, by block, per phase** — roll the flat op table up into blocks and give
   ms and % of phase: MLA/attention flash kernel · attention projections + RoPE · DSA indexer ·
   MoE routed-expert GEMMs · MoE shared expert · MoE local data movement (router, permute,
   unpermute, combine) · **collectives, each named with its ms** · dense FFN · norms/residual.
   Sum the rollup in Python from `--layer-csv` (`Role` in `op`/`fused`, column `Layer Total (ms)`)
   and check it equals the phase total. Classify collectives by `ICI` being the only non-zero time,
   not by op name — `Dispatch (hierarchical AG)` has no "gather" in it. Then the HBM / MXU / VPU /
   ICI boundness split, and the top rows in **ms**, not just percent. 30% of a 5 ms phase is not
   worth anyone's time.
4. **Per-layer latency** — one table per phase: layer-kind name (step 4) + key, count, ms/layer,
   ms per chunk (prefill), all-layer ms and %. Then the op table for the kind that owns each phase,
   per step 5, and one line saying what the other kinds add or remove relative to it.
5. **Hand recomputes** — at least four numbers recomputed in an executed Python script **from the
   formula**, not read back from the script: the largest compute row in prefill, a
   padding-affected row where one exists, the decode weight read (Σ weight MiB per step / hbm_bw),
   and throughput from T_prefill and T_step. A table with columns: number, formula, recomputed,
   script. A mismatch is a finding. Report it and do not smooth it over.
6. **Caveats** — hardware spec used; kernel efficiency 100% and collectives fully exposed, so the
   result is a ceiling; the share of each phase that is collectives, because that share has the
   widest error bar; numerics that were carried or defaulted; unmodelled terms (layout/copy, the
   quantize pass after bf16 staging writes, host/scheduler overhead across `n_c` steps); the
   **model 2 delta** from `serving.py --both`, with the reason one model fits this cell better —
   e.g. at 1 req/core there is nothing to mix, so model 2's steady-state closure does not apply;
   and anything in the output you could not reconcile.

Call the result **modelled**. It is not a measured result and it does not settle an A/B —
it tells you whether a benchmark is worth running.

---

## Changing `roofline.py`: check `model/common/` in the same edit

`model/common/` is the **specification** and {{script:roofline-analysis/roofline.py}} **implements** it. A change to
either is a claim about both, and the pair drifting apart is how this skill produces confident
wrong numbers. Decide which case yours is **before** you edit:

| what you found | fix |
| --- | --- |
| code contradicts a correct spec | code only — but say which spec line you were failing |
| spec is wrong or unqualified | **both**, spec first |
| neither says anything | **both**: the spec gets the rule, the code the arithmetic |

Two checks on any op you touch, each of which has caught a real bug:

1. **Does the output have a different width from the input?** KV-cache writes, top-k indices, fp8
   GEMMs accumulating to bf16. Every single-dtype helper in `roofline.py` has been wrong about one.
2. **Grep the other specs for the same shape of claim** — "kept in VMEM", "can be ignored if
   fused", "`* bytes`" with no dtype. A wrong assumption is rarely written once.

Then re-run `--selftest` **and** reconcile the op's FLOPs against the spec formula in Python: the
selftest pins the code against itself, not against `model/common/`.

## Files

| | |
| --- | --- |
| [`roofline.py`]({{script:roofline-analysis/roofline.py}}) | the arithmetic, one function per op; `weight_terms` / `kv_terms` split memory per tensor |
| [`derivation.py`]({{script:roofline-analysis/derivation.py}}) | one operating point, step by step, each step checked against `serving.py` — what {{skill:sharding-strategy}}'s worked examples print |
| [`README.md`]({{reference:roofline-analysis/README.md}}) | how it works and what it gets wrong |
| [`model/common/`]({{reference:roofline-analysis/model/common}}) | what ops a component has |
| [`model/architecture/`]({{reference:roofline-analysis/model/architecture}}) | per-model shapes |
| [`parallelism/`]({{reference:roofline-analysis/parallelism}}) | which collectives a sharding issues |
| [`COLLECTIVE.md`]({{reference:roofline-analysis/COLLECTIVE.md}}) | how to price a collective |
| [`tpu/`]({{reference:roofline-analysis/tpu}}) | hardware numbers |

Weak spots worth flagging when they matter: layout and copy time is not modelled at all (7% of
prefill in a real profile), collectives are assumed fully exposed when they are really only partly
so, and the fixed cost of a small collective is underpriced by about 2.6×.
