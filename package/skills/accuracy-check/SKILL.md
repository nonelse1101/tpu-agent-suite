---
name: accuracy-check
description: Measures a model's accuracy on real hardware against a recorded baseline, using the eval recipe this skill ships and the protocol that produced that baseline, then reports whether the score clears the 1.5pp gate. Use before calling an optimization `accepted`, when asked to "run the accuracy check", "run MMLU-Pro", "check the eval score", "is this change quality-neutral", or to establish the correctness evidence a benchmarked run needs.
scripts:
- scripts/accuracy-check/eval_recipe.sh
---

# Accuracy check on TPU

You boot one vLLM server on a TPU slice and run `lm_eval` against it, then compare the score to
the baseline recorded for that config. The output is a verdict a caller can act on,
not a number.

This is the **expensive** tier. A real gate run is hours. {{skill:change-preflight}} owns the cheap tier
and is not gated by this skill; reach for that first.

## One size: `--limit 100`

`mmlu_pro` runs at `--limit 100` — 100 questions per subject, 14 subjects, **~1400 samples**.
That is the gate: there is no sizing decision to make and no larger run required before a change
is checked in. The full set (omit `--limit`, 12,032 samples) stays available for an absolute
quality number; it is never a precondition for `accepted`.

`--limit` is applied **per subject**, not overall, which is why 100 gives ~1400 and `--limit 5`
gives 70. Anything below the gate size is a plumbing test, not a result: say so in the report and
do not compare it to a baseline.

**The one rule that replaces sizing: baseline and candidate must be at the same size.** A
`--limit 100` score checked against a full-set baseline is not a paired comparison, and the
mismatch is worth more than the gate.

## The gate is 1.5pp, and it is not configurable

**A drop of more than 1.5 percentage points against the `mmlu_pro` baseline fails.** Fixed, for
every model and every config. The 1.5pp figure is calibrated on `mmlu_pro` at `--limit 100`; it is
not a generic tolerance to carry to another task.

## Earlier job records

The baseline and evidence job ids in this file (`j-…`) are records of runs on an earlier job
system. They are identifiers for provenance only. `{{runtime}} compute job` cannot read them back.

## Step 0: Get the baseline

**Get the baseline before anything else.** A baseline, the protocol that produced it, and the job
that measured it belong together. Take them, in this order, from:

1. the caller, if the caller gave you a baseline;
2. the accuracy record of an earlier run of this skill on the same config, in that run's artifacts
   directory, `{{runs}}/<earlier run_id>/artifacts/` (Step 5 writes it);
3. the per-model tables in Step 1 and "Server command and baseline, per model".

You need the baseline **for `task: mmlu_pro`** — score, stderr, `n`, tier, the server shape it was
measured at, and the `EVAL_CMD` that produced it. If a source offers baselines on other tasks,
discard them.

If no source has an accuracy baseline for the model, see "Server command and baseline, per model"
below — you are establishing one rather than checking one, and that changes what you report.

### The task is `mmlu_pro`. You will be offered others — do not take them.

**This skill's gate is defined against `mmlu_pro` and nothing else.** Ignore accuracy baselines on
any other task, no matter how they are described.

You will hit this on **Qwen3.5-397B-A17B-FP8**, which has two accuracy baselines:

| | `mmlu_pro` — **use this** | `gsm8k` — do not use |
|---|---|---|
| Recipe | `vllm-torchtpu-generic-eval` | `vllm-torchtpu-generic-**bench**`, `BENCH_CMD` swapped |
| Task source | `--tasks mmlu_pro` | `<InferenceX>/utils/evals/gsm8k.yaml`, cloned at runtime |
| Shape | limit-100, 1400 samples, thinking off | full set, 5-shot, `temperature=0` |
| Score | 0.8250 ± 0.0100 · `j-8846a1e0-1b60-4785-944c` | 0.9689 ± 0.0048 · `j-52e7603c-99b7-48a8-b1eb` |

If a caller explicitly asks for gsm8k, that is a serving job (`{{runtime}} compute submit --serve`)
with the gsm8k client in place of the benchmark client, not this skill. Say so rather than approximating it.

## Inputs you need

- **`DOCKER_IMAGE`** — reuse the image the caller gave you. **Do not build one.** A rebuild is
  the caller's decision, not a step you take silently. If no image was given, say so and stop.

  **The repo to use is your team's `vllm-torchtpu` image repository:**

  ```
  <registry>/vllm-torchtpu:<commit sha>
  ```

  **Prefer a commit-sha tag over `:latest`.** `:latest` is rolling — on 2026-09-04 it was
  `sha256:00990af4…`, built 2026-09-01, carrying vllm-torchtpu `27ff56aa0a295cdb43a5c835b4cb5f58b4cc7389`
  — and a comparison against a baseline is only about your change if the image is held fixed. If
  you must use `:latest`, record the commit the eval job prints, never the tag.
- **The config name** — e.g. `qwen3.5-35b-fp8-dp8-ep`. It selects both the protocol and the
  baseline, and it must match the layout actually being served. Not the model id: the same model
  has several configs and they do not share baselines.
- **`VLLM_SERVER_CMD`** — the server under test. When this run is gating a benchmark, pass the
  **same** command that benchmark used, read back from that benchmark job's own record of what ran.
  An accuracy number measured at a different shape describes a configuration nobody benchmarked.
- **`MODEL_ID`** — the HuggingFace id, org included. Serve it and pass
  `--served-model-name=<MODEL_ID>`; the recipe takes the same value as `SERVED_MODEL_NAME`.
  `vllm serve <MODEL_ID>` downloads the weights from HuggingFace on the target. A gated model needs
  the target's HuggingFace token. If the target mounts a shared HuggingFace cache, the weights
  download once; check what the target offers with `{{runtime}} compute inspect --json`.
- **the size** — `EVAL_LIMIT=100`, unless the caller explicitly asked for the full set.

## Step 1: The protocol is fixed — copy it, do not rebuild it

**There is one protocol per model, and it is whichever one produced the baseline you are comparing
against.** Read it out of the baseline record from Step 0 and copy it verbatim.
Do not rebuild it, do not merge the two below, and do not assume either generalizes — they differ
in every field that matters. The recipe this skill ships has no default `EVAL_CMD`: it exits
non-zero without one (see Step 2).

| | **Qwen3.5-397B-A17B-FP8** | **Qwen3.5-35B-A3B (BF16 + FP8 KV)** | **GLM-5.2-FP8** |
|---|---|---|---|
| Task | `mmlu_pro` (see Step 0) | `mmlu_pro` | `mmlu_pro` |
| Baseline recipe | `vllm-torchtpu-generic-eval` | `accuracy-check eval_recipe.sh` | `vllm-torchtpu-generic-eval-2node` |
| Slice | 1 node, TP=1 / DP=8 / EP | 1 node (`v6e-8`), TP=1 / DP=8 / EP | 2 nodes, DP=16 / EP over Ray |
| `--model_args` form | JSON | JSON | comma-kv |
| `num_concurrent` | 512 | 512 | 32 |
| Thinking disabled | yes, in two places | yes, in two places | no |
| `--log_samples` | yes | yes | no — **add it to a candidate** |
| `until` | task default | task default | `["<\|im_end\|>"]` |
| Wall clock, `--limit 100` | 1h29m | 3m10s (after 611s compile) | 3h53m |
| Baseline | 0.8250 ± 0.0100 · `j-8846a1e0-1b60-4785-944c` | 0.7607 ± 0.0111 · `acc-v6e8-dp8ep8-mmlupro-r3` (GPU BF16 ref: 0.7729 ± 0.0110) | 0.7500 ± 0.0113 · `j-012c51a9-a247-4d76-914b` |

Both baselines: commit `ec4bcecf78`, lm_eval 0.5.0.dev1, n=1400, **0 null-content answers**.

---

### Qwen3.5-397B-A17B-FP8 — 1 node

```
lm_eval --model local-chat-completions \
  --model_args "{\"model\": \"${SERVED_MODEL_NAME}\", \"base_url\": \"${EVAL_BASE_URL}\", \"enable_thinking\": false, \"num_concurrent\": 512}" \
  --tasks ${EVAL_TASKS} --apply_chat_template --log_samples \
  --gen_kwargs "{\"chat_template_kwargs\": {\"enable_thinking\": false}}" \
  --output_path ${EVAL_OUTPUT_DIR} ${EVAL_LIMIT_ARG} --seed 42
```

Only `${EVAL_TASKS}` and `${EVAL_LIMIT_ARG}` vary. Every other part is load-bearing:

- **`enable_thinking: false` appears twice, deliberately** — once in `model_args`, once in
  `gen_kwargs`.
- **`--log_samples` is not optional.** When a score comes back wrong, the samples are what tell
  you whether the model is emitting `<think>`, hitting the token cap, or answering fine and being
  mis-extracted. Without it that costs another multi-hour job.
- **JSON `--model_args` requires lm_eval 0.5.x.** 0.4.x parses it with `simple_parse_args_string`,
  which splits on `,` and `=` and silently produces garbage keys. `vllm/vllm-tpu:nightly` ships
  0.4.12; the torchtpu image and the recipe's fallback pin give 0.5.0.dev1.

### GLM-5.2-FP8 — 2 nodes

```
lm_eval --model local-chat-completions --model_args base_url=${EVAL_BASE_URL},model=${SERVED_MODEL_NAME},tokenizer=${SERVED_MODEL_NAME},max_gen_toks=4096,max_length=32768,num_concurrent=32,timeout=7200 --apply_chat_template --tasks ${EVAL_TASKS} --seed 42 ${EVAL_LIMIT_ARG} --gen_kwargs "max_new_tokens=4096,until=[\"<|im_end|>\"]" --output_path ${EVAL_OUTPUT_DIR}
```

**The baseline is sound — read its own log before believing otherwise.** From the eval log of
`j-012c51a9-a247-4d76-914b`:

```
lm_eval version: 0.5.0.dev1
smoke: finish_reason=stop completion_tokens=130
smoke: content=391
eval-summary: null-content answers=0
eval-summary: mmlu_pro exact_match,custom-extract=0.75 ... sample_len=1400
```

Two properties of the command travel with it, neither of which impugns the score:

1. **`until=["<|im_end|>"]` replaces `mmlu_pro`'s own `until: ['Question:']`** with a token GLM
   does not emit, so generation runs to the 4096-token cap on every sample. That costs wall clock,
   not accuracy — it is why the run takes 3h53m.
2. **No `--log_samples`**, so individual answers cannot be inspected afterwards. **Add it to a
   candidate run**: it changes what is written, never what is generated, and is the one safe
   departure from the baseline's command. The recipe this skill ships also counts null-content
   answers from the samples file, so without `--log_samples` it reports that count as `unknown`.

Thinking is not explicitly disabled here, against a server running `--reasoning-parser=glm45`.
With zero nulls there is no evidence this cost anything, and no evidence thinking was off — the
chain of thought may have gone to `reasoning` rather than `content`. Say which you know.

### Rules that apply to both

- **Reuse the baseline's command; do not "improve" on it.** A candidate measured under a different
  command is not comparable to its baseline, whatever either number looks like.
- **Both JSON arguments are wrapped in `"…"` with inner quotes escaped.** Escaping alone is not
  enough when the JSON contains spaces — the outer quotes keep it one argv element through the
  recipe's `eval`. Full quoting rules in Step 2.
- **A model with no generic-recipe baseline has no protocol to copy.** You are establishing one,
  not checking against one — see "Server command and baseline, per model".

### Record the harness version; do not assume a mapping

Read the **lm_eval version** and the **group and subtask versions** `mmlu_pro` reports out of the
results table (and the `versions` map in the results JSON), and record both in the accuracy
record's protocol field. When a candidate and its baseline differ on either, say so and do not
treat the delta as clean.

That is the whole rule. Do **not** carry a version-to-schema mapping here — the schema strings
have been observed to differ between lm_eval versions on the same task, so a table written today
would be wrong tomorrow. Record what the run printed and compare it to what the baseline printed.

The one version constraint that *is* confirmed is above: JSON `--model_args` needs 0.5.x. The pin
itself belongs in the recipe, not here.

### Server command and baseline, per model

The **authority for a server command is the record of a job that ran it** — the
`effective VLLM_SERVER_CMD:` line the recipe prints (Step 2), or the equivalent read-back of the
reference job — not a copy in this file. A second copy drifts, and the read-back is what actually
ran. Take the command from the reference job, change only what the caller is testing.

**Numbers are not kept in this file.** The baseline, the protocol that produced it and the
commands live in the baseline record; this table only says where to look. A number copied here
is a second copy that drifts.

| Model / config | Baseline recipe | Baseline job (`--limit 100`) |
|---|---|---|
| `qwen3.5-397b-fp8-tp1-dp8-ep`, agent shape | `vllm-torchtpu-generic-eval` | `j-8846a1e0-1b60-4785-944c` |
| `qwen3.5-35b-bf16-dp8ep8-v6e8`, `fp8_e4m3` KV | `accuracy-check eval_recipe.sh` | `acc-v6e8-dp8ep8-mmlupro-r3` (0.7607 ± 0.0111; paired GPU BF16: 0.7729 ± 0.0110) |
| `glm-5.2-fp8-dp16-ep` | `vllm-torchtpu-generic-eval-2node` | `j-012c51a9-a247-4d76-914b` |

Two consequences, and neither is optional:

- **To reproduce a baseline exactly, pass that tag.** A run that does not pin `DOCKER_IMAGE` to
  the baseline's image reproduces nothing.

**Job ids are full ids here on purpose.** The short form appears everywhere in prose and resolves
nowhere. Config names use the canonical `<model>-<quant>-<sharding>` form so that one name selects
one baseline.

**Only generic-recipe jobs count.** A baseline, and the protocol that produced it, is a
`vllm-torchtpu-generic-eval` or `-2node` job, or a run of this skill's recipe, with a recorded
protocol. Runs on a dedicated per-model recipe, numbers in
`<vllm-torchtpu>/scripts/vllm/benchmarking/baselines/eval/`, and protocols derived from
`<vllm-torchtpu>/configs/<config>.sh` or `run_eval_flow.sh` are **not** comparators — they were
measured at a different shape, under a different harness, or by a job that is attributable to no
commit.

**So there are exactly two things a score can be checked against:**

1. **The recorded `mmlu_pro` baseline** for this model — a generic-recipe job, measured at
   the agent's server shape, carrying the protocol and the job id that produced it. This is the
   normal case, since the accuracy run inherits the benchmark's `VLLM_SERVER_CMD`.
2. **A baseline you measure yourself**, when there is none. That is a run of this same skill on
   the baseline commit at the same size, recorded in the run's artifacts directory,
   `{{runs}}/<run_id>/artifacts/`. Say in the report that the baseline is yours and n=1,
   unreplicated.

**If the model has no generic-recipe run you are in case 2** — establishing a baseline, not
checking one. Measure it, record it, and report it as a baseline rather than as a gate result.
There is no protocol to derive from elsewhere.

## Step 2: Submit the job

The recipe is {{script:accuracy-check/eval_recipe.sh}}. In one job it starts the server from
`vllm_server_cmd.txt`, waits for `/health`, prints the versions, sends one smoke request, runs
`eval_cmd.txt` verbatim, prints the `eval-summary:` lines, and exits with `lm_eval`'s exit code.
The two commands travel as files, so they reach the recipe without a second layer of shell quoting.

Write the two files into `<work>`:

- `<work>/vllm_server_cmd.txt` — the server under test, one shell command line.
- `<work>/eval_cmd.txt` — the command from step 1, for THIS model.

Then submit:

```bash
{{runtime}} compute submit \
  --recipe {{script:accuracy-check/eval_recipe.sh}} --interpreter bash \
  --image <image> \
  --input <work>/vllm_server_cmd.txt:vllm_server_cmd.txt \
  --input <work>/eval_cmd.txt:eval_cmd.txt \
  --output eval_out.tar.gz --output server.log \
  --run-id <run_id> --name accuracy-check-<config> \
  -- SERVED_MODEL_NAME=<HF id> EVAL_TASKS=mmlu_pro EVAL_LIMIT=100
JOB_ID=$({{runtime}} compute job list --run-id <run_id> --json --no-refresh \
  | python3 -c "import json,sys;print(next(r['job_id'] for r in json.load(sys.stdin) if r['job_name']=='accuracy-check-<config>'))")
```

Pass `EVAL_LIMIT=full` only when the caller asked for the full set. The recipe accepts only the
keys `SERVED_MODEL_NAME`, `EVAL_TASKS`, `EVAL_LIMIT`, `SERVER_PORT`, `SERVER_WAIT_TIMEOUT_S` and
`LM_EVAL_VERSION`, and exits non-zero on any other key.

**The recipe is single-host.** The GLM-5.2 protocol needs two hosts. A 2-node run needs a
multi-host backend and a recipe that creates the Ray workload itself; this skill does not ship
one. Check the configured target with `{{runtime}} compute inspect --json`.
If it cannot hold the model's baseline layout, report that as the blocker and stop. Do not run the
model at a different layout and compare it to the baseline. A 2-node GLM-5.2 run must carry
the values its baseline ran with:

- `MODEL_ID=zai-org/GLM-5.2-FP8`, `SERVED_MODEL_NAME=zai-org/GLM-5.2-FP8`
- `EVAL_TASKS=mmlu_pro`, `EVAL_LIMIT=100`
- `VLLM_ENGINE_READY_TIMEOUT_S=9600`, and a server wait of the same length
- the server env `VLLM_DISABLE_COMPILE_CACHE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TORCHINDUCTOR_AUTOGRAD_CACHE=0`
- `EVAL_CMD` = the GLM command from step 1

**Always pass `EVAL_CMD` explicitly.** The recipe has no default for it and exits non-zero without
`eval_cmd.txt`. The baselines' original recipes carried a default `EVAL_CMD` that matched neither
baseline; do not reconstruct one.

Two more parameters matter when porting a command from a dedicated recipe:

- **`SERVER_PORT`** (default 8000) — the recipe builds `${EVAL_BASE_URL}` from it, so the
  `--port` in `VLLM_SERVER_CMD` must match. Porting GLM's command meant changing its 47903 to
  8000 (passing `SERVER_PORT=47903` would have worked equally).
- **Server environment** — put it in the server command: `export A=1 B=2 && vllm serve …` inside
  `VLLM_SERVER_CMD`. That has no limit on spaces in a value (`LIBTPU_INIT_ARGS`). An `export`
  there is how `VLLM_ENGINE_READY_TIMEOUT_S=9600` is set. When you raise it, raise the recipe's
  `SERVER_WAIT_TIMEOUT_S` (default 3600) to match, or the recipe stops waiting first.

**The recipe guards `EVAL_CMD` before it starts the server.** It checks that the string mentions
`base_url` and exits non-zero if not. It accepts both spellings, comma-kv **and** JSON.

**`EVAL_CMD` is run verbatim — nothing is appended to it.** This is the difference from a bench
recipe, which appends the server host: `vllm bench serve` takes its endpoint as `--host`,
`lm_eval` takes it inside `--model_args`. Use `${EVAL_BASE_URL}` (already the full
`/v1/chat/completions` URL), `${EVAL_OUTPUT_DIR}`, `${EVAL_TASKS}`, `${EVAL_LIMIT_ARG}` and
`${SERVED_MODEL_NAME}`; the recipe exports all of them before the `eval`.

### Quoting: the rule that still applies

The eval command is `eval`'d in the container. That means:

- **JSON needs its double quotes escaped as `\"`.** `eval` performs quote removal, so an
  unescaped `{"enable_thinking":false}` reaches the program as `{enable_thinking:false}` and dies
  in argument parsing. Escaped, it arrives intact:

  ```
  --gen_kwargs {\"chat_template_kwargs\":{\"enable_thinking\":false}}
  ```

  Verified both directions on 2026-09-01: escaped survives as one argv element; unescaped splits
  into `image:0`, `video:0`, which is exactly how job `j-09ab11f8` died.

  **The rule is about wrapping, not about spaces.** JSON wrapped in outer `"…"` may contain
  spaces freely — the baseline `EVAL_CMD` and its `--limit-mm-per-prompt` both do, and both work.
  It is *unwrapped* JSON that must not contain them. Do not strip spaces out of a working wrapped
  command to satisfy a rule it is not subject to.

- **A list-valued `gen_kwargs` keeps the baseline's spelling.** The GLM stop list is written with
  escaped double quotes inside the wrapped value:

  ```
  --gen_kwargs "max_new_tokens=4096,until=[\"<|im_end|>\"]"
  ```

  `lm_eval` parses the value with `literal_eval`, so `["<|im_end|>"]` and `['<|im_end|>']` are the
  same list. It reaches lm_eval as one argv element, `max_new_tokens=4096,until=["<|im_end|>"]`.
  The `|` characters are safe because they are inside the quotes.

**Confirm the image tag exists before submitting.** A tag that does not resolve schedules the
slice and then fails to pull the image, having reserved the hardware to do it:

```bash
gcloud artifacts docker tags list <repo path> --filter=tag:<tag> --format='value(tag)'
```

**Record the recipe version at submit time.** The recipe is part of this package, so the package
tag, `{{package_tag}}`, identifies what ran. Carry it into the report.

**Read back what was actually submitted**, on every job. The recipe prints every effective value
before it starts the server:

```bash
{{runtime}} compute job logs --job-id $JOB_ID | grep '^effective '
```

**Check the keys.** The recipe rejects an unknown key, so a mistyped `KEY=VALUE` fails the job
instead of being ignored. Still confirm `effective EVAL_LIMIT:` — that is the one that costs a
gate. Do the same for `effective EVAL_CMD:` and `effective VLLM_SERVER_CMD:`; those two are what
the report must quote.

## Step 3: Watch the job

```bash
{{runtime}} compute job wait --job-id $JOB_ID \
  --until "log:vllm-server is up" --fail-on-log '^eval exited rc=[1-9]' --json
{{runtime}} compute job wait --job-id $JOB_ID --until done --json
{{runtime}} compute job logs --job-id $JOB_ID
```

A gate run takes hours. Exit 3 from `job wait` means it is still waiting; do what its `next`
says, which while the gate runs is to run the same command again.

The job log carries the recipe's own lines. The server's output goes to `server.log`, which comes
back with the artifacts (Step 4). When the server exits or does not answer `/health` in time, the
recipe prints the last 200 lines of `server.log` into the job log before it exits.

The recipe's exit code is the job's status, so a failed job is a real failure. A successful job is
necessary and not sufficient: it says the process exited 0, not that the score is meaningful.

Lines to find in the job log, in order:

1. `lm_eval version: <v>` — expect **0.5.0.dev1**. A preceding `lm_eval not present; installing`
   is not a problem in itself: the recipe pins the fallback
   (`uv pip install --system --no-config "lm-eval[api]==0.5.0.dev1"`), so both paths land on the
   same version. It does tell you the image lacks the harness, which is worth noting. What
   actually matters is the version *printed*: if it differs from the baseline's, say so and do not
   treat the delta as clean.
2. `vllm version: <v>` and `vllm-torchtpu commit: <sha>` — the serving stack that produced the
   score. A score without them is not reproducible; carry both into the report. The recipe reads
   the commit from the image's `VLLM_TORCHTPU_COMMIT` variable and prints `unknown` when the image
   does not set it; then record the commit from the image tag and say where it came from.
3. `Eval endpoint: http://…/v1/chat/completions`
4. `smoke: content=<...>` — one real answer before the run commits hours of TPU time. A
   reasoning model may instead print `content=<null>` followed by `smoke: content was null;
   reasoning=…`, which is the chain of thought landing in `reasoning` rather than a broken
   server. Do not treat either as diagnostic on its own: GLM-5.2's baseline, served with
   `--reasoning-parser=glm45`, printed `content=391` and `finish_reason=stop`. The line that
   decides is `eval-summary: null-content answers=<n>` at the end.
5. `eval-summary: null-content answers=<n>` — see the task-version and null-answer notes below.
6. `eval-summary: <task> exact_match,…=<score> … sample_len=<n>`, then one row per subtask.
7. `eval exited rc=0`

## Step 4: Verify the results artifact

A score printed is not a score computed over what you asked for. Before reporting:

```bash
{{runtime}} compute job artifacts --job-id $JOB_ID
```

`eval_out.tar.gz` and `server.log` land at `{{runs}}/<run_id>/artifacts/<job_id>/`. Unpack
`eval_out.tar.gz` there: it holds `results_<ts>.json`, and `samples_*.jsonl` only with
`--log_samples`.

**Do not assume `results_*.json` exists.** A run that failed before lm_eval wrote it has an
empty archive. The `eval-summary:` lines in the job log carry score, stderr and
`sample_len`, so the log is a complete substitute for the JSON.

Confirm that **`sample_len` matches the size you ran** — ~1400 for `--limit 100`, 12,032 for the full set. A mismatch means the run answered a different question — treat it exactly as the bench skill treats a typo'd recipe key.

**Sanity-check the magnitude before believing it.** MMLU-Pro is 10-way, so a score near 0.10 is
chance and means the answers are not being parsed at all — read a sample rather than reporting it.
A score far below baseline is more often a protocol error than a regression; step 1 is where to
look first.

## Step 5: Report

```
accuracy:
  config:      <config name>, <1node | 2node>
  image:       <image path>
  recipe:      accuracy-check eval recipe · <package tag>
  server:      <VLLM_SERVER_CMD, as read back from the `effective VLLM_SERVER_CMD:` line>
  eval:        <EVAL_CMD, as read back>, protocol derived from <files read>
  job:         <job id> · <job status>
  harness:     lm_eval <version, from the log>
  score:       <task> <metric>=<value> ± <stderr> (n=<sample_len>)
  baseline:    <value> ± <stderr> (n=<samples>) from <baseline record>,
               measured at <which server shape>
  size:        <--limit 100 (n~1400, the gate) | full (no --limit, n=12032) | plumbing (smaller)>
  paired:      <is the baseline at the same size? if not, say so — the delta is not comparable>
  gate:        1.5pp drop — <delta vs baseline, and whether it clears>
  verdict:     <✅ passed | ❌ failed | ⚠️ inconclusive> — <why>
```

`not recorded` is not a value here. If a field is empty because that work did not happen, say so
in words — a hollow block read quickly looks like a check that ran.

**Quote numbers, never summarize them**, and quote the commands as read back from the job log
rather than as you meant to send them.

`⚠️ inconclusive` is the honest verdict for any run below `--limit 100`, whatever the score was,
and for a run whose baseline was measured at a different size — that delta is not a paired
comparison. It is not a failure of the run; it is a statement about what the run can support.

`verdict:` always takes one of the three values above. Do not use `provisional` as a verdict.
If the baseline carries a limitation, write the qualifier **inside** the verdict rather than
replacing it: `✅ passed (provisional — <limitation>)`.

Report a regression plainly. A score that drops is the outcome this check exists to produce, and
it is worth more to your caller than the hours it saves.

**Then close the loop: record the run.** Write the report block to the run's artifacts directory,
`{{runs}}/<run_id>/artifacts/accuracy-check-<config>-<job_id>.md` — score, stderr, `n`, tier, the
baseline it was checked against, the protocol, and the job id. That is what makes this run's
number the next run's baseline, and it is the reason Step 0 has anything to read.

## Verified mechanisms

These records come from runs of the earlier generic eval recipes. They establish the protocol
and the harness behaviour this skill relies on.

> **The eval path was verified end to end on 2026-09-01** by `j-fa9c39fa-e4de-4c0f-a8c9`
> (`Succeeded`, 1 node, Qwen3.5-35B-A3B-FP8 DP8/EP, `--limit 5`). `lm_eval version: 0.5.0.dev1`
> came from the image with no runtime install; checkpoint copy **4.0 GiB/s** for 35 GB;
> `vllm-server is up after 955s`; eval 14.4 min for 70 samples; `results_*.json` landed in
> the job's output directory; the `eval-summary` lines printed. Score 0.400 — later shown to be a protocol
> error, not a model result.
>
> **The protocol was validated on 2026-09-01** by `j-0a9b1f22-f43d-4c6d-a4e1` (`Succeeded`, same
> config, single change: `--gen_kwargs {\"chat_template_kwargs\":{\"enable_thinking\":false}}`).
> `mmlu_pro exact_match,custom-extract = 0.7571 ± 0.0515` at n=70, against 0.400 for the same
> config with thinking left on. Server up in 600 s. That is the evidence for the `enable_thinking`
> rule above; it establishes the field matters, not that any external number is reproduced.
>
> **The full path was validated end to end on Qwen3.5-397B on 2026-09-02** by the three-tier sweep
> `j-b1817590` / `j-8846a1e0` / `j-3c80a2b6` (TP=1/DP=8/EP, agent shape, lm_eval 0.5.0.dev1,
> `mmlu_pro` group 2.0 / subtasks 3.0, as read from `results_*.json`). Scores 0.8314 ± 0.0139 (n≈700), 0.8250 ± 0.0100 (n≈1400),
> **0.8201 ± 0.0034 (n=12,032)** — converging monotonically, every tier inside the previous one's
> interval. All three are generic-recipe jobs, which is what makes them the authority here, and
> they are the evidence behind the tier table at the top of this file. `j-8846a1e0` was deleted
> afterwards; its score was recovered from its archived eval log.
>
> **GLM-5.2's baseline** is `j-012c51a9-a247-4d76-914b`, generic 2-node, `--limit 100`,
> 0.7500 ± 0.0113 at n=1400 with **0 null-content answers**, lm_eval 0.5.0.dev1, 3h53m. There is
> no generic-recipe full-set run for this model.
>
> **Three failure modes cost a job each, all fixed by rules above and `eval_recipe.sh`.**
> `j-bd2a68c6-fbad-410f-9e05` died at `ValueError: Weight input_size_per_partition = 64 is not
> divisible by weight quantization block_k = 128` — TP=8 shards this model's expert `down_proj`
> below the fp8 block-quant floor; the DP8/TP1 layout is the one that works.
> `j-09ab11f8-b120-49c0-bed2` died at `--limit-mm-per-prompt: Value image:0 cannot be converted`,
> the unescaped-JSON trap. Both were red, correctly: the recipe propagates the exit code.
> `acc-v6e8-dp8ep8-mmlupro-r2` died at `RuntimeError: Bad StatusOr access: INTERNAL: Failed to open
> /dev/vfio/0: Device or resource busy` when `lm_eval` ran in the container's system Python (where
> `torch_tpu` is autoloaded and tries to acquire the TPU device on `torch.manual_seed` while `vllm
> serve` already holds it). Fixed in `eval_recipe.sh` by installing `lm-eval[api]` with CPU-only
> `torch` into an isolated `/opt/lmeval` venv whenever `torch_tpu` is present in system Python.
>
> Correctness of the *model* was not at issue in any of these — they establish the harness.
