#!/usr/bin/env bash
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

# Single-host accuracy-check recipe for `compute submit`.
#
# Starts one vLLM server from vllm_server_cmd.txt, waits for /health, prints
# the harness and serving-stack versions, sends one smoke request, runs the
# lm_eval command in eval_cmd.txt verbatim through `eval`, prints the
# eval-summary lines, and exits with lm_eval's exit code.
#
# Inputs (staged next to this recipe with --input):
#   vllm_server_cmd.txt  the server command, run with `bash -c`
#   eval_cmd.txt         the lm_eval command, run with `eval`
#
# Arguments (KEY=VALUE, after `--`). An unknown key is an error.
#   SERVED_MODEL_NAME    required; the --served-model-name of the server
#   EVAL_TASKS           default mmlu_pro
#   EVAL_LIMIT           default 100; "full" runs the whole task (no --limit)
#   SERVER_PORT          default 8000; must match --port in the server command
#   SERVER_WAIT_TIMEOUT_S  default 3600; seconds to wait for /health
#   LM_EVAL_VERSION      default 0.5.0.dev1; installed only if lm_eval is absent
#
# Outputs (in the recipe's working directory, fetch with --output):
#   eval_out.tar.gz      the lm_eval --output_path tree (results_*.json, samples_*.jsonl)
#   server.log           the full vLLM server log

set -uo pipefail

WORKDIR="$(pwd)"
EVAL_OUTPUT_DIR="${WORKDIR}/eval_out"
SERVER_LOG="${WORKDIR}/server.log"
mkdir -p "${EVAL_OUTPUT_DIR}"
: > "${SERVER_LOG}"

SERVED_MODEL_NAME=""
EVAL_TASKS="mmlu_pro"
EVAL_LIMIT="100"
SERVER_PORT="8000"
SERVER_WAIT_TIMEOUT_S="3600"
LM_EVAL_VERSION="0.5.0.dev1"

SERVER_PID=""

pack_outputs() {
  tar -czf "${WORKDIR}/eval_out.tar.gz" -C "${WORKDIR}" eval_out 2>/dev/null \
    || tar -czf "${WORKDIR}/eval_out.tar.gz" --files-from /dev/null
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
}

finish() {
  local rc="$1"
  stop_server
  pack_outputs
  echo "eval exited rc=${rc}"
  exit "${rc}"
}

fail() {
  echo "ERROR: $*" >&2
  finish 1
}

for kv in "$@"; do
  key="${kv%%=*}"
  value="${kv#*=}"
  if [[ "${kv}" != *=* ]]; then
    fail "argument '${kv}' is not KEY=VALUE"
  fi
  case "${key}" in
    SERVED_MODEL_NAME|EVAL_TASKS|EVAL_LIMIT|SERVER_PORT|SERVER_WAIT_TIMEOUT_S|LM_EVAL_VERSION)
      printf -v "${key}" '%s' "${value}" ;;
    *)
      fail "unknown key '${key}'; accepted: SERVED_MODEL_NAME EVAL_TASKS EVAL_LIMIT SERVER_PORT SERVER_WAIT_TIMEOUT_S LM_EVAL_VERSION" ;;
  esac
done

[[ -n "${SERVED_MODEL_NAME}" ]] || fail "SERVED_MODEL_NAME is required"
[[ -s "${WORKDIR}/vllm_server_cmd.txt" ]] || fail "vllm_server_cmd.txt is missing or empty; pass it with --input"
[[ -s "${WORKDIR}/eval_cmd.txt" ]] || fail "eval_cmd.txt is missing or empty; pass it with --input"

VLLM_SERVER_CMD="$(cat "${WORKDIR}/vllm_server_cmd.txt")"
EVAL_CMD="$(cat "${WORKDIR}/eval_cmd.txt")"

# The eval command takes its endpoint inside --model_args; reject it before the
# server spends a model load if it cannot reach the server at all.
if [[ "${EVAL_CMD}" != *base_url* ]]; then
  fail "eval_cmd.txt does not mention base_url"
fi

if [[ "${EVAL_LIMIT}" == "full" || -z "${EVAL_LIMIT}" ]]; then
  EVAL_LIMIT_ARG=""
else
  EVAL_LIMIT_ARG="--limit ${EVAL_LIMIT}"
fi
EVAL_BASE_URL="http://localhost:${SERVER_PORT}/v1/chat/completions"
export SERVED_MODEL_NAME EVAL_TASKS EVAL_LIMIT EVAL_LIMIT_ARG EVAL_BASE_URL EVAL_OUTPUT_DIR SERVER_PORT

echo "effective SERVED_MODEL_NAME: ${SERVED_MODEL_NAME}"
echo "effective EVAL_TASKS: ${EVAL_TASKS}"
echo "effective EVAL_LIMIT: ${EVAL_LIMIT}"
echo "effective SERVER_PORT: ${SERVER_PORT}"
echo "effective VLLM_SERVER_CMD: ${VLLM_SERVER_CMD}"
echo "effective EVAL_CMD: ${EVAL_CMD}"

# Harness version. Install the pinned fallback only when the image lacks it,
# or into an isolated CPU-torch venv (/opt/lmeval) when system python has
# torch_tpu autoloaded (which otherwise tries to open /dev/vfio/0 on
# torch.manual_seed while vllm serve already holds the TPU device).
LMEVAL_BIN_DIR=""
if python3 -c "import torch_tpu" 2>/dev/null; then
  echo "torch_tpu detected in system python; installing lm-eval[api]==${LM_EVAL_VERSION} into isolated CPU venv /opt/lmeval to avoid /dev/vfio/0 contention"
  python3 -m venv /opt/lmeval || fail "venv /opt/lmeval creation failed"
  /opt/lmeval/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu || fail "CPU torch install failed"
  /opt/lmeval/bin/pip install -q "lm-eval[api]==${LM_EVAL_VERSION}" || fail "lm_eval venv install failed"
  LMEVAL_BIN_DIR="/opt/lmeval/bin"
  echo "lm_eval version: $(/opt/lmeval/bin/python -c 'import importlib.metadata as m; print(m.version("lm_eval"))' 2>/dev/null || echo unknown)"
else
  if ! python3 -c "import lm_eval" 2>/dev/null; then
    echo "lm_eval not present; installing lm-eval[api]==${LM_EVAL_VERSION}"
    if command -v uv >/dev/null 2>&1; then
      uv pip install --system --no-config "lm-eval[api]==${LM_EVAL_VERSION}" || fail "lm_eval install failed"
    else
      python3 -m pip install "lm-eval[api]==${LM_EVAL_VERSION}" || fail "lm_eval install failed"
    fi
  fi
  echo "lm_eval version: $(python3 -c 'import importlib.metadata as m; print(m.version("lm_eval"))' 2>/dev/null || echo unknown)"
fi
echo "vllm version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
echo "vllm-torchtpu commit: ${VLLM_TORCHTPU_COMMIT:-unknown}"

# Server, in its own process group so that stop_server reaches its workers.
setsid bash -c "${VLLM_SERVER_CMD}" >>"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
start=$(date +%s)
while true; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vllm-server exited before /health answered; last 200 lines of server.log:"
    tail -n 200 "${SERVER_LOG}"
    finish 1
  fi
  if curl -sf "http://localhost:${SERVER_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if (( $(date +%s) - start > SERVER_WAIT_TIMEOUT_S )); then
    echo "vllm-server not up after ${SERVER_WAIT_TIMEOUT_S}s; last 200 lines of server.log:"
    tail -n 200 "${SERVER_LOG}"
    finish 1
  fi
  sleep 10
done
echo "vllm-server is up after $(( $(date +%s) - start ))s"
echo "Eval endpoint: ${EVAL_BASE_URL}"

# One real answer before the run commits hours of TPU time.
python3 - "${EVAL_BASE_URL}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys
import urllib.request

url, model = sys.argv[1], sys.argv[2]
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
    "max_tokens": 512,
    "temperature": 0,
}).encode()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
except Exception as exc:  # report and continue; the eval decides
    print(f"smoke: request failed: {exc}")
    sys.exit(0)
choice = (out.get("choices") or [{}])[0]
msg = choice.get("message") or {}
usage = out.get("usage") or {}
print(f"smoke: finish_reason={choice.get('finish_reason')} completion_tokens={usage.get('completion_tokens')}")
content = msg.get("content")
if content is None:
    print("smoke: content=<null>")
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    print(f"smoke: content was null; reasoning={reasoning[:200]!r}")
else:
    print(f"smoke: content={content.strip()[:200]}")
PY

# The eval command runs verbatim. Nothing is appended to it.
if [[ -n "${LMEVAL_BIN_DIR}" ]]; then
  export PATH="${LMEVAL_BIN_DIR}:${PATH}"
fi
eval "${EVAL_CMD}"
rc=$?

# Summary lines, from the results and samples files lm_eval wrote.
python3 - "${EVAL_OUTPUT_DIR}" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
results = sorted(glob.glob(os.path.join(root, "**", "results_*.json"), recursive=True),
                 key=os.path.getmtime)
samples = glob.glob(os.path.join(root, "**", "samples_*.jsonl"), recursive=True)

if samples:
    nulls = 0
    for path in samples:
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                resps = rec.get("resps") or []
                flat = [r for group in resps for r in (group if isinstance(group, list) else [group])]
                if not flat or any(r is None or (isinstance(r, str) and not r.strip()) for r in flat):
                    nulls += 1
    print(f"eval-summary: null-content answers={nulls}")
else:
    print("eval-summary: null-content answers=unknown (no samples_*.jsonl; run with --log_samples)")

if not results:
    print("eval-summary: no results_*.json written")
    sys.exit(0)

with open(results[-1]) as f:
    data = json.load(f)
n_samples = data.get("n-samples") or {}
versions = data.get("versions") or {}
groups = data.get("groups") or {}
res = data.get("results") or {}


def line(task, metrics):
    parts = []
    for key, value in metrics.items():
        if key == "alias" or "_stderr" in key:
            continue
        stderr = metrics.get(key.replace(",", "_stderr,", 1))
        if isinstance(stderr, (int, float)):
            parts.append(f"{key}={value} +- {stderr}")
        else:
            parts.append(f"{key}={value}")
    ns = n_samples.get(task) or {}
    sample_len = metrics.get("sample_len") or ns.get("effective")
    if sample_len is None and task in groups:
        sample_len = sum((n_samples.get(t) or {}).get("effective", 0)
                         for t in res if t != task and t.startswith(task))
    print(f"eval-summary: {task} {' '.join(parts)} sample_len={sample_len} version={versions.get(task)}")


heads = [t for t in groups] or list(res)[:1]
for task in heads:
    line(task, groups.get(task) or res.get(task) or {})
for task, metrics in res.items():
    if task not in heads:
        line(task, metrics)
PY

finish "${rc}"
