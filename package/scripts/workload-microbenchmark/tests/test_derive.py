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
"""CPU tests for core/jobdata.py, core/derive.py, core/cells.py and core/deploy.py: serve args, overrides, reference derivations, recipes and env audit.

Run from scripts/workload-microbenchmark/: `python3 -m unittest discover tests`.
"""

import glob
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

from core import cells, derive, jobdata, project  # noqa: E402
from core.deploy import Deploy, apply_overrides  # noqa: E402


class ServeArgs(unittest.TestCase):

    def test_array_form_with_appended_flag_and_redirect(self):
        script = '''VLLM_ARGS=(
  vllm serve "$MODEL_PATH"
  --tensor-parallel-size=16
  --block-size=256
  --async-scheduling
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  "${OFFLOAD_ARGS[@]}"
)
"${VLLM_ARGS[@]}" --no-dcp-q-replicate 2>&1 | tee "$LOG"
'''
        args, model = jobdata._parse_serve_args(script)
        self.assertEqual(model, ["$MODEL_PATH"])
        self.assertEqual(args["tensor-parallel-size"], "16")
        self.assertIs(args["async-scheduling"], True)
        self.assertIs(args["no-dcp-q-replicate"], True)  # not "2>&1"
        self.assertEqual(
            json.loads(args["speculative-config"])["method"], "mtp")

    def test_plain_line_form(self):
        args, _ = jobdata._parse_serve_args(
            "vllm serve m --max-num-seqs 8 --enable-prefix-caching > log")
        self.assertEqual(args, {
            "max-num-seqs": "8",
            "enable-prefix-caching": True
        })


class Recompute(unittest.TestCase):
    D = Deploy(block_size=256,
               dcp=16,
               prefix_caching=True,
               spec_method="mtp",
               spec_tokens=3)

    def _req(self, conv, isl, osl, phase="profiling"):
        return {"conv": conv, "isl": isl, "osl": osl, "phase": phase}

    def test_floor_is_one_block_plus_misalignment_with_eagle_drop(self):
        # prior turn 100,000 + 200 out; next prompt appends 500 tokens
        reqs = [
            self._req("a", 100_000, 200, "warmup"),
            self._req("a", 100_700, 50)
        ]
        r = cells.recompute(reqs, self.D)[0]
        G = 4096
        self.assertEqual(r["hit"], (100_200 // G) * G -
                         G)  # full blocks of the prefix, minus one
        self.assertTrue(G < r["computed"] <= 2 * G + 700)

    def test_no_eagle_no_drop(self):
        d = Deploy(block_size=256, dcp=16, prefix_caching=True)
        reqs = [
            self._req("a", 100_000, 200, "warmup"),
            self._req("a", 100_700, 50)
        ]
        self.assertEqual(
            cells.recompute(reqs, d)[0]["hit"], (100_200 // 4096) * 4096)

    def test_cold_request_computes_everything(self):
        r = cells.recompute([self._req("b", 50_000, 10)], self.D)[0]
        self.assertEqual(r["computed"], 50_000)

    def test_block_granularity_scales_with_dcp(self):
        self.assertEqual(self.D.cache_block_tokens, 4096)
        self.assertEqual(
            apply_overrides(self.D, {
                "dcp": "8"
            }).cache_block_tokens, 2048)


class Overrides(unittest.TestCase):

    def test_unknown_key_is_an_error(self):
        with self.assertRaises(ValueError):
            apply_overrides(Deploy(), {"dcpp": "8"})

    def test_env_overrides_do_not_touch_deploy(self):
        self.assertEqual(apply_overrides(Deploy(dcp=2), {"env.X": "1"}),
                         Deploy(dcp=2))


def _reference():
    p = sorted(
        glob.glob(
            os.path.join(SCRIPTS, "GLM-5.2-FP8", "config", "*.source.json")))
    return json.load(open(p[0])) if p else None


@unittest.skipIf(_reference() is None, "no GLM-5.2-FP8 reference snapshot")
class GlmReference(unittest.TestCase):
    snap = _reference()

    def _twin(self, world=8):
        cfg = derive.derive(self.snap, "GLM-5.2-FP8")
        return cfg, project.project(cfg, self.snap,
                                    derive.load_model("GLM-5.2-FP8"), world)

    def test_mirror_mode_scales_nothing_and_spans_hosts(self):
        """twin_world == the served world: only the layer count shrinks; nothing deviates."""
        cfg, tw = self._twin(16)
        f, hf, sv = cfg["facts"], tw["twin"]["hf_overrides"], tw["twin"][
            "server_args"]
        self.assertEqual(tw["mode"], "mirror")
        self.assertEqual((tw["twin"]["multihost"], tw["twin"]["hosts"]),
                         (True, 2))
        self.assertEqual(tw["context_scale"], 1.0)
        self.assertEqual((hf["num_attention_heads"], hf["n_routed_experts"],
                          hf["num_experts_per_tok"], hf["intermediate_size"]),
                         (f["heads"], f["experts"], f["topk"], f["dense_ffn"]))
        self.assertEqual(
            (sv["tensor-parallel-size"], sv["decode-context-parallel-size"]),
            (cfg["deploy"]["tp"], cfg["deploy"]["dcp"]))
        self.assertEqual(hf["num_hidden_layers"], 3)
        self.assertEqual([
            r["per_rank"]
            for r in tw["projection"] if r["status"] == "deviation"
        ], [])
        _, sc = self._twin(8)
        self.assertEqual((sc["mode"], sc["twin"]["multihost"]),
                         ("scaled", False))

    def test_twin_rule_preserves_per_rank_shapes(self):
        """global' / degree' == global / degree for every scaled tensor."""
        cfg, tw = self._twin()
        d, f, t = cfg["deploy"], cfg["facts"], tw
        hf, sv = tw["twin"]["hf_overrides"], tw["twin"]["server_args"]
        ep_real, ep_twin = d["tp"] * d["dp"], sv["tensor-parallel-size"] * sv[
            "data-parallel-size"]
        self.assertEqual(
            hf["num_attention_heads"] // sv["tensor-parallel-size"],
            f["heads"] // d["tp"])
        self.assertEqual(hf["n_routed_experts"] // ep_twin,
                         f["experts"] // ep_real)
        self.assertEqual(hf["num_experts_per_tok"] / ep_twin,
                         f["topk"] / ep_real)
        self.assertEqual(hf["intermediate_size"] // sv["tensor-parallel-size"],
                         f["dense_ffn"] // d["tp"])
        # the per-rank KV page is the served one (page-granular work matches), and so is the
        # per-rank KV pool; the prefix-cache block (block x DCP) is not used by forward_run
        self.assertEqual(sv["block-size"], d["block_size"])
        pool = {
            r["per_rank"]: r
            for r in tw["projection"]
        }["kv_pool_tokens_per_rank"]
        self.assertEqual(pool["real"], pool["twin"])
        self.assertEqual(t["layers"], ["dense_idx", "moe_idx", "moe"])
        # every row the rule controls matches; only group-size rows and the shared expert deviate
        dev = {
            r["per_rank"]
            for r in tw["projection"] if r["status"] == "deviation"
        }
        self.assertEqual(
            dev, {
                "heads_in_attention", "indexer_merge_width",
                "shared_expert_ffn_per_rank"
            })
        self.assertEqual(hf["index_topk_pattern"], ["F", "F", "S"])

    def test_fp8_skip_list_follows_layer_kind(self):
        """Routers, indexer projections and MTP projections keep their served dtype in the twin."""
        q = self._twin()[1]["twin"]["hf_overrides"]["quantization_config"]
        skip = set(q["modules_to_not_convert"])
        self.assertIn("model.layers.1.mlp.gate",
                      skip)  # twin layer 1 = MoE + indexer
        self.assertIn("model.layers.1.self_attn.indexers_proj", skip)
        self.assertNotIn("model.layers.2.self_attn.indexers_proj",
                         skip)  # reuse layer
        self.assertIn("model.layers.3.eh_proj",
                      skip)  # MTP layer is id 3 in the twin
        self.assertFalse(any(m.startswith("model.layers.4") for m in skip))

    def test_counts_reproduce_the_server(self):
        c = derive.derive(self.snap, "GLM-5.2-FP8")["cells"]
        self.assertAlmostEqual(c["recompute"]["model_over_server"],
                               1.0,
                               delta=0.03)
        self.assertAlmostEqual(sum(p["weight"] for p in c["prefill_cells"]),
                               1.0,
                               delta=0.01)
        self.assertAlmostEqual(sum(q["weight"] for q in c["decode_cells"]),
                               1.0,
                               delta=0.01)

    def test_dcp1_is_refused_with_alternatives(self):
        cfg = derive.derive(self.snap, "GLM-5.2-FP8", {"dcp": "1"})
        self.assertFalse(cfg["feasibility"]["feasible"])
        self.assertNotIn("cells", cfg)
        self.assertGreater(
            cfg["feasibility"]["kv"]["min_dcp_for_recorded_peak"], 1)

    def test_scheduling_override_is_labelled_L2(self):
        cfg = derive.derive(self.snap, "GLM-5.2-FP8", {"mnbt": "8192"})
        self.assertTrue(cfg["decidable_at"].startswith("L1 per-layer"))
        self.assertEqual(
            derive.derive(
                self.snap, "GLM-5.2-FP8",
                {"env.TPU_STREAMIDX_CHUNK_TOKENS": "128"})["decidable_at"],
            "L1")


class RecipeEval(unittest.TestCase):
    """The recipe script is evaluated as bash would, and nothing in it actually runs."""
    SCRIPT = r'''set -euo pipefail
gsutil -m rsync -r gs://x /root/ckpt || exit 1
/bin/touch "$MARK"
echo hi > "$MARK.redirect"
if [ "${USE_MOE_FUSED_EP_KERNEL:-0}" = "1" ]; then
  export TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER="${TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER:-1}"
fi
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }--flag=$(( 2 * 4096 ))"
MNBT="16384"
if [ "$JOBSET_JOB_INDEX" = "0" ]; then
  until ray status 2>/dev/null | grep -q "16.0 TPU"; do sleep 5; done
  vllm serve /root/ckpt \
    --tensor-parallel-size=8 \
    --max-num-batched-tokens=${MNBT} \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' 2>&1 | tee "$MARK.log"
else
  ray start --block
fi
'''

    def _run(self, env):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mark = os.path.join(tmp, "ran")
            ev = jobdata._eval_script(self.SCRIPT, {
                "JOBSET_JOB_INDEX": "0",
                "MARK": mark,
                **env
            })
            self.assertFalse(os.path.exists(mark),
                             "an absolute-path command ran")
            self.assertFalse(os.path.exists(mark + ".redirect"),
                             "a file redirect was written")
            self.assertFalse(os.path.exists(mark + ".log"))
            return ev

    def test_args_expanded_and_env_from_branches(self):
        argv, env = self._run({
            "USE_MOE_FUSED_EP_KERNEL": "1",
            "LIBTPU_INIT_ARGS": "--base"
        })
        args, model = jobdata._args_from_tokens(argv)
        self.assertEqual(model, ["/root/ckpt"])
        self.assertEqual(args["max-num-batched-tokens"],
                         "16384")  # shell var expanded
        self.assertEqual(
            json.loads(args["speculative-config"])["method"], "mtp")
        self.assertEqual(env["TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER"],
                         "1")  # branch taken
        self.assertEqual(env["LIBTPU_INIT_ARGS"], "--base --flag=8192")

    def test_branch_not_taken(self):
        _, env = self._run({"USE_MOE_FUSED_EP_KERNEL": "0"})
        self.assertNotIn("TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER", env)
        self.assertEqual(env["LIBTPU_INIT_ARGS"], "--flag=8192")

    def test_never_reaches_vllm(self):
        self.assertIsNone(
            jobdata._eval_script(
                "python3 -m vllm.entrypoints.api_server --x 1\n", {}))


class PrefixChainAndConcurrency(unittest.TestCase):

    def test_chain_and_gap_fallback(self):
        reqs = [
            {
                "conv": "a",
                "turn": 0,
                "phase": "warmup",
                "t0": 0,
                "isl": 100,
                "osl": 50
            },
            {
                "conv": "a",
                "turn": 1,
                "phase": "profiling",
                "t0": 1,
                "isl": 200,
                "osl": 80
            },
            # shorter than turn 1's prompt+output (280): falls back to turn 0 (150)
            {
                "conv": "a",
                "turn": 2,
                "phase": "profiling",
                "t0": 2,
                "isl": 260,
                "osl": 10
            }
        ]
        r = jobdata.prefix_chain(reqs)
        self.assertEqual([(x["reusable"], x["src_turn"], x["gap"]) for x in r],
                         [(0, None, False), (150, 0, False), (150, 0, True)])
        self.assertEqual(r[2]["new"], 110)

    def test_concurrency_time_weighted(self):
        reqs = [{
            "phase": "profiling",
            "t0": 0,
            "t1": 10
        }, {
            "phase": "profiling",
            "t0": 0,
            "t1": 2
        }, {
            "phase": "warmup",
            "t0": 0,
            "t1": 10
        }]
        c = jobdata.concurrency(reqs)
        self.assertEqual(c["time_share"], {1: 0.8, 2: 0.2})
        self.assertEqual((c["p50"], c["p90"], c["max"]), (1, 2, 2))


class EnvAudit(unittest.TestCase):
    REC = {
        "serve_parse": "bash-eval",
        "env_from_script": {
            "LIBTPU_INIT_ARGS": "--x"
        }
    }
    ENV = {
        "LIBTPU_INIT_ARGS": "--x",
        "USE_MOE_SPARSE_CORE": "1",
        "VLLM_DCP_Q_REPLICATE": "0"
    }

    def test_sources(self):
        rows = {
            r["name"]: r
            for r in derive.env_provenance(self.REC, self.ENV,
                                           {"env.USE_MOE_SPARSE_CORE": "1"})
        }
        self.assertEqual(rows["LIBTPU_INIT_ARGS"]["source"], "script")
        self.assertEqual(rows["USE_MOE_SPARSE_CORE"]["source"], "override")
        self.assertEqual(rows["VLLM_DCP_Q_REPLICATE"]["source"], "container")
        self.assertFalse(any("review" in r
                             for r in rows.values()))  # no resharding

    def test_resharding_flags_sharding_env_for_review(self):
        rows = {
            r["name"]: r
            for r in derive.env_provenance(self.REC, self.ENV, {
                "dcp": "8",
                "env.VLLM_DCP_Q_REPLICATE": "1"
            })
        }
        self.assertIn("review", rows["LIBTPU_INIT_ARGS"])
        self.assertNotIn("review",
                         rows["VLLM_DCP_Q_REPLICATE"])  # set explicitly
        self.assertNotIn("review",
                         rows["USE_MOE_SPARSE_CORE"])  # not a sharding knob

    def test_unevaluated_script_is_labelled(self):
        rows = derive.env_provenance({"serve_parse": "regex-fallback"},
                                     {"A": "1"}, {})
        self.assertIn("NOT evaluated", rows[0]["source"])

    def test_node_ids_are_plumbing(self):
        self.assertEqual(
            derive.server_env({
                "TPU_NODE_ID": "0",
                "TPU_WORKER_ID": "0",
                "X": "1"
            }), {"X": "1"})


class SnapshotFormat(unittest.TestCase):
    """Snapshots are written one record per line and read back unchanged."""

    def test_round_trip_and_layout(self):
        snap = {
            "job_id": "j-1",
            "requests": [{
                "conv": i,
                "isl": 100 + i,
                "osl": 20
            } for i in range(3)],
            "step_slices": {
                "rows": [[float(i), 0, 1] for i in range(40)]
            },
            "recipe": {
                "script": "x" * 300
            },
        }
        text = jobdata.dumps_snapshot(snap, width=80)
        self.assertEqual(json.loads(text), snap)
        lines = text.splitlines()
        # A record that fits stays on one line.
        self.assertIn('  {"conv": 0, "isl": 100, "osl": 20},', lines)
        # Short flat rows share a line.
        rows = [line for line in lines if line.strip().startswith("[0.0")]
        self.assertIn("[1.0, 0, 1]", rows[0])
        # An object too long for the width is expanded, a long string is not split.
        self.assertTrue(
            any(line.startswith('  "script": "xxx') for line in lines))


class Qwen35_35B_A3B_AdapterTest(unittest.TestCase):
    """Unit tests for Qwen3.5-35B-A3B-FP8 and Qwen3.5-35B-A3B-BF16 model adapters."""

    CKPT = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
            "full_attention_interval": 4,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "mtp_num_hidden_layers": 1,
            "vocab_size": 248320,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "modules_to_not_convert": [
                "model.language_model.layers.0.mlp.gate",
                "model.language_model.layers.3.mlp.gate",
            ],
        },
    }

    def test_fp8_facts_and_twin_mirror(self):
        mod = derive.load_model("Qwen3.5-35B-A3B-FP8")
        f = mod.facts(self.CKPT, kv_cache_dtype="fp8")
        self.assertEqual(f.layers, 40)
        self.assertEqual(f.kinds.count("gdn"), 30)
        self.assertEqual(f.kinds.count("attn"), 10)
        self.assertEqual(f.kv_cache_bytes_per_token_per_layer, 1024)
        self.assertEqual(f.kv_bytes_token, 10240)
        # 30 GDN layers x (2,097,152 B SSM + 49,152 B conv) = 64,389,120 B
        self.assertEqual(f.state_bytes_seq, 30 * (2_097_152 + 49_152))

        d_tp2 = Deploy(tp=2, dp=4, ep=True, block_size=256, max_model_len=9216)
        self.assertEqual(mod.kv_bytes_per_token_per_rank(f, d_tp2), 5120.0)
        rows_tp2 = mod.per_rank_rows(f, d_tp2)
        self.assertEqual(rows_tp2["attn_kv_heads_per_rank"], 1)
        self.assertFalse(rows_tp2["attn_kv_replicated"])
        self.assertEqual(rows_tp2["gdn_k_heads_per_rank"], 8)
        self.assertEqual(rows_tp2["gdn_v_heads_per_rank"], 16)

        # At TP=8, KV heads (2) replicate so each rank holds 1 KV head
        d_tp8 = Deploy(tp=8, dp=1, ep=True, block_size=256, max_model_len=9216)
        rows_tp8 = mod.per_rank_rows(f, d_tp8)
        self.assertEqual(rows_tp8["attn_kv_heads_per_rank"], 1)
        self.assertTrue(rows_tp8["attn_kv_replicated"])

        tw = mod.twin_overrides(f, d_tp2, twin_world=8, ckpt=self.CKPT)
        self.assertEqual(tw["layers"], ["gdn", "attn"])
        tf = mod.twin_facts(f, tw["hf_overrides"])
        self.assertEqual(tf.layers, 2)
        self.assertEqual(tf.kinds, ("gdn", "attn"))

    def test_bf16_facts_and_weights(self):
        mod_fp8 = derive.load_model("Qwen3.5-35B-A3B-FP8")
        mod_bf16 = derive.load_model("Qwen3.5-35B-A3B-BF16")
        f_fp8 = mod_fp8.facts(self.CKPT, kv_cache_dtype="fp8")
        f_bf16 = mod_bf16.facts(self.CKPT, kv_cache_dtype="bf16")
        self.assertEqual(f_bf16.kv_bytes_token, 2 * f_fp8.kv_bytes_token)
        d = Deploy(tp=2, dp=4, ep=True)
        w_fp8 = mod_fp8.param_bytes_per_rank(f_fp8, d)
        w_bf16 = mod_bf16.param_bytes_per_rank(f_bf16, d)
        self.assertGreater(w_bf16["layers"], 1.95 * w_fp8["layers"])


if __name__ == "__main__":
    unittest.main()
