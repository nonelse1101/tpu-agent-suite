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
"""Qwen3.5-35B-A3B-BF16 adapter for workload-microbenchmark. Reviewed code: change via PR.

BF16 variant of Qwen3.5-35B-A3B (40 layers: 30 Gated DeltaNet + 10 gated GQA layers,
256 routed experts top-8 + 1 shared expert per layer, 2 bytes per parameter and 2 bytes
per KV cache element by default).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from core.deploy import Deploy, Facts

MODEL_DIR = "Qwen3.5-35B-A3B-BF16"
MODEL_TYPES = ("qwen3_5_moe", "qwen3_5_moe_text")

#: Layer kind per `layer_types` entry (the model's own names, shortened).
KIND = {"linear_attention": "gdn", "full_attention": "attn"}
#: Twin layer stack: one layer of each kind, in served order (layer 0 is GDN, layer 3 attention).
TWIN_ORDER = ("gdn", "attn")


def _text(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def _layer_types(t: dict) -> list[str]:
    lt = t.get("layer_types")
    if lt:
        return list(lt)
    k = t.get("full_attention_interval", 4)
    return [
        "full_attention" if (i + 1) % k == 0 else "linear_attention"
        for i in range(t["num_hidden_layers"])
    ]


def _gdn_state_bytes(t: dict, tp: int = 1, spec: int = 0) -> int:
    """GDN state per sequence per layer on one TP rank: conv (bf16) + SSM (mamba_ssm_dtype)."""
    kh, vh = t["linear_num_key_heads"], t["linear_num_value_heads"]
    kd, vd = t["linear_key_head_dim"], t["linear_value_head_dim"]
    conv_dim = kd * kh * 2 + vd * vh
    conv = conv_dim // tp * (t["linear_conv_kernel_dim"] - 1 + spec) * 2
    ssm_b = 4 if str(t.get("mamba_ssm_dtype", "float32")) in ("float32", "fp32") else 2
    return conv + vh // tp * vd * kd * ssm_b


#: The checkpoint's text config, kept by facts() for the GDN dims Facts has no field for.
_GDN: dict[str, Any] = {}


def facts(cfg: dict, kv_cache_dtype: str = "bf16") -> Facts:
    if cfg.get("model_type") not in MODEL_TYPES:
        raise ValueError(
            f"{MODEL_DIR}: unexpected model_type {cfg.get('model_type')!r}")
    t = _text(cfg)
    kv_b = 1 if "fp8" in kv_cache_dtype else 2
    lt = _layer_types(t)
    kinds = tuple(KIND[x] for x in lt)
    n_attn = kinds.count("attn")
    hd = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    kv_layer = 2 * t["num_key_value_heads"] * hd * kv_b  # K + V, all KV heads
    mtp = t.get("mtp_num_hidden_layers", 0)
    _GDN.clear()
    _GDN.update({
        k: t[k]
        for k in ("linear_num_key_heads", "linear_num_value_heads",
                  "linear_key_head_dim", "linear_value_head_dim",
                  "linear_conv_kernel_dim")
    })
    _GDN["mamba_ssm_dtype"] = t.get("mamba_ssm_dtype", "float32")
    _GDN["head_dim"] = hd
    return Facts(
        name=MODEL_DIR,
        layers=t["num_hidden_layers"],
        dense_layers=0,
        hidden=t["hidden_size"],
        heads=t["num_attention_heads"],
        kv_heads=t["num_key_value_heads"],
        mla=False,
        experts=t.get("num_experts", 0),
        topk=t.get("num_experts_per_tok", 0),
        expert_ffn=t.get("moe_intermediate_size", 0),
        shared_ffn=t.get("shared_expert_intermediate_size", 0),
        dense_ffn=0,
        kv_cache_bytes_per_token_per_layer=kv_layer,
        mtp_layers=mtp,
        vocab=t.get("vocab_size", 0),
        kinds=kinds,
        kv_bytes_token=kv_layer * n_attn,
        state_bytes_seq=_gdn_state_bytes(t) * kinds.count("gdn"),
        mtp_kv_bytes_token=kv_layer * mtp,
    )


def _kv_heads_rank(f: Facts, d: Deploy) -> int:
    return max(1, f.kv_heads // max(1, d.tp))


def kv_bytes_per_token_per_rank(f: Facts, d: Deploy) -> float:
    return f.kv_bytes_token * _kv_heads_rank(f, d) / f.kv_heads / d.dcp


def state_bytes_per_seq_per_rank(f: Facts, d: Deploy) -> float:
    n_gdn = f.kinds.count("gdn")
    return _gdn_state_bytes(_GDN, max(1, d.tp), d.spec_tokens) * n_gdn


def per_rank_rows(f: Facts,
                  d: Deploy,
                  pool_tokens: int | None = None) -> dict[str, Any]:
    tp = max(1, d.tp)
    return {
        "attn_kv_heads_per_rank":
        _kv_heads_rank(f, d),
        "attn_kv_replicated":
        tp > f.kv_heads,
        "gdn_k_heads_per_rank":
        _GDN["linear_num_key_heads"] // tp,
        "gdn_v_heads_per_rank":
        _GDN["linear_num_value_heads"] // tp,
        "gdn_state_bytes_per_seq_per_layer_per_rank":
        _gdn_state_bytes(_GDN, tp, d.spec_tokens),
        "kv_bytes_per_token_per_rank_gqa":
        round(kv_bytes_per_token_per_rank(f, d), 1),
    }


PROJECTION_ROWS = [
    "attn_kv_heads_per_rank", "gdn_k_heads_per_rank", "gdn_v_heads_per_rank",
    "gdn_state_bytes_per_seq_per_layer_per_rank"
]

TWIN_CARRY_ARGS = [
    "attention-backend", "mamba-cache-mode", "language-model-only"
]

TRACE = {
    "embed_all_reduces": 1,
    "all_reduces_per_layer": 2,
    "backbone_by_all_reduces": True,
    "boundary_regex": r"^all-reduce\.\d+(\.cloned\.\d+\.call-done)?$",
    "census": {
        "families": {
            "gdn": {
                "regex": r"fused_conv1d_gdn",
                "part": "attention"
            },
            "attn": {
                "regex": r"^RPA[a-z]*-",
                "part": "attention"
            },
            "experts": {
                "regex": r"gmm|router_topk",
                "part": "ffn"
            }
        },
        "kinds": {
            "gdn": ["gdn", "experts"],
            "attn": ["attn", "experts"]
        },
    },
    "per_step_ops": [
        {
            "what": "RoPE cos/sin cache copy",
            "op": r"^copy$",
            "out": r"^bf16\[\d{5,},64\]$"
        },
        {
            "what": "RPA metadata schedule",
            "op": r"^rpa_metadata_schedule$"
        },
        {
            "what": "attention metadata H2D copy",
            "op": r"^copy-done$",
            "out": r"^s32\[\d+\]$"
        },
        {
            "what": "mRoPE position gather",
            "op": r"^gather_offload_async_done$",
            "out": r"^bf16\[\d+,64\]$"
        },
    ],
}

ROUTER = {"weight_suffix": "mlp.gate.weight", "zero_suffixes": [], "std": 0.05}

SAMPLING = {
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "source": "model.py default: the checkpoint's generation_config.json"
}

DRAFTER = {
    "file": "spec_decode/eagle3.py",
    "cls": "Eagle3Proposer",
    "replay": "run_dp_dummy_draft",
    "gate": "_dp_lockstep_sharded",
    "source_funcs":
    ["Eagle3Proposer.run_dp_dummy_draft", "Eagle3Proposer.propose"]
}


def kernels(env: dict[str, str], d: Deploy) -> dict[str, Any]:
    moe = (
        "fused_ep" if env.get("USE_MOE_FUSED_EP_KERNEL") == "1" else
        "gmm_sparsecore" if env.get("USE_MOE_SPARSE_CORE") == "1" else "gmm")
    knobs = {
        k: v
        for k, v in sorted(env.items())
        if k.startswith(("TPU_MOE", "USE_MOE", "ONEHOT_MOE", "MOE_",
                         "ENABLE_QUANTIZED", "TPU_ROPE", "TPU_GDN",
                         "TPU_ENABLE_GDN", "TPU_RPA", "VLLM_KV_CACHE_LAYOUT",
                         "VLLM_TPU_FUSED", "TPU_TOKEN_BUCKET"))
    }
    return {
        "gdn":
        "fused_conv1d_gdn (kernels/gdn/v3)" +
        (", dynamic tiling"
         if env.get("TPU_ENABLE_GDN_DYNAMIC_TILING") == "1" else ""),
        "attention":
        "ragged paged attention (CUSTOM)",
        "moe":
        moe,
        "linear":
        "bf16",
        "env":
        knobs,
    }


def twin_overrides(f: Facts,
                   real: Deploy,
                   twin_world: int,
                   pool_tokens: int | None = None,
                   ckpt: dict | None = None) -> dict[str, Any]:
    if twin_world != real.world:
        raise ValueError(
            f"{MODEL_DIR}: twin world {twin_world} != served world {real.world}; "
            "only the served world (mirror) is supported for this model")
    kinds = [k for k in TWIN_ORDER if k in f.layer_kinds]
    inv = {v: k for k, v in KIND.items()}
    hf: dict[str, Any] = {
        "text_config": {
            "num_hidden_layers": len(kinds),
            "layer_types": [inv[k] for k in kinds]
        }
    }
    server: dict[str, Any] = {
        "tensor-parallel-size": real.tp,
        "data-parallel-size": real.dp,
        "decode-context-parallel-size": real.dcp,
        "block-size": real.block_size,
        "max-model-len": real.max_model_len,
        "load-format": "dummy",
    }
    deviations = []
    if pool_tokens:
        server[
            "num-gpu-blocks-override"] = pool_tokens // real.dcp // real.block_size
        deviations.append({
            "what":
            "KV pool blocks",
            "real":
            "num_blocks from the server log (vLLM's reported pool tokens exclude the "
            "hybrid padding layers)",
            "twin":
            f"{server['num-gpu-blocks-override']} = reported pool tokens / page",
            "covered_by":
            "none: distinct pages per request, dummy KV values; no whole-pool op on this model's step path"
        })
    deviations.append({
        "what":
        "GDN recurrent state",
        "real":
        "per-request slots",
        "twin":
        "distinct non-null slot per request (forward_run _mamba_slots); "
        "dummy state values",
        "covered_by":
        "state read/write cost per request is the same; values differ"
    })
    return {
        "twin_world":
        twin_world,
        "twin_deploy":
        real.label(),
        "layers":
        kinds,
        "hf_overrides":
        hf,
        "server_args":
        server,
        "context_scale":
        1.0,
        "deviations":
        deviations,
        "problems": [],
        "verify_before_boot": [
            "hf_overrides.text_config.layer_types honoured (layer census: 1 gdn + 1 attn)",
            f"resolved page size {real.block_size} logged again by tpu_block_size_utils",
            "load-format dummy + language-model-only boot through vllm-torchtpu",
        ],
    }


def twin_facts(f: Facts, hf: dict) -> Facts:
    t = hf["text_config"]
    kinds = tuple(KIND[x] for x in t["layer_types"])
    return replace(
        f,
        layers=t["num_hidden_layers"],
        kinds=kinds,
        kv_bytes_token=f.kv_cache_bytes_per_token_per_layer *
        kinds.count("attn"),
        state_bytes_seq=(f.state_bytes_seq // max(1, f.kinds.count("gdn"))) *
        kinds.count("gdn"))


def param_bytes_per_rank(f: Facts,
                         d: Deploy,
                         layers: dict[str, int] | None = None) -> dict:
    """Estimated resident weight bytes on one rank (bf16 weights: 2 bytes/param across
    all linear, MoE, embedding, and router tensors)."""
    kinds = layers or f.layer_kinds
    D, tp = f.hidden, max(1, d.tp)
    hd = _GDN["head_dim"]
    w_b = 2  # bf16
    attn = w_b * (
        D * f.heads * hd * 2  # q + output gate
        + 2 * D * _kv_heads_rank(f, d) * hd * tp  # k, v (per rank x tp)
        + f.heads * hd * D) / tp
    kh, vh = _GDN["linear_num_key_heads"], _GDN["linear_num_value_heads"]
    kd, vd = _GDN["linear_key_head_dim"], _GDN["linear_value_head_dim"]
    gdn = w_b * (D * (2 * kh * kd + 2 * vh * vd) +
                 vh * vd * D) / tp  # in_proj qkvz + out_proj
    moe = w_b * ((f.experts / max(1, d.ep_group)) * 3 * D * f.expert_ffn +
                 3 * D * f.shared_ffn / tp) + 2 * D * f.experts
    per = {"gdn": gdn + moe, "attn": attn + moe}
    layer_b = sum(per[k] * n for k, n in kinds.items())
    mtp_b = f.mtp_layers * (per["attn"] + 2 * D * D)
    emb_b = 2 * 2 * f.vocab * D / tp
    return {
        "layers": int(layer_b),
        "mtp": int(mtp_b),
        "embed_head": int(emb_b),
        "total": int(layer_b + mtp_b + emb_b)
    }


COMPONENTS = {}
