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
"""Canonical model specifications and numerics presets for roofline-analysis.

Each entry corresponds to a verified architecture specification under
`references/roofline-analysis/model/architecture/` and `configs/*.json`.
"""

from __future__ import annotations

from roofline import Model, Numerics

QWEN3_5_35B_A3B = Model(
    name="Qwen3.5-35B-A3B-FP8",
    layers=40,
    dense_layers=0,
    gdn_layers=30,
    full_attention_interval=4,
    hidden=2048,
    heads=16,
    kv_heads=2,
    head_dim=256,
    attn_output_gate=True,
    partial_rotary_factor=0.25,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    gdn_chunk_size=64,
    mamba_ssm_dtype="fp32",
    moe_dispatch_strategy="hierarchical_ag",
    moe_fused_activation=True,
    n_experts=256,
    topk=8,
    expert_ffn=512,
    n_shared_experts=1,
    shared_ffn=512,
    vocab=248320,
    tie_embeddings=False,
)

QWEN3_5_35B_A3B_BF16 = Model(
    name="Qwen3.5-35B-A3B-BF16",
    layers=40,
    dense_layers=0,
    gdn_layers=30,
    full_attention_interval=4,
    hidden=2048,
    heads=16,
    kv_heads=2,
    head_dim=256,
    attn_output_gate=True,
    partial_rotary_factor=0.25,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    gdn_chunk_size=64,
    mamba_ssm_dtype="fp32",
    moe_dispatch_strategy="hierarchical_ag",
    moe_fused_activation=True,
    n_experts=256,
    topk=8,
    expert_ffn=512,
    n_shared_experts=1,
    shared_ffn=512,
    vocab=248320,
    tie_embeddings=False,
)

QWEN3_5_397B_A17B = Model(
    name="Qwen3.5-397B-A17B-FP8",
    layers=60,
    dense_layers=0,
    gdn_layers=45,
    full_attention_interval=4,
    hidden=4096,
    heads=32,
    kv_heads=2,
    head_dim=256,
    attn_output_gate=True,
    partial_rotary_factor=0.25,
    linear_num_key_heads=16,
    linear_num_value_heads=64,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    gdn_chunk_size=64,
    mamba_ssm_dtype="fp32",
    moe_dispatch_strategy="hierarchical_ag",
    moe_fused_activation=True,
    n_experts=512,
    topk=10,
    expert_ffn=1024,
    n_shared_experts=1,
    shared_ffn=1024,
    vocab=248320,
    tie_embeddings=False,
)

GLM_5_2_FP8 = Model(
    name="GLM-5.2-FP8",
    layers=78,
    dense_layers=3,
    indexer_layers=21,
    indexer_freq=4,
    indexer_offset=3,
    hidden=6144,
    heads=64,
    kv_lora_rank=512,
    q_lora_rank=2048,
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    v_head_dim=256,
    indexer_heads=32,
    indexer_head_dim=128,
    indexer_topk=2048,
    indexer_cache_bytes_per_token=132,
    sparse_attention_strategy="gathered",
    moe_dispatch_strategy="hierarchical_ag",
    dense_fused_activation=False,
    moe_fused_activation=True,
    dense_ffn=12288,
    n_experts=256,
    topk=8,
    expert_ffn=2048,
    n_shared_experts=1,
    shared_ffn=2048,
    vocab=154880,
    tie_embeddings=False,
)

NUMERICS_FP8 = Numerics(
    default="bf16",
    sites={
        "weights": "fp8",
        "moe_weights": "fp8",
        "kv_cache": "fp8",
        "kv_write": "bf16",
        "act_in": "fp8",
        "act_out": "bf16",
        "attn_qk": "bf16",
        "attn_av": "bf16",
        "tp_allreduce": "bf16",
        "expert_dispatch": "fp8",
        "expert_collect": "bf16",
        "router_logits": "fp32",
        "dcp_gather": "fp32",
    },
)

NUMERICS_BF16 = Numerics(
    default="bf16",
    sites={
        "weights": "bf16",
        "moe_weights": "bf16",
        "kv_cache": "bf16",
        "kv_write": "bf16",
        "act_in": "bf16",
        "act_out": "bf16",
        "attn_qk": "bf16",
        "attn_av": "bf16",
        "tp_allreduce": "bf16",
        "expert_dispatch": "bf16",
        "expert_collect": "bf16",
        "router_logits": "fp32",
        "dcp_gather": "fp32",
    },
)

MODELS: dict[str, Model] = {
    "Qwen3.5-35B-A3B-FP8": QWEN3_5_35B_A3B,
    "Qwen3.5-35B-A3B-BF16": QWEN3_5_35B_A3B_BF16,
    "Qwen3.5-397B-A17B-FP8": QWEN3_5_397B_A17B,
    "GLM-5.2-FP8": GLM_5_2_FP8,
}

NUMERICS_BY_MODEL: dict[str, Numerics] = {
    "Qwen3.5-35B-A3B-FP8": NUMERICS_FP8,
    "Qwen3.5-35B-A3B-BF16": NUMERICS_BF16,
    "Qwen3.5-397B-A17B-FP8": NUMERICS_FP8,
    "GLM-5.2-FP8": NUMERICS_FP8,
}


def get_model(name: str) -> Model:
    """Return a registered Model by name."""
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name]


def get_numerics(name: str) -> Numerics:
    """Return the default Numerics preset for a registered Model."""
    if name not in NUMERICS_BY_MODEL:
        raise KeyError(
            f"Unknown model {name!r}; available: {sorted(NUMERICS_BY_MODEL)}")
    return NUMERICS_BY_MODEL[name]
