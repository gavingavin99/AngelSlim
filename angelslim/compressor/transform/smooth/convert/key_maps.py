# Copyright 2025 Tencent Inc. All Rights Reserved.
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

"""Per-architecture key maps used by the offline weight converter.

Each ``KEY_MAP`` specifies attribute names for projections (``q_proj``,
``k_proj``, ...), the smooth-stats key suffixes captured by the vLLM
hooks (``.q``, ``.k``, ``.attn_out``, ...), regex patterns for
discovering ``down_proj`` stats and MLP containers, and — for fused MoE
architectures — the names of the 3-D ``down_proj`` / ``gate_up_proj``
parameters.
"""

__all__ = [
    "HY_V3_KEY_MAP",
    "LLAMA_KEY_MAP",
    "MIXTRAL_KEY_MAP",
    "QWEN3_MOE_KEY_MAP",
    "PREDEFINED_KEY_MAPS",
    "DEFAULT_KEY_MAP",
]


HY_V3_KEY_MAP = {
    # projection attribute names
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
    "down_proj": "down_proj",
    "up_proj": "up_proj",
    # qk_norm
    "qk_norm_flag": "use_qk_norm",
    "q_norm": "q_norm",
    "k_norm": "k_norm",
    # stats key suffixes
    "stat_k": ".k",
    "stat_q": ".q",
    "stat_attn_out": ".attn_out",
    # vLLM -> HF path translation
    "attn_strip": ".attn",
    # down_proj stat regex patterns
    "down_patterns": [
        r"\.shared_experts\.down_proj",
        r"\.shared_mlp\.down_proj",
        r"\.mlp\.down_proj",
        r"\.experts\.\d+\.down_proj",
    ],
    # MLP container regex (used for verification forward)
    "mlp_containers": [
        r"\.mlp$",
        r"mlp\.shared_experts$",
        r"mlp\.experts\.\d+$",
    ],
    # vLLM stat_key -> HF module name aliases (regex sub)
    "stat_path_aliases": [
        (r"\.shared_mlp\b", ".shared_experts"),
    ],
    # Fused-experts MoE (HYV3Experts-style 3D nn.Parameter)
    #     gate, up = F.linear(x, gate_up_proj[i]).chunk(2, dim=-1)
    "fused_down_attr": "down_proj",
    "fused_gate_up_attr": "gate_up_proj",
}


LLAMA_KEY_MAP = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
    "down_proj": "down_proj",
    "up_proj": "up_proj",
    "qk_norm_flag": "use_qk_norm",
    "q_norm": "q_norm",
    "k_norm": "k_norm",
    "stat_k": ".k",
    "stat_q": ".q",
    "stat_attn_out": ".attn_out",
    "attn_strip": ".attn",
    "down_patterns": [r"\.mlp\.down_proj"],
    "mlp_containers": [r"mlp$"],
}


MIXTRAL_KEY_MAP = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
    "down_proj": "w2",
    "up_proj": "w3",
    "qk_norm_flag": "use_qk_norm",
    "q_norm": "q_norm",
    "k_norm": "k_norm",
    "stat_k": ".k",
    "stat_q": ".q",
    "stat_attn_out": ".attn_out",
    "attn_strip": ".attn",
    "down_patterns": [r"\.block_sparse_moe\.experts\.\d+\.w2"],
    "mlp_containers": [r"block_sparse_moe\.experts\.\d+$"],
}


QWEN3_MOE_KEY_MAP = {
    # projection attribute names
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
    "down_proj": "down_proj",
    "up_proj": "up_proj",
    # qk_norm: Qwen3MoeAttention.forward applies q_norm/k_norm in the
    # head_dim dim, so we must always fold smooth into them rather than
    # into q_proj/k_proj (RMSNorm would otherwise absorb the proj-side
    # scale and break equivalence).  Pointing qk_norm_flag at the always-
    # truthy ``q_norm`` attribute forces the qk_norm code path.
    "qk_norm_flag": "q_norm",
    "q_norm": "q_norm",
    "k_norm": "k_norm",
    # stats key suffixes
    "stat_k": ".k",
    "stat_q": ".q",
    "stat_attn_out": ".attn_out",
    # vLLM -> HF path translation
    "attn_strip": ".attn",
    # down_proj stat regex patterns
    "down_patterns": [
        r"\.mlp\.down_proj",
        r"\.experts\.\d+\.down_proj",
    ],
    "mlp_containers": [
        r"mlp$",
        r"mlp\.experts\.\d+$",
    ],
    # Fused-experts MoE (Qwen3MoeExperts)
    "fused_down_attr": "down_proj",
    "fused_gate_up_attr": "gate_up_proj",
}


PREDEFINED_KEY_MAPS = {
    "hy_v3": HY_V3_KEY_MAP,
    "llama": LLAMA_KEY_MAP,
    "mixtral": MIXTRAL_KEY_MAP,
    "qwen3_moe": QWEN3_MOE_KEY_MAP,
}

DEFAULT_KEY_MAP = HY_V3_KEY_MAP
