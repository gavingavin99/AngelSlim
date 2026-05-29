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

"""QK smooth — apply RoPE-aware smooth weight to ``q_proj`` / ``k_proj``
(or fold into ``q_norm`` / ``k_norm`` for RMSNorm-based models).
"""

import torch

from ..core.tensor_ops import inplace_div_fp32, inplace_mul_fp32
from .key_maps import DEFAULT_KEY_MAP
from .module_finder import attn_key_to_hf_prefix, get_submodule_safe, maybe_materialize

__all__ = [
    "apply_qk_smooth",
]


def apply_qk_smooth(
    model: torch.nn.Module,
    smooth_stats: dict,
    alpha: float = 0.6,
    head_dim: int = None,
    km: dict = None,
) -> None:
    """Apply QK smooth to every layer that has both ``attn.k`` and
    ``attn.q`` stats.

    Smooth weight formula (RoPE-aware pairing)::

        k_absmax reshape -> [num_kv_heads, head_dim]
        paired[h, d] = max(k_absmax[h, d], k_absmax[h, d + head_dim/2])
        smooth_weight[h, d] = smooth_weight[h, d + head_dim/2] = paired[h, d] ^ alpha

    GQA handling: expand ``smooth_weight`` from kv_heads to q_heads via
    ``repeat_interleave``.

    qk_norm path: fuse ``smooth`` into ``q_norm`` / ``k_norm`` weights
    instead of proj weights.
    """
    km = km or DEFAULT_KEY_MAP
    stat_k = km["stat_k"]
    stat_q = km["stat_q"]

    k_keys = {k for k in smooth_stats if k.endswith(stat_k)}
    q_keys = {k for k in smooth_stats if k.endswith(stat_q)}
    k_bases = {k[: -len(stat_k)] for k in k_keys}
    q_bases = {k[: -len(stat_q)] for k in q_keys}
    valid_bases = k_bases & q_bases

    print(
        f"\n[QK Smooth] {len(valid_bases)} layers eligible " f"(have both attn.k and attn.q stats)"
    )

    for base in sorted(valid_bases):
        k_stat_key = base + stat_k
        hf_prefix = attn_key_to_hf_prefix(base, km)

        attn_module = get_submodule_safe(model, hf_prefix)
        if attn_module is None:
            print(f"  [SKIP] {hf_prefix}: module not found in model")
            continue

        use_qk_norm = getattr(attn_module, km["qk_norm_flag"], False) or getattr(
            model.config, "qk_norm", False
        )
        q_norm = getattr(attn_module, km["q_norm"], None) if use_qk_norm else None
        k_norm = getattr(attn_module, km["k_norm"], None) if use_qk_norm else None
        if use_qk_norm and (q_norm is None or k_norm is None):
            print(f"  [SKIP] {hf_prefix}: use_qk_norm=True but q_norm/k_norm not found")
            continue

        q_proj = getattr(attn_module, km["q_proj"], None)
        k_proj = getattr(attn_module, km["k_proj"], None)
        if q_proj is None or k_proj is None:
            print(f"  [SKIP] {hf_prefix}: q_proj or k_proj not found")
            continue

        k_scale = smooth_stats[k_stat_key]["scale"]
        if k_scale is None:
            print(f"  [SKIP] {hf_prefix}: k scale is None")
            continue

        k_absmax = k_scale.to(torch.float32)
        D = k_absmax.shape[0]

        if head_dim is None:
            raise ValueError(f"head_dim must be provided for QK smooth (layer {hf_prefix})")

        num_kv_heads_loc = D // head_dim
        half_hd = head_dim // 2

        k_per_head = k_absmax.reshape(num_kv_heads_loc, head_dim)
        paired_hd = torch.maximum(k_per_head[:, :half_hd], k_per_head[:, half_hd:]).clamp(min=1e-8)
        smooth_per_head = torch.empty(num_kv_heads_loc, head_dim, dtype=torch.float32)
        smooth_per_head[:, :half_hd] = paired_hd
        smooth_per_head[:, half_hd:] = paired_hd
        smooth_weight = smooth_per_head.pow(alpha)

        q_out_dim = q_proj.weight.shape[0]
        k_out_dim = k_proj.weight.shape[0]
        is_gqa = q_out_dim != k_out_dim
        device = q_proj.weight.device if not q_proj.weight.is_meta else torch.device("cpu")

        if use_qk_norm:
            per_dim_max = smooth_weight.max(dim=0).values
            norm_sw = torch.maximum(per_dim_max[:half_hd], per_dim_max[half_hd:])
            norm_smooth_weight = torch.empty(head_dim, dtype=torch.float32)
            norm_smooth_weight[:half_hd] = norm_sw
            norm_smooth_weight[half_hd:] = norm_sw

            norm_smooth_weight = norm_smooth_weight.to(device=device)

            with maybe_materialize(q_norm, device):
                inplace_mul_fp32(q_norm.weight, norm_smooth_weight)
            with maybe_materialize(k_norm, device):
                inplace_div_fp32(k_norm.weight, norm_smooth_weight)

            print(
                f"  [QK-norm] {hf_prefix}: q_out={q_out_dim}, k_out={k_out_dim}, "
                f"norm_smooth range=["
                f"{norm_smooth_weight.float().min():.4f}, "
                f"{norm_smooth_weight.float().max():.4f}], "
                f"gqa={'yes (n_groups=' + str(q_out_dim // k_out_dim) + ')' if is_gqa else 'no'}"
            )

        else:
            smooth_weight_flat = smooth_weight.reshape(D)

            if is_gqa:
                num_kv_heads = k_out_dim // head_dim
                n_groups = q_out_dim // k_out_dim
                sw_kv = smooth_weight_flat.reshape(num_kv_heads, head_dim)
                sw_q = sw_kv.repeat_interleave(n_groups, dim=0)
                q_smooth_weight = sw_q.reshape(q_out_dim)
                smooth_k_weight = smooth_weight_flat
            else:
                q_smooth_weight = smooth_weight_flat
                smooth_k_weight = smooth_weight_flat

            q_smooth_weight = q_smooth_weight.to(device=device)
            smooth_k_weight = smooth_k_weight.to(device=device)

            with maybe_materialize(q_proj, device):
                inplace_mul_fp32(q_proj.weight, q_smooth_weight.view(-1, 1))
                if q_proj.bias is not None:
                    inplace_mul_fp32(q_proj.bias, q_smooth_weight)

            with maybe_materialize(k_proj, device):
                inplace_div_fp32(k_proj.weight, smooth_k_weight.view(-1, 1))
                if k_proj.bias is not None:
                    inplace_div_fp32(k_proj.bias, smooth_k_weight)

            print(
                f"  [QK] {hf_prefix}: q_out={q_out_dim}, k_out={k_out_dim}, "
                f"gqa={'yes (n_groups=' + str(q_out_dim // k_out_dim) + ')' if is_gqa else 'no'}"
            )
