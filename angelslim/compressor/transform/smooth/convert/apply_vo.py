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

"""VO smooth — fold attention-output absmax into ``v_proj`` / ``o_proj``."""

import torch

from ..core.tensor_ops import inplace_div_fp32, inplace_mul_fp32
from .key_maps import DEFAULT_KEY_MAP
from .module_finder import attn_key_to_hf_prefix, get_submodule_safe, maybe_materialize

__all__ = [
    "apply_vo_smooth",
]


def apply_vo_smooth(
    model: torch.nn.Module,
    smooth_stats: dict,
    alpha: float = 0.5,
    head_dim: int = None,
    km: dict = None,
) -> None:
    """Apply VO smooth to every layer that has ``attn_out`` stats.

    Smooth weight formula::

        smooth_weight[i] = attn_out_absmax[i]^alpha
                           / max|o_proj.weight[:, i]|^(1 - alpha)

    GQA: max-pool ``smooth_weight`` across head groups for ``v_proj``.
    """
    km = km or DEFAULT_KEY_MAP
    stat_attn_out = km["stat_attn_out"]

    attn_out_keys = [k for k in smooth_stats if k.endswith(stat_attn_out)]
    print(f"\n[VO Smooth] {len(attn_out_keys)} layers with attn_out stats")

    for attn_out_key in sorted(attn_out_keys):
        base = attn_out_key[: -len(stat_attn_out)]
        hf_prefix = attn_key_to_hf_prefix(base, km)

        attn_module = get_submodule_safe(model, hf_prefix)
        if attn_module is None:
            print(f"  [SKIP] {hf_prefix}: module not found in model")
            continue

        v_proj = getattr(attn_module, km["v_proj"], None)
        o_proj = getattr(attn_module, km["o_proj"], None)
        if v_proj is None or o_proj is None:
            print(f"  [SKIP] {hf_prefix}: v_proj or o_proj not found")
            continue

        attn_out_scale = smooth_stats[attn_out_key]["scale"]
        if attn_out_scale is None:
            print(f"  [SKIP] {hf_prefix}: attn_out scale is None")
            continue

        attn_out_absmax = attn_out_scale.to(torch.float32).clamp(min=1e-8)
        attn_out_dim = attn_out_absmax.shape[0]

        # weight_absmax for o_proj — handle accelerate meta-device hooks.
        o_hook = getattr(o_proj, "_hf_hook", None)
        if o_hook is not None and o_proj.weight.device.type == "meta":
            _tmp_device = torch.device("cpu")
            _orig_exec = o_hook.execution_device
            o_hook.execution_device = _tmp_device
            o_hook.pre_forward(o_proj)
            o_proj_max = (o_proj.weight.detach().float().abs().max(dim=0).values).clamp(min=1e-8)
            o_hook.post_forward(o_proj, None)
            o_hook.execution_device = _orig_exec
        else:
            o_proj_max = (o_proj.weight.detach().float().abs().max(dim=0).values).clamp(min=1e-8)

        smooth_weight = attn_out_absmax.pow(alpha) / o_proj_max.pow(1.0 - alpha)

        v_out_dim = v_proj.weight.shape[0]
        is_gqa = v_out_dim != attn_out_dim

        if is_gqa:
            assert (
                attn_out_dim % v_out_dim == 0
            ), f"{hf_prefix}: attn_out_dim={attn_out_dim} not divisible by v_out_dim={v_out_dim}"
            n_groups = attn_out_dim // v_out_dim

            if head_dim is None:
                raise ValueError(
                    f"head_dim must be provided for GQA VO smooth (layer {hf_prefix})"
                )
            num_kv_heads = v_out_dim // head_dim

            sw_grouped = smooth_weight.reshape(num_kv_heads, n_groups, head_dim)
            pooled = sw_grouped.max(dim=1).values
            smooth_v_weight = pooled.reshape(v_out_dim)

            smooth_weight = (
                pooled.unsqueeze(1).expand(num_kv_heads, n_groups, head_dim).reshape(attn_out_dim)
            )
        else:
            smooth_v_weight = smooth_weight

        device = v_proj.weight.device if not v_proj.weight.is_meta else torch.device("cpu")
        smooth_v_weight = smooth_v_weight.to(device=device)
        smooth_weight = smooth_weight.to(device=device)

        with maybe_materialize(v_proj, device):
            inplace_div_fp32(v_proj.weight, smooth_v_weight.view(-1, 1))
            if v_proj.bias is not None:
                inplace_div_fp32(v_proj.bias, smooth_v_weight)

        with maybe_materialize(o_proj, device):
            inplace_mul_fp32(o_proj.weight, smooth_weight.view(1, -1))

        print(
            f"  [VO] {hf_prefix}: v_out={v_out_dim}, attn_out={attn_out_dim}, "
            f"gqa={'yes (n_groups=' + str(attn_out_dim // v_out_dim) + ')' if is_gqa else 'no'}"
        )
