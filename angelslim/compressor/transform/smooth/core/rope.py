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

"""RoPE-aware geometry utilities for QK smooth.

RoPE rotates within each head along the pair ``(d, d + head_dim/2)``, so
the smooth-weight assignment must be done at *head granularity* —
splitting along ``D / 2`` globally would couple unrelated heads and
break numerical equivalence.

This module provides backend-agnostic helpers used by the offline
weight converter (``convert/apply_qk.py``) and any future online path
that wants to perform the same fold.
"""

import torch

__all__ = [
    "pair_rope_kv_absmax",
    "rope_kv_smooth_weight",
    "fold_smooth_into_qk_norm",
]


def pair_rope_kv_absmax(
    k_absmax: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Apply RoPE-aware symmetrisation to a per-channel K absmax vector.

    For each head ``h`` and dimension ``d`` in ``[0, head_dim/2)``:
        ``paired[h, d] = paired[h, d + head_dim/2]
            = max(k_absmax[h, d], k_absmax[h, d + head_dim/2])``

    Args:
        k_absmax: 1-D tensor of shape ``[num_kv_heads * head_dim]``.
        head_dim: per-head dimension (must divide ``k_absmax.numel()``).

    Returns:
        ``[num_kv_heads, head_dim]`` symmetrised absmax (each row's
        first half equals its second half).
    """
    if k_absmax.dim() != 1:
        raise ValueError(f"Expected 1-D k_absmax, got shape {tuple(k_absmax.shape)}")
    D = k_absmax.numel()
    if D % head_dim != 0:
        raise ValueError(f"k_absmax len {D} not divisible by head_dim {head_dim}")
    num_kv_heads = D // head_dim
    half_hd = head_dim // 2

    k_per_head = k_absmax.reshape(num_kv_heads, head_dim)
    paired = torch.empty(num_kv_heads, head_dim, dtype=k_per_head.dtype, device=k_per_head.device)
    paired_first_half = torch.maximum(k_per_head[:, :half_hd], k_per_head[:, half_hd:])
    paired[:, :half_hd] = paired_first_half
    paired[:, half_hd:] = paired_first_half
    return paired


def rope_kv_smooth_weight(
    k_absmax: torch.Tensor,
    head_dim: int,
    alpha: float,
) -> torch.Tensor:
    """Compute the per-head RoPE-aware smooth weight from K absmax.

    ``smooth[h, d] = paired[h, d]^alpha``, where ``paired`` comes from
    :func:`pair_rope_kv_absmax`.

    Returns:
        ``[num_kv_heads, head_dim]`` float32 smooth weight.
    """
    paired = pair_rope_kv_absmax(k_absmax.float(), head_dim)
    return paired.pow(alpha)


def fold_smooth_into_qk_norm(
    smooth_per_head: torch.Tensor,
) -> torch.Tensor:
    """Reduce a per-head ``[num_kv_heads, head_dim]`` smooth weight to a
    head-shared ``[head_dim]`` weight.

    For models where ``Q = q_norm(q_proj(x))`` (RMSNorm after proj), the
    proj-level scale is annihilated by the norm, so the smooth weight
    must be folded into the norm weight instead.  Since the norm weight
    has shape ``[head_dim]`` and is shared across heads, we collapse the
    per-head smooth weight by taking the channel-wise max across heads.

    Args:
        smooth_per_head: ``[num_kv_heads, head_dim]``.

    Returns:
        ``[head_dim]`` float32 norm-level smooth weight.
    """
    if smooth_per_head.dim() != 2:
        raise ValueError(f"Expected 2-D smooth_per_head, got shape {tuple(smooth_per_head.shape)}")
    return smooth_per_head.float().max(dim=0).values
