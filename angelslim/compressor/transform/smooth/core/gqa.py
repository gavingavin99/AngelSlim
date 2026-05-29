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

"""Grouped-Query Attention helpers for smooth weight expansion / pooling.

* :func:`expand_kv_smooth_to_q` — KV-side smooth weight has fewer heads
  than Q-side; ``repeat_interleave`` along the head axis so each Q head
  inside a group shares the same smooth weight.
* :func:`pool_attn_out_smooth_to_v` — VO smooth: attn output has
  ``num_q_heads * head_dim`` channels, V has ``num_kv_heads * head_dim``;
  reduce by channel-wise max within each head group.
"""

import torch

__all__ = [
    "expand_kv_smooth_to_q",
    "pool_attn_out_smooth_to_v",
]


def expand_kv_smooth_to_q(
    smooth_kv: torch.Tensor,
    head_dim: int,
    num_q_heads: int,
) -> torch.Tensor:
    """Expand a ``[num_kv_heads * head_dim]`` smooth weight to
    ``[num_q_heads * head_dim]`` via per-head repeat_interleave.

    Each KV head is shared by ``n_groups = num_q_heads // num_kv_heads``
    consecutive Q heads, so we tile its smooth weight that many times.

    Args:
        smooth_kv: 1-D, length ``num_kv_heads * head_dim``.
        head_dim: per-head dimension.
        num_q_heads: target Q-head count.

    Returns:
        1-D tensor of length ``num_q_heads * head_dim``.
    """
    kv_dim = smooth_kv.numel()
    num_kv_heads = kv_dim // head_dim
    if kv_dim % head_dim != 0:
        raise ValueError(f"smooth_kv len {kv_dim} not divisible by head_dim {head_dim}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads {num_q_heads} not divisible by num_kv_heads {num_kv_heads}")
    n_groups = num_q_heads // num_kv_heads
    sw_kv = smooth_kv.reshape(num_kv_heads, head_dim)
    sw_q = sw_kv.repeat_interleave(n_groups, dim=0)
    return sw_q.reshape(num_q_heads * head_dim)


def pool_attn_out_smooth_to_v(
    smooth_attn_out: torch.Tensor,
    head_dim: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Reduce a ``[num_q_heads * head_dim]`` attn-output smooth weight
    to a ``[num_kv_heads * head_dim]`` V-output smooth weight by per-head
    channel-wise max within each KV-head group.

    Args:
        smooth_attn_out: 1-D, length ``num_q_heads * head_dim``.
        head_dim: per-head dimension.
        num_kv_heads: target KV-head count.

    Returns:
        1-D tensor of length ``num_kv_heads * head_dim``.
    """
    out_dim = smooth_attn_out.numel()
    num_q_heads = out_dim // head_dim
    if out_dim % head_dim != 0:
        raise ValueError(f"smooth len {out_dim} not divisible by head_dim {head_dim}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads {num_q_heads} not divisible by num_kv_heads {num_kv_heads}")
    n_groups = num_q_heads // num_kv_heads
    sw_q = smooth_attn_out.reshape(num_kv_heads, n_groups, head_dim)
    sw_v = sw_q.max(dim=1).values  # [num_kv_heads, head_dim]
    return sw_v.reshape(num_kv_heads * head_dim)
