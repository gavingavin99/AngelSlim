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

"""Backend-agnostic per-layer alpha grid search.

This is the *pure-tensor* core of :class:`SmoothAlphaSearcher`: it
takes already-collected ``x``, ``w``, ``act_absmax`` tensors (typically
on a single TP shard) and returns the best alpha + smooth_weight by
minimising fake-quant MSE.

No distributed primitives, no module discovery, no FusedMoE awareness —
those concerns live in ``smooth/vllm/searcher_dist.py``.
"""

from typing import Tuple

import torch

from .formulas import smooth_default, smooth_per_tensor_act_first
from .qdq import smooth_qdq_act, smooth_qdq_weight

__all__ = [
    "smooth_alpha_search_layer",
]


def smooth_alpha_search_layer(
    x: torch.Tensor,
    w: torch.Tensor,
    act_absmax: torch.Tensor,
    config,
) -> Tuple[float, torch.Tensor, float]:
    """Run the per-layer grid search.

    Args:
        x: ``[N, in_features]`` raw activation (float, on the search device).
        w: ``[out_features, in_features]`` weight tensor (float, on device).
        act_absmax: ``[in_features]`` per-channel activation absmax/EMA.
        config: ``SmoothAlphaSearchConfig`` instance.

    Returns:
        Tuple of ``(best_alpha_or_mul, best_smooth_weight, best_loss)``.
        For ``smooth_search_mode="per-tensor-act-first"`` the first
        element is the optimal multiplier; for ``"default"`` it is the
        optimal ``alpha``.
    """
    cfg = config
    device = x.device
    act_absmax = act_absmax.float().to(device).clamp(min=1e-8)
    weight_absmax = w.abs().max(dim=0).values.clamp(min=1e-8)

    # Reference output (computed once)
    y_ref = x @ w.T  # [N, out_feat]

    if cfg.smooth_search_mode == "default":
        alphas = torch.linspace(cfg.alpha_min, cfg.alpha_max, cfg.alpha_steps).tolist()
        losses = torch.empty(len(alphas), device=device)
        for i, alpha in enumerate(alphas):
            smooth = smooth_default(
                act_absmax,
                weight_absmax,
                alpha=alpha,
                smooth_min=cfg.smooth_min,
                smooth_max=cfg.smooth_max,
            )
            x_smoothed = x / smooth.unsqueeze(0)
            w_smoothed = w * smooth.unsqueeze(0)
            x_q = smooth_qdq_act(
                x_smoothed,
                method=cfg.act_quant_method,
                quant_type=cfg.act_quant_type,
                bits=cfg.weight_quant_bits,
            )
            w_q = smooth_qdq_weight(
                w_smoothed,
                method=cfg.weight_quant_method,
                quant_type=cfg.weight_quant_type,
                bits=cfg.weight_quant_bits,
                group_size=cfg.weight_group_size,
                block_size=cfg.block_size,
            )
            y_q = x_q @ w_q.T
            losses[i] = ((y_ref - y_q) ** 2).mean()

        best_idx = int(losses.argmin().item())
        best_alpha = alphas[best_idx]
        best_smooth = smooth_default(
            act_absmax,
            weight_absmax,
            alpha=best_alpha,
            smooth_min=cfg.smooth_min,
            smooth_max=cfg.smooth_max,
        )
        return best_alpha, best_smooth, losses[best_idx].item()

    elif cfg.smooth_search_mode == "per-tensor-act-first":
        muls = torch.linspace(cfg.act_mul_min, cfg.act_mul_max, cfg.alpha_steps).tolist()
        losses = torch.empty(len(muls), device=device)
        for i, mul in enumerate(muls):
            smooth = smooth_per_tensor_act_first(
                act_absmax,
                mul=mul,
                smooth_min=cfg.smooth_min,
                smooth_max=cfg.smooth_max,
            )
            x_smoothed = x / smooth.unsqueeze(0)
            w_smoothed = w * smooth.unsqueeze(0)
            x_q = smooth_qdq_act(
                x_smoothed,
                method=cfg.act_quant_method,
                quant_type=cfg.act_quant_type,
                bits=cfg.weight_quant_bits,
            )
            w_q = smooth_qdq_weight(
                w_smoothed,
                method=cfg.weight_quant_method,
                quant_type=cfg.weight_quant_type,
                bits=cfg.weight_quant_bits,
                group_size=cfg.weight_group_size,
                block_size=cfg.block_size,
            )
            y_q = x_q @ w_q.T
            losses[i] = ((y_ref - y_q) ** 2).mean()

        best_idx = int(losses.argmin().item())
        best_mul = muls[best_idx]
        best_smooth = smooth_per_tensor_act_first(
            act_absmax,
            mul=best_mul,
            smooth_min=cfg.smooth_min,
            smooth_max=cfg.smooth_max,
        )
        return best_mul, best_smooth, losses[best_idx].item()

    else:
        raise ValueError(f"Unknown smooth_search_mode: {cfg.smooth_search_mode}")
