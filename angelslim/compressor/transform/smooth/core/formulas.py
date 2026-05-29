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

"""Pure-tensor smooth-weight formulas.

These functions are deliberately backend-agnostic: they accept torch
tensors and return torch tensors with no awareness of TP / Linear /
FusedMoE.  Both the online searcher and the offline weight converter
import them.
"""

import torch

__all__ = [
    "smooth_default",
    "smooth_per_tensor_act_first",
]


def smooth_default(
    act_absmax: torch.Tensor,
    weight_absmax: torch.Tensor,
    alpha: float,
    smooth_min: float = 1e-6,
    smooth_max: float = 1e6,
) -> torch.Tensor:
    """Classic SmoothQuant per-channel formula.

    ``smooth_j = max|x_j|^alpha / max|w_j|^(1 - alpha)``

    Args:
        act_absmax: ``[C]`` per-channel activation absmax (or EMA).
        weight_absmax: ``[C]`` per-channel weight absmax along the
            input dimension.
        alpha: smoothing strength in ``[0, 1]``.  Larger values bias the
            scale toward activation distribution.
        smooth_min / smooth_max: clamp bounds against degenerate values.

    Returns:
        ``[C]`` smooth-weight tensor on the same device / dtype as the
        inputs (after `.pow` they are floating-point).
    """
    act = act_absmax.clamp(min=1e-8)
    weight = weight_absmax.clamp(min=1e-8)
    smooth = act.pow(alpha) / weight.pow(1.0 - alpha)
    return smooth.clamp(smooth_min, smooth_max)


def smooth_per_tensor_act_first(
    act_absmax: torch.Tensor,
    mul: float,
    smooth_min: float = 1e-6,
    smooth_max: float = 1e6,
) -> torch.Tensor:
    """Per-tensor-act-first smooth formula.

    Pick a global target absmax ``mul * max(act_absmax)`` and rescale
    each channel so its activation magnitude approaches that target.

    ``tgt = mul * act_absmax.max()``
    ``smooth = act_absmax / tgt``

    Args:
        act_absmax: ``[C]`` per-channel activation absmax.
        mul: multiplier searched in the grid (``act_mul_min..act_mul_max``).
        smooth_min / smooth_max: clamp bounds.

    Returns:
        ``[C]`` smooth-weight tensor.
    """
    act = act_absmax.clamp(min=1e-8)
    per_tensor_absmax = act.max()
    tgt = mul * per_tensor_absmax
    smooth = act / tgt
    return smooth.clamp(smooth_min, smooth_max)
