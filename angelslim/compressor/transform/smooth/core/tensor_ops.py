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

"""Backend-agnostic tensor utilities used by both the vLLM online
calibration path and the offline weight conversion path.

* :func:`per_channel_absmax` — fast topk-based per-channel absmax with
  optional percentile clipping (was ``_per_channel_absmax``).
* :func:`inplace_mul_fp32` / :func:`inplace_div_fp32` — high-precision
  in-place weight scaling (cast to float32, multiply / divide, cast back)
  shared by the offline weight converter.
"""

import os

import torch

__all__ = [
    "per_channel_absmax",
    "inplace_mul_fp32",
    "inplace_div_fp32",
    "set_percentile_subsample",
    "get_percentile_subsample",
]


# ---------------------------------------------------------------------------
# Stride-sampling configuration for percentile estimation.
# ---------------------------------------------------------------------------
_SAMPLE_STRIDE = 683  # prime, coprime with all common hidden_dims
_SAMPLE_MIN_KEEP = 1024  # numel <= STRIDE * MIN_KEEP → skip sampling

# Initial value taken from env var; can be flipped at runtime via
# set_percentile_subsample().
_SAMPLE_ENABLE = os.getenv("VLLM_CALIB_PERCENTILE_SUBSAMPLE", "0") == "1"


def set_percentile_subsample(enable: bool) -> None:
    """Enable / disable stride sub-sampling for percentile estimation.

    Affects :func:`per_channel_absmax`. Takes effect on the next call
    (safe to flip between calibration batches).
    """
    global _SAMPLE_ENABLE
    _SAMPLE_ENABLE = bool(enable)


def get_percentile_subsample() -> bool:
    """Return the current sub-sampling flag."""
    return _SAMPLE_ENABLE


def per_channel_absmax(tensor: torch.Tensor, token_clip: float = -1.0) -> torch.Tensor:
    """Per-channel absmax over ``dim=0``, with optional percentile clipping.

    Args:
        tensor: shape ``[num_tokens, dim]`` (1-D input is first unsqueezed).
        token_clip: clip mode for the per-channel statistic.
            * ``<= 0`` (default): absolute max along ``dim=0``.
            * In ``(0, 1]``: quantile fraction (e.g. 0.999).
            * In ``(1, 100)``: percentile value (e.g. 99.9).

    Implementation notes:
        * Uses ``torch.topk(..., dim=0)`` instead of ``torch.kthvalue`` —
          much faster on GPU when ``k`` is small (always the case for
          tail percentiles like 0.999).
        * Sub-samples rows with a coprime stride (683) when
          ``num_tokens`` is large.  683 is prime and coprime with all
          common hidden dims, so the sampling is uniform across channels.
        * Returns a CPU tensor for compatibility with running stats that
          live on the host side.

    Returns:
        CPU tensor of shape ``[dim]``.
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    abs_t = tensor.abs()

    use_percentile = token_clip is not None and token_clip > 0
    if not use_percentile:
        return abs_t.max(dim=0).values.detach().cpu()

    num_tokens = abs_t.shape[0]
    if num_tokens == 0:
        return abs_t.max(dim=0).values.detach().cpu()

    # Normalise token_clip to a quantile fraction in (0, 1].
    tc = float(token_clip)
    q = tc if tc <= 1.0 else tc / 100.0
    q = max(0.5, min(q, 1.0 - 1e-6))

    if not abs_t.is_floating_point():
        abs_t = abs_t.float()

    if _SAMPLE_ENABLE and num_tokens > _SAMPLE_STRIDE * _SAMPLE_MIN_KEEP:
        abs_t = abs_t[::_SAMPLE_STRIDE]
        num_tokens = abs_t.shape[0]

    try:
        k_high_raw = int(round((1.0 - q) * num_tokens))
        if k_high_raw <= 1:
            return abs_t.max(dim=0).values.detach().cpu()

        k_high = min(num_tokens, k_high_raw)
        top = torch.topk(abs_t, k_high, dim=0, largest=True, sorted=False).values
        return top.min(dim=0).values.detach().cpu()
    except Exception as e:  # pragma: no cover - defensive fallback
        print(
            f"[per_channel_absmax] token_clip topk failed (shape="
            f"{tuple(abs_t.shape)}, q={q}): {e}. Fallback to absolute max."
        )
        return abs_t.max(dim=0).values.detach().cpu()


def inplace_mul_fp32(weight: torch.Tensor, scale: torch.Tensor) -> None:
    """Multiply ``weight`` in-place by ``scale`` at fp32 precision.

    The operation is computed in float32 to avoid bfloat16 / float16
    accumulation errors and the result is written back into the original
    storage with the original dtype preserved.
    """
    orig_dtype = weight.dtype
    weight.data.copy_(weight.data.float().mul_(scale.float()).to(orig_dtype))


def inplace_div_fp32(weight: torch.Tensor, scale: torch.Tensor) -> None:
    """Divide ``weight`` in-place by ``scale`` at fp32 precision."""
    orig_dtype = weight.dtype
    weight.data.copy_(weight.data.float().div_(scale.float()).to(orig_dtype))
