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

"""Forward-hook classes for online smooth-stat collection.

* :class:`SmoothAttnHook` — captures ``q`` / ``k`` inputs and the
  attention output (= ``o_proj`` input) on a vLLM ``Attention`` module.
* :class:`SmoothDownProjInputHook` — captures the input of a dense MLP
  ``down_proj`` (which is ``silu(gate) * up``).
* :class:`SmoothAlphaValueHook` — captures the *raw* down_proj input
  tensors needed by the alpha grid search.

Migrated verbatim from
``angelslim/compressor/quant/core/vllm_calibrate_utils/{hooks,search}.py``;
the only change is the import of ``per_channel_absmax`` from the
shared ``smooth/core`` layer instead of being defined locally.
"""

import torch

from ..core.tensor_ops import per_channel_absmax

__all__ = [
    "SmoothAttnHook",
    "SmoothDownProjInputHook",
    "SmoothAlphaValueHook",
]


class SmoothAttnHook:
    """Hook for collecting per-channel (last-dim) absmax and EMA on
    Attention ``q`` / ``k`` inputs and the attention output.

    ``v`` is intentionally skipped.

    EMA update rule::
        ema = ema_momentum * ema + (1 - ema_momentum) * current_absmax

    Args:
        layer_name: Attention layer name.
        smooth_stats: Shared dict that stores per-key running absmax/ema.
        ema_momentum: EMA decay factor.
        token_clip: Optional percentile-based clipping for the per-channel
            absmax. See :func:`per_channel_absmax` for accepted values.
            ``<=0`` (default) preserves the original absolute-max behaviour.
    """

    def __init__(self, layer_name, smooth_stats, ema_momentum=0.9, token_clip=-1):
        self.layer_name = layer_name
        self.smooth_stats = smooth_stats
        self.ema_momentum = ema_momentum
        self.token_clip = token_clip
        self.call_count = 0

    def _update(self, key, tensor):
        """Update absmax and EMA for a named tensor slot."""
        with torch.no_grad():
            cur_absmax = per_channel_absmax(tensor, token_clip=self.token_clip)

            stats = self.smooth_stats[key]
            if stats["absmax"] is None:
                stats["absmax"] = cur_absmax
            else:
                stats["absmax"] = torch.maximum(stats["absmax"], cur_absmax)
            if stats["ema"] is None:
                stats["ema"] = cur_absmax.clone()
            else:
                stats["ema"] = (
                    self.ema_momentum * stats["ema"] + (1.0 - self.ema_momentum) * cur_absmax
                )
            stats["call_count"] = self.call_count

    def __call__(self, module, input, output):
        self.call_count += 1
        # --- inputs: q, k (v skipped) ---
        q = input[0] if len(input) > 0 else None
        k = input[1] if len(input) > 1 else None
        for tag, tensor in [("q", q), ("k", k)]:
            if tensor is not None and isinstance(tensor, torch.Tensor):
                self._update(f"{self.layer_name}.{tag}", tensor)
        # --- output: attention result = o_proj input ---
        attn_out = output[0] if isinstance(output, tuple) else output
        if attn_out is not None and isinstance(attn_out, torch.Tensor):
            self._update(f"{self.layer_name}.attn_out", attn_out)


class SmoothDownProjInputHook:
    """Hook for collecting per-channel (last-dim) absmax and EMA on the
    INPUT of a dense MLP ``down_proj`` layer.

    This captures ``silu(gate) * up`` — the true activation that
    ``down_proj`` sees and the correct signal for SmoothQuant calibration.

    Input shape: ``[num_tokens, intermediate_size]``.
    """

    def __init__(self, layer_name, smooth_stats, ema_momentum=0.9, token_clip=-1):
        self.layer_name = layer_name
        self.smooth_stats = smooth_stats
        self.ema_momentum = ema_momentum
        self.token_clip = token_clip
        self.call_count = 0

    def __call__(self, module, input, output):
        self.call_count += 1
        act = input[0] if isinstance(input, tuple) else input
        if not isinstance(act, torch.Tensor):
            return

        with torch.no_grad():
            cur_absmax = per_channel_absmax(act, token_clip=self.token_clip)

            stats = self.smooth_stats[self.layer_name]
            if stats["absmax"] is None:
                stats["absmax"] = cur_absmax
            else:
                stats["absmax"] = torch.maximum(stats["absmax"], cur_absmax)
            if stats["ema"] is None:
                stats["ema"] = cur_absmax.clone()
            else:
                stats["ema"] = (
                    self.ema_momentum * stats["ema"] + (1.0 - self.ema_momentum) * cur_absmax
                )
            stats["call_count"] = self.call_count


class SmoothAlphaValueHook:
    """Capture raw down_proj input activation for alpha grid search.

    Stores up to ``max_tokens`` token-rows per layer via CPU offload
    with uniform sub-sampling when the cap is exceeded.

    Works alongside :class:`SmoothDownProjInputHook` (which collects
    absmax / ema statistics); this hook stores the actual tensor values.

    Args:
        layer_name: down_proj module name (same key used in
            ``_smooth_stats``).
        alpha_search_values: shared dict
            ``{layer_name: {"tokens": Tensor | None}}``.
        max_tokens: maximum token-rows to keep per layer.
    """

    def __init__(self, layer_name, alpha_search_values, max_tokens=4096):
        self.layer_name = layer_name
        self.alpha_search_values = alpha_search_values
        self.max_tokens = max_tokens

    def __call__(self, module, input, output):
        act = input[0] if isinstance(input, tuple) else input
        if not isinstance(act, torch.Tensor):
            return

        with torch.no_grad():
            act_cpu = act.detach().float().cpu()
            stored = self.alpha_search_values[self.layer_name]
            if stored["tokens"] is None:
                stored["tokens"] = act_cpu
            else:
                stored["tokens"] = torch.cat([stored["tokens"], act_cpu], dim=0)
            # Uniform sub-sample when cap exceeded
            if stored["tokens"].shape[0] > self.max_tokens:
                indices = torch.linspace(0, stored["tokens"].shape[0] - 1, self.max_tokens).long()
                stored["tokens"] = stored["tokens"][indices]
