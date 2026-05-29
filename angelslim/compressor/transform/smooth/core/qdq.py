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

"""Quantize-Dequantize helpers used by the alpha grid search.

These are intentionally backend-agnostic — both the vLLM online searcher
and any offline simulator can call them with plain torch tensors.

Migrated from
``angelslim/compressor/quant/core/vllm_calibrate_utils/search.py``
(``_smooth_qdq_act`` / ``_smooth_qdq_weight``) — leading underscore
dropped, behaviour unchanged.
"""

import torch

__all__ = [
    "smooth_qdq_act",
    "smooth_qdq_weight",
]


def smooth_qdq_act(
    x: torch.Tensor,
    method: str = "per_token",
    quant_type: str = "int8",
    bits: int = 8,
) -> torch.Tensor:
    """Quantize-dequantize activation (already divided by smooth_weight).

    Args:
        x: ``[N, in_features]``, float32.
        method: ``"per_tensor"`` | ``"per_token"``.
        quant_type: ``"int8"`` | ``"int4"`` | ``"int"`` | ``"fp8"``.
        bits: bit width for INT.

    Returns:
        ``x_qdq``: ``[N, in_features]``, float32.
    """
    if quant_type in ("int8", "int4", "int"):
        bnt = (1 << (bits - 1)) - 1  # 127 for int8
        if method == "per_tensor":
            scale = x.abs().max().clamp(min=1e-8) / bnt
        else:  # per_token
            scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / bnt
        return (x / scale).round().clamp(-bnt - 1, bnt) * scale

    elif quant_type == "fp8":
        fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
        if method == "per_tensor":
            scale = x.abs().max().clamp(min=1e-8) / fp8_max
        else:  # per_token
            scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / fp8_max
        q = (x / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return q.float() * scale

    else:
        raise ValueError(f"Unknown act quant_type: {quant_type}")


def smooth_qdq_weight(
    w: torch.Tensor,
    method: str = "per_channel",
    quant_type: str = "int8",
    bits: int = 8,
    group_size: int = 128,
    block_size: int = 128,
) -> torch.Tensor:
    """Quantize-dequantize weight (already multiplied by smooth_weight).

    Args:
        w: ``[out_features, in_features]``, float32.
        method: ``"per_tensor"`` | ``"per_channel"`` | ``"per_group"`` | ``"per_block"``.
        quant_type: ``"int8"`` | ``"int4"`` | ``"int"`` | ``"fp8"``.
        bits: bit width for INT.
        group_size: for ``per_group`` (along ``dim=1``).
        block_size: for ``per_block`` fp8.

    Returns:
        ``w_qdq``: ``[out_features, in_features]``, float32.
    """
    if quant_type in ("int8", "int4", "int"):
        bnt = (1 << (bits - 1)) - 1

        if method == "per_tensor":
            scale = w.abs().max().clamp(min=1e-8) / bnt
            return (w / scale).round().clamp(-bnt - 1, bnt) * scale

        elif method == "per_channel":
            # per output-channel (per row)
            scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / bnt
            return (w / scale).round().clamp(-bnt - 1, bnt) * scale

        elif method == "per_group":
            # group along in_features (dim=1)
            out_feat, in_feat = w.shape
            gs = group_size if group_size > 0 else in_feat
            if in_feat % gs != 0:
                # pad to group_size multiple
                pad_len = gs - in_feat % gs
                w_padded = torch.nn.functional.pad(w, (0, pad_len))
            else:
                w_padded = w
                pad_len = 0
            w_grouped = w_padded.reshape(out_feat, -1, gs)
            scale = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / bnt
            w_qdq = (w_grouped / scale).round().clamp(-bnt - 1, bnt) * scale
            w_qdq = w_qdq.reshape(out_feat, -1)
            if pad_len > 0:
                w_qdq = w_qdq[:, :in_feat]
            return w_qdq

        else:
            raise ValueError(f"Unknown weight method for INT: {method}")

    elif quant_type == "fp8":
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        if method == "per_tensor":
            scale = w.abs().max().clamp(min=1e-8) / fp8_max
            q = (w / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            return q.float() * scale

        elif method == "per_channel":
            scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / fp8_max
            q = (w / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            return q.float() * scale

        elif method == "per_block":
            out_feat, in_feat = w.shape
            # pad to block_size multiples
            pad_r = (block_size - in_feat % block_size) % block_size
            pad_b = (block_size - out_feat % block_size) % block_size
            if pad_r > 0 or pad_b > 0:
                w_padded = torch.nn.functional.pad(w, (0, pad_r, 0, pad_b))
            else:
                w_padded = w
            po, pi = w_padded.shape
            w_blocks = w_padded.reshape(po // block_size, block_size, pi // block_size, block_size)
            w_blocks = w_blocks.permute(0, 2, 1, 3)  # [bo, bi, bs, bs]
            scale = w_blocks.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8) / fp8_max
            q = (w_blocks / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            w_qdq = (q.float() * scale).permute(0, 2, 1, 3).reshape(po, pi)
            return w_qdq[:out_feat, :in_feat]

        elif method == "per_group":
            # FP8 per_group: group along dim=1
            out_feat, in_feat = w.shape
            gs = group_size if group_size > 0 else in_feat
            if in_feat % gs != 0:
                pad_len = gs - in_feat % gs
                w_padded = torch.nn.functional.pad(w, (0, pad_len))
            else:
                w_padded = w
                pad_len = 0
            w_grouped = w_padded.reshape(out_feat, -1, gs)
            scale = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / fp8_max
            q = (w_grouped / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            w_qdq = (q.float() * scale).reshape(out_feat, -1)
            if pad_len > 0:
                w_qdq = w_qdq[:, :in_feat]
            return w_qdq

        else:
            raise ValueError(f"Unknown weight method for FP8: {method}")

    else:
        raise ValueError(f"Unknown weight quant_type: {quant_type}")
