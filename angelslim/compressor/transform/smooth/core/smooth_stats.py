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

"""Smooth-stat data structure & JSON I/O.

This is the single bridge between Phase 1 (vLLM online stat collection)
and Phase 2 (offline weight conversion): the JSON file written by
``get_smooth_stats`` is consumed by ``apply_qk_smooth`` /
``apply_vo_smooth`` / ``apply_down_proj_smooth`` etc.

The dataclass-style helper here is purely for convenience — most call
sites still operate on the plain ``dict[str, dict]`` shape produced by
``get_smooth_stats`` / ``json.load``, so we expose both views.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch

__all__ = [
    "SmoothStats",
    "load_smooth_stats",
    "save_smooth_stats",
    "load_alpha_search_results",
    "save_alpha_search_results",
]


@dataclass
class SmoothStats:
    """A single layer's smooth statistics."""

    absmax: Optional[torch.Tensor]  # [C] per-channel absmax
    ema: Optional[torch.Tensor]  # [C] per-channel EMA
    call_count: int = 0


def load_smooth_stats(json_path: str, use_ema: bool = False) -> dict:
    """Load smooth stats from JSON file and convert lists back to tensors.

    The original ``smooth_stats.json`` schema is::

        {
          "<layer_key>": {
            "absmax": [float, ...],
            "ema":    [float, ...],
            "call_count": int
          },
          ...
        }

    Args:
        json_path: path to smooth_stats.json.
        use_ema: if True, use the EMA field as the per-channel scale;
            otherwise use absmax.

    Returns:
        ``dict[layer_key, {"scale": Tensor[C], "call_count": int}]``.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    field = "ema" if use_ema else "absmax"
    out = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        vals = entry.get(field)
        if vals is None:
            continue
        out[key] = {
            "scale": torch.tensor(vals, dtype=torch.float32),
            "call_count": int(entry.get("call_count", 0)),
        }
    return out


def save_smooth_stats(stats: dict, json_path: str) -> None:
    """Serialise ``model._smooth_stats``-shaped data to JSON.

    Args:
        stats: the dict returned by ``get_smooth_stats(model)`` — already
            JSON-serialisable (lists, ints).
        json_path: target file path.
    """
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)


def load_alpha_search_results(json_path: str) -> dict:
    """Load smooth_alpha_search.json.

    Returns the inner ``"results"`` dict if present, otherwise the whole
    payload (for forwards compat with older / flatter formats)::

        {
          "<layer_key>": {
            "alpha": [float, ...],     # per-rank list under TP
            "smooth_weight": [float],  # full [intermediate_size]
            "loss": float
          },
          ...
        }
    """
    with open(json_path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "results" in raw:
        return raw["results"]
    return raw


def save_alpha_search_results(payload: dict, json_path: str) -> None:
    """Save alpha search payload (config + results) to JSON."""
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
