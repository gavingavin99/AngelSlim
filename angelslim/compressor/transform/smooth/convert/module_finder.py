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

"""Module-discovery helpers used by the offline weight converter.

* :func:`get_submodule_safe` — name-based submodule lookup with a None
  fallback.
* :func:`maybe_materialize` — context manager that temporarily
  materialises a meta-device weight via accelerate hooks for in-place
  modification, then offloads it again.
* :func:`attn_key_to_hf_prefix` — translate a vLLM smooth-stat key to
  a HuggingFace ``self_attn`` module path.
"""

import contextlib

import torch

from .key_maps import DEFAULT_KEY_MAP

__all__ = [
    "get_submodule_safe",
    "maybe_materialize",
    "attn_key_to_hf_prefix",
]


def get_submodule_safe(model: torch.nn.Module, name: str):
    """Return a nested submodule by dotted name, or ``None`` if not found."""
    try:
        return model.get_submodule(name)
    except AttributeError:
        return None


@contextlib.contextmanager
def maybe_materialize(linear: torch.nn.Module, target_device):
    """Temporarily materialise meta-device weights for in-place
    modification; offload them back via accelerate hooks.

    If the linear layer has no accelerate hook or weights are not on
    meta device, this is a no-op and the caller can modify weights
    directly.

    Usage::

        with maybe_materialize(proj, device):
            proj.weight.data.mul_(scale)
    """
    hook = getattr(linear, "_hf_hook", None)
    if hook is not None and linear.weight.device.type == "meta":
        original_exec_device = hook.execution_device
        hook.execution_device = target_device
        hook.pre_forward(linear)
        try:
            yield
        finally:
            hook.post_forward(linear, None)
            hook.execution_device = original_exec_device
    else:
        yield


def attn_key_to_hf_prefix(smooth_key_base: str, km: dict = None) -> str:
    """Map vLLM Attention layer name to HuggingFace module prefix.

    vLLM wraps the raw attention op in a sub-module ``.attn``, so::

        "model.layers.0.self_attn.attn"  ->  "model.layers.0.self_attn"

    If the key does not end with the ``attn_strip`` suffix we return
    it unchanged (fallback behaviour).
    """
    km = km or DEFAULT_KEY_MAP
    attn_strip = km["attn_strip"]
    if smooth_key_base.endswith(attn_strip):
        return smooth_key_base[: -len(attn_strip)]
    return smooth_key_base
