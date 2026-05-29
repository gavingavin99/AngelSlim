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

"""Setup / teardown for online smooth-stat hooks and alpha-search hooks.

These are called via ``llm.apply_model(...)`` from the Phase 1 driver
``tools/smooth/run_vllm_smooth.py``.  They register PyTorch forward
hooks on dense Attention / down_proj layers and wire FusedMoE layers
for kernel-injected stat collection.
"""

import os

import torch

from angelslim.compressor.quant.core.vllm_calibrate_utils._common import _find_layers

from .hooks import SmoothAlphaValueHook, SmoothAttnHook, SmoothDownProjInputHook
from .moe_inject import set_moe_collect_alpha_search, set_moe_collect_smooth

__all__ = [
    "setup_smooth_hooks",
    "remove_smooth_hooks",
    "setup_smooth_alpha_search_hooks",
    "remove_smooth_alpha_search_hooks",
]


def setup_smooth_hooks(
    model,
    ema_momentum=0.9,
    collect_attn=True,
    collect_down_proj=True,
    collect_moe=True,
    token_clip=-1,
):
    """Register smooth hooks on the model to collect per-channel absmax
    and EMA stats.

    Args:
        model: The nn.Module model instance (called inside ``llm.apply_model``).
        ema_momentum: EMA decay factor (default 0.9).
        collect_attn: If True, register hooks on
            ``vllm.attention.layer.Attention`` to collect q / k inputs and
            attn output (= o_proj input).
        collect_down_proj: If True, register hooks on dense MLP
            ``down_proj`` layers to collect ``silu(gate)*up``.
        collect_moe: If True, wire FusedMoE layers for kernel-injected
            smooth stats.  Requires ``VLLM_MOE_COLLECT_SMOOTH_STATS=1``
            env var AND the corresponding call to
            :func:`collect_fused_moe_smooth_stats` inside the vLLM
            FusedMoE kernel source.
        token_clip: Optional percentile-based clipping for per-channel
            absmax collection.
            * ``<= 0`` (default): disabled, use absolute max.
            * In ``(0, 1]``: quantile fraction (e.g. 0.999).
            * In ``(1, 100)``: percentile value (e.g. 99.9).

    Statistics are stored in ``model._smooth_stats``::

        {
          "<attn_layer>.q":        {"absmax": Tensor, "ema": Tensor, "call_count": int},
          "<attn_layer>.k":        {...},
          "<attn_layer>.attn_out": {...},
          "<down_proj_layer>":     {...},
          "<moe_layer>.<i>.down_proj": {...},
        }
    """
    try:
        from vllm.model_executor.layers.attention import Attention
    except ImportError:
        from vllm.attention.layer import Attention
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase

    # ------------------------------------------------------------------
    # 1. Discover layers for each requested scope
    # ------------------------------------------------------------------
    attn_layers = _find_layers(model, layers=[Attention]) if collect_attn else {}
    down_proj_layers = {}
    if collect_down_proj:
        all_linear = _find_layers(model, layers=[torch.nn.Linear, LinearBase])
        down_proj_layers = {
            name: module
            for name, module in all_linear.items()
            if name.split(".")[-1] == "down_proj"
        }
    moe_layers = _find_layers(model, layers=[FusedMoE]) if collect_moe else {}

    # ------------------------------------------------------------------
    # EP (Expert Parallelism) check: smooth stat collection is NOT
    # compatible with EP — abort early to prevent silent wrong results.
    # ------------------------------------------------------------------
    if collect_moe and moe_layers:
        os.environ["VLLM_MOE_COLLECT_SMOOTH_STATS"] = "1"
        set_moe_collect_smooth(True)
        ep_layers = {
            name: layer
            for name, layer in moe_layers.items()
            if getattr(layer, "use_ep", False) and getattr(layer, "ep_size", 1) > 1
        }
        if ep_layers:
            ep_example = next(iter(ep_layers))
            ep_size = getattr(ep_layers[ep_example], "ep_size", "?")
            raise RuntimeError(
                f"[setup_smooth_hooks] Expert Parallelism (EP) is NOT supported for "
                f"smooth stat calibration. Detected {len(ep_layers)} FusedMoE layer(s) "
                f"with ep_size={ep_size} (e.g. '{ep_example}'). "
                f"Please re-run calibration with pure TP (set ep_size=1 / "
                f"disable expert parallelism)."
            )

    collect_moe_smooth = collect_moe and os.getenv("VLLM_MOE_COLLECT_SMOOTH_STATS", "0") == "1"

    print(
        f"---------Smooth hooks scope: "
        f"attn={'ON' if collect_attn else 'OFF'} ({len(attn_layers)} layers), "
        f"down_proj={'ON' if collect_down_proj else 'OFF'} ({len(down_proj_layers)} layers), "
        f"moe={'ON' if collect_moe else 'OFF'} ({len(moe_layers)} layers, "
        f"kernel_inject={'ON' if collect_moe_smooth else 'OFF'})---------"
    )
    if collect_moe and moe_layers and not collect_moe_smooth:
        print(
            "---------[WARNING] MoE scope enabled but VLLM_MOE_COLLECT_SMOOTH_STATS"
            " is not set to 1 — MoE stats will NOT be collected---------"
        )

    # ------------------------------------------------------------------
    # 2. Initialise stats storage
    # ------------------------------------------------------------------
    if not hasattr(model, "_smooth_stats"):
        model._smooth_stats = {}
    if not hasattr(model, "_smooth_stats_meta"):
        model._smooth_stats_meta = {}

    for name, _attn_module in attn_layers.items():
        # Read kv_head replica count from the qkv_proj inside the parent
        # LlamaAttention (or equivalent).
        kv_replicas = 1
        try:
            from vllm.model_executor.layers.linear import QKVParallelLinear

            parent_name = ".".join(name.split(".")[:-1])  # drop ".attn" suffix
            parent = model
            for part in parent_name.split("."):
                parent = getattr(parent, part)
            qkv_proj = getattr(parent, "qkv_proj", None)
            if isinstance(qkv_proj, QKVParallelLinear):
                kv_replicas = qkv_proj.num_kv_head_replicas
        except Exception:
            pass

        for tag in ("q", "k", "attn_out"):
            key = f"{name}.{tag}"
            if key not in model._smooth_stats:
                model._smooth_stats[key] = {"absmax": None, "ema": None, "call_count": 0}
            model._smooth_stats_meta[key] = {
                "kv_replicas": kv_replicas if tag == "k" else 1,
            }

    for name in down_proj_layers:
        if name not in model._smooth_stats:
            model._smooth_stats[name] = {"absmax": None, "ema": None, "call_count": 0}

    if collect_moe_smooth:
        for name, layer in moe_layers.items():
            num_experts = getattr(layer, "global_num_experts", None) or getattr(
                layer, "local_num_experts", None
            )
            if num_experts is None:
                print(
                    f"[WARNING] Cannot determine num_experts for {name}, "
                    f"skipping pre-allocation"
                )
                continue
            for expert_id in range(num_experts):
                key = f"{name}.{expert_id}.down_proj"
                if key not in model._smooth_stats:
                    model._smooth_stats[key] = {"absmax": None, "ema": None, "call_count": 0}

    # ------------------------------------------------------------------
    # 3. Register PyTorch forward hooks (idempotent guard)
    # ------------------------------------------------------------------
    if not hasattr(model, "_smooth_hooks"):
        model._smooth_hooks = []

        for name, layer in attn_layers.items():
            hook = SmoothAttnHook(
                name,
                model._smooth_stats,
                ema_momentum=ema_momentum,
                token_clip=token_clip,
            )
            handle = layer.register_forward_hook(hook)
            model._smooth_hooks.append(handle)

        for name, layer in down_proj_layers.items():
            hook = SmoothDownProjInputHook(
                name,
                model._smooth_stats,
                ema_momentum=ema_momentum,
                token_clip=token_clip,
            )
            handle = layer.register_forward_hook(hook)
            model._smooth_hooks.append(handle)

    # ------------------------------------------------------------------
    # 4. Wire FusedMoE layers for kernel-injected smooth stats
    # ------------------------------------------------------------------
    if collect_moe_smooth:
        for name, layer in moe_layers.items():
            if hasattr(layer, "w13_weight") and layer.w13_weight is not None:
                layer.w13_weight._vllm_layer_name_smooth = name
                layer.w13_weight._smooth_stats_of_model = model._smooth_stats
                layer.w13_weight._smooth_ema_momentum = ema_momentum
            else:
                print(
                    f"[DEBUG] Cannot set smooth attrs on w13_weight for {name}: "
                    f"hasattr={hasattr(layer, 'w13_weight')}, "
                    f"is_none={getattr(layer, 'w13_weight', None) is None}"
                )

    print(f"---------Smooth hooks registered: {len(model._smooth_hooks)} total---------")
    return f"Registered {len(model._smooth_hooks)} smooth hooks"


def remove_smooth_hooks(model):
    """Remove smooth hooks registered by :func:`setup_smooth_hooks`."""
    if hasattr(model, "_smooth_hooks"):
        for handle in model._smooth_hooks:
            handle.remove()
        model._smooth_hooks.clear()
    os.environ.pop("VLLM_MOE_COLLECT_SMOOTH_STATS", None)
    set_moe_collect_smooth(False)
    return "Smooth hooks removed"


def setup_smooth_alpha_search_hooks(
    model,
    max_tokens=4096,
    collect_moe=True,
):
    """Register hooks to capture raw down_proj activations for alpha
    grid search.

    Must be called via ``llm.apply_model()`` AFTER
    :func:`setup_smooth_hooks`.

    Dense MLP: registers :class:`SmoothAlphaValueHook` on each
    ``down_proj`` layer.
    MoE: wires ``w13_weight`` attributes so that
    :func:`collect_fused_moe_alpha_search_values` (called from the kernel
    injection point) stores raw tensors per expert.
    """
    from vllm.model_executor.layers.linear import LinearBase

    if not hasattr(model, "_alpha_search_values"):
        model._alpha_search_values = {}
    if not hasattr(model, "_alpha_search_hooks"):
        model._alpha_search_hooks = []

    # --- Dense MLP down_proj ---
    all_linear = _find_layers(model, layers=[torch.nn.Linear, LinearBase])
    down_proj_layers = {
        name: module for name, module in all_linear.items() if name.split(".")[-1] == "down_proj"
    }

    for name, layer in down_proj_layers.items():
        if name not in model._alpha_search_values:
            model._alpha_search_values[name] = {"tokens": None}
        hook = SmoothAlphaValueHook(name, model._alpha_search_values, max_tokens)
        handle = layer.register_forward_hook(hook)
        model._alpha_search_hooks.append(handle)

    # --- MoE (kernel injection) ---
    moe_count = 0
    if collect_moe:
        try:
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE

            moe_layers = _find_layers(model, layers=[FusedMoE])
        except ImportError:
            moe_layers = {}

        if moe_layers:
            os.environ["VLLM_MOE_COLLECT_ALPHA_SEARCH"] = "1"
            set_moe_collect_alpha_search(True)

        per_expert_max = max(1, max_tokens // 8)  # for topk=8
        for name, layer in moe_layers.items():
            if hasattr(layer, "w13_weight") and layer.w13_weight is not None:
                layer.w13_weight._alpha_search_values_of_model = model._alpha_search_values
                layer.w13_weight._vllm_layer_name_smooth = name
                layer.w13_weight._alpha_search_max_tokens = per_expert_max
                moe_count += 1
            else:
                print(
                    f"[DEBUG] Cannot set smooth attrs on w13_weight for {name}: "
                    f"hasattr={hasattr(layer, 'w13_weight')}, "
                    f"is_none={getattr(layer, 'w13_weight', None) is None}"
                )
    total = len(model._alpha_search_hooks)
    msg = (
        f"Registered {total} alpha-search hooks (dense down_proj), "
        f"{moe_count} MoE layers wired"
    )
    print(f"[Alpha Search] {msg}")
    return msg


def remove_smooth_alpha_search_hooks(model):
    """Remove alpha-search hooks and free stored activation tensors."""
    if hasattr(model, "_alpha_search_hooks"):
        for handle in model._alpha_search_hooks:
            handle.remove()
        model._alpha_search_hooks.clear()
    if hasattr(model, "_alpha_search_values"):
        del model._alpha_search_values
    os.environ.pop("VLLM_MOE_COLLECT_ALPHA_SEARCH", None)
    set_moe_collect_alpha_search(False)
    return "Alpha-search hooks removed"
