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

"""Down-projection smooth — fold MLP / MoE down_proj input absmax into
``down_proj`` and ``up_proj`` (or fused 3-D MoE ``down_proj`` /
``gate_up_proj``) weights.

Two flavours:

* :func:`apply_down_proj_smooth` computes ``smooth_weight`` from
  ``smooth_stats`` using a fixed alpha (``smooth = absmax^alpha
  / weight_absmax^(1-alpha)``).
* :func:`apply_down_proj_smooth_from_search` reads pre-computed
  per-layer ``smooth_weight`` produced by the alpha grid searcher.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from ..core.tensor_ops import inplace_div_fp32, inplace_mul_fp32
from .key_maps import DEFAULT_KEY_MAP
from .module_finder import get_submodule_safe, maybe_materialize

__all__ = [
    "apply_down_proj_smooth",
    "apply_down_proj_smooth_from_search",
]


# ---------------------------------------------------------------------------
# Fused-experts MoE smooth helper
# ---------------------------------------------------------------------------


def _apply_smooth_fused_expert(
    experts_module: torch.nn.Module,
    expert_idx: int,
    smooth_weight: torch.Tensor,
    fused_down_attr: str = "down_proj",
    fused_gate_up_attr: str = "gate_up_proj",
):
    """Apply down_proj smooth to a single expert in fused 3-D-tensor MoE
    (HYV3Experts / Qwen3MoeExperts style).

    Layout::

        experts.down_proj      shape [E, H, I]
        experts.gate_up_proj   shape [E, 2*I, H]   (gate then up)
            gate_part = gate_up_proj[i, :I, :]   (untouched)
            up_part   = gate_up_proj[i, I:, :]   (divided by smooth_weight)

    All math is done in fp32 then cast back to each tensor's original
    dtype.  Different ``expert_idx`` values touch disjoint storage
    regions of the same Parameter, so concurrent invocation across
    threads is safe.
    """
    down_param = getattr(experts_module, fused_down_attr)
    gate_up_param = getattr(experts_module, fused_gate_up_attr)

    intermediate_dim = down_param.shape[2]
    if smooth_weight.shape != (intermediate_dim,):
        raise ValueError(
            f"smooth_weight shape {tuple(smooth_weight.shape)} != ({intermediate_dim},)"
        )
    if gate_up_param.shape[1] != 2 * intermediate_dim:
        raise ValueError(
            f"gate_up_proj rows ({gate_up_param.shape[1]}) != 2*intermediate_dim "
            f"({2 * intermediate_dim}); fused gate-then-up layout assumption violated."
        )

    # Pick a non-meta target device if available.
    target_device = None
    for p in (down_param, gate_up_param):
        if not p.is_meta:
            target_device = p.device
            break
    if target_device is None:
        target_device = torch.device("cpu")

    hook = getattr(experts_module, "_hf_hook", None)
    use_hook = hook is not None and (down_param.is_meta or gate_up_param.is_meta)
    if use_hook:
        original_exec_device = hook.execution_device
        hook.execution_device = target_device
        hook.pre_forward(experts_module)

    absmax_before, absmax_after = None, None
    try:
        absmax_before = down_param.data[expert_idx].abs().max().item()

        sw = smooth_weight.to(device=target_device).float()

        # 1) down_proj[expert_idx] *= sw   (broadcast on input channel)
        dp_orig_dtype = down_param.dtype
        dp_slice = down_param.data[expert_idx]  # view [H, I]
        dp_slice.copy_(dp_slice.float().mul_(sw.view(1, -1)).to(dp_orig_dtype))

        # 2) gate_up_proj[expert_idx, I:, :] /= sw   (only the up half)
        gu_orig_dtype = gate_up_param.dtype
        gu_up_slice = gate_up_param.data[expert_idx, intermediate_dim:, :]  # [I, H]
        gu_up_slice.copy_(gu_up_slice.float().div_(sw.view(-1, 1)).to(gu_orig_dtype))

        absmax_after = down_param.data[expert_idx].abs().max().item()
    finally:
        if use_hook:
            hook.post_forward(experts_module, None)
            hook.execution_device = original_exec_device
    return absmax_before, absmax_after


# ---------------------------------------------------------------------------
# Resolve a smooth-stat key to a target module (linear / fused) or skip
# ---------------------------------------------------------------------------


def _resolve_down_target(
    model: torch.nn.Module,
    stat_key: str,
    all_named_modules: dict,
    up_proj_map: dict,
    km: dict,
):
    """Resolve a smooth-stats key to a target for down_proj smooth.

    Returns one of::

        ("linear", down_proj_module, up_proj_module,
                   down_proj_full_name, up_proj_full_name)
            -- per-expert nn.Linear / dense MLP / shared MLP path.

        ("fused",  experts_module, expert_idx, parent_path,
                   fused_down_attr, fused_gate_up_attr)
            -- fused 3D-tensor MoE path.

        ("none",   reason_str)
            -- could not resolve, caller should skip with reason.

    Strategy: try the nn.Linear path first (covers dense MLP and legacy
    per-expert MoE).  Fall back to fused-experts pattern matching only
    when no Linear submodule with a 2-D weight can be located.
    """
    down_name = km["down_proj"]

    # Build vLLM->HF candidate keys via stat_path_aliases.
    aliases = km.get("stat_path_aliases", [])
    candidate_keys = [stat_key]
    for _pat, _dst in aliases:
        translated = re.sub(_pat, _dst, stat_key)
        if translated != stat_key and translated not in candidate_keys:
            candidate_keys.append(translated)

    # ---------------- nn.Linear path ----------------
    matched_name = None
    down_proj_module = None
    for cand in candidate_keys:
        mod = get_submodule_safe(model, cand)
        if mod is None:
            continue
        w = getattr(mod, "weight", None)
        if w is not None and hasattr(w, "dim") and w.dim() == 2:
            matched_name = cand
            down_proj_module = mod
            break

    if matched_name is None:
        # Suffix match: stat_key may have missing/extra prefix.
        for name, mod in all_named_modules.items():
            if any(name.endswith(c) or c.endswith(name) for c in candidate_keys):
                w = getattr(mod, "weight", None)
                if w is not None and hasattr(w, "dim") and w.dim() == 2:
                    matched_name = name
                    down_proj_module = mod
                    break

    if matched_name is not None:
        parent = matched_name[: -(len(down_name) + 1)]
        up_proj_name = up_proj_map.get(parent)
        if up_proj_name is None:
            return ("none", f"{matched_name}: cannot find matching up_proj")
        up_proj_module = all_named_modules.get(up_proj_name)
        if up_proj_module is None or not hasattr(up_proj_module, "weight"):
            return ("none", f"{matched_name}: up_proj {up_proj_name} not usable")
        return ("linear", down_proj_module, up_proj_module, matched_name, up_proj_name)

    # ---------------- fused-experts path ----------------
    fused_down_attr = km.get("fused_down_attr")
    fused_gate_up_attr = km.get("fused_gate_up_attr")
    if not fused_down_attr or not fused_gate_up_attr:
        return (
            "none",
            f"{stat_key}: cannot resolve to nn.Linear; fused experts not "
            f"configured in key_map (missing fused_down_attr / fused_gate_up_attr)",
        )

    fused_re = re.compile(
        r"^(?P<prefix>.*\.experts)\.(?P<idx>\d+)\." + re.escape(down_name) + r"$"
    )

    m = None
    for cand in candidate_keys:
        m = fused_re.match(cand)
        if m is not None:
            break
    if m is None:
        for name in all_named_modules:
            if any(name.endswith(c) or c.endswith(name) for c in candidate_keys):
                m = fused_re.match(name)
                if m:
                    break

    if m is None:
        return (
            "none",
            f"{stat_key}: no nn.Linear and pattern .experts.<i>.{down_name} does not match",
        )

    parent_path = m.group("prefix")
    expert_idx = int(m.group("idx"))

    experts_module = all_named_modules.get(parent_path) or get_submodule_safe(model, parent_path)
    if experts_module is None:
        return ("none", f"{stat_key}: experts container {parent_path!r} not found")

    down_param = getattr(experts_module, fused_down_attr, None)
    gate_up_param = getattr(experts_module, fused_gate_up_attr, None)

    if not isinstance(down_param, torch.nn.Parameter) or down_param.dim() != 3:
        return (
            "none",
            f"{parent_path}.{fused_down_attr} is not a 3D nn.Parameter "
            f"(got {type(down_param).__name__})",
        )
    if not isinstance(gate_up_param, torch.nn.Parameter) or gate_up_param.dim() != 3:
        return (
            "none",
            f"{parent_path}.{fused_gate_up_attr} is not a 3D nn.Parameter "
            f"(got {type(gate_up_param).__name__})",
        )

    if expert_idx < 0 or expert_idx >= down_param.shape[0]:
        return (
            "none",
            f"{parent_path}: expert_idx={expert_idx} out of range [0, {down_param.shape[0]})",
        )

    return ("fused", experts_module, expert_idx, parent_path, fused_down_attr, fused_gate_up_attr)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def apply_down_proj_smooth(
    model: torch.nn.Module,
    smooth_stats: dict,
    alpha: float = 0.5,
    num_workers: int = 8,
    exclude_keys=None,
    km: dict = None,
) -> None:
    """Apply down_proj smooth to every layer that has down_proj stats.

    Smooth weight formula::

        smooth_weight = absmax^alpha / weight_absmax^(1-alpha)

    Application::

        down_proj.weight *= smooth_weight   (per input channel, dim=1)
        up_proj.weight   /= smooth_weight   (per output channel, dim=0)

    Parallelism:
        Uses a ``ThreadPoolExecutor``; falls back to serial when
        accelerate hooks are detected.
    """
    km = km or DEFAULT_KEY_MAP
    up_name = km["up_proj"]

    _DOWN_PATTERNS = [re.compile(p) for p in km["down_patterns"]]
    down_keys = [k for k in smooth_stats if any(pat.search(k) for pat in _DOWN_PATTERNS)]

    if exclude_keys:
        before = len(down_keys)
        down_keys = [k for k in down_keys if k not in exclude_keys]
        skipped = before - len(down_keys)
        if skipped > 0:
            print(f"  [Down Smooth] Skipped {skipped} keys already handled by alpha search")

    print(
        f"\n[Down Smooth] {len(down_keys)} down_proj stat entries to process "
        f"(fixed alpha={alpha})"
    )

    if not down_keys:
        print("  [SKIP] No down_proj stats found in smooth-stats")
        return

    # Build named_modules dict and O(1) up_proj lookup table.
    all_named_modules = {name: mod for name, mod in model.named_modules()}

    up_proj_map = {}
    for name in all_named_modules:
        if name.endswith("." + up_name):
            parent = name[: -(len(up_name) + 1)]
            up_proj_map[parent] = name

    has_hf_hook = any(
        getattr(mod, "_hf_hook", None) is not None for mod in all_named_modules.values()
    )
    effective_workers = 1 if has_hf_hook else num_workers
    if has_hf_hook and num_workers > 1:
        print("  [Parallel] accelerate _hf_hook detected -> fallback to serial (num_workers=1)")
    else:
        print(f"  [Parallel] num_workers={effective_workers}")

    def _process_one(stat_key: str) -> None:
        target = _resolve_down_target(model, stat_key, all_named_modules, up_proj_map, km)
        kind = target[0]
        if kind == "none":
            print(f"  [SKIP] {stat_key}: {target[1]}")
            return

        scale = smooth_stats[stat_key]["scale"]
        if scale is None:
            print(f"  [SKIP] {stat_key}: scale is None")
            return

        raw = scale.to(torch.float32)
        if torch.any(~torch.isfinite(raw)) or torch.any(raw <= 0):
            n_neginf = torch.sum(torch.isneginf(raw)).item()
            n_nan = torch.sum(torch.isnan(raw)).item()
            n_nonpos = torch.sum(raw <= 0).item()
            print(
                f"  [SKIP] {stat_key}: invalid scale values "
                f"(-inf={n_neginf}, nan={n_nan}, <=0={n_nonpos}), skipping this layer"
            )
            return

        absmax = raw.clamp(min=1e-8)

        if kind == "linear":
            (_, down_proj_module, up_proj_module, down_proj_module_name, up_proj_name) = target

            device = (
                down_proj_module.weight.device
                if not down_proj_module.weight.is_meta
                else torch.device("cpu")
            )

            d_hook = getattr(down_proj_module, "_hf_hook", None)
            if d_hook is not None and down_proj_module.weight.device.type == "meta":
                _tmp_device = torch.device("cpu")
                _orig_exec = d_hook.execution_device
                d_hook.execution_device = _tmp_device
                d_hook.pre_forward(down_proj_module)
                weight_absmax = (
                    down_proj_module.weight.detach().float().abs().max(dim=0).values
                ).clamp(min=1e-8)
                d_hook.post_forward(down_proj_module, None)
                d_hook.execution_device = _orig_exec
            else:
                weight_absmax = (
                    down_proj_module.weight.detach().float().abs().max(dim=0).values
                ).clamp(min=1e-8)

            smooth_weight = absmax.pow(alpha) / weight_absmax.pow(1.0 - alpha)
            smooth_weight = smooth_weight.to(device=device)

            with maybe_materialize(down_proj_module, device):
                inplace_mul_fp32(down_proj_module.weight, smooth_weight.view(1, -1))

            up_device = (
                up_proj_module.weight.device
                if not up_proj_module.weight.is_meta
                else torch.device("cpu")
            )
            smooth_weight_up = smooth_weight.to(device=up_device)

            with maybe_materialize(up_proj_module, up_device):
                inplace_div_fp32(up_proj_module.weight, smooth_weight_up.view(-1, 1))
                if up_proj_module.bias is not None:
                    inplace_div_fp32(up_proj_module.bias, smooth_weight_up)

            print(
                f"  [Down] {down_proj_module_name} <-> {up_proj_name}: "
                f"in_features={absmax.shape[0]}, "
                f"smooth range=["
                f"{smooth_weight.min().item():.4f}, "
                f"{smooth_weight.max().item():.4f}]"
            )

        elif kind == "fused":
            (_, experts_module, expert_idx, parent_path, fdn_attr, fgu_attr) = target

            down_param = getattr(experts_module, fdn_attr)

            d_hook = getattr(experts_module, "_hf_hook", None)
            if d_hook is not None and down_param.is_meta:
                _orig_exec = d_hook.execution_device
                d_hook.execution_device = torch.device("cpu")
                d_hook.pre_forward(experts_module)
                weight_absmax = (
                    down_param.detach()[expert_idx].float().abs().max(dim=0).values
                ).clamp(min=1e-8)
                d_hook.post_forward(experts_module, None)
                d_hook.execution_device = _orig_exec
            else:
                weight_absmax = (
                    down_param.detach()[expert_idx].float().abs().max(dim=0).values
                ).clamp(min=1e-8)

            smooth_weight = absmax.pow(alpha) / weight_absmax.pow(1.0 - alpha)

            absmax_before, absmax_after = _apply_smooth_fused_expert(
                experts_module,
                expert_idx,
                smooth_weight,
                fused_down_attr=fdn_attr,
                fused_gate_up_attr=fgu_attr,
            )

            print(
                f"  [Down-Fused] {parent_path}[{expert_idx}]: "
                f"in_features={absmax.shape[0]}, "
                f"absmax_before={absmax_before}, absmax_after={absmax_after}, "
                f"smooth range=["
                f"{smooth_weight.min().item():.4f}, "
                f"{smooth_weight.max().item():.4f}]"
            )

    sorted_keys = sorted(down_keys)

    if effective_workers <= 1:
        for stat_key in sorted_keys:
            _process_one(stat_key)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_process_one, k): k for k in sorted_keys}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    key = futures[fut]
                    print(f"  [ERROR] {key}: unhandled exception: {exc}")


def apply_down_proj_smooth_from_search(
    model: torch.nn.Module,
    alpha_search_results: dict,
    num_workers: int = 8,
    km: dict = None,
):
    """Apply pre-computed ``smooth_weight`` from alpha search to
    ``down_proj`` / ``up_proj`` (or fused 3-D MoE).

    Unlike :func:`apply_down_proj_smooth`, this function does NOT
    compute ``smooth_weight`` from absmax — it reads the pre-computed
    ``smooth_weight`` directly from ``alpha_search_results`` (output of
    ``SmoothAlphaSearcher``).

    ``alpha_search_results`` format::

        {
          "config": {...},
          "results": {
            "model.layers.0.mlp.down_proj": {
              "alpha": [...],
              "smooth_weight": [float, ...],   # complete [in_features]
              "loss": float
            },
            ...
          }
        }

    Application is identical to :func:`apply_down_proj_smooth`.

    Returns:
        Set of stat_keys successfully processed (used by callers to
        exclude them from a follow-up fixed-alpha pass).
    """
    km = km or DEFAULT_KEY_MAP
    up_name = km["up_proj"]

    results = alpha_search_results.get("results", alpha_search_results)
    if not results:
        print("[Down Smooth Search] No results found in alpha_search_results")
        return set()

    print(f"\n[Down Smooth Search] {len(results)} entries from alpha search")

    all_named_modules = {name: mod for name, mod in model.named_modules()}
    up_proj_map = {}
    for name in all_named_modules:
        if name.endswith("." + up_name):
            parent = name[: -(len(up_name) + 1)]
            up_proj_map[parent] = name

    has_hf_hook = any(
        getattr(mod, "_hf_hook", None) is not None for mod in all_named_modules.values()
    )
    effective_workers = 1 if has_hf_hook else num_workers
    if has_hf_hook and num_workers > 1:
        print("  [Parallel] accelerate _hf_hook detected -> fallback to serial")
    else:
        print(f"  [Parallel] num_workers={effective_workers}")

    def _process_one(stat_key: str, entry: dict):
        smooth_weight_list = entry.get("smooth_weight")
        if smooth_weight_list is None:
            print(f"  [SKIP] {stat_key}: no smooth_weight in search results")
            return None

        smooth_weight = torch.tensor(smooth_weight_list, dtype=torch.float32)

        if torch.any(~torch.isfinite(smooth_weight)):
            print(f"  [SKIP] {stat_key}: smooth_weight contains inf/nan")
            return None

        target = _resolve_down_target(model, stat_key, all_named_modules, up_proj_map, km)
        kind = target[0]
        if kind == "none":
            print(f"  [SKIP] {stat_key}: {target[1]}")
            return None

        alpha_info = entry.get("alpha", "?")

        if kind == "linear":
            (_, down_proj_module, up_proj_module, down_proj_module_name, up_proj_name) = target

            device = (
                down_proj_module.weight.device
                if not down_proj_module.weight.is_meta
                else torch.device("cpu")
            )
            smooth_weight_dev = smooth_weight.to(device=device)

            with maybe_materialize(down_proj_module, device):
                absmax_down_before = down_proj_module.weight.abs().max().item()
                inplace_mul_fp32(down_proj_module.weight, smooth_weight_dev.view(1, -1))
                absmax_down_after = down_proj_module.weight.abs().max().item()

            up_device = (
                up_proj_module.weight.device
                if not up_proj_module.weight.is_meta
                else torch.device("cpu")
            )
            smooth_weight_up = smooth_weight.to(device=up_device)

            with maybe_materialize(up_proj_module, up_device):
                inplace_div_fp32(up_proj_module.weight, smooth_weight_up.view(-1, 1))
                if up_proj_module.bias is not None:
                    inplace_div_fp32(up_proj_module.bias, smooth_weight_up)

            print(
                f"  [Down-Search] {down_proj_module_name} <-> {up_proj_name}: "
                f"in_features={smooth_weight.shape[0]}, "
                f"absmax_down_before={absmax_down_before:.4f}, "
                f"absmax_down_after={absmax_down_after:.4f}, "
                f"smooth range=["
                f"{smooth_weight.min().item():.4f}, "
                f"{smooth_weight.max().item():.4f}], "
                f"alpha={alpha_info}"
            )
            return stat_key

        elif kind == "fused":
            (_, experts_module, expert_idx, parent_path, fdn_attr, fgu_attr) = target

            absmax_before, absmax_after = _apply_smooth_fused_expert(
                experts_module,
                expert_idx,
                smooth_weight,
                fused_down_attr=fdn_attr,
                fused_gate_up_attr=fgu_attr,
            )

            print(
                f"  [Down-Search-Fused] {parent_path}[{expert_idx}]: "
                f"in_features={smooth_weight.shape[0]}, "
                f"absmax_before={absmax_before:.4f}, absmax_after={absmax_after:.4f}, "
                f"smooth range=["
                f"{smooth_weight.min().item():.4f}, "
                f"{smooth_weight.max().item():.4f}], "
                f"alpha={alpha_info}"
            )
            return stat_key

        return None

    processed_keys = set()
    sorted_items = sorted(results.items())
    if effective_workers <= 1:
        for stat_key, entry in sorted_items:
            result = _process_one(stat_key, entry)
            if result is not None:
                processed_keys.add(result)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_process_one, k, v): k for k, v in sorted_items}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    key = futures[fut]
                    print(f"  [ERROR] {key}: {exc}")
                else:
                    result = fut.result()
                    if result is not None:
                        processed_keys.add(result)

    print(
        f"  [Down-Search] Successfully processed "
        f"{len(processed_keys)} / {len(results)} entries"
    )
    return processed_keys
