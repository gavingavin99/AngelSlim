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

"""Pre/post-transform forward snapshot & numerical verification.

Two flavours:
* attention-level — :func:`snapshot_attn_output_before` records the
  output of the first attention module and
  :func:`verify_attn_output_diff` re-runs it after the smooth transform
  and prints the difference.
* MLP-level — :func:`snapshot_mlp_outputs_before` /
  :func:`verify_mlp_output_diff` do the same for MLP containers.
"""

import contextlib
import inspect
import re

import torch

from .key_maps import DEFAULT_KEY_MAP
from .module_finder import attn_key_to_hf_prefix, get_submodule_safe

__all__ = [
    "find_first_attn_module",
    "snapshot_attn_output_before",
    "verify_attn_output_diff",
    "snapshot_mlp_outputs_before",
    "verify_mlp_output_diff",
]


# ---------------------------------------------------------------------------
# Attention snapshot / verify
# ---------------------------------------------------------------------------


def find_first_attn_module(model: torch.nn.Module, smooth_stats: dict, km: dict = None):
    """Find the first ``self_attn`` submodule that exists in both the
    model and the smooth-stats dict.

    Returns:
        ``(hf_prefix, attn_module)`` tuple, or ``(None, None)``.
    """
    km = km or DEFAULT_KEY_MAP
    stat_k = km["stat_k"]
    stat_attn_out = km["stat_attn_out"]

    bases = set()
    for key in smooth_stats:
        if key.endswith(stat_k):
            bases.add(key[: -len(stat_k)])
        elif key.endswith(stat_attn_out):
            bases.add(key[: -len(stat_attn_out)])

    for base in sorted(bases):
        hf_prefix = attn_key_to_hf_prefix(base, km)
        attn_module = get_submodule_safe(model, hf_prefix)
        if attn_module is None:
            continue
        if all(
            getattr(attn_module, km[p], None) is not None
            for p in ("q_proj", "k_proj", "v_proj", "o_proj")
        ):
            return hf_prefix, attn_module

    return None, None


def _run_attn_forward(
    attn_module,
    hidden_states: torch.Tensor,
    rope_cache=None,
) -> torch.Tensor:
    """Run ``self_attn`` forward, using ``inspect`` to detect accepted
    parameters.

    ``hidden_states`` shape: ``(1, seq_len, hidden_size)`` (batch=1).

    Returns:
        Output tensor ``(1, seq_len, hidden_size)``, detached on cpu in float32.
    """
    seq_len = hidden_states.shape[1]
    device = hidden_states.device
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    cache_pos = torch.arange(seq_len, dtype=torch.long, device=device)

    # Construct position_embeddings (cos, sin) for newer model architectures.
    position_embeddings = None
    if rope_cache is not None:
        _head_dim = rope_cache.shape[-1] // 2
        cos_part = (
            rope_cache[:seq_len, :_head_dim]
            .unsqueeze(0)
            .to(dtype=hidden_states.dtype, device=device)
        )
        sin_part = (
            rope_cache[:seq_len, _head_dim:]
            .unsqueeze(0)
            .to(dtype=hidden_states.dtype, device=device)
        )
        position_embeddings = (cos_part, sin_part)

    all_kwargs = dict(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        rope_cache=rope_cache,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=cache_pos,
    )

    try:
        sig = inspect.signature(attn_module.forward)
        params = sig.parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_var_keyword:
            kwargs = all_kwargs
        else:
            kwargs = {k: v for k, v in all_kwargs.items() if k in params}
    except (ValueError, TypeError):
        kwargs = all_kwargs

    cfg = getattr(attn_module, "config", None)
    orig_attn_impl = getattr(cfg, "_attn_implementation", None) if cfg else None
    if orig_attn_impl not in (None, "eager", "flash_attention_2"):
        cfg._attn_implementation = "eager"

    try:
        out = attn_module(**kwargs)
    finally:
        if orig_attn_impl is not None and cfg is not None:
            cfg._attn_implementation = orig_attn_impl

    result = out[0] if isinstance(out, (tuple, list)) else out
    return result.detach().cpu().float()


def snapshot_attn_output_before(
    model: torch.nn.Module,
    smooth_stats: dict,
    seq_len: int = 8,
    seed: int = 42,
    km: dict = None,
):
    """Record attn forward output BEFORE smooth transformation for verification.

    Returns:
        Dict with keys ``hf_prefix``, ``x``, ``out_before``,
        ``rope_cache``, ``seed``; or ``None`` if no suitable layer is found.
    """
    km = km or DEFAULT_KEY_MAP
    hf_prefix, attn_module = find_first_attn_module(model, smooth_stats, km)
    if attn_module is None:
        print("  [Verify] No suitable attn layer found, skipping verification")
        return None

    weight = getattr(attn_module, km["q_proj"]).weight
    hidden_size = weight.shape[1]
    model_dtype = weight.dtype
    model_device = weight.device

    cfg = model.config
    _head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    _max_seq = getattr(cfg, "max_position_embeddings", 4096)
    _base = getattr(cfg, "rope_theta", 10000.0)
    freqs = 1.0 / (
        _base
        ** (
            torch.arange(0, _head_dim, 2, dtype=torch.float32, device=model_device)[
                : (_head_dim // 2)
            ]
            / _head_dim
        )
    )
    t = torch.arange(_max_seq, dtype=freqs.dtype, device=model_device)
    idx_theta = torch.outer(t, freqs)
    freqs_cat = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs_cat.cos(), freqs_cat.sin()], dim=-1)
    print(
        f"  [Verify] Generated rope_cache: shape={list(rope_cache.shape)}, "
        f"head_dim={_head_dim}, base={_base}, device={model_device}"
    )

    torch.manual_seed(seed)
    x = torch.randn(1, seq_len, hidden_size, dtype=model_dtype, device=model_device)

    print(
        f"\n[Verify] Recording pre-transform output -> layer={hf_prefix}, "
        f"input shape={list(x.shape)}, dtype={model_dtype}, "
        f"device={model_device}, seed={seed}"
    )

    with torch.no_grad():
        out_before = _run_attn_forward(attn_module, x, rope_cache=rope_cache)

    return {
        "hf_prefix": hf_prefix,
        "x": x,
        "out_before": out_before,
        "rope_cache": rope_cache,
        "seed": seed,
    }


def verify_attn_output_diff(
    snap,
    model: torch.nn.Module,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    km: dict = None,
) -> None:
    """Compare attn forward output before and after smooth transformation."""
    km = km or DEFAULT_KEY_MAP
    print("\n" + "=" * 70)
    print("[Verify] Attention output diff (before vs after transform)")

    if snap is None:
        print("  Skipped: no snapshot available")
        print("=" * 70)
        return

    hf_prefix = snap["hf_prefix"]
    attn_module = get_submodule_safe(model, hf_prefix)
    if attn_module is None:
        print(f"  Skipped: module {hf_prefix} not found after transform")
        print("=" * 70)
        return

    if getattr(attn_module, km["qk_norm_flag"], False) or getattr(model.config, "qk_norm", False):
        print(f"  [NOTE] {hf_prefix} has qk_norm=True, use QK smooth.")

    x = snap["x"]
    out_before = snap["out_before"]
    rope_cache = snap.get("rope_cache")

    with torch.no_grad():
        out_after = _run_attn_forward(attn_module, x, rope_cache=rope_cache)

    diff = (out_after - out_before).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_max = (diff / out_before.abs().clamp(min=1e-8)).max().item()
    is_close = torch.allclose(out_before, out_after, atol=atol, rtol=rtol)

    print(f"  Layer        : {hf_prefix}")
    print(f"  Input shape  : {list(x.shape)}  (seed={snap['seed']})")
    print(f"  Output shape : {list(out_before.shape)}")
    print(f"  max  |diff|  : {max_diff:.6e}")
    print(f"  mean |diff|  : {mean_diff:.6e}")
    print(f"  max rel diff : {rel_max:.6e}")
    print(f"  allclose(atol={atol}, rtol={rtol}): {'PASS' if is_close else 'FAIL'}")

    if not is_close:
        flat_diff = diff.reshape(-1)
        topk_vals, topk_idx = flat_diff.topk(min(5, flat_diff.numel()))
        print("  Top-5 largest diff positions (flat index):")
        for rank, (idx, val) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist()), 1):
            b_val = out_before.reshape(-1)[idx].item()
            a_val = out_after.reshape(-1)[idx].item()
            print(
                f"    #{rank}  idx={idx:6d}  "
                f"before={b_val:+.6f}  after={a_val:+.6f}  diff={val:.6e}"
            )

    print("=" * 70)


# ---------------------------------------------------------------------------
# MLP snapshot / verify
# ---------------------------------------------------------------------------


def _collect_mlp_linears(model: torch.nn.Module, max_layers: int, km: dict = None):
    """Collect MLP container modules for verification, matching patterns
    from ``key_map["mlp_containers"]``.
    """
    km = km or DEFAULT_KEY_MAP
    _container_patterns = [re.compile(p) for p in km["mlp_containers"]]

    seen: set = set()
    results: list = []

    def _add(name: str, mod) -> bool:
        if name not in seen:
            seen.add(name)
            results.append((name, mod))
            print(f"  [Verify] Collected MLP container: {name}")
        return len(results) >= max_layers

    for pat in _container_patterns:
        cnt = 0
        for name, mod in model.named_modules():
            if pat.search(name):
                cnt += 1
                _add(name, mod)
                if cnt >= max_layers:
                    break

    return results


@contextlib.contextmanager
def _force_eager_attn(model: torch.nn.Module):
    """Temporarily switch ``_attn_implementation`` to ``"eager"`` for verification."""
    cfg = getattr(model, "config", None)
    orig = getattr(cfg, "_attn_implementation", None) if cfg is not None else None
    if cfg is not None and orig not in (None, "eager", "flash_attention_2"):
        cfg._attn_implementation = "eager"
    try:
        yield
    finally:
        if cfg is not None and orig is not None:
            cfg._attn_implementation = orig


def _run_module_forward(mod: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward a single module, return first output tensor (float32, cpu)."""
    with torch.no_grad():
        out = mod(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().cpu().float()


def snapshot_mlp_outputs_before(
    model: torch.nn.Module,
    patch_n: int = 3,
    seq_len: int = 8,
    seed: int = 42,
    km: dict = None,
):
    """Collect MLP modules and record their pre-transform forward output."""
    km = km or DEFAULT_KEY_MAP
    mlp_mods = _collect_mlp_linears(model, patch_n, km)
    if not mlp_mods:
        print("[Verify MLP] No MLP modules found, skipping.")
        return []

    rng = torch.Generator()
    rng.manual_seed(seed)
    snap = []
    for name, mod in mlp_mods:
        hidden = None
        if isinstance(mod, torch.nn.Linear):
            hidden = mod.in_features
        else:
            for sub in mod.modules():
                if isinstance(sub, torch.nn.Linear):
                    hidden = sub.in_features
                    break
        if hidden is None:
            print(f"  [Verify MLP] Cannot infer hidden_size for {name}, skipping.")
            continue

        try:
            p = next(mod.parameters())
            dev, dt = p.device, p.dtype
        except StopIteration:
            dev, dt = torch.device("cpu"), torch.float32

        x = torch.randn(1, seq_len, hidden, generator=rng, dtype=dt, device=dev)
        out_before = _run_module_forward(mod, x)
        snap.append({"name": name, "mod": mod, "x": x, "out_before": out_before})
        print(
            f"  [Verify MLP] Recorded pre-transform output: {name}  " f"shape={out_before.shape}"
        )

    return snap


def verify_mlp_output_diff(
    snap,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    """Compare MLP module output before and after transformation."""
    if not snap:
        print("[Verify MLP] No snapshot, skipping.")
        return
    print("\n[Verify MLP] MLP module output comparison (before vs after):")
    all_pass = True
    for entry in snap:
        name = entry["name"]
        mod = entry["mod"]
        x = entry["x"]
        out_before = entry["out_before"]
        out_after = _run_module_forward(mod, x)
        diff = (out_after - out_before).abs()
        max_err = diff.max().item()
        mse = (diff**2).mean().sqrt().item()
        ok = max_err <= atol + rtol * out_before.abs().max().item()
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status} {name:60s}  max_abs_err={max_err:.4e}  rmse={mse:.4e}")

    if all_pass:
        print("[Verify MLP] All MLP modules within tolerance. PASS")
    else:
        print("[Verify MLP] Some MLP modules exceed tolerance! FAIL")
