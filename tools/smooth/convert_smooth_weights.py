#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline SmoothQuant weight conversion.

Usage:
    python convert_smooth_weights.py \
        --model-path /path/to/model \
        --smooth-stats /path/to/smooth_stats.json \
        --save-path /path/to/output \
        [--alpha-qk 0.6] [--alpha-vo 0.5] \
        [--use-ema]

Smooth stats JSON format (from run_vllm_smooth.py):
    {
      "<attn_layer>.q":        {"absmax": [...], "ema": [...], "call_count": N},
      "<attn_layer>.k":        {...},
      "<attn_layer>.attn_out": {...},
      ...
    }
  where <attn_layer> is the vLLM Attention layer name,
  e.g. "model.layers.0.self_attn.attn"
"""

import argparse
import contextlib
import inspect
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Offline SmoothQuant weight conversion")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the pretrained model directory (safetensors format)",
    )
    parser.add_argument(
        "--smooth-stats",
        type=str,
        default=None,
        help="Path to smooth stats JSON file. If not set, auto-detected from output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory containing smooth_stats.json and smooth_alpha_search.json "
        "Used for auto-detection when --smooth-stats is not set.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Directory to save the transformed model",
    )
    parser.add_argument(
        "--alpha-qk",
        type=float,
        default=0.6,
        help="Alpha for QK smooth weight formula: smooth = k_absmax^alpha",
    )
    parser.add_argument(
        "--alpha-vo",
        type=float,
        default=0.5,
        help="Alpha for VO smooth: smooth = attn_out^alpha / o_proj_max^(1-alpha)",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA stats instead of absmax for smooth weight computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the model on (default: cpu). Use 'cuda' or 'auto' for GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Dtype to load the model in (default: auto)",
    )
    parser.add_argument(
        "--smooth-qk",
        action="store_true",
        help="Apply QK smooth (q_proj/k_proj weight transformation). "
        "Requires attn.q and attn.k stats in smooth-stats.",
    )
    parser.add_argument(
        "--smooth-vo",
        action="store_true",
        help="Apply VO smooth (v_proj/o_proj weight transformation). "
        "Requires attn_out stats in smooth-stats.",
    )
    parser.add_argument(
        "--smooth-down",
        action="store_true",
        help="Apply down_proj smooth (down_proj/up_proj weight transformation). "
        "Requires down_proj stats in smooth-stats.",
    )
    parser.add_argument(
        "--alpha-down",
        type=float,
        default=0.5,
        help="Alpha for down_proj smooth: "
        "smooth = absmax^alpha / weight_absmax^(1-alpha) (default: 0.5)",
    )
    parser.add_argument(
        "--enable-alpha-search",
        action="store_true",
        help="Enable alpha search result loading. When set, auto-detects "
        "smooth_alpha_search.json from output-dir. Per-key: if key has search "
        "result, use precomputed smooth_weight; otherwise use --alpha-down.",
    )
    parser.add_argument(
        "--alpha-search-results",
        type=str,
        default=None,
        help="Path to smooth_alpha_search.json (output of alpha grid search). "
        "When provided, down_proj smooth uses pre-computed smooth_weight "
        "from search results instead of computing from smooth-stats + fixed alpha.",
    )
    parser.add_argument(
        "--down-workers",
        type=int,
        default=1,
        help="Number of parallel threads for down_proj smooth (default: 128). ",
    )
    parser.add_argument(
        "--save-workers",
        type=int,
        default=16,
        help="Number of parallel threads for writing safetensors shards (default: 16). "
        "Increase for fast NVMe/parallel NFS; each thread writes one shard file independently.",
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=8.0,
        help="Target size (GiB) of each safetensors shard file (default: 8.0).",
    )
    parser.add_argument(
        "--patch-mlp-layers",
        type=int,
        default=3,
        help="Number of MLP Linear layers to patch for logit loss verification",
    )
    parser.add_argument(
        "--verify-seq-len",
        type=int,
        default=8,
        help="Sequence length for smooth verification forward passes (default: 8)",
    )

    # Architecture key map
    parser.add_argument(
        "--arch",
        type=str,
        default="hy_v3",
        choices=list(PREDEFINED_KEY_MAPS.keys()) + ["custom"],
        help="Architecture key map to use (default: hy_v3). "
        "Use 'custom' with --key-map-file for user-defined mapping.",
    )
    parser.add_argument(
        "--key-map-file",
        type=str,
        default=None,
        help="Custom key_map JSON file path (used when --arch=custom).",
    )

    # YAML config support
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="YAML config file path. Values override argparse defaults; "
        "explicit CLI flags still take final precedence.",
    )

    args = parser.parse_args()

    # Lazy-import _yaml_args (located in tools/). Done here instead of at
    # module top so flake8 doesn't trip on a sys.path mutation between
    # imports.
    import sys

    _tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    from _yaml_args import apply_yaml_config

    apply_yaml_config(parser, args)

    # Auto-resolve smooth_stats path from output_dir
    if args.smooth_stats is None and args.output_dir is not None:
        args.smooth_stats = os.path.join(args.output_dir, "smooth_stats.json")

    if args.smooth_stats is None:
        parser.error("Either --smooth-stats or --output-dir must be specified.")

    # Auto-resolve alpha_search_results from output_dir or enable_alpha_search
    if args.alpha_search_results is None and args.enable_alpha_search:
        if args.output_dir is not None:
            candidate = os.path.join(args.output_dir, "smooth_alpha_search.json")
            if os.path.isfile(candidate):
                args.alpha_search_results = candidate

    return args


# ---------------------------------------------------------------------------
# KEY_MAP
# ---------------------------------------------------------------------------

HY_V3_KEY_MAP = {
    # projection 属性名
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
    "down_proj": "down_proj",
    "up_proj": "up_proj",
    # qk_norm 相关
    "qk_norm_flag": "use_qk_norm",
    "q_norm": "q_norm",
    "k_norm": "k_norm",
    # stats key 后缀
    "stat_k": ".k",
    "stat_q": ".q",
    "stat_attn_out": ".attn_out",
    # vLLM -> HF 路径转换
    "attn_strip": ".attn",
    # down_proj stat 匹配模式
    "down_patterns": [
        r"\.shared_experts\.down_proj",
        r"\.shared_mlp\.down_proj",
        r"\.mlp\.down_proj",
        r"\.experts\.\d+\.down_proj",
    ],
    # 验证用 MLP container 匹配 (HF 侧路径; vLLM 名 shared_mlp 在 HF 模型里
    # 不存在, 由下方 stat_path_aliases 在 _resolve_down_target 里翻译.)
    "mlp_containers": [
        r"\.mlp$",
        r"mlp\.shared_experts$",
        r"mlp\.experts\.\d+$",
    ],
    # vLLM stat_key -> HF module name 路径别名 (regex sub).
    # vLLM HuyuanV3 把 shared MLP 命名成 ".shared_mlp", 而 HuggingFace
    # transformers 里这个子模块叫 ".shared_experts". 校准产物里仍是 vLLM
    # 名, _resolve_down_target 必须先把名字翻译过来才能定位到 HF nn.Linear.
    "stat_path_aliases": [
        (r"\.shared_mlp\b", ".shared_experts"),
    ],
    # === Fused-experts MoE (HYV3Experts 风格的 3D nn.Parameter) ===
    # 当 stat_key 形如 "...experts.<i>.down_proj" 但 ".experts.<i>" 在模型中
    # 并不是 nn.Module 子模块时, resolver 会回退到 "...experts" 容器上读取
    # 下列两个 3D Parameter, 并按 expert_idx 做切片级 smooth。
    # forward 约定:
    #     gate, up = F.linear(x, gate_up_proj[i]).chunk(2, dim=-1)
    # 因此:
    #     gate 半段 = gate_up_proj[i, :I, :]   (smooth 时不变)
    #     up   半段 = gate_up_proj[i, I:, :]   (smooth 时除以 smooth_weight)
    "fused_down_attr": "down_proj",
    "fused_gate_up_attr": "gate_up_proj",
}

LLAMA_KEY_MAP = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
    "down_proj": "down_proj",
    "up_proj": "up_proj",
    "qk_norm_flag": "use_qk_norm",
    "q_norm": "q_norm",
    "k_norm": "k_norm",
    "stat_k": ".k",
    "stat_q": ".q",
    "stat_attn_out": ".attn_out",
    "attn_strip": ".attn",
    "down_patterns": [r"\.mlp\.down_proj"],
    "mlp_containers": [r"mlp$"],
}

MIXTRAL_KEY_MAP = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
    "down_proj": "w2",
    "up_proj": "w3",
    "qk_norm_flag": "use_qk_norm",
    "q_norm": "q_norm",
    "k_norm": "k_norm",
    "stat_k": ".k",
    "stat_q": ".q",
    "stat_attn_out": ".attn_out",
    "attn_strip": ".attn",
    "down_patterns": [r"\.block_sparse_moe\.experts\.\d+\.w2"],
    "mlp_containers": [r"block_sparse_moe\.experts\.\d+$"],
}

QWEN3_MOE_KEY_MAP = {
    # projection 属性名 (transformers 内置 Qwen3MoeAttention / Qwen3MoeMLP 命名)
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "o_proj",
    "down_proj": "down_proj",
    "up_proj": "up_proj",
    # qk_norm 相关
    # ----------------------------------------------------------------------
    # Qwen3MoeAttention 的 forward 是
    #     q = self.q_norm(self.q_proj(x).view(*, num_heads, head_dim))
    #     k = self.k_norm(self.k_proj(x).view(*, num_kv_heads, head_dim))
    # 即 q_norm / k_norm 在 head_dim 维做 RMSNorm, 总是存在, 模型也没有
    # 类似 HYV3 `use_qk_norm` 这种条件 flag, config 里也没有 `qk_norm`.
    #
    # apply_qk_smooth 判断 use_qk_norm 的逻辑是
    #     getattr(attn_module, km["qk_norm_flag"], False)
    #     or getattr(model.config, "qk_norm", False)
    # 因此把 qk_norm_flag 指向 attn 模块上一定存在且 truthy 的属性 "q_norm"
    # 自身, 让 getattr 拿到 Qwen3MoeRMSNorm 实例(非 None, 非空), 从而 use_qk_norm
    # 永远为真, 强制把 smooth 融进 q_norm/k_norm 而不是 q_proj/k_proj.
    # 这是必要的: RMSNorm 不是逐元素线性缩放, 把 smooth 放在 q_proj/k_proj 上
    # 与原模型不数学等价.
    # ----------------------------------------------------------------------
    "qk_norm_flag": "q_norm",
    "q_norm": "q_norm",
    "k_norm": "k_norm",
    # stats key 后缀
    "stat_k": ".k",
    "stat_q": ".q",
    "stat_attn_out": ".attn_out",
    # vLLM -> HF 路径转换 (vLLM 把 attention op 包了一层 ".attn")
    "attn_strip": ".attn",
    # down_proj stat 匹配模式
    # ----------------------------------------------------------------------
    # Qwen3 MoE 没有 shared experts. 每层根据 mlp_only_layers /
    # decoder_sparse_step 走 dense 还是 sparse:
    #   dense layer  : layer.mlp == Qwen3MoeMLP  -> stats key 形如
    #                  "...mlp.down_proj", 命中第 1 条 pattern, 按 nn.Linear
    #                  路径处理.
    #   sparse layer : layer.mlp == Qwen3MoeSparseMoeBlock, 其中
    #                  layer.mlp.experts == Qwen3MoeExperts 是融合 3D Parameter
    #                  -> stats key 形如 "...mlp.experts.<i>.down_proj",
    #                  命中第 2 条 pattern, _resolve_down_target 自动 fallback
    #                  到 fused 路径处理.
    # 注意: r"\.mlp\.down_proj" 不会误匹配 ".mlp.experts.5.down_proj"
    # (中间有 ".experts.5." 隔开, 子串不连续).
    # ----------------------------------------------------------------------
    "down_patterns": [
        r"\.mlp\.down_proj",
        r"\.experts\.\d+\.down_proj",
    ],
    # 验证用 MLP container 匹配
    # ----------------------------------------------------------------------
    # r"mlp$" 同时覆盖 dense / sparse 两种 layer.mlp:
    #   - Qwen3MoeMLP            : 有 gate_proj/up_proj/down_proj 的 nn.Linear
    #                              子模块, _collect_mlp_linears 能推断 hidden
    #                              并跑 forward 验证.
    #   - Qwen3MoeSparseMoeBlock : 没有 nn.Linear 子模块(experts 是 Parameter,
    #                              gate 是 Parameter), _collect_mlp_linears
    #                              拿不到 hidden, 会打印一行 "Cannot infer
    #                              hidden_size" 然后跳过. smooth/save 流程不
    #                              受影响, 仅是 sparse 层 MLP 数值验证降级.
    # 第 2 条 r"mlp\.experts\.\d+$" 在 fused 模式下匹配不到任何子模块,
    # 也是 graceful no-op, 留着是为了将来如果有 per-expert nn.ModuleList 实现
    # 也能复用同一个 KEY_MAP.
    # ----------------------------------------------------------------------
    "mlp_containers": [
        r"mlp$",
        r"mlp\.experts\.\d+$",
    ],
    # === Fused-experts MoE (Qwen3MoeExperts) ===
    # 与 HYV3Experts 完全同构:
    #     experts.gate_up_proj   shape [E, 2*I, H]
    #     experts.down_proj      shape [E, H, I]
    # forward 里
    #     gate, up = F.linear(x, gate_up_proj[i]).chunk(2, dim=-1)
    # 故 gate_up_proj[i, :I, :] 是 gate (smooth 时不变),
    #    gate_up_proj[i, I:, :] 是 up   (smooth 时除以 smooth_weight).
    "fused_down_attr": "down_proj",
    "fused_gate_up_attr": "gate_up_proj",
}

PREDEFINED_KEY_MAPS = {
    "hy_v3": HY_V3_KEY_MAP,
    "llama": LLAMA_KEY_MAP,
    "mixtral": MIXTRAL_KEY_MAP,
    "qwen3_moe": QWEN3_MOE_KEY_MAP,
}

DEFAULT_KEY_MAP = HY_V3_KEY_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_smooth_stats(json_path: str, use_ema: bool = False) -> dict:
    """
    Load smooth stats from JSON file and convert lists back to tensors.

    Returns dict keyed by layer-tag string, e.g. "model.layers.0.self_attn.attn.k",
    with values {"scale": Tensor[D], "call_count": int}.
    The "scale" field is taken from "ema" if use_ema else "absmax".
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    stats = {}
    for key, val in raw.items():
        field = "ema" if use_ema else "absmax"
        data = val.get(field)
        if data is None:
            data = val.get("absmax")  # fallback to absmax
        stats[key] = {
            "scale": torch.tensor(data, dtype=torch.float32) if data is not None else None,
            "call_count": val.get("call_count", 0),
        }
    return stats


def attn_key_to_hf_prefix(smooth_key_base: str, km: dict = None) -> str:
    """
    Map vLLM Attention layer name to HuggingFace module prefix.

    vLLM wraps the raw attention op in a sub-module ".attn", so:
      "model.layers.0.self_attn.attn"  ->  "model.layers.0.self_attn"

    If the key does not end with the attn_strip suffix we return it unchanged (fallback).
    """
    km = km or DEFAULT_KEY_MAP
    attn_strip = km["attn_strip"]
    if smooth_key_base.endswith(attn_strip):
        return smooth_key_base[: -len(attn_strip)]
    return smooth_key_base


def get_submodule_safe(model: torch.nn.Module, name: str):
    """Return a nested submodule by dotted name, or None if not found."""
    try:
        return model.get_submodule(name)
    except AttributeError:
        return None


@contextlib.contextmanager
def _maybe_materialize(linear: torch.nn.Module, target_device):
    """
    Context manager that temporarily materializes meta-device weights for
    in-place modification, then offloads them back via accelerate hooks.

    If the linear layer has no accelerate hook or weights are not on meta
    device, this is a no-op and the caller can modify weights directly.

    Usage:
        with _maybe_materialize(proj, device):
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


# ---------------------------------------------------------------------------
# High-precision weight update helpers
# ---------------------------------------------------------------------------


def _inplace_mul_fp32(weight: torch.Tensor, scale: torch.Tensor) -> None:
    """weight *= scale, computed in float32 then cast back to original dtype."""
    orig_dtype = weight.dtype
    weight.data.copy_(weight.data.float().mul_(scale.float()).to(orig_dtype))


def _inplace_div_fp32(weight: torch.Tensor, scale: torch.Tensor) -> None:
    """weight /= scale, computed in float32 then cast back to original dtype."""
    orig_dtype = weight.dtype
    weight.data.copy_(weight.data.float().div_(scale.float()).to(orig_dtype))


# ---------------------------------------------------------------------------
# Output diff verification
# ---------------------------------------------------------------------------


def _find_first_attn_module(model: torch.nn.Module, smooth_stats: dict, km: dict = None):
    """
    Find the first self_attn submodule that exists in both model and smooth_stats.

    Returns:
        (hf_prefix, attn_module) or (None, None)
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
    """
    Run self_attn forward, using inspect to detect accepted parameters.

    hidden_states shape: (1, seq_len, hidden_size)  (batch=1)

    Returns:
        output tensor (1, seq_len, hidden_size), detached on cpu in float32
    """
    seq_len = hidden_states.shape[1]
    device = hidden_states.device
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    cache_pos = torch.arange(seq_len, dtype=torch.long, device=device)

    # Construct position_embeddings (cos, sin) tuple for newer model architectures
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
) -> dict | None:
    """
    Record attn forward output BEFORE smooth transformation for verification.

    Returns:
        dict with keys: hf_prefix, x (input), out_before (output tensor)
        or None if no suitable layer found.
    """
    km = km or DEFAULT_KEY_MAP
    hf_prefix, attn_module = _find_first_attn_module(model, smooth_stats, km)
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
        f"input shape={list(x.shape)}, dtype={model_dtype}, device={model_device}, seed={seed}"
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
    snap: dict | None,
    model: torch.nn.Module,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    km: dict = None,
) -> None:
    """
    Compare attn forward output before and after smooth transformation.
    """
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
    print(f"  allclose(atol={atol}, rtol={rtol}): " f"{'PASS' if is_close else 'FAIL'}")

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
# MLP layer patch + logit-loss verification
# ---------------------------------------------------------------------------


def _collect_mlp_linears(
    model: torch.nn.Module, max_layers: int, km: dict = None
) -> list[tuple[str, torch.nn.Module]]:
    """
    Collect MLP container modules for verification, matching patterns from key_map.
    """
    import re

    km = km or DEFAULT_KEY_MAP
    _container_patterns = [re.compile(p) for p in km["mlp_containers"]]

    seen: set[str] = set()
    results: list[tuple[str, torch.nn.Module]] = []

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
    """Temporarily switch _attn_implementation to eager for verification."""
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
) -> list[dict]:
    """
    Collect MLP modules and record their pre-transform forward output.
    """
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
        print(f"  [Verify MLP] Recorded pre-transform output: {name}  shape={out_before.shape}")

    return snap


def verify_mlp_output_diff(
    snap: list[dict],
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


# ---------------------------------------------------------------------------
# QK smooth
# ---------------------------------------------------------------------------


def apply_qk_smooth(
    model: torch.nn.Module,
    smooth_stats: dict,
    alpha: float = 0.6,
    head_dim: int = None,
    km: dict = None,
) -> None:
    """
    Apply QK smooth to every layer that has both attn.k and attn.q stats.

    Smooth weight formula (RoPE-aware pairing):
        k_absmax reshape -> [num_kv_heads, head_dim]
        paired[h, d] = max(k_absmax[h, d], k_absmax[h, d + head_dim/2])
        smooth_weight[h, d] = smooth_weight[h, d + head_dim/2] = paired[h, d] ^ alpha

    GQA handling: expand smooth_weight from kv_heads to q_heads via repeat_interleave.

    qk_norm path: fuse smooth into q_norm/k_norm weights instead of proj weights.
    """
    km = km or DEFAULT_KEY_MAP
    stat_k = km["stat_k"]
    stat_q = km["stat_q"]

    k_keys = {k for k in smooth_stats if k.endswith(stat_k)}
    q_keys = {k for k in smooth_stats if k.endswith(stat_q)}
    k_bases = {k[: -len(stat_k)] for k in k_keys}
    q_bases = {k[: -len(stat_q)] for k in q_keys}
    valid_bases = k_bases & q_bases

    print(f"\n[QK Smooth] {len(valid_bases)} layers eligible (have both attn.k and attn.q stats)")

    for base in sorted(valid_bases):
        k_stat_key = base + stat_k
        hf_prefix = attn_key_to_hf_prefix(base, km)

        attn_module = get_submodule_safe(model, hf_prefix)
        if attn_module is None:
            print(f"  [SKIP] {hf_prefix}: module not found in model")
            continue

        use_qk_norm = getattr(attn_module, km["qk_norm_flag"], False) or getattr(
            model.config, "qk_norm", False
        )
        q_norm = getattr(attn_module, km["q_norm"], None) if use_qk_norm else None
        k_norm = getattr(attn_module, km["k_norm"], None) if use_qk_norm else None
        if use_qk_norm and (q_norm is None or k_norm is None):
            print(f"  [SKIP] {hf_prefix}: use_qk_norm=True but q_norm/k_norm not found")
            continue

        q_proj = getattr(attn_module, km["q_proj"], None)
        k_proj = getattr(attn_module, km["k_proj"], None)
        if q_proj is None or k_proj is None:
            print(f"  [SKIP] {hf_prefix}: q_proj or k_proj not found")
            continue

        k_scale = smooth_stats[k_stat_key]["scale"]
        if k_scale is None:
            print(f"  [SKIP] {hf_prefix}: k scale is None")
            continue

        k_absmax = k_scale.to(torch.float32)
        D = k_absmax.shape[0]

        if head_dim is None:
            raise ValueError(f"head_dim must be provided for QK smooth (layer {hf_prefix})")

        num_kv_heads_loc = D // head_dim
        half_hd = head_dim // 2

        k_per_head = k_absmax.reshape(num_kv_heads_loc, head_dim)
        paired_hd = torch.maximum(k_per_head[:, :half_hd], k_per_head[:, half_hd:]).clamp(min=1e-8)
        smooth_per_head = torch.empty(num_kv_heads_loc, head_dim, dtype=torch.float32)
        smooth_per_head[:, :half_hd] = paired_hd
        smooth_per_head[:, half_hd:] = paired_hd
        smooth_weight = smooth_per_head.pow(alpha)

        q_out_dim = q_proj.weight.shape[0]
        k_out_dim = k_proj.weight.shape[0]
        is_gqa = q_out_dim != k_out_dim
        device = q_proj.weight.device if not q_proj.weight.is_meta else torch.device("cpu")

        if use_qk_norm:
            per_dim_max = smooth_weight.max(dim=0).values
            norm_sw = torch.maximum(per_dim_max[:half_hd], per_dim_max[half_hd:])
            norm_smooth_weight = torch.empty(head_dim, dtype=torch.float32)
            norm_smooth_weight[:half_hd] = norm_sw
            norm_smooth_weight[half_hd:] = norm_sw

            norm_smooth_weight = norm_smooth_weight.to(device=device)

            with _maybe_materialize(q_norm, device):
                _inplace_mul_fp32(q_norm.weight, norm_smooth_weight)
            with _maybe_materialize(k_norm, device):
                _inplace_div_fp32(k_norm.weight, norm_smooth_weight)

            print(
                f"  [QK-norm] {hf_prefix}: q_out={q_out_dim}, k_out={k_out_dim}, "
                f"norm_smooth range=["
                f"{norm_smooth_weight.float().min():.4f}, "
                f"{norm_smooth_weight.float().max():.4f}], "
                f"gqa={'yes (n_groups=' + str(q_out_dim // k_out_dim) + ')' if is_gqa else 'no'}"
            )

        else:
            smooth_weight_flat = smooth_weight.reshape(D)

            if is_gqa:
                num_kv_heads = k_out_dim // head_dim
                n_groups = q_out_dim // k_out_dim
                sw_kv = smooth_weight_flat.reshape(num_kv_heads, head_dim)
                sw_q = sw_kv.repeat_interleave(n_groups, dim=0)
                q_smooth_weight = sw_q.reshape(q_out_dim)
                smooth_k_weight = smooth_weight_flat
            else:
                q_smooth_weight = smooth_weight_flat
                smooth_k_weight = smooth_weight_flat

            q_smooth_weight = q_smooth_weight.to(device=device)
            smooth_k_weight = smooth_k_weight.to(device=device)

            with _maybe_materialize(q_proj, device):
                _inplace_mul_fp32(q_proj.weight, q_smooth_weight.view(-1, 1))
                if q_proj.bias is not None:
                    _inplace_mul_fp32(q_proj.bias, q_smooth_weight)

            with _maybe_materialize(k_proj, device):
                _inplace_div_fp32(k_proj.weight, smooth_k_weight.view(-1, 1))
                if k_proj.bias is not None:
                    _inplace_div_fp32(k_proj.bias, smooth_k_weight)

            print(
                f"  [QK] {hf_prefix}: q_out={q_out_dim}, k_out={k_out_dim}, "
                f"gqa={'yes (n_groups=' + str(q_out_dim // k_out_dim) + ')' if is_gqa else 'no'}"
            )


# ---------------------------------------------------------------------------
# VO smooth
# ---------------------------------------------------------------------------


def apply_vo_smooth(
    model: torch.nn.Module,
    smooth_stats: dict,
    alpha: float = 0.5,
    head_dim: int = None,
    km: dict = None,
) -> None:
    """
    Apply VO smooth to every layer that has attn_out stats.

    Smooth weight formula:
        smooth_weight[i] = attn_out_absmax[i]^alpha / max|o_proj.weight[:,i]|^(1-alpha)

    GQA: max-pool smooth_weight across groups for v_proj.
    """
    km = km or DEFAULT_KEY_MAP
    stat_attn_out = km["stat_attn_out"]

    attn_out_keys = [k for k in smooth_stats if k.endswith(stat_attn_out)]
    print(f"\n[VO Smooth] {len(attn_out_keys)} layers with attn_out stats")

    for attn_out_key in sorted(attn_out_keys):
        base = attn_out_key[: -len(stat_attn_out)]
        hf_prefix = attn_key_to_hf_prefix(base, km)

        attn_module = get_submodule_safe(model, hf_prefix)
        if attn_module is None:
            print(f"  [SKIP] {hf_prefix}: module not found in model")
            continue

        v_proj = getattr(attn_module, km["v_proj"], None)
        o_proj = getattr(attn_module, km["o_proj"], None)
        if v_proj is None or o_proj is None:
            print(f"  [SKIP] {hf_prefix}: v_proj or o_proj not found")
            continue

        attn_out_scale = smooth_stats[attn_out_key]["scale"]
        if attn_out_scale is None:
            print(f"  [SKIP] {hf_prefix}: attn_out scale is None")
            continue

        attn_out_absmax = attn_out_scale.to(torch.float32).clamp(min=1e-8)
        attn_out_dim = attn_out_absmax.shape[0]

        o_hook = getattr(o_proj, "_hf_hook", None)
        if o_hook is not None and o_proj.weight.device.type == "meta":
            _tmp_device = torch.device("cpu")
            _orig_exec = o_hook.execution_device
            o_hook.execution_device = _tmp_device
            o_hook.pre_forward(o_proj)
            o_proj_max = (o_proj.weight.detach().float().abs().max(dim=0).values).clamp(min=1e-8)
            o_hook.post_forward(o_proj, None)
            o_hook.execution_device = _orig_exec
        else:
            o_proj_max = (o_proj.weight.detach().float().abs().max(dim=0).values).clamp(min=1e-8)

        smooth_weight = attn_out_absmax.pow(alpha) / o_proj_max.pow(1.0 - alpha)

        v_out_dim = v_proj.weight.shape[0]
        is_gqa = v_out_dim != attn_out_dim

        if is_gqa:
            assert (
                attn_out_dim % v_out_dim == 0
            ), f"{hf_prefix}: attn_out_dim={attn_out_dim} not divisible by v_out_dim={v_out_dim}"
            n_groups = attn_out_dim // v_out_dim

            if head_dim is None:
                raise ValueError(
                    f"head_dim must be provided for GQA VO smooth (layer {hf_prefix})"
                )
            num_kv_heads = v_out_dim // head_dim

            sw_grouped = smooth_weight.reshape(num_kv_heads, n_groups, head_dim)
            pooled = sw_grouped.max(dim=1).values
            smooth_v_weight = pooled.reshape(v_out_dim)

            smooth_weight = (
                pooled.unsqueeze(1).expand(num_kv_heads, n_groups, head_dim).reshape(attn_out_dim)
            )
        else:
            smooth_v_weight = smooth_weight

        device = v_proj.weight.device if not v_proj.weight.is_meta else torch.device("cpu")
        smooth_v_weight = smooth_v_weight.to(device=device)
        smooth_weight = smooth_weight.to(device=device)

        with _maybe_materialize(v_proj, device):
            _inplace_div_fp32(v_proj.weight, smooth_v_weight.view(-1, 1))
            if v_proj.bias is not None:
                _inplace_div_fp32(v_proj.bias, smooth_v_weight)

        with _maybe_materialize(o_proj, device):
            _inplace_mul_fp32(o_proj.weight, smooth_weight.view(1, -1))

        print(
            f"  [VO] {hf_prefix}: v_out={v_out_dim}, attn_out={attn_out_dim}, "
            f"gqa={'yes (n_groups=' + str(attn_out_dim // v_out_dim) + ')' if is_gqa else 'no'}"
        )


# ---------------------------------------------------------------------------
# Fused-experts MoE smooth helpers
# ---------------------------------------------------------------------------


def _apply_smooth_fused_expert(
    experts_module: torch.nn.Module,
    expert_idx: int,
    smooth_weight: torch.Tensor,
    fused_down_attr: str = "down_proj",
    fused_gate_up_attr: str = "gate_up_proj",
) -> None:
    """
    Apply down_proj smooth to a single expert in HYV3Experts-style fused MoE.

    Semantics match the per-expert nn.Linear case:
        old_down_proj.weight  *=  smooth_weight   (input channel,  dim=1)
        old_up_proj.weight    /=  smooth_weight   (output channel, dim=0)

    Layout for the fused 3D-tensor MoE (HYV3Experts):
        experts.down_proj          shape [E, H, I]
        experts.gate_up_proj       shape [E, 2*I, H]   (gate then up)
            gate_part = gate_up_proj[i, :I, :]   (untouched)
            up_part   = gate_up_proj[i, I:, :]   (divided by smooth_weight)
        because forward unpacks
            gate, up = F.linear(x, gate_up_proj[i]).chunk(2, dim=-1)

    All math is done in fp32 then cast back to each tensor's original dtype.

    Note (parallel safety): different ``expert_idx`` values touch disjoint
    storage regions of the same Parameter, so concurrent invocation from
    multiple threads is safe as long as the surrounding code does not also
    rely on accelerate hooks (which is already serialized upstream).
    """
    down_param = getattr(experts_module, fused_down_attr)
    gate_up_param = getattr(experts_module, fused_gate_up_attr)

    intermediate_dim = down_param.shape[2]
    if smooth_weight.shape != (intermediate_dim,):
        raise ValueError(
            f"smooth_weight shape {tuple(smooth_weight.shape)} != " f"({intermediate_dim},)"
        )
    if gate_up_param.shape[1] != 2 * intermediate_dim:
        raise ValueError(
            f"gate_up_proj rows ({gate_up_param.shape[1]}) != 2*intermediate_dim "
            f"({2 * intermediate_dim}); fused gate-then-up layout assumption "
            f"violated."
        )

    # Pick a non-meta target device if available
    target_device = None
    for p in (down_param, gate_up_param):
        if not p.is_meta:
            target_device = p.device
            break
    if target_device is None:
        target_device = torch.device("cpu")

    # Materialize the whole experts module via accelerate hook if either
    # fused param is on meta. We use `experts_module` (not a Linear) since
    # the params live directly on it.
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
        gu_up_slice = gate_up_param.data[expert_idx, intermediate_dim:, :]  # view [I, H]
        gu_up_slice.copy_(gu_up_slice.float().div_(sw.view(-1, 1)).to(gu_orig_dtype))

        absmax_after = down_param.data[expert_idx].abs().max().item()
    finally:
        if use_hook:
            hook.post_forward(experts_module, None)
            hook.execution_device = original_exec_device
    return absmax_before, absmax_after


def _resolve_down_target(
    model: torch.nn.Module,
    stat_key: str,
    all_named_modules: dict,
    up_proj_map: dict,
    km: dict,
):
    """
    Resolve a smooth-stats key to a target for down_proj smooth.

    Returns one of:
        ("linear", down_proj_module, up_proj_module,
                   down_proj_full_name, up_proj_full_name)
            -- per-expert nn.Linear / dense MLP / shared MLP path.

        ("fused",  experts_module, expert_idx, parent_path,
                   fused_down_attr, fused_gate_up_attr)
            -- fused 3D-tensor MoE path (HYV3Experts-style).

        ("none",   reason_str)
            -- could not resolve, caller should skip with reason.

    Strategy: try the nn.Linear path first (covers dense MLP and legacy
    per-expert MoE). Fall back to fused-experts pattern matching only when
    no Linear submodule with a 2D weight can be located -- this preserves
    the original behavior for legacy checkpoints.
    """
    import re as _re

    down_name = km["down_proj"]

    # Build vLLM->HF candidate keys via stat_path_aliases (e.g. HuyuanV3
    # vLLM names shared MLP ".shared_mlp" while HF names it ".shared_experts").
    # The original stat_key always stays at index 0 so behaviour is unchanged
    # when no alias matches or when the key map declares no aliases.
    aliases = km.get("stat_path_aliases", [])
    candidate_keys = [stat_key]
    for _pat, _dst in aliases:
        translated = _re.sub(_pat, _dst, stat_key)
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
        # suffix match: stat_key may be missing/extra prefix relative to model
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
            f"{stat_key}: cannot resolve to nn.Linear; fused experts "
            f"not configured in key_map (missing fused_down_attr / "
            f"fused_gate_up_attr)",
        )

    fused_re = _re.compile(
        r"^(?P<prefix>.*\.experts)\.(?P<idx>\d+)\." + _re.escape(down_name) + r"$"
    )

    m = None
    for cand in candidate_keys:
        m = fused_re.match(cand)
        if m is not None:
            break
    if m is None:
        # rebuild via suffix match against named modules and re-match
        for name in all_named_modules:
            if any(name.endswith(c) or c.endswith(name) for c in candidate_keys):
                m = fused_re.match(name)
                if m:
                    break

    if m is None:
        return (
            "none",
            f"{stat_key}: no nn.Linear and pattern " f".experts.<i>.{down_name} does not match",
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
            f"{parent_path}.{fused_gate_up_attr} is not a 3D "
            f"nn.Parameter (got {type(gate_up_param).__name__})",
        )

    if expert_idx < 0 or expert_idx >= down_param.shape[0]:
        return (
            "none",
            f"{parent_path}: expert_idx={expert_idx} out of range " f"[0, {down_param.shape[0]})",
        )

    return ("fused", experts_module, expert_idx, parent_path, fused_down_attr, fused_gate_up_attr)


# ---------------------------------------------------------------------------
# down_proj smooth
# ---------------------------------------------------------------------------


def apply_down_proj_smooth(
    model: torch.nn.Module,
    smooth_stats: dict,
    alpha: float = 0.5,
    num_workers: int = 8,
    exclude_keys: set | None = None,
    km: dict = None,
) -> None:
    """
    Apply down_proj smooth to every layer that has down_proj stats.

    Key matching: stats keys that match patterns from key_map["down_patterns"].

    Smooth weight formula:
        smooth_weight = absmax^alpha / weight_absmax^(1-alpha)

    Application:
        down_proj.weight *= smooth_weight   (per input channel, dim=1)
        up_proj.weight   /= smooth_weight   (per output channel, dim=0)

    Parallelism:
        Uses ThreadPoolExecutor. Falls back to serial if accelerate hooks detected.
    """
    import re as _re

    km = km or DEFAULT_KEY_MAP
    up_name = km["up_proj"]

    _DOWN_PATTERNS = [_re.compile(p) for p in km["down_patterns"]]

    down_keys = [k for k in smooth_stats if any(pat.search(k) for pat in _DOWN_PATTERNS)]

    # Filter out keys already handled by alpha search
    if exclude_keys:
        before = len(down_keys)
        down_keys = [k for k in down_keys if k not in exclude_keys]
        skipped = before - len(down_keys)
        if skipped > 0:
            print(f"  [Down Smooth] Skipped {skipped} keys already handled by alpha search")

    print(
        f"\n[Down Smooth] {len(down_keys)} down_proj stat entries to process (fixed alpha={alpha})"
    )

    if not down_keys:
        print("  [SKIP] No down_proj stats found in smooth-stats")
        return

    # Build named_modules dict and O(1) up_proj lookup table
    all_named_modules: dict = {name: mod for name, mod in model.named_modules()}

    up_proj_map: dict = {}
    for name in all_named_modules:
        if name.endswith("." + up_name):
            parent = name[: -(len(up_name) + 1)]
            up_proj_map[parent] = name

    # Detect accelerate hooks -> fall back to serial for thread safety
    has_hf_hook = any(
        getattr(mod, "_hf_hook", None) is not None for mod in all_named_modules.values()
    )
    effective_workers = 1 if has_hf_hook else num_workers
    if has_hf_hook and num_workers > 1:
        print("  [Parallel] accelerate _hf_hook detected -> fallback to serial (num_workers=1)")
    else:
        print(f"  [Parallel] num_workers={effective_workers}")

    def _process_one(stat_key: str) -> None:
        """Process a single down_proj stat_key (thread-safe, independent per key)."""
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

        # Validate: skip layer if contains -inf, nan, or <=0 values
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

            # Handle meta device for weight_absmax computation
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

            with _maybe_materialize(down_proj_module, device):
                _inplace_mul_fp32(down_proj_module.weight, smooth_weight.view(1, -1))

            up_device = (
                up_proj_module.weight.device
                if not up_proj_module.weight.is_meta
                else torch.device("cpu")
            )
            smooth_weight_up = smooth_weight.to(device=up_device)

            with _maybe_materialize(up_proj_module, up_device):
                _inplace_div_fp32(up_proj_module.weight, smooth_weight_up.view(-1, 1))
                if up_proj_module.bias is not None:
                    _inplace_div_fp32(up_proj_module.bias, smooth_weight_up)

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

            # weight_absmax for the i-th expert: max-abs over the H (output) dim
            # of down_param[expert_idx], yielding shape [I].
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

    # Execute: serial or parallel
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


# ---------------------------------------------------------------------------
# down_proj smooth from alpha search results (pre-computed smooth_weight)
# ---------------------------------------------------------------------------


def apply_down_proj_smooth_from_search(
    model: torch.nn.Module,
    alpha_search_results: dict,
    num_workers: int = 8,
    km: dict = None,
) -> set:
    """
    Apply pre-computed smooth_weight from alpha search to down_proj/up_proj.

    Unlike apply_down_proj_smooth(), this function does NOT compute
    smooth_weight from absmax -- it reads the pre-computed smooth_weight
    directly from alpha_search_results (output of SmoothAlphaSearcher).

    alpha_search_results format:
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

    Application (same as apply_down_proj_smooth):
        down_proj.weight *= smooth_weight   (dim=1, input channel)
        up_proj.weight   /= smooth_weight   (dim=0, output channel)
        up_proj.bias     /= smooth_weight   (if exists)

    Returns:
        set of stat_keys that were successfully processed.
    """
    km = km or DEFAULT_KEY_MAP
    up_name = km["up_proj"]

    results = alpha_search_results.get("results", alpha_search_results)
    if not results:
        print("[Down Smooth Search] No results found in alpha_search_results")
        return set()

    print(f"\n[Down Smooth Search] {len(results)} entries from alpha search")

    all_named_modules: dict = {name: mod for name, mod in model.named_modules()}
    up_proj_map: dict = {}
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

    def _process_one(stat_key: str, entry: dict) -> str | None:
        """Process one key. Returns stat_key on success, None on skip."""
        smooth_weight_list = entry.get("smooth_weight")
        if smooth_weight_list is None:
            print(f"  [SKIP] {stat_key}: no smooth_weight in search results")
            return None

        smooth_weight = torch.tensor(smooth_weight_list, dtype=torch.float32)

        # Validate
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

            # Apply smooth_weight
            device = (
                down_proj_module.weight.device
                if not down_proj_module.weight.is_meta
                else torch.device("cpu")
            )
            smooth_weight_dev = smooth_weight.to(device=device)

            # down_proj.weight *= smooth_weight  (input channel, dim=1)
            with _maybe_materialize(down_proj_module, device):
                absmax_down_before = down_proj_module.weight.abs().max().item()
                _inplace_mul_fp32(down_proj_module.weight, smooth_weight_dev.view(1, -1))
                absmax_down_after = down_proj_module.weight.abs().max().item()

            # up_proj.weight /= smooth_weight  (output channel, dim=0)
            up_device = (
                up_proj_module.weight.device
                if not up_proj_module.weight.is_meta
                else torch.device("cpu")
            )
            smooth_weight_up = smooth_weight.to(device=up_device)

            with _maybe_materialize(up_proj_module, up_device):
                _inplace_div_fp32(up_proj_module.weight, smooth_weight_up.view(-1, 1))
                if up_proj_module.bias is not None:
                    _inplace_div_fp32(up_proj_module.bias, smooth_weight_up)

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

    # Execute and collect successfully processed keys
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

    print(f"  [Down-Search] Successfully processed {len(processed_keys)} / {len(results)} entries")
    return processed_keys


# ---------------------------------------------------------------------------
# MTP / extra weight recovery: load tensors dropped by AutoModelForCausalLM
# ---------------------------------------------------------------------------


def load_missing_tensors(model_path: str, loaded_keys: set) -> dict:
    """
    Load weights that were NOT loaded by AutoModelForCausalLM (e.g. MTP layers).

    Typical case: models with Multi-Token Prediction heads whose extra layers
    are not registered in the HF architecture and get dropped with
    "Some weights of the model checkpoint were not used" warning.

    Args:
        model_path:   original model directory path
        loaded_keys:  model.state_dict().keys() (keys already loaded by HF)

    Returns:
        dict[str, torch.Tensor] -- missing key -> CPU tensor mapping;
        empty dict if nothing is missing.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        print("  [WARNING] safetensors not available, cannot supplement missing weights")
        return {}

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
    elif os.path.exists(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as f_st:
            weight_map = {k: "model.safetensors" for k in f_st.keys()}
    else:
        print("  [WARNING] No safetensors index/file found in model_path, skipping")
        return {}

    missing_keys = set(weight_map.keys()) - loaded_keys
    if not missing_keys:
        print("  [MTP] No missing keys, model parameters are complete.")
        return {}

    print(
        f"  [MTP] Found {len(missing_keys)} weight keys not loaded by HF "
        f"(e.g. MTP layers), supplementing:"
    )
    for k in sorted(missing_keys)[:8]:
        print(f"    {k}")
    if len(missing_keys) > 8:
        print(f"    ... total {len(missing_keys)}")

    # Group by shard file, open each shard only once
    shard_to_keys: dict[str, list[str]] = {}
    for k in missing_keys:
        shard_to_keys.setdefault(weight_map[k], []).append(k)

    missing_tensors: dict[str, torch.Tensor] = {}
    for shard_file, keys in sorted(shard_to_keys.items()):
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f_st:
            for k in keys:
                t = f_st.get_tensor(k)
                if not t.is_contiguous():
                    t = t.contiguous()
                missing_tensors[k] = t
        print(f"    Loaded {len(keys)} missing keys from {shard_file}")

    return missing_tensors


# ---------------------------------------------------------------------------
# Parallel safetensors save
# ---------------------------------------------------------------------------


def save_model_parallel(
    model: torch.nn.Module,
    save_path: str,
    shard_size_gb: float = 4.0,
    num_workers: int = 4,
    extra_state_dict: dict | None = None,
) -> None:
    """
    Save model weights as safetensors with parallel shard writing.

    Much faster than save_pretrained (which serializes sequentially).

    Flow:
      1. Collect state_dict, split into shards by shard_size_gb
      2. ThreadPoolExecutor writes shards concurrently
      3. Generate model.safetensors.index.json
      4. Single shard -> write model.safetensors directly (no index)

    Args:
        model:             transformed model
        save_path:         output directory (must exist)
        shard_size_gb:     target size per shard (GiB)
        num_workers:       concurrent writer threads
        extra_state_dict:  extra weights to merge (e.g. MTP layers from
                           load_missing_tensors())
    """
    try:
        from safetensors.torch import save_file as st_save_file
    except ImportError:
        raise ImportError(
            "safetensors is required for parallel saving. " "Install with: pip install safetensors"
        )

    shard_size_bytes = int(shard_size_gb * 1024**3)

    # 1. Collect state_dict (all contiguous cpu tensors)
    print("  [Save] Collecting state_dict ...")
    raw_state_dict = model.state_dict()
    state_dict: dict[str, torch.Tensor] = {}
    for name, t in raw_state_dict.items():
        if t.device.type != "cpu":
            t = t.cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        state_dict[name] = t

    # Merge extra_state_dict (e.g. MTP layers)
    if extra_state_dict:
        n_extra = 0
        for name, t in extra_state_dict.items():
            if name in state_dict:
                print(f"  [Save][WARNING] key {name!r} already in state_dict, extra ignored")
                continue
            if t.device.type != "cpu":
                t = t.cpu()
            if not t.is_contiguous():
                t = t.contiguous()
            state_dict[name] = t
            n_extra += 1
        print(f"  [Save] Supplemented {n_extra} extra weights (MTP etc.) into state_dict")

    total_bytes = sum(t.nbytes for t in state_dict.values())
    total_gb = total_bytes / 1024**3
    n_shards = max(1, math.ceil(total_bytes / shard_size_bytes))
    print(
        f"  [Save] Total params: {total_gb:.2f} GiB, splitting into {n_shards} shards "
        f"(each <= {shard_size_gb} GiB)"
    )

    # 2. Assign tensors to shards (greedy, no cross-shard splitting)
    shard_dicts: list[dict[str, torch.Tensor]] = []
    cur_shard: dict[str, torch.Tensor] = {}
    cur_bytes = 0

    for name, tensor in state_dict.items():
        nb = tensor.nbytes
        if cur_bytes + nb > shard_size_bytes and cur_shard:
            shard_dicts.append(cur_shard)
            cur_shard = {}
            cur_bytes = 0
        cur_shard[name] = tensor
        cur_bytes += nb

    if cur_shard:
        shard_dicts.append(cur_shard)

    n_shards = len(shard_dicts)

    # 3. Write shard files concurrently
    if n_shards == 1:
        out_file = os.path.join(save_path, "model.safetensors")
        print(f"  [Save] Single shard -> {out_file}")
        st_save_file(shard_dicts[0], out_file)
        print(f"  [Save] Done: {out_file}")
        return

    pad = len(str(n_shards))
    shard_filenames = [
        f"model-{str(i + 1).zfill(pad)}-of-{str(n_shards).zfill(pad)}.safetensors"
        for i in range(n_shards)
    ]

    def _write_shard(idx: int) -> tuple[int, str]:
        fname = shard_filenames[idx]
        out_file = os.path.join(save_path, fname)
        shard_bytes = sum(t.nbytes for t in shard_dicts[idx].values())
        print(
            f"  [Save] [{idx+1}/{n_shards}] Writing {fname}  "
            f"({shard_bytes / 1024**3:.2f} GiB, "
            f"{len(shard_dicts[idx])} tensors) ..."
        )
        st_save_file(shard_dicts[idx], out_file)
        return idx, fname

    effective_workers = min(num_workers, n_shards)
    print(f"  [Save] Writing {n_shards} shards concurrently, workers={effective_workers}")

    results_list: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(_write_shard, i): i for i in range(n_shards)}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                raise RuntimeError(f"Shard {futures[fut]} write failed: {exc}") from exc
            results_list.append(fut.result())

    results_list.sort()

    # 4. Generate model.safetensors.index.json
    weight_map: dict[str, str] = {}
    for idx, fname in results_list:
        for key in shard_dicts[idx]:
            weight_map[key] = fname

    index = {
        "metadata": {"total_size": str(total_bytes)},
        "weight_map": weight_map,
    }
    index_path = os.path.join(save_path, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"  [Save] Index written: {index_path}")
    print(f"  [Save] All {n_shards} shards written successfully")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    print("=" * 70)
    print("[Config]")
    print(f"  model-path:      {args.model_path}")
    print(f"  smooth-stats:    {args.smooth_stats}")
    print(f"  save-path:       {args.save_path}")
    print(f"  alpha-qk:        {args.alpha_qk}")
    print(f"  alpha-vo:        {args.alpha_vo}")
    print(f"  use-ema:         {args.use_ema}")
    print(f"  device:          {args.device}")
    print(f"  dtype:           {args.dtype}")
    print(f"  smooth-qk:       {args.smooth_qk}")
    print(f"  smooth-vo:       {args.smooth_vo}")
    print(f"  smooth-down:     {args.smooth_down}")
    print(f"  alpha-down:      {args.alpha_down}")
    print(f"  down-workers:    {args.down_workers}")
    print(f"  save-workers:    {args.save_workers}")
    print(f"  shard-size-gb:   {args.shard_size_gb}")
    print(f"  patch-mlp-layers:{args.patch_mlp_layers}")
    print(f"  verify-seq-len:  {args.verify_seq_len}")
    print(f"  alpha-search:    {args.alpha_search_results or '(none, use fixed alpha)'}")
    print("=" * 70)

    # Resolve key_map
    if args.arch == "custom" and args.key_map_file:
        with open(args.key_map_file, "r") as f:
            km = json.load(f)
        print(f"  [KeyMap] Loaded custom key_map from {args.key_map_file}")
    else:
        km = PREDEFINED_KEY_MAPS[args.arch]
        print(f"  [KeyMap] Using predefined key_map: {args.arch}")

    if not args.smooth_qk and not args.smooth_vo and not args.smooth_down:
        print(
            "\n[WARNING] Neither --smooth-qk, --smooth-vo, nor --smooth-down is set. "
            "No weight transformation will be applied. "
            "Use --smooth-qk / --smooth-vo / --smooth-down to enable conversion."
        )
        return

    # ------------------------------------------------------------------
    # Step 1: Load model
    # ------------------------------------------------------------------
    print(f"\n[Step 1] Loading model from {args.model_path} ...")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(args.dtype, "auto")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Model loaded: {type(model).__name__}")

    # Load tokenizer (optional, for saving alongside model)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        print(f"  Tokenizer loaded: {type(tokenizer).__name__}")
    except Exception as e:
        print(f"  [WARNING] Could not load tokenizer: {e}")

    # Infer head_dim from model config
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        num_heads = getattr(cfg, "num_attention_heads", None)
        hidden_size = getattr(cfg, "hidden_size", None)
        if num_heads and hidden_size:
            head_dim = hidden_size // num_heads
    print(f"  head_dim inferred: {head_dim}")

    # ------------------------------------------------------------------
    # Step 2: Load smooth stats
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Loading smooth stats from {args.smooth_stats} ...")
    smooth_stats = load_smooth_stats(args.smooth_stats, use_ema=args.use_ema)
    print(f"  Loaded {len(smooth_stats)} stat entries")
    sample_keys = list(smooth_stats.keys())[:6]
    for k in sample_keys:
        sc = smooth_stats[k]["scale"]
        print(f"    {k!r:70s}  shape={list(sc.shape) if sc is not None else None}")
    if len(smooth_stats) > 6:
        print(f"    ... and {len(smooth_stats) - 6} more")

    # ------------------------------------------------------------------
    # Step 2.5: Record pre-transform attn output (for verification)
    # ------------------------------------------------------------------
    verify_snap = snapshot_attn_output_before(
        model, smooth_stats, seq_len=args.verify_seq_len, seed=42, km=km
    )

    # ------------------------------------------------------------------
    # Step 2.6: Record pre-transform MLP module output (for verification)
    # ------------------------------------------------------------------
    mlp_snap = snapshot_mlp_outputs_before(
        model,
        patch_n=args.patch_mlp_layers,
        seq_len=args.verify_seq_len,
        seed=42,
        km=km,
    )

    # ------------------------------------------------------------------
    # Step 3: Apply smooth transforms
    # ------------------------------------------------------------------
    print("\n[Step 3] Applying offline smooth transforms ...")

    # QK smooth
    if args.smooth_qk:
        has_qk = any(k.endswith(km["stat_k"]) for k in smooth_stats) and any(
            k.endswith(km["stat_q"]) for k in smooth_stats
        )
        if has_qk:
            apply_qk_smooth(model, smooth_stats, alpha=args.alpha_qk, head_dim=head_dim, km=km)
        else:
            print("  [SKIP] QK smooth: no attn.k / attn.q stats found in smooth-stats")
    else:
        print("  [SKIP] QK smooth: --smooth-qk not set")

    # VO smooth
    if args.smooth_vo:
        has_vo = any(k.endswith(km["stat_attn_out"]) for k in smooth_stats)
        if has_vo:
            apply_vo_smooth(model, smooth_stats, alpha=args.alpha_vo, head_dim=head_dim, km=km)
        else:
            print("  [SKIP] VO smooth: no attn_out stats found in smooth-stats")
    else:
        print("  [SKIP] VO smooth: --smooth-vo not set")

    # Down proj smooth (two-step: alpha search first, then fixed alpha fallback)
    if args.smooth_down:
        search_processed_keys = set()

        # Step 1: If alpha search results provided, apply them first
        if args.alpha_search_results:
            print(f"\n[Down Smooth] Using alpha search results: {args.alpha_search_results}")
            with open(args.alpha_search_results, "r") as f:
                alpha_search = json.load(f)
            search_processed_keys = apply_down_proj_smooth_from_search(
                model, alpha_search, num_workers=args.down_workers, km=km
            )

        # Step 2: Fallback -- for keys NOT in alpha search, use fixed alpha
        import re as _re

        _DOWN_PATTERNS = [_re.compile(p) for p in km["down_patterns"]]
        has_down = any(any(pat.search(k) for pat in _DOWN_PATTERNS) for k in smooth_stats)
        if has_down:
            apply_down_proj_smooth(
                model,
                smooth_stats,
                alpha=args.alpha_down,
                num_workers=args.down_workers,
                exclude_keys=search_processed_keys,
                km=km,
            )
        elif not search_processed_keys:
            print("  [SKIP] Down smooth: no down_proj stats found in smooth-stats")
    else:
        print("  [SKIP] Down smooth: --smooth-down not set")

    # ------------------------------------------------------------------
    # Step 3.5: Verify attn output diff
    # ------------------------------------------------------------------
    verify_attn_output_diff(verify_snap, model, atol=1e-3, rtol=1e-3, km=km)

    # ------------------------------------------------------------------
    # Step 3.6: Verify MLP output diff
    # ------------------------------------------------------------------
    verify_mlp_output_diff(mlp_snap, atol=1e-2, rtol=1e-2)

    # ------------------------------------------------------------------
    # Step 4: Save transformed model
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Saving transformed model to {args.save_path} ...")
    os.makedirs(args.save_path, exist_ok=True)

    # Supplement MTP / extra weights dropped by AutoModelForCausalLM
    print("\n[Step 4.1] Detecting and supplementing extra weights (MTP etc.)...")
    loaded_keys = set(model.state_dict().keys())
    extra_state_dict = load_missing_tensors(args.model_path, loaded_keys)

    # Parallel safetensors shard writing
    save_model_parallel(
        model,
        args.save_path,
        shard_size_gb=args.shard_size_gb,
        num_workers=args.save_workers,
        extra_state_dict=extra_state_dict if extra_state_dict else None,
    )

    # Copy config / tokenizer / non-weight files from original model directory
    _NON_WEIGHT_EXTS = {".json", ".txt", ".py", ".model", ".tiktoken"}
    _SKIP_FILES = {"model.safetensors.index.json"}
    copied = []
    for fname in os.listdir(args.model_path):
        if fname in _SKIP_FILES:
            continue
        if fname.startswith("model-") and fname.endswith(".safetensors"):
            continue
        if fname == "model.safetensors":
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in _NON_WEIGHT_EXTS:
            src = os.path.join(args.model_path, fname)
            dst = os.path.join(args.save_path, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(fname)
    if copied:
        print(f"  [Save] Copied config files: {copied}")

    if tokenizer is not None:
        tokenizer.save_pretrained(args.save_path)
    print(f"  Done. Saved to {args.save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
