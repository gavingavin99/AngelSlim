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

"""TP-aware smooth-stats retrieval & summary printing.

The all-gather logic is performance-tuned: shards are grouped by
``(tensor_size, kv_replicas)`` and gathered in a single NCCL call per
group, reducing thousands of round-trips on large MoE models to a
handful of collectives.
"""

from collections import defaultdict

import torch

from angelslim.compressor.quant.core.vllm_calibrate_utils._common import _get_dist_info

__all__ = [
    "get_smooth_stats",
    "print_smooth_stats",
]


def get_smooth_stats(model):
    """Retrieve smooth statistics from the model.

    Performs all-reduce (MAX for absmax, mean for ema) across all
    workers.  Returns a serialisable dict::

        {
          "<key>": {"absmax": [float, ...], "ema": [float, ...], "call_count": int},
          ...
        }

    TP-aware gather:
        Each rank holds one shard of the full channel dimension.
        For ``k`` under GQA + TP (``kv_replicas > 1``), only every
        ``kv_replicas``-th rank carries a unique shard; the rest are
        duplicates and are skipped.  ``setup_smooth_hooks`` raises
        ``RuntimeError`` for EP, so EP never reaches here.

    Performance optimisation:
        Keys are grouped by ``(tensor_size, kv_replicas)``; each group
        runs **one** ``dist.all_gather`` instead of one per key.
    """
    if not hasattr(model, "_smooth_stats"):
        return None

    import torch.distributed as dist

    rank, world_size = _get_dist_info()

    if world_size > 1 and dist.is_initialized():
        meta = getattr(model, "_smooth_stats_meta", {})

        # Step 1: group keys by (tensor_size, kv_replicas)
        groups = defaultdict(list)  # (size, kv_reps) -> [(key, has_ema), ...]
        for key, stats in model._smooth_stats.items():
            if stats["absmax"] is None:
                continue
            kv_replicas = meta.get(key, {}).get("kv_replicas", 1)
            size = stats["absmax"].shape[0]
            has_ema = stats["ema"] is not None
            groups[(size, kv_replicas)].append((key, has_ema))

        # Step 2: per-group batched all_gather
        for (_size, kv_replicas), keys_info in groups.items():
            rows = []
            row_map = []  # tracks (key, field) for each row
            for key, has_ema in keys_info:
                rows.append(model._smooth_stats[key]["absmax"])
                row_map.append((key, "absmax"))
                if has_ema:
                    rows.append(model._smooth_stats[key]["ema"])
                    row_map.append((key, "ema"))

            stacked = torch.stack(rows, dim=0).cuda()  # [num_rows, size]
            gathered = [torch.zeros_like(stacked) for _ in range(world_size)]
            dist.all_gather(gathered, stacked)

            # De-duplicate replicated shards (GQA k heads)
            unique = gathered[::kv_replicas]

            full = torch.cat(unique, dim=1).cpu()  # [num_rows, full_size]

            for i, (key, field) in enumerate(row_map):
                model._smooth_stats[key][field] = full[i]

            del stacked, gathered, full

    # Serialise to plain Python lists (JSON-friendly)
    result = {}
    for key, stats in model._smooth_stats.items():
        result[key] = {
            "absmax": stats["absmax"].tolist() if stats["absmax"] is not None else None,
            "ema": stats["ema"].tolist() if stats["ema"] is not None else None,
            "call_count": stats.get("call_count", 0),
        }
    return result


def print_smooth_stats(model, max_rows=20):
    """Pretty-print smooth statistics summary (rank-0 only)."""
    if not hasattr(model, "_smooth_stats"):
        print("No smooth statistics available")
        return

    rank, _ = _get_dist_info()
    if rank != 0:
        return

    print("\n" + "=" * 90)
    print("Smooth Statistics  (per-channel absmax / ema — showing scalar max over channels)")
    print("=" * 90)
    rows = 0
    for key, stats in model._smooth_stats.items():
        if stats["absmax"] is None:
            absmax_repr = "N/A"
            ema_repr = "N/A"
        else:
            absmax_repr = f"{stats['absmax'].max().item():.6f}"
            ema_repr = f"{stats['ema'].max().item():.6f}" if stats["ema"] is not None else "N/A"
        print(
            f"{key:70s} | absmax_max: {absmax_repr:>12} | ema_max: {ema_repr:>12}"
            f" | calls: {stats.get('call_count', 0):4d}"
        )
        rows += 1
        if rows >= max_rows:
            remaining = len(model._smooth_stats) - max_rows
            if remaining > 0:
                print(f"  ... and {remaining} more entries (use get_smooth_stats() for full data)")
            break
    print("=" * 90 + "\n")
