#!/usr/bin/env python3
"""tools/qsa_overlap_sim.py -- QSA Cross-Layer and Temporal Overlap Simulation & Feasibility Gate.

Phase 0 Feasibility Gate:
Checks whether QSA block selection (512 micro-blocks of 4 tokens each) between
Layer L and Layer L+4 in step t, combined with Layer L+4 in step t-1, achieves
>= 90% hit rate for asynchronous host-offloading prefetch.

Prediction model:
  Target:  TopK(L+4, t)
  P_cross: TopK(L, t)
  P_temp:  TopK(L+4, t-1)
  P_union: TopK(L, t) U TopK(L+4, t-1)

Gate Decision:
  - If hit_rate(P_union) >= 90%: Path B (Host-Offloading) is viable.
  - If hit_rate(P_union) <  90%: Focus primarily on Path A (KVarN 1:1 Compression).
"""

import argparse
import itertools
import os
import sys
from collections import defaultdict

import numpy as np

QSA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]


def parse_dump_file(path):
    """Parses QSA dump file.

    Format per line: layer \\t token_idx \\t position \\t sel_0 ... sel_{topk-1}
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            layer = int(parts[0])
            tok_idx = int(parts[1])
            pos = int(parts[2])
            sels = [int(x) for x in parts[3:] if int(x) >= 0]
            records.append({
                "layer": layer,
                "tok_idx": tok_idx,
                "pos": pos,
                "sels": set(sels),
            })
    return records


def group_by_steps(records):
    """Groups records by step (indexed by pos or chronological sequence).

    Returns list of dicts: step -> { layer: set(block_ids) }
    """
    steps = defaultdict(dict)
    if records and records[0]["pos"] >= 0:
        for r in records:
            steps[r["pos"]][r["layer"]] = r["sels"]
        sorted_pos = sorted(steps.keys())
        return [steps[p] for p in sorted_pos]

    step_list = []
    curr = {}
    for r in records:
        if r["layer"] == QSA_LAYERS[0] and curr:
            step_list.append(curr)
            curr = {}
        curr[r["layer"]] = r["sels"]
    if curr:
        step_list.append(curr)
    return step_list


def generate_synthetic_trace(n_ctx, n_steps=200, topk=512, ratio=4, seed=42):
    """Synthesizes realistic QSA block selections for a sequence of length n_ctx across n_steps.

    In long-context LLMs (Qwen/DeepSeek):
    - Sinks / initial tokens (~16 blocks = 64 tokens) are attended by all layers.
    - Local window (~32 blocks = 128 tokens) is attended by all layers.
    - Semantic clusters / retrieval blocks (middle context):
      * Temporal persistence is high (drift ~2-5% per decode step).
      * Cross-layer variation: Different layers attend to different semantic subsets
        due to layer-specific projection heads and transformer depth representation drift.
    """
    rng = np.random.RandomState(seed)
    n_blocks = max(n_ctx // ratio, topk)

    sink_blocks = set(range(min(16, n_blocks)))

    layer_centroids = {}
    for l in QSA_LAYERS:
        layer_centroids[l] = rng.choice(n_blocks, size=12, replace=False)

    step_list = []

    layer_semantic = {}
    for l in QSA_LAYERS:
        candidates = []
        for c in layer_centroids[l]:
            cluster = np.clip(
                rng.normal(c, scale=20, size=50).astype(int), 0, n_blocks - 1
            )
            candidates.extend(cluster)
        chosen = set(candidates) - sink_blocks
        while len(chosen) < topk - 48:
            chosen.add(int(rng.randint(0, n_blocks)))
        layer_semantic[l] = set(list(chosen)[: topk - 48])

    curr_pos = n_ctx
    for _ in range(n_steps):
        step_data = {}
        curr_pos += 1
        local_blocks = set(
            range(max(0, (curr_pos // ratio) - 32), curr_pos // ratio)
        )

        for l in QSA_LAYERS:
            prev = list(layer_semantic[l])
            n_drift = max(1, int(0.03 * len(prev)))
            to_remove = set(rng.choice(prev, size=n_drift, replace=False))
            kept = set(prev) - to_remove
            while len(kept) < topk - 48:
                new_block = int(rng.randint(0, n_blocks))
                if (
                    new_block not in sink_blocks
                    and new_block not in local_blocks
                ):
                    kept.add(new_block)
            layer_semantic[l] = kept

            all_selected = sink_blocks | local_blocks | layer_semantic[l]
            step_data[l] = set(list(all_selected)[:topk])

        step_list.append(step_data)

    return step_list


def analyze_overlap(step_list):
    """Computes cross-layer, temporal, and combined union hit rates."""
    if len(step_list) < 2:
        return None

    cross_hits = []
    temp_hits = []
    union_hits = []
    prefetch_sizes = []
    cache_misses = []

    pairs = list(itertools.pairwise(QSA_LAYERS))

    for t in range(1, len(step_list)):
        curr_step = step_list[t]
        prev_step = step_list[t - 1]

        for L_curr, L_next in pairs:
            if (
                L_curr not in curr_step
                or L_next not in curr_step
                or L_next not in prev_step
            ):
                continue

            target = curr_step[L_next]
            if not target:
                continue

            s_cross = curr_step[L_curr]
            s_temp = prev_step[L_next]
            s_union = s_cross | s_temp

            hit_cross = len(s_cross & target) / len(target)
            hit_temp = len(s_temp & target) / len(target)
            hit_union = len(s_union & target) / len(target)

            new_prefetches = len(s_union - s_temp)
            miss_count = len(target - s_union)

            cross_hits.append(hit_cross)
            temp_hits.append(hit_temp)
            union_hits.append(hit_union)
            prefetch_sizes.append(new_prefetches)
            cache_misses.append(miss_count)

    return {
        "cross_mean": float(np.mean(cross_hits)) if cross_hits else 0.0,
        "cross_std": float(np.std(cross_hits)) if cross_hits else 0.0,
        "temp_mean": float(np.mean(temp_hits)) if temp_hits else 0.0,
        "temp_std": float(np.std(temp_hits)) if temp_hits else 0.0,
        "union_mean": float(np.mean(union_hits)) if union_hits else 0.0,
        "union_std": float(np.std(union_hits)) if union_hits else 0.0,
        "new_prefetches_mean": float(np.mean(prefetch_sizes))
        if prefetch_sizes
        else 0.0,
        "miss_count_mean": float(np.mean(cache_misses))
        if cache_misses
        else 0.0,
        "n_samples": len(union_hits),
    }


def print_report(res, title):
    print("=" * 80)
    print(f"QSA BLOCK SELECTION OVERLAP REPORT: {title}")
    print("=" * 80)
    print(f"Total step transitions analyzed: {res['n_samples']}")
    print("1. Cross-Layer Hit Rate [TopK(L_t) -> TopK(L+4_t)]:")
    print(
        f"   Mean: {res['cross_mean'] * 100:.2f}% (std: {res['cross_std'] * 100:.2f}%)"
    )
    print("2. Temporal Persistence [TopK(L+4_{{t-1}}) -> TopK(L+4_t)]:")
    print(
        f"   Mean: {res['temp_mean'] * 100:.2f}% (std: {res['temp_std'] * 100:.2f}%)"
    )
    print("3. Combined Union Predictor [TopK(L_t) U TopK(L+4_{{t-1}})]:")
    print(
        f"   Mean: {res['union_mean'] * 100:.2f}% (std: {res['union_std'] * 100:.2f}%)"
    )
    print(
        f"   Avg New Prefetched Blocks / Step: {res['new_prefetches_mean']:.1f}"
    )
    print(
        f"   Avg Cache Misses / Step:          {res['miss_count_mean']:.1f} blocks"
    )
    print("-" * 80)

    gate_threshold = 0.90
    print(f"GATE THRESHOLD: {gate_threshold * 100:.1f}%")
    if res["union_mean"] >= gate_threshold:
        print("RESULT: PASS [Gate Condition >= 90% Satisfied]")
        print(
            "RECOMMENDATION: Path B (QSA Cross-Layer Host-Offloading) is HIGHLY VIABLE."
        )
        print("  - Cross-layer + temporal union covers >= 90% of working set.")
        print(
            "  - Remaining <= 10% misses handled gracefully via PCIe Zero-Copy fallback."
        )
    else:
        print("RESULT: FAIL / MARGINAL [Gate Condition < 90%]")
        print(
            "RECOMMENDATION: Focus primarily on Path A (KVarN 1:1 Compression)."
        )
        print(
            "  - VRAM compression delivers deterministic, lossless memory scaling."
        )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="QSA Selection Overlap Simulator & Gate"
    )
    parser.add_argument(
        "--dump", type=str, help="Path to RADIANCE_DUMP_QSA_SEL trace file"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic evaluation on 32k and 64k",
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of decode steps"
    )
    parser.add_argument(
        "--topk", type=int, default=512, help="QSA budget / ratio (512)"
    )
    args = parser.parse_args()

    if args.dump:
        if not os.path.exists(args.dump):
            print(f"Error: dump file {args.dump} not found", file=sys.stderr)
            sys.exit(1)
        records = parse_dump_file(args.dump)
        steps = group_by_steps(records)
        res = analyze_overlap(steps)
        if res:
            print_report(res, f"Real Trace: {args.dump}")
        else:
            print("Error: insufficient data in trace", file=sys.stderr)
            sys.exit(1)
    else:
        print(
            "Running Phase 0 Feasibility Gate on 32k and 64k contexts (200 decode steps each)..."
        )
        for ctx_len in [32768, 65536]:
            steps = generate_synthetic_trace(
                n_ctx=ctx_len, n_steps=args.steps, topk=args.topk
            )
            res = analyze_overlap(steps)
            print_report(
                res, f"Context {ctx_len // 1024}k Tokens ({args.steps} Decode Steps)"
            )


if __name__ == "__main__":
    main()
