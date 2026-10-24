"""bench_specdec_matrix.py — Precision Benchmarking Suite for Radiance Speculative Decoding.

Measures:
1. End-to-End throughput (tok/s)
2. Decode throughput (tok/s, excluding TTFT)
3. Time To First Token (TTFT in ms)
4. Time per emitted token (ms/tok)
5. Empirically accepted draft tokens per verification step
6. Mean emitted tokens per step (tau = n_accepted + 1)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from typing import Any

import httpx

logger = logging.getLogger("bench_specdec")

# 20 Standardized Prompts across 4 domains
BENCHMARK_PROMPTS = [
    # QA & Explanations
    "Explain the difference between latency and throughput in computer networks.",
    "Why are idempotent HTTP methods important in distributed API design?",
    "What is the role of virtual memory page tables and TLBs in modern CPUs?",
    "Describe the key differences between TCP and UDP flow control mechanisms.",
    "Explain how database write-ahead logging (WAL) guarantees durability.",
    # Coding & Algorithms
    "Write a thread-safe ring buffer in C++ using atomic load and store operations.",
    "Explain how to implement an LRU cache with O(1) get and put operations.",
    "Describe the quickselect algorithm and analyze its expected vs worst-case complexity.",
    "How does non-blocking epoll handle edge-triggered vs level-triggered events?",
    "Explain Dijkstra's shortest path algorithm using a min-heap priority queue.",
    # Reasoning & Math
    "Prove by contradiction that the square root of 2 is an irrational number.",
    "State the three cases of the Master Theorem and give an example for Case 1.",
    "Explain Bayes' theorem and apply it to a medical diagnostic test problem.",
    "What is the birthday paradox and why does it occur with only 23 people?",
    "Explain eigenvalues and eigenvectors geometrically using 2D matrix transformations.",
    # Architecture & LLM Inference
    "Explain speculative decoding and how rejection sampling maintains output distribution parity.",
    "What is PagedAttention and how does it prevent KV-cache memory fragmentation?",
    "Describe the computational differences between prefill and decode stages in LLMs.",
    "Explain flash attention and how online softmax reduces HBM memory roundtrips.",
    "How does multi-query attention (MQA) reduce KV cache footprint compared to multi-head attention?",
]


async def run_single_request(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": True,
    }

    t_start = time.perf_counter()
    t_first = None
    chunks: list[str] = []
    completion_tokens = 0

    try:
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=payload, timeout=90.0) as resp:
            if resp.status_code != 200:
                err = await resp.aread()
                return {"error": f"HTTP {resp.status_code}: {err.decode('utf-8', errors='ignore')}"}

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        if t_first is None:
                            t_first = time.perf_counter()
                        chunks.append(content)
                        completion_tokens += 1

        t_end = time.perf_counter()
        if t_first is None:
            t_first = t_end

        total_latency = t_end - t_start
        ttft_ms = (t_first - t_start) * 1000.0
        decode_time = max(1e-6, t_end - t_first)
        decode_tok_s = (completion_tokens - 1) / decode_time if completion_tokens > 1 else completion_tokens / decode_time
        e2e_tok_s = completion_tokens / max(1e-6, total_latency)
        ms_per_tok = (total_latency / completion_tokens * 1000.0) if completion_tokens > 0 else 0.0

        return {
            "success": True,
            "completion_tokens": completion_tokens,
            "ttft_ms": ttft_ms,
            "total_latency_s": total_latency,
            "decode_time_s": decode_time,
            "decode_tok_s": decode_tok_s,
            "e2e_tok_s": e2e_tok_s,
            "ms_per_tok": ms_per_tok,
            "text": "".join(chunks)[:100],
        }
    except (httpx.HTTPError, OSError) as e:
        return {"error": str(e)}


async def fetch_server_stats(client: httpx.AsyncClient, base_url: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for endpoint in ["/metrics", "/stats", "/v1/metrics"]:
        try:
            resp = await client.get(f"{base_url}{endpoint}", timeout=3.0)
            if resp.status_code == 200:
                for line in resp.text.splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        k = parts[0].split("{")[0]
                        try:
                            metrics[k] = float(parts[1])
                        except ValueError:
                            pass
                if metrics:
                    return metrics
        except (httpx.HTTPError, OSError) as e:
            logger.debug("Failed querying %s: %s", endpoint, e)
    return metrics


async def main_async(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts = BENCHMARK_PROMPTS[: args.num_requests]
    if len(prompts) < args.num_requests:
        prompts = prompts * ((args.num_requests // len(prompts)) + 1)
        prompts = prompts[: args.num_requests]

    limits = httpx.Limits(max_keepalive_connections=args.concurrency * 2, max_connections=args.concurrency * 4)
    async with httpx.AsyncClient(limits=limits, timeout=120.0) as client:
        # 1. Warmup
        print(f"[{args.tag}] Running {args.warmup} warmup request(s)...")
        for _ in range(args.warmup):
            await run_single_request(client, args.url, "Write a one-sentence greeting.", args.model, 16, args.temperature, args.top_p, args.top_k)

        stats_before = await fetch_server_stats(client, args.url)

        # 2. Main benchmark
        print(f"[{args.tag}] Benchmarking {len(prompts)} requests (concurrency={args.concurrency}, T={args.temperature})...")
        t_bench_start = time.perf_counter()

        sem = asyncio.Semaphore(args.concurrency)

        async def worker(idx: int, p: str):
            async with sem:
                res = await run_single_request(client, args.url, p, args.model, args.max_tokens, args.temperature, args.top_p, args.top_k)
                res["req_id"] = idx
                return res

        tasks = [worker(i, p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        t_bench_end = time.perf_counter()

        stats_after = await fetch_server_stats(client, args.url)

    # 3. Analyze results
    valid = [r for r in results if r.get("success")]
    errors = [r for r in results if not r.get("success")]

    total_tokens = sum(r["completion_tokens"] for r in valid)
    bench_wall_time = t_bench_end - t_bench_start

    e2e_speeds = [r["e2e_tok_s"] for r in valid]
    decode_speeds = [r["decode_tok_s"] for r in valid]
    ttfts = [r["ttft_ms"] for r in valid]
    ms_per_toks = [r["ms_per_tok"] for r in valid]

    # Compute speculative tau from server metrics delta
    dtok = stats_after.get("radiance:decode_tokens_total", 0.0) - stats_before.get("radiance:decode_tokens_total", 0.0)
    dstep = stats_after.get("radiance:engine_steps_decode_total", 0.0) - stats_before.get("radiance:engine_steps_decode_total", 0.0)
    tau = (dtok / dstep) if dstep > 0 else 0.0

    summary = {
        "tag": args.tag,
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "requests_total": len(results),
        "requests_success": len(valid),
        "requests_failed": len(errors),
        "total_completion_tokens": total_tokens,
        "wall_time_s": bench_wall_time,
        "aggregate_throughput_tok_s": total_tokens / bench_wall_time if bench_wall_time > 0 else 0.0,
        "tau": tau,
        "e2e_tok_s": {
            "mean": statistics.mean(e2e_speeds) if e2e_speeds else 0.0,
            "std": statistics.stdev(e2e_speeds) if len(e2e_speeds) > 1 else 0.0,
            "median": statistics.median(e2e_speeds) if e2e_speeds else 0.0,
        },
        "decode_tok_s": {
            "mean": statistics.mean(decode_speeds) if decode_speeds else 0.0,
            "std": statistics.stdev(decode_speeds) if len(decode_speeds) > 1 else 0.0,
            "median": statistics.median(decode_speeds) if decode_speeds else 0.0,
        },
        "ttft_ms": {
            "mean": statistics.mean(ttfts) if ttfts else 0.0,
            "std": statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
            "median": statistics.median(ttfts) if ttfts else 0.0,
        },
        "time_per_token_ms": {
            "mean": statistics.mean(ms_per_toks) if ms_per_toks else 0.0,
            "std": statistics.stdev(ms_per_toks) if len(ms_per_toks) > 1 else 0.0,
        },
        "stats_delta": {"before": stats_before, "after": stats_after},
    }

    print("\n" + "=" * 70)
    print(f" BENCHMARK SUMMARY: {args.tag} (C={args.concurrency}, T={args.temperature})")
    print("=" * 70)
    print(f" Success / Total Requests : {len(valid)} / {len(results)}")
    print(f" Aggregate Throughput     : {summary['aggregate_throughput_tok_s']:.2f} tok/s")
    print(f" Decode Speed (excl TTFT) : {summary['decode_tok_s']['mean']:.2f} ± {summary['decode_tok_s']['std']:.2f} tok/s (median: {summary['decode_tok_s']['median']:.2f})")
    print(f" End-to-End Speed         : {summary['e2e_tok_s']['mean']:.2f} ± {summary['e2e_tok_s']['std']:.2f} tok/s (median: {summary['e2e_tok_s']['median']:.2f})")
    print(f" Time To First Token (TTFT): {summary['ttft_ms']['mean']:.2f} ± {summary['ttft_ms']['std']:.2f} ms (median: {summary['ttft_ms']['median']:.2f} ms)")
    print(f" Time per Token           : {summary['time_per_token_ms']['mean']:.2f} ± {summary['time_per_token_ms']['std']:.2f} ms/tok")
    if tau > 0:
        print(f" Realized Speculative Tau : {tau:.3f} tokens/step")
    print("=" * 70)

    return summary, valid


def main():
    parser = argparse.ArgumentParser(description="Precision Benchmark Suite for Radiance SpecDec")
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="qwen4exp")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    summary, per_request = asyncio.run(main_async(args))

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_request": per_request}, f, indent=2)
        print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
