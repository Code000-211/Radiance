"""bench_5runs.py - Multi-Run Benchmark Suite for Radiance Speculative Drafter.

Executes 5 consecutive multi-turn benchmark rounds against Port 8002,
measuring throughput, acceptance length (tau), latency, and stability.
"""

import json
import os
import statistics
import subprocess
import sys
import time

TARGET_URL = "http://127.0.0.1:8002"
NUM_REQUESTS = 30
CONCURRENCY = 4
MAX_TOKENS = 64
NUM_RUNS = 5
OUTPUT_JSON = "tests/live_results/better_bench_5runs.json"


def run_benchmark():
    os.makedirs("tests/live_results", exist_ok=True)
    runs_data = []

    print("=" * 72)
    print(" RADIANCE SPECULATIVE DRAFTER - 5-RUN BENCHMARK SUITE")
    print(f" Target: {TARGET_URL} | Runs: {NUM_RUNS} | Reqs/Run: {NUM_REQUESTS} | Concurrency: {CONCURRENCY}")
    print("=" * 72)

    for run_idx in range(1, NUM_RUNS + 1):
        temp_out = f"tests/live_results/run_{run_idx}.json"
        cmd = [
            sys.executable,
            "tests/stress/multi_turn_500.py",
            "--url", TARGET_URL,
            "--num-requests", str(NUM_REQUESTS),
            "--concurrency", str(CONCURRENCY),
            "--max-tokens", str(MAX_TOKENS),
            "--output-json", temp_out,
        ]

        print(f"\n--- [Run {run_idx}/{NUM_RUNS}] Executing benchmark ({NUM_REQUESTS} requests, c={CONCURRENCY})... ---")
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        wall_time = time.time() - t0

        if not os.path.exists(temp_out):
            print(f"❌ Run {run_idx} failed! (exit code: {res.returncode})")
            print("STDERR:", res.stderr[-500:])
            return 1

        with open(temp_out, "r") as f:
            data = json.load(f)

        summary = data.get("summary", data)
        tok_per_sec = summary.get("throughput_tok_per_sec", 0.0)
        tau = summary.get("tau", 0.0)
        req_per_sec = summary.get("requests_per_sec", 0.0)
        lat_avg = summary.get("latency", {}).get("avg_secs", 0.0)
        lat_p95 = summary.get("latency", {}).get("p95_secs", 0.0)
        crashes = summary.get("zero_failure_assertions", {}).get("server_crashes", 0)
        nans = summary.get("zero_failure_assertions", {}).get("nans_detected", 0)

        run_info = {
            "run": run_idx,
            "wall_time_secs": wall_time,
            "throughput_tok_per_sec": tok_per_sec,
            "tau": tau,
            "requests_per_sec": req_per_sec,
            "latency_avg_secs": lat_avg,
            "latency_p95_secs": lat_p95,
            "crashes": crashes,
            "nans": nans,
            "tokens_generated": summary.get("tokens_generated", 0),
        }
        runs_data.append(run_info)

        print(f"✓ Run {run_idx} complete: {tok_per_sec:.2f} tok/s | tau={tau:.3f} | {req_per_sec:.2f} req/s | avg_lat={lat_avg:.3f}s | p95={lat_p95:.3f}s")
        time.sleep(2)  # Settle pause

    # Aggregate Statistics
    toks = [r["throughput_tok_per_sec"] for r in runs_data]
    taus = [r["tau"] for r in runs_data]
    reqs = [r["requests_per_sec"] for r in runs_data]
    lats = [r["latency_avg_secs"] for r in runs_data]
    p95s = [r["latency_p95_secs"] for r in runs_data]

    stats = {
        "runs": runs_data,
        "aggregate": {
            "throughput_tok_per_sec": {
                "mean": statistics.mean(toks),
                "stddev": statistics.stdev(toks) if len(toks) > 1 else 0.0,
                "min": min(toks),
                "max": max(toks),
                "median": statistics.median(toks),
            },
            "tau": {
                "mean": statistics.mean(taus),
                "stddev": statistics.stdev(taus) if len(taus) > 1 else 0.0,
                "min": min(taus),
                "max": max(taus),
            },
            "requests_per_sec": {
                "mean": statistics.mean(reqs),
                "min": min(reqs),
                "max": max(reqs),
            },
            "latency_avg_secs": {
                "mean": statistics.mean(lats),
                "min": min(lats),
                "max": max(lats),
            },
            "latency_p95_secs": {
                "mean": statistics.mean(p95s),
                "min": min(p95s),
                "max": max(p95s),
            },
            "total_crashes": sum(r["crashes"] for r in runs_data),
            "total_nans": sum(r["nans"] for r in runs_data),
        }
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    # Print Formatted Report
    print("\n" + "=" * 72)
    print(" 5-RUN BENCHMARK SUMMARY REPORT")
    print("=" * 72)
    print(f"{'Run':<6} | {'Throughput (tok/s)':<20} | {'Tau':<8} | {'Req/s':<8} | {'Avg Lat':<9} | {'P95 Lat':<9} | {'Status'}")
    print("-" * 72)
    for r in runs_data:
        status = "PASSED" if r["crashes"] == 0 and r["nans"] == 0 else "FAILED"
        print(f"#{r['run']:<5} | {r['throughput_tok_per_sec']:<20.2f} | {r['tau']:<8.3f} | {r['requests_per_sec']:<8.2f} | {r['latency_avg_secs']:<9.3f}s | {r['latency_p95_secs']:<9.3f}s | {status}")

    agg = stats["aggregate"]
    print("-" * 72)
    print(f"Mean   | {agg['throughput_tok_per_sec']['mean']:<20.2f} | {agg['tau']['mean']:<8.3f} | {agg['requests_per_sec']['mean']:<8.2f} | {agg['latency_avg_secs']['mean']:<9.3f}s | {agg['latency_p95_secs']['mean']:<9.3f}s | 0 Crashes")
    print(f"StdDev | {agg['throughput_tok_per_sec']['stddev']:<20.2f} | {agg['tau']['stddev']:<8.3f} |")
    print(f"Min    | {agg['throughput_tok_per_sec']['min']:<20.2f} | {agg['tau']['min']:<8.3f} |")
    print(f"Max    | {agg['throughput_tok_per_sec']['max']:<20.2f} | {agg['tau']['max']:<8.3f} |")
    print("=" * 72)
    print(f"Results saved to: {OUTPUT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(run_benchmark())
