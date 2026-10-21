"""run_full_matrix.py — Automated Execution of Speculative Decoding Benchmark Matrix.

Orchestrates:
- Test A: Non-Speculative Baseline (Teacher only, K=0)
- Test B: Drafter 72.5k (linear K=12)
- Test C: Drafter 92.5k Kandidat (linear K=12)

Measures:
- C1 (Concurrency=1) at T=1.0 (5 runs per config, alternating)
- C4 (Concurrency=4) at T=1.0 (5 runs per config)
- Metrics: Decode tok/s, E2E tok/s, TTFT (ms), ms/tok, draft accepted tokens, tau
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("matrix_runner")

REPO_ROOT = Path("/home/orca/RadianceMine/worktrees/tap_record")
RESULTS_DIR = REPO_ROOT / "tests" / "live_results" / "matrix_benchmark"


def run_cmd(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    logger.info("Executing: %s", " ".join(cmd))
    res = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    if check and res.returncode != 0:
        logger.error("Command failed (rc=%d):\nSTDOUT: %s\nSTDERR: %s", res.returncode, res.stdout[-800:], res.stderr[-800:])
        raise RuntimeError(f"Command failed with rc {res.returncode}")
    return res


def stop_server():
    logger.info("Stopping Radiance server on port 8002...")
    run_cmd(["bash", "tests/stress/stop_engine_8002.sh"], check=False)
    time.sleep(3)


def is_matching_server_running(spec_tokens: int, sidecar_path: str | None) -> bool:
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:8002/v1/models", timeout=2.0)
        if r.status_code != 200:
            return False
        res = subprocess.run(["pgrep", "-f", "radiance.*--port 8002"], capture_output=True, text=True, check=False)
        pids = res.stdout.strip().split()
        if not pids:
            return False
        pid = pids[0]
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
        environ = Path(f"/proc/{pid}/environ").read_bytes().decode(errors="ignore")
        if f"--num-speculative-tokens\x00{spec_tokens}\x00" not in cmdline:
            return False
        if spec_tokens == 0:
            if "RADIANCE_FULLDRAFTER=1\x00" in environ:
                return False
        else:
            if sidecar_path and f"RADIANCE_FULLDRAFTER_SIDECAR={sidecar_path}\x00" not in environ:
                return False
        return True
    except (httpx.HTTPError, OSError, subprocess.SubprocessError):
        return False


def start_server(config_name: str, sidecar_path: str | None, spec_tokens: int):
    if is_matching_server_running(spec_tokens, sidecar_path):
        logger.info("Server for %s is already running and healthy on port 8002, reusing.", config_name)
        return
    stop_server()
    env = os.environ.copy()
    env["PORT"] = "8002"
    env["NUM_SPECULATIVE_TOKENS"] = str(spec_tokens)

    if spec_tokens == 0 or sidecar_path is None:
        env["RADIANCE_FULLDRAFTER"] = "0"
        env.pop("RADIANCE_FULLDRAFTER_SIDECAR", None)
    else:
        env["RADIANCE_FULLDRAFTER"] = "1"
        env["RADIANCE_FULLDRAFTER_SIDECAR"] = sidecar_path

    logger.info("Launching server for %s (spec_tokens=%d, sidecar=%s)...", config_name, spec_tokens, sidecar_path)
    res = run_cmd(["bash", "tests/stress/launch_engine_8002.sh"], env=env, check=False)
    if res.returncode != 0:
        logger.error("Launch script failed:\n%s\n%s", res.stdout, res.stderr)
        raise RuntimeError(f"Failed to start server for {config_name}")
    logger.info("Server for %s reported healthy on port 8002!", config_name)


def run_benchmark_pass(tag: str, concurrency: int, num_requests: int = 20) -> dict[str, Any]:
    out_file = RESULTS_DIR / f"{tag}_c{concurrency}.json"
    cmd = [
        sys.executable,
        "tests/stress/bench_specdec_matrix.py",
        "--url", "http://127.0.0.1:8002",
        "--concurrency", str(concurrency),
        "--num-requests", str(num_requests),
        "--max-tokens", "64",
        "--temperature", "1.0",
        "--tag", f"{tag}_C{concurrency}",
        "--output-json", str(out_file),
    ]
    logger.info("Running benchmark pass: %s (C=%d, N=%d)...", tag, concurrency, num_requests)
    run_cmd(cmd)

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["summary"]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "A_baseline": {"sidecar": None, "spec_tokens": 0, "name": "Teacher Baseline (K=0)"},
        "B_72500": {
            "sidecar": "/home/orca/drafter_data/checkpoints/flash_next_k12_exp/drafter_step_72500.safetensors",
            "spec_tokens": 12,
            "name": "Drafter 72.5k (linear K=12)",
        },
        "C_92500": {
            "sidecar": "/home/orca/drafter_data/checkpoints/flash_next_k12_exp/drafter_step_92500.safetensors",
            "spec_tokens": 12,
            "name": "Drafter 92.5k Kandidat (linear K=12)",
        },
    }

    all_summaries: dict[str, list[dict[str, Any]]] = {}

    # Schedule: 5 alternating passes for C1 and C4
    configs_order = ["A_baseline", "B_72500", "C_92500"]

    for cfg_key in configs_order:
        cfg = checkpoints[cfg_key]
        start_server(cfg["name"], cfg["sidecar"], cfg["spec_tokens"])

        # Run 5 independent measurement rounds for C=1
        logger.info("=== Starting C=1 Evaluations for %s ===", cfg["name"])
        for run_idx in range(1, 6):
            tag = f"{cfg_key}_run{run_idx}"
            summary_c1 = run_benchmark_pass(tag, concurrency=1, num_requests=20)
            all_summaries.setdefault(f"{cfg_key}_c1", []).append(summary_c1)
            time.sleep(2)

        # Run 5 independent measurement rounds for C=4
        logger.info("=== Starting C=4 Evaluations for %s ===", cfg["name"])
        for run_idx in range(1, 6):
            tag = f"{cfg_key}_run{run_idx}"
            summary_c4 = run_benchmark_pass(tag, concurrency=4, num_requests=24)
            all_summaries.setdefault(f"{cfg_key}_c4", []).append(summary_c4)
            time.sleep(2)

    stop_server()

    # Save complete matrix results
    master_file = RESULTS_DIR / "master_summary.json"
    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info("Master benchmark matrix completed and saved to %s!", master_file)

    report_md = generate_markdown_report(all_summaries)
    md_file = RESULTS_DIR / "master_summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print("\n" + report_md + "\n")
    logger.info("Summary report saved to %s!", md_file)


def generate_markdown_report(all_summaries: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Speculative Decoding Serving Benchmark (Dual AMD R9700 gfx1201)",
        "",
        "| Configuration | Concurrency | Decode tok/s | E2E tok/s | TTFT (ms) | ms/tok | Mean Acc. Tokens | Gross $\\tau$ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for key, runs in all_summaries.items():
        if not runs:
            continue
        cfg_name = runs[0].get("tag", key).split("_run")[0]
        c = runs[0].get("concurrency", 1)

        decode_means = [r["decode_tok_s"]["mean"] for r in runs if "decode_tok_s" in r]
        e2e_means = [r["e2e_tok_s"]["mean"] for r in runs if "e2e_tok_s" in r]
        ttft_means = [r["ttft_ms"]["mean"] for r in runs if "ttft_ms" in r]
        ms_tok_means = [r["time_per_token_ms"]["mean"] for r in runs if "time_per_token_ms" in r]

        d_mean = statistics.mean(decode_means) if decode_means else 0.0
        d_std = statistics.stdev(decode_means) if len(decode_means) > 1 else 0.0
        e_mean = statistics.mean(e2e_means) if e2e_means else 0.0
        e_std = statistics.stdev(e2e_means) if len(e2e_means) > 1 else 0.0
        t_mean = statistics.mean(ttft_means) if ttft_means else 0.0
        t_std = statistics.stdev(ttft_means) if len(ttft_means) > 1 else 0.0
        m_mean = statistics.mean(ms_tok_means) if ms_tok_means else 0.0
        m_std = statistics.stdev(ms_tok_means) if len(ms_tok_means) > 1 else 0.0

        # Calculate acceptance from stats_delta across runs
        acc_per_step_list = []
        for r in runs:
            s_before = r.get("stats_delta", {}).get("before", {})
            s_after = r.get("stats_delta", {}).get("after", {})
            d_acc = s_after.get("draft_accept", 0) - s_before.get("draft_accept", 0)
            d_steps = s_after.get("steps_decode", 0) - s_before.get("steps_decode", 0)
            if d_steps > 0:
                acc_per_step_list.append(d_acc / d_steps)

        if acc_per_step_list:
            acc_m = statistics.mean(acc_per_step_list)
            tau_m = acc_m + 1.0
            acc_str = f"{acc_m:.2f}"
            tau_str = f"{tau_m:.2f}"
        else:
            acc_str = "0.00 (Baseline)"
            tau_str = "1.00"

        lines.append(
            f"| **{cfg_name}** | C={c} | {d_mean:.2f} ± {d_std:.2f} | {e_mean:.2f} ± {e_std:.2f} | {t_mean:.2f} ± {t_std:.2f} | {m_mean:.2f} ± {m_std:.2f} | {acc_str} | {tau_str} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
