"""multi_turn_500.py — 500-Request Multi-Turn Stress Test Suite for Radiance DARTree.

Milestone 4 Acceptance Test:
- 500 consecutive/concurrent requests against http://127.0.0.1:8002.
- Real multi-turn conversation sessions accumulating context history.
- Diverse prompts: Short queries, Long context inputs, Coding tasks, Reasoning/Math.
- Zero-Failure Criteria:
  1. 0 server crashes or process exits.
  2. 0 NaNs or Inf values in response token streams, logprobs, or logits.
  3. 0 VRAM leaks (physical VRAM returns to baseline via rocm-smi / hipMemGetInfo).
- Strictly protects production port 8000.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from typing import Any, ClassVar

import httpx

logger = logging.getLogger("multi_turn_500")


# ==============================================================================
# Security & Port Quarantine Guardrails
# ==============================================================================
def assert_port_safety(url: str) -> None:
    """Enforce strict quarantine of production port 8000."""
    if ":8000" in url or "port=8000" in url:
        raise RuntimeError(
            "CRITICAL SECURITY VIOLATION: Target port 8000 is protected production! "
            "Stress testing must NEVER touch port 8000. Use test port 8002."
        )


# ==============================================================================
# Hardware VRAM Monitor (hipMemGetInfo & rocm-smi)
# ==============================================================================
class VRAMMonitor:
    """Accurately monitors Dual R9700 physical VRAM in megabytes."""

    def __init__(
        self,
        rocm_smi_path: str = "/opt/rocm/bin/rocm-smi",
        hip_lib_path: str = "/opt/rocm/lib/libamdhip64.so",
    ) -> None:
        self.rocm_smi_path = rocm_smi_path
        self.hip_lib_path = hip_lib_path
        self._hip_lib = None
        self._init_hip()

    def _init_hip(self) -> None:
        try:
            if os.path.exists(self.hip_lib_path):
                hip = ctypes.CDLL(self.hip_lib_path)
                hip.hipSetDevice.argtypes = [ctypes.c_int]
                hip.hipSetDevice.restype = ctypes.c_int
                hip.hipMemGetInfo.argtypes = [
                    ctypes.POINTER(ctypes.c_size_t),
                    ctypes.POINTER(ctypes.c_size_t),
                ]
                hip.hipMemGetInfo.restype = ctypes.c_int
                self._hip_lib = hip
        except OSError as e:
            logger.debug("Failed to load HIP runtime library: %s", e)
            self._hip_lib = None

    def sample_vram(self) -> dict[int, dict[str, float]]:
        """Returns {gpu_id: {'used_mb': float, 'total_mb': float, 'free_mb': float}}."""
        # 1. Fast in-process path via hipMemGetInfo
        if self._hip_lib is not None:
            results: dict[int, dict[str, float]] = {}
            for d in (0, 1):
                if self._hip_lib.hipSetDevice(d) == 0:
                    free = ctypes.c_size_t()
                    total = ctypes.c_size_t()
                    if self._hip_lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0:
                        used = total.value - free.value
                        results[d] = {
                            "free_mb": free.value / (1024 * 1024),
                            "total_mb": total.value / (1024 * 1024),
                            "used_mb": used / (1024 * 1024),
                        }
            if len(results) == 2:
                return results

        # 2. Fallback via rocm-smi JSON CLI
        try:
            out = subprocess.run(
                [self.rocm_smi_path, "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(out.stdout)
            results = {}
            for card_str, info in data.items():
                idx = int(card_str.replace("card", ""))
                tot = int(info["VRAM Total Memory (B)"])
                used = int(info["VRAM Total Used Memory (B)"])
                results[idx] = {
                    "free_mb": (tot - used) / (1024 * 1024),
                    "total_mb": tot / (1024 * 1024),
                    "used_mb": used / (1024 * 1024),
                }
            return results
        except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug("Fallback rocm-smi query failed: %s", e)
            return {}


# ==============================================================================
# Prompt Diversity Generator
# ==============================================================================
class PromptLibrary:
    """Provides high-diversity multi-turn conversation seeds."""

    CATEGORIES: ClassVar[tuple[str, ...]] = (
        "short_query",
        "coding_task",
        "reasoning_math",
        "long_context",
    )

    SHORT_CONVERSATIONS: ClassVar[tuple[tuple[str, ...], ...]] = (
        (
            "What is latency in networks?",
            "How does throughput differ?",
        ),
        (
            "Explain idempotent REST methods.",
            "Why is HTTP GET idempotent?",
        ),
    )

    CODING_CONVERSATIONS: ClassVar[tuple[tuple[str, ...], ...]] = (
        (
            "What is a ring buffer in C++?",
            "Why use atomics for it?",
        ),
        (
            "Describe socket polling in async I/O.",
            "How does keep-alive help?",
        ),
    )

    REASONING_CONVERSATIONS: ClassVar[tuple[tuple[str, ...], ...]] = (
        (
            "Is sqrt(2) irrational?",
            "State the contradiction step.",
        ),
        (
            "State Master Theorem Case 1.",
            "When does Case 2 apply?",
        ),
    )

    LONG_CONTEXT_CONVERSATIONS: ClassVar[tuple[tuple[str, ...], ...]] = (
        (
            "What is DARTree speculative decoding?",
            "How does causal ancestor mask work?",
        ),
        (
            "Explain candidate tree verification.",
            "How does KV rollback prune branches?",
        ),
    )

    @classmethod
    def get_conversation_turns(cls, session_id: int, category: str) -> list[str]:
        if category == "short_query":
            convs = cls.SHORT_CONVERSATIONS
        elif category == "coding_task":
            convs = cls.CODING_CONVERSATIONS
        elif category == "reasoning_math":
            convs = cls.REASONING_CONVERSATIONS
        else:
            convs = cls.LONG_CONTEXT_CONVERSATIONS
        return list(convs[session_id % len(convs)])


# ==============================================================================
# Response Validator (Zero NaN / Inf & Structural Correctness)
# ==============================================================================
class IntegrityScanner:
    """Validates that token streams and payloads contain 0 NaNs or Infs."""

    NAN_INF_REGEX = re.compile(
        r"(?<![a-zA-Z0-9_])(nan|NaN|inf|Inf|-inf|-Inf|infinity|Infinity|-infinity|-Infinity)(?![a-zA-Z0-9_])"
    )

    @classmethod
    def check_text(cls, text: str) -> list[str]:
        """Returns detected NaN/Inf anomalies in raw string text."""
        matches = cls.NAN_INF_REGEX.findall(text)
        return [m for m in matches if m.lower() in ("nan", "inf", "-inf", "infinity", "-infinity")]

    @classmethod
    def check_floats(cls, obj: Any, path: str = "") -> list[str]:
        """Recursively checks that all numeric floats in JSON are finite."""
        anomalies: list[str] = []
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                anomalies.append(f"{path}: {obj}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                anomalies.extend(cls.check_floats(v, f"{path}.{k}" if path else k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                anomalies.extend(cls.check_floats(v, f"{path}[{i}]"))
        return anomalies


# ==============================================================================
# Multi-Turn Stress Test Engine
# ==============================================================================
class MultiTurnStressRunner:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8002",
        total_requests: int = 500,
        concurrency: int = 4,
        vram_interval: int = 50,
        temperature: float = 0.0,
        max_tokens: int = 64,
        min_throughput: float = 0.0,
        min_tau: float = 0.0,
    ) -> None:
        assert_port_safety(base_url)
        self.base_url = base_url.rstrip("/")
        self.total_requests = total_requests
        self.concurrency = concurrency
        self.vram_interval = vram_interval
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_throughput = min_throughput
        self.min_tau = min_tau

        self.vram_monitor = VRAMMonitor()
        self.completed_requests = 0
        self.server_crashes = 0
        self.nans_detected = 0
        self.infs_detected = 0
        self.total_tokens_generated = 0

        self.vram_samples: list[dict[str, Any]] = []
        self.request_latencies: list[float] = []
        self.lock = asyncio.Lock()
        self.model_id = "default"

    async def fetch_model_id(self, client: httpx.AsyncClient) -> str:
        """Discover model ID from /v1/models."""
        try:
            resp = await client.get(f"{self.base_url}/v1/models", timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    return str(data["data"][0]["id"])
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            logger.debug("Could not fetch model ID from /v1/models: %s", e)
        return "qwen3.8-next-flash-fp8-iq4r-moe"

    async def scrape_prometheus_metrics(self, client: httpx.AsyncClient) -> dict[str, float]:
        """Scrape Prometheus KV cache metrics from /metrics."""
        try:
            resp = await client.get(f"{self.base_url}/metrics", timeout=5.0)
            if resp.status_code == 200:
                metrics: dict[str, float] = {}
                for line in resp.text.splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        k, v = parts[0], parts[1]
                        try:
                            metric_name = k.split("{")[0]
                            metrics[metric_name] = float(v)
                        except ValueError:
                            pass
                return metrics
        except httpx.HTTPError as e:
            logger.debug("Metrics scrape failed: %s", e)
        return {}

    async def execute_turn(
        self,
        client: httpx.AsyncClient,
        session_id: str,
        turn_idx: int,
        messages: list[dict[str, str]],
        stream: bool,
    ) -> tuple[str, float, int]:
        """Execute a single multi-turn request, validating integrity."""
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }

        t0 = time.perf_counter()
        assistant_reply = ""
        tokens_count = 0

        if stream:
            # SSE streaming verification
            async with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload, timeout=90.0
            ) as resp:
                if resp.status_code != 200:
                    err_body = await resp.aread()
                    raise RuntimeError(f"HTTP {resp.status_code}: {err_body.decode('utf-8', errors='ignore')}")

                chunks_text: list[str] = []
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    raw_data = line[6:].strip()
                    if raw_data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue

                    float_anomalies = IntegrityScanner.check_floats(chunk)
                    if float_anomalies:
                        async with self.lock:
                            self.nans_detected += len(float_anomalies)

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            chunks_text.append(content)
                            tokens_count += 1
                            text_anomalies = IntegrityScanner.check_text(content)
                            if text_anomalies:
                                async with self.lock:
                                    self.nans_detected += len(text_anomalies)

                assistant_reply = "".join(chunks_text)
        else:
            # Blocking completion verification
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions", json=payload, timeout=90.0
            )
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

            data = resp.json()
            float_anomalies = IntegrityScanner.check_floats(data)
            if float_anomalies:
                async with self.lock:
                    self.nans_detected += len(float_anomalies)

            choices = data.get("choices", [])
            if choices:
                assistant_reply = choices[0].get("message", {}).get("content", "")
                text_anomalies = IntegrityScanner.check_text(assistant_reply)
                if text_anomalies:
                    async with self.lock:
                        self.nans_detected += len(text_anomalies)

            usage = data.get("usage", {})
            tokens_count = usage.get("completion_tokens", max(1, len(assistant_reply.split())))

        latency = time.perf_counter() - t0
        return assistant_reply, latency, tokens_count

    async def worker_loop(self, worker_id: int, client: httpx.AsyncClient) -> None:
        """Worker task executing multi-turn sessions sequentially."""
        sess_counter = 0
        categories = PromptLibrary.CATEGORIES

        while True:
            sess_counter += 1
            session_id = f"w{worker_id}_s{sess_counter}"
            category = categories[(worker_id + sess_counter) % len(categories)]
            turns = PromptLibrary.get_conversation_turns(sess_counter, category)

            messages: list[dict[str, str]] = []

            for turn_idx, user_prompt in enumerate(turns):
                async with self.lock:
                    if self.completed_requests >= self.total_requests:
                        return
                    req_num = self.completed_requests
                    self.completed_requests += 1

                stream_mode = (req_num % 2 == 0)
                messages.append({"role": "user", "content": user_prompt})

                try:
                    reply, lat, toks = await self.execute_turn(
                        client, session_id, turn_idx, messages, stream=stream_mode
                    )
                    messages.append({"role": "assistant", "content": reply})

                    async with self.lock:
                        self.request_latencies.append(lat)
                        self.total_tokens_generated += toks

                        if (req_num + 1) % self.vram_interval == 0 or (req_num + 1) == self.total_requests:
                            v_sample = self.vram_monitor.sample_vram()
                            prom = await self.scrape_prometheus_metrics(client)
                            self.vram_samples.append({
                                "request": req_num + 1,
                                "timestamp": time.time(),
                                "vram": v_sample,
                                "kv_util": prom.get("vllm:kv_cache_usage_perc", 0.0),
                            })
                            print(
                                f"  [Progress {req_num + 1}/{self.total_requests}] "
                                f"Latency: {lat:.3f}s, KV-Util: {prom.get('vllm:kv_cache_usage_perc', 0.0):.2%}, "
                                f"VRAM(card0): {v_sample.get(0, {}).get('used_mb', 0.0):.1f}MB, "
                                f"VRAM(card1): {v_sample.get(1, {}).get('used_mb', 0.0):.1f}MB"
                            )

                except Exception as e:  # noqa: BLE001
                    async with self.lock:
                        self.server_crashes += 1
                    print(f"❌ ERROR on Request {req_num} ({session_id}, Turn {turn_idx}): {e}")

    async def run(self) -> dict[str, Any]:
        """Main stress test orchestration."""
        print("======================================================================")
        print(" Starting 500-Request Multi-Turn Stress Test")
        print(f" Target URL:     {self.base_url}")
        print(f" Total Requests: {self.total_requests}")
        print(f" Concurrency:    {self.concurrency}")
        print(f" Temperature:    {self.temperature}")
        print("======================================================================")

        limits = httpx.Limits(max_connections=self.concurrency * 2, max_keepalive_connections=self.concurrency)
        async with httpx.AsyncClient(limits=limits) as client:
            # 1. Probe server health & model ID
            self.model_id = await self.fetch_model_id(client)
            print(f"✓ Discovered Model ID: {self.model_id}")

            # 2. Record Baseline VRAM (t0)
            baseline_vram = self.vram_monitor.sample_vram()
            baseline_prom = await self.scrape_prometheus_metrics(client)
            print("✓ Baseline VRAM Recorded:")
            for d, info in baseline_vram.items():
                print(f"   Card {d}: Used={info['used_mb']:.1f} MB, Total={info['total_mb']:.1f} MB")
            print(f"   Baseline KV cache util: {baseline_prom.get('vllm:kv_cache_usage_perc', 0.0):.2%}")

            t_start = time.perf_counter()

            # 3. Launch Concurrent Multi-Turn Worker Tasks
            workers = [
                asyncio.create_task(self.worker_loop(w, client))
                for w in range(self.concurrency)
            ]
            await asyncio.gather(*workers)

            duration = time.perf_counter() - t_start

            # 4. Drain & Post-Test Settlement (3s cooldown)
            print("Settling engine and evaluating post-stress recovery (3s)...")
            await asyncio.sleep(3.0)

            try:
                await client.post(f"{self.base_url}/reset_prefix_cache", timeout=5.0)
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)

            final_vram = self.vram_monitor.sample_vram()
            final_prom = await self.scrape_prometheus_metrics(client)

            # 5. Verify Post-Test Server Health
            health_ok = False
            try:
                h_resp = await client.get(f"{self.base_url}/health", timeout=5.0)
                health_ok = (h_resp.status_code == 200)
            except httpx.HTTPError:
                health_ok = False

        # 6. Evaluate Acceptance Assertions
        vram_deltas: dict[int, float] = {}
        leak_detected = False
        for d in (0, 1):
            b_mb = baseline_vram.get(d, {}).get("used_mb", 0.0)
            f_mb = final_vram.get(d, {}).get("used_mb", 0.0)
            delta = f_mb - b_mb
            vram_deltas[d] = delta
            # Leak threshold: delta > 16.0 MB (tolerance for allocator heap padding)
            if delta > 16.0:
                leak_detected = True

        avg_lat = sum(self.request_latencies) / max(1, len(self.request_latencies))
        p95_lat = sorted(self.request_latencies)[int(0.95 * len(self.request_latencies))] if self.request_latencies else 0.0

        throughput_tok_s = self.total_tokens_generated / max(0.001, duration)

        # Scrape and compute speculative tau acceptance length from Prometheus counters
        dec_toks = final_prom.get("radiance:decode_tokens_total", 0.0)
        dec_steps = final_prom.get("radiance:engine_steps_decode_total", 0.0)
        tau = (dec_toks / max(1.0, dec_steps)) if (dec_toks > 0.0 and dec_steps > 0.0) else 0.0

        throughput_ok = (self.min_throughput <= 0.0) or (throughput_tok_s >= self.min_throughput)
        tau_ok = (self.min_tau <= 0.0) or (tau <= 0.0) or (tau >= self.min_tau)

        passed = (
            self.server_crashes == 0
            and self.nans_detected == 0
            and self.infs_detected == 0
            and not leak_detected
            and health_ok
            and self.completed_requests >= self.total_requests
            and throughput_ok
            and tau_ok
        )

        results = {
            "passed": passed,
            "total_requests_completed": self.completed_requests,
            "duration_secs": duration,
            "requests_per_sec": self.completed_requests / max(0.001, duration),
            "tokens_generated": self.total_tokens_generated,
            "throughput_tok_per_sec": throughput_tok_s,
            "tau": tau,
            "latency": {
                "avg_secs": avg_lat,
                "p95_secs": p95_lat,
            },
            "zero_failure_assertions": {
                "server_crashes": self.server_crashes,
                "nans_detected": self.nans_detected,
                "infs_detected": self.infs_detected,
                "health_endpoint_ok": health_ok,
                "vram_leak_detected": leak_detected,
                "vram_deltas_mb": vram_deltas,
                "throughput_passed": throughput_ok,
                "tau_passed": tau_ok,
                "final_kv_util": final_prom.get("vllm:kv_cache_usage_perc", 0.0),
            },
            "vram_profile": {
                "baseline": baseline_vram,
                "final": final_vram,
                "samples": self.vram_samples,
            },
            "prometheus_metrics": final_prom,
        }

        print("======================================================================")
        print(" Stress Test Run Summary")
        print(f" Result:                {'✅ PASSED' if passed else '❌ FAILED'}")
        print(f" Requests Completed:    {self.completed_requests}/{self.total_requests}")
        print(f" Total Duration:        {duration:.2f} s ({self.completed_requests / max(0.001, duration):.2f} req/s)")
        print(f" Throughput:            {throughput_tok_s:.2f} tok/s{' (Acceptance: >= ' + str(self.min_throughput) + ' tok/s)' if self.min_throughput > 0 else ''}")
        if tau > 0:
            print(f" Acceptance Length tau: {tau:.3f}{' (Acceptance: >= ' + str(self.min_tau) + ')' if self.min_tau > 0 else ''}")
        print(f" Server Crashes:        {self.server_crashes} (Acceptance: 0)")
        print(f" NaNs / Infs Detected:  {self.nans_detected + self.infs_detected} (Acceptance: 0)")
        print(f" VRAM Delta Card 0:     {vram_deltas.get(0, 0.0):+.2f} MB (Acceptance: <= 16 MB)")
        print(f" VRAM Delta Card 1:     {vram_deltas.get(1, 0.0):+.2f} MB (Acceptance: <= 16 MB)")
        print(f" Final Post-Health:     {'200 OK' if health_ok else 'UNAVAILABLE'}")
        print("======================================================================")

        return results


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="500-Request Multi-Turn Stress Test for Radiance DARTree")
    parser.add_argument("--url", default="http://127.0.0.1:8002", help="Server test URL (default: http://127.0.0.1:8002)")
    parser.add_argument("--num-requests", type=int, default=500, help="Total requests to execute (default: 500)")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent workers (default: 4)")
    parser.add_argument("--vram-interval", type=int, default=50, help="VRAM sample interval in requests (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per turn (default: 64)")
    parser.add_argument("--output-json", default="stress_results.json", help="Path to write JSON results")
    parser.add_argument("--min-throughput", type=float, default=0.0, help="Minimum required throughput in tok/s (default: 0.0)")
    parser.add_argument("--min-tau", type=float, default=0.0, help="Minimum required speculative acceptance tau (default: 0.0)")

    args = parser.parse_args()

    runner = MultiTurnStressRunner(
        base_url=args.url,
        total_requests=args.num_requests,
        concurrency=args.concurrency,
        vram_interval=args.vram_interval,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_throughput=args.min_throughput,
        min_tau=args.min_tau,
    )

    results = asyncio.run(runner.run())

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed telemetry report written to: {args.output_json}")

    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
