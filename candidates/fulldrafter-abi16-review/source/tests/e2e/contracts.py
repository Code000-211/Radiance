"""E2E Test Infrastructure: Contracts and Behavioral Specifications.

This module defines public interface contracts, schemas, dataclasses, and
invariant validators for Features 1 through 9 of the FullDrafter GPU Kernel
Optimization project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ==============================================================================
# Feature 1: Batched 128-bit Vectorized LM-Head Multi-GEMV Contract
# ==============================================================================
@dataclass(frozen=True)
class LMHeadGEMVContract:
    """Interface contract for r4d_fulldrafter_lmhead_gemv6.

    Computes Z_base = O * W_LM^T for 6 slots simultaneously using 128-bit vector
    loads (ushort8 / float4), streaming W_LM once per stage.
    """
    num_slots: int = 6
    hidden_dim: int = 2560
    vocab_shard: int = 124160
    vector_bits: int = 128
    vector_elements_bf16: int = 8
    block_dim_x: int = 256

    def validate_inputs(self, hidden_shape: tuple[int, ...], weight_shape: tuple[int, ...]) -> None:
        """Validate input tensor shapes against contract specifications."""
        if hidden_shape != (self.num_slots, self.hidden_dim):
            raise ValueError(
                f"Invalid hidden state shape: expected {(self.num_slots, self.hidden_dim)}, got {hidden_shape}"
            )
        if weight_shape != (self.vocab_shard, self.hidden_dim):
            raise ValueError(
                f"Invalid LM head weight shape: expected {(self.vocab_shard, self.hidden_dim)}, got {weight_shape}"
            )
        if self.hidden_dim % self.vector_elements_bf16 != 0:
            raise ValueError(
                f"Hidden dimension {self.hidden_dim} is not aligned to 128-bit vector width ({self.vector_elements_bf16} elements)"
            )

    def validate_output(self, output_shape: tuple[int, ...]) -> None:
        """Validate output logits shape."""
        if output_shape != (self.num_slots, self.vocab_shard):
            raise ValueError(
                f"Invalid output logits shape: expected {(self.num_slots, self.vocab_shard)}, got {output_shape}"
            )


# ==============================================================================
# Feature 2: LDS Markov Bigram Fusion & In-LDS Top-16 Reduction Contract
# ==============================================================================
@dataclass(frozen=True)
class MarkovSelectContract:
    """Interface contract for r4d_fulldrafter_slot_markov_select.

    Cooperative Top-16 reduction and rank-64 Markov bigram correction in LDS,
    eliminating 3856 B stack spill and merge kernel.
    """
    markov_rank: int = 64
    active_vocab: int = 248077
    total_vocab: int = 248320
    vocab_shard: int = 124160
    top_k: int = 16
    max_lds_scratchpad_bytes: int = 16384

    def validate_matrices(self, shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
        """Validate Markov factor matrices shapes."""
        if shape_a != (self.total_vocab, self.markov_rank):
            raise ValueError(
                f"Invalid Markov Matrix A shape: expected {(self.total_vocab, self.markov_rank)}, got {shape_a}"
            )
        if shape_b != (self.markov_rank, self.vocab_shard):
            raise ValueError(
                f"Invalid Markov Matrix B shape: expected {(self.markov_rank, self.vocab_shard)}, got {shape_b}"
            )

    def validate_masking(self, active_cols: int, tail_cols: int) -> None:
        """Validate active vocabulary masking: tail columns must be masked to -inf."""
        if active_cols + tail_cols != self.total_vocab:
            raise ValueError("Active columns plus tail columns must sum to total vocabulary size")
        if active_cols != self.active_vocab:
            raise ValueError(f"Active vocabulary count must be {self.active_vocab}, got {active_cols}")


# ==============================================================================
# Feature 3: Bitwise Deterministic Candidate Selection Contract
# ==============================================================================
@dataclass(frozen=True)
class DeterministicSelectionContract:
    """Contract for deterministic candidate tie-breaking and TP=2 equivalence.

    Tie-breaking rule: when two candidate logits are identical, the strictly
    smaller candidate index wins.
    """
    top_k: int = 16
    tp_ranks: int = 2

    def resolve_tie(self, score_a: float, id_a: int, score_b: float, id_b: int) -> int:
        """Deterministically breaks ties between two candidate tokens."""
        if math.isclose(score_a, score_b, abs_tol=1e-7):
            return min(id_a, id_b)
        return id_a if score_a > score_b else id_b


# ==============================================================================
# Feature 4: High-Occupancy Fused Transformer Backbone (H=64) Contract
# ==============================================================================
@dataclass(frozen=True)
class TransformerBackboneContract:
    """Interface contract for FullDrafter fused transformer blocks.

    Must use Head Dimension H=64 (12 Q heads, 4 KV heads = 768 hidden dimension).
    Operations requiring H=128 must be rejected.
    """
    hidden_size: int = 768
    num_q_heads: int = 12
    num_kv_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 2304
    num_stages: int = 2
    block_k: int = 6
    total_k: int = 12
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0

    def validate_head_dimension(self, candidate_head_dim: int) -> None:
        """Ensure head dimension is strictly 64; reject H=128."""
        if candidate_head_dim != self.head_dim:
            raise ValueError(
                f"Head dimension mismatch: FullDrafter requires H={self.head_dim}, received H={candidate_head_dim}. "
                f"H=128 is strictly prohibited."
            )


# ==============================================================================
# Feature 5: Asynchronous Stream Pipeline & Memory Stability Contract
# ==============================================================================
@dataclass(frozen=True)
class AsyncStreamContract:
    """Contract for non-blocking stream execution and memory stability.

    Prohibits hipDeviceSynchronize and hipStreamSynchronize during forward steps.
    Zero dynamic hipMalloc calls allowed in forward execution path.
    """
    required_gpu_headroom_mib: int = 384
    scratch_bytes: int = 1048576  # 1 MB scratch space
    allow_cpu_sync_in_step: bool = False
    allow_dynamic_malloc_in_step: bool = False

    def check_operation_safety(self, is_in_forward_step: bool, operation: str) -> None:
        """Enforces stream safety and allocation rules."""
        if is_in_forward_step:
            if operation in ("hipDeviceSynchronize", "hipStreamSynchronize"):
                raise RuntimeError(
                    f"CRITICAL SYNC VIOLATION: CPU barrier '{operation}' detected during asynchronous forward step!"
                )
            if operation in ("hipMalloc", "malloc"):
                raise RuntimeError(
                    f"CRITICAL ALLOC VIOLATION: Dynamic allocation '{operation}' detected during forward step!"
                )


# ==============================================================================
# Feature 6: Standalone GPU Latency <= 3.5ms Verification Contract
# ==============================================================================
@dataclass(frozen=True)
class LatencyContract:
    """Contract for FullDrafter forward pass execution latency."""
    max_latency_ms: float = 3.5
    measured_device: str = "Dual AMD Radeon AI PRO R9700 (gfx1201)"

    def check_latency(self, elapsed_ms: float) -> bool:
        """Checks if elapsed forward pass latency satisfies <= 3.5 ms threshold."""
        return elapsed_ms <= self.max_latency_ms


# ==============================================================================
# Feature 7: Bitwise Mathematical Parity Verification Contract
# ==============================================================================
@dataclass(frozen=True)
class BitwiseParityContract:
    """Contract for parity verification against reference benchmarks.

    Requires 100% token equivalence on drafter_parity_test (18 test cases)
    and drafter_gpu_exec_test cross-GPU equivalence.
    """
    parity_test_cases_count: int = 18
    required_token_equivalence_ratio: float = 1.0

    def check_cross_gpu_equivalence(self, gpu0_tokens: list[int], gpu1_tokens: list[int]) -> bool:
        """Verifies bitwise token equivalence across dual GPUs."""
        if len(gpu0_tokens) != 12 or len(gpu1_tokens) != 12:
            return False
        return gpu0_tokens == gpu1_tokens


# ==============================================================================
# Feature 8: Multi-Turn Live Serving Throughput >= 150 tok/s Contract
# ==============================================================================
@dataclass(frozen=True)
class ServingThroughputContract:
    """Contract for serving performance on test Port 8002."""
    test_port: int = 8002
    min_throughput_tok_per_sec: float = 150.0
    min_acceptance_length_tau: float = 3.2

    def check_throughput(self, tok_per_sec: float) -> bool:
        """Evaluates whether serving throughput meets or exceeds 150 tok/s."""
        return tok_per_sec >= self.min_throughput_tok_per_sec

    def check_tau(self, tau: float) -> bool:
        """Evaluates whether acceptance length meets or exceeds tau >= 3.2."""
        return tau >= self.min_acceptance_length_tau


# ==============================================================================
# Feature 9: Multi-Turn Stability & VRAM Leak Isolation Contract
# ==============================================================================
@dataclass(frozen=True)
class MultiTurnStabilityContract:
    """Contract for multi-turn stress test stability and VRAM leak isolation."""
    min_requests: int = 100
    max_allowed_crashes: int = 0
    max_allowed_nans: int = 0
    max_vram_delta_mib: float = 16.0
    quarantined_production_port: int = 8000
    test_port: int = 8002

    def validate_port(self, target_port: int) -> None:
        """Enforces strict isolation of production port 8000."""
        if target_port == self.quarantined_production_port:
            raise RuntimeError(
                f"SECURITY VIOLATION: Port {self.quarantined_production_port} is protected production. "
                f"Testing must only target port {self.test_port}."
            )

    def validate_stability(self, crashes: int, nans: int, vram_delta_mib: float) -> bool:
        """Verifies zero crashes, zero NaNs, and VRAM delta <= 16 MB."""
        return (
            crashes <= self.max_allowed_crashes
            and nans <= self.max_allowed_nans
            and vram_delta_mib <= self.max_vram_delta_mib
        )
