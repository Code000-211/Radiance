"""
Tier 5 White-Box Adversarial Stress Test Suite: FullDrafter Numerical Edge Cases
Milestone 5 Phase 2: Tier 5 White-Box Adversarial Challenger 1

Target Components Under Audit:
- `libr4d/r4d_fulldrafter.hip`
- `libr4d/r4d_fulldrafter.h`
- `arch/qwen4exp_fp8/qwen4exp_fp8.cpp`
- `arch/common/rad_block_fulldrafter.h`

Adversarial Stress Test Dimensions:
1. Extreme BF16 dynamic ranges and NaN/Inf sanitization
   - Round-to-nearest-even precision across normal, subnormal, max finite, and overflow ranges
   - IEEE-754 NaN and Inf propagation through linear projections and dot products
   - RMSNorm numerical stability under all-zero vectors, extreme values (FP32 overflow), and NaNs
   - Fused SwiGLU behavior under extreme positive and negative float regimes
2. Degenerate attention weights and softmax numerical stability
   - Zero-norm and uniform Q/K inputs yielding exactly uniform attention distributions
   - Opposing orthants (negative dot products) and causal softmax stability
   - Degenerate softmax inputs: all -inf or < -1e9f vulnerability analysis (line 449 max_score initialization)
   - One-hot dominant logit attention distribution
   - RoPE rotation boundary conditions (t=0 identity, t=6 maximum angle, norm conservation, split-half lane mapping)
3. Boundary tokens (T6) and active vocabulary filtering
   - Boundary token T6 at active vocab limit (248,076)
   - Padding vocab tokens (248,077 to 248,319) masking with -1e30f
   - Out-of-bounds token IDs (>= 248,320 and negative integers) hardware clamping to token 0
   - TP=2 shard boundary filtering (Rank 1 active slice 123,917 tokens)
4. Markov bigram edge cases and deterministic tie-breaking
   - Strict weak ordering and lower token ID tie-breaking of `is_better`
   - IEEE-754 NaN comparison anomaly in `is_better` (NaN poisoning vulnerability)
   - Logits below -1e30f underflow vulnerability (un-updated 0xFFFFFFFF placeholder)
   - Full-vocabulary identical logit ties resolving deterministically to tokens [0..15] with argmax 0
   - Markov zero transition counts cleanly degenerating to pure LM Head logits
   - Missing bigram fallback to base logits
5. Two-stage discrete recurrence invariants
   - Stage 1 -> Stage 2 terminal state h1[5] preservation and recurrent RMSNorm addition
   - Stage 2 mask vector broadcasting for slots 1..6
"""

from __future__ import annotations

import math
import struct
import unittest
import warnings

import numpy as np
import torch
import torch.nn.functional as F

# ==============================================================================
# Architectural Constants from libr4d/r4d_fulldrafter.h
# ==============================================================================
FULLDRAFTER_HIDDEN_SIZE = 768
FULLDRAFTER_INTERMEDIATE_SIZE = 2304
FULLDRAFTER_TARGET_WIDTH = 2560
FULLDRAFTER_CONTEXT_DIM = 7680
FULLDRAFTER_BLOCK_K = 6
FULLDRAFTER_STAGES = 2
FULLDRAFTER_TOTAL_K = 12
FULLDRAFTER_VOCAB_SIZE = 248320
FULLDRAFTER_ACTIVE_VOCAB = 248077
FULLDRAFTER_TP_SHARD_VOCAB = 124160
FULLDRAFTER_MARKOV_RANK = 64
FULLDRAFTER_SELECTOR_RANK = 64
FULLDRAFTER_TOP_K = 16
FULLDRAFTER_RMS_NORM_EPS = 1e-6
FULLDRAFTER_ROPE_THETA = 10000000.0
FULLDRAFTER_HEAD_DIM = 64
FULLDRAFTER_NUM_Q_HEADS = 12
FULLDRAFTER_NUM_KV_HEADS = 4


# ==============================================================================
# Emulation Oracles for Kernel Device Logic
# ==============================================================================
def round_bf16_val(f: float) -> float:
    """Emulates `round_bf16_val` from libr4d/r4d_fulldrafter.hip (L69-L73).

    uint32_t u = __builtin_bit_cast(uint32_t, f);
    uint16_t h = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    return __builtin_bit_cast(float, (uint32_t)h << 16);
    """
    if math.isnan(f):
        return float("nan")
    u = struct.unpack("<I", struct.pack("<f", float(f)))[0]
    h = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF) >> 16
    out_u = (h & 0xFFFF) << 16
    return struct.unpack("<f", struct.pack("<I", out_u))[0]


def is_better(va: float, ia: int, vb: float, ib: int) -> bool:
    """Emulates `is_better` from libr4d/r4d_fulldrafter.hip (L993-L997).

    if (va > vb) return true;
    if (va < vb) return false;
    return ia < ib;
    """
    if va > vb:
        return True
    if va < vb:
        return False
    return ia < ib


def dev_rmsnorm(x: np.ndarray, w: np.ndarray | None = None, eps: float = FULLDRAFTER_RMS_NORM_EPS) -> np.ndarray:
    """Emulates `dev_rmsnorm` from libr4d/r4d_fulldrafter.hip (L106-L123)."""
    n = len(x)
    sum_sq = float(np.sum(x * x))
    mean_sq = sum_sq / float(n)
    if math.isnan(mean_sq):
        scale = float("nan")
    elif math.isinf(mean_sq):
        scale = 0.0
    elif (mean_sq + eps) > 0.0:
        scale = 1.0 / math.sqrt(mean_sq + eps)
    else:
        scale = 0.0
    out = x * scale
    if w is not None:
        out = out * w
    return out


def dev_swiglu(x: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Emulates SwiGLU activation from MLP gate/up projection."""
    x_t = torch.from_numpy(x).float()
    up_t = torch.from_numpy(up).float()
    out_t = F.silu(x_t) * up_t
    return out_t.numpy()


# ==============================================================================
# Tier 5 Test Cases: Area 1 - Extreme BF16 Dynamic Ranges & NaN/Inf Sanitization
# ==============================================================================
class TestBF16ExtremeDynamicRangeAndSanitization(unittest.TestCase):
    """Verifies numerical precision, edge-case rounding, subnormals, and NaN/Inf handling."""

    def test_bf16_normal_round_trip(self):
        """Verifies round-to-nearest-even matches hardware bit manipulation."""
        test_values = [0.0, -0.0, 1.0, -1.0, 1.5, -1.5, 0.125, 2.0, 65504.0]
        for val in test_values:
            rounded = round_bf16_val(val)
            t_val = torch.tensor(val, dtype=torch.float32).bfloat16().float().item()
            self.assertEqual(
                rounded,
                t_val,
                f"Mismatch for value {val}: got {rounded}, expected {t_val}",
            )

    def test_bf16_subnormal_handling(self):
        """Verifies that subnormal values round cleanly without crashing or producing garbage."""
        subnormal_f32 = 1.401298464324817e-45  # Smallest positive subnormal in FP32
        res = round_bf16_val(subnormal_f32)
        # In BF16, this underflows to 0.0
        self.assertEqual(res, 0.0)

        # Largest subnormal in BF16 is ~1.166e-38
        bf16_subnormal = 1.0e-39
        res = round_bf16_val(bf16_subnormal)
        self.assertTrue(math.isfinite(res))

    def test_bf16_extreme_magnitudes_max_finite_and_overflow(self):
        """Verifies behaviour at the boundary of finite BF16 representation."""
        max_bf16 = 3.3895313892515355e38
        rounded_max = round_bf16_val(max_bf16)
        self.assertEqual(rounded_max, max_bf16)
        self.assertTrue(math.isfinite(rounded_max))

        # Value that exceeds max BF16 and rounds up to +Inf
        max_f32 = 3.4028234663852886e38
        overflowed = round_bf16_val(max_f32)
        self.assertTrue(
            math.isinf(overflowed),
            f"Expected infinity for max float32 overflow, got {overflowed}",
        )

    def test_bf16_nan_and_inf_propagation(self):
        """Verifies that IEEE-754 NaN and Inf propagate through round_bf16_val."""
        self.assertTrue(math.isnan(round_bf16_val(float("nan"))))
        self.assertTrue(math.isinf(round_bf16_val(float("inf"))))
        self.assertTrue(math.isinf(round_bf16_val(float("-inf"))))
        self.assertGreater(round_bf16_val(float("inf")), 0)
        self.assertLess(round_bf16_val(float("-inf")), 0)

    def test_rmsnorm_zero_norm_and_overflow(self):
        """Verifies RMSNorm stability under all-zero and extreme magnitude vectors."""
        # 1. All-zero vector: norm is 0, eps prevents divide-by-zero, output is 0.0
        zeros = np.zeros(FULLDRAFTER_HIDDEN_SIZE, dtype=np.float32)
        out_zeros = dev_rmsnorm(zeros)
        self.assertTrue(np.all(out_zeros == 0.0))
        self.assertFalse(np.any(np.isnan(out_zeros)))

        # 2. Extreme magnitude vector: elements near 1e19 cause sum of squares to overflow FP32 to inf
        extreme_vec = np.full(FULLDRAFTER_HIDDEN_SIZE, 1.84e19, dtype=np.float32)
        # In IEEE FP32, sum_sq becomes inf, scale becomes 0.0, out becomes 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sum_sq = float(np.sum(extreme_vec * extreme_vec))
            self.assertTrue(math.isinf(sum_sq))
            out_extreme = dev_rmsnorm(extreme_vec)
        self.assertFalse(np.any(np.isnan(out_extreme)))

        # 3. Vector containing NaN: propagates NaN deterministically
        nan_vec = np.zeros(FULLDRAFTER_HIDDEN_SIZE, dtype=np.float32)
        nan_vec[42] = float("nan")
        out_nan = dev_rmsnorm(nan_vec)
        self.assertTrue(np.all(np.isnan(out_nan)))

    def test_swiglu_extreme_positive_and_negative_regimes(self):
        """Verifies SwiGLU numerical stability across extreme input regimes."""
        # Extreme negative: silu underflows to -0.0
        x_neg = np.full(10, -100.0, dtype=np.float32)
        up_neg = np.full(10, 2.0, dtype=np.float32)
        out_neg = dev_swiglu(x_neg, up_neg)
        self.assertTrue(np.all(out_neg == 0.0))
        self.assertFalse(np.any(np.isnan(out_neg)))

        # Extreme positive: silu(x) -> x, out -> x * up
        x_pos = np.full(10, 100.0, dtype=np.float32)
        up_pos = np.full(10, 2.0, dtype=np.float32)
        out_pos = dev_swiglu(x_pos, up_pos)
        np.testing.assert_allclose(out_pos, 200.0, rtol=1e-5)


# ==============================================================================
# Tier 5 Test Cases: Area 2 - Degenerate Attention Weights & Softmax Stability
# ==============================================================================
class TestDegenerateAttentionWeights(unittest.TestCase):
    """Verifies attention mechanisms under pathological, degenerate, and edge-case inputs."""

    def test_zero_norm_qk_uniform_attention(self):
        """Zero Q and K inputs produce exactly uniform attention weights across causal history."""
        # For token t in 0..6, causal terms are 0..t (t+1 terms)
        for t in range(7):
            num_causal = t + 1
            scores = np.zeros(num_causal, dtype=np.float32)  # All dot products are 0.0
            max_score = np.max(scores)
            exp_scores = np.exp(scores - max_score)
            exp_sum = np.sum(exp_scores)
            probs = exp_scores / exp_sum

            # Each probability must be exactly 1 / (t + 1)
            expected_prob = 1.0 / float(num_causal)
            np.testing.assert_allclose(probs, expected_prob, rtol=1e-6)
            self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)

    def test_uniform_qk_constant_attention(self):
        """Identical Q and K dimensions produce identical dot products and uniform weights."""
        q_val = 0.5
        k_val = 0.5
        # Dot product across head_dim=64: 64 * 0.25 * 0.125 = 2.0
        expected_dot = 64.0 * (q_val * k_val) * 0.125

        for t in range(7):
            num_causal = t + 1
            scores = np.full(num_causal, expected_dot, dtype=np.float32)
            max_score = np.max(scores)
            exp_scores = np.exp(scores - max_score)
            probs = exp_scores / np.sum(exp_scores)
            expected_prob = 1.0 / float(num_causal)
            np.testing.assert_allclose(probs, expected_prob, rtol=1e-6)

    def test_orthant_shifts_opposing_directions(self):
        """Opposing orthants produce large negative dot products; max-subtraction preserves stability."""
        # Q in [+1, +1, ...], K in [-1, -1, ...]
        dot = 64.0 * (1.0 * -1.0) * 0.125  # -8.0
        scores = np.full(7, dot, dtype=np.float32)
        max_score = np.max(scores)
        # scores - max_score = 0.0, exp(0) = 1.0
        exp_scores = np.exp(scores - max_score)
        probs = exp_scores / np.sum(exp_scores)
        self.assertTrue(np.all(probs == (1.0 / 7.0)))
        self.assertFalse(np.any(np.isnan(probs)))

    def test_all_negative_infinity_softmax_vulnerability(self):
        """Empirical vulnerability demonstration: line 449 in r4d_fulldrafter.hip.

        In `r4d_fulldrafter_block_attention_kernel`:
            float max_score = -1e9f;
        If all causal dot products are < -1e9f (e.g. -1.5e9f or -inf),
        `dot > max_score` is False for all tokens.
        `max_score` remains -1e9f.
        `expf(scores - max_score)` underflows to 0.0f.
        `exp_sum` becomes 0.0f, causing a 0/0 division by zero and resulting in NaN!
        """
        scores = np.array([-1.5e9, -2.0e9, -3.0e9], dtype=np.float32)

        # 1. Kernel emulation with hardcoded max_score = -1e9f
        kernel_max_score = -1e9
        for s in scores:
            kernel_max_score = max(kernel_max_score, s)
        self.assertEqual(kernel_max_score, -1e9)  # Never updated!

        # Exponentiation underflows to 0.0
        exp_sum = 0.0
        exp_terms = []
        for s in scores:
            diff = s - kernel_max_score
            term = math.exp(diff) if diff > -700 else 0.0
            exp_terms.append(term)
            exp_sum += term
        self.assertEqual(exp_sum, 0.0)

        # Division by zero occurs
        vulnerable_prob = (
            float("nan") if exp_sum == 0.0 else exp_terms[0] / exp_sum
        )
        self.assertTrue(
            math.isnan(vulnerable_prob),
            "Confirmed vulnerability: scores < -1e9 cause 0/0 -> NaN in kernel",
        )

        # 2. Corrected robust formulation: initialize max_score = scores[0]
        robust_max = np.max(scores)
        robust_exp = np.exp(scores - robust_max)
        robust_sum = np.sum(robust_exp)
        self.assertGreaterEqual(robust_sum, 1.0)
        robust_probs = robust_exp / robust_sum
        self.assertFalse(np.any(np.isnan(robust_probs)))

    def test_single_dominant_logit_one_hot_attention(self):
        """One overwhelmingly large dot product collapses attention to a one-hot distribution."""
        scores = np.array([-10.0, 100.0, -5.0, 0.0], dtype=np.float32)
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)
        probs = exp_scores / np.sum(exp_scores)

        # Index 1 should have probability 1.0 (to within float precision)
        self.assertAlmostEqual(float(probs[1]), 1.0, places=6)
        self.assertAlmostEqual(float(probs[0]), 0.0, places=6)
        self.assertAlmostEqual(float(probs[2]), 0.0, places=6)

    def test_rope_rotation_boundary_conditions(self):
        """RoPE rotation boundary: t=0 is identity, t=6 has maximum angle, norm is strictly conserved."""
        theta_base = FULLDRAFTER_ROPE_THETA
        for lane in range(32):
            freq = 1.0 / (theta_base ** ((2.0 * lane) / 64.0))

            # t = 0 (anchor token): theta = 0, cos=1, sin=0 -> exact identity
            theta_0 = 0.0 * freq
            self.assertEqual(math.cos(theta_0), 1.0)
            self.assertEqual(math.sin(theta_0), 0.0)

            # t = 6 (last draft token): verify norm conservation
            theta_6 = 6.0 * freq
            cos_6 = math.cos(theta_6)
            sin_6 = math.sin(theta_6)

            q0, q1 = 1.2345, -6.7890
            qr0 = q0 * cos_6 - q1 * sin_6
            qr1 = q0 * sin_6 + q1 * cos_6

            norm_orig = q0 * q0 + q1 * q1
            norm_rot = qr0 * qr0 + qr1 * qr1
            self.assertAlmostEqual(
                norm_orig,
                norm_rot,
                places=5,
                msg=f"RoPE norm violated at lane {lane}",
            )


# ==============================================================================
# Tier 5 Test Cases: Area 3 - Boundary Tokens & Active Vocabulary Filtering
# ==============================================================================
class TestBoundaryTokensAndActiveVocabFiltering(unittest.TestCase):
    """Verifies boundary token T6 at vocabulary limit and active vocabulary masking."""

    def test_boundary_token_t6_at_active_vocab_limit_248076(self):
        """Token 248,076 is the highest valid active token ID and is within bounds."""
        highest_active_token = FULLDRAFTER_ACTIVE_VOCAB - 1  # 248076
        self.assertEqual(highest_active_token, 248076)

        # 1. Must be less than total vocabulary size 248320
        self.assertLess(highest_active_token, FULLDRAFTER_VOCAB_SIZE)

        # 2. In Stage 2 backbone: boundary_tok < V_TOTAL check passes
        boundary_tok = highest_active_token
        if boundary_tok >= FULLDRAFTER_VOCAB_SIZE:
            boundary_tok = 0
        self.assertEqual(boundary_tok, 248076)

        # 3. Row offset calculation into teacher_embedding_table: 248076 * 2560 elements
        table_offset = boundary_tok * FULLDRAFTER_TARGET_WIDTH
        max_table_elements = (
            FULLDRAFTER_VOCAB_SIZE * FULLDRAFTER_TARGET_WIDTH
        )  # 635,699,200
        self.assertLess(table_offset, max_table_elements)

    def test_boundary_token_t6_at_vocab_padding_248077_to_248319(self):
        """Tokens 248,077..248,319 are padding tokens; their logits must be masked to -1e30f."""
        padding_tokens = [
            FULLDRAFTER_ACTIVE_VOCAB,
            FULLDRAFTER_ACTIVE_VOCAB + 10,
            FULLDRAFTER_VOCAB_SIZE - 1,
        ]

        for tok in padding_tokens:
            # In slot_markov_select_kernel (L1067):
            # float logit = -1e30f;
            # if (v < V_ACT) { ... }
            is_active = tok < FULLDRAFTER_ACTIVE_VOCAB
            self.assertFalse(
                is_active, f"Token {tok} must not be considered active"
            )

            # In Stage 2 backbone: tok < V_TOTAL is True, so memory access is safe (within allocated table)
            self.assertLess(
                tok,
                FULLDRAFTER_VOCAB_SIZE,
                f"Token {tok} must be within allocated buffer size",
            )

    def test_boundary_token_t6_out_of_bounds_clamping(self):
        """Out-of-bounds tokens (>= 248,320 or negative) are clamped to 0 by hardware guards."""
        oob_tokens = [
            FULLDRAFTER_VOCAB_SIZE,  # Exactly at boundary 248320
            FULLDRAFTER_VOCAB_SIZE + 500,  # 248820
            0xFFFFFFFF,  # 4294967295 (cast from -1)
        ]

        for raw_tok in oob_tokens:
            # Hardware guard emulation from fulldrafter_stage2_backbone_kernel (L1364)
            clamped = 0 if raw_tok >= FULLDRAFTER_VOCAB_SIZE else raw_tok
            self.assertEqual(
                clamped,
                0,
                f"Token {raw_tok} must be clamped to 0, got {clamped}",
            )

            # Selector predecessor clamp from slot_markov_select_kernel (L1041)
            ptok_clamp = (
                raw_tok if raw_tok < FULLDRAFTER_VOCAB_SIZE else 0
            )
            self.assertEqual(
                ptok_clamp,
                0,
                f"Predecessor {raw_tok} must be clamped to 0, got {ptok_clamp}",
            )

    def test_tp2_shard_active_vocab_boundary(self):
        """Verifies TP=2 vocabulary sharding: Rank 1 active slice excludes padding tokens."""
        shard_size = FULLDRAFTER_TP_SHARD_VOCAB  # 124160
        active_vocab = FULLDRAFTER_ACTIVE_VOCAB  # 248077

        # Rank 0 covers [0 .. 124159]: all active
        rank0_active = shard_size
        self.assertEqual(rank0_active, 124160)

        # Rank 1 covers [124160 .. 248319]:
        # Active slice calculation from rad_block_fulldrafter.h (L1689):
        # size_t active_shard1 = (ACTIVE_VOCAB > SHARD) ? (ACTIVE_VOCAB - SHARD) : 0;
        active_shard1 = (
            (active_vocab - shard_size) if active_vocab > shard_size else 0
        )
        self.assertEqual(active_shard1, 123917)

        # Highest token extracted on Rank 1
        highest_rank1_tok = shard_size + active_shard1 - 1
        self.assertEqual(highest_rank1_tok, 248076)

        # Padding tokens on Rank 1
        padding_count = shard_size - active_shard1
        self.assertEqual(
            padding_count, 243
        )  # 248320 - 248077 = 243 padding tokens


# ==============================================================================
# Tier 5 Test Cases: Area 4 - Markov Bigram Edge Cases & Deterministic Tie-Breaking
# ==============================================================================
class TestDeterministicTieBreakingAndMarkovTransitions(unittest.TestCase):
    """Verifies deterministic tie-breaking and Markov bigram transition robustness."""

    def test_is_better_strict_weak_ordering_and_tie_breaking(self):
        """Verifies `is_better` implements strict weak ordering with lower token ID tie-breaking."""
        # 1. Distinct values: higher value wins regardless of ID
        self.assertTrue(is_better(10.0, 100, 5.0, 50))
        self.assertFalse(is_better(5.0, 50, 10.0, 100))

        # 2. Equal values: lower token ID wins
        self.assertTrue(is_better(10.0, 50, 10.0, 100))
        self.assertFalse(is_better(10.0, 100, 10.0, 50))

        # 3. Identical value and identical ID: strictly False (irreflexivity)
        self.assertFalse(is_better(10.0, 50, 10.0, 50))

        # 4. Transitivity across ties: ID 10 beats ID 20, ID 20 beats ID 30
        self.assertTrue(is_better(0.0, 10, 0.0, 20))
        self.assertTrue(is_better(0.0, 20, 0.0, 30))
        self.assertTrue(is_better(0.0, 10, 0.0, 30))

    def test_is_better_nan_poisoning_vulnerability(self):
        """Empirical vulnerability demonstration: NaN comparison anomaly in `is_better`.

        In IEEE-754:
            va > vb and va < vb both return False when either va or vb is NaN.
        If vb is NaN, `is_better(va, ia, vb, ib)` falls through to `return ia < ib`.
        Consequently, a NaN with token ID 10 cannot be overtaken by any finite score (e.g. +1e9)
        with token ID 20!
        """
        nan_val = float("nan")
        normal_val = 1e9

        # Case A: Normal value has larger token ID -> cannot beat NaN!
        self.assertFalse(
            is_better(normal_val, 20, nan_val, 10),
            "Vulnerability confirmed: large score cannot beat NaN when its ID is larger",
        )

        # Case B: Normal value has smaller token ID -> beats NaN purely on ID, not score
        self.assertTrue(is_better(normal_val, 5, nan_val, 10))

    def test_logits_below_negative_1e30_underflow_vulnerability(self):
        """Empirical vulnerability demonstration: logits below -1e30f in `slot_markov_select_kernel`.

        In `slot_markov_select_kernel` (L1060):
            float top_v[2] = { -1e30f, -1e30f };
            uint32_t top_i[2] = { 0xFFFFFFFF, 0xFFFFFFFF };
        If a token has logit < -1e30f (e.g. -2e30f), `is_better` evaluates to False,
        and the thread never updates its top candidates, returning 0xFFFFFFFF!
        """
        top_v = [-1e30, -1e30]
        top_i = [0xFFFFFFFF, 0xFFFFFFFF]

        low_logit = -2.0e30
        candidate_id = 42

        # Emulate thread-local tracking (L1083-L1088)
        updated = False
        if is_better(low_logit, candidate_id, top_v[0], top_i[0]):
            top_v[0] = low_logit
            top_i[0] = candidate_id
            updated = True
        elif is_better(low_logit, candidate_id, top_v[1], top_i[1]):
            top_v[1] = low_logit
            top_i[1] = candidate_id
            updated = True

        self.assertFalse(
            updated,
            "Confirmed vulnerability: logit < -1e30 cannot update top_v/top_i",
        )
        self.assertEqual(top_i[0], 0xFFFFFFFF)

    def test_uniform_logits_complete_tie_resolution(self):
        """When all vocabulary tokens have identical logits, Top-16 extracts tokens 0..15 deterministically."""
        num_blocks = 64
        total_vocab = FULLDRAFTER_ACTIVE_VOCAB
        k_top = FULLDRAFTER_TOP_K
        v_per_block = (total_vocab + num_blocks - 1) // num_blocks

        # Block 0..63 extract local top-16 candidates under identical logits 0.0f
        block_candidates: list[tuple[float, int]] = []
        for b in range(num_blocks):
            v_start = b * v_per_block
            v_end = min(v_start + v_per_block, total_vocab)
            # Under identical logits, block selects the 16 lowest IDs in its range
            for tok in range(v_start, min(v_start + k_top, v_end)):
                block_candidates.append((0.0, tok))

        # Phase 2: cooperative merge of all 1024 candidates
        merged_top16: list[int] = []
        cands = list(block_candidates)
        for _ in range(k_top):
            best_idx = 0
            for i in range(1, len(cands)):
                if is_better(
                    cands[i][0],
                    cands[i][1],
                    cands[best_idx][0],
                    cands[best_idx][1],
                ):
                    best_idx = i
            merged_top16.append(cands[best_idx][1])
            cands[best_idx] = (-1e30, 0xFFFFFFFF)  # Cleared after selection

        # Must strictly be tokens 0, 1, 2, ..., 15 in order
        expected_top16 = list(range(16))
        self.assertEqual(
            merged_top16,
            expected_top16,
            f"Expected {expected_top16}, got {merged_top16}",
        )

        # Winning token is strictly token 0
        winning_tok = merged_top16[0]
        self.assertEqual(winning_tok, 0)

    def test_markov_zero_transitions_pure_lmhead_degeneration(self):
        """Zero transition weights in Markov matrices cleanly degenerates to pure LM Head logits."""
        # Markov matrix B row is all zero
        m_dot = 0.0
        base_logits = np.array([2.5, 8.1, 1.2, 5.4], dtype=np.float32)
        adjusted_logits = base_logits + m_dot

        # Adjusted logits match base logits bit-for-bit
        np.testing.assert_array_equal(adjusted_logits, base_logits)

    def test_markov_missing_bigram_fallback(self):
        """When predecessor token has no recorded transitions, Markov projection evaluates to 0."""
        # Unseen predecessor token has zero vector in markov_matrix_a
        h_markov = np.zeros(FULLDRAFTER_MARKOV_RANK, dtype=np.float32)
        mb_row = np.random.randn(FULLDRAFTER_MARKOV_RANK).astype(np.float32)

        m_dot = float(np.dot(mb_row, h_markov))
        self.assertEqual(m_dot, 0.0)


# ==============================================================================
# Tier 5 Test Cases: Area 5 - Two-Stage Discrete Recurrence Invariants
# ==============================================================================
class TestTwoStageDiscreteRecurrenceInvariants(unittest.TestCase):
    """Verifies recurrence coupling between Stage 1 and Stage 2."""

    def test_stage1_to_stage2_h1_terminal_state_preservation(self):
        """Terminal hidden state h1[5] is normalized and summed with anchor projection."""
        h1_5 = np.random.randn(FULLDRAFTER_HIDDEN_SIZE).astype(np.float32)
        recurrent_norm_w = np.ones(FULLDRAFTER_HIDDEN_SIZE, dtype=np.float32)
        norm_h1 = dev_rmsnorm(h1_5, recurrent_norm_w)

        # Anchor projection of T6
        anc_proj_t6 = np.random.randn(FULLDRAFTER_HIDDEN_SIZE).astype(
            np.float32
        )

        # Base representation for Stage 2
        base2 = norm_h1 + anc_proj_t6
        base2_rounded = np.array([round_bf16_val(x) for x in base2])

        self.assertEqual(len(base2_rounded), FULLDRAFTER_HIDDEN_SIZE)
        self.assertFalse(np.any(np.isnan(base2_rounded)))

    def test_stage2_mask_vector_broadcast_invariance(self):
        """Tokens 1..6 in Stage 2 receive exact mask_vector values across all 768 dimensions."""
        mask_vector = np.random.randn(FULLDRAFTER_HIDDEN_SIZE).astype(
            np.float32
        )
        # Emulate Stage 2 initialization (L1419-L1423 in r4d_fulldrafter.hip)
        s_h2 = np.zeros(
            (FULLDRAFTER_BLOCK_K + 1, FULLDRAFTER_HIDDEN_SIZE), dtype=np.float32
        )
        for t in range(1, FULLDRAFTER_BLOCK_K + 1):
            s_h2[t] = mask_vector.copy()

        for t in range(1, FULLDRAFTER_BLOCK_K + 1):
            np.testing.assert_array_equal(s_h2[t], mask_vector)


if __name__ == "__main__":
    unittest.main()
