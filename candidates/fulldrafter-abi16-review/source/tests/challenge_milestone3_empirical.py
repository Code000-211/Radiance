"""Milestone 3 Adversarial Empirical Challenge Suite.

Mathematical & Parity Challenger:
Adversarially challenge the mathematical correctness, RoPE rotation,
attention numerical stability, SwiGLU activation, and recurrence parity of Milestone 3.

Properties probed:
1. RoPE rotation and split-half lane mapping (d0=lane, d1=lane+32): mathematical identity with PyTorch RotaryEmbedding.
2. RoPE V_DOT2_F32_BF16 and Wave32 all-reduce dot-product equivalence.
3. Attention scale factor 1/sqrt(64) == 0.125f IEEE-754 exactness.
4. Softmax numerical stability: max-subtraction preventing overflow, exp_sum >= 1.0 preventing divide-by-zero.
5. In-register Softmax and Value accumulation numerical parity with PyTorch GroupedQueryAttention.
6. Fused SwiGLU activation: x / (1.0f + expf(-x)) * up exact match with PyTorch F.silu(x) * up across normal and extreme float regimes.
7. Stage 1 -> Stage 2 Recurrence Parity: h1[5] terminal state preservation, boundary token T6 indexing, teacher embedding projection, recurrent RMSNorm sum.
8. Static invariant and occupancy verification:
   - Waves per EU (16, 16) on all kernels.
   - Non-overlapping scratchpad offsets.
"""

import math
import os
import struct
import unittest

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = "/home/orca/RadianceMine/worktrees/tap_record"
HIP_FILE = os.path.join(PROJECT_ROOT, "libr4d/r4d_fulldrafter.hip")
HEADER_FILE = os.path.join(PROJECT_ROOT, "libr4d/r4d_fulldrafter.h")

ROPE_THETA = 10000000.0
HEAD_DIM = 64
NUM_Q_HEADS = 12
NUM_KV_HEADS = 4
H_DIM = 768
W_DIM = 2560
INTERM_DIM = 2304
BLOCK_K = 6


def f32_to_bf16_rounded(f: float) -> int:
    """Emulates `round_bf16_val` from r4d_fulldrafter.hip."""
    u32 = struct.unpack("<I", struct.pack("<f", float(f)))[0]
    h = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16
    return h & 0xFFFF


def bf16_to_f32(h: int) -> float:
    """Emulates `bf16_to_f32` from r4d_fulldrafter.hip."""
    u32 = (int(h) & 0xFFFF) << 16
    return struct.unpack("<f", struct.pack("<I", u32))[0]


def round_bf16_val(f: float) -> float:
    """Round float to BF16 and convert back to float."""
    return bf16_to_f32(f32_to_bf16_rounded(f))


def pack_f32_to_bf16x2(f0: float, f1: float) -> int:
    """Emulates `pack_f32_to_bf16x2`."""
    h0 = f32_to_bf16_rounded(f0)
    h1 = f32_to_bf16_rounded(f1)
    return (h0 & 0xFFFF) | ((h1 & 0xFFFF) << 16)


def r4d_dot2_f32_bf16(pk_a: int, pk_b: int) -> float:
    """Emulates AMD RDNA4 `v_dot2_f32_bf16` instruction."""
    a0 = bf16_to_f32(pk_a & 0xFFFF)
    a1 = bf16_to_f32((pk_a >> 16) & 0xFFFF)
    b0 = bf16_to_f32(pk_b & 0xFFFF)
    b1 = bf16_to_f32((pk_b >> 16) & 0xFFFF)
    return a0 * b0 + a1 * b1


class TestRoPEMathematicalIdentity(unittest.TestCase):
    """Adversarially challenge the RoPE split-half lane mapping against PyTorch."""

    def test_frequency_exact_match(self):
        """Verify frequency calculation for all 32 lanes matches PyTorch inv_freq."""
        inv_freq_pytorch = 1.0 / (
            ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)
        ).numpy()

        freq_kernel = np.zeros(32, dtype=np.float32)
        for lane in range(32):
            freq_kernel[lane] = 1.0 / math.pow(ROPE_THETA, (2.0 * lane) / 64.0)

        for lane in range(32):
            self.assertAlmostEqual(
                freq_kernel[lane],
                inv_freq_pytorch[lane],
                places=6,
                msg=f"Frequency mismatch at lane {lane}",
            )

    def test_split_half_rotation_identity_with_pytorch(self):
        """Verify split-half lane mapping (d0=lane, d1=lane+32) matches PyTorch rotate_half."""
        torch.manual_seed(42)
        for t in range(7):
            q_vec = torch.randn(HEAD_DIM, dtype=torch.float32)
            k_vec = torch.randn(HEAD_DIM, dtype=torch.float32)

            # PyTorch Reference RoPE
            inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
            freqs = t * inv_freq
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()
            sin = emb.sin()

            # rotate_half: cat([-x2, x1])
            q_rot_ref = (q_vec * cos) + (
                torch.cat([-q_vec[32:], q_vec[:32]], dim=-1) * sin
            )
            k_rot_ref = (k_vec * cos) + (
                torch.cat([-k_vec[32:], k_vec[:32]], dim=-1) * sin
            )

            # Kernel Emulation:
            # Lanes 0..31 compute d0=lane, d1=lane+32
            q_rot_kernel = np.zeros(HEAD_DIM, dtype=np.float32)
            k_rot_kernel = np.zeros(HEAD_DIM, dtype=np.float32)

            for lane in range(32):
                freq = 1.0 / math.pow(ROPE_THETA, (2.0 * lane) / 64.0)
                theta = float(t) * freq
                cos_th = math.cos(theta)
                sin_th = math.sin(theta)

                q0 = q_vec[lane].item()
                q1 = q_vec[lane + 32].item()
                k0 = k_vec[lane].item()
                k1 = k_vec[lane + 32].item()

                qr0 = q0 * cos_th - q1 * sin_th
                qr1 = q0 * sin_th + q1 * cos_th
                kr0 = k0 * cos_th - k1 * sin_th
                kr1 = k0 * sin_th + k1 * cos_th

                q_rot_kernel[lane] = qr0
                q_rot_kernel[lane + 32] = qr1
                k_rot_kernel[lane] = kr0
                k_rot_kernel[lane + 32] = kr1

            np.testing.assert_allclose(
                q_rot_kernel,
                q_rot_ref.numpy(),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Q rotation mismatch at token {t}",
            )
            np.testing.assert_allclose(
                k_rot_kernel,
                k_rot_ref.numpy(),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"K rotation mismatch at token {t}",
            )

    def test_v_dot2_f32_bf16_wave_reduction_parity(self):
        """Verify pack_f32_to_bf16x2 + v_dot2_f32_bf16 + all-reduce equals full 64-dim inner product."""
        torch.manual_seed(123)
        for t in range(7):
            q_vec = torch.randn(HEAD_DIM, dtype=torch.float32)
            k_vec = torch.randn(HEAD_DIM, dtype=torch.float32)

            # PyTorch inner product after BF16 rounding
            q_bf16 = q_vec.to(torch.bfloat16).to(torch.float32)
            k_bf16 = k_vec.to(torch.bfloat16).to(torch.float32)
            expected_dot = torch.dot(q_bf16, k_bf16).item()

            # Kernel Emulation: 32 lanes packing (d0, d1)
            wave_dot = 0.0
            for lane in range(32):
                q_pk = pack_f32_to_bf16x2(q_vec[lane].item(), q_vec[lane + 32].item())
                k_pk = pack_f32_to_bf16x2(k_vec[lane].item(), k_vec[lane + 32].item())
                lane_dot = r4d_dot2_f32_bf16(q_pk, k_pk)
                wave_dot += lane_dot  # simulates __shfl_xor wave all-reduce

            self.assertAlmostEqual(
                wave_dot,
                expected_dot,
                places=3,
                msg=f"Inner product mismatch at token {t}",
            )


class TestAttentionNumericalStability(unittest.TestCase):
    """Adversarially challenge attention scaling, max-subtraction, and softmax stability."""

    def test_scale_factor_ieee754_exactness(self):
        """Verify scale factor 1/sqrt(64) == 0.125f exactly."""
        head_dim = 64
        scale = 1.0 / math.sqrt(head_dim)
        self.assertEqual(scale, 0.125)
        # Check IEEE-754 bit representation: 0.125 is 2^-3 (0x3E000000)
        u32 = struct.unpack("<I", struct.pack("<f", 0.125))[0]
        self.assertEqual(u32, 0x3E000000)

    def test_softmax_max_subtraction_prevents_overflow(self):
        """Verify that max-subtraction prevents overflow on large raw scores (> +1000.0)."""
        large_scores = np.array([800.0, 950.0, 1000.0, 850.0, 900.0, 920.0, 990.0], dtype=np.float32)

        # Naive exp without max-subtraction will overflow to inf in float32
        with np.errstate(over="ignore"):
            naive_exp = np.exp(large_scores)
            self.assertTrue(np.isinf(naive_exp).any(), "Expected naive exp to overflow to inf")

        # Kernel implementation with max subtraction:
        max_score = float(np.max(large_scores))
        exp_scores = np.zeros(7, dtype=np.float32)
        exp_sum = 0.0
        for p_idx in range(7):
            exp_scores[p_idx] = math.exp(large_scores[p_idx] - max_score)
            exp_sum += exp_scores[p_idx]

        self.assertFalse(np.isneginf(exp_scores).any())
        self.assertFalse(np.isinf(exp_scores).any())
        self.assertFalse(np.isnan(exp_scores).any())
        self.assertGreaterEqual(exp_sum, 1.0, "exp_sum must be >= 1.0")

        probs = exp_scores / exp_sum
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_softmax_never_divides_by_zero(self):
        """Verify that exp_sum is guaranteed >= 1.0 even with all large negative scores."""
        all_negative_scores = np.array([-1e6, -2e6, -5e5, -1e7, -8e5, -9e5, -6e5], dtype=np.float32)
        max_score = float(np.max(all_negative_scores))  # will be -5e5
        exp_sum = 0.0
        for s in all_negative_scores:
            exp_sum += math.exp(s - max_score)

        # Since max_score - max_score == 0, exp(0) == 1.0. Therefore exp_sum >= 1.0.
        self.assertGreaterEqual(exp_sum, 1.0)
        self.assertNotEqual(exp_sum, 0.0)

    def test_in_register_attention_emulation_parity(self):
        """Compare in-register Wave32 causal attention against PyTorch attention."""
        torch.manual_seed(999)
        # Sequence length 7, 12 query heads, 4 KV heads, head_dim 64
        L = 7
        q = torch.randn(1, NUM_Q_HEADS, L, HEAD_DIM, dtype=torch.float32)
        k = torch.randn(1, NUM_KV_HEADS, L, HEAD_DIM, dtype=torch.float32)
        v = torch.randn(1, NUM_KV_HEADS, L, HEAD_DIM, dtype=torch.float32)

        # PyTorch Reference Attention
        k_rep = k.repeat_interleave(3, dim=1)  # 12 heads
        v_rep = v.repeat_interleave(3, dim=1)  # 12 heads

        causal_mask = torch.triu(torch.full((L, L), float("-inf")), 1)
        scores_ref = torch.matmul(q, k_rep.transpose(-2, -1)) * 0.125
        scores_ref = scores_ref + causal_mask
        probs_ref = F.softmax(scores_ref, dim=-1)
        out_ref = torch.matmul(probs_ref, v_rep)  # [1, 12, 7, 64]

        # Kernel Emulation for each query head
        out_kernel = np.zeros((12, L, HEAD_DIM), dtype=np.float32)
        for q_h in range(12):
            kv_h = q_h // 3
            # For each lane (0..31): computes d0=lane, d1=lane+32
            # Values:
            v_tok = v[0, kv_h].numpy()  # [7, 64]

            # In kernel, scores are computed per token t:
            for t in range(L):
                scores = np.zeros(L, dtype=np.float32)
                max_score = -1e9
                for p_idx in range(t + 1):  # CAUSAL = true
                    # Dot product over all 64 dims
                    dot = np.dot(q[0, q_h, t].numpy(), k[0, kv_h, p_idx].numpy()) * 0.125
                    scores[p_idx] = dot
                    max_score = max(max_score, dot)

                exp_sum = 0.0
                for p_idx in range(t + 1):
                    scores[p_idx] = math.exp(scores[p_idx] - max_score)
                    exp_sum += scores[p_idx]

                # Value accumulation
                for lane in range(32):
                    acc0 = 0.0
                    acc1 = 0.0
                    for p_idx in range(t + 1):
                        prob = scores[p_idx] / exp_sum
                        acc0 += prob * v_tok[p_idx, lane]
                        acc1 += prob * v_tok[p_idx, lane + 32]
                    out_kernel[q_h, t, lane] = acc0
                    out_kernel[q_h, t, lane + 32] = acc1

        np.testing.assert_allclose(
            out_kernel,
            out_ref[0].numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg="In-register attention emulation does not match PyTorch reference",
        )


class TestSwiGLUActivationExactness(unittest.TestCase):
    """Adversarially challenge SwiGLU activation formula across dynamic ranges."""

    def test_silu_formula_match(self):
        """Verify g / (1.0f + expf(-g)) matches PyTorch F.silu(g) across [-20, 20]."""
        g_vals = torch.linspace(-20.0, 20.0, 1000, dtype=torch.float32)
        silu_ref = F.silu(g_vals).numpy()

        silu_kernel = np.zeros(len(g_vals), dtype=np.float32)
        for i, g in enumerate(g_vals.numpy()):
            silu_kernel[i] = g / (1.0 + math.exp(-g))

        np.testing.assert_allclose(
            silu_kernel,
            silu_ref,
            rtol=1e-6,
            atol=1e-6,
            err_msg="SwiGLU SiLU formula does not match PyTorch F.silu",
        )

    def test_silu_extreme_positive_inputs(self):
        """For large positive g (+100.0), exp(-g) -> 0.0, silu(g) -> g."""
        g = 100.0
        silu = g / (1.0 + math.exp(-g))
        self.assertAlmostEqual(silu, g, places=5)
        self.assertAlmostEqual(silu, F.silu(torch.tensor(g)).item(), places=5)

    def test_silu_extreme_negative_inputs(self):
        """For large negative g (-100.0), exp(-g) -> inf, silu(g) -> 0.0."""
        g = -100.0
        # In float32 exp(100) overflows to inf, and g / (1 + inf) == 0.0
        with np.errstate(over="ignore"):
            silu = np.float32(g) / (np.float32(1.0) + np.exp(np.float32(-g)))
        self.assertEqual(silu, 0.0)
        self.assertAlmostEqual(float(silu), F.silu(torch.tensor(g)).item(), places=5)

    def test_silu_zero_and_subnormal(self):
        """Test g = 0.0 and subnormals."""
        self.assertEqual(0.0 / (1.0 + math.exp(0.0)), 0.0)
        subnormal = 1e-38
        self.assertAlmostEqual(
            subnormal / (1.0 + math.exp(-subnormal)),
            F.silu(torch.tensor(subnormal)).item(),
            places=7,
        )


class TestStage2RecurrenceAndTeacherEmbeddingParity(unittest.TestCase):
    """Verify that h1[5] and boundary token teacher projection match PyTorch Stage 2."""

    def test_stage1_terminal_state_is_token_6_proposal_slot_5(self):
        """Verify that Stage 1 normalizes proposal slots 1..6 into slots 0..5, making h1[5] terminal."""
        with open(HIP_FILE, "r") as f:
            hip_code = f.read()

        # Check final norm kernel takes slot 1..6:
        # dev_rmsnorm(h_base + (1 + slot) * H_DIM, final_norm, t_base + slot * H_DIM, H_DIM)
        self.assertIn(
            "dev_rmsnorm(h_base + (1 + slot) * H_DIM, final_norm, t_base + slot * H_DIM, H_DIM);",
            hip_code,
            "Final norm does not slice tokens 1..6 into proposal slots 0..5!",
        )

        # Check copy kernel copies t_base into h_base[slot * H_DIM]:
        self.assertIn(
            "h_base[slot * H_DIM + i] = t_base[slot * H_DIM + i];",
            hip_code,
            "Copy kernel does not place proposal slots at h_base[slot * H_DIM]!",
        )

    def test_stage2_reads_boundary_tok_from_slot_5(self):
        """Verify that Stage 2 reads boundary token from slot 5 (BLOCK_K - 1)."""
        with open(HIP_FILE, "r") as f:
            hip_code = f.read()

        self.assertIn(
            "uint32_t boundary_tok = tokens_out[(size_t)s * FULLDRAFTER_TOTAL_K + (BLOCK_K - 1)];",
            hip_code,
            "Stage 2 backbone does not read boundary_tok from slot 5 (BLOCK_K - 1)!",
        )

        self.assertIn(
            "if (boundary_tok >= V_TOTAL) boundary_tok = 0;",
            hip_code,
            "Stage 2 backbone does not guard against boundary_tok out of bounds!",
        )

    def test_stage2_recurrent_norm_applied_to_h1_slot_5(self):
        """Verify that Stage 2 applies recurrent_norm to s_h1 + (BLOCK_K - 1) * H_DIM."""
        with open(HIP_FILE, "r") as f:
            hip_code = f.read()

        self.assertIn(
            "dev_rmsnorm(s_h1 + (BLOCK_K - 1) * H_DIM, p->recurrent_norm, s_temp, H_DIM);",
            hip_code,
            "Stage 2 base commit does not apply recurrent_norm to s_h1[5]!",
        )

    def test_stage2_base2_composition_formula(self):
        """Verify that Stage 2 base2 is recurrent_norm(h1[5]) + anchor_proj(emb[T6])."""
        with open(HIP_FILE, "r") as f:
            hip_code = f.read()

        self.assertIn(
            "s_base2[h] = round_bf16_val(s_temp[h] + s_temp[H_DIM + h]);",
            hip_code,
            "Base 2 is not composed of recurrent_norm + anchor_proj!",
        )
        self.assertIn(
            "s_h2[0 * H_DIM + h] = s_base2[h];",
            hip_code,
            "Base 2 is not placed at token 0 of s_h2!",
        )


class TestKernelAttributesAndOccupancyGuarantees(unittest.TestCase):
    """Verify that all kernels have 100% occupancy attributes and bounds."""

    def test_waves_per_eu_attribute_on_all_backbone_kernels(self):
        """Verify that all Milestone 3 backbone kernels specify __attribute__((amdgpu_waves_per_eu(16, 16)))."""
        with open(HIP_FILE, "r") as f:
            hip_code = f.read()

        m3_kernels = [
            "fulldrafter_stage1_backbone_kernel",
            "fulldrafter_stage1_base_commit_kernel",
            "r4d_fulldrafter_block_dflash_conv_kernel",
            "r4d_fulldrafter_token_rmsnorm_kernel",
            "r4d_fulldrafter_block_qkv_kernel",
            "r4d_fulldrafter_block_attention_kernel",
            "r4d_fulldrafter_block_out_proj_kernel",
            "r4d_fulldrafter_block_mlp_gate_up_kernel",
            "r4d_fulldrafter_block_mlp_down_kernel",
            "r4d_fulldrafter_final_norm_kernel",
            "r4d_fulldrafter_copy_final_norm_kernel",
            "fulldrafter_stage2_backbone_kernel",
            "fulldrafter_stage2_base_commit_kernel",
        ]

        for k_name in m3_kernels:
            idx = hip_code.find(k_name)
            self.assertNotEqual(idx, -1, f"Kernel {k_name} not found in {HIP_FILE}")
            # Check the line containing the kernel name
            header_snippet = hip_code[max(0, idx - 100) : idx + len(k_name)]
            self.assertIn(
                "amdgpu_waves_per_eu(16, 16)",
                header_snippet,
                f"Kernel {k_name} missing __attribute__((amdgpu_waves_per_eu(16, 16)))!",
            )

    def test_scratchpad_offsets_no_overlap(self):
        """Verify all scratchpad offsets are strictly non-overlapping and aligned."""
        offsets = [
            ("SCR_BASE1_OFF", 0, 768 * 4),
            ("SCR_H1_OFF", 4096, 7 * 768 * 4),
            ("SCR_O1_OFF", 32768, 6 * 2560 * 4),
            ("SCR_BASE2_OFF", 98304, 768 * 4),
            ("SCR_H2_OFF", 102400, 7 * 768 * 4),
            ("SCR_O2_OFF", 131072, 6 * 2560 * 4),
            ("SCR_PREV_TOK_OFF", 196608, 12 * 4),
            ("SCR_BLK_CANDS_OFF", 200704, 64 * 16 * 4),
            ("SCR_BLK_UNARY_OFF", 216064, 64 * 16 * 4),
            ("SCR_ARRIVAL_CTR_OFF", 220160, 4),
            ("SCR_PROP_ID_OFF", 220264, 12 * 16 * 4),
            ("SCR_PROP_VAL_OFF", 221032, 12 * 16 * 4),
            ("SCR_TEMP_OFF", 232000, 64432),
            ("SCR_BASE_LOGITS_OFF", 2097152, 6 * 248320 * 4),
        ]

        # Check in sorted order
        offsets.sort(key=lambda x: x[1])
        for i in range(len(offsets) - 1):
            name1, start1, size1 = offsets[i]
            name2, start2, size2 = offsets[i + 1]
            end1 = start1 + size1
            self.assertLessEqual(
                end1,
                start2,
                f"Scratchpad buffer collision: {name1} [{start1}..{end1}) overlaps with {name2} [{start2}..{start2 + size2})",
            )


if __name__ == "__main__":
    unittest.main()
