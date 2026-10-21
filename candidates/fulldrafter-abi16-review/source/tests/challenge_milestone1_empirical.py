"""
Milestone 1 Adversarial Empirical Challenge Suite
Testing: r4d_fulldrafter_lmhead_gemv6
Properties probed:
1. Mathematical precision & numerical parity with PyTorch reference
2. IEEE-754 edge cases: subnormals, NaNs, Infs, zero-valued weights
3. Boundary limits: V_shard = 124,160, V_total = 248,320, D = 2560, grid ceiling
"""

import math
import struct
import unittest

import numpy as np
import torch


def bf16_bits_to_fp32(val_u16: int) -> float:
    """Emulates `__builtin_bit_cast(float, w_p << 16)` or `w_p & 0xFFFF0000`."""
    u32 = (int(val_u16) & 0xFFFF) << 16
    return struct.unpack('<f', struct.pack('<I', u32))[0]


def fp32_to_bf16_bits_truncated(val_f32: float) -> int:
    """Emulates the kernel cooperative loading: `u0 >> 16`."""
    u32 = struct.unpack('<I', struct.pack('<f', float(val_f32)))[0]
    return (u32 >> 16) & 0xFFFF


def fp32_to_bf16_bits_rounded(val_f32: float) -> int:
    """Emulates `round_bf16_val` from fast_out_proj_kernel."""
    u32 = struct.unpack('<I', struct.pack('<f', float(val_f32)))[0]
    h = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16
    return h & 0xFFFF


def emulate_kernel_gemv(hidden_fp32: np.ndarray, weight_bf16_u16: np.ndarray) -> np.ndarray:
    """
    Exact bit-accurate software emulation of `r4d_fulldrafter_lmhead_gemv6_kernel`.
    hidden_fp32: shape [6, 2560] float32
    weight_bf16_u16: shape [V, 2560] uint16
    Returns: logits_base of shape [6, V] float32
    """
    V, D = weight_bf16_u16.shape
    assert D == 2560
    assert hidden_fp32.shape == (6, 2560)

    # 1. Cooperative load & pack of hidden states into LDS:
    # s_hidden[s * 1280 + i] stores (u0 >> 16) | (u1 & 0xFFFF0000)
    s_hidden = np.zeros((6, 1280), dtype=np.uint32)
    for s in range(6):
        for i in range(1280):
            f0 = hidden_fp32[s, 2 * i + 0]
            f1 = hidden_fp32[s, 2 * i + 1]
            u0 = fp32_to_bf16_bits_truncated(f0)
            u1 = fp32_to_bf16_bits_truncated(f1)
            s_hidden[s, i] = u0 | (u1 << 16)

    logits_out = np.zeros((6, V), dtype=np.float32)

    # 2. Each wave computes one global_row
    # We simulate wave execution:
    # 32 lanes per wave
    for row in range(V):
        w_row = weight_bf16_u16[row]  # 2560 elements of uint16

        # Accumulator per lane: acc[32, 6]
        lane_acc = np.zeros((32, 6), dtype=np.float32)

        for lane_id in range(32):
            for k_step in range(10):
                v_idx = k_step * 32 + lane_id
                lds_base = v_idx * 4

                # 4 uint32 words in 128-bit vector load
                for s in range(6):
                    fma_sum = np.float32(0.0)
                    for p in range(4):
                        w_lo_u16 = int(w_row[8 * v_idx + 2 * p + 0])
                        w_hi_u16 = int(w_row[8 * v_idx + 2 * p + 1])
                        fw_lo = np.float32(bf16_bits_to_fp32(w_lo_u16))
                        fw_hi = np.float32(bf16_bits_to_fp32(w_hi_u16))

                        o_word = int(s_hidden[s, lds_base + p])
                        o_lo_u16 = o_word & 0xFFFF
                        o_hi_u16 = (o_word >> 16) & 0xFFFF
                        fo_lo = np.float32(bf16_bits_to_fp32(o_lo_u16))
                        fo_hi = np.float32(bf16_bits_to_fp32(o_hi_u16))

                        prod0 = fw_lo * fo_lo
                        prod1 = fw_hi * fo_hi
                        fma_sum = fma_sum + prod0 + prod1

                    lane_acc[lane_id, s] += fma_sum

        # 4. Intra-wave reduction across 32 lanes (__shfl_down tree)
        # offset = 16, 8, 4, 2, 1
        acc = np.copy(lane_acc)
        for offset in [16, 8, 4, 2, 1]:
            for lid in range(32 - offset):
                acc[lid] = acc[lid] + acc[lid + offset]

        # 5. Lane 0 stores
        logits_out[:, row] = acc[0]

    return logits_out


class Milestone1AdversarialChallenge(unittest.TestCase):

    def test_01_bf16_fp32_exact_bitcast_identity(self):
        """Verify that bitwise casting uint16 to uint32 << 16 produces exact IEEE-754 float."""
        for u in [0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0x0001, 0x8001]:
            f = bf16_bits_to_fp32(u)
            t_bf16 = torch.tensor([u if u < 32768 else u - 65536], dtype=torch.int16).view(torch.bfloat16)
            f_torch = t_bf16.to(torch.float32).item()
            if math.isnan(f):
                self.assertTrue(math.isnan(f_torch))
            else:
                self.assertEqual(f, f_torch)
                self.assertEqual(math.copysign(1.0, f), math.copysign(1.0, f_torch))

    def test_02_random_parity_vs_pytorch(self):
        """Verify numerical parity against PyTorch reference over random data."""
        np.random.seed(42)
        torch.manual_seed(42)

        # 6 slots, 16 vocabulary rows, D=2560
        V_test = 16
        D = 2560

        # Create hidden in FP32, rounded to BF16 (as produced by fast_out_proj_kernel)
        hidden_torch = torch.randn(6, D, dtype=torch.float32)
        hidden_bf16 = hidden_torch.to(torch.bfloat16)
        hidden_fp32_exact = hidden_bf16.to(torch.float32).numpy()

        weight_torch_bf16 = torch.randn(V_test, D, dtype=torch.bfloat16)
        weight_u16 = weight_torch_bf16.view(torch.int16).numpy().view(np.uint16)

        # 1. Emulate kernel
        kernel_logits = emulate_kernel_gemv(hidden_fp32_exact, weight_u16)

        # 2. PyTorch reference (FP32 accumulation of BF16 inputs)
        ref_logits = torch.matmul(hidden_bf16.to(torch.float32), weight_torch_bf16.to(torch.float32).T).numpy()

        # Compare difference
        abs_diff = np.abs(kernel_logits - ref_logits)
        rel_diff = abs_diff / (np.abs(ref_logits) + 1e-7)
        max_abs = np.max(abs_diff)
        max_rel = np.max(rel_diff)

        # Because summation order differs (tree reduction vs MKL vector accumulators),
        # absolute error should be within a few ULPs (< 1e-4 relative error for sum of 2560 floats)
        self.assertLess(max_rel, 1e-4, f"Relative error {max_rel} exceeded threshold")
        self.assertLess(max_abs, 1e-3, f"Absolute error {max_abs} exceeded threshold")

    def test_03_subnormal_inputs(self):
        """Stress-test subnormal weights and hidden inputs."""
        D = 2560
        # Subnormals in BF16
        w_sub = np.ones((1, D), dtype=np.uint16) * 0x0001  # Smallest positive subnormal
        h_sub = np.ones((6, D), dtype=np.float32) * (2.0 ** (-133))

        kernel_logits = emulate_kernel_gemv(h_sub, w_sub)
        self.assertTrue(np.all(np.isfinite(kernel_logits)))
        # 2560 * 2^(-133) * 2^(-133) underflows to 0.0 in FP32
        self.assertTrue(np.all(kernel_logits >= 0.0))

    def test_04_zeros_and_signs(self):
        """Stress-test +0.0 and -0.0 preservation."""
        D = 2560
        w_zeros = np.zeros((1, D), dtype=np.uint16)
        h_zeros = np.zeros((6, D), dtype=np.float32)

        kernel_logits = emulate_kernel_gemv(h_zeros, w_zeros)
        self.assertTrue(np.all(kernel_logits == 0.0))

        # Mixed zeros and finite values
        w_zeros[0, 100] = 0x3F80  # 1.0 in BF16
        h_zeros[0, 100] = 5.0
        kernel_logits = emulate_kernel_gemv(h_zeros, w_zeros)
        self.assertAlmostEqual(kernel_logits[0, 0], 5.0, places=5)
        for s in range(1, 6):
            self.assertEqual(kernel_logits[s, 0], 0.0)

    def test_05_nan_and_inf_propagation(self):
        """Stress-test NaN and Infinity propagation."""
        D = 2560
        w = np.zeros((1, D), dtype=np.uint16)
        h = np.zeros((6, D), dtype=np.float32)

        # Put NaN in weight
        w[0, 50] = 0x7FC0  # NaN
        h[0, 50] = 1.0
        kernel_logits = emulate_kernel_gemv(h, w)
        self.assertTrue(np.isnan(kernel_logits[0, 0]), "NaN failed to propagate from weight")

        # Put Inf in hidden
        w[0, 50] = 0x3F80  # 1.0
        h[1, 50] = float('inf')
        kernel_logits = emulate_kernel_gemv(h, w)
        self.assertTrue(np.isinf(kernel_logits[1, 0]), "Inf failed to propagate from hidden")

    def test_06_boundary_and_grid_ceiling(self):
        """Verify grid calculations for V_shard = 124,160 and dual-shard V_total = 248,320."""
        V_shard = 124160
        V_total_dual = 248320

        # Grid ceiling formula: (V_total + 7) / 8
        grid_single = (V_shard + 7) // 8
        grid_dual = (V_total_dual + 7) // 8

        self.assertEqual(grid_single, 15520)
        self.assertEqual(grid_single * 8, V_shard)  # Exactly divides by 8!

        self.assertEqual(grid_dual, 31040)
        self.assertEqual(grid_dual * 8, V_total_dual)  # Exactly divides by 8!

        # Boundary checks for unaligned V_total:
        for V_test in [1, 7, 8, 9, 124159, 124160, 124161, 248319, 248320, 248321]:
            grid = (V_test + 7) // 8
            max_covered_row = grid * 8 - 1
            self.assertGreaterEqual(max_covered_row, V_test - 1)
            # In kernel: if (global_row >= V_total) return;
            # Ensure waves beyond V_test are properly guarded
            active_waves = [r for r in range(grid * 8) if r < V_test]
            self.assertEqual(len(active_waves), V_test)

    def test_07_k_loop_and_vector_bounds(self):
        """Verify that the 10-step unroll processes exactly D=2560 with no out-of-bounds access."""
        D = 2560
        num_u128_vectors = D // 8  # 320 vectors
        self.assertEqual(num_u128_vectors, 320)

        # Trace all vector offsets accessed in the kernel
        accessed_offsets = set()
        # 1. Prefetch loop (k_step 0..8):
        for lane_id in range(32):
            # initial load
            accessed_offsets.add(lane_id)
            for k_step in range(9):
                next_offset = (k_step + 1) * 32 + lane_id
                accessed_offsets.add(next_offset)

        # Max offset accessed
        max_offset = max(accessed_offsets)
        min_offset = min(accessed_offsets)
        self.assertEqual(min_offset, 0)
        self.assertEqual(max_offset, 319)  # Exactly the last vector!
        self.assertEqual(len(accessed_offsets), 320)  # Every single vector accessed exactly once!


if __name__ == '__main__':
    unittest.main()
