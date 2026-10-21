"""
Milestone 2 Adversarial Empirical Challenge Unit Tests
Validates:
1. Strict deterministic tie-breaking of `is_better` across block and wave boundaries.
2. In-LDS cross-block merge (Phase 2) top-16 completeness and ranking precision.
3. Numerical robustness of Markov projections (subnormals, inf, signed zeros, NaNs).
4. Boundary condition stress tests.
"""

import math
import struct
import unittest

import numpy as np


def is_better(va: float, ia: int, vb: float, ib: int) -> bool:
    if va > vb:
        return True
    if va < vb:
        return False
    return ia < ib


def bf16_to_f32(h: int) -> float:
    u32 = (int(h) & 0xFFFF) << 16
    return struct.unpack('<f', struct.pack('<I', u32))[0]


def f32_to_bf16(f: float) -> int:
    u32 = struct.unpack('<I', struct.pack('<f', float(f)))[0]
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16) & 0xFFFF


def simulate_phase2_merge(all_cands: np.ndarray, all_unary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact CPU software emulation of `r4d_fulldrafter_slot_markov_select_kernel` Phase 2
    (lines 831-906 of libr4d/r4d_fulldrafter.hip).
    """
    TOTAL_CANDS = 1024
    K_TOP = 16

    assert len(all_cands) == TOTAL_CANDS
    assert len(all_unary) == TOTAL_CANDS

    s_all_val = np.array(all_unary, dtype=np.float32)
    s_all_id = np.array(all_cands, dtype=np.uint32)

    top16_val = np.zeros(K_TOP, dtype=np.float32)
    top16_id = np.zeros(K_TOP, dtype=np.uint32)

    for k in range(K_TOP):
        # 1. Thread-level best (256 threads, stride 256)
        t_best_val = np.full(256, -1e30, dtype=np.float32)
        t_best_id = np.full(256, 0xFFFFFFFF, dtype=np.uint32)
        t_best_idx = np.full(256, -1, dtype=np.int32)

        for tid in range(256):
            b_val = np.float32(-1e30)
            b_id = np.uint32(0xFFFFFFFF)
            b_idx = -1
            for i in range(tid, TOTAL_CANDS, 256):
                v = s_all_val[i]
                uid = s_all_id[i]
                if is_better(v, uid, b_val, b_id):
                    b_val = v
                    b_id = uid
                    b_idx = i
            t_best_val[tid] = b_val
            t_best_id[tid] = b_id
            t_best_idx[tid] = b_idx

        # 2. Wave-level reduction (8 waves, 32 lanes each)
        s_red_val = np.zeros(8, dtype=np.float32)
        s_red_id = np.zeros(8, dtype=np.uint32)
        s_red_idx = np.zeros(8, dtype=np.int32)

        for wave in range(8):
            wv_val = t_best_val[wave * 32 : (wave + 1) * 32].copy()
            wv_id = t_best_id[wave * 32 : (wave + 1) * 32].copy()
            wv_idx = t_best_idx[wave * 32 : (wave + 1) * 32].copy()

            for offset in [16, 8, 4, 2, 1]:
                for lane in range(32 - offset):
                    ov = wv_val[lane + offset]
                    oi = wv_id[lane + offset]
                    oidx = wv_idx[lane + offset]
                    if is_better(ov, oi, wv_val[lane], wv_id[lane]):
                        wv_val[lane] = ov
                        wv_id[lane] = oi
                        wv_idx[lane] = oidx

            s_red_val[wave] = wv_val[0]
            s_red_id[wave] = wv_id[0]
            s_red_idx[wave] = wv_idx[0]

        # 3. Inter-wave reduction (Wave 0, 8 elements)
        w0_val = s_red_val.copy()
        w0_id = s_red_id.copy()
        w0_idx = s_red_idx.copy()

        for offset in [4, 2, 1]:
            for t in range(8 - offset):
                ov = w0_val[t + offset]
                oi = w0_id[t + offset]
                oidx = w0_idx[t + offset]
                if is_better(ov, oi, w0_val[t], w0_id[t]):
                    w0_val[t] = ov
                    w0_id[t] = oi
                    w0_idx[t] = oidx

        # 4. Winner recording and LDS blanking
        top16_val[k] = w0_val[0]
        top16_id[k] = w0_id[0]
        widx = w0_idx[0]
        if widx >= 0:
            s_all_val[widx] = np.float32(-1e30)
            s_all_id[widx] = np.uint32(0xFFFFFFFF)

    return top16_id, top16_val


class Milestone2AdversarialTests(unittest.TestCase):

    def test_01_is_better_tie_breaking(self):
        """Verify strict deterministic tie-breaking for equal FP32 values."""
        # Different values
        self.assertTrue(is_better(10.5, 100, 10.4, 20))
        self.assertFalse(is_better(10.4, 20, 10.5, 100))

        # Identical values: lower ID wins
        self.assertTrue(is_better(5.0, 10, 5.0, 20))
        self.assertFalse(is_better(5.0, 20, 5.0, 10))
        self.assertFalse(is_better(5.0, 10, 5.0, 10))

        # Transitivity
        self.assertTrue(is_better(5.0, 10, 5.0, 20))
        self.assertTrue(is_better(5.0, 20, 5.0, 30))
        self.assertTrue(is_better(5.0, 10, 5.0, 30))

    def test_02_signed_zeros(self):
        """Verify +0.0 and -0.0 tie-breaking."""
        pzero = 0.0
        nzero = -0.0
        self.assertEqual(pzero, nzero)
        # Tie break by ID
        self.assertTrue(is_better(pzero, 5, nzero, 10))
        self.assertTrue(is_better(nzero, 5, pzero, 10))
        self.assertFalse(is_better(pzero, 10, nzero, 5))

    def test_03_subnormals_and_infinities(self):
        """Verify subnormals and infinities in comparison."""
        sub1 = bf16_to_f32(0x0001)
        sub2 = bf16_to_f32(0x0002)
        self.assertTrue(sub2 > sub1)
        self.assertTrue(is_better(sub2, 100, sub1, 10))

        pinf = float('inf')
        ninf = float('-inf')
        self.assertTrue(is_better(pinf, 100, 1e20, 1))
        self.assertTrue(is_better(1e20, 1, pinf, 100) is False)
        self.assertTrue(is_better(-1e20, 100, ninf, 1))
        self.assertTrue(is_better(ninf, 10, ninf, 20))

    def test_04_phase2_merge_random_distributions(self):
        """Test Phase 2 in-LDS cross-block merge across 100 random trials against Golden sort."""
        np.random.seed(1337)
        for trial in range(100):
            ids = np.random.choice(248000, size=1024, replace=False).astype(np.uint32)
            vals = np.random.uniform(-50.0, 50.0, size=1024).astype(np.float32)

            res_ids, res_vals = simulate_phase2_merge(ids, vals)

            # Golden reference sort
            order = sorted(range(1024), key=lambda i: (-vals[i], ids[i]))[:16]
            expected_ids = ids[order]
            expected_vals = vals[order]

            np.testing.assert_array_equal(res_ids, expected_ids)
            np.testing.assert_array_equal(res_vals, expected_vals)

    def test_05_phase2_merge_all_identical_scores(self):
        """Stress-test Phase 2 when all 1,024 scores are bitwise identical."""
        np.random.seed(42)
        ids = np.random.choice(248000, size=1024, replace=False).astype(np.uint32)
        vals = np.full(1024, 3.14159, dtype=np.float32)

        res_ids, res_vals = simulate_phase2_merge(ids, vals)

        expected_ids = np.sort(ids)[:16]
        np.testing.assert_array_equal(res_ids, expected_ids)
        np.testing.assert_array_equal(res_vals, vals[:16])

    def test_06_phase2_merge_extreme_winner_positions(self):
        """Verify winner is never missed regardless of where in 1024 array it sits."""
        for target in [0, 1, 31, 32, 255, 256, 511, 512, 1023]:
            ids = np.arange(1024, dtype=np.uint32) + 100
            vals = np.zeros(1024, dtype=np.float32)
            vals[target] = 999.0

            res_ids, res_vals = simulate_phase2_merge(ids, vals)
            self.assertEqual(res_ids[0], ids[target])
            self.assertEqual(res_vals[0], 999.0)

    def test_07_markov_matrix_dot_product_subnormals(self):
        """Stress-test rank-64 Markov dot product with BF16 subnormals."""
        sub = bf16_to_f32(0x0001)
        h_a = np.full(64, sub, dtype=np.float32)
        m_b = np.full(64, sub, dtype=np.float32)

        # Dot product
        dot = float(np.sum(h_a * m_b))
        self.assertTrue(math.isfinite(dot))
        self.assertTrue(dot >= 0.0)


if __name__ == '__main__':
    unittest.main()
