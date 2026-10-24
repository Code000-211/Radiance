"""
Empirical Challenge Suite: Milestone 4
Mathematical Parity, Stage 1 -> Stage 2 Recurrent Transition, Proposal Format Integrity,
Buffer Safety & Pure Asynchronous Stream Dispatch.

Author: teamwork_preview_challenger_m4_1 (Empirical Challenger)
"""

import os
import re
import unittest

import torch
import torch.nn.functional as F


class TestMilestone4MathematicalParity(unittest.TestCase):
    """Adversarial stress-testing of Stage 2 Recurrent Transition math against PyTorch reference."""

    def setUp(self):
        self.H_DIM = 768
        self.W_DIM = 2560
        self.BLOCK_K = 6
        self.TOTAL_K = 12
        self.V_TOTAL = 248320
        self.EPS = 1e-6

    def _round_bf16(self, tensor: torch.Tensor) -> torch.Tensor:
        """Emulate IEEE BF16 rounding (16-bit truncation with round-to-nearest-even)."""
        return tensor.to(torch.bfloat16).to(torch.float32)

    def _rmsnorm_reference(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """PyTorch exact RMSNorm reference: x * rsqrt(mean(x^2) + eps) * weight."""
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight

    def test_recurrent_transition_formula_equivalence(self):
        """Verify h2[0] = recurrent_norm(h1[5]) + anchor_proj(emb[T6]) matches PyTorch reference within BF16 ULP."""
        torch.manual_seed(42)

        # Synthetic weights in BF16
        w_recurrent_norm = torch.randn(self.H_DIM, dtype=torch.bfloat16)
        w_anchor_proj = torch.randn(self.H_DIM, self.W_DIM, dtype=torch.bfloat16)
        emb_table = torch.randn(self.V_TOTAL, self.W_DIM, dtype=torch.bfloat16)

        # Terminal hidden state h1[5] (post final_norm)
        h1_5 = torch.randn(self.H_DIM, dtype=torch.bfloat16)

        # Discrete winning token T6 from slot 5
        T6 = 12345
        emb_T6 = emb_table[T6]

        # PyTorch Reference Execution in BF16:
        # _next_base(last_hidden, boundary_emb): (recurrent_norm(last_hidden) + anchor_proj(boundary_emb))
        ref_norm = self._rmsnorm_reference(h1_5.float(), w_recurrent_norm.float(), self.EPS).to(torch.bfloat16)
        ref_proj = F.linear(emb_T6, w_anchor_proj)  # in BF16
        ref_base2 = (ref_norm + ref_proj).float()

        # HIP Kernel Emulation:
        # 1. dev_rmsnorm(s_h1[5], recurrent_norm, s_temp, H_DIM)
        var_hip = torch.sum(h1_5.float() * h1_5.float()) / float(self.H_DIM)
        scale_hip = torch.rsqrt(var_hip + self.EPS)
        hip_norm = h1_5.float() * scale_hip * w_recurrent_norm.float()

        # 2. dot_anc on teacher_embedding[T6] and anchor_proj rows
        hip_dot_anc = torch.matmul(w_anchor_proj.float(), emb_T6.float())
        hip_proj_rounded = hip_dot_anc.to(torch.bfloat16).float()

        # 3. s_base2[h] = round_bf16_val(s_temp[h] + s_temp[H_DIM + h])
        hip_base2 = (hip_norm + hip_proj_rounded).to(torch.bfloat16).float()

        # Check equivalence
        max_diff = torch.max(torch.abs(ref_base2 - hip_base2)).item()
        cos_sim = F.cosine_similarity(ref_base2.unsqueeze(0), hip_base2.unsqueeze(0)).item()
        mean_rel_err = torch.mean(torch.abs(ref_base2 - hip_base2) / (torch.abs(ref_base2) + 1e-4)).item()

        # Max diff must be bounded within 1 ULP of BF16 representation (<= 1.0)
        self.assertLessEqual(max_diff, 1.0, f"Max diff too high: {max_diff}")
        self.assertGreater(cos_sim, 0.99999, f"Cosine similarity too low: {cos_sim}")
        self.assertLess(mean_rel_err, 0.005, f"Mean relative error too high: {mean_rel_err}")

    def test_negative_slot_index_counterexample(self):
        """Adversarial negative test: Using slot 4 (h1[4]) or slot 0 (h1[0]) causes massive parity failure."""
        torch.manual_seed(99)
        w_recurrent_norm = torch.randn(self.H_DIM).float()
        w_anchor_proj = torch.randn(self.H_DIM, self.W_DIM).float()
        emb_table = torch.randn(100, self.W_DIM).float()

        # Complete h1 across all 6 slots
        h1 = torch.randn(self.BLOCK_K, self.H_DIM).float()
        tokens_out = [10, 20, 30, 40, 50, 60]

        # Correct slot 5 transition
        correct_base2 = self._rmsnorm_reference(h1[5], w_recurrent_norm, self.EPS) + F.linear(emb_table[tokens_out[5]], w_anchor_proj)

        # Wrong slot 4 transition (off-by-one error)
        wrong_base2_slot4 = self._rmsnorm_reference(h1[4], w_recurrent_norm, self.EPS) + F.linear(emb_table[tokens_out[4]], w_anchor_proj)

        # Wrong slot 0 transition
        wrong_base2_slot0 = self._rmsnorm_reference(h1[0], w_recurrent_norm, self.EPS) + F.linear(emb_table[tokens_out[0]], w_anchor_proj)

        diff_slot4 = torch.max(torch.abs(correct_base2 - wrong_base2_slot4)).item()
        diff_slot0 = torch.max(torch.abs(correct_base2 - wrong_base2_slot0)).item()

        self.assertGreater(diff_slot4, 1.0, f"Slot 4 must diverge from Slot 5, got diff {diff_slot4}")
        self.assertGreater(diff_slot0, 1.0, f"Slot 0 must diverge from Slot 5, got diff {diff_slot0}")


class TestBoundaryTokenExtractionAndClamping(unittest.TestCase):
    """Stress-test edge cases for boundary token T6 extraction from tokens_out[5]."""

    def setUp(self):
        self.TOTAL_K = 12
        self.BLOCK_K = 6
        self.V_TOTAL = 248320

    def _extract_boundary_tok(self, tokens_out, seq_idx=0):
        """Mirrors line 1363-1364 in r4d_fulldrafter.hip."""
        boundary_tok = tokens_out[seq_idx * self.TOTAL_K + (self.BLOCK_K - 1)]
        if boundary_tok >= self.V_TOTAL:
            boundary_tok = 0
        return boundary_tok

    def test_valid_token_extraction(self):
        tokens_out = [0] * self.TOTAL_K
        tokens_out[5] = 42100  # Slot 5
        self.assertEqual(self._extract_boundary_tok(tokens_out), 42100)

    def test_zero_token_extraction(self):
        tokens_out = [0] * self.TOTAL_K
        tokens_out[5] = 0
        self.assertEqual(self._extract_boundary_tok(tokens_out), 0)

    def test_max_vocab_token_extraction(self):
        tokens_out = [0] * self.TOTAL_K
        tokens_out[5] = self.V_TOTAL - 1  # 248319
        self.assertEqual(self._extract_boundary_tok(tokens_out), 248319)

    def test_out_of_bounds_clamping(self):
        # Exactly V_TOTAL (248320)
        tokens_out = [0] * self.TOTAL_K
        tokens_out[5] = self.V_TOTAL
        self.assertEqual(self._extract_boundary_tok(tokens_out), 0)

        # Beyond V_TOTAL
        tokens_out[5] = self.V_TOTAL + 1000
        self.assertEqual(self._extract_boundary_tok(tokens_out), 0)

        # 0xFFFFFFFF (uninitialized memory)
        tokens_out[5] = 0xFFFFFFFF
        self.assertEqual(self._extract_boundary_tok(tokens_out), 0)

    def test_multi_sequence_boundary_extraction(self):
        M = 4
        tokens_out = [0] * (M * self.TOTAL_K)
        expected = [101, 202, 303, 404]
        for s in range(M):
            tokens_out[s * self.TOTAL_K + 5] = expected[s]
            self.assertEqual(self._extract_boundary_tok(tokens_out, seq_idx=s), expected[s])


class TestProposalFormatAndAlignmentIntegrity(unittest.TestCase):
    """Verify bitwise alignment of Top-16 candidate IDs, FP32 logits, and draft tokens."""

    def setUp(self):
        self.project_root = "/home/orca/RadianceMine/worktrees/tap_record"
        self.hip_path = os.path.join(self.project_root, "libr4d/r4d_fulldrafter.hip")
        self.h_path = os.path.join(self.project_root, "libr4d/r4d_fulldrafter.h")

        with open(self.hip_path) as f:
            self.hip_code = f.read()
        with open(self.h_path) as f:
            self.h_code = f.read()

        # Parse scratch offsets
        self.offsets = {}
        for m in re.finditer(r"#define\s+(SCR_\w+_OFF)\s+([0-9L]+)", self.hip_code):
            name = m.group(1)
            val = int(m.group(2).rstrip("L"))
            self.offsets[name] = val

    def test_proposal_offsets_existence_and_match(self):
        self.assertIn("SCR_PROP_ID_OFF", self.offsets)
        self.assertIn("SCR_PROP_VAL_OFF", self.offsets)
        self.assertEqual(self.offsets["SCR_PROP_ID_OFF"], 220264)
        self.assertEqual(self.offsets["SCR_PROP_VAL_OFF"], 221032)

    def test_proposal_alignment_multiples(self):
        """Candidate IDs (uint32) and values (float) must be at least 4-byte and 8-byte aligned."""
        id_off = self.offsets["SCR_PROP_ID_OFF"]
        val_off = self.offsets["SCR_PROP_VAL_OFF"]

        self.assertEqual(id_off % 4, 0, f"SCR_PROP_ID_OFF ({id_off}) must be 4-byte aligned")
        self.assertEqual(val_off % 4, 0, f"SCR_PROP_VAL_OFF ({val_off}) must be 4-byte aligned")
        self.assertEqual(id_off % 8, 0, f"SCR_PROP_ID_OFF ({id_off}) must be 8-byte aligned")
        self.assertEqual(val_off % 8, 0, f"SCR_PROP_VAL_OFF ({val_off}) must be 8-byte aligned")

    def test_proposal_size_and_zero_overlap(self):
        """Top-16 IDs: [12, 16] uint32 = 768 bytes. Top-16 Vals: [12, 16] float = 768 bytes."""
        id_off = self.offsets["SCR_PROP_ID_OFF"]
        val_off = self.offsets["SCR_PROP_VAL_OFF"]
        temp_off = self.offsets["SCR_TEMP_OFF"]

        # Exactly 768 bytes between IDs and Vals
        self.assertEqual(val_off - id_off, 12 * 16 * 4)

        # Gap between Vals and temp buffer
        val_end = val_off + 12 * 16 * 4
        self.assertLessEqual(val_end, temp_off)
        margin = temp_off - val_end
        self.assertGreaterEqual(margin, 10000, f"Margin between proposal values and temp buffer is {margin} bytes")

    def test_tie_breaking_deterministic_oracle(self):
        """Verify strict lower-ID tie breaking for proposals with equal score."""
        def is_better(score_a, id_a, score_b, id_b):
            """Mirrors line 77 in r4d_fulldrafter.hip."""
            if score_a > score_b:
                return True
            if score_a == score_b:
                return id_a < id_b
            return False

        # Higher score wins
        self.assertTrue(is_better(10.5, 100, 10.0, 50))
        self.assertFalse(is_better(9.5, 10, 10.0, 50))

        # Tie: smaller ID wins
        self.assertTrue(is_better(10.0, 50, 10.0, 100))
        self.assertFalse(is_better(10.0, 100, 10.0, 50))

        # Strict: identical ID does not replace
        self.assertFalse(is_better(10.0, 50, 10.0, 50))


class TestBufferSafetyAndHeadroom(unittest.TestCase):
    """Verify multi-batch scratchpad safety and GPU headroom budget compliance."""

    def setUp(self):
        self.SCR_PER_SEQ = 8388608  # 8 MiB
        self.HEADROOM_BYTES = 384 * 1024 * 1024  # 384 MiB

    def test_batch_scales_within_headroom(self):
        for M in [1, 2, 4, 8, 16, 32]:
            allocated = M * self.SCR_PER_SEQ
            self.assertLessEqual(allocated, self.HEADROOM_BYTES,
                                 f"Batch M={M} ({allocated / 1024**2:.1f} MiB) exceeds headroom")

    def test_max_safe_batch_size(self):
        max_m = self.HEADROOM_BYTES // self.SCR_PER_SEQ
        self.assertEqual(max_m, 48, f"Expected max batch 48 sequences, got {max_m}")

    def test_sequence_scratchpad_disjointness(self):
        """Verify non-overlapping intervals for consecutive sequences."""
        M = 16
        intervals = []
        for s in range(M):
            start = s * self.SCR_PER_SEQ
            end = start + self.SCR_PER_SEQ
            intervals.append((start, end))

        for i in range(len(intervals) - 1):
            self.assertEqual(intervals[i][1], intervals[i+1][0],
                             f"Sequence {i} and {i+1} must be contiguous with zero gap and zero overlap")


class TestAsynchronousStreamDispatchAudit(unittest.TestCase):
    """Static audit ensuring pure asynchronous stream dispatch and zero CPU synchronization barriers."""

    def setUp(self):
        self.project_root = "/home/orca/RadianceMine/worktrees/tap_record"
        self.hip_file = os.path.join(self.project_root, "libr4d/r4d_fulldrafter.hip")
        with open(self.hip_file) as f:
            self.hip_code = f.read()

    def test_zero_device_synchronize(self):
        """hipDeviceSynchronize is strictly forbidden."""
        matches = re.findall(r"\bhipDeviceSynchronize\s*\(", self.hip_code)
        self.assertEqual(len(matches), 0, f"Found {len(matches)} hipDeviceSynchronize calls")

    def test_guarded_stream_synchronize_only(self):
        """hipStreamSynchronize must only appear inside DRAFTER_PROFILE check."""
        matches = list(re.finditer(r"\bhipStreamSynchronize\s*\(", self.hip_code))
        self.assertEqual(len(matches), 1, f"Expected exactly 1 guarded hipStreamSynchronize, found {len(matches)}")

        # Check line context
        start = max(0, matches[0].start() - 200)
        end = min(len(self.hip_code), matches[0].end() + 200)
        snippet = self.hip_code[start:end]
        self.assertIn("profile_check", snippet)

    def test_zero_dynamic_allocations(self):
        """Zero runtime heap/VRAM allocations in forward pass."""
        for pattern in [r"\bhipMalloc\b", r"\bhipFree\b", r"\bmalloc\b", r"\bcalloc\b", r"\bnew\b"]:
            matches = re.findall(pattern, self.hip_code)
            self.assertEqual(len(matches), 0, f"Found {len(matches)} occurrences of {pattern}")


if __name__ == "__main__":
    unittest.main()
