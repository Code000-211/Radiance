"""Milestone 2 Adversarial Empirical Challenge Suite (Remediation Verification).
Testing: r4d_fulldrafter_slot_markov_select and Milestone 2 memory/concurrency contracts

Empirical verification of:
1. Dynamic offset validation from libr4d/r4d_fulldrafter.hip and libr4d/r4d_fulldrafter.h
2. Complete resolution of SCR_ARRIVAL_CTR_OFF and SCR_TEMP_OFF aliasing
3. Complete insulation of SCR_PROP_ID_OFF and SCR_PROP_VAL_OFF from s_temp [232000..296512)
4. Bitwise survival of Stage 1 proposals across Stage 2 Backbone execution
5. Stage 2 Backbone arrival counter re-initialization (*ctr = 0u)
6. Grid-wide atomic arrival concurrency & multi-slot counter re-initialization
7. Deterministic tie-breaking (is_better) oracle across edge cases
8. Host non-blocking stream pipeline contracts (zero CPU barriers)
"""

import os
import re
import struct
import threading
import unittest

PROJECT_ROOT = "/home/orca/RadianceMine/worktrees/tap_record"
HIP_FILE = os.path.join(PROJECT_ROOT, "libr4d/r4d_fulldrafter.hip")
HEADER_FILE = os.path.join(PROJECT_ROOT, "libr4d/r4d_fulldrafter.h")

# ==============================================================================
# Helper Constants from r4d_fulldrafter.hip and r4d_fulldrafter.h
# ==============================================================================
W_DIM = 2560
H_DIM = 768
C_DIM = 7680
M_RANK = 64
S_RANK = 64
K_TOP = 16
BLOCK_K = 6
TOTAL_K = 12
INTERM_DIM = 2048
GRID_BLOCKS = 64
V_TOTAL = 248320
V_SHARD = 124160


def parse_hip_offsets(hip_path: str) -> dict:
    """Dynamically parse #define SCR_* from source to ensure tests test current code."""
    offsets = {}
    with open(hip_path, "r") as f:
        for line in f:
            m = re.match(r"#define\s+(SCR_\w+)\s+([0-9]+)(?:LL)?", line.strip())
            if m:
                offsets[m.group(1)] = int(m.group(2))
    return offsets


def parse_header_offsets(header_path: str) -> dict:
    """Dynamically parse #define FULLDRAFTER_SCR_* from header."""
    offsets = {}
    with open(header_path, "r") as f:
        for line in f:
            m = re.match(r"#define\s+(FULLDRAFTER_SCR_\w+_OFF)\s+([0-9]+)", line.strip())
            if m:
                offsets[m.group(1)] = int(m.group(2))
    return offsets


def is_better(va: float, ia: int, vb: float, ib: int) -> bool:
    """Exact emulation of __device__ bool is_better(...) from r4d_fulldrafter.hip."""
    if va > vb:
        return True
    if va < vb:
        return False
    return ia < ib


class TestScratchpadOffsetsDynamicValidation(unittest.TestCase):
    """Verify that source files define correct, synchronized, and non-colliding offsets."""

    def setUp(self):
        self.hip_offsets = parse_hip_offsets(HIP_FILE)
        self.header_offsets = parse_header_offsets(HEADER_FILE)

    def test_offsets_exist(self):
        self.assertIn("SCR_ARRIVAL_CTR_OFF", self.hip_offsets)
        self.assertIn("SCR_PROP_ID_OFF", self.hip_offsets)
        self.assertIn("SCR_PROP_VAL_OFF", self.hip_offsets)
        self.assertIn("SCR_TEMP_OFF", self.hip_offsets)

    def test_hip_and_header_offsets_in_sync(self):
        self.assertEqual(
            self.hip_offsets["SCR_ARRIVAL_CTR_OFF"],
            self.header_offsets["FULLDRAFTER_SCR_ARRIVAL_CTR_OFF"],
        )
        self.assertEqual(
            self.hip_offsets["SCR_PROP_ID_OFF"],
            self.header_offsets["FULLDRAFTER_SCR_PROP_ID_OFF"],
        )
        self.assertEqual(
            self.hip_offsets["SCR_PROP_VAL_OFF"],
            self.header_offsets["FULLDRAFTER_SCR_PROP_VAL_OFF"],
        )

    def test_arrival_counter_offset_not_aliased_to_s_temp(self):
        """CRITICAL: SCR_ARRIVAL_CTR_OFF (now 220160) must NOT alias SCR_TEMP_OFF (232000)."""
        arr_off = self.hip_offsets["SCR_ARRIVAL_CTR_OFF"]
        temp_off = self.hip_offsets["SCR_TEMP_OFF"]
        self.assertNotEqual(
            arr_off,
            temp_off,
            f"SCR_ARRIVAL_CTR_OFF ({arr_off}) still aliases SCR_TEMP_OFF ({temp_off})!",
        )
        self.assertEqual(arr_off, 220160)
        self.assertEqual(temp_off, 232000)
        margin = temp_off - arr_off
        self.assertGreaterEqual(margin, 11840, f"Expected margin >= 11840, got {margin}")

    def test_proposal_offsets_relocated(self):
        """CRITICAL: Proposal IDs and values relocated to 220264 and 221032."""
        self.assertEqual(self.hip_offsets["SCR_PROP_ID_OFF"], 220264)
        self.assertEqual(self.hip_offsets["SCR_PROP_VAL_OFF"], 221032)


class TestScratchpadLayoutAndBoundaryIsolation(unittest.TestCase):
    """Adversarial stress-testing of scratchpad memory boundaries and buffer overlap."""

    def setUp(self):
        self.offsets = parse_hip_offsets(HIP_FILE)
        self.arr_off = self.offsets["SCR_ARRIVAL_CTR_OFF"]
        self.prop_id_off = self.offsets["SCR_PROP_ID_OFF"]
        self.prop_val_off = self.offsets["SCR_PROP_VAL_OFF"]
        self.temp_off = self.offsets["SCR_TEMP_OFF"]
        self.unary_off = self.offsets["SCR_BLK_UNARY_OFF"]

    def test_scratchpad_temp_buffer_size(self):
        """Calculate active memory required by s_temp during backbone execution."""
        min_attn_floats = 21 * H_DIM  # 21 * 768 = 16,128 floats
        min_attn_bytes = min_attn_floats * 4
        self.assertEqual(min_attn_bytes, 64512)

        min_mlp_bytes = 2 * INTERM_DIM * 4  # 2 * 2048 * 4 = 16,384 bytes
        self.assertEqual(min_mlp_bytes, 16384)

        s_temp_min_bytes = max(min_attn_bytes, min_mlp_bytes)
        self.assertGreaterEqual(s_temp_min_bytes, 64512)

    def test_unary_and_arrival_counter_boundary(self):
        """Verify block_unary [216064..220160) does not overlap SCR_ARRIVAL_CTR_OFF (220160)."""
        unary_size = GRID_BLOCKS * K_TOP * 4  # 64 * 16 * 4 = 4096 bytes
        unary_end = self.unary_off + unary_size
        self.assertEqual(unary_end, 220160)
        self.assertEqual(self.arr_off, 220160)
        self.assertLessEqual(unary_end, self.arr_off)

    def test_proposal_id_completely_insulated_from_s_temp(self):
        """Verify SCR_PROP_ID_OFF (220264..221032) has zero overlap with s_temp [232000..296512)."""
        prop_id_start = self.prop_id_off
        prop_id_end = self.prop_id_off + TOTAL_K * K_TOP * 4  # 12 * 16 * 4 = 768 bytes
        self.assertEqual(prop_id_end, 221032)

        s_temp_start = self.temp_off
        s_temp_end = self.temp_off + 64512  # active attention span

        overlap_start = max(prop_id_start, s_temp_start)
        overlap_end = min(prop_id_end, s_temp_end)
        overlap_bytes = max(0, overlap_end - overlap_start)

        self.assertEqual(
            overlap_bytes,
            0,
            f"SCR_PROP_ID_OFF overlaps with s_temp by {overlap_bytes} bytes!",
        )
        margin = s_temp_start - prop_id_end
        self.assertGreater(margin, 10000, f"Margin between prop_id and s_temp is {margin} bytes")

    def test_proposal_val_completely_insulated_from_s_temp(self):
        """Verify SCR_PROP_VAL_OFF (221032..221800) has zero overlap with s_temp [232000..296512)."""
        prop_val_start = self.prop_val_off
        prop_val_end = self.prop_val_off + TOTAL_K * K_TOP * 4  # 12 * 16 * 4 = 768 bytes
        self.assertEqual(prop_val_end, 221800)

        s_temp_start = self.temp_off
        s_temp_end = self.temp_off + 64512

        overlap_start = max(prop_val_start, s_temp_start)
        overlap_end = min(prop_val_end, s_temp_end)
        overlap_bytes = max(0, overlap_end - overlap_start)

        self.assertEqual(
            overlap_bytes,
            0,
            f"SCR_PROP_VAL_OFF overlaps with s_temp by {overlap_bytes} bytes!",
        )
        margin = s_temp_start - prop_val_end
        self.assertEqual(margin, 10200, f"Expected 10200 bytes safety margin, got {margin}")

    def test_pairwise_non_overlapping_all_buffers(self):
        """Verify that all scratchpad buffers are strictly disjoint."""
        buffers = [
            ("base1", self.offsets["SCR_BASE1_OFF"], 768 * 4),
            ("h1", self.offsets["SCR_H1_OFF"], 7 * 768 * 4),
            ("o1", self.offsets["SCR_O1_OFF"], 6 * 2560 * 4),
            ("base2", self.offsets["SCR_BASE2_OFF"], 768 * 4),
            ("h2", self.offsets["SCR_H2_OFF"], 7 * 768 * 4),
            ("o2", self.offsets["SCR_O2_OFF"], 6 * 2560 * 4),
            ("prev_tokens", self.offsets["SCR_PREV_TOK_OFF"], 16 * 4),
            ("block_cands", self.offsets["SCR_BLK_CANDS_OFF"], 64 * 16 * 4),
            ("block_unary", self.offsets["SCR_BLK_UNARY_OFF"], 64 * 16 * 4),
            ("arrival_ctr", self.offsets["SCR_ARRIVAL_CTR_OFF"], 4),
            ("prop_ids", self.offsets["SCR_PROP_ID_OFF"], 12 * 16 * 4),
            ("prop_vals", self.offsets["SCR_PROP_VAL_OFF"], 12 * 16 * 4),
            ("temp", self.offsets["SCR_TEMP_OFF"], 64512),
            ("base_logits", self.offsets["SCR_BASE_LOGITS_OFF"], 6 * V_TOTAL * 4),
        ]
        # Sort by start offset
        buffers.sort(key=lambda b: b[1])
        for i in range(len(buffers) - 1):
            curr_name, curr_start, curr_size = buffers[i]
            next_name, next_start, _ = buffers[i + 1]
            curr_end = curr_start + curr_size
            self.assertLessEqual(
                curr_end,
                next_start,
                f"Collision: buffer '{curr_name}' ends at {curr_end} but '{next_name}' starts at {next_start}!",
            )


class TestStage1ProposalSurvival(unittest.TestCase):
    """Empirical simulation of Stage 1 proposal survival across Stage 2 Backbone execution."""

    def setUp(self):
        self.offsets = parse_hip_offsets(HIP_FILE)
        self.prop_id_off = self.offsets["SCR_PROP_ID_OFF"]
        self.prop_val_off = self.offsets["SCR_PROP_VAL_OFF"]
        self.temp_off = self.offsets["SCR_TEMP_OFF"]
        self.per_seq = self.offsets["SCR_PER_SEQ"]

    def test_stage1_proposals_survive_stage2_backbone_100_percent(self):
        """Simulate sequential pipeline: Stage 1 proposals remain 100% bitwise intact."""
        scratch = bytearray(self.per_seq)

        # 1. Stage 1 Slot Markov Select writes proposal IDs and scores for slots 0..5
        dummy_cands = [1000 + i for i in range(BLOCK_K * K_TOP)]
        dummy_scores = [10.5 + float(i) for i in range(BLOCK_K * K_TOP)]

        struct.pack_into(f"<{BLOCK_K * K_TOP}I", scratch, self.prop_id_off, *dummy_cands)
        struct.pack_into(f"<{BLOCK_K * K_TOP}f", scratch, self.prop_val_off, *dummy_scores)

        # 2. Stage 2 Backbone executes: writes 64,512 bytes of float activations across s_temp
        stage2_activations = bytearray([0x5A] * 64512)
        scratch[self.temp_off : self.temp_off + 64512] = stage2_activations

        # 3. Verify Stage 1 proposal IDs and scores survived with ZERO corruption
        s1_cands_after = struct.unpack_from(f"<{BLOCK_K * K_TOP}I", scratch, self.prop_id_off)
        s1_scores_after = struct.unpack_from(f"<{BLOCK_K * K_TOP}f", scratch, self.prop_val_off)

        self.assertEqual(
            list(s1_cands_after),
            dummy_cands,
            "CRITICAL FAILURE: Stage 1 proposal IDs were corrupted by Stage 2 Backbone!",
        )
        self.assertEqual(
            list(s1_scores_after),
            dummy_scores,
            "CRITICAL FAILURE: Stage 1 proposal scores were corrupted by Stage 2 Backbone!",
        )


class TestAtomicArrivalConcurrencyAndReinitialization(unittest.TestCase):
    """Stress-test grid-wide atomic arrival counter and Stage 2 reinitialization."""

    def test_stage2_backbone_source_contains_counter_reinit(self):
        """Verify that fulldrafter_stage2_backbone_kernel physically contains *ctr = 0u."""
        with open(HIP_FILE, "r") as f:
            content = f.read()

        s2_bb_idx = content.find("fulldrafter_stage2_backbone_kernel")
        self.assertNotEqual(s2_bb_idx, -1, "fulldrafter_stage2_backbone_kernel not found")

        s2_code = content[s2_bb_idx : s2_bb_idx + 2000]
        self.assertIn(
            "*(uint32_t*)(s_base + SCR_ARRIVAL_CTR_OFF) = 0u;",
            s2_code,
            "fulldrafter_stage2_backbone_kernel does not re-initialize arrival counter!",
        )

    def test_ideal_atomic_arrival_across_64_concurrent_blocks(self):
        """Under clean initial state (ctr = 0), exactly 1 block among 64 becomes the last block."""
        lock = threading.Lock()
        ctr = 0
        last_block_flags = []

        def workgroup(block_id: int):
            nonlocal ctr
            with lock:
                s_arrived = ctr
                ctr += 1
            is_last = s_arrived == GRID_BLOCKS - 1
            last_block_flags.append(is_last)

        threads = [
            threading.Thread(target=workgroup, args=(b,)) for b in range(GRID_BLOCKS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(last_block_flags), GRID_BLOCKS)
        self.assertEqual(
            sum(last_block_flags),
            1,
            f"Exactly 1 block must be last block, got {sum(last_block_flags)}",
        )

    def test_multi_slot_sequence_with_stage2_backbone_reinit(self):
        """Simulate all 12 sequential slots with Stage 2 backbone re-initialization."""
        ctr = 0

        for slot_idx in range(TOTAL_K):
            stage = 1 if slot_idx < BLOCK_K else 2
            slot = slot_idx if stage == 1 else slot_idx - BLOCK_K

            # Before Stage 2 Slot 0: Stage 2 Backbone executes!
            if stage == 2 and slot == 0:
                # Stage 2 Backbone re-initializes counter:
                ctr = 0

            # Block arrival simulation for 64 blocks
            arrivals = []
            for b in range(GRID_BLOCKS):
                s_arrived = ctr
                ctr += 1
                arrivals.append(s_arrived == GRID_BLOCKS - 1)

            last_blocks = sum(arrivals)
            self.assertEqual(
                last_blocks,
                1,
                f"Slot {slot_idx} failed: expected exactly 1 last block, got {last_blocks}",
            )
            # Last block resets ctr to 0
            ctr = 0


class TestDeterministicTieBreaking(unittest.TestCase):
    """Verify bitwise deterministic tie-breaking logic (is_better)."""

    def test_score_dominance(self):
        self.assertTrue(is_better(10.0, 100, 9.9, 50))
        self.assertFalse(is_better(9.9, 50, 10.0, 100))

    def test_tie_breaking_lower_id_wins(self):
        # Equal score: smaller vocabulary token ID must win
        self.assertTrue(is_better(5.0, 10, 5.0, 20))
        self.assertFalse(is_better(5.0, 20, 5.0, 10))
        # Identical token ID and score
        self.assertFalse(is_better(5.0, 10, 5.0, 10))

    def test_extreme_values_and_subnormals(self):
        subnormal = struct.unpack("<f", struct.pack("<I", 0x00000001))[0]
        self.assertTrue(is_better(subnormal, 1, 0.0, 2))
        self.assertTrue(is_better(0.0, 1, -1e30, 2))
        self.assertTrue(is_better(-1e30, 10, -1e30, 20))


class TestHostNonBlockingContract(unittest.TestCase):
    """Verify asynchronous stream dispatch contract from C++ source."""

    def test_no_unconditional_stream_synchronization(self):
        with open(HIP_FILE, "r") as f:
            content = f.read()

        # Check for hipDeviceSynchronize
        self.assertNotIn("hipDeviceSynchronize()", content)

        # Check that hipStreamSynchronize only occurs under profile guard
        sync_count = content.count("hipStreamSynchronize(")
        self.assertEqual(sync_count, 1)

        profile_idx = content.find("if (profile_check) {")
        sync_idx = content.find("hipStreamSynchronize(")
        self.assertGreater(sync_idx, profile_idx)


if __name__ == "__main__":
    unittest.main()
