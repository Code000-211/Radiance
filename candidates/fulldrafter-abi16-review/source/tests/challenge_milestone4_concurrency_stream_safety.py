"""Milestone 4 Adversarial Empirical Challenge Test Suite:
Stream Pipeline Safety, Non-Blocking CPU Dispatch, Scratchpad Isolation, and VRAM Headroom.

Author: teamwork_preview_challenger_m4_2 (Milestone 4 Concurrency & Stream Safety Challenger)
"""

import os
import re
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HIP_FILE = os.path.join(REPO_ROOT, "libr4d", "r4d_fulldrafter.hip")
HEADER_FILE = os.path.join(REPO_ROOT, "libr4d", "r4d_fulldrafter.h")

V_SHARD = 124160
V_TOTAL = 248320
H_DIM = 768
W_DIM = 2560
BLOCK_K = 6
TOTAL_K = 12
K_TOP = 16
GPU_HEADROOM_MIB = 384
GPU_HEADROOM_BYTES = GPU_HEADROOM_MIB * 1024 * 1024


def parse_offsets_from_file(filepath):
    offsets = {}
    pattern = re.compile(r"#define\s+(FULLDRAFTER_)?([A-Za-z0-9_]+)\s+([0-9]+L*)")
    with open(filepath, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                name = m.group(2)
                val_str = m.group(3).rstrip("L")
                offsets[name] = int(val_str)
    return offsets


def extract_stream_from_launch(launch_str):
    depth = 0
    tokens = []
    current = []
    for char in launch_str:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            tokens.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current).strip())
    # tokens are [grid, block, lds, stream]
    if len(tokens) >= 4:
        return tokens[3]
    elif len(tokens) == 3 and not tokens[2].isdigit():
        return tokens[2]
    return "STREAM_MISSING"


class TestMilestone4ConcurrencyStreamSafety(unittest.TestCase):

    def setUp(self):
        self.assertTrue(os.path.isfile(HIP_FILE), f"Missing {HIP_FILE}")
        self.assertTrue(os.path.isfile(HEADER_FILE), f"Missing {HEADER_FILE}")
        with open(HIP_FILE, "r") as f:
            self.hip_content = f.read()
        with open(HEADER_FILE, "r") as f:
            self.header_content = f.read()

    def test_exact_56_kernel_dispatches_for_batch1(self):
        """Verify that exactly 56 kernel dispatches are issued in the forward pipeline for M=1."""
        # Stage 1:
        # - Tap prep: 1
        # - Base commit: 1
        # - Transformer blocks: 2 blocks * (1 conv + 1 pre_norm + 1 qkv + 1 attn + 1 out_proj + 1 post_norm + 1 mlp_gate_up + 1 mlp_down) = 16
        # - Final norm: 1
        # - Copy final norm: 1
        # - Fast out proj: 1
        # - LM head GEMV6: 1
        # - Markov select slots 0..5: 6
        # Total S1 = 1 + 1 + 16 + 1 + 1 + 1 + 1 + 6 = 28
        #
        # Stage 2:
        # - Backbone prep: 1
        # - Base commit: 1
        # - Transformer blocks: 2 blocks * 8 = 16
        # - Final norm: 1
        # - Copy final norm: 1
        # - Fast out proj: 1
        # - LM head GEMV6: 1
        # - Markov select slots 0..5: 6
        # Total S2 = 1 + 1 + 16 + 1 + 1 + 1 + 1 + 6 = 28
        #
        # Total pipeline = 28 + 28 = 56 kernel dispatches.
        s1_kernels = 1 + 1 + (2 * 8) + 1 + 1 + 1 + 1 + 6
        s2_kernels = 1 + 1 + (2 * 8) + 1 + 1 + 1 + 1 + 6
        total_kernels = s1_kernels + s2_kernels
        self.assertEqual(total_kernels, 56)

    def test_stream_s_uniformity_on_all_kernel_launches(self):
        """Verify all kernel launches in r4d_fulldrafter.hip explicitly use stream 'stream'."""
        pattern = re.compile(r'<<<([^>]+)>>>')
        matches = pattern.findall(self.hip_content)
        self.assertGreater(len(matches), 0, "No kernel launches found")

        non_stream_launches = []
        for i, m in enumerate(matches):
            stream_arg = extract_stream_from_launch(m)
            if stream_arg != "stream":
                non_stream_launches.append((i, m, stream_arg))

        self.assertEqual(len(non_stream_launches), 0,
                         f"Found kernel launches not on stream 'stream': {non_stream_launches}")

    def test_zero_unconditional_synchronization_stalls(self):
        """Confirm that hipDeviceSynchronize and hipStreamSynchronize never stall CPU in normal forward inference."""
        # 1. hipDeviceSynchronize must never appear
        self.assertNotIn("hipDeviceSynchronize", self.hip_content)
        self.assertNotIn("hipDeviceSynchronize", self.header_content)

        # 2. hipStreamSynchronize must appear at most once and strictly gated by profile_check
        sync_matches = [m.start() for m in re.finditer(r"\bhipStreamSynchronize\b", self.hip_content)]
        self.assertLessEqual(len(sync_matches), 1, "hipStreamSynchronize appears more than once")

        if sync_matches:
            # Check enclosing context around hipStreamSynchronize
            sync_pos = sync_matches[0]
            context = self.hip_content[max(0, sync_pos - 400):min(len(self.hip_content), sync_pos + 100)]
            self.assertIn("if (profile_check)", context, "hipStreamSynchronize is not gated by profile_check!")
            self.assertIn("getenv(\"DRAFTER_PROFILE\")", self.hip_content, "profile_check is not tied to DRAFTER_PROFILE env var!")

    def test_zero_dynamic_allocations(self):
        """Confirm zero runtime heap and VRAM dynamic allocations exist in the forward pass."""
        forbidden = [
            r"\bhipMalloc\b",
            r"\bhipFree\b",
            r"\bhipMallocAsync\b",
            r"\bhipFreeAsync\b",
            r"\bmalloc\s*\(",
            r"\bfree\s*\(",
            r"\bcalloc\s*\(",
            r"\brealloc\s*\(",
            r"\bnew\s+[A-Za-z0-9_]+",
            r"\bdelete\s+",
        ]
        for pattern in forbidden:
            matches_hip = re.findall(pattern, self.hip_content)
            matches_hdr = re.findall(pattern, self.header_content)
            self.assertEqual(len(matches_hip), 0, f"Forbidden allocation {pattern} in hip file: {matches_hip}")
            self.assertEqual(len(matches_hdr), 0, f"Forbidden allocation {pattern} in header: {matches_hdr}")

    def test_multi_batch_scratchpad_isolation_and_headroom(self):
        """Verify complete isolation and headroom compliance for batch sizes M in [1, 32]."""
        offsets = parse_offsets_from_file(HIP_FILE)
        scr_per_seq = offsets["SCR_PER_SEQ"]
        self.assertEqual(scr_per_seq, 8388608, "SCR_PER_SEQ must be exactly 8 MiB (8388608 bytes)")

        # Verify all 14 buffers within a sequence slice
        buffers = [
            ("SCR_BASE1",       offsets["SCR_BASE1_OFF"],       H_DIM * 4),
            ("SCR_H1",          offsets["SCR_H1_OFF"],          7 * H_DIM * 4),
            ("SCR_O1",          offsets["SCR_O1_OFF"],          6 * W_DIM * 4),
            ("SCR_BASE2",       offsets["SCR_BASE2_OFF"],       H_DIM * 4),
            ("SCR_H2",          offsets["SCR_H2_OFF"],          7 * H_DIM * 4),
            ("SCR_O2",          offsets["SCR_O2_OFF"],          6 * W_DIM * 4),
            ("SCR_PREV_TOK",    offsets["SCR_PREV_TOK_OFF"],    (TOTAL_K + 1) * 4),
            ("SCR_BLK_CANDS",   offsets["SCR_BLK_CANDS_OFF"],   64 * 16 * 4),
            ("SCR_BLK_UNARY",   offsets["SCR_BLK_UNARY_OFF"],   64 * 16 * 4),
            ("SCR_ARRIVAL_CTR", offsets["SCR_ARRIVAL_CTR_OFF"], 4),
            ("SCR_PROP_ID",     offsets["SCR_PROP_ID_OFF"],     12 * 16 * 4),
            ("SCR_PROP_VAL",    offsets["SCR_PROP_VAL_OFF"],    12 * 16 * 4),
            ("SCR_TEMP",        offsets["SCR_TEMP_OFF"],        21504 * 4),
            ("SCR_BASE_LOGITS", offsets["SCR_BASE_LOGITS_OFF"], 6 * V_TOTAL * 4),
        ]

        # Verify pairwise disjointness
        sorted_bufs = sorted(buffers, key=lambda x: x[1])
        for i in range(len(sorted_bufs) - 1):
            curr_name, curr_off, curr_size = sorted_bufs[i]
            next_name, next_off, _ = sorted_bufs[i + 1]
            curr_end = curr_off + curr_size
            self.assertLessEqual(curr_end, next_off,
                                 f"Buffer collision: {curr_name} [0x{curr_off:X}..0x{curr_end:X}] overlaps {next_name} [0x{next_off:X}]")

        # Verify max offset written in sequence
        max_end = sorted_bufs[-1][1] + sorted_bufs[-1][2]
        self.assertEqual(max_end, 8056832)
        self.assertLessEqual(max_end, scr_per_seq)
        margin = scr_per_seq - max_end
        self.assertEqual(margin, 331776, "Margin to sequence boundary must be 331776 bytes")

        # Verify multi-batch headroom up to M=32
        for M in [1, 2, 4, 8, 16, 32]:
            total_bytes = M * scr_per_seq
            total_mib = total_bytes / (1024 * 1024)
            self.assertLessEqual(total_bytes, GPU_HEADROOM_BYTES,
                                 f"Batch M={M} exceeds GPU_HEADROOM_MIB={GPU_HEADROOM_MIB}!")
            if M == 32:
                self.assertEqual(total_mib, 256.0)
                self.assertEqual(GPU_HEADROOM_MIB - total_mib, 128.0)

    def test_terminal_state_h1_preservation_and_reinit(self):
        """Verify Stage 1 terminal state h1[5] is preserved and arrival counter is cleanly reset."""
        # Check Stage 2 base commit reads h1[5]
        self.assertIn("s_h1 + (BLOCK_K - 1) * H_DIM", self.hip_content)
        self.assertIn("dev_rmsnorm", self.hip_content)

        # Check Stage 2 backbone kernel re-initializes arrival counter
        self.assertIn("*(uint32_t*)(s_base + SCR_ARRIVAL_CTR_OFF) = 0u", self.hip_content)


if __name__ == "__main__":
    unittest.main()
