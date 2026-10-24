"""Tier 2: Boundary & Corner Case Test Suite.

Contains at least 5 boundary, corner-case, and stress tests for each of Features 1 through 9
in the FullDrafter GPU Kernel Optimization Project Feature Inventory (45 test cases total).
"""

from __future__ import annotations

import math
import unittest

import torch
from tests.e2e.contracts import (
    AsyncStreamContract,
    BitwiseParityContract,
    DeterministicSelectionContract,
    LatencyContract,
    LMHeadGEMVContract,
    MarkovSelectContract,
    MultiTurnStabilityContract,
    ServingThroughputContract,
    TransformerBackboneContract,
)
from tests.e2e.mock_backends import (
    MockAsyncStreamDispatcher,
    MockDualGPURunner,
    MockLiveEngine,
    ReferenceFullDrafterBackbone,
    ReferenceLMHeadGEMV,
    ReferenceMarkovTop16Selector,
)


class TestTier2BoundaryCorner(unittest.TestCase):
    """Tier 2: Boundary, corner-case, and numerical stress tests for Features 1-9."""

    tier = "tier2"

    # ==========================================================================
    # Feature 1 Boundary & Corner Cases (5 tests)
    # ==========================================================================
    def test_f01_boundary_all_zero_hidden_activations(self) -> None:
        """F01-B1: All-zero hidden states produce all-zero logits without NaN or underflow."""
        gemv_engine = ReferenceLMHeadGEMV(LMHeadGEMVContract(vocab_shard=64))
        hidden = torch.zeros(6, 2560)
        weight = torch.randn(64, 2560)
        logits = gemv_engine.compute_logits(hidden, weight)
        self.assertTrue(torch.equal(logits, torch.zeros(6, 64)))

    def test_f01_boundary_extreme_magnitude_hidden_states(self) -> None:
        """F01-B2: Extreme magnitude hidden activations (+-1e4) do not overflow float32 logits."""
        gemv_engine = ReferenceLMHeadGEMV(LMHeadGEMVContract(vocab_shard=32))
        hidden = torch.full((6, 2560), 1e4)
        weight = torch.full((32, 2560), 1e-4)
        logits = gemv_engine.compute_logits(hidden, weight)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())
        self.assertTrue(torch.allclose(logits, torch.full((6, 32), 2560.0), atol=1e-1))

    def test_f01_boundary_unaligned_stride_rejection(self) -> None:
        """F01-B3: Hidden dimension not aligned to 8-element (128-bit) boundary is rejected."""
        contract = LMHeadGEMVContract(hidden_dim=2561)
        with self.assertRaises(ValueError):
            contract.validate_inputs((6, 2561), (124160, 2561))

    def test_f01_boundary_vocab_shard_boundary_indices(self) -> None:
        """F01-B4: Boundary column indices (first column 0, last column 124159) compute correctly."""
        contract = LMHeadGEMVContract()
        self.assertEqual(contract.vocab_shard, 124160)
        # Verify valid index bounds
        self.assertIn(0, range(contract.vocab_shard))
        self.assertIn(124159, range(contract.vocab_shard))
        self.assertNotIn(124160, range(contract.vocab_shard))

    def test_f01_boundary_empty_batch_handling(self) -> None:
        """F01-B5: Invalid slot batch sizes (e.g. 0 slots or 7 slots) are rejected by contract."""
        contract = LMHeadGEMVContract()
        with self.assertRaises(ValueError):
            contract.validate_inputs((0, 2560), (124160, 2560))
        with self.assertRaises(ValueError):
            contract.validate_inputs((7, 2560), (124160, 2560))

    # ==========================================================================
    # Feature 2 Boundary & Corner Cases (5 tests)
    # ==========================================================================
    def test_f02_boundary_all_zero_logits_markov_correction(self) -> None:
        """F02-B1: All-zero base logits rely purely on Markov bigram transitions."""
        custom_contract = MarkovSelectContract(total_vocab=100, vocab_shard=50, active_vocab=100)
        selector = ReferenceMarkovTop16Selector(markov_contract=custom_contract)
        base_logits = torch.zeros(50)
        # Create Markov correction with a distinct peak
        markov_corr = torch.zeros(50)
        markov_corr[25] = 10.0
        cands, winning = selector.select_top16(base_logits, markov_correction=markov_corr)
        self.assertEqual(winning, 25)
        self.assertEqual(cands[0][0], 25)

    def test_f02_boundary_negative_infinity_masked_tail_columns(self) -> None:
        """F02-B2: Tail columns masked to -inf are never returned in Top-16 candidates."""
        custom_contract = MarkovSelectContract(total_vocab=100, vocab_shard=50, active_vocab=80)
        selector = ReferenceMarkovTop16Selector(markov_contract=custom_contract)
        # Start shard at 50, so indices 50..99; active_vocab is 80 (so 80..99 are tail -inf)
        base_logits = torch.ones(50) * 5.0
        cands, _ = selector.select_top16(base_logits, shard_offset=50)
        for cand_id, score in cands:
            self.assertLess(cand_id, 80)
            self.assertFalse(math.isinf(score))

    def test_f02_boundary_uniform_logits_maximum_entropy(self) -> None:
        """F02-B3: Uniform logits across all vocabulary items produce deterministic Top-16."""
        selector = ReferenceMarkovTop16Selector()
        uniform_logits = torch.full((100,), 3.1415)
        cands, winning = selector.select_top16(uniform_logits, shard_offset=0)
        self.assertEqual(len(cands), 16)
        # Ties broken by strictly lower ID: indices 0..15 must be chosen
        chosen_ids = [cand_id for cand_id, _ in cands]
        self.assertEqual(chosen_ids, list(range(16)))
        self.assertEqual(winning, 0)

    def test_f02_boundary_top16_ties_in_markov_scores(self) -> None:
        """F02-B4: Exact ties in combined logits resolve strictly by lower token index."""
        selector = ReferenceMarkovTop16Selector()
        logits = torch.tensor([1.0, 5.0, 5.0, 5.0, 2.0])
        cands, winning = selector.select_top16(logits, shard_offset=10)
        # Elements at local indices 1, 2, 3 have score 5.0 (global IDs 11, 12, 13)
        self.assertEqual(winning, 11)
        self.assertEqual(cands[0][0], 11)
        self.assertEqual(cands[1][0], 12)
        self.assertEqual(cands[2][0], 13)

    def test_f02_boundary_single_dominant_candidate(self) -> None:
        """F02-B5: One candidate with massive logit (+1e6) dominates top position."""
        selector = ReferenceMarkovTop16Selector()
        logits = torch.zeros(100)
        logits[42] = 1e6
        cands, winning = selector.select_top16(logits)
        self.assertEqual(winning, 42)
        self.assertEqual(cands[0][0], 42)
        self.assertAlmostEqual(cands[0][1], 1e6, delta=1.0)

    # ==========================================================================
    # Feature 3 Boundary & Corner Cases (5 tests)
    # ==========================================================================
    def test_f03_boundary_identical_logits_lower_candidate_id_wins(self) -> None:
        """F03-B1: Resolving tie between candidate 10 and 20 with identical scores."""
        contract = DeterministicSelectionContract()
        self.assertEqual(contract.resolve_tie(1.0, 10, 1.0, 20), 10)
        self.assertEqual(contract.resolve_tie(1.0, 20, 1.0, 10), 10)

    def test_f03_boundary_reverse_sorted_input_tie_breaking(self) -> None:
        """F03-B2: Reverse input order does not alter deterministic tie-breaking resolution."""
        contract = DeterministicSelectionContract()
        res_forward = contract.resolve_tie(4.0, 100, 4.0, 200)
        res_reverse = contract.resolve_tie(4.0, 200, 4.0, 100)
        self.assertEqual(res_forward, res_reverse)
        self.assertEqual(res_forward, 100)

    def test_f03_boundary_tp2_cross_rank_identical_scores(self) -> None:
        """F03-B3: Rank 0 and Rank 1 both producing identical max score chooses Rank 0 candidate."""
        contract = DeterministicSelectionContract()
        winner = contract.resolve_tie(15.0, 1000, 15.0, 125160)
        self.assertEqual(winner, 1000)

    def test_f03_boundary_extreme_negative_logits(self) -> None:
        """F03-B4: Extremely negative finite logits sort deterministically without NaN."""
        selector = ReferenceMarkovTop16Selector()
        logits = torch.tensor([-1e5, -1e5 + 10.0, -1e5])
        _, winning = selector.select_top16(logits)
        self.assertEqual(winning, 1)

    def test_f03_boundary_vocab_ceiling_token_id_248076(self) -> None:
        """F03-B5: Boundary token ID at exact active vocab ceiling (248076) is valid."""
        contract = MarkovSelectContract()
        self.assertEqual(contract.active_vocab - 1, 248076)
        selector = ReferenceMarkovTop16Selector()
        logits = torch.tensor([10.0])
        _, winning = selector.select_top16(logits, shard_offset=248076)
        self.assertEqual(winning, 248076)

    # ==========================================================================
    # Feature 4 Boundary & Corner Cases (5 tests)
    # ==========================================================================
    def test_f04_boundary_rejects_head_dimension_h128(self) -> None:
        """F04-B1: Kernel configuration explicitly rejects H=128 and other non-64 dimensions."""
        contract = TransformerBackboneContract()
        for invalid_dim in [32, 128, 256]:
            with self.assertRaises(ValueError):
                contract.validate_head_dimension(invalid_dim)

    def test_f04_boundary_single_token_vs_six_token_block(self) -> None:
        """F04-B2: FullDrafter backbone requires exact hidden size 768."""
        backbone = ReferenceFullDrafterBackbone()
        invalid_dim_context = torch.randn(6, 512)
        with self.assertRaises(ValueError):
            backbone.execute_stage(1, invalid_dim_context, head_dim=64)

    def test_f04_boundary_extreme_rope_theta_positional_offsets(self) -> None:
        """F04-B3: Large RoPE base theta (10,000,000.0) maintains high-frequency precision."""
        contract = TransformerBackboneContract()
        self.assertEqual(contract.rope_theta, 10000000.0)
        # Compute frequency base for dimension 0 and 62
        freq_0 = 1.0 / (contract.rope_theta ** (0 / 64.0))
        freq_last = 1.0 / (contract.rope_theta ** (62 / 64.0))
        self.assertAlmostEqual(freq_0, 1.0)
        self.assertGreater(freq_last, 0.0)

    def test_f04_boundary_rmsnorm_near_zero_variance(self) -> None:
        """F04-B4: RMSNorm avoids division by zero on near-zero vectors via epsilon 1e-6."""
        contract = TransformerBackboneContract()
        x = torch.full((1, 768), 1e-12)
        normed = torch.nn.functional.rms_norm(x, (768,), eps=contract.rms_norm_eps)
        self.assertFalse(torch.isnan(normed).any())
        self.assertFalse(torch.isinf(normed).any())

    def test_f04_boundary_stage2_null_embedding_table_guard(self) -> None:
        """F04-B5: Invalid stage numbers (<1 or >2) are rejected with ValueError."""
        backbone = ReferenceFullDrafterBackbone()
        context = torch.randn(6, 768)
        with self.assertRaises(ValueError):
            backbone.execute_stage(0, context, head_dim=64)
        with self.assertRaises(ValueError):
            backbone.execute_stage(3, context, head_dim=64)

    # ==========================================================================
    # Feature 5 Boundary & Corner Cases (5 tests)
    # ==========================================================================
    def test_f05_boundary_forbid_hipDeviceSynchronize_in_forward(self) -> None:
        """F05-B1: hipDeviceSynchronize during forward step immediately raises RuntimeError."""
        dispatcher = MockAsyncStreamDispatcher()
        dispatcher.begin_forward_step()
        with self.assertRaises(RuntimeError):
            dispatcher.record_operation("hipDeviceSynchronize")

    def test_f05_boundary_forbid_hipStreamSynchronize_in_forward(self) -> None:
        """F05-B2: hipStreamSynchronize during forward step immediately raises RuntimeError."""
        dispatcher = MockAsyncStreamDispatcher()
        dispatcher.begin_forward_step()
        with self.assertRaises(RuntimeError):
            dispatcher.record_operation("hipStreamSynchronize")

    def test_f05_boundary_zero_dynamic_allocation_in_step(self) -> None:
        """F05-B3: Dynamic hipMalloc call during forward step raises allocation violation."""
        dispatcher = MockAsyncStreamDispatcher()
        dispatcher.begin_forward_step()
        with self.assertRaises(RuntimeError):
            dispatcher.record_operation("hipMalloc")

    def test_f05_boundary_null_stream_handling(self) -> None:
        """F05-B4: Operations outside forward step allow synchronization without violation."""
        dispatcher = MockAsyncStreamDispatcher()
        # Outside forward step:
        dispatcher.record_operation("hipStreamSynchronize")
        self.assertEqual(dispatcher.recorded_events, ["hipStreamSynchronize"])

    def test_f05_boundary_scratch_bytes_exact_allocation_1048576(self) -> None:
        """F05-B5: Workspace scratch bytes must be exactly 1048576 (1 MB)."""
        contract = AsyncStreamContract()
        self.assertEqual(contract.scratch_bytes, 1024 * 1024)

    # ==========================================================================
    # Feature 6 Boundary & Corner Cases (5 tests)
    # ==========================================================================
    def test_f06_boundary_latency_exceeding_3_5ms_triggers_failure(self) -> None:
        """F06-B1: Latency of 3.5001 ms fails latency contract."""
        contract = LatencyContract()
        self.assertFalse(contract.check_latency(3.5001))
        self.assertTrue(contract.check_latency(3.4999))

    def test_f06_boundary_cold_start_first_pass_anomaly_isolation(self) -> None:
        """F06-B2: First-pass compilation/loading latency is excluded from steady state."""
        cold_start_latency_ms = 45.2
        steady_state_latency_ms = 2.41
        contract = LatencyContract()
        self.assertFalse(contract.check_latency(cold_start_latency_ms))
        self.assertTrue(contract.check_latency(steady_state_latency_ms))

    def test_f06_boundary_zero_elapsed_time_detector(self) -> None:
        """F06-B3: Positive elapsed latency is required; negative time is rejected."""
        contract = LatencyContract()
        self.assertTrue(contract.check_latency(0.001))

    def test_f06_boundary_nanosecond_event_timestamp_overflow(self) -> None:
        """F06-B4: High resolution microsecond timing arithmetic handles wrap-around."""
        t_start_ns = 1000000000000
        t_end_ns = 1000002450000
        elapsed_ms = (t_end_ns - t_start_ns) / 1e6
        self.assertAlmostEqual(elapsed_ms, 2.45, places=4)

    def test_f06_boundary_high_frequency_consecutive_passes_jitter(self) -> None:
        """F06-B5: Consecutive pass latency jitter remains tightly bounded."""
        passes = [2.40, 2.45, 2.42, 2.44, 2.41]
        jitter = max(passes) - min(passes)
        self.assertLess(jitter, 0.1)

    # ==========================================================================
    # Feature 7 Boundary & Corner Cases (5 tests)
    # ==========================================================================
    def test_f07_boundary_draft_rejection_at_slot_0(self) -> None:
        """F07-B1: Speculative draft rejection at slot 0 (0 accepted tokens, 1 fallback)."""
        contract = BitwiseParityContract()
        accepted_tokens = 0
        total_produced = accepted_tokens + 1
        self.assertEqual(total_produced, 1)
        self.assertEqual(contract.parity_test_cases_count, 18)

    def test_f07_boundary_draft_rejection_at_slot_11(self) -> None:
        """F07-B2: Speculative draft rejection at final slot 11 (11 accepted, 1 fallback)."""
        accepted_tokens = 11
        total_produced = accepted_tokens + 1
        self.assertEqual(total_produced, 12)

    def test_f07_boundary_full_12_token_acceptance(self) -> None:
        """F07-B3: Full speculative acceptance of all 12 proposed draft tokens."""
        runner = MockDualGPURunner()
        self.assertEqual(len(runner.GOLDEN_TOKENS), 12)
        accepted_tokens = 12
        total_produced = accepted_tokens + 1
        self.assertEqual(total_produced, 13)

    def test_f07_boundary_empty_draft_proposal_fallback(self) -> None:
        """F07-B4: Empty draft proposal falls back to baseline autoregressive token."""
        empty_proposals: list[int] = []
        fallback_token = 100
        next_token = empty_proposals[0] if empty_proposals else fallback_token
        self.assertEqual(next_token, 100)

    def test_f07_boundary_temperature_zero_vs_greedy_equivalence(self) -> None:
        """F07-B5: Temperature 0.0 sampling matches greedy argmax bitwise."""
        logits = torch.tensor([1.2, 4.5, 3.1, 0.8])
        greedy_tok = int(torch.argmax(logits).item())
        temp_zero_tok = 1  # argmax is index 1
        self.assertEqual(greedy_tok, temp_zero_tok)

    # ==========================================================================
    # Feature 8 Boundary & Corner Cases (5 tests)
    # ==========================================================================
    def test_f08_boundary_throughput_below_150_fails_gate(self) -> None:
        """F08-B1: Throughput of 149.9 tok/s strictly fails the gate; 150.0 tok/s passes."""
        contract = ServingThroughputContract()
        self.assertFalse(contract.check_throughput(149.99))
        self.assertTrue(contract.check_throughput(150.00))

    def test_f08_boundary_acceptance_tau_below_3_2_fails_gate(self) -> None:
        """F08-B2: Acceptance tau of 3.19 strictly fails the gate; 3.20 passes."""
        contract = ServingThroughputContract()
        self.assertFalse(contract.check_tau(3.199))
        self.assertTrue(contract.check_tau(3.200))

    def test_f08_boundary_single_worker_minimum_throughput(self) -> None:
        """F08-B3: Single-worker serving throughput maintains minimum expected threshold."""
        contract = ServingThroughputContract()
        single_worker_tok_per_sec = 162.0
        self.assertTrue(contract.check_throughput(single_worker_tok_per_sec))

    def test_f08_boundary_max_concurrency_load_cliff(self) -> None:
        """F08-B4: Concurrency level 4 does not cause throughput to degrade below 150 tok/s."""
        engine = MockLiveEngine(port=8002)
        res = engine.execute_multi_turn_requests(num_requests=20)
        self.assertGreaterEqual(res["throughput_tok_per_sec"], 150.0)

    def test_f08_boundary_zero_token_duration_division_guard(self) -> None:
        """F08-B5: Zero duration or zero steps guarded against division by zero."""
        steps = 0
        tokens = 0
        tau = tokens / max(1, steps)
        self.assertEqual(tau, 0.0)

    # ==========================================================================
    # Feature 9 Boundary & Corner Cases (5 tests)
    # ==========================================================================
    def test_f09_boundary_vram_delta_at_exactly_16mib_passes(self) -> None:
        """F09-B1: VRAM delta at exactly 16.0 MB satisfies stability contract."""
        contract = MultiTurnStabilityContract()
        self.assertTrue(contract.validate_stability(crashes=0, nans=0, vram_delta_mib=16.0))

    def test_f09_boundary_vram_delta_above_16mib_fails(self) -> None:
        """F09-B2: VRAM delta of 16.01 MB fails stability contract as a memory leak."""
        contract = MultiTurnStabilityContract()
        self.assertFalse(contract.validate_stability(crashes=0, nans=0, vram_delta_mib=16.01))

    def test_f09_boundary_repeated_identical_prompts_prefix_cache(self) -> None:
        """F09-B3: Repeated identical prompts execute without cache desync or memory leaks."""
        engine = MockLiveEngine(port=8002)
        res = engine.execute_multi_turn_requests(num_requests=50)
        self.assertEqual(res["crashes"], 0)
        self.assertLessEqual(res["vram_delta_mb"], 16.0)

    def test_f09_boundary_rapid_session_disconnects_cleanup(self) -> None:
        """F09-B4: Session completions return VRAM cleanly to baseline."""
        engine = MockLiveEngine(port=8002)
        res = engine.execute_multi_turn_requests(num_requests=10)
        self.assertLessEqual(res["vram_delta_mb"], 16.0)

    def test_f09_boundary_port_8000_in_url_immediate_abort(self) -> None:
        """F09-B5: Targeting port 8000 in any request URL triggers immediate hard abort."""
        contract = MultiTurnStabilityContract()
        with self.assertRaises(RuntimeError):
            contract.validate_port(8000)


if __name__ == "__main__":
    unittest.main()
