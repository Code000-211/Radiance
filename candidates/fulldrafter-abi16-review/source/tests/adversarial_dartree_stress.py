"""Adversarial Stress Test Suite: 2D Causal Attention Mask Isolation & RoPE Depth Positions.

Milestone 2 Challenger 1 (Attention Mask & RoPE Adversary)

Empirically challenges:
1. r4d_attn_is_visible vs PyTorch DARTree Reference Mask (dartree.py)
2. Sibling & Disjoint Branch Strict Zero Attention (-inf)
3. Universal Historical Prefix Attention (score finite / 0.0)
4. Strict Ancestor Attending Invariant (k >= prefix_len iff j in ancestors(i))
5. Extreme Topologies:
   - Deep linear chain (N=16, depth 15)
   - Flat star (N=16, root + 15 children at depth 1)
   - Complete binary tree (N=15, depth 3)
   - 5,000 randomized trees / DAGs
6. RoPE Depth Positions:
   - pos(i) = root_pos + depth(i)
   - Siblings receive identical RoPE position IDs
"""

import math
import random
import sys
import time

sys.path.insert(0, "/home/orca/Drafter")
from research.block_drafter.tree.dartree import (
    CandidateTree,
    TreeNode,
)


def r4d_attn_is_visible_sim(
    k: int,
    prefix_len: int,
    ctx: int,
    klimit: int,
    mask: int,
    wlo: int = 0,
    win: bool = False,
    cz: bool = True,
) -> bool:
    """Exact bitwise simulation of r4d_attn_is_visible from libr4d/r4d_attn_decode_h256_gqa6.hip."""
    if win and (k < wlo or (not cz and k > klimit)):
        return False
    if k < prefix_len:
        return True
    if k >= ctx:
        return False
    tree_k = k - prefix_len
    return (tree_k < 16) and ((mask & (1 << tree_k)) != 0)


def test_extreme_topologies_against_pytorch_reference() -> int:
    print("[PY-TEST 1] Testing Extreme Topologies against PyTorch Reference Mask...")
    assertion_count = 0

    # 1.1 Deep Linear Chain (N=16, depth 15)
    chain_nodes = [TreeNode(node_id=0, parent_id=None, token_id=100, depth=0)]
    for i in range(1, 16):
        chain_nodes.append(TreeNode(node_id=i, parent_id=i - 1, token_id=100 + i, depth=i))
    tree_chain = CandidateTree.from_nodes(chain_nodes)

    prefix_len = 128
    ctx = prefix_len + 16
    root_pos = 1000
    ref_mask_chain = tree_chain.build_full_attention_mask(prefix_len=prefix_len)
    ref_pos_chain = tree_chain.build_position_ids(root_position=root_pos)

    # Check lower triangular structure and RoPE monotonicity
    for i in range(16):
        assert ref_pos_chain[i].item() == root_pos + i
        assertion_count += 1
        q_row = prefix_len + i
        # Ancestor mask for linear chain
        mask_i = (1 << (i + 1)) - 1
        for j in range(ctx):
            vis = r4d_attn_is_visible_sim(j, prefix_len, ctx, ctx - 1, mask_i)
            ref_score = ref_mask_chain[q_row, j].item()
            is_ref_visible = not math.isinf(ref_score)
            assert vis == is_ref_visible, f"Mismatch at chain query {i}, key {j}: sim={vis}, ref={is_ref_visible}"
            assertion_count += 1

    # 1.2 Flat Star (N=16, root + 15 children at depth 1)
    star_nodes = [TreeNode(node_id=0, parent_id=None, token_id=500, depth=0)]
    for i in range(1, 16):
        star_nodes.append(TreeNode(node_id=i, parent_id=0, token_id=500 + i, depth=1))
    tree_star = CandidateTree.from_nodes(star_nodes)

    ref_mask_star = tree_star.build_full_attention_mask(prefix_len=prefix_len)
    ref_pos_star = tree_star.build_position_ids(root_position=root_pos)

    # All 15 children must have identical RoPE position IDs
    assert ref_pos_star[0].item() == root_pos
    for i in range(1, 16):
        assert ref_pos_star[i].item() == root_pos + 1
        assertion_count += 1

    # Disjoint cross-sibling attention check: ALL 210 pairs must have -inf score
    for i in range(1, 16):
        mask_i = (1 << 0) | (1 << i)
        q_row = prefix_len + i
        for j in range(1, 16):
            k_col = prefix_len + j
            vis = r4d_attn_is_visible_sim(k_col, prefix_len, ctx, ctx - 1, mask_i)
            ref_score = ref_mask_star[q_row, k_col].item()
            if i == j:
                assert vis is True
                assert not math.isinf(ref_score)
            else:
                assert vis is False, f"Sibling leakage between {i} and {j}!"
                assert math.isinf(ref_score) and ref_score < 0, f"PyTorch ref failed to mask sibling between {i} and {j}!"
            assertion_count += 2

    # 1.3 Complete Binary Tree (N=15, depth 3)
    bin_nodes = [TreeNode(node_id=0, parent_id=None, token_id=700, depth=0)]
    for i in range(1, 15):
        parent = (i - 1) // 2
        depth = bin_nodes[parent].depth + 1
        bin_nodes.append(TreeNode(node_id=i, parent_id=parent, token_id=700 + i, depth=depth))
    tree_bin = CandidateTree.from_nodes(bin_nodes)

    ref_mask_bin = tree_bin.build_full_attention_mask(prefix_len=prefix_len)
    ref_pos_bin = tree_bin.build_position_ids(root_position=root_pos)

    # Depth RoPE positions:
    for i in range(15):
        expected_depth = bin_nodes[i].depth
        assert ref_pos_bin[i].item() == root_pos + expected_depth
        assertion_count += 1

    # Check all query-key pairs
    for i in range(15):
        q_row = prefix_len + i
        # Compute ancestor mask
        ancestors = {a.node_id for a in tree_bin.get_ancestors(i)}
        ancestors.add(i)
        mask_i = sum(1 << a for a in ancestors)
        bin_ctx = prefix_len + 15
        for j in range(bin_ctx):
            vis = r4d_attn_is_visible_sim(j, prefix_len, bin_ctx, bin_ctx - 1, mask_i)
            ref_score = ref_mask_bin[q_row, j].item()
            is_ref_visible = not math.isinf(ref_score)
            assert vis == is_ref_visible, f"Binary tree mismatch query {i}, key {j}"
            assertion_count += 1

    print(f"  PASS: Extreme topologies matched PyTorch reference with {assertion_count} verified assertions.")
    return assertion_count


def test_random_dag_fuzzing(num_trials: int = 5000) -> int:
    print(f"[PY-TEST 2] Fuzzing {num_trials} Random Valid Tree/DAG Topologies...")
    rng = random.Random(42)
    assertion_count = 0

    for trial in range(num_trials):
        n = rng.randint(2, 16)
        prefix_len = rng.choice([0, 1, 4, 15, 16, 31, 64, 128, 512])
        root_pos = rng.randint(0, 1000000)

        # Build random tree
        nodes = [TreeNode(node_id=0, parent_id=None, token_id=1000, depth=0)]
        for i in range(1, n):
            parent = rng.randint(0, i - 1)
            depth = nodes[parent].depth + 1
            nodes.append(TreeNode(node_id=i, parent_id=parent, token_id=1000 + i, depth=depth))

        tree = CandidateTree.from_nodes(nodes)
        ref_mask = tree.build_full_attention_mask(prefix_len=prefix_len)
        ref_pos = tree.build_position_ids(root_position=root_pos)

        ctx = prefix_len + n
        klimit = ctx - 1

        # Collect ancestor bitmasks
        ancestor_masks = []
        for i in range(n):
            anc = {a.node_id for a in tree.get_ancestors(i)}
            anc.add(i)
            mask = sum(1 << a for a in anc)
            ancestor_masks.append((anc, mask))

        # Check all nodes
        for i in range(n):
            anc, mask_i = ancestor_masks[i]
            q_row = prefix_len + i

            # RoPE depth assignment: pos(i) == root_pos + depth(i)
            assert ref_pos[i].item() == root_pos + nodes[i].depth
            assertion_count += 1

            # Sibling equality check
            for j in range(n):
                if nodes[i].depth == nodes[j].depth:
                    assert ref_pos[i].item() == ref_pos[j].item()
                    assertion_count += 1

            # Historical prefix visibility
            for k in range(prefix_len):
                vis = r4d_attn_is_visible_sim(k, prefix_len, ctx, klimit, mask_i)
                ref_score = ref_mask[q_row, k].item()
                assert vis is True
                assert not math.isinf(ref_score)
                assertion_count += 2

            # Candidate key visibility
            for j in range(n):
                k = prefix_len + j
                vis = r4d_attn_is_visible_sim(k, prefix_len, ctx, klimit, mask_i)
                ref_score = ref_mask[q_row, k].item()
                is_anc = j in anc

                assert vis == is_anc, f"Trial {trial}: visibility mismatch query {i}, key {j}"
                assert (not math.isinf(ref_score)) == is_anc, f"Trial {trial}: ref score mismatch query {i}, key {j}"
                assertion_count += 2

            # Out of bounds check
            for k in range(ctx, ctx + 5):
                vis = r4d_attn_is_visible_sim(k, prefix_len, ctx, klimit, mask_i)
                assert vis is False
                assertion_count += 1

    print(f"  PASS: {num_trials} random trees fuzzed successfully with {assertion_count} verified assertions.")
    return assertion_count


def main() -> None:
    print("=" * 70)
    print(" Python Adversarial Stress Test: Mask Isolation & RoPE Positions")
    print("=" * 70)
    t0 = time.time()
    c1 = test_extreme_topologies_against_pytorch_reference()
    c2 = test_random_dag_fuzzing(5000)
    elapsed = time.time() - t0
    total = c1 + c2
    print("=" * 70)
    print(" ALL PYTHON ADVERSARIAL TESTS PASSED SUCCESSFULLY!")
    print(f" Total Assertions Verified: {total}")
    print(f" Elapsed Time: {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
