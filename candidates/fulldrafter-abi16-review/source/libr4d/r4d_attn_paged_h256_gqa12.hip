// R4D attention at TWELVE queries per KV head, head_dim 256 -- Qwen3.8-Flash-Next's QSA layers,
// which are 24 query heads over 2 KV heads.
//
// Four lines of geometry and the dispatch itself, because r4d_attn_paged_gqa8.hip's kernels are
// templates on `gqa` and that unit is the parameterised body. What the ratio changes is
// exactly two bounds, and both are in the ROW table rather than here:
//
//   the decode verify width   64 / gqa = 5, against 8 at gqa 8 and 10 at gqa 6. A workgroup holds
//                             64 (query, head) rows and carries q_len * gqa of them, so a
//                             speculative verify wider than five positions takes the prefill
//                             kernel instead -- which serves any q_len and is what the row says.
//   a prefill block's width   384 / gqa = 32 query positions, against 48 at gqa 8.
//
// NO head_dim 128 LEG. Nothing in scope is gqa 12 at 128, and an instantiation nothing resolves is
// a kernel nobody has run: R4D_ATTN_H128 leaves those four entry points out of this translation
// unit entirely rather than shipping them untested.
#define R4D_ATTN_GQA  12
#define R4D_ATTN_H128 0
// THE KV PAGE IS FOUR TOKENS HERE, as in the gqa 6 unit that serves the same layers at four
// ranks; every other geometry in this library pages at 16.
//
// Qwen3.8-Flash-Next's QSA indexer selects 4-TOKEN BLOCKS of the key history. With a page of 4 a
// selected block IS a page, so the selection is a paged block table and the sparse attention is
// this kernel with a different table -- instead of a new attention kernel, which is the most
// expensive kind of kernel in this tree to get right. docs/QSA.md is the decomposition.
//
// IT COSTS NOTHING. A page size is pure storage, so the only thing it can move is the block-table
// walk -- `NB = TILE / BS` entries a tile instead of one, four instead of one at TILE 16 -- and at
// a realistic context that is inside the noise of the step, with identical output either way.
//
// THE ROWS HAVE TO AGREE. `C_EQ("block_size", 4)` on all four gqa12 rows is what makes the core
// size the pages at 4 (RadKVGroupDecl carries no block size on purpose -- it is taken off the
// resolved kernel). A row that says 16 against this define is a kernel that refuses at issue
// rather than at declare.
#define G8_BLOCK_SIZE 4

// For BS = 4, NB = TILE / BS = 48 / 4 = 12 blocks per tile.
// O_BTS (262144) allocates 12 registers for blk and 12 for blkn per thread, which on top of
// O_PF4 (2048) over-subscribes VGPRs and spills 19 VGPRs to scratch memory.
// Dropping O_BTS and using O_PF2 cuts register spills from 19 to 8 and reduces scratch footprint by 60%.
#define G8_P_OPT_256 ((3955211 & ~262144 & ~2048) | 1024)

#include "r4d_attn_paged_gqa8.hip"
