#ifndef RADIANCE_ARCH_RAD_RDNA4_ISA_H
#define RADIANCE_ARCH_RAD_RDNA4_ISA_H

#include <cstdint>
#include <algorithm>
#include <cmath>

namespace rad {
namespace isa {
namespace rdna4 {

/* ==============================================================================
 * Rule 1: VGPR Allocation Granularity (24 for Wave32)
 *
 * Microarchitectural Ground Truth:
 * - AMD RDNA4 (gfx1201) SIMD register file contains 1536 32-bit VGPRs per SIMD.
 * - Hardware allocation quanta: 24 VGPRs for Wave32, 12 VGPRs for Wave64.
 * - Max active Wave32 wavefronts per SIMD = 16 (100% occupancy).
 * - Max VGPRs per thread in Wave32 = 256.
 *
 * Occupancy table (Wave32):
 *   <= 96 VGPRs: 16 waves (100.0%)
 *   120   VGPRs: 12 waves (75.0%)
 *   144   VGPRs: 10 waves (62.5%)  <-- 128 VGPRs falls into this bucket!
 *   168   VGPRs:  9 waves (56.25%)
 *   192   VGPRs:  8 waves (50.0%)
 *   216   VGPRs:  7 waves (43.75%)
 *   240   VGPRs:  6 waves (37.5%)
 *   256   VGPRs:  5 waves (31.25%)
 * ============================================================================== */
constexpr int kSimdVgprs = 1536;
constexpr int kWave32VgprGranularity = 24;
constexpr int kWave64VgprGranularity = 12;
constexpr int kMaxWave32WavesPerSimd = 16;
constexpr int kMaxVgprsPerWave32 = 256;

constexpr int allocated_vgprs_wave32(int vgpr) {
    if (vgpr <= 0) return 0;
    return ((vgpr + kWave32VgprGranularity - 1) / kWave32VgprGranularity) * kWave32VgprGranularity;
}

constexpr int simd_occupancy_waves(int vgpr) {
    if (vgpr <= 0) return kMaxWave32WavesPerSimd;
    int alloc = allocated_vgprs_wave32(vgpr);
    return std::min(kMaxWave32WavesPerSimd, kSimdVgprs / alloc);
}

constexpr float simd_occupancy_pct(int vgpr) {
    return (static_cast<float>(simd_occupancy_waves(vgpr)) / static_cast<float>(kMaxWave32WavesPerSimd)) * 100.0f;
}

constexpr bool is_in_cliff_danger_zone(int vgpr) {
    constexpr int cliffs[] = {96, 120, 144, 168, 192, 216, 240};
    for (int b : cliffs) {
        if (vgpr > b && vgpr <= b + 8) return true;
    }
    return false;
}

/* ==============================================================================
 * Rule 2: Zero-Waterfall Loop Standard
 *
 * Microarchitectural Ground Truth:
 * - Dynamic VGPR indexing uses the scalar M0 register.
 * - Divergent indices across wavefront lanes force LLVM to emit v_movrels_b32
 *   inside s_and_saveexec_b32 / s_cbranch_execnz serialization loops.
 * - All unrolling on wavefront paths must use compile-time constant bounds
 *   (templates or #pragma unroll) with predication inside rather than dynamic loop limits.
 * ============================================================================== */

/* ==============================================================================
 * Rule 3: Dual-Issue VOPD Constraints
 *
 * Microarchitectural Ground Truth:
 * - OpX has 14 opcodes (0..13), OpY has 17 opcodes (0..13, 16..18).
 * - Legal only in Wave32 mode.
 * - Destination parity: one even, one odd (vdstY[0] == ~vdstX[0]).
 * - Bank conflict: 4 banks (reg % 4). Sources must come from different banks
 *   or be identical registers.
 * - Dual SRC2 parity: if both operations read SRC2 (FMAC, FMAMK, DOT2ACC),
 *   one SRC2 must be even and the other odd.
 * ============================================================================== */
constexpr int kVopdOpXCount = 14;
constexpr int kVopdOpYCount = 17;
constexpr int kVgprBanks = 4;

constexpr bool vopd_valid_dest_parity(int dstX, int dstY) {
    return (dstX & 1) != (dstY & 1);
}

constexpr bool vopd_valid_bank_pairing(int srcX, int srcY) {
    return (srcX == srcY) || ((srcX % kVgprBanks) != (srcY % kVgprBanks));
}

constexpr bool vopd_valid_dual_src2(int src2X, int src2Y) {
    return (src2X & 1) != (src2Y & 1);
}

/* ==============================================================================
 * Rule 4: Canonical K-Ordering for GLOBAL_LOAD_TR
 *
 * Microarchitectural Ground Truth:
 * - global_load_tr_b64 and _b128 load in AMD hardware canonical K-order:
 *     Lanes 0..15  hold K-indices {0, 1, 2, 3,  8, 9, 10, 11}
 *     Lanes 16..31 hold K-indices {4, 5, 6, 7, 12, 13, 14, 15}
 * - Invariant: Both operands to WMMA must follow this exact order.
 * ============================================================================== */
constexpr int kCanonicalK_Lo[8] = {0, 1, 2, 3, 8, 9, 10, 11};
constexpr int kCanonicalK_Hi[8] = {4, 5, 6, 7, 12, 13, 14, 15};

constexpr int canonical_k_order(int lane, int slot) {
    int half = lane / 16;  // 0 for lanes 0..15, 1 for lanes 16..31
    if (slot < 4) {
        return half * 4 + slot;
    } else {
        return 8 + half * 4 + (slot - 4);
    }
}

/* ==============================================================================
 * Rule 5: Split Cache Temporal Hints (TH)
 *
 * Microarchitectural Ground Truth:
 * - Near (CU L0) vs Far (L2) cache retention policies:
 *   0: RT    (Regular temporal near + far)
 *   1: NT    (Non-temporal near + far, streaming stores)
 *   2: HT    (High-priority temporal near + far)
 *   3: LU    (Last-use: non-temporal + discard dirty)
 *   4: NT_RT (Non-temporal near / bypass L0, regular far L2 -> layer weight streaming)
 *   5: RT_NT (Regular near, non-temporal far)
 *   6: NT_HT (Non-temporal near / bypass L0, high-priority far L2 -> resident weights)
 *   7: NT_WB (Non-temporal near, write-back far)
 * ============================================================================== */
enum class TemporalHint : uint8_t {
    RT    = 0,
    NT    = 1,
    HT    = 2,
    LU    = 3,
    NT_RT = 4,
    RT_NT = 5,
    NT_HT = 6,
    NT_WB = 7,
};

constexpr bool th_is_weight_streaming(TemporalHint th) {
    return th == TemporalHint::NT_RT || th == TemporalHint::NT_HT;
}

constexpr bool th_is_last_use(TemporalHint th) {
    return th == TemporalHint::LU;
}

constexpr bool th_is_streaming_store(TemporalHint th) {
    return th == TemporalHint::NT || th == TemporalHint::NT_WB;
}

/* ==============================================================================
 * Rule 6: Independent Wait Counters & s_wait_alu Guard
 *
 * Microarchitectural Ground Truth:
 * - Deprecates legacy s_waitcnt.
 * - Independent counters: s_wait_loadcnt, s_wait_storecnt, s_wait_dscnt, s_wait_kmcnt.
 * - Hardware hazard guards for SGPR / VCC register pairs:
 *     0xFFFD: VALU write to VCC -> VALU read of VCC
 *     0xF1FF: VALU write to SGPR -> VALU read of SGPR
 *     0xFFFE: SALU write to SGPR/VCC -> SALU or VALU read of SGPR/VCC
 * ============================================================================== */
constexpr uint16_t kWaitAluValuVcc  = 0xFFFD;
constexpr uint16_t kWaitAluValuSgpr = 0xF1FF;
constexpr uint16_t kWaitAluSaluAny  = 0xFFFE;

/* ==============================================================================
 * Rule 7: Early Deallocation Epilogue Contract
 *
 * Microarchitectural Ground Truth:
 * - Emit s_sendmsg sendmsg(MSG_DEALLOC_VGPRS) BEFORE s_wait_storecnt 0.
 * - Wave releases its VGPR allocation back to the SIMD allocator while memory
 *   stores flush, allowing new wavefronts to launch without waiting for DRAM ACK.
 * ============================================================================== */
enum class EpilogueStep {
    IssueStores,
    DeallocVGPRs,
    WaitStores,
    EndPgm
};

constexpr bool validate_epilogue_order(bool has_dealloc, bool has_wait_stores, int dealloc_idx, int wait_stores_idx) {
    if (!has_dealloc || !has_wait_stores) return false;
    return dealloc_idx < wait_stores_idx;
}

} // namespace rdna4
} // namespace isa
} // namespace rad

#endif // RADIANCE_ARCH_RAD_RDNA4_ISA_H
