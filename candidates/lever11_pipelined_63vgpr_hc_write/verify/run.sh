#!/bin/bash
# Byte-exactness check of the pipelined r4d_ar_hc_write_word against the base header, on a real
# gfx1201 card.
#
# The two kernels are the same launch shell; one inlines the base commit's header, the other this
# tree's header. Every output buffer is compared byte for byte, plus guard bytes, input
# immutability, a known-value write and a negative control that must make the buffers differ.
#
#   bash candidates/lever11_pipelined_63vgpr_hc_write/verify/run.sh [card]
#
# Environment: BASE (default cb5a69ac3f993560ed8dcb32fa524b3b9f987c62), ROCM (default
# /opt/rocm/core-10.0), CXX (default /opt/rocm/llvm/bin/clang++).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../../.." && pwd)
BASE=${BASE:-cb5a69ac3f993560ed8dcb32fa524b3b9f987c62}
ROCM=${ROCM:-/opt/rocm/core-10.0}
CXX=${CXX:-/opt/rocm/llvm/bin/clang++}
CARD=${1:-0}
OUT=$HERE/out
mkdir -p "$OUT/base/libr4d"
# BASE_HDR lets a caller substitute a modified copy of the base header (verify/mutation.sh does);
# without it the file is read from the base commit.
if [ -n "${BASE_HDR:-}" ]; then
    cp "$BASE_HDR" "$OUT/base/libr4d/r4d_ar_hc.h" || {
        echo "cannot read BASE_HDR=$BASE_HDR" >&2; exit 2; }
else
    git -C "$TREE" show "$BASE:libr4d/r4d_ar_hc.h" > "$OUT/base/libr4d/r4d_ar_hc.h" || {
        echo "cannot read $BASE:libr4d/r4d_ar_hc.h" >&2; exit 2; }
fi
FLAGS="-x hip --rocm-path=$ROCM --rocm-device-lib-path=$ROCM/amdgcn/bitcode --offload-arch=gfx1201 \
-std=c++17 -O3 -fno-fast-math -ffp-contract=off -I$TREE/abi"
# Same header name, two include roots: the base checkout for one unit, this tree for the other.
$CXX $FLAGS -I"$OUT/base/libr4d" -I"$TREE/libr4d" -c "$HERE/hc-baseline.hip" \
    -o "$OUT/hc-baseline.o" 2> "$OUT/hc-baseline.err"
r=$?; echo "hc-baseline compile rc=$r"; [ $r -ne 0 ] && { cat "$OUT/hc-baseline.err"; exit 1; }
$CXX $FLAGS -I"$TREE/libr4d" -c "$HERE/hc-candidate.hip" -o "$OUT/hc-candidate.o" \
    2> "$OUT/hc-candidate.err"
r=$?; echo "hc-candidate compile rc=$r"; [ $r -ne 0 ] && { cat "$OUT/hc-candidate.err"; exit 1; }
$CXX $FLAGS -I"$TREE/libr4d" -c "$HERE/compare.hip" -o "$OUT/compare.o" 2> "$OUT/compare.err"
r=$?; echo "compare compile rc=$r"; [ $r -ne 0 ] && { cat "$OUT/compare.err"; exit 1; }
# The objects are compiled HIP code; linking must not re-parse them as source.
$CXX --rocm-path="$ROCM" --offload-arch=gfx1201 -L"$ROCM/lib" -lamdhip64 \
    "$OUT/hc-baseline.o" "$OUT/hc-candidate.o" "$OUT/compare.o" -o "$OUT/hc-compare" \
    2> "$OUT/link.err"
r=$?; echo "link rc=$r"; [ $r -ne 0 ] && { cat "$OUT/link.err"; exit 1; }
"$OUT/hc-compare" "$CARD"
r=$?; echo "compare rc=$r"; exit $r
