#!/bin/sh
# run.sh -- build this candidate and run the part of its test that needs no card.
#
#   sh test/run.sh [SOURCE_TREE]
#
# SOURCE_TREE is the radiance source the row is compiled against, defaulting to the tree this
# candidate lives in (../..). Only the ABI headers and libr4d's sources are read from it, so an
# installed prefix is not needed; that is also why this is not the plugin's CMake build (which
# does want one). What it produces is the same DSO: rows.cpp, libr4d's int8 layout, and the device
# source, with the ABI's export set and nothing else visible.
#
# The compilers are ROCm's clang for both languages, as the candidate's CMakeLists requires:
# hipcc's wrapper would add its own flags to the host half.
#
# NO CARD IS TOUCHED. The DSO links libamdhip64, so every child here runs with HIP_VISIBLE_DEVICES
# and ROCR_VISIBLE_DEVICES emptied: HIP initialises without a device, a launch that got past the
# body's refusals would fail instead of running, and no KFD context is claimed. Each child also
# runs under `timeout`, because a child that hangs keeps whatever context it did claim, and a
# leaked KFD context on this host is not free: it holds 150+ MB that the scheduler's measured
# budget then misses by (tests/kbench_mutant_test.cpp and tests/oot_plugins.sh hide devices the
# same way). Neither is optional.
set -eu
S=${1:-"$(cd "$(dirname "$0")/../.." && pwd)/.."}
ROCM=${ROCM:-/opt/rocm}
CLANG=$ROCM/llvm/bin/clang++
ROCMLIB=${ROCM_LIB:-$ROCM/core-10.0}
ARCH=${ARCH:-gfx1201}
LIMIT=${LIMIT:-120}
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT

[ -f "$S/libr4d/r4d_fp8_layout.cpp" ] || { echo "not a radiance source tree: $S" >&2; exit 2; }

echo "building shared_i8a8_down.so from $S"
"$CLANG" -std=c++17 -O2 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
         -ffp-contract=off -I"$S/abi" -I"$S/libr4d" -I"$S/candidates/shared-i8a8-down" \
         -c "$S/candidates/shared-i8a8-down/rows.cpp" -o "$D/rows.o"
"$CLANG" -std=c++17 -O2 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
         -ffp-contract=off -I"$S/abi" -I"$S/libr4d" -I"$S/candidates/shared-i8a8-down" \
         -c "$S/libr4d/r4d_fp8_layout.cpp" -o "$D/layout.o"
"$CLANG" -x hip --rocm-path="$ROCMLIB" --rocm-device-lib-path="$ROCMLIB/amdgcn/bitcode" \
         --offload-arch="$ARCH" -std=c++17 -O3 -fno-fast-math -ffp-contract=off -fPIC \
         -I"$S/abi" -I"$S/libr4d" -I"$S/candidates/shared-i8a8-down" \
         -c "$S/candidates/shared-i8a8-down/shared_i8a8_down.hip" -o "$D/hip.o"
"$CLANG" -shared -fPIC -o "$D/shared_i8a8_down.so" "$D/rows.o" "$D/hip.o" "$D/layout.o" \
         -L"$ROCMLIB/lib" -lamdhip64 -ldl

# The export set is the ABI's, and the DSO carries no undefined symbol from libr4d: the layout
# hooks are compiled in, not linked out.
exports=$(nm -D --defined-only "$D/shared_i8a8_down.so" | awk '$2 == "T" { print $3 }' | sort)
want=$(printf '%s\n' rad_kernel_at rad_kernel_concurrent rad_kernel_count rad_kernel_schema_at \
                       rad_kernel_schema_count rad_plugin_abi_version rad_plugin_info | sort)
[ "$exports" = "$want" ] || { echo "FAIL: exported symbols differ from the ABI's set:" >&2
                              echo "$exports" >&2; exit 1; }
if nm -D -u "$D/shared_i8a8_down.so" | grep -q r4d_; then
  echo "FAIL: the DSO has an undefined libr4d symbol" >&2; exit 1
fi
echo "exports: the ABI's set, nothing from libr4d"

"$CLANG" -std=c++17 -O1 -Wall -Wextra -I"$S/abi" \
         "$S/candidates/shared-i8a8-down/test/si8_test.cpp" -o "$D/si8_test" -ldl

echo
# A child that outlives its case is a leaked KFD context, so every one of them is bounded here and
# sealed off from the card. See the note at the top of this file.
NOCARD="HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=99 RADIANCE_HOST_VRAM_MIB=512"
echo "--- plugin off (RADIANCE_SHARED_I8A8_DOWN unset)"
env -u RADIANCE_SHARED_I8A8_DOWN $NOCARD timeout "$LIMIT" \
    "$D/si8_test" --off "$D/shared_i8a8_down.so"
echo "--- plugin on (RADIANCE_SHARED_I8A8_DOWN=1)"
env $NOCARD RADIANCE_SHARED_I8A8_DOWN=1 timeout "$LIMIT" \
    "$D/si8_test" --on "$D/shared_i8a8_down.so"
echo
echo "host tests passed. Not run here: the numerics (compare.hip) and every served-path number,"
echo "both of which need a gfx1201 card."
