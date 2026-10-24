#!/bin/sh
# One row per device kernel in a build tree: VGPRs, SGPRs, LDS bytes, scratch bytes, spilled
# registers, the waves a SIMD the VGPR count alone allows, and the workgroup size.
#
#   BUILD=/path/to/build res.sh
#
# SPILL IS THE LOUD COLUMN, and SCRATCH after it: spilled registers are a memory round trip the
# source does not show, and scratch without spills is a private array the compiler could not keep
# in registers. WAVES is the other one: on gfx1201 a SIMD has 1536 VGPRs in wave32 allocated in
# groups of 24, so 256 VGPRs is 5 waves, 240 is 6 waves, and 128 is 10 waves (allocating 144) --
# and LDS can cut it further, which this column does NOT model.
#
# The metadata is msgpack printed as YAML and its keys are ALPHABETICAL, so .vgpr_count comes after
# .symbol and .sgpr_count before it. Emitting a row at .symbol prints the previous kernel's VGPR
# count. The row is flushed at .wavefront_size, which is last.
: "${ROCM:=/opt/rocm}"
: "${BUILD:=build}"
: "${ISA_OUT:=/tmp/r4disa}"
D="$ISA_OUT/res"; rm -rf "$D"; mkdir -p "$D"
here=$(cd "$(dirname "$0")" && pwd)
for o in $(find "$BUILD" -name '*.hip.o'); do
  n=$(basename "$o" .hip.o)
  ISA_OUT="$D" "$here/co.sh" "$o" >/dev/null 2>&1 || continue
  "$ROCM/llvm/bin/llvm-readelf" --notes "$D/$n.co" 2>/dev/null > "$D/$n.md"
done
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' vgpr sgpr lds scratch spill waves wg kernel
awk -v OFS='\t' '
  /^ *\.group_segment_fixed_size:/   { lds = $2 }
  /^ *\.private_segment_fixed_size:/ { sc  = $2 }
  /^ *\.sgpr_count:/                 { s   = $2 }
  /^ *\.sgpr_spill_count:/           { ss  = $2 }
  /^ *\.vgpr_count:/                 { v   = $2 }
  /^ *\.vgpr_spill_count:/           { vs  = $2 }
  /^ *\.max_flat_workgroup_size:/    { wg  = $2 }
  /^ *\.name:/                       { nm  = $2 }
  /^ *\.wavefront_size:/ { if (nm != "") { g = int((v + 23) / 24) * 24; w = (g > 0) ? int(1536 / g) : 16
                                           if (w > 16) w = 16
                                           print v, s, lds, sc, vs + ss, w, wg, nm; nm = "" } }
' "$D"/*.md | sort -k5,5nr -k4,4nr -k1,1nr
