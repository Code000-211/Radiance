#!/bin/bash
# Proves the comparison harness has teeth: mutates the base header and requires the comparison to
# FAIL. The mutation is a wrong store (the last stream is skipped when hc > 1), so the hc = 1
# known-value check still passes and only the byte comparison can catch it. Nothing in the tree is
# touched: the mutated copy is passed to run.sh via BASE_HDR.
#
#   bash candidates/lever11_pipelined_63vgpr_hc_write/verify/mutation.sh [card]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../../.." && pwd)
BASE=${BASE:-cb5a69ac3f993560ed8dcb32fa524b3b9f987c62}
OUT=$HERE/out
mkdir -p "$OUT/base/libr4d"
SRC=$OUT/base/libr4d/r4d_ar_hc.h
git -C "$TREE" show "$BASE:libr4d/r4d_ar_hc.h" > "$SRC" || {
    echo "cannot read $BASE:libr4d/r4d_ar_hc.h" >&2; exit 2; }

echo "== the two headers the harness compiles actually differ"
if diff -q "$SRC" "$TREE/libr4d/r4d_ar_hc.h" > /dev/null; then
    echo "FAIL base and candidate header are identical -- nothing to compare" >&2; exit 1
fi
diff -u "$SRC" "$TREE/libr4d/r4d_ar_hc.h" | grep -cE '^[+-][^+-]' | sed 's/^/  changed lines: /'

MUT=$OUT/base.libr4d.h.mutated
python3 - "$SRC" "$MUT" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as fh:
    s = fh.read()
old = "  for (int c = 0; c < hc; ++c) {"
assert old in s, "anchor not found in the base header"
s = s.replace(old, "  for (int c = 0; c < hc - (hc > 1 ? 1 : 0); ++c) {", 1)
with open(dst, "w") as fh:
    fh.write(s)
PY
echo "== mutated base: last stream not written when hc > 1; comparison must FAIL"
BASE_HDR=$MUT bash "$HERE/run.sh" "${1:-0}" 2>&1 | tail -4
r=${PIPESTATUS[0]}
echo "mutated run rc=$r (expected nonzero)"
[ "$r" -ne 0 ] || { echo "FAIL harness did not catch a wrong store" >&2; exit 1; }
rm -f "$MUT"
echo "== pass: the comparison rejected the mutated header"
