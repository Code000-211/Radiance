#!/bin/bash
# convert_test.sh PYTHON RAD_CONVERT RADIANCE_HOME WORKDIR -- the end-to-end convert (ctest exl3_import_convert):
#  1. a synthetic DeepSeek-shaped EXL3 checkpoint (tests/make_fixture.py) through tools/exl3_convert.sh, and every
#     entry of the container checked against the checkpoint (tests/check_container.py);
#  2. the same container with one codes byte flipped must FAIL the check (the check can fail);
#  3. a checkpoint with a half-integer K, a layer of mixed experts and a linear not in 128s must be refused
#     before rad-convert runs, each by name, with no container written.
# Exit 77 (skipped) without numpy.
set -euo pipefail
py=$1 conv=$2 home=$3 work=$4
here=$(cd "$(dirname "$0")/.." && pwd)
"$py" -c 'import numpy' 2>/dev/null || { echo "SKIPPED: $py with numpy is needed"; exit 77; }
rm -rf "$work" && mkdir -p "$work/models" "$work/out"
export RAD_CONVERT=$conv RADIANCE_HOME=$home PYTHON=$py

"$py" -I "$here/tests/make_fixture.py" "$work/models/dsv4-exl3"
"$here/tools/exl3_convert.sh" "$work/models/dsv4-exl3" "$work/out/dsv4.rad"
"$py" -I "$here/tests/check_container.py" "$work/out/dsv4.rad" "$work/models/dsv4-exl3" --arch deepseekv4

"$py" -I -c '
import sys; sys.path.insert(0, sys.argv[3])
from check_container import Rad
r = Rad(sys.argv[1]); b = bytearray(r.b)
e = next(x for x in r.entries if x["name"] == "model.layers.0.self_attn.q_proj.weight")
b[e["planes"][0][0] + 5] ^= 0x10
open(sys.argv[2], "wb").write(b)' "$work/out/dsv4.rad" "$work/out/mutated.rad" "$here/tests"
if "$py" -I "$here/tests/check_container.py" "$work/out/mutated.rad" "$work/models/dsv4-exl3" --arch deepseekv4; then
    echo "FAIL: a flipped codes byte passed the check"; exit 1
fi

"$py" -I "$here/tests/make_fixture.py" "$work/models/bad-exl3" --half-k --mixed-experts --odd-dims
if "$here/tools/exl3_convert.sh" "$work/models/bad-exl3" "$work/out/bad.rad" > "$work/bad.log" 2>&1; then
    echo "FAIL: the bad checkpoint converted"; exit 1
fi
grep -q "half-integer K = 3.5" "$work/bad.log" || { echo "FAIL: the half-integer K was not named"; exit 1; }
grep -q "differs between experts" "$work/bad.log" || { echo "FAIL: the mixed experts were not named"; exit 1; }
grep -q "wkv_a: \[192, 256\].*multiples of 128" "$work/bad.log" || { echo "FAIL: the 192-row linear was not named"; exit 1; }
[ ! -e "$work/out/bad.rad" ] || { echo "FAIL: a container was written for the bad checkpoint"; exit 1; }
echo "convert_test: PASS"
