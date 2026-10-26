#!/bin/bash
# exl3_convert.sh -- an exllamav3 (EXL3) checkpoint into a .rad container, weight for weight, with
# plugin/exl3import's convert-only architecture and the `exl3` importer. Nothing is quantised.
#
#   exl3_convert.sh SRC OUT.rad              # inventory, plan, convert
#   PLAN_ONLY=1 exl3_convert.sh SRC OUT.rad  # inventory and plan only: writes nothing but the hole file
#
# SRC is the checkpoint directory as downloaded (config.json, model.safetensors.index.json, the shards, the
# tokenizer files). It is read, never written.
#
# STEPS (each stops the run with a FAIL line and a non-zero exit):
#  1. tools/exl3_inventory.py SRC: every EXL3 linear by K and codebook, and every reason the container could
#     not be written (half-integer K, a layer whose experts differ, ...). Its report is OUT.inventory.json.
#  2. the hole file OUT.holes.safetensors: one <X>.weight a linear, sparse (~0 on disk), which rad-convert
#     reads through --add. Kept beside OUT and reused while it is newer than SRC's index.
#  3. free space on OUT's filesystem >= the inventory's predicted bytes + 5%.
#  4. rad-convert --plan-only: the plugin declares, the recipe plans every weight.
#  5. the convert, to OUT.part, renamed to OUT when rad-convert succeeds.
#
# THE RECORDED NAMES. rad-convert records the checkpoint as its last two path components and the recipe with
# $EXL3_SRC expanded, so it runs in SRC's parent directory and names SRC relatively: the container says
# "src=<SRC's directory name>", never a path of this machine.
#
# Env:
#   RAD_CONVERT       the rad-convert binary (default: rad-convert on PATH)
#   RADIANCE_HOME     the plugin home(s) rad-convert loads, ':'-separated; must hold exl3_import and exl3_quant
#                     (a build tree's radiance_home holds both). Default: rad-convert's own.
#   EXL3_IMPORT_ARCH  the architecture id the container carries. Default: the checkpoint's (config.json
#                     model_type without underscores, e.g. deepseekv4). The plugin that serves it must claim it.
#   PYTHON            default python3
#   PLAN_ONLY         1 = stop after step 4
#   RAD_CONVERT_ARGS  more rad-convert arguments, word-split (e.g. "--report 0")
set -euo pipefail

die() { printf 'exl3_convert: FAIL: %s\n' "$*" >&2; exit 1; }
say() { printf 'exl3_convert: %s\n' "$*"; }

[ $# -eq 2 ] || die "usage: exl3_convert.sh SRC OUT.rad"
here=$(cd "$(dirname "$0")/.." && pwd)
src=$(cd "$1" 2>/dev/null && pwd) || die "SRC '$1' is not a directory"
out_dir=$(cd "$(dirname "$2")" 2>/dev/null && pwd) || die "OUT's directory '$(dirname "$2")' does not exist"
out="$out_dir/$(basename "$2")"
: "${RAD_CONVERT:=rad-convert}"
: "${PYTHON:=python3}"
: "${PLAN_ONLY:=0}"
command -v "$RAD_CONVERT" >/dev/null || die "rad-convert not found (set RAD_CONVERT)"
[ -e "$out" ] && die "$out exists; this script never overwrites a container"
[ -f "$src/config.json" ] || die "$src has no config.json"

# 1. the inventory
inv="$out.inventory.json"
holes="$out.holes.safetensors"
index="$src/model.safetensors.index.json"
[ -f "$index" ] || index="$src/model.safetensors"
hole_args=(--holes "$holes")
if [ -f "$holes" ] && [ "$holes" -nt "$index" ]; then
    hole_args=()
    say "reusing $holes"
fi
"$PYTHON" -I "$here/tools/exl3_inventory.py" "$src" --json "$inv" "${hole_args[@]}" ||
    die "the inventory refuses $src (above; every finding is in $inv)"
[ -f "$holes" ] || die "no hole file at $holes"

arch=${EXL3_IMPORT_ARCH:-$("$PYTHON" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["arch"])' "$inv")}
[ -n "$arch" ] || die "config.json has no model_type; set EXL3_IMPORT_ARCH to the id the container should carry"
need=$("$PYTHON" -I -c 'import json,sys; b=json.load(open(sys.argv[1]))["bytes"]; print(int((b["exl3"]+b["plain"])*1.05)+(64<<20))' "$inv")

# 3. space
free=$(df -P -B1 "$out_dir" | awk 'NR==2 {print $4}')
if [ "$PLAN_ONLY" != 1 ] && [ "$free" -lt "$need" ]; then
    die "$out_dir has $((free >> 30)) GiB free and the container needs about $((need >> 30)) GiB"
fi

# 4-5. plan, then convert, from SRC's parent so the recorded names are relative
recipe="$out.recipe"
cat > "$recipe" <<'EOF'
# plugin/exl3import: every EXL3 linear imported at its own K and codebook; every other tensor as the
# checkpoint holds it. The importer declines a <X>.weight whose <X>.trellis the source does not hold.
*.weight  exl3  src=$EXL3_SRC bits=0 codebook=auto
EOF
name=$(basename "$src")
home_args=()
[ -n "${RADIANCE_HOME:-}" ] && home_args=(--home "$RADIANCE_HOME")
# shellcheck disable=SC2206
extra=(${RAD_CONVERT_ARGS:-})
run() {
    (cd "$(dirname "$src")" && EXL3_SRC="$name" EXL3_IMPORT_ARCH="$arch" EXL3_IMPORT_QUANT=exl3 \
        "$RAD_CONVERT" "$name" --add "$holes" --recipe "$recipe" --set "exl3_import.src=$name" \
        "${home_args[@]}" "${extra[@]}" "$@")
}
say "arch $arch, quant exl3; plan"
run --plan-only -o "$out.part" || die "rad-convert --plan-only refused (above)"
if [ "$PLAN_ONLY" = 1 ]; then
    say "PLAN_ONLY=1: stopping before the convert; $holes and $recipe are kept"
    exit 0
fi
say "convert to $out (about $((need >> 30)) GiB)"
rm -f "$out.part"
run -o "$out.part" || { rm -f "$out.part"; die "rad-convert failed (above)"; }
mv "$out.part" "$out"
say "wrote $out ($(du -h --apparent-size "$out" | cut -f1)); $holes can go"
