#!/usr/bin/env bash
# Machine comparison of one kernel between two revisions -- the PLE gather's entry, by default
# bdb4454 (where the row cache's stream stages landed) against the checked-out tree.
#
# It exports each revision with git archive into its own tree, compiles the one translation unit
# with the same hipcc line, takes the code object out of the object's .hip_fatbin, and prints and
# diffs, for ple_embed_lookup_q_kernel: the .text size, the mangled name, the kernarg segment
# (size, align and the per-argument block), and the disassembly. The disassemblies, the metadata
# and the function body are left as files next to the output, so the comparison can be read
# without trusting this script.
#
# No GPU, no benchmark: it compiles and reads ELF. Builds in this repo are serialised with
# /tmp/radiance-tp2-bench.lock, so run it under the lock:
#
#   flock -n -x /tmp/radiance-tp2-bench.lock tools/verify-ple-kernel-neutrality.sh
#
# usage: verify-ple-kernel-neutrality.sh [ref] [cur]     (default: bdb4454, HEAD)
# exit: 0 the two agree on all four; 1 they differ; 2 the run could not be made.
set -u

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REF=${1:-bdb4454}
CUR=${2:-HEAD}
ARCH=${ARCH:-gfx1201}
KERNEL=${KERNEL:-ple_embed_lookup_q_kernel}
OUT=${OUT:-$REPO/build/verify-ple-kernel-neutrality}

HIPCC=${HIPCC:-$(command -v hipcc || echo /opt/rocm/bin/hipcc)}
LLVM_BIN=${LLVM_BIN:-$(dirname "$(command -v llvm-objdump || echo /opt/rocm/llvm/bin/llvm-objdump)")}
for t in "$HIPCC" "$LLVM_BIN/llvm-objcopy" "$LLVM_BIN/llvm-objdump" \
         "$LLVM_BIN/clang-offload-bundler" python3 git tar; do
    command -v "$t" >/dev/null || { echo "missing tool: $t" >&2; exit 2; }
done

rm -rf "$OUT" && mkdir -p "$OUT" || exit 2

# One hipcc line: the flags cmake/RadAddPlugin.cmake gives libr4d's units (O3, uncompressed fat
# binary, no fp contraction; VOPD stays on, as this LLVM version is the repeatable one). Only each
# tree's include root differs, because each revision is compiled out of its own export.
hipcc_line() {  # $1 tree
    echo "$HIPCC -x hip -c --offload-arch=$ARCH -O3 -DNDEBUG -g0 --no-offload-compress" \
         "-ffp-contract=off -I$1/libr4d -I$1/abi" \
         "$1/libr4d/r4d_model_bf16.hip"
}

export_tree() {  # $1 rev  $2 tree
    mkdir -p "$2" || return 1
    git -C "$REPO" archive "$1" | tar -x -C "$2" || return 1
    [ -f "$2/libr4d/r4d_model_bf16.hip" ] ||
        { echo "$1 has no libr4d/r4d_model_bf16.hip" >&2; return 1; }
}

build_one() {  # $1 rev  $2 tree
    local rev=$1 tree=$2 obj=$OUT/$1.o cmd
    cmd=$(hipcc_line "$tree") || return 1
    echo "== $rev: $cmd" >&2
    # shellcheck disable=SC2086
    ( cd "$tree" && $cmd -o "$obj" ) || { echo "$rev: compile failed" >&2; return 1; }
    "$LLVM_BIN/llvm-objcopy" --dump-section .hip_fatbin="$OUT/$rev.fatbin" "$obj" || return 1
    "$LLVM_BIN/clang-offload-bundler" --unbundle --type=o \
        --targets="hip-amdgcn-amd-amdhsa--$ARCH" \
        --input="$OUT/$rev.fatbin" --output="$OUT/$rev.code.o" || return 1
}

# The kernel's metadata row, decoded from the code object's .note (msgpack): name, symbol, kernarg
# size and align, segment sizes, register counts, and the argument block itself.
metadata() {  # $1 code object
    "$LLVM_BIN/llvm-objcopy" --dump-section .note="$1.note" "$1" || return 1
    python3 - "$1.note" "$1.meta.txt" "$KERNEL" <<'PY'
import struct, sys
note, meta_out, kernel = sys.argv[1], sys.argv[2], sys.argv[3]
b = open(note, 'rb').read()
namesz, descsz, _ = struct.unpack_from('<III', b, 0)
desc = b[12 + ((namesz + 3) // 4) * 4:][:descsz]

def dec(i):
    t = desc[i]
    if t <= 0x7f: return t, i + 1
    if t >= 0xe0: return t - 0x100, i + 1
    if 0x80 <= t <= 0x8f: return ('map', t & 0xf), i + 1
    if 0x90 <= t <= 0x9f: return ('arr', t & 0xf), i + 1
    if 0xa0 <= t <= 0xbf:
        n = t & 0x1f; return desc[i + 1:i + 1 + n].decode(), i + 1 + n
    if t == 0xc0: return None, i + 1
    if t == 0xc2: return False, i + 1
    if t == 0xc3: return True, i + 1
    if t == 0xcc: return desc[i + 1], i + 2
    if t in (0xcd, 0xce, 0xcf):
        n = {0xcd: 2, 0xce: 4, 0xcf: 8}[t]
        return int.from_bytes(desc[i + 1:i + 1 + n], 'big'), i + 1 + n
    if t in (0xd0, 0xd1, 0xd2):
        n = {0xd0: 1, 0xd1: 2, 0xd2: 4}[t]
        return int.from_bytes(desc[i + 1:i + 1 + n], 'big', signed=True), i + 1 + n
    if t == 0xd9:
        n = desc[i + 1]; return desc[i + 2:i + 2 + n].decode(), i + 2 + n
    if t in (0xda, 0xdb):
        w = 2 if t == 0xda else 4
        n = int.from_bytes(desc[i + 1:i + 1 + w], 'big')
        return desc[i + 1 + w:i + 1 + w + n].decode(), i + 1 + w + n
    if t in (0xdc, 0xdd, 0xde, 0xdf):
        w = 2 if t in (0xdc, 0xde) else 4
        n = int.from_bytes(desc[i + 1:i + 1 + w], 'big')
        return ('map' if t in (0xde, 0xdf) else 'arr', n), i + 1 + w
    raise SystemExit('msgpack tag %02x at %d' % (desc[i], i))

def parse(i):
    v, i = dec(i)
    if isinstance(v, tuple):
        kind, n = v
        if kind == 'map':
            o = {}
            for _ in range(n):
                k, i = parse(i); o[k], i = parse(i)
            return o, i
        o = []
        for _ in range(n):
            x, i = parse(i); o.append(x)
        return o, i
    return v, i

meta, _ = parse(0)
open(meta_out, 'w').write(repr(meta))
for k in meta['amdhsa.kernels']:
    if kernel in (k.get('.name', '') + k.get('.symbol', '')):
        print('name    %s' % k['.name'])
        print('symbol  %s' % k['.symbol'])
        print('kernarg %d 0x%x align %d' % (k['.kernarg_segment_size'],
              k['.kernarg_segment_size'], k['.kernarg_segment_align']))
        print('group %d private %d uniform_workgroup %d' % (
            k['.group_segment_fixed_size'], k['.private_segment_fixed_size'],
            k['.uniform_work_group_size']))
        print('vgpr %d sgpr %d wavefront %d' % (k['.vgpr_count'], k['.sgpr_count'],
              k['.wavefront_size']))
        print('args')
        for a in k['.args']:
            print('  %-12s offset %-3d size %-2d %-16s %s' % (
                a.get('.name', ''), a['.offset'], a['.size'], a['.value_kind'],
                a.get('.address_space', '')))
        break
else:
    raise SystemExit('%s not in %s' % (kernel, note))
PY
}

describe() {  # $1 rev
    local rev=$1 code=$OUT/$1.code.o dis=$OUT/$1.disasm.txt meta sym text kernarg kd kd_says
    "$LLVM_BIN/llvm-objdump" -d --mcpu="$ARCH" "$code" >"$dis" || return 1
    meta=$(metadata "$code") || return 1
    # The body alone, and without the code addresses (an unrelated kernel's code cannot make this
    # look equal or unequal, and a shifted section address cannot make it look different). Every
    # instruction's mnemonic and encoding stays, and the raw listing stays in $rev.disasm.txt.
    awk -v sym="$KERNEL" '!p && index($0, sym) && /^[0-9a-f]+ </ {p=1}
                          p && /^$/ {exit} p' "$dis" |
        sed -E 's|^[0-9a-f]+ <|<|; s|[[:space:]]*// [0-9A-F]{8,}: |  |' \
        >"$OUT/$rev.body.txt"
    sym=$("$LLVM_BIN/llvm-objdump" -t "$code" |
          awk -v k="$KERNEL" '$3 == "F" && index($NF, k) {print $NF; exit}')
    text=$("$LLVM_BIN/llvm-objdump" -t "$code" |
           awk -v k="$KERNEL" '$3 == "F" && index($NF, k) {print $5; exit}')
    [ -n "$sym" ] || { echo "$rev: $KERNEL not in the code object" >&2; return 1; }
    printf '%s\n' "$meta"
    printf 'text    %s bytes (%d)\n' "$text" "$((16#$text))"
    printf 'mangled %s\n' "$sym"
    # The metadata's kernarg size against the kernel descriptor's own field (.kd+8): the number
    # above is then not one reading of one file.
    kernarg=$(printf '%s\n' "$meta" | awk '/^kernarg / {print $2}')
    kd=$("$LLVM_BIN/llvm-objdump" -t "$code" |
         awk -v s="$sym.kd" 'index($0, s) {print $1; exit}')
    if [ -n "$kd" ]; then
        kd_says=$(python3 - "$code" "$kd" <<'PY'
import struct, sys
code, addr = sys.argv[1], int(sys.argv[2], 16)
b = open(code, 'rb').read()
shoff, shentsize, shnum, shstrndx = (struct.unpack_from('<Q', b, 0x28)[0],
                                     struct.unpack_from('<H', b, 0x3a)[0],
                                     struct.unpack_from('<H', b, 0x3c)[0],
                                     struct.unpack_from('<H', b, 0x3e)[0])
secs = [struct.unpack_from('<IIQQQQIIQQ', b, shoff + i * shentsize) for i in range(shnum)]
strtab = secs[shstrndx]
for s in secs:
    name = b[strtab[4] + s[0]:b.index(b'\0', strtab[4] + s[0])].decode()
    if name == '.rodata':
        group, private, kernarg = struct.unpack_from('<III', b, s[4] + (addr - s[3]))
        print('%d group %d private %d' % (kernarg, group, private))
        break
else:
    raise SystemExit('no .rodata')
PY
        ) || { echo "$rev: cannot read the kernel descriptor" >&2; return 1; }
        printf 'kd      kernarg %s\n' "$kd_says"
        [ "${kd_says%% *}" = "$kernarg" ] ||
            { echo "$rev: .kd says ${kd_says%% *}, metadata says $kernarg" >&2; return 1; }
    else
        echo "$rev: no .kd for $KERNEL" >&2
        return 1
    fi
}

for rev in "$REF" "$CUR"; do
    tree=$OUT/tree-$rev
    export_tree "$rev" "$tree" || exit 2
    build_one "$rev" "$tree" || exit 2
    describe "$rev" >"$OUT/$rev.report.txt" || { cat "$OUT/$rev.report.txt"; exit 2; }
done

echo "kernel: $KERNEL, arch: $ARCH"
for rev in "$REF" "$CUR"; do
    echo "== $rev"
    sed 's/^/   /' "$OUT/$rev.report.txt"
done
echo
status=0
for f in report body; do
    if diff -q "$OUT/$REF.$f.txt" "$OUT/$CUR.$f.txt" >/dev/null; then
        echo "same: $f"
    else
        echo "DIFFERS: $f"
        diff -u "$OUT/$REF.$f.txt" "$OUT/$CUR.$f.txt" | sed 's/^/   /'
        status=1
    fi
done
echo "files: $OUT"
if [ $status = 0 ]; then
    echo "RESULT: $KERNEL is the same code and the same kernarg block in $REF and $CUR"
else
    echo "RESULT: $KERNEL differs between $REF and $CUR"
fi
exit $status
