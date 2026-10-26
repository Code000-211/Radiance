#!/usr/bin/env python3
"""check_container.py -- every entry of a .rad that plugin/exl3import wrote, against the EXL3 checkpoint it came from.

    check_container.py CONTAINER SRC --arch ID

Reads the container with its own reader of abi/rad_format.h (not radiance's) and checks:
  the header's architecture id and quantisation descriptor;
  one entry for every EXL3 linear (<X>.weight) and every other tensor, and nothing else;
  an EXL3 entry: scheme exl3, the codes plane at the trellis's K, the codebook its marker names, the logical shape
    [n, k], its codes plane equal to the numpy reference's (exl3_ref.codes_plane, written from the format
    statement, independently of the C importer), suh and svh byte-equal to the source's;
  any other entry: plain, its dtype and shape the source's, its bytes the source's;
  the movement groups (layer, expert, slot) the README's contract states.
Exit 0 with one summary line, or 1 naming every mismatch.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exl3_ref as ref  # noqa: E402

MAX_RANK, ENC_STR, ENC_MAX_PLANES = 6, 24, 8
DT = {1: "f32", 2: "f16", 3: "bf16", 4: "fp8_e4m3", 5: "fp8_e5m2", 8: "i8", 9: "u8", 10: "i16", 11: "i32", 12: "i64",
      13: "u32", 20: "u2", 24: "u3", 19: "u4", 26: "u5", 28: "u6", 30: "u7"}
ST = {"F32": "f32", "F16": "f16", "BF16": "bf16", "F8_E4M3": "fp8_e4m3", "I8": "i8", "U8": "u8", "I16": "i16",
      "I32": "i32", "I64": "i64", "U32": "u32"}
MARKERS = {"mul1": 0x83DCD12D, "mcg": 0xCBAC1FED}


class Rad:
    def __init__(self, path):
        self.b = Path(path).read_bytes()
        h = struct.unpack_from("<IIQ5Q16Q", self.b, 0)
        if h[0] != 0x31444152:
            raise SystemExit(f"{path}: not a .rad")
        (self.str_off, _, _, _, self.dir_off, self.dir_n, self.plane_off, _, self.enc_off, _) = h[8:18]
        self.arch, self.name, self.quant = (self.s(h[3]), self.s(h[4]), self.s(h[5]))
        self.entries = [self.entry(i) for i in range(self.dir_n)]

    def s(self, off):
        if off == 0:
            return ""
        p = self.str_off + off
        return self.b[p: self.b.index(b"\0", p)].decode()

    def enc(self, i):
        p = self.enc_off + i * 568
        cstr = lambda q: self.b[q: q + ENC_STR].split(b"\0")[0].decode()  # noqa: E731
        n = struct.unpack_from("<i", self.b, p + 48)[0]
        planes = []
        for k in range(n):
            q = p + 56 + k * 64
            dtype, kind, b0, b1 = struct.unpack_from("<IIqq", self.b, q + ENC_STR)
            planes.append({"role": cstr(q), "dtype": DT.get(dtype, dtype), "kind": kind, "block": (b0, b1)})
        return {"scheme": cstr(p), "transform": cstr(p + ENC_STR), "planes": planes}

    def entry(self, i):
        p = self.dir_off + i * 136
        name, qz, opts, enc, rank = struct.unpack_from("<QQQII", self.b, p)
        shape = struct.unpack_from(f"<{MAX_RANK}q", self.b, p + 32)[:rank]
        layer, expert, slot, n_planes, plane_first, off, nbytes = struct.unpack_from("<iiiIQQQ", self.b, p + 80)
        planes = [struct.unpack_from("<QQ", self.b, self.plane_off + 16 * (plane_first + k)) for k in range(n_planes)]
        return {"name": self.s(name), "quantizer": self.s(qz), "options": self.s(opts), "enc": self.enc(enc),
                "shape": list(shape), "group": (layer, expert, slot), "planes": planes}

    def plane(self, e, k):
        off, n = e["planes"][k]
        return self.b[off: off + n]


def load_src(src):
    src = Path(src)
    out = {}
    for fn in sorted(set(json.loads((src / "model.safetensors.index.json").read_text())["weight_map"].values())):
        raw = (src / fn).read_bytes()
        (n,) = struct.unpack_from("<Q", raw, 0)
        h = json.loads(raw[8: 8 + n])
        h.pop("__metadata__", None)
        for k, v in h.items():
            a, b = v["data_offsets"]
            out[k] = (v["dtype"], v["shape"], raw[8 + n + a: 8 + n + b])
    return out, json.loads((src / "config.json").read_text())


def groups(forms, n_layers):
    """(layer, expert, slot) for every name, as the README's contract states it. `forms` maps each name to its
    form (encoding and shape); a layer's expert group is its most common whole expert form, ties to the
    lowest-numbered expert's, numbered densely; every other expert's weights are tensors of the layer."""
    def lay(n):
        parts, got = n.split("."), None
        for i in range(len(parts) - 2):
            if parts[i] == "layers" and parts[i + 1].isdigit():
                got = (".".join(parts[:i]), int(parts[i + 1]))
        return got

    names = sorted(forms)
    count = {}
    for n in names:
        g = lay(n)
        if g:
            count[g[0]] = max(count.get(g[0], 0), g[1] + 1)
    trunk = next((p for p in sorted(count) if count[p] == n_layers), min(count) if count else None)
    base, nxt = {trunk: 0}, max(n_layers, count.get(trunk, 0))
    for p in sorted(count):
        if p != trunk:
            base[p], nxt = nxt, nxt + count[p]
    info, units = {}, {}
    for n in names:
        g = lay(n)
        layer = base[g[0]] + g[1] if g else -1
        parts = n.split(".")
        real, key = -1, None
        for i in range(1, len(parts) - 2):
            if parts[i] == "experts" and parts[i + 1].isdigit():
                real, key = int(parts[i + 1]), ".".join(parts[: i + 1] + ["*"] + parts[i + 2:])
                break
        info[n] = (layer, real, key)
        if layer >= 0 and real >= 0:
            units.setdefault(layer, {}).setdefault(real, {})[key] = forms[n]
    place, keys = {}, {}
    for layer, experts in units.items():
        keys[layer] = sorted(set().union(*(set(u) for u in experts.values())))
        by_form, order = {}, []
        for real in sorted(experts):
            whole = tuple(experts[real].get(k) for k in keys[layer])
            if whole not in by_form:
                order.append(whole)
            by_form.setdefault(whole, []).append(real)
        best = order[0]
        for f in order:
            if len(by_form[f]) > len(by_form[best]):
                best = f
        place[layer] = {real: i for i, real in enumerate(by_form[best])}
    out = {}
    for n, (layer, real, key) in info.items():
        if real >= 0 and layer >= 0 and real in place[layer]:
            out[n] = (layer, place[layer][real], keys[layer].index(key))
        else:
            out[n] = (layer, -1, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("container")
    ap.add_argument("src")
    ap.add_argument("--arch", required=True)
    a = ap.parse_args()
    rad = Rad(a.container)
    src, cfg = load_src(a.src)
    bad = []
    if rad.arch != a.arch:
        bad.append(f"header arch '{rad.arch}', not '{a.arch}'")
    if rad.quant != "exl3":
        bad.append(f"header quant '{rad.quant}', not 'exl3'")

    linears = sorted(n[: -len(".trellis")] for n in src if n.endswith(".trellis"))
    parts = {f"{x}.{p}" for x in linears for p in ("trellis", "suh", "svh", "mul1", "mcg")}
    want = {x + ".weight" for x in linears} | {n for n in src if n not in parts}
    have = {e["name"]: e for e in rad.entries}
    for n in sorted(want - set(have)):
        bad.append(f"{n}: not in the container")
    for n in sorted(set(have) - want):
        bad.append(f"{n}: in the container, not in the checkpoint")
    forms = {}
    for x in linears:
        sh = src[x + ".trellis"][1]
        cb = next((m for m in MARKERS if x + "." + m in src), "3inst")
        forms[x + ".weight"] = ("exl3", sh[2] // 16, cb, sh[1] * 16, sh[0] * 16)
    for n in want - {x + ".weight" for x in linears}:
        forms[n] = ("plain", src[n][0], tuple(src[n][1]))
    grp = groups(forms, cfg["num_hidden_layers"])

    n_exl3 = n_plain = 0
    for x in linears:
        e = have.get(x + ".weight")
        if not e:
            continue
        dt, sh, raw = src[x + ".trellis"]
        tr = np.frombuffer(raw, dtype=np.int16).reshape(sh)
        bits, k, n = sh[2] // 16, sh[0] * 16, sh[1] * 16
        cb = next((m for m in MARKERS if x + "." + m in src), "3inst")
        enc = e["enc"]
        roles = [p["role"] for p in enc["planes"]]
        if enc["scheme"] != "exl3" or enc["transform"] != f"{cb}/h128" or roles != ["codes", "suh", "svh"]:
            bad.append(f"{x}: encoding {enc['scheme']} {enc['transform']} {roles}, not exl3 {cb}/h128 codes,suh,svh")
            continue
        if enc["planes"][0]["dtype"] != f"u{bits}":
            bad.append(f"{x}: codes {enc['planes'][0]['dtype']}, not u{bits}")
        if e["shape"] != [n, k]:
            bad.append(f"{x}: shape {e['shape']}, not [{n}, {k}]")
        if rad.plane(e, 0) != ref.codes_plane(tr, bits).tobytes():
            bad.append(f"{x}: the codes plane differs from the reference's")
        if rad.plane(e, 1) != src[x + ".suh"][2]:
            bad.append(f"{x}: suh differs from the source")
        if rad.plane(e, 2) != src[x + ".svh"][2]:
            bad.append(f"{x}: svh differs from the source")
        if e["quantizer"] != "exl3":
            bad.append(f"{x}: written by '{e['quantizer']}', not exl3")
        if e["group"] != grp[x + ".weight"]:
            bad.append(f"{x}: group {e['group']}, the contract says {grp[x + '.weight']}")
        n_exl3 += 1
    for name in sorted(want - {x + ".weight" for x in linears}):
        e = have.get(name)
        if not e:
            continue
        dt, sh, raw = src[name]
        enc = e["enc"]
        if enc["scheme"] != "plain" or enc["planes"][0]["dtype"] != ST.get(dt):
            bad.append(f"{name}: {enc['scheme']} {enc['planes'][0]['dtype']}, not plain {ST.get(dt)}")
        if e["shape"] != sh:
            bad.append(f"{name}: shape {e['shape']}, not {sh}")
        if rad.plane(e, 0) != raw:
            bad.append(f"{name}: bytes differ from the source")
        if e["group"] != grp[name]:
            bad.append(f"{name}: group {e['group']}, the contract says {grp[name]}")
        n_plain += 1

    for x in bad:
        print(f"check_container: {x}")
    print(f"check_container: {a.container}: {n_exl3} EXL3 linears and {n_plain} plain tensors checked against "
          f"{a.src}, {len(bad)} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
