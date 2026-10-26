#!/usr/bin/env python3
"""exl3_inventory.py -- what an exllamav3 (EXL3) checkpoint holds, whether plugin/exl3import can write it into a
.rad container, and the hole file rad-convert reads it through.

    exl3_inventory.py SRC [--json REPORT] [--holes HOLES.safetensors] [--strict]

SRC is the checkpoint directory (config.json with quantization_config.quant_method "exl3", and safetensors shards
named by model.safetensors.index.json, or one model.safetensors). Only headers are read, plus the 4-byte codebook
markers; nothing is loaded.

PRINTS
  the architecture id rad-convert derives (model_type without underscores) and the quantisation descriptor;
  every EXL3 linear by K and codebook, and their bytes; the other tensors by dtype;
  every finding that stops a convert (FAIL) or deserves a look (WARN).

FAIL (exit 1; nothing is written):
  a half-integer K (trellis last dimension 16K+8, exllamav3 1.5.2+): no decoder here reads it
  a K outside 2..8, a trellis not I16 of rank 3, a linear without .suh/.svh, a codebook marker of a wrong value
  a <X>.weight beside an <X>.trellis (two weights of one name)
  a layer whose experts differ in K, codebook or shape at one slot: the container addresses a layer's experts
  by a fixed stride and refuses that (core/format/radfile.h)
  a tensor dtype the safetensors reader does not take (core/format/safetensors.cpp)
  an EXL3 linear whose rows or columns are not a multiple of 128 (the importer refuses it)
WARN (exit 0, or 1 with --strict): tensors of a linear the importer does not read (.su / .sv / unknown parts),
  a bias on an EXL3 linear (written as its own plain weight).

--holes writes the file rad-convert takes with --add: one <X>.weight BF16 [n, k] for every EXL3 linear, a
safetensors header and a hole as long as the data (a sparse file: apparent size 2 bytes a weight, ~0 on disk). The
exl3 quantiser ignores the rows it is handed and reads the planes from SRC, so nothing reads real values there.
"""
import argparse
import json
import os
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

PARTS = ("trellis", "suh", "svh", "mul1", "mcg")
UNREAD_PARTS = ("su", "sv")
MARKERS = {"mul1": 0x83DCD12D, "mcg": 0xCBAC1FED}
# What core/format/safetensors.cpp reads (st_dtype); F64, F8_E8M0, U16 and U64 it refuses.
DTYPES = ("F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I8", "U8", "I16", "I32", "I64", "U32", "BOOL")
HAD = 128   # both rotations are 128-wide Hadamards: an EXL3 linear's rows and columns come in 128s


def fail(msg):
    raise SystemExit(f"exl3_inventory: {msg}")


def read_header(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            fail(f"{path}: not a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if n > 1 << 30:
            fail(f"{path}: a header of {n} bytes")
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    for v in h.values():
        v["file"] = path
        v["begin"] = 8 + n + v["data_offsets"][0]
    return h


def load(src):
    src = Path(src)
    cfg = json.loads((src / "config.json").read_text())
    tensors = {}
    index = src / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif (src / "model.safetensors").exists():
        files = ["model.safetensors"]
    else:
        fail(f"{src} has neither model.safetensors.index.json nor model.safetensors")
    for fn in files:
        if "/" in fn:
            fail(f"the index names {fn}, which is not a file in {src}")
        p = src / fn
        if not p.exists():
            fail(f"{p} is missing (the index names it)")
        for k, v in read_header(p).items():
            if k in tensors:
                fail(f"{k} is in two shards")
            tensors[k] = v
    return cfg, tensors


def marker(t):
    with open(t["file"], "rb") as f:
        f.seek(t["begin"])
        return struct.unpack("<I", f.read(4))[0]


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def layer_expert(name):
    """(prefix, layer number, expert number, slot key) as plugin/exl3import/arch/exl3_import.cpp groups them."""
    parts = name.split(".")
    prefix, layer = None, None
    for i in range(len(parts) - 1):
        if parts[i] == "layers" and parts[i + 1].isdigit() and i + 2 < len(parts):
            prefix, layer = ".".join(parts[:i]) + ("." if i else ""), int(parts[i + 1])
    expert, key = None, None
    for i in range(len(parts) - 1):
        if parts[i] == "experts" and parts[i + 1].isdigit() and i + 2 < len(parts) and i > 0:
            expert = int(parts[i + 1])
            key = ".".join(parts[: i + 1] + ["*"] + parts[i + 2:])
            break
    return prefix, layer, expert, key


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("--json", help="write the report as JSON here")
    ap.add_argument("--holes", help="write the hole file rad-convert reads with --add")
    ap.add_argument("--strict", action="store_true", help="exit 1 on a WARN too")
    a = ap.parse_args()

    cfg, tensors = load(a.src)
    qc = cfg.get("quantization_config") or {}
    fails, warns = [], []
    if qc.get("quant_method") != "exl3":
        fails.append(f"config.json quantization_config.quant_method is {qc.get('quant_method')!r}, not 'exl3'")
    arch = (cfg.get("model_type") or "").replace("_", "")
    quant = qc.get("quant_method", "") + ("_" + qc["fmt"] if "fmt" in qc else "")
    tc = cfg.get("text_config") or {}
    n_layers = tc.get("num_hidden_layers", cfg.get("num_hidden_layers"))

    linears = sorted(k[: -len(".trellis")] for k in tensors if k.endswith(".trellis"))
    lin_set = set(linears)
    lin = {}
    for x in linears:
        t = tensors[x + ".trellis"]
        sh = t["shape"]
        rec = {"name": x + ".weight"}
        if t["dtype"] != "I16" or len(sh) != 3:
            fails.append(f"{x}.trellis is {t['dtype']} {sh}; exllamav3 writes I16 [k/16, n/16, 16*K]")
            continue
        if sh[2] % 16 == 8:
            fails.append(f"{x}: half-integer K = {sh[2] // 16}.5 (trellis last dimension {sh[2]}); "
                         f"nothing here decodes it")
            continue
        if sh[2] % 16 or not 2 <= sh[2] // 16 <= 8:
            fails.append(f"{x}: trellis last dimension {sh[2]} is not 16*K for a K of 2..8")
            continue
        k, n, bits = sh[0] * 16, sh[1] * 16, sh[2] // 16
        rec.update(rows=n, cols=k, bits=bits)
        if n % HAD or k % HAD:
            fails.append(f"{x}: [{n}, {k}] (out, in); the importer needs both in multiples of {HAD} (its rotations)")
        cbs = []
        for m, want in MARKERS.items():
            mt = tensors.get(f"{x}.{m}")
            if mt is None:
                continue
            if mt["dtype"] != "I32" or mt["shape"] != []:
                fails.append(f"{x}.{m} is {mt['dtype']} {mt['shape']}; a codebook marker is an I32 scalar")
            elif marker(mt) != want:
                fails.append(f"{x}.{m} holds 0x{marker(mt):08X}, not the {m} marker 0x{want:08X}")
            cbs.append(m)
        if len(cbs) > 1:
            fails.append(f"{x} carries both the mul1 and the mcg marker")
        rec["codebook"] = cbs[0] if cbs else "3inst"
        for p, want in (("suh", [k]), ("svh", [n])):
            pt = tensors.get(f"{x}.{p}")
            if pt is None or pt["dtype"] != "F16" or pt["shape"] != want:
                fails.append(f"{x}.{p} is {'absent' if pt is None else pt['dtype'] + ' ' + str(pt['shape'])}; "
                             f"the import needs F16 {want}")
        for p in UNREAD_PARTS:
            if f"{x}.{p}" in tensors:
                warns.append(f"{x}.{p} is not read by the importer (it reads suh/svh): an older exllamav3 layout?")
        if f"{x}.bias" in tensors:
            warns.append(f"{x}.bias is written as a plain weight of its own beside the EXL3 linear")
        if f"{x}.weight" in tensors:
            fails.append(f"{x}.weight and {x}.trellis are both in the checkpoint")
        rec["bytes"] = n * (k * bits + 7) // 8 + 2 * (n + k)
        lin[x] = rec

    plain = {}
    for name, t in tensors.items():
        base, _, part = name.rpartition(".")
        if base in lin_set and (part in PARTS or part in UNREAD_PARTS):
            continue
        if base in lin_set and part not in ("bias",):
            warns.append(f"{name}: a tensor of the EXL3 linear {base} the importer does not know; written as a "
                         f"plain weight")
        if t["dtype"] not in DTYPES:
            fails.append(f"{name} is {t['dtype']}, which radiance's safetensors reader does not take")
        plain[name] = {"dtype": t["dtype"], "shape": t["shape"],
                       "bytes": t["data_offsets"][1] - t["data_offsets"][0]}

    # A layer's experts: one K, codebook and shape at each slot, and every expert with every slot.
    slots = defaultdict(dict)   # (prefix, layer) -> key -> {expert: form}
    for x, r in lin.items():
        prefix, layer, expert, key = layer_expert(x + ".weight")
        if expert is not None and layer is not None:
            slots[(prefix, layer)].setdefault(key, {})[expert] = ("exl3", r["bits"], r["codebook"], r["rows"],
                                                                  r["cols"])
    for name, r in plain.items():
        prefix, layer, expert, key = layer_expert(name)
        if expert is not None and layer is not None:
            slots[(prefix, layer)].setdefault(key, {})[expert] = ("plain", r["dtype"], tuple(r["shape"]))
    experts_by_layer = {}
    for (prefix, layer), keys in sorted(slots.items()):
        ids = set()
        for key, forms in keys.items():
            ids |= set(forms)
            kinds = Counter(forms.values())
            if len(kinds) > 1:
                fails.append(f"{prefix}layers.{layer}: {key} differs between experts ({dict(kinds)}); the "
                             f"container needs one form a slot in a layer")
        for key, forms in keys.items():
            if set(forms) != ids:
                fails.append(f"{prefix}layers.{layer}: {key} is missing for experts "
                             f"{sorted(ids - set(forms))[:8]}")
        if len(keys) > 8:
            fails.append(f"{prefix}layers.{layer}: {len(keys)} weights an expert; a movement group holds 8")
        experts_by_layer[f"{prefix}layers.{layer}"] = {"experts": len(ids), "slots": sorted(keys)}

    by_k = Counter((r["bits"], r["codebook"]) for r in lin.values())
    lin_bytes = sum(r["bytes"] for r in lin.values())
    plain_bytes = sum(r["bytes"] for r in plain.values())
    n_weights = sum(r["rows"] * r["cols"] for r in lin.values())
    print(f"source      {a.src}")
    print(f"arch        {arch or '(no model_type)'}   quant {quant or '(none)'}   layers {n_layers}")
    print(f"exl3        {len(lin)} linears, {n_weights / 1e9:.2f} G weights, {lin_bytes / 2**30:.2f} GiB "
          f"({8 * lin_bytes / max(n_weights, 1):.2f} bits a weight with scales)")
    for (bits, cb), c in sorted(by_k.items()):
        print(f"            K={bits} {cb:5s}  {c} linears")
    for dt, c in sorted(Counter(r["dtype"] for r in plain.values()).items()):
        print(f"plain       {dt:8s}  {c} tensors")
    print(f"plain       {plain_bytes / 2**30:.2f} GiB")
    n_exp = sum(1 for v in experts_by_layer.values() if v["experts"])
    if n_exp:
        e = next(iter(experts_by_layer.values()))
        print(f"experts     {n_exp} layers with experts, e.g. {e['experts']} experts x {len(e['slots'])} weights")
    print(f"container   about {(lin_bytes + plain_bytes) / 2**30:.2f} GiB")
    for tag, items in (("WARN", warns), ("FAIL", fails)):
        for x in items[:20]:
            print(f"{tag}  {x}")
        if len(items) > 20:
            print(f"{tag}  ... and {len(items) - 20} more" + (" (all in --json)" if a.json else " (--json lists all)"))

    report = {"source": str(a.src), "arch": arch, "quant": quant, "num_hidden_layers": n_layers,
              "exl3": {k: v for k, v in lin.items()}, "plain": plain, "experts": experts_by_layer,
              "bytes": {"exl3": lin_bytes, "plain": plain_bytes}, "warn": warns, "fail": fails}
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, sort_keys=True))
    if fails or (a.strict and warns):
        print(f"exl3_inventory: {len(fails)} FAIL, {len(warns)} WARN" + ("; no hole file written" if a.holes else ""))
        return 1

    if a.holes:
        write_holes(a.holes, [(r["name"], r["rows"], r["cols"]) for r in lin.values()])
    return 0


def write_holes(path, entries):
    header, off = {}, 0
    for name, rows, cols in sorted(entries):
        size = rows * cols * 2
        header[name] = {"dtype": "BF16", "shape": [rows, cols], "data_offsets": [off, off + size]}
        off += size
    header["__metadata__"] = {"format": "pt", "exl3_import": "holes: rows are never read (plugin/exl3import)"}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.truncate(8 + len(raw) + off)
    os.replace(tmp, path)
    st = os.stat(path)
    print(f"holes       {path}: {len(entries)} tensors, {st.st_size / 2**30:.2f} GiB apparent, "
          f"{st.st_blocks * 512 / 2**20:.2f} MiB on disk")


if __name__ == "__main__":
    sys.exit(main())
