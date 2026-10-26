#!/usr/bin/env python3
"""make_fixture.py -- a tiny exllamav3 (EXL3) checkpoint shaped like a DeepSeek MoE, for the convert test.

    make_fixture.py OUT [--half-k] [--mixed-experts]

Not a model anyone runs: random trellis symbols (every symbol sequence is a valid EXL3 encoding) and random
scales. What it covers is what an import must get right: EXL3 linears at K 2, 3, 4 and 6 in all three codebooks
(mul1, mcg by marker, 3inst by no marker), a bias on an EXL3 linear, plain F16/BF16/F32 tensors of rank 1-3, routed
experts with three weights each, shared experts, a router, an MTP layer under its own prefix, and two shards
behind an index.

--half-k     one linear at half-integer K (trellis last dimension 16K+8): the inventory must refuse it
--mixed-experts  one expert of layer 0 at another K: the inventory must refuse the layer
--odd-dims   one linear of 192 output rows (not a multiple of 128): the inventory must refuse it
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

H, E, FF, SFF, V, L = 256, 4, 128, 128, 512, 2
MARK = {"mul1": 0x83DCD12D, "mcg": 0xCBAC1FED}


def linear(t, rng, name, n, k, bits, cb, half=False):
    last = 16 * bits + (8 if half else 0)
    t[name + ".trellis"] = rng.integers(-32768, 32768, size=(k // 16, n // 16, last), dtype=np.int64).astype(np.int16)
    t[name + ".suh"] = (rng.choice([-1, 1], size=k) * rng.uniform(0.5, 2, size=k)).astype(np.float16)
    t[name + ".svh"] = (rng.choice([-1, 1], size=n) * rng.uniform(1e-3, 4e-3, size=n)).astype(np.float16)
    if cb in MARK:
        t[name + "." + cb] = np.array(struct.unpack("<i", struct.pack("<I", MARK[cb]))[0], dtype=np.int32)


def byte_unicode():
    """GPT-2's byte -> printable character table, as ByteLevel spells bytes."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def build(half_k, mixed, odd=False):
    rng = np.random.default_rng(1017)
    t = {}
    t["model.embed_tokens.weight"] = rng.standard_normal((V, H)).astype(np.float16)
    t["model.norm.weight"] = np.ones(H, np.float16)
    linear(t, rng, "lm_head", V, H, 6, "3inst")
    for l in range(L):
        p = f"model.layers.{l}."
        t[p + "input_layernorm.weight"] = np.ones(H, np.float16)
        t[p + "post_attention_layernorm.weight"] = np.ones(H, np.float16)
        linear(t, rng, p + "self_attn.q_proj", H, H, 3, "mcg", half=half_k and l == 1)
        linear(t, rng, p + "self_attn.kv_proj", 256, H, 4, "mul1")
        linear(t, rng, p + "self_attn.o_proj", H, H, 4, "mul1")
        t[p + "self_attn.o_proj.bias"] = rng.standard_normal(H).astype(np.float16)
        t[p + "self_attn.kv_norm.weight"] = np.ones(256, np.float16)
        t[p + "mlp.gate.weight"] = rng.standard_normal((E, H)).astype(np.float32)
        t[p + "mlp.gate.e_score_correction_bias"] = np.zeros(E, np.float32)
        for e in range(E):
            q = f"{p}mlp.experts.{e}."
            kb = 3 if (mixed and l == 0 and e == 2) else 2
            linear(t, rng, q + "gate_proj", FF, H, kb, "mul1")
            linear(t, rng, q + "up_proj", FF, H, 2, "mul1")
            linear(t, rng, q + "down_proj", H, FF, 2, "mul1")
        for proj, n, k in (("gate_proj", SFF, H), ("up_proj", SFF, H), ("down_proj", H, SFF)):
            linear(t, rng, f"{p}mlp.shared_experts.{proj}", n, k, 3, "mul1")
    m = "mtp.layers.0."
    t[m + "enorm.weight"] = np.ones(H, np.float16)
    t[m + "hnorm.weight"] = np.ones(H, np.float16)
    t[m + "eh_proj.weight"] = rng.standard_normal((H, 2 * H)).astype(np.float16).view(np.uint16)  # stored bf16-ish
    for e in range(E):
        for proj, n, k in (("gate_proj", FF, H), ("up_proj", FF, H), ("down_proj", H, FF)):
            linear(t, rng, f"{m}mlp.experts.{e}.{proj}", n, k, 2, "mul1")
    if odd:
        linear(t, rng, "model.layers.0.self_attn.wkv_a", 192, H, 4, "mul1")
    t["model.rope.inv_freq_table"] = rng.standard_normal((2, 4, 8)).astype(np.float32)
    return t


def st_dtype(name, a):
    if name.endswith("eh_proj.weight"):
        return "BF16"
    return {np.dtype(np.float16): "F16", np.dtype(np.float32): "F32", np.dtype(np.int16): "I16",
            np.dtype(np.int32): "I32"}[a.dtype]


def write_shard(path, tensors):
    header, off, blobs = {}, 0, []
    for name in sorted(tensors):
        a = np.asarray(tensors[name]).copy(order="C")   # ascontiguousarray would make a scalar [1]
        b = a.tobytes()
        header[name] = {"dtype": st_dtype(name, a), "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--half-k", action="store_true")
    ap.add_argument("--mixed-experts", action="store_true")
    ap.add_argument("--odd-dims", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = build(a.half_k, a.mixed_experts, a.odd_dims)
    names = sorted(t)
    shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    wm = {}
    for fn, ns in shards.items():
        write_shard(out / fn, {n: t[n] for n in ns})
        wm.update({n: fn for n in ns})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}, indent=1))
    cfg = {"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4", "hidden_size": H,
           "num_hidden_layers": L, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 64,
           "moe_intermediate_size": FF, "n_routed_experts": E, "num_experts_per_tok": 2, "n_shared_experts": 1,
           "vocab_size": V, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "num_nextn_predict_layers": 1,
           "quantization_config": {"quant_method": "exl3", "version": "0.0.x", "bits": 2.5, "head_bits": 6,
                                   "calibration": {"rows": 0, "cols": 0}}}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    # A byte-level BPE over the 256 bytes and no merges: enough for rad-convert to bake a vocab.
    b2u = byte_unicode()
    vocab = {b2u[i]: i for i in range(256)}
    vocab.update({f"<extra_{i}>": i for i in range(256, V - 2)})
    tok = {"version": "1.0", "truncation": None, "padding": None,
           "added_tokens": [{"id": V - 2, "content": "<|begin_of_sentence|>", "single_word": False, "lstrip": False,
                             "rstrip": False, "normalized": False, "special": True},
                            {"id": V - 1, "content": "<|end_of_sentence|>", "single_word": False, "lstrip": False,
                             "rstrip": False, "normalized": False, "special": True}],
           "normalizer": None,
           "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
           "post_processor": None, "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True,
                                                 "use_regex": True},
           "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                     "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                     "vocab": vocab, "merges": []}}
    (out / "tokenizer.json").write_text(json.dumps(tok))
    (out / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<|begin_of_sentence|>",
                                                           "eos_token": "<|end_of_sentence|>"}))
    print(f"make_fixture: {len(t)} tensors in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
