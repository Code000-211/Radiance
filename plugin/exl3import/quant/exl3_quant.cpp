/* exl3_quant.cpp -- the `exl3` quantiser: IMPORTS an exllamav3 checkpoint's EXL3 linears into the canonical
 * EXL3 encoding (include/exl3_encoding.h). PLAN.md D2: nothing is quantised here -- exllamav3's quantiser is
 * CUDA/torch -- so the f32 weight rad-convert hands over is ignored and the planes come from `src=DIR`.
 *
 * From Dyluhn/radiance-exl3 v1.3.0 (c88ba2b, quantizers/exl3/exl3_quant.cpp; Apache-2.0, NOTICE.radiance-exl3),
 * which imports the routed experts of radiance's qwen4exp names only. This copy adds the GENERIC form, for a
 * container whose weights keep the checkpoint's own names (arch/exl3_import.cpp):
 *
 *   blk.*.ffn_*_exps.*.weight   exl3   src=$EXL3_SRC bits=4 codebook=mul1        (qwen4exp, as upstream)
 *   *.weight                    exl3   src=$EXL3_SRC bits=0 codebook=auto        (generic)
 *
 * GENERIC: a weight `<X>.weight` is the EXL3 linear `<X>` when the source holds `<X>.trellis`, and is declined
 * (RAD_E_UNSUPPORTED: the recipe's next rule decides it) when it does not. bits=0 takes each weight's K from
 * its trellis and codebook=auto its codebook from its marker; a number or a name instead checks every weight
 * against it. A generic weight is never stacked, and its `<X>.bias`, if any, is a weight of its own.
 *
 * WHICH TENSORS. The logical weight blk.L.ffn_gate_up_exps.E.weight (stacked: gate rows, then up rows) is
 * <P>.E.gate_proj and <P>.E.up_proj; blk.L.ffn_down_exps.E.weight is <P>.E.down_proj; each projection is
 * {trellis, suh, svh} and a codebook marker. <P> is the checkpoint prefix radiance's qwen4exp architecture
 * maps the block's experts from (arch/qwen4exp_fp8: `model.language_model.layers.L.mlp.experts` for a trunk
 * layer, `mtp.layers.0.mlp.experts` for the MTP head, which it declares at layer index num_hidden_layers).
 * The real names were confirmed against turboderp/Qwen3.8-Flash-Next-exl3 @ 55a732e's index.
 *
 * WHAT IS REFUSED, BY NAME (REQUIREMENTS R12) -- each has a test (tests/exl3_quant_test.cpp):
 *   a source tensor of another K than bits=            RAD_E_DTYPE
 *   a codebook marker other than codebook= (mcg, none) RAD_E_DTYPE
 *   a marker scalar with the wrong value, or two       RAD_E_FORMAT
 *   half-integer K (trellis last dim 16K+8, 1.5.2)     RAD_E_UNSUPPORTED
 *   a .bias                                            RAD_E_UNSUPPORTED
 *   a missing tensor / a layer the source lacks        RAD_E_NOTFOUND
 *   a source that is not EXL3, a shape that disagrees  RAD_E_FORMAT / RAD_E_SHAPE
 * A name that is not a routed expert is DECLINED (RAD_E_UNSUPPORTED from `encoding`): the recipe's next
 * rule decides it, as the rad_quant.h contract says.
 *
 * ROW BLOCK 0 -- the whole weight in one call. Not for the import, which could go 16 rows at a time, but for
 * the converter's error report, which decodes each call's rows as a weight of their own: a block of a
 * stacked gate_up would be read with the wrong half's input scales (suh.1 covers "the second half of the
 * rows"), and rows not in 128s cannot be rotated at all. An expert is 13 MB of f32; one call is nothing.
 */
#include "rad_quant.h"

#include "exl3_encoding.h"
#include "exl3_layout.h"
#include "exl3_src.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char kName[] = "exl3";

const RadQuantOption kOptions[] = {
    {"src", RAD_P_STR, nullptr,
     "directory of the EXL3 checkpoint the expert planes are imported from (exllamav3's layout, e.g. "
     "turboderp's conversion): config.json with quantization_config.quant_method exl3, and safetensors"},
    {"bits", RAD_P_INT, nullptr,
     "K, bits a weight, 2..8; every imported trellis must be K (its last dimension 16*K). 0: each "
     "weight's own K, read from its trellis (generic names only)"},
    {"codebook", RAD_P_STR, "mul1",
     "mul1 | mcg | 3inst | auto; every imported projection's marker must name it (.mul1 / .mcg / none). "
     "auto: each weight's own codebook, read from its marker (generic names only)"},
};

void say(const char* weight, const std::string& why) {
    std::fprintf(stderr, "radiance: exl3: '%s': %s\n", weight ? weight : "?", why.c_str());
}

enum { CB_AUTO = -1 };

struct Opts {
    std::string src;
    int bits = -1;                  /* 0: the weight's own (generic) */
    int codebook = EXL3_CB_MUL1;    /* CB_AUTO: the weight's own (generic) */
};

/* The options, typed by the converter from kOptions. */
bool read_opts(const RadParam* o, int n, Opts* out, std::string* why) {
    for (int i = 0; i < n; ++i) {
        const char* k = o[i].key;
        if (!std::strcmp(k, "src") && o[i].kind == RAD_P_STR && o[i].sval) out->src = o[i].sval;
        else if (!std::strcmp(k, "bits") && o[i].kind == RAD_P_INT) out->bits = (int)o[i].ival;
        else if (!std::strcmp(k, "codebook") && o[i].kind == RAD_P_STR && o[i].sval) {
            out->codebook = !std::strcmp(o[i].sval, "auto") ? (int)CB_AUTO : exl3_codebook_from_name(o[i].sval);
            if (out->codebook < 0 && out->codebook != CB_AUTO) {
                *why = std::string("codebook=") + o[i].sval + " is not one of mul1, mcg, 3inst, auto";
                return false;
            }
        } else {
            *why = std::string("option '") + k + "' is not one this quantiser takes (src, bits, codebook)";
            return false;
        }
    }
    if (out->src.empty()) { *why = "src=DIR (the EXL3 checkpoint) is required"; return false; }
    if (out->bits != 0 && (out->bits < EXL3_GEMM_BITS_MIN || out->bits > EXL3_GEMM_BITS_MAX)) {
        *why = out->bits < 0 ? "bits=K is required"
                             : "bits=" + std::to_string(out->bits) +
                                   ": EXL3 expert dtypes exist for K 2..8 (exl3k2..exl3k8), or 0 for each "
                                   "weight's own";
        return false;
    }
    return true;
}

/* blk.<L>.ffn_gate_up_exps.<E>.weight | blk.<L>.ffn_down_exps.<E>.weight, or (generic) <X>.weight */
struct Name {
    int layer = -1;
    long long expert = -1;
    bool stacked = false;
    bool generic = false;
    std::string base;   /* generic: <X> */
};

bool parse_name(const char* s, Name* out) {
    int at = -1;
    if (!s) return false;
    const size_t len = std::strlen(s);
    if (std::strncmp(s, "blk.", 4) != 0) {
        static const char kSuffix[] = ".weight";
        const size_t n = sizeof kSuffix - 1;
        if (len <= n || std::strcmp(s + len - n, kSuffix) != 0) return false;
        out->generic = true;
        out->base.assign(s, len - n);
        return true;
    }
    if (std::sscanf(s, "blk.%d.ffn_gate_up_exps.%lld.weight%n", &out->layer, &out->expert, &at) == 2 &&
        at > 0 && s[at] == 0) {
        out->stacked = true;
    } else if (at = -1, std::sscanf(s, "blk.%d.ffn_down_exps.%lld.weight%n", &out->layer, &out->expert,
                                    &at) == 2 && at > 0 && s[at] == 0) {
        out->stacked = false;
    } else {
        return false;
    }
    return out->layer >= 0 && out->expert >= 0;
}

/* A probe's answer for "not an EXL3 linear" (not a RAD_ status; those are negative): the weight is declined, and
 * the recipe's next rule decides it. */
constexpr int kDecline = 1;

/* A generic weight's K and codebook from its source tensors, checked against the options. RAD_OK; kDecline
 * when the source has no <base>.trellis; or a refusal (status, `why`). */
int probe(exl3::Source* s, const std::string& base, const Opts& op, int* bits, int* cb, std::string* why);

int encoding(const RadParam* o, int n_o, const RadQuantWeight* w, RadEncoding* out) {
    Name nm;
    Opts op;
    std::string why;
    if (!parse_name(w->name, &nm)) return RAD_E_UNSUPPORTED;
    if (!read_opts(o, n_o, &op, &why)) { say(w->name, why); return RAD_E_INVAL; }
    if (!nm.generic && (op.bits == 0 || op.codebook == CB_AUTO)) {
        say(w->name, "bits=0 and codebook=auto are for generic names; a qwen4exp expert takes its K and "
                     "codebook from the recipe");
        return RAD_E_INVAL;
    }
    int bits = op.bits, cb = op.codebook;
    if (nm.generic) {
        exl3::Source* s = exl3::Source::open(op.src, &why);
        if (!s) { say(w->name, why); return RAD_E_FORMAT; }
        const int st = probe(s, nm.base, op, &bits, &cb, &why);
        if (st == kDecline) return RAD_E_UNSUPPORTED;
        if (st != RAD_OK) { say(w->name, why); return st; }
    }
    if (w->rank != 2 || w->rows <= 0 || w->cols <= 0 || w->rows % EXL3_HAD || w->cols % EXL3_HAD ||
        (nm.stacked && w->rows % (2 * EXL3_HAD))) {
        say(w->name, "a [" + std::to_string(w->rows) + ", " + std::to_string(w->cols) + "] weight: EXL3 needs "
                     "rows and columns in 128s (both rotations), a stacked gate_up's halves too");
        return RAD_E_SHAPE;
    }
    return exl3_encoding_make(bits, cb, nm.stacked, out);
}

int64_t row_block(const RadParam*, int, const RadQuantWeight*) { return 0; }

/* ------------------------------------------------------------------ the import */

struct Part {
    const char* proj;    /* gate_proj | up_proj | down_proj; null for a generic weight (its base is whole) */
    int64_t row0, rows;  /* its rows of the logical weight */
    int suh_plane;
};

/* The codebook a projection's marker names: 1 found (in *cb), or a refusal (status, `why`). */
int marker(exl3::Source* s, const std::string& base, int* cb, std::string* why) {
    static const struct { const char* tag; uint32_t value; int cb; } kMarkers[] = {
        {"mul1", EXL3_MUL1_MARKER, EXL3_CB_MUL1}, {"mcg", EXL3_MCG_MARKER, EXL3_CB_MCG}};
    int found = 0;
    *cb = EXL3_CB_3INST;   /* no marker: exllamav3's 3inst */
    for (const auto& m : kMarkers) {
        exl3::Tensor t;
        uint32_t v = 0;
        const int f = s->find(base + "." + m.tag, &t, why);
        if (f < 0) return RAD_E_IO;
        if (f == 0) continue;
        if (t.dtype != "I32" || !t.shape.empty() || t.bytes() != 4) {
            *why = base + "." + m.tag + " is " + t.dtype + " of rank " + std::to_string(t.shape.size()) +
                   "; a codebook marker is an I32 scalar";
            return RAD_E_FORMAT;
        }
        if (!s->read(t, &v, why)) return RAD_E_IO;
        if (v != m.value) {
            char buf[160];
            std::snprintf(buf, sizeof buf, " holds 0x%08" PRIX32 ", not the %s marker 0x%08" PRIX32, v, m.tag,
                          m.value);
            *why = base + "." + m.tag + buf;
            return RAD_E_FORMAT;
        }
        *cb = m.cb;
        ++found;
    }
    if (found > 1) {
        *why = base + " carries both the mul1 and the mcg marker; a projection has one codebook";
        return RAD_E_FORMAT;
    }
    return RAD_OK;
}

/* One tensor, checked for dtype and shape, read into dst. */
int fetch(exl3::Source* s, const std::string& name, const char* dtype, std::vector<int64_t> shape, void* dst,
          std::string* why) {
    exl3::Tensor t;
    const int f = s->find(name, &t, why);
    if (f < 0) return RAD_E_IO;
    if (f == 0) { *why = name + " is not in the source"; return RAD_E_NOTFOUND; }
    if (t.dtype != dtype || t.shape != shape) {
        std::string want, got;
        for (int64_t d : shape) want += (want.empty() ? "" : ", ") + std::to_string(d);
        for (int64_t d : t.shape) got += (got.empty() ? "" : ", ") + std::to_string(d);
        *why = name + " is " + t.dtype + " [" + got + "]; the weight needs " + dtype + " [" + want + "]";
        return RAD_E_SHAPE;
    }
    return s->read(t, dst, why) ? RAD_OK : RAD_E_IO;
}

/* A projection's trellis tensor and its K and codebook, each checked: the trellis's form, half-integer K,
 * K against `want_bits` (0: any of 2..8) and the codebook against `want_cb` (CB_AUTO: any). RAD_OK; kDecline
 * when the source has no <base>.trellis and `absent_declines`; or a refusal (status, `why`). */
int read_trellis(exl3::Source* s, const std::string& base, int want_bits, int want_cb, bool absent_declines,
                 exl3::Tensor* t, int* bits, int* cb, std::string* why) {
    int st = 0;
    /* the trellis first: an absent projection has no marker either, and must not read as 3inst */
    if ((st = s->find(base + ".trellis", t, why)) <= 0) {
        if (st < 0) return RAD_E_IO;
        if (absent_declines) return kDecline;
        *why = base + ".trellis is not in the source";
        return RAD_E_NOTFOUND;
    }
    if ((st = marker(s, base, cb, why)) != RAD_OK) return st;
    if (want_cb != CB_AUTO && *cb != want_cb) {
        *why = base + " is codebook " + exl3_codebook_name(*cb) + " (by its marker); the recipe says codebook=" +
               exl3_codebook_name(want_cb);
        return RAD_E_DTYPE;
    }
    if (t->dtype != "I16" || t->shape.size() != 3) {
        *why = base + ".trellis is " + t->dtype + " of rank " + std::to_string(t->shape.size()) +
               "; exllamav3 writes I16 [k/16, n/16, 16*K]";
        return RAD_E_FORMAT;
    }
    if (t->shape[2] % 16 == 8) {
        *why = base + ".trellis has a last dimension of " + std::to_string(t->shape[2]) + ": half-integer K = " +
               std::to_string(t->shape[2] / 16) + ".5 (exllamav3 1.5.2), which nothing here decodes";
        return RAD_E_UNSUPPORTED;
    }
    if (t->shape[2] % 16 != 0) {
        *why = base + ".trellis has a last dimension of " + std::to_string(t->shape[2]) + ", not 16*K";
        return RAD_E_FORMAT;
    }
    *bits = (int)(t->shape[2] / 16);
    if (want_bits == 0 && (*bits < EXL3_GEMM_BITS_MIN || *bits > EXL3_GEMM_BITS_MAX)) {
        *why = base + " is K=" + std::to_string(*bits) + " (trellis last dimension " + std::to_string(t->shape[2]) +
               "); EXL3 dtypes exist for K 2..8";
        return RAD_E_UNSUPPORTED;
    }
    if (want_bits != 0 && *bits != want_bits) {
        *why = base + " is K=" + std::to_string(*bits) + " (trellis last dimension " +
               std::to_string(t->shape[2]) + "); the recipe says bits=" + std::to_string(want_bits);
        return RAD_E_DTYPE;
    }
    return RAD_OK;
}

int probe(exl3::Source* s, const std::string& base, const Opts& op, int* bits, int* cb, std::string* why) {
    exl3::Tensor t;
    return read_trellis(s, base, op.bits, op.codebook, true, &t, bits, cb, why);
}

/* One projection: `base` names its tensors in the source. `bias_refused` for the qwen4exp experts, whose ops
 * have no bias term; a generic weight's bias is a weight of its own. */
int import_part(exl3::Source* s, const std::string& base, const Part& p, int want_bits, int want_cb,
                bool bias_refused, const RadQuantWeight* w, void* const* planes, std::string* why) {
    const int64_t k = w->cols, n = p.rows;
    exl3::Tensor t;
    int bits = 0, cb = 0, st = 0;

    if ((st = read_trellis(s, base, want_bits, want_cb, false, &t, &bits, &cb, why)) != RAD_OK) return st;
    exl3::Tensor b;
    if (bias_refused && (st = s->find(base + ".bias", &b, why)) != 0) {
        if (st < 0) return RAD_E_IO;
        *why = base + " has a .bias; the EXL3 expert ops have no bias term";
        return RAD_E_UNSUPPORTED;
    }
    if (t.shape[0] * 16 != k || t.shape[1] * 16 != n) {
        *why = base + ".trellis is [" + std::to_string(t.shape[0]) + ", " + std::to_string(t.shape[1]) +
               ", " + std::to_string(t.shape[2]) + "], tiles of a [" + std::to_string(t.shape[0] * 16) + ", " +
               std::to_string(t.shape[1] * 16) + "] weight; this part of '" + w->name + "' is k=" +
               std::to_string(k) + " in, n=" + std::to_string(n) + " out";
        return RAD_E_SHAPE;
    }

    /* suh and svh straight into their planes; the trellis through the layout into the codes plane */
    uint16_t* svh = static_cast<uint16_t*>(planes[EXL3_PLANE_SVH]) + p.row0;
    if ((st = fetch(s, base + ".suh", "F16", {k}, planes[p.suh_plane], why)) != RAD_OK) return st;
    if ((st = fetch(s, base + ".svh", "F16", {n}, svh, why)) != RAD_OK) return st;
    std::vector<uint16_t> trellis((size_t)(t.bytes() / 2));
    if (!s->read(t, trellis.data(), why)) return RAD_E_IO;
    const int64_t rb = exl3_codes_row_bytes(k, bits);
    uint8_t* codes = static_cast<uint8_t*>(planes[EXL3_PLANE_CODES]) + p.row0 * rb;
    if (exl3_record_to_codes(trellis.data(), n, k, bits, codes, rb) != EXL3_OK) {
        *why = base + ": the trellis does not fit the codes plane";
        return RAD_E_SHAPE;
    }
    return RAD_OK;
}

int quantize(const RadParam* o, int n_o, const RadQuantWeight* w, float* /* ignored: D2 */, int64_t row0,
             int64_t rows, void* const* planes) {
    Name nm;
    Opts op;
    std::string why;
    if (!parse_name(w->name, &nm)) { say(w->name, "is not a routed expert weight"); return RAD_E_UNSUPPORTED; }
    if (!read_opts(o, n_o, &op, &why)) { say(w->name, why); return RAD_E_INVAL; }
    if (row0 != 0 || rows != w->rows) {
        say(w->name, "rows [" + std::to_string(row0) + ", " + std::to_string(row0 + rows) + ") of " +
                     std::to_string(w->rows) + ": this quantiser takes a weight whole (row_block 0)");
        return RAD_E_STATE;
    }
    exl3::Source* s = exl3::Source::open(op.src, &why);
    if (!s) { say(w->name, why); return RAD_E_FORMAT; }

    if (nm.generic) {
        const Part whole{nullptr, 0, rows, EXL3_PLANE_SUH};
        const int st = import_part(s, nm.base, whole, op.bits, op.codebook, false, w, planes, &why);
        if (st != RAD_OK) { say(w->name, why); return st; }
        return RAD_OK;
    }

    char prefix[160];
    if (nm.layer < s->n_layers())
        std::snprintf(prefix, sizeof prefix, "model.language_model.layers.%d.mlp.experts.%lld", nm.layer,
                      nm.expert);
    else if (nm.layer == s->n_layers())
        std::snprintf(prefix, sizeof prefix, "mtp.layers.0.mlp.experts.%lld", nm.expert);
    else {
        say(w->name, "layer " + std::to_string(nm.layer) + " is past the source's " +
                     std::to_string(s->n_layers()) + " layers and its MTP head (layer " +
                     std::to_string(s->n_layers()) + ")");
        return RAD_E_NOTFOUND;
    }
    const Part gate_up[2] = {{"gate_proj", 0, rows / 2, EXL3_PLANE_SUH},
                             {"up_proj", rows / 2, rows / 2, EXL3_PLANE_SUH1}};
    const Part down[1] = {{"down_proj", 0, rows, EXL3_PLANE_SUH}};
    const Part* parts = nm.stacked ? gate_up : down;
    for (int i = 0; i < (nm.stacked ? 2 : 1); ++i) {
        const int st = import_part(s, std::string(prefix) + "." + parts[i].proj, parts[i], op.bits, op.codebook,
                                   true, w, planes, &why);
        if (st != RAD_OK) { say(w->name, why); return st; }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the decoder (oracle, kbench, error report) */

int decode(const RadEncoding* e, const RadQuantWeight* w, const void* const* planes, int64_t row0, int64_t rows,
           float* out) {
    int bits = 0, cb = 0, stacked = 0;
    const int st = exl3_encoding_read(e, &bits, &cb, &stacked);
    if (st != RAD_OK) return st;   /* RAD_E_UNSUPPORTED: not ours; RAD_E_FORMAT: says exl3, is not */
    Exl3View v;
    std::memset(&v, 0, sizeof v);
    v.rows = w->rows;
    v.cols = w->cols;
    v.bits = bits;
    v.codebook = cb;
    v.codes = static_cast<const uint8_t*>(planes[EXL3_PLANE_CODES]);
    v.suh = static_cast<const uint16_t*>(planes[EXL3_PLANE_SUH]);
    v.svh = static_cast<const uint16_t*>(planes[EXL3_PLANE_SVH]);
    v.suh1 = stacked ? static_cast<const uint16_t*>(planes[EXL3_PLANE_SUH1]) : nullptr;
    switch (exl3_decode_rows(&v, row0, rows, out)) {
    case EXL3_OK: return RAD_OK;
    case EXL3_E_SHAPE: return RAD_E_SHAPE;
    case EXL3_E_NOMEM: return RAD_E_NOMEM;
    default: return RAD_E_INVAL;
    }
}

const RadQuantizerInfo kQuant[] = {{
    kName,
    "imports EXL3 (exllamav3 trellis) linears from src=DIR into the canonical exl3 encoding: qwen4exp's routed "
    "experts, or any <X>.weight whose <X>.trellis the source holds; quantises nothing (the source's planes, "
    "permuted losslessly), decodes for the oracle",
    kOptions, (int)(sizeof kOptions / sizeof kOptions[0]),
    encoding, row_block, quantize, decode,
}};

const RadPluginInfo kInfo = {
    RAD_PLUGIN_QUANT, "exl3_quant", "0.1.0",
    "the exl3 quantiser: EXL3 linears imported from an exllamav3 checkpoint", "host",
};

}  // namespace

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
extern "C" int rad_quant_count(void) { return 1; }
extern "C" const RadQuantizerInfo* rad_quant_at(int i) { return i == 0 ? &kQuant[0] : nullptr; }
