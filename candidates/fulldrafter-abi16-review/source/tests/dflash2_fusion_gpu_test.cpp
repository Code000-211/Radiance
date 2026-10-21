/* tests/dflash2_fusion_gpu_test.cpp -- the executable oracle for the DFlash2 local fusions.
 *
 * WHY THIS FILE EXISTS. tools/rad_kbench.cpp:2682-2686 resolves every step of a `replaces` chain
 * INSIDE THE FUSED ROW'S OWN PLUGIN. `mine` publishes `dflash_conv_add` and
 * `dflash_k_norm_rope_store` and no unfused row, and it must not publish one: a second copy of
 * libr4d's arithmetic would be the very thing the fusion is checked against. So stock rad-kbench's
 * `--fuse` output for these rows is a SKIP, never a pass, and no claim of parity may rest on it.
 *
 * THIS COMPARATOR NAMES BOTH SIDES EXPLICITLY.
 *
 *   candidate   exactly ONE row, resolved through the ordinary registry for a geometry, and then
 *               required to have come back as plugin `mine` and the pinned kernel name. No
 *               fallback: a different winner, a missing row or a zero-case run is a failure.
 *   reference   each unfused step, pinned by kernel NAME inside plugin `libr4d` and looked up in
 *               that plugin's own row table -- never through resolve(), which could answer with
 *               another plugin's substitute.
 *
 * The two sides run on SEPARATE device buffers holding identical initial bytes. Every buffer is
 * framed in sentinels; after the run the comparator checks (a) candidate bytes against reference
 * bytes and (b) every byte the operation must not write against the pristine image -- the framing,
 * the gaps of a pitched row, the unused half of the delta projection, skipped KV slots. There is
 * no tolerance anywhere in this file: the declared contract is bit equality.
 *
 * THE ROW IDENTITY IS CHECKED, NOT ASSUMED. The candidate's winning row must be plugin `mine` AND
 * the pinned kernel name; a pinned reference row must belong to `libr4d` and be a device row with a
 * launcher. The plugins are also identified by their own `RadPluginInfo`, not by the file name.
 *
 * THE BF16 GATE IS WIRED FROM ITS SETTLED CONTRACT (mine/dflash_gated_bf16.h). Its candidate is
 * `mine_dflash_gated_bf16`; its reference is libr4d's `gemm_bf16_nt_m64` (`gemm_nt`) followed by
 * `silu_mul_bf16`, with the REAL [M, 2n] bf16 intermediate kept as its own device buffer. For
 * M <= 16 libr4d's selector would serve `gemm_bf16_nt_m16`, which forwards to `m64`; this
 * comparator pins `m64` directly and asserts that identity, so one reference row covers every M.
 *
 * NO SKIP IS A PASS. A missing device, a missing row, a refused launch, a zero selected case or an
 * unknown case name all return nonzero. `--help` is answered before a device is touched.
 *
 * `--benchmark` times model-sized cases and is NOT a wider parity claim. Each benchmark case runs
 * its own byte comparison first and aborts on mismatch; only then are the fused row and the
 * complete unfused chain timed, on the same scene, in alternating order over paired rounds. The
 * numbers are host submission + queue execution + final sync per rank on synthetic inputs with hot
 * buffers; HIP events are not used, and the result line says "finite-cases-only".
 *
 * THE GATE GETS A SECOND SCENE BECAUSE HOT BUFFERS ARE A BEST CASE. Re-reading one gate matrix on
 * every launch lets it stay in cache; the block owns five different gate matrices, so production
 * re-reads 39 MB of cold weights each step. `gate/model-C1-M4-stream5` therefore rotates five
 * independent weight buffers per arm, with the same shape, the same inputs, the same byte
 * comparison on all five outputs, and the same calibration, alternating rounds and sync. Its
 * `BENCH_NOTE` line says `hot-buffers=0 weight-rotation=5`; the hot scene stays as it was and says
 * `hot-buffers=1`.
 */
#include "core/device/device.h"
#include "core/plugin/registry.h"
#include "abi/rad_plugin.h"      /* rad_f32_to_bf16: the ABI's RNE narrowing, not a local one */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

/* ------------------------------------------------------------------------------------ basics */

/* Both take a view: the error string is built ONLY when something failed. A literal label on a
 * launch path therefore costs nothing, which matters because the benchmark calls these once per
 * launch and an eager allocation would be charged to whichever side launches fewer kernels. */
[[noreturn]] void fail(std::string_view what, std::string_view detail = {}) {
    throw std::runtime_error(detail.empty() ? std::string(what)
                                            : std::string(what) + ": " + std::string(detail));
}

void require(bool value, std::string_view message) {
    if (!value) fail(message);
}

void check(int status, std::string_view what) {
    if (status == RAD_OK) return;
    const char* detail = rad_dev_last_error();
    fail(what, std::string(rad_strerror(status)) + "; " + (detail ? detail : ""));
}

uint16_t to_bf(float v) { return rad_f32_to_bf16(v); }

/* Deterministic noise in [-0.5, 0.5), narrowed to bf16 before use, so neither side ever sees a
 * denormal, an infinity or a NaN it could resolve differently. */
uint32_t g_rng = 0x13579bdfu;
float rnd() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return float((g_rng >> 9) & 0x3fffu) / 16384.0f - 0.5f;
}

/* --------------------------------------------------------------------------------- device */

struct DeviceBuffer {
    void* data = nullptr;
    explicit DeviceBuffer(size_t bytes) {
        if (bytes) data = rad_dev_alloc(int64_t(bytes), RAD_MEM_DEVICE);
        require(bytes == 0 || data != nullptr, "device allocation failed");
    }
    ~DeviceBuffer() { if (data) rad_dev_free(data, RAD_MEM_DEVICE); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : data(other.data) { other.data = nullptr; }
};

struct Session {
    RadStream stream = nullptr;
    Session() {
        check(rad_dev_set(0), "select GPU 0");
        RadDeviceProps props{};
        check(rad_dev_props(0, &props), "GPU properties");
        require(!props.is_host_backend && std::string(props.arch) == "gfx1201",
                "dflash2 fusion parity requires a real gfx1201 device; a host backend is not a pass");
        check(rad_stream_create(&stream, 0), "create comparator stream");
    }
    ~Session() { if (stream) rad_stream_destroy(stream); }
};

/* ----------------------------------------------------------------------------------- scene */
/* A scene owns the HOST image of every buffer once. Both sides upload the same bytes and both
 * operand lists index into that image, so "identical initialisation" is a property of the data
 * rather than a promise made twice. */

constexpr size_t kPad = 64;   /* framing each side of a payload; deliberately not a payload size */

struct OpSpec {
    int      buf = 0;
    int64_t  offset = 0;      /* bytes into the payload */
    uint32_t dtype = 0;
    int      rank = 0;
    int64_t  shape[4] = {};
    int64_t  stride[4] = {};
};

struct Scene {
    std::vector<std::string>          name;
    std::vector<std::vector<uint8_t>> image;    /* framed: [kPad][payload][kPad] */
    std::vector<std::vector<uint8_t>> mask;     /* empty = unchecked; else 1 = may be written */
    std::vector<OpSpec>               cand;
    std::vector<OpSpec>               ref;
    std::vector<int>                  compare;  /* buffers compared byte for byte */
    /* Gate scenes only: one operand list per weight buffer. A streaming case rotates them, and
     * `cand`/`ref` above are always the first slice, so every other case is unchanged. */
    std::vector<std::vector<OpSpec>>  cand_ops;
    std::vector<std::vector<OpSpec>>  ref_ops;

    int add_buffer(std::string n, const std::vector<uint8_t>& payload, std::vector<uint8_t> written) {
        std::vector<uint8_t> framed(kPad + payload.size() + kPad, 0);
        for (size_t i = 0; i < framed.size(); ++i)
            framed[i] = uint8_t(0x5au ^ uint8_t(i * 37u + 11u));
        if (!payload.empty()) std::memcpy(framed.data() + kPad, payload.data(), payload.size());
        require(written.empty() || written.size() == framed.size(), "mask size must equal the frame");
        name.push_back(std::move(n));
        image.push_back(std::move(framed));
        mask.push_back(std::move(written));
        return int(name.size()) - 1;
    }
};

std::vector<uint8_t> no_writes(size_t bytes) { return std::vector<uint8_t>(bytes, 0); }

size_t framed_at(size_t payload_offset) { return kPad + payload_offset; }

void put_bf(std::vector<uint8_t>& payload, size_t element, uint16_t value) {
    require(element * 2 + 2 <= payload.size(), "bf16 write past the payload");
    std::memcpy(payload.data() + element * 2, &value, 2);
}

void put_i32(std::vector<uint8_t>& payload, size_t element, int32_t value) {
    require(element * 4 + 4 <= payload.size(), "i32 write past the payload");
    std::memcpy(payload.data() + element * 4, &value, 4);
}

void put_f32(std::vector<uint8_t>& payload, size_t element, float value) {
    require(element * 4 + 4 <= payload.size(), "f32 write past the payload");
    std::memcpy(payload.data() + element * 4, &value, 4);
}

/* ------------------------------------------------------------------------------------ sides */

struct Side {
    std::vector<std::unique_ptr<DeviceBuffer>> buf;

    void upload(const Session& s, const Scene& sc) {
        buf.clear();
        for (const auto& bytes : sc.image) {
            buf.emplace_back(new DeviceBuffer(bytes.size()));
            if (!bytes.empty())
                check(rad_memcpy_async(buf.back()->data, bytes.data(), int64_t(bytes.size()),
                                       s.stream), "upload scene buffer");
        }
        check(rad_stream_sync(s.stream), "finish scene upload");
    }

    std::vector<RadTensor> bind(const std::vector<OpSpec>& ops) const {
        std::vector<RadTensor> out;
        out.reserve(ops.size());
        for (const OpSpec& o : ops) {
            RadTensor t{};
            t.data = static_cast<uint8_t*>(buf[size_t(o.buf)]->data) + kPad + o.offset;
            t.dtype = o.dtype;
            t.rank = uint32_t(o.rank);
            for (int i = 0; i < o.rank; ++i) { t.shape[i] = o.shape[i]; t.stride[i] = o.stride[i]; }
            out.push_back(t);
        }
        return out;
    }

    void download(const Session& s, const Scene& sc, std::vector<std::vector<uint8_t>>& out) const {
        out.assign(sc.image.size(), {});
        for (size_t i = 0; i < buf.size(); ++i) {
            out[i].resize(sc.image[i].size());
            if (!out[i].empty())
                check(rad_memcpy_async(out[i].data(), buf[i]->data, int64_t(out[i].size()),
                                       s.stream), "download scene buffer");
        }
        check(rad_stream_sync(s.stream), "finish scene download");
    }
};

/* ----------------------------------------------------------------------------- registry use */

/* A reference row is pinned by (plugin, kernel) directly in the loaded tables. resolve() is not
 * used here on purpose: it is a selector, and a selector's answer is exactly what must not be
 * allowed to substitute for the named step. */
const rad::KernelRow* pin_row(rad::Registry& reg, const char* op, const char* kernel,
                              const char* plugin) {
    const auto* rows = reg.rows_for(op);
    require(rows != nullptr && !rows->empty(), std::string("no row at all for op '") + op + "'");
    const rad::KernelRow* hit = nullptr;
    for (const rad::KernelRow* r : *rows) {
        if (!r->info || !r->info->name || std::strcmp(r->info->name, kernel) != 0) continue;
        require(r->plugin == plugin, std::string("kernel '") + kernel + "' is published by '" +
                                    r->plugin + "', expected '" + plugin + "'");
        require(hit == nullptr, std::string("duplicate row named '") + kernel + "'");
        hit = r;
    }
    require(hit != nullptr, std::string("missing pinned row ") + plugin + "/" + kernel +
                            " for op '" + op + "'");
    require(hit->info->launch != nullptr, std::string("pinned row ") + kernel + " has no launcher");
    require(hit->info->domain == RAD_DOMAIN_DEVICE,
            std::string("pinned row ") + kernel + " is not a device row");
    std::printf("ROW role=pinned plugin=%s kernel=%s op=%s\n", hit->plugin.c_str(),
                hit->info->name, op);
    return hit;
}

rad::Resolved resolve_candidate(rad::Registry& reg, const char* op, const char* kernel,
                                const rad::Geometry& geometry, int expected_operands) {
    const RadOpSchema* schema = reg.schema(op);
    require(schema != nullptr, std::string("no schema registered for op '") + op + "'");
    require(schema->n_operands == expected_operands,
            std::string("op '") + op + "' declares " + std::to_string(schema->n_operands) +
            " operands, this comparator binds " + std::to_string(expected_operands));
    std::string error;
    const int v = reg.validate(op, geometry.params(), geometry.n_params(), &error);
    require(v == RAD_OK, std::string("parameters rejected for '") + op + "': " + error);

    rad::SelectionTrace why;
    rad::Resolved r = reg.resolve(op, geometry, RAD_DOMAIN_DEVICE, &why);
    require(r.row != nullptr, std::string("candidate '") + op + "' did not resolve: " + why.reason);
    require(r.row->plugin == "mine", std::string("candidate '") + op + "' resolved to plugin '" +
                                    r.row->plugin + "', not 'mine'");
    require(r.row->info && r.row->info->name &&
            std::strcmp(r.row->info->name, kernel) == 0,
            std::string("candidate '") + op + "' resolved to kernel '" +
            (r.row->info && r.row->info->name ? r.row->info->name : "?") +
            "', expected '" + kernel + "' (no fallback is accepted)");
    std::printf("ROW role=candidate plugin=%s kernel=%s op=%s\n", r.row->plugin.c_str(),
                r.row->info->name, op);
    return r;
}

rad::Resolved bind_row(rad::Registry& reg, const rad::KernelRow* row,
                       const rad::Geometry& geometry) {
    rad::Resolved r;
    r.row = row;
    r.geom = geometry;
    check(reg.init_instance(r, 0, 1), std::string("init row '") + row->info->name + "'");
    return r;
}

/* NO SCRATCH IS FAKED. Every row this comparator pins declares `scratch == nullptr`, so a row that
 * starts declaring one is a contract change, not something to paper over with a host vector (which
 * a device kernel cannot read anyway). It fails here, on every path, rather than only when timed. */
void launch(const rad::Resolved& r, const std::vector<RadTensor>& tensors, RadStream stream,
            std::string_view what) {
    RadArgs a{};
    a.t = tensors.data();
    a.n_t = int(tensors.size());
    a.p = r.geom.params();
    a.n_p = r.geom.n_params();
    a.instance = r.instance;
    a.rank = 0;
    a.world_size = 1;
    if (r.row->info->scratch) {
        const int64_t bytes = r.row->info->scratch(&a);
        if (bytes < 0) fail(what, "negative scratch");
        if (bytes != 0) fail(what, "row declares device scratch this comparator cannot supply");
    }
    check(r.row->info->launch(&a, stream), what);
}

/* ------------------------------------------------------------------------------- comparison */

size_t compare_side(const Session& s, const Scene& sc, Side& cand, Side& ref,
                    const std::string& label) {
    std::vector<std::vector<uint8_t>> c, r;
    cand.download(s, sc, c);
    ref.download(s, sc, r);
    size_t compared = 0;

    for (int idx : sc.compare) {
        const auto& a = c[size_t(idx)];
        const auto& b = r[size_t(idx)];
        require(a.size() == b.size(), label + ": buffer size disagreement");
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] == b[i]) continue;
            char msg[512];
            std::snprintf(msg, sizeof msg,
                          "%s: buffer '%s' differs at frame offset %zu (payload %zu): "
                          "candidate=0x%02x reference=0x%02x",
                          label.c_str(), sc.name[size_t(idx)].c_str(), i,
                          i >= kPad ? i - kPad : 0, a[i], b[i]);
            throw std::runtime_error(msg);
        }
        compared += a.size();
    }

    /* SENTINELS AND UNWRITTEN REGIONS. Comparing candidate against reference alone would not catch
     * both sides writing a byte nobody asked for; the mask is the set of bytes the operation may
     * write, and every other byte must still hold the pristine image. */
    for (size_t idx = 0; idx < sc.mask.size(); ++idx) {
        if (sc.mask[idx].empty()) continue;
        const auto& a = c[idx];
        for (size_t i = 0; i < a.size(); ++i) {
            if (sc.mask[idx][i] || a[i] == sc.image[idx][i]) continue;
            char msg[512];
            std::snprintf(msg, sizeof msg,
                          "%s: buffer '%s' wrote an unowned byte at frame offset %zu "
                          "(candidate=0x%02x pristine=0x%02x)",
                          label.c_str(), sc.name[idx].c_str(), i, a[i], sc.image[idx][i]);
            throw std::runtime_error(msg);
        }
    }
    std::printf("COMPARE_PASS %s buffers=%zu bytes=%zu\n", label.c_str(), sc.compare.size(),
                compared);
    return compared;
}

/* ---------------------------------------------------------------------------------- geometry */

rad::Geometry conv_geometry(int64_t T, int64_t H, int64_t block, int64_t taps = 2,
                            int64_t group = 16, const char* dtype = "bf16") {
    rad::Geometry g;
    g.set_i("T", T);
    g.set_i("hidden_size", H);
    g.set_i("taps", taps);
    g.set_i("group_size", group);
    g.set_i("NG", H / group);
    g.set_i("block_size", block);
    g.set_s("dtype", dtype);
    return g;
}

rad::Geometry kstore_geometry(int64_t M, int64_t nkv, int64_t hd, int64_t block,
                              const char* kv_dtype = "bf16") {
    rad::Geometry g;
    g.set_i("M", M);
    g.set_i("n_head_kv", nkv);
    g.set_i("head_dim", hd);
    g.set_f("eps", 1e-6);
    g.set_f("wadd", 0.0);
    g.set_f("theta", 10000.0);
    g.set_f("scale", 1.0);
    g.set_i("block_size", block);
    g.set_s("kv_dtype", kv_dtype);
    return g;
}

rad::Geometry gate_geometry(int64_t M, int64_t N, int64_t K, const char* dtype = "bf16") {
    rad::Geometry g;
    g.set_i("M", M);
    g.set_i("N", N);
    g.set_i("K", K);
    g.set_s("dtype", dtype);
    return g;
}

/* ------------------------------------------------------------------------------ conv scene */

struct ConvCase {
    const char* name = "";
    int64_t rows = 0;
    int64_t H = 0;
    int64_t block = 0;
    int64_t band = 0;        /* the T parameter: the real row count, or a ceiling above it */
    int     delta_half = 0;  /* which half of [T, 2, taps, NG] this call's slice names */
    bool    inplace = true;  /* out aliases residual, the block's real call */
};

Scene conv_scene(const ConvCase& c) {
    const int64_t taps = 2, group = 16;
    const int64_t NG = c.H / group;
    const int64_t wide = taps * NG;          /* the slice's own width */
    const int64_t dpitch = 2 * taps * NG;    /* the projection's row pitch */
    const int64_t T = c.band;

    Scene sc;

    /* ROUNDING BOUNDARIES, constructed rather than hoped for. With x = 1.0, base = 1.0 (or -1.0)
     * and tap 1 skipped (t % block_size == 0), the convolution's f32 accumulator is exactly one of
     * the bf16 midpoints at magnitude 1:
     *   group 0:  1 + 1*2^-8    RNE -> 1.0          (even mantissa wins)
     *   group 1:  1 + 3*2^-8    RNE -> 1 + 2^-7     (even mantissa wins)
     *   group 2: -1 - 1*2^-8    RNE -> -1.0
     * Each is exactly representable in f32 and NOT in bf16, so the mandated RNE narrowing is what
     * decides the bit: truncation and round-half-up both differ here. The residual add repeats the
     * test one step later, because bf16(1 + 2^-8) = 1.0 and 2^-8 + 1.0 is again a midpoint. */
    std::vector<uint8_t> x(size_t(c.rows * c.H) * 2, 0);
    std::vector<uint8_t> proj(size_t(T * dpitch) * 2, 0);
    std::vector<uint8_t> base(size_t(taps * c.H) * 2, 0);
    std::vector<uint8_t> resid(size_t(c.rows * c.H) * 2, 0);
    std::vector<uint8_t> h2(size_t(c.rows * c.H) * 2, 0);

    const float tie_delta = 1.0f / 256.0f;   /* 2^-8 */
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t tap = 0; tap < taps; ++tap) {
            for (int64_t g = 0; g < NG; ++g) {
                float dv = rnd() * 0.25f;
                if (tap == 0 && g < 3)
                    dv = (g == 0) ? tie_delta : (g == 1 ? 3.0f * tie_delta : -tie_delta);
                const bool live = (t < c.rows) && (tap == 0 || t % c.block >= tap);
                /* The half this call names carries the real coefficients; the other half carries a
                 * different pattern, so a wrong-half read cannot agree by construction. */
                const size_t lo = size_t(t * dpitch + tap * NG + g);
                const size_t hi = size_t(t * dpitch + wide + tap * NG + g);
                if (c.delta_half == 0) {
                    put_bf(proj, lo, live ? to_bf(dv) : to_bf(rnd() * 0.25f));
                    put_bf(proj, hi, to_bf(rnd() * 0.25f));
                } else {
                    put_bf(proj, lo, to_bf(rnd() * 0.25f));
                    put_bf(proj, hi, live ? to_bf(dv) : to_bf(rnd() * 0.25f));
                }
            }
        }
    }

    for (int64_t tap = 0; tap < taps; ++tap) {
        for (int64_t h = 0; h < c.H; ++h) {
            const int64_t g = h / group;
            float v = rnd() * 0.5f;
            if (tap == 0 && g < 3) v = (g == 2) ? -1.0f : 1.0f;
            put_bf(base, size_t(tap * c.H + h), to_bf(v));
        }
    }

    for (int64_t t = 0; t < c.rows; ++t) {
        for (int64_t h = 0; h < c.H; ++h) {
            const int64_t g = h / group;
            const float xv = (g < 3) ? 1.0f : rnd() * 0.5f;
            put_bf(x, size_t(t * c.H + h), to_bf(xv));

            /* The residual is the add's tie operand: 2^-8 against a value near 1.0. */
            const float rv = (g < 3) ? tie_delta : rnd() * 0.25f;
            put_bf(resid, size_t(t * c.H + h), to_bf(rv));

            /* A distinctive pattern the kernel must overwrite on every row it owns. */
            put_bf(h2, size_t(t * c.H + h), to_bf(-0.75f));
        }
    }

    const int xb = sc.add_buffer("x", x, no_writes(kPad + x.size() + kPad));
    const int db = sc.add_buffer("delta_projection", proj, no_writes(kPad + proj.size() + kPad));
    const int bb = sc.add_buffer("base", base, no_writes(kPad + base.size() + kPad));
    /* `conv_h2` is the reference chain's own temporary: the candidate has no such buffer, so it is
     * not compared and carries no write mask. Its content still matters -- the add consumes it --
     * but that is proven by the output bytes, not by inspecting it. */
    const int hb = sc.add_buffer("conv_h2", h2, {});

    /* residual/out share ONE buffer when in place, exactly as the block issues the add. Out of
     * place, the residual must come back untouched, so its mask permits nothing. */
    std::vector<uint8_t> rm = no_writes(kPad + resid.size() + kPad);
    for (int64_t t = 0; t < c.rows; ++t)
        for (size_t i = 0; i < size_t(c.H) * 2; ++i)
            rm[framed_at(size_t(t * c.H) * 2 + i)] = 1;
    const int rb = sc.add_buffer("residual", resid, c.inplace ? rm : no_writes(rm.size()));
    int ob = rb;
    if (!c.inplace) ob = sc.add_buffer("out", resid, rm);

    OpSpec xo{};
    xo.buf = xb; xo.dtype = RAD_BF16; xo.rank = 2;
    xo.shape[0] = c.rows; xo.shape[1] = c.H; xo.stride[0] = c.H; xo.stride[1] = 1;

    OpSpec d = xo;
    d.buf = db; d.offset = int64_t(c.delta_half) * wide * 2;
    d.shape[0] = T; d.shape[1] = wide; d.stride[0] = dpitch; d.stride[1] = 1;

    OpSpec bo{};
    bo.buf = bb; bo.dtype = RAD_BF16; bo.rank = 2;
    bo.shape[0] = taps; bo.shape[1] = c.H; bo.stride[0] = c.H; bo.stride[1] = 1;

    OpSpec ro{};
    ro.buf = rb; ro.dtype = RAD_BF16; ro.rank = 2;
    ro.shape[0] = c.rows; ro.shape[1] = c.H; ro.stride[0] = c.H; ro.stride[1] = 1;

    OpSpec oo = ro;
    oo.buf = ob;

    OpSpec ho = xo;
    ho.buf = hb;

    sc.cand = { xo, d, bo, ro, oo };
    sc.ref = { xo, d, bo, ho,   /* dflash_conv: x, delta, base, out */
               ro, ho, oo };    /* add: a, b, y */
    sc.compare = { xb, db, bb, rb };
    if (!c.inplace) sc.compare.push_back(ob);
    return sc;
}

size_t run_conv_case(const Session& s, rad::Registry& reg, const rad::KernelRow* conv,
                     const rad::KernelRow* add, const ConvCase& c) {
    Scene sc = conv_scene(c);
    rad::Resolved cand = resolve_candidate(reg, "dflash_conv_add",
                                           "mine_dflash_conv_add_t2_g16_bf16",
                                           conv_geometry(c.band, c.H, c.block), 5);
    check(reg.init_instance(cand, 0, 1), "init candidate dflash_conv_add");

    rad::Resolved ref_conv = bind_row(reg, conv, conv_geometry(c.band, c.H, c.block));
    rad::Geometry ag;
    ag.set_i("M", c.rows);
    ag.set_i("n", c.H);
    ag.set_s("dtype", "bf16");
    rad::Resolved ref_add = bind_row(reg, add, ag);

    Side cand_side, ref_side;
    cand_side.upload(s, sc);
    ref_side.upload(s, sc);

    launch(cand, cand_side.bind(sc.cand), s.stream, "candidate dflash_conv_add");

    /* The unfused pair: the convolution into the real bf16 plane, then the add that reads it. */
    std::vector<RadTensor> rt = ref_side.bind(sc.ref);
    std::vector<RadTensor> conv_t(rt.begin(), rt.begin() + 4);
    std::vector<RadTensor> add_t(rt.begin() + 4, rt.end());
    launch(ref_conv, conv_t, s.stream, "reference dflash_conv");
    launch(ref_add, add_t, s.stream, "reference add");

    check(rad_stream_sync(s.stream), "conv case completion");
    char label[256];
    std::snprintf(label, sizeof label,
                  "convadd/%s rows=%lld H=%lld block=%lld band=%lld half=%d %s",
                  c.name, (long long)c.rows, (long long)c.H, (long long)c.block,
                  (long long)c.band, c.delta_half, c.inplace ? "in-place" : "out-of-place");
    const size_t bytes = compare_side(s, sc, cand_side, ref_side, label);
    std::printf("CASE_PASS %s\n", label);
    return bytes;
}

/* ----------------------------------------------------------------------------- conv refusals */
/* Each refusal runs through the SAME candidate row the parity cases use, with exactly one
 * condition broken, so the code returned is the contract's and not a test-local guess. */
struct ConvRefusal {
    const char* name;
    int expected;
};

size_t run_conv_refusals(rad::Registry& reg) {
    const int64_t rows = 8, block = 8, max_H = 160;
    const int64_t taps = 2;
    /* DEVICE scratch, not host memory. Every case is expected to be refused by the validation, so
     * nothing is dereferenced -- but if a guard ever regressed and the launch proceeded, it would
     * address real device memory instead of host pointers. */
    const int64_t lane_bytes = 8 * max_H * 2;
    DeviceBuffer pool(size_t(6) * size_t(lane_bytes) + 4096);
    uint8_t* base_ptr = static_cast<uint8_t*>(pool.data);
    auto slot = [&](int i) { return base_ptr + size_t(i) * size_t(lane_bytes); };
    /* The alignment cases expect RAD_E_ALIGN from a PITCH, not from the pointer, so the scratch
     * itself must satisfy the 16-byte rule the validators check first. */
    require((reinterpret_cast<uintptr_t>(base_ptr) & 15u) == 0,
            "conv scratch must be 16-byte aligned for the alignment refusal cases");

    const ConvRefusal runs[] = {
        { "dtype-param-f32",            RAD_E_DTYPE },       /* geometry says f32 */
        { "taps-3",                     RAD_E_UNSUPPORTED },
        { "hidden-not-multiple-of-64",  RAD_E_UNSUPPORTED },
        { "delta-packed-pitch",         RAD_E_STRIDE },      /* the pitch no shape check sees */
        { "pitch-not-16b-aligned",      RAD_E_ALIGN },
        { "out-partial-alias-residual", RAD_E_INVAL },
        { "out-alias-x",                RAD_E_INVAL },
        { "operand-dtype-f32",          RAD_E_DTYPE },
    };
    size_t done = 0;
    const rad::KernelRow* row = pin_row(reg, "dflash_conv_add",
                                        "mine_dflash_conv_add_t2_g16_bf16", "mine");
    for (const ConvRefusal& r : runs) {
        const std::string n = r.name;
        const int64_t H = n == "hidden-not-multiple-of-64" ? 120 : 128;
        const int64_t taps_p = n == "taps-3" ? 3 : taps;
        const int64_t NG = H / 16;
        const int64_t dpitch = 2 * taps_p * NG;
        const int64_t dspitch = n == "delta-packed-pitch" ? taps_p * NG : dpitch;
        const int64_t rpitch = n == "pitch-not-16b-aligned" ? 132 : H;
        const uint32_t dt = n == "operand-dtype-f32" ? RAD_F32 : RAD_BF16;
        const char* dtype_param = n == "dtype-param-f32" ? "f32" : "bf16";

        RadTensor t[5]{};
        auto fill = [&](int i, void* data, int rank, std::initializer_list<int64_t> shape) {
            t[i].data = data; t[i].dtype = dt; t[i].rank = uint32_t(rank);
            int k = 0;
            for (int64_t v : shape) t[i].shape[k++] = v;
            int64_t stride = 1;
            for (int k2 = rank - 1; k2 >= 0; --k2) { t[i].stride[k2] = stride; stride *= t[i].shape[k2]; }
        };
        fill(0, slot(0), 2, {rows, H});                       /* x */
        fill(1, slot(1), 2, {rows, taps_p * NG});             /* delta slice */
        t[1].stride[0] = dspitch;
        fill(2, slot(2), 2, {taps_p, H});                     /* base */
        fill(3, slot(3), 2, {rows, H});                       /* residual */
        t[3].stride[0] = rpitch;
        fill(4, slot(4), 2, {rows, H});                       /* out */
        t[4].stride[0] = rpitch;
        if (n == "out-alias-x") t[4].data = t[0].data;
        if (n == "out-partial-alias-residual") t[4].data = static_cast<uint8_t*>(t[3].data) + 16;

        rad::Geometry g = conv_geometry(rows, H, block, taps_p, 16, dtype_param);
        RadArgs a{};
        a.t = t; a.n_t = 5;
        a.p = g.params(); a.n_p = g.n_params();
        a.rank = 0; a.world_size = 1;
        const int got = row->info->launch(&a, nullptr);
        if (got != r.expected)
            throw std::runtime_error(std::string("convadd refusal '") + r.name + "': got " +
                                     rad_strerror(got) + " (" + std::to_string(got) +
                                     "), expected " + rad_strerror(r.expected) + " (" +
                                     std::to_string(r.expected) + ")");
        std::printf("REFUSAL_PASS convadd/%s status=%d\n", r.name, got);
        ++done;
    }
    return done;
}

/* ---------------------------------------------------------------------------- kstore scene */

/* EXCEPTIONAL VALUES ARE PART OF THE CONTRACT, not decoration.
 *
 * The previous BF16 conversion helper in this tree truncated instead of rounding to even, and a
 * truncating f32->bf16 differs from RNE exactly at midpoints and on overflow/underflow edges; a
 * payload of ordinary noise never lands there. The norm's `rsqrtf(ss/n + eps)` also has two
 * regimes no ordinary row reaches: `ss` overflowing to +Inf (s becomes 0, so the row collapses to
 * signed zeros) and `ss` underflowing to 0 (s becomes `rsqrtf(eps)`). Both sides must agree bit for
 * bit in every one of those regimes, and NaN payloads must come out the same, because the
 * comparison is `memcmp`.
 *
 * Each mode touches only HEAD 0 of row 0 (or a whole head where the mode is about the reduction),
 * so the remaining heads stay finite and the ordinary path is exercised in the same launch. */
enum class ValueMode {
    kNormal,       /* the spread of ordinary magnitudes */
    kExceptional,  /* Inf, -Inf, three quiet NaN payloads, -0, min/max subnormal, max finite both
                    * signs, at columns 0..11, 64, 127 of row 0 head 0; v a different shift */
    kOverflow,     /* row 0 head 0 is all max-finite: the sum of squares overflows f32 to +Inf and
                    * the scale becomes rsqrtf(Inf) = 0, so the row collapses to signed zeros */
    kUnderflow,    /* row 0 head 0 is all min-subnormal: the sum of squares must underflow to 0 */
};

constexpr uint16_t kBfPosInf    = 0x7F80u;
constexpr uint16_t kBfNegInf    = 0xFF80u;
/* QUIET NaNs ONLY, with DIFFERENT payloads. A signalling NaN is quieted on first use, and which
 * operand's payload survives quieting is hardware-defined; a difference there would be an artefact
 * of the probe, not of the kernel. Quiet payloads propagate unchanged through every arithmetic
 * operation both sides perform, so a canonicalising implementation is still caught. */
constexpr uint16_t kBfNanA      = 0x7FC1u;   /* quiet, payload 1 */
constexpr uint16_t kBfNanB      = 0x7FC5u;   /* quiet, payload 5 */
constexpr uint16_t kBfNegNan    = 0xFFC3u;   /* quiet, negative, payload 3 */
constexpr uint16_t kBfNegZero   = 0x8000u;
constexpr uint16_t kBfMinSub    = 0x0001u;   /* 2^-133, the smallest bf16 subnormal */
constexpr uint16_t kBfMaxSub    = 0x007Fu;
constexpr uint16_t kBfMaxFinite = 0x7F7Fu;   /* 3.3895e38 */
constexpr uint16_t kBfNegMax    = 0xFF7Fu;

/* Twelve slots, indexed directly for h < 12 so every class is present exactly once per row, plus
 * two probes further down the head. */
constexpr uint16_t kExceptionalTable[] = {
    kBfPosInf, kBfNegInf, kBfNanA, kBfNanB, kBfNegNan, kBfNegZero,
    kBfMinSub, kBfMaxSub, kBfMaxFinite, kBfNegMax,
    0x3F80u /* 1.0 */, 0x0000u /* +0 */,
};

bool is_exceptional_slot(int64_t h) {
    return h < int64_t(sizeof(kExceptionalTable) / sizeof(kExceptionalTable[0])) || h == 64 || h == 127;
}

uint16_t exceptional_bf(int64_t h) {
    if (h < int64_t(sizeof(kExceptionalTable) / sizeof(kExceptionalTable[0])))
        return kExceptionalTable[h];
    return h == 64 ? kBfMaxFinite : kBfMinSub;
}

/* The gain is f32 in this model, so its exceptional entries are f32 BIT PATTERNS: +Inf, a NaN with
 * a payload, -0, the smallest f32 subnormal, and the largest finite f32. */
void put_f32_bits(std::vector<uint8_t>& payload, size_t element, uint32_t bits) {
    require(element * 4 + 4 <= payload.size(), "f32 write past the payload");
    std::memcpy(payload.data() + element * 4, &bits, 4);
}

struct KCase {
    const char* name;
    int64_t rows;
    int64_t nkv;
    int64_t block;
    int64_t blocks;
    int64_t kpitch;             /* >= nkv*128; pitched only where the reference can read it */
    std::vector<int32_t> pos;
    std::vector<int32_t> slot;  /* distinct; negative and past-the-end are the store's own rule */
    ValueMode mode = ValueMode::kNormal;
    bool extreme_gain = false;
};

Scene kstore_scene(const KCase& c) {
    const int64_t hd = 128;
    const int64_t lane = c.nkv * hd;
    require(c.kpitch >= lane, "kstore scene pitch narrower than the row");
    require(int64_t(c.pos.size()) == c.rows && int64_t(c.slot.size()) == c.rows,
            "kstore scene position/slot count");

    Scene sc;
    std::vector<uint8_t> k(size_t(c.rows * c.kpitch) * 2, 0);
    std::vector<uint8_t> v(size_t(c.rows * c.kpitch) * 2, 0);
    std::vector<uint8_t> w(size_t(hd) * 4, 0);
    std::vector<uint8_t> pos(size_t(c.rows) * 4, 0);
    std::vector<uint8_t> slot(size_t(c.rows) * 4, 0);
    std::vector<uint8_t> kv(size_t(c.blocks * c.nkv * c.block * 2 * hd) * 2, 0);

    for (int64_t t = 0; t < c.rows; ++t) {
        for (int64_t h = 0; h < lane; ++h) {
            /* Exact powers of two mixed into the spread, so the norm is exercised on values whose
             * scaling does not itself round. */
            const float kx = (h % 7 == 0) ? std::ldexp(1.0f, int((h % 5) - 2)) : rnd() * 2.0f;
            uint16_t kb = to_bf(kx);
            const bool head0 = h < hd;           /* head 0 of this row */
            switch (c.mode) {
            case ValueMode::kExceptional:
                if (t == 0 && is_exceptional_slot(h)) kb = exceptional_bf(h);
                break;
            case ValueMode::kOverflow:
                /* EVERY element of row 0 head 0 is the largest FINITE bf16 (3.3895e38, which is a
                 * finite f32 too). Its square overflows f32, so the sum of squares saturates to
                 * +Inf and the scale is rsqrtf(Inf) = 0. The product is `finite * 0`, i.e. +0 or
                 * -0 by the sign of the input -- NOT a NaN. A NaN here would need an infinite
                 * input or an infinite gain, which is what the exceptional cases supply. */
                if (t == 0 && head0) kb = kBfMaxFinite;
                break;
            case ValueMode::kUnderflow:
                /* EVERY element is 2^-133, whose square is 2^-266: the sum underflows to +0, so the
                 * scale is rsqrtf(0 + eps) -- the other end of the same expression. */
                if (t == 0 && head0) kb = kBfMinSub;
                break;
            case ValueMode::kNormal:
                break;
            }
            put_bf(k, size_t(t * c.kpitch + h), kb);
            /* v is untouched by both sides; different bits so a wrong-slot write cannot agree. */
            put_bf(v, size_t(t * c.kpitch + h),
                   (t == 0 && is_exceptional_slot(h)) ? exceptional_bf(h + 1) : to_bf(rnd() * 2.0f));
        }
        put_i32(pos, size_t(t), c.pos[size_t(t)]);
        put_i32(slot, size_t(t), c.slot[size_t(t)]);
    }
    if (c.extreme_gain) {
        /* A gain row with an infinity, a NaN payload, -0, the smallest f32 subnormal and the
         * largest finite f32 -- the second place a BF16 narrowing or an `rsqrtf` regime can differ.
         * The rest stays near 1 so every other head keeps a finite result. */
        for (int64_t h = 0; h < hd; ++h) put_f32(w, size_t(h), 1.0f + float(h % 5) * 0.125f);
        put_f32_bits(w, 0, 0x7F800000u);   /* +Inf   */
        put_f32_bits(w, 1, 0x7FC00001u);   /* NaN, payload 1 */
        put_f32_bits(w, 2, 0x80000000u);   /* -0     */
        put_f32_bits(w, 3, 0x00000001u);   /* smallest f32 subnormal */
        put_f32_bits(w, 127, 0x7F7FFFFF);  /* largest finite f32 */
    } else {
        for (int64_t h = 0; h < hd; ++h) put_f32(w, size_t(h), 1.0f + float(h % 5) * 0.125f);
    }
    /* The cache starts as a pattern no store would produce. */
    for (size_t i = 0; i < kv.size(); i += 2) {
        const uint16_t seed = to_bf(float((i / 2) % 23) * 0.03125f - 0.25f);
        std::memcpy(kv.data() + i, &seed, 2);
    }

    std::vector<uint8_t> km = no_writes(kPad + k.size() + kPad);
    for (int64_t t = 0; t < c.rows; ++t)
        for (size_t i = 0; i < size_t(lane) * 2; ++i)
            km[framed_at(size_t(t * c.kpitch) * 2 + i)] = 1;

    std::vector<uint8_t> kvm = no_writes(kPad + kv.size() + kPad);
    for (int64_t t = 0; t < c.rows; ++t) {
        const int32_t s = c.slot[size_t(t)];
        if (s < 0 || s / c.block >= c.blocks) continue;   /* the store's own skip rule */
        const int64_t blk = s / c.block, off = s % c.block;
        for (int64_t h = 0; h < c.nkv; ++h) {
            const size_t at = size_t((((blk * c.nkv + h) * c.block) + off) * 2 * hd);
            for (size_t i = 0; i < size_t(2 * hd) * 2; ++i) kvm[framed_at(at * 2 + i)] = 1;
        }
    }

    const int kb = sc.add_buffer("k", k, km);
    const int vb = sc.add_buffer("v", v, no_writes(kPad + v.size() + kPad));
    const int wb = sc.add_buffer("k_norm_gain", w, no_writes(kPad + w.size() + kPad));
    const int pb = sc.add_buffer("positions", pos, no_writes(kPad + pos.size() + kPad));
    const int sb = sc.add_buffer("slot_mapping", slot, no_writes(kPad + slot.size() + kPad));
    const int cb = sc.add_buffer("kv_cache", kv, kvm);

    OpSpec ko{};
    ko.buf = kb; ko.dtype = RAD_BF16; ko.rank = 2;
    ko.shape[0] = c.rows; ko.shape[1] = lane; ko.stride[0] = c.kpitch; ko.stride[1] = 1;

    OpSpec vo = ko;
    vo.buf = vb;

    OpSpec wo{};
    wo.buf = wb; wo.dtype = RAD_F32; wo.rank = 1;
    wo.shape[0] = hd; wo.stride[0] = 1;

    OpSpec po{};
    po.buf = pb; po.dtype = RAD_I32; po.rank = 1;
    po.shape[0] = c.rows; po.stride[0] = 1;

    OpSpec so = po;
    so.buf = sb;

    OpSpec kvo{};
    kvo.buf = cb; kvo.dtype = RAD_BF16; kvo.rank = 4;
    kvo.shape[0] = c.blocks; kvo.shape[1] = c.nkv; kvo.shape[2] = c.block; kvo.shape[3] = 2 * hd;
    kvo.stride[0] = c.nkv * c.block * 2 * hd;
    kvo.stride[1] = c.block * 2 * hd;
    kvo.stride[2] = 2 * hd;
    kvo.stride[3] = 1;

    sc.cand = { ko, vo, wo, po, so, kvo };
    sc.ref = sc.cand;
    sc.compare = { kb, vb, wb, pb, sb, cb };
    return sc;
}

size_t run_kstore_case(const Session& s, rad::Registry& reg, const rad::KernelRow* rms,
                       const rad::KernelRow* rope, const rad::KernelRow* store, const KCase& c) {
    Scene sc = kstore_scene(c);
    rad::Resolved cand = resolve_candidate(reg, "dflash_k_norm_rope_store",
                                           "mine_dflash_k_norm_rope_store_hd128",
                                           kstore_geometry(c.rows, c.nkv, 128, c.block), 6);
    check(reg.init_instance(cand, 0, 1), "init candidate dflash_k_norm_rope_store");

    rad::Geometry ng;
    ng.set_i("M", c.rows * c.nkv);   /* the norm is per (token, head) */
    ng.set_i("n", 128);
    ng.set_f("eps", 1e-6);
    ng.set_f("wadd", 0.0);
    ng.set_s("dtype", "bf16");
    rad::Resolved ref_norm = bind_row(reg, rms, ng);

    rad::Geometry rg;
    rg.set_i("M", c.rows);
    rg.set_i("head_dim", 128);
    rg.set_i("n_head", c.nkv);
    rg.set_i("n_head_kv", 0);        /* K only: every head present is rotated */
    rg.set_f("theta", 10000.0);
    rg.set_f("scale", 1.0);
    rg.set_s("mode", "neox");
    rad::Resolved ref_rope = bind_row(reg, rope, rg);

    rad::Geometry kg;
    kg.set_i("M", c.rows);
    kg.set_i("head_dim", 128);
    kg.set_i("n_head_kv", c.nkv);
    kg.set_i("block_size", c.block);
    kg.set_s("kv_dtype", "bf16");
    rad::Resolved ref_store = bind_row(reg, store, kg);

    Side cand_side, ref_side;
    cand_side.upload(s, sc);
    ref_side.upload(s, sc);

    launch(cand, cand_side.bind(sc.cand), s.stream, "candidate dflash_k_norm_rope_store");

    std::vector<RadTensor> rt = ref_side.bind(sc.ref);
    std::vector<RadTensor> norm_t = { rt[0], rt[2], rt[0] };            /* x, w, y (in place) */
    std::vector<RadTensor> rope_t = { rt[0], rt[3] };                   /* qkv, positions */
    std::vector<RadTensor> store_t = { rt[0], rt[1], rt[4], rt[5] };    /* k, v, slot, cache */
    launch(ref_norm, norm_t, s.stream, "reference rmsnorm");
    launch(ref_rope, rope_t, s.stream, "reference rope");
    launch(ref_store, store_t, s.stream, "reference kv_store");

    check(rad_stream_sync(s.stream), "kstore case completion");
    char label[256];
    static const char* const modes[] = { "normal", "exceptional", "overflow", "underflow" };
    std::snprintf(label, sizeof label,
                  "kstore/%s rows=%lld nkv=%lld block=%lld blocks=%lld pitch=%lld values=%s%s",
                  c.name, (long long)c.rows, (long long)c.nkv, (long long)c.block,
                  (long long)c.blocks, (long long)c.kpitch, modes[int(c.mode)],
                  c.extreme_gain ? " extreme-gain" : "");
    const size_t bytes = compare_side(s, sc, cand_side, ref_side, label);
    std::printf("CASE_PASS %s\n", label);
    return bytes;
}

/* -------------------------------------------------------------------------- kstore refusals */

size_t run_kstore_refusals(rad::Registry& reg) {
    const int64_t rows = 4, nkv = 2, hd = 128, block = 16, blocks = 2;
    const int64_t lane = nkv * hd;
    /* DEVICE scratch: every case is expected to be refused, so nothing is dereferenced, but an
     * accidental launch must not address host memory. */
    DeviceBuffer pool(6u << 20);
    uint8_t* base_ptr = static_cast<uint8_t*>(pool.data);
    auto slot = [&](int i) { return base_ptr + size_t(i) * (1u << 20); };
    require((reinterpret_cast<uintptr_t>(base_ptr) & 15u) == 0,
            "device scratch must be 16-byte aligned for the alignment refusal cases");
    struct Run {
        const char* name; int expected; int64_t hd_p; int64_t kpitch; bool contig; bool missing;
        int cache_dtype; const char* kv_dtype;
    };
    const Run runs[] = {
        { "head-dim-64",              RAD_E_UNSUPPORTED, 64, 128,  true,  false, RAD_BF16, "bf16" },
        { "kv-dtype-f32",             RAD_E_DTYPE,       128, lane, true,  false, RAD_BF16, "f32"  },
        { "cache-dtype-f32",          RAD_E_DTYPE,       128, lane, true,  false, RAD_F32,  "bf16" },
        { "pitched-plane-multi-head", RAD_E_STRIDE,      128, lane + 8, true, false, RAD_BF16, "bf16" },
        { "cache-not-packed",         RAD_E_STRIDE,      128, lane, false, false, RAD_BF16, "bf16" },
        { "missing-cache",            RAD_E_INVAL,       128, lane, true,  true,  RAD_BF16, "bf16" },
    };

    const rad::KernelRow* row = pin_row(reg, "dflash_k_norm_rope_store",
                                        "mine_dflash_k_norm_rope_store_hd128", "mine");
    size_t done = 0;
    for (const Run& r : runs) {
        RadTensor t[6]{};
        const int64_t width = nkv * r.hd_p;
        t[0].data = slot(0); t[0].dtype = RAD_BF16; t[0].rank = 2;
        t[0].shape[0] = rows; t[0].shape[1] = width;
        t[0].stride[0] = r.kpitch; t[0].stride[1] = 1;
        t[1] = t[0]; t[1].data = slot(1);
        t[2].data = slot(2); t[2].dtype = RAD_F32; t[2].rank = 1;
        t[2].shape[0] = r.hd_p; t[2].stride[0] = 1;
        t[3].data = slot(3); t[3].dtype = RAD_I32; t[3].rank = 1;
        t[3].shape[0] = rows; t[3].stride[0] = 1;
        t[4] = t[3]; t[4].data = slot(4);
        t[5].data = r.missing ? nullptr : slot(5);
        t[5].dtype = r.cache_dtype; t[5].rank = 4;
        t[5].shape[0] = blocks; t[5].shape[1] = nkv;
        t[5].shape[2] = block; t[5].shape[3] = 2 * r.hd_p;
        t[5].stride[0] = nkv * block * 2 * r.hd_p;
        t[5].stride[1] = block * 2 * r.hd_p;
        t[5].stride[2] = 2 * r.hd_p;
        t[5].stride[3] = r.contig ? 1 : 2;

        rad::Geometry g = kstore_geometry(rows, nkv, r.hd_p, block, r.kv_dtype);
        RadArgs a{};
        a.t = t; a.n_t = 6;
        a.p = g.params(); a.n_p = g.n_params();
        a.rank = 0; a.world_size = 1;
        const int got = row->info->launch(&a, nullptr);
        if (got != r.expected)
            throw std::runtime_error(std::string("kstore refusal '") + r.name + "': got " +
                                     rad_strerror(got) + " (" + std::to_string(got) +
                                     "), expected " + rad_strerror(r.expected) + " (" +
                                     std::to_string(r.expected) + ")");
        std::printf("REFUSAL_PASS kstore/%s status=%d\n", r.name, got);
        ++done;
    }
    return done;
}

/* ------------------------------------------------------------------------------ gate scene */
/* The unfused pair's REAL intermediate: [M, N] bf16, N = 2n, written by the GEMM and read by
 * silu_mul. It is the buffer whose bf16 rounding the fusion must preserve, so it is a device
 * allocation with the pristine image, not a host computation. */

struct GateCase {
    const char* name = "";
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
};

/* `weight_streams` > 1 builds a streaming scene: the same activation and output geometry, but one
 * weight buffer per stream and one output pair per stream, so every stream is a complete call whose
 * output is byte-compared on its own. Stream 0 keeps exactly the bytes a single-stream scene would
 * have produced, because the noise generator is consumed in the same order. */
Scene gate_scene(const GateCase& c, int weight_streams = 1) {
    require(weight_streams >= 1, "a gate scene needs at least one weight stream");
    Scene sc;
    std::vector<uint8_t> a(size_t(c.M * c.K) * 2, 0);
    std::vector<uint8_t> y(size_t(c.M * (c.N / 2)) * 2, 0);
    std::vector<uint8_t> gu(size_t(c.M * c.N) * 2, 0);

    for (int64_t m = 0; m < c.M; ++m)
        for (int64_t k = 0; k < c.K; ++k)
            put_bf(a, size_t(m * c.K + k), to_bf(rnd() * 1.5f));
    for (size_t i = 0; i < y.size(); i += 2) put_bf(y, i / 2, to_bf(0.5f));
    for (size_t i = 0; i < gu.size(); i += 2) put_bf(gu, i / 2, to_bf(-0.5f));

    /* `y` is the only buffer either side may write, and only over its own [M, n] payload. */
    std::vector<uint8_t> ym = no_writes(kPad + y.size() + kPad);
    for (size_t i = 0; i < y.size(); ++i) ym[framed_at(i)] = 1;

    const int ab = sc.add_buffer("a", a, no_writes(kPad + a.size() + kPad));

    std::vector<int> wb, yb, gb;
    for (int stream = 0; stream < weight_streams; ++stream) {
        std::vector<uint8_t> b(size_t(c.N * c.K) * 2, 0);
        for (int64_t n = 0; n < c.N; ++n)
            for (int64_t k = 0; k < c.K; ++k) {
                /* Powers of two mixed with noise: the gate and up products then land on values
                 * whose own rounding is exact, so the two mandated bf16 roundings are what the
                 * comparison actually tests. */
                const float w = (k % 11 == 0) ? std::ldexp(1.0f, int((k % 5) - 3))
                                              : rnd() * 0.02f;
                put_bf(b, size_t(n * c.K + k), to_bf(w));
            }
        const std::string suffix = weight_streams == 1 ? std::string() : std::to_string(stream);
        wb.push_back(sc.add_buffer("gate_up_weight" + suffix, b,
                                   no_writes(kPad + b.size() + kPad)));
        yb.push_back(sc.add_buffer("y" + suffix, y, ym));
        gb.push_back(sc.add_buffer("gemm_out" + suffix, gu, {}));
    }

    OpSpec ao{};
    ao.buf = ab; ao.dtype = RAD_BF16; ao.rank = 2;
    ao.shape[0] = c.M; ao.shape[1] = c.K; ao.stride[0] = c.K; ao.stride[1] = 1;

    OpSpec yo{};
    yo.buf = yb[0]; yo.dtype = RAD_BF16; yo.rank = 2;
    yo.shape[0] = c.M; yo.shape[1] = c.N / 2; yo.stride[0] = c.N / 2; yo.stride[1] = 1;

    for (int stream = 0; stream < weight_streams; ++stream) {
        OpSpec bo{};
        bo.buf = wb[size_t(stream)]; bo.dtype = RAD_BF16; bo.rank = 2;
        bo.shape[0] = c.N; bo.shape[1] = c.K; bo.stride[0] = c.K; bo.stride[1] = 1;

        OpSpec ys = yo;
        ys.buf = yb[size_t(stream)];

        OpSpec gs = yo;
        gs.buf = gb[size_t(stream)];
        gs.shape[0] = c.M; gs.shape[1] = c.N; gs.stride[0] = c.N; gs.stride[1] = 1;

        sc.cand_ops.push_back({ ao, bo, ys });
        sc.ref_ops.push_back({ ao, bo, gs,    /* gemm_nt: a, b(weight), y */
                               gs, ys });     /* silu_mul: gate_up, y */
    }
    sc.cand = sc.cand_ops.front();
    sc.ref = sc.ref_ops.front();
    /* A single-stream scene compares the operands too, exactly as before. A streaming scene lists
     * only the five outputs for the candidate-versus-reference pass: the weights are read-only, so
     * a difference there could only mean one arm wrote them, and the sentinel mask below already
     * fails on that with the offending buffer named. */
    if (weight_streams == 1) sc.compare = { ab, wb[0], yb[0] };
    else                     sc.compare.assign(yb.begin(), yb.end());
    return sc;
}

size_t run_gate_case(const Session& s, rad::Registry& reg, const rad::KernelRow* gemm,
                     const rad::KernelRow* silu, const GateCase& c) {
    Scene sc = gate_scene(c);
    require(c.N >= 2048 && c.N % 32 == 0 && c.K % 16 == 0,
            "gate case outside the candidate's declared band");
    rad::Resolved cand = resolve_candidate(reg, "dflash_gated_bf16",
                                           "mine_dflash_gated_bf16",
                                           gate_geometry(c.M, c.N, c.K), 3);
    check(reg.init_instance(cand, 0, 1), "init candidate dflash_gated_bf16");

    /* THE REFERENCE GEMM IS PINNED TO sk = 1, because that is the only reduction tree the
     * candidate reproduces. `tune_i` (libr4d/r4d_rows.cpp:5093-5099) reads each tuned key from the
     * argument list FIRST and falls back to the table default only when the caller did not name
     * it, so naming it here makes a tune cache unable to substitute a multi-slice split. `wv` is
     * pinned to its own default for the same reason; neither can change what an element adds.
     *
     * At these shapes the shape ladder would answer sk = 1 anyway (n_tiles * 1 already reaches
     * kFlatTileTarget), and the pin removes the dependence on that arithmetic rather than trusting
     * it. The candidate refuses a pinned sk > 1 at its launcher; a REFERENCE that somehow ran at
     * sk > 1 would be a different baseline, not a candidate failure. */
    rad::Geometry rg = gate_geometry(c.M, c.N, c.K);
    rg.set_i("sk", 1);
    rg.set_i("wv", 1);
    rad::Resolved ref_gemm = bind_row(reg, gemm, rg);
    /* silu_mul's parameters are M and n; K is not its parameter. */
    rad::Geometry sg;
    sg.set_i("M", c.M);
    sg.set_i("n", c.N / 2);
    sg.set_s("dtype", "bf16");
    rad::Resolved ref_silu = bind_row(reg, silu, sg);

    Side cand_side, ref_side;
    cand_side.upload(s, sc);
    ref_side.upload(s, sc);

    launch(cand, cand_side.bind(sc.cand), s.stream, "candidate dflash_gated_bf16");

    std::vector<RadTensor> rt = ref_side.bind(sc.ref);
    std::vector<RadTensor> gemm_t(rt.begin(), rt.begin() + 3);
    std::vector<RadTensor> silu_t(rt.begin() + 3, rt.end());
    launch(ref_gemm, gemm_t, s.stream, "reference gemm_nt");
    launch(ref_silu, silu_t, s.stream, "reference silu_mul");

    check(rad_stream_sync(s.stream), "gate case completion");
    char label[256];
    std::snprintf(label, sizeof label, "gate/%s M=%lld N=%lld K=%lld",
                  c.name, (long long)c.M, (long long)c.N, (long long)c.K);
    const size_t bytes = compare_side(s, sc, cand_side, ref_side, label);
    std::printf("CASE_PASS %s\n", label);
    return bytes;
}

/* --------------------------------------------------------------------------- gate refusals */
/* M = 65 is the band with no candidate row: the contract says the geometry resolves to nothing
 * rather than falling back, so it is checked at SELECTION, not at launch. Every other case is a
 * launch-time refusal of the candidate row itself. */

size_t run_gate_refusals(rad::Registry& reg) {
    const int64_t M = 8, N = 2048, K = 256;
    /* DEVICE scratch: every case is expected to be refused, so nothing is dereferenced, but an
     * accidental launch must not address host memory. The largest weight a case describes is
     * N*K bf16 (2064*264 = 1.09 MB), so each 2 MiB slot covers it with room to spare. */
    DeviceBuffer pool(8u << 20);
    uint8_t* base_ptr = static_cast<uint8_t*>(pool.data);
    auto slot = [&](int i) { return base_ptr + size_t(i) * (2u << 20); };
    require((reinterpret_cast<uintptr_t>(base_ptr) & 15u) == 0,
            "gate scratch must be 16-byte aligned for the alignment refusal case");

    struct Run {
        const char* name; int expected; int64_t M_p; int64_t N_p; int64_t K_p;
        const char* dtype; int rank_a; int misalign_y; int alias_y_a; const char* weight_dtype;
    };
    const Run runs[] = {
        { "dtype-f32",       RAD_E_DTYPE,        M, N, K, "f32", 2, 0, 0, "bf16" },
        { "N-below-2048",    RAD_E_UNSUPPORTED,  M, 1024, K, "bf16", 2, 0, 0, "bf16" },
        { "N-not-32-step",   RAD_E_UNSUPPORTED,  M, 2064, K, "bf16", 2, 0, 0, "bf16" },
        { "K-not-16-step",   RAD_E_UNSUPPORTED,  M, N, 264, "bf16", 2, 0, 0, "bf16" },
        { "rank-3-a",        RAD_E_UNSUPPORTED,  M, N, K, "bf16", 3, 0, 0, "bf16" },
        { "unaligned-y",     RAD_E_ALIGN,        M, N, K, "bf16", 2, 1, 0, "bf16" },
        { "y-aliases-a",     RAD_E_INVAL,        M, N, K, "bf16", 2, 0, 1, "bf16" },
        { "weight-dtype-f32", RAD_E_DTYPE,       M, N, K, "bf16", 2, 0, 0, "f32" },
    };

    const rad::KernelRow* row = pin_row(reg, "dflash_gated_bf16", "mine_dflash_gated_bf16", "mine");
    size_t done = 0;
    for (const Run& r : runs) {
        RadTensor t[3]{};
        t[0].data = slot(0); t[0].dtype = RAD_BF16; t[0].rank = uint32_t(r.rank_a);
        t[0].shape[0] = r.M_p;
        if (r.rank_a == 2) {
            t[0].shape[1] = r.K_p;
        } else {
            /* A rank-3 activation, small enough that its span stays inside the scratch slot. */
            t[0].shape[1] = 1;
            t[0].shape[2] = r.K_p;
        }
        int64_t stride = 1;
        for (int i = r.rank_a - 1; i >= 0; --i) { t[0].stride[i] = stride; stride *= t[0].shape[i]; }

        t[1].data = slot(1);
        t[1].dtype = std::strcmp(r.weight_dtype, "f32") == 0 ? RAD_F32 : RAD_BF16;
        t[1].rank = 2; t[1].shape[0] = r.N_p; t[1].shape[1] = r.K_p;
        t[1].stride[0] = r.K_p; t[1].stride[1] = 1;

        t[2].data = r.alias_y_a ? t[0].data
                                : static_cast<uint8_t*>(slot(2)) + (r.misalign_y ? 2 : 0);
        t[2].dtype = RAD_BF16; t[2].rank = 2;
        t[2].shape[0] = r.M_p; t[2].shape[1] = r.N_p / 2;
        t[2].stride[0] = r.N_p / 2; t[2].stride[1] = 1;

        rad::Geometry g = gate_geometry(r.M_p, r.N_p, r.K_p, r.dtype);
        RadArgs a{};
        a.t = t; a.n_t = 3;
        a.p = g.params(); a.n_p = g.n_params();
        a.rank = 0; a.world_size = 1;
        const int got = row->info->launch(&a, nullptr);
        if (got != r.expected)
            throw std::runtime_error(std::string("gate refusal '") + r.name + "': got " +
                                     rad_strerror(got) + " (" + std::to_string(got) +
                                     "), expected " + rad_strerror(r.expected) + " (" +
                                     std::to_string(r.expected) + ")");
        std::printf("REFUSAL_PASS gate/%s status=%d\n", r.name, got);
        ++done;
    }

    /* M = 65 HAS NO CANDIDATE ROW. Selection must answer with nothing -- a fallback would be a
     * different kernel under the fused op's name. */
    rad::SelectionTrace why;
    const rad::Resolved above = reg.resolve("dflash_gated_bf16", gate_geometry(65, N, K),
                                            RAD_DOMAIN_DEVICE, &why);
    require(above.row == nullptr,
            std::string("gate M=65 must resolve to no row, but selected ") +
            (above.row && above.row->info ? above.row->info->name : "?"));
    std::printf("REFUSAL_PASS gate/M-above-64 unresolved: %s\n", why.reason.c_str());
    ++done;
    return done;
}

/* --------------------------------------------------------------------------------- benchmark */

/* A BENCHMARK CASE IS ADMITTED ONLY AFTER ITS PARITY CASE PASSES. The byte comparison runs first,
 * in full, on the same scene and the same rows the timing then uses; a mismatch throws before a
 * clock is read, so no number below can describe an unverified pairing.
 *
 * WHAT IS TIMED. The engine's own path: host submission, queue execution and the final stream
 * sync, measured with steady_clock around a batch of launches and one checked `rad_stream_sync`.
 * HIP runtime events are deliberately absent -- the report behind this file is host submission
 * plus execution, and a native barrier event's timestamp is not a reliable substitute.
 *
 * WHAT IT IS NOT. These are per-rank microbenchmarks on SYNTHETIC inputs of the model's sizes: the
 * weights carry random bf16 values, not checkpoint bytes, and no engine graph, scheduler or
 * multi-rank traffic is involved. A ratio here is the fused row against the unfused chain on the
 * same buffers, nothing wider.
 *
 * HOT BUFFERS ARE REPEATED. Every iteration writes the same output buffer and, for the gate, reads
 * the same weight buffer, so both sit in cache and the numbers are a best case. Mutable buffers are
 * restored from the pristine image outside the measured window (the reset syncs before the clock
 * starts), so no batch inherits the previous batch's state. */

struct BenchConfig {
    int rounds = 7;              /* paired rounds, alternating which side runs first */
    int warmup = 32;             /* unmeasured iterations before calibration */
    double target_us = 20000.0;  /* a batch aims at this many microseconds */
    int cap = 20000;
};

/* Restore the buffers an operation writes. The copy is asynchronous, so the sync is what keeps the
 * transfer OUT of the measured window: the clock starts after it returns. */
void reload(const Session& s, const Scene& sc, const std::vector<int>& buffers, Side& side) {
    for (size_t i = 0; i < buffers.size(); ++i) {
        bool seen = false;
        for (size_t j = 0; j < i; ++j) seen = seen || buffers[j] == buffers[i];
        if (seen) continue;   /* in-place ops name one buffer twice */
        const int b = buffers[i];
        const std::vector<uint8_t>& image = sc.image[size_t(b)];
        check(rad_memcpy_async(side.buf[size_t(b)]->data, image.data(), int64_t(image.size()),
                               s.stream), "benchmark reset");
    }
    check(rad_stream_sync(s.stream), "benchmark reset completion");
}

template <class F>
double timed_batch(const Session& s, F&& body, int iters, const char* what) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) body();
    check(rad_stream_sync(s.stream), what);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / double(iters);
}

template <class F>
int calibrate(const Session& s, F&& body, const BenchConfig& bc, const char* what) {
    for (int i = 0; i < bc.warmup; ++i) body();
    check(rad_stream_sync(s.stream), what);
    const double per = timed_batch(s, body, 3, what);
    const int n = int(std::llround(bc.target_us / std::max(per, 1e-3)));
    return std::max(1, std::min(bc.cap, n));
}

double median_of(std::vector<double> v) {
    require(!v.empty(), "median of an empty sample");
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* The paired loop both ops share: per-round numbers, then the median of each side. */
template <class C, class R>
void paired_rounds(const char* id, const BenchConfig& bc, int iters_c, int iters_r,
                   C&& candidate, R&& reference) {
    std::vector<double> c_us, r_us;
    for (int round = 0; round < bc.rounds; ++round) {
        const bool cand_first = (round % 2) == 0;
        double tc = 0, tr = 0;
        if (cand_first) { tc = candidate(); tr = reference(); }
        else            { tr = reference(); tc = candidate(); }
        c_us.push_back(tc);
        r_us.push_back(tr);
        std::printf("BENCH_ROUND %s round=%d order=%s cand_us=%.3f ref_us=%.3f "
                    "iters_cand=%d iters_ref=%d\n",
                    id, round + 1, cand_first ? "cand-first" : "ref-first", tc, tr,
                    iters_c, iters_r);
    }
    const double cm = median_of(c_us), rm = median_of(r_us);
    std::printf("BENCH_MEDIAN %s cand_us=%.3f ref_us=%.3f ratio=%.4f rounds=%d "
                "iters_cand=%d iters_ref=%d\n",
                id, cm, rm, rm > 0 ? cm / rm : 0.0, bc.rounds, iters_c, iters_r);
}

size_t bench_conv(const Session& s, rad::Registry& reg, const rad::KernelRow* conv,
                  const rad::KernelRow* add, const ConvCase& c, const BenchConfig& bc) {
    Scene sc = conv_scene(c);
    rad::Resolved cand = resolve_candidate(reg, "dflash_conv_add",
                                           "mine_dflash_conv_add_t2_g16_bf16",
                                           conv_geometry(c.band, c.H, c.block), 5);
    check(reg.init_instance(cand, 0, 1), "init candidate dflash_conv_add");
    rad::Resolved ref_conv = bind_row(reg, conv, conv_geometry(c.band, c.H, c.block));
    rad::Geometry ag;
    ag.set_i("M", c.rows);
    ag.set_i("n", c.H);
    ag.set_s("dtype", "bf16");
    rad::Resolved ref_add = bind_row(reg, add, ag);

    Side cand_side, ref_side;
    cand_side.upload(s, sc);
    ref_side.upload(s, sc);

    std::vector<RadTensor> cand_t = cand_side.bind(sc.cand);
    std::vector<RadTensor> rt = ref_side.bind(sc.ref);
    std::vector<RadTensor> conv_t(rt.begin(), rt.begin() + 4);
    std::vector<RadTensor> add_t(rt.begin() + 4, rt.end());

    launch(cand, cand_t, s.stream, "bench candidate dflash_conv_add");
    launch(ref_conv, conv_t, s.stream, "bench reference dflash_conv");
    launch(ref_add, add_t, s.stream, "bench reference add");
    check(rad_stream_sync(s.stream), "bench conv parity completion");

    char id[128];
    std::snprintf(id, sizeof id, "convadd/%s", c.name);
    const size_t bytes = compare_side(s, sc, cand_side, ref_side, id);
    std::printf("CASE_PASS %s\n", id);
    std::printf("BENCH_CASE %s op=dflash_conv_add cand_plugin=mine "
                "cand_kernel=mine_dflash_conv_add_t2_g16_bf16 "
                "ref_plugin=libr4d ref_chain=dflash_conv_t2_g16_bf16+add_bf16 "
                "T=%lld hidden_size=%lld NG=%lld block_size=%lld taps=2 group_size=16\n",
                id, (long long)c.band, (long long)c.H, (long long)(c.H / 16),
                (long long)c.block);
    std::printf("BENCH_NOTE %s parity=byte-exact buffers=%zu synthetic-bf16-weights "
                "host-binding=per-launch hot-buffers=1 per-rank=1 sync=per-batch\n",
                id, sc.compare.size());

    const std::vector<int> written = { sc.cand[3].buf, sc.cand[4].buf };
    auto reset = [&](Side& side) { reload(s, sc, written, side); };

    auto cand_body = [&] { launch(cand, cand_t, s.stream, "bench candidate dflash_conv_add"); };
    auto ref_body = [&] {
        launch(ref_conv, conv_t, s.stream, "bench reference dflash_conv");
        launch(ref_add, add_t, s.stream, "bench reference add");
    };

    reset(cand_side);
    reset(ref_side);
    const int iters_c = calibrate(s, cand_body, bc, "bench candidate convadd");
    const int iters_r = calibrate(s, ref_body, bc, "bench reference convadd");
    paired_rounds(id, bc, iters_c, iters_r,
                  [&] { reset(cand_side); return timed_batch(s, cand_body, iters_c, "candidate"); },
                  [&] { reset(ref_side);  return timed_batch(s, ref_body, iters_r, "reference"); });
    return bytes;
}

size_t bench_kstore(const Session& s, rad::Registry& reg, const rad::KernelRow* rms,
                    const rad::KernelRow* rope, const rad::KernelRow* store, const KCase& c,
                    const BenchConfig& bc) {
    Scene sc = kstore_scene(c);
    rad::Resolved cand = resolve_candidate(reg, "dflash_k_norm_rope_store",
                                           "mine_dflash_k_norm_rope_store_hd128",
                                           kstore_geometry(c.rows, c.nkv, 128, c.block), 6);
    check(reg.init_instance(cand, 0, 1), "init candidate dflash_k_norm_rope_store");

    rad::Geometry ng;
    ng.set_i("M", c.rows * c.nkv);
    ng.set_i("n", 128);
    ng.set_f("eps", 1e-6);
    ng.set_f("wadd", 0.0);
    ng.set_s("dtype", "bf16");
    rad::Resolved ref_norm = bind_row(reg, rms, ng);

    rad::Geometry rg;
    rg.set_i("M", c.rows);
    rg.set_i("head_dim", 128);
    rg.set_i("n_head", c.nkv);
    rg.set_i("n_head_kv", 0);
    rg.set_f("theta", 10000.0);
    rg.set_f("scale", 1.0);
    rg.set_s("mode", "neox");
    rad::Resolved ref_rope = bind_row(reg, rope, rg);

    rad::Geometry kg;
    kg.set_i("M", c.rows);
    kg.set_i("head_dim", 128);
    kg.set_i("n_head_kv", c.nkv);
    kg.set_i("block_size", c.block);
    kg.set_s("kv_dtype", "bf16");
    rad::Resolved ref_store = bind_row(reg, store, kg);

    Side cand_side, ref_side;
    cand_side.upload(s, sc);
    ref_side.upload(s, sc);

    std::vector<RadTensor> cand_t = cand_side.bind(sc.cand);
    std::vector<RadTensor> rt = ref_side.bind(sc.ref);
    std::vector<RadTensor> norm_t = { rt[0], rt[2], rt[0] };
    std::vector<RadTensor> rope_t = { rt[0], rt[3] };
    std::vector<RadTensor> store_t = { rt[0], rt[1], rt[4], rt[5] };

    launch(cand, cand_t, s.stream, "bench candidate dflash_k_norm_rope_store");
    launch(ref_norm, norm_t, s.stream, "bench reference rmsnorm");
    launch(ref_rope, rope_t, s.stream, "bench reference rope");
    launch(ref_store, store_t, s.stream, "bench reference kv_store");
    check(rad_stream_sync(s.stream), "bench kstore parity completion");

    char id[128];
    std::snprintf(id, sizeof id, "kstore/%s", c.name);
    const size_t bytes = compare_side(s, sc, cand_side, ref_side, id);
    std::printf("CASE_PASS %s\n", id);
    std::printf("BENCH_CASE %s op=dflash_k_norm_rope_store cand_plugin=mine "
                "cand_kernel=mine_dflash_k_norm_rope_store_hd128 "
                "ref_plugin=libr4d ref_chain=rmsnorm_bf16+rope_bf16+kv_store_bf16 "
                "M=%lld n_head_kv=%lld head_dim=128 block_size=%lld blocks=%lld pitch=%lld\n",
                id, (long long)c.rows, (long long)c.nkv, (long long)c.block,
                (long long)c.blocks, (long long)c.kpitch);
    std::printf("BENCH_NOTE %s parity=byte-exact buffers=%zu synthetic-bf16-weights "
                "host-binding=per-launch hot-buffers=1 per-rank=1 sync=per-batch\n",
                id, sc.compare.size());

    const std::vector<int> written = { sc.cand[0].buf, sc.cand[5].buf };
    auto reset = [&](Side& side) { reload(s, sc, written, side); };

    auto cand_body = [&] {
        launch(cand, cand_t, s.stream, "bench candidate dflash_k_norm_rope_store");
    };
    auto ref_body = [&] {
        launch(ref_norm, norm_t, s.stream, "bench reference rmsnorm");
        launch(ref_rope, rope_t, s.stream, "bench reference rope");
        launch(ref_store, store_t, s.stream, "bench reference kv_store");
    };

    reset(cand_side);
    reset(ref_side);
    const int iters_c = calibrate(s, cand_body, bc, "bench candidate kstore");
    const int iters_r = calibrate(s, ref_body, bc, "bench reference kstore");
    paired_rounds(id, bc, iters_c, iters_r,
                  [&] { reset(cand_side); return timed_batch(s, cand_body, iters_c, "candidate"); },
                  [&] { reset(ref_side);  return timed_batch(s, ref_body, iters_r, "reference"); });
    return bytes;
}

size_t bench_gate(const Session& s, rad::Registry& reg, const rad::KernelRow* gemm,
                  const rad::KernelRow* silu, const GateCase& c, const BenchConfig& bc) {
    Scene sc = gate_scene(c);
    require(c.N >= 2048 && c.N % 32 == 0 && c.K % 16 == 0,
            "gate case outside the candidate's declared band");
    rad::Resolved cand = resolve_candidate(reg, "dflash_gated_bf16", "mine_dflash_gated_bf16",
                                           gate_geometry(c.M, c.N, c.K), 3);
    check(reg.init_instance(cand, 0, 1), "init candidate dflash_gated_bf16");

    rad::Geometry rg = gate_geometry(c.M, c.N, c.K);
    rg.set_i("sk", 1);
    rg.set_i("wv", 1);
    rad::Resolved ref_gemm = bind_row(reg, gemm, rg);
    rad::Geometry sg;
    sg.set_i("M", c.M);
    sg.set_i("n", c.N / 2);
    sg.set_s("dtype", "bf16");
    rad::Resolved ref_silu = bind_row(reg, silu, sg);

    Side cand_side, ref_side;
    cand_side.upload(s, sc);
    ref_side.upload(s, sc);

    std::vector<RadTensor> cand_t = cand_side.bind(sc.cand);
    std::vector<RadTensor> rt = ref_side.bind(sc.ref);
    std::vector<RadTensor> gemm_t(rt.begin(), rt.begin() + 3);
    std::vector<RadTensor> silu_t(rt.begin() + 3, rt.end());

    launch(cand, cand_t, s.stream, "bench candidate dflash_gated_bf16");
    launch(ref_gemm, gemm_t, s.stream, "bench reference gemm_nt");
    launch(ref_silu, silu_t, s.stream, "bench reference silu_mul");
    check(rad_stream_sync(s.stream), "bench gate parity completion");

    char id[128];
    std::snprintf(id, sizeof id, "gate/%s", c.name);
    const size_t bytes = compare_side(s, sc, cand_side, ref_side, id);
    std::printf("CASE_PASS %s\n", id);
    std::printf("BENCH_CASE %s op=dflash_gated_bf16 cand_plugin=mine "
                "cand_kernel=mine_dflash_gated_bf16 "
                "ref_plugin=libr4d ref_chain=gemm_bf16_nt_m64+silu_mul_bf16 "
                "M=%lld N=%lld K=%lld\n",
                id, (long long)c.M, (long long)c.N, (long long)c.K);
    std::printf("BENCH_NOTE %s parity=byte-exact buffers=%zu synthetic-bf16-weights "
                "host-binding=per-launch hot-buffers=1 per-rank=1 sync=per-batch\n",
                id, sc.compare.size());

    const std::vector<int> written = { sc.cand[2].buf };
    auto reset = [&](Side& side) { reload(s, sc, written, side); };

    auto cand_body = [&] { launch(cand, cand_t, s.stream, "bench candidate dflash_gated_bf16"); };
    auto ref_body = [&] {
        launch(ref_gemm, gemm_t, s.stream, "bench reference gemm_nt");
        launch(ref_silu, silu_t, s.stream, "bench reference silu_mul");
    };

    reset(cand_side);
    reset(ref_side);
    const int iters_c = calibrate(s, cand_body, bc, "bench candidate gate");
    const int iters_r = calibrate(s, ref_body, bc, "bench reference gate");
    paired_rounds(id, bc, iters_c, iters_r,
                  [&] { reset(cand_side); return timed_batch(s, cand_body, iters_c, "candidate"); },
                  [&] { reset(ref_side);  return timed_batch(s, ref_body, iters_r, "reference"); });
    return bytes;
}

/* THE STREAMING SCENE. The hot-buffer case above re-reads ONE 39 MB gate matrix for every launch,
 * so it can sit in cache after the first pass and report a best case no production step sees: the
 * block has five different gate matrices (one per gate/up pair). This case rotates five independent
 * weight buffers per arm, so consecutive launches never re-read the previous launch's image and the
 * 196 MB working set cannot be served from a cache smaller than that.
 *
 * IT IS A CLOSER SCENE, NOT A GUARANTEED DRAM READ. Rotation removes reuse between consecutive
 * launches; whether a given matrix survives in cache across a round is the device's business. The
 * line reports `hot-buffers=0 weight-rotation=5` so the reader can tell the two scenes apart.
 *
 * EVERYTHING ELSE IS THE HOT CASE'S CONTRACT. Activation, weight and output geometry are the
 * production M4, N7680, K2560 shape; both arms see the same host image; all five outputs are
 * byte-compared, in both arms, and every weight buffer is sentinel-checked as unwritten; the same
 * calibration, alternating paired rounds and one sync per batch are used. The only difference is
 * that the rotation index advances inside the measured loop.
 *
 * NO ALLOCATION IS CHARGED TO THE LOOP. Both arms' operand lists are bound ONCE, outside every
 * timed batch, into one vector per (stream, arm); the body only selects one.
 *
 * COST. Five 39.3 MB weight images per arm, so ~393 MB of device memory for both arms plus one
 * 196 MB host image set, and all five arithmetic paths are byte-compared. That is the scene's
 * price, and it is why it runs once rather than per case. */
size_t bench_gate_streaming(const Session& s, rad::Registry& reg, const rad::KernelRow* gemm,
                            const rad::KernelRow* silu, const GateCase& c, int streams,
                            const BenchConfig& bc) {
    require(streams >= 2, "a streaming gate case needs at least two weight buffers");
    require(c.N >= 2048 && c.N % 32 == 0 && c.K % 16 == 0,
            "gate case outside the candidate's declared band");
    Scene sc = gate_scene(c, streams);
    rad::Resolved cand = resolve_candidate(reg, "dflash_gated_bf16", "mine_dflash_gated_bf16",
                                           gate_geometry(c.M, c.N, c.K), 3);
    check(reg.init_instance(cand, 0, 1), "init candidate dflash_gated_bf16");

    rad::Geometry rg = gate_geometry(c.M, c.N, c.K);
    rg.set_i("sk", 1);
    rg.set_i("wv", 1);
    rad::Resolved ref_gemm = bind_row(reg, gemm, rg);
    rad::Geometry sg;
    sg.set_i("M", c.M);
    sg.set_i("n", c.N / 2);
    sg.set_s("dtype", "bf16");
    rad::Resolved ref_silu = bind_row(reg, silu, sg);

    Side cand_side, ref_side;
    cand_side.upload(s, sc);
    ref_side.upload(s, sc);

    std::vector<std::vector<RadTensor>> cand_t, gemm_t, silu_t;
    cand_t.reserve(size_t(streams));
    gemm_t.reserve(size_t(streams));
    silu_t.reserve(size_t(streams));
    for (int i = 0; i < streams; ++i) {
        cand_t.push_back(cand_side.bind(sc.cand_ops[size_t(i)]));
        std::vector<RadTensor> rt = ref_side.bind(sc.ref_ops[size_t(i)]);
        gemm_t.emplace_back(rt.begin(), rt.begin() + 3);
        silu_t.emplace_back(rt.begin() + 3, rt.end());
    }

    /* PARITY ON EVERY STREAM, not just the first: each weight image has its own bytes, so each is
     * its own arithmetic. */
    for (int i = 0; i < streams; ++i) {
        launch(cand, cand_t[size_t(i)], s.stream, "bench candidate dflash_gated_bf16");
        launch(ref_gemm, gemm_t[size_t(i)], s.stream, "bench reference gemm_nt");
        launch(ref_silu, silu_t[size_t(i)], s.stream, "bench reference silu_mul");
    }
    check(rad_stream_sync(s.stream), "bench gate streaming parity completion");

    char id[160];
    std::snprintf(id, sizeof id, "gate/%s-stream%d", c.name, streams);
    const size_t bytes = compare_side(s, sc, cand_side, ref_side, id);
    std::printf("CASE_PASS %s\n", id);
    std::printf("BENCH_CASE %s op=dflash_gated_bf16 cand_plugin=mine "
                "cand_kernel=mine_dflash_gated_bf16 "
                "ref_plugin=libr4d ref_chain=gemm_bf16_nt_m64+silu_mul_bf16 "
                "M=%lld N=%lld K=%lld weight_buffers=%d weight_bytes_per_buffer=%lld\n",
                id, (long long)c.M, (long long)c.N, (long long)c.K, streams,
                (long long)(c.N * c.K * 2));
    std::printf("BENCH_NOTE %s parity=byte-exact buffers=%zu synthetic-bf16-weights "
                "host-binding=per-launch hot-buffers=0 weight-rotation=%d per-rank=1 "
                "sync=per-batch\n",
                id, sc.compare.size(), streams);

    /* Only the outputs are restored; the weights are read-only and the sentinel check above has
     * already established that neither arm wrote them. The rotation restarts at stream 0 for every
     * batch, so the two arms walk the five images in the same order even though their calibrated
     * iteration counts differ. */
    std::vector<int> written;
    written.reserve(size_t(streams));
    for (int i = 0; i < streams; ++i) written.push_back(sc.cand_ops[size_t(i)][2].buf);

    int next_cand = 0, next_ref = 0;
    auto reset = [&](Side& side) {
        next_cand = next_ref = 0;
        reload(s, sc, written, side);
    };

    auto cand_body = [&] {
        const int i = next_cand;
        next_cand = (next_cand + 1) % streams;
        launch(cand, cand_t[size_t(i)], s.stream, "bench candidate dflash_gated_bf16");
    };
    auto ref_body = [&] {
        const int i = next_ref;
        next_ref = (next_ref + 1) % streams;
        launch(ref_gemm, gemm_t[size_t(i)], s.stream, "bench reference gemm_nt");
        launch(ref_silu, silu_t[size_t(i)], s.stream, "bench reference silu_mul");
    };

    reset(cand_side);
    reset(ref_side);
    const int iters_c = calibrate(s, cand_body, bc, "bench candidate gate streaming");
    const int iters_r = calibrate(s, ref_body, bc, "bench reference gate streaming");
    paired_rounds(id, bc, iters_c, iters_r,
                  [&] { reset(cand_side); return timed_batch(s, cand_body, iters_c, "candidate"); },
                  [&] { reset(ref_side);  return timed_batch(s, ref_body, iters_r, "reference"); });
    return bytes;
}

/* Model-sized geometries. hidden_size 2560, intermediate 7680, 8 KV heads of 128 and KV block 16
 * are Qwen3.8-Flash-Next-DFlash2's own (config.json). Both widths here are PER RANK after a TP2
 * split: the gate's N 7680 is 2*3840 (the fused gate and up halves of a 3840-wide local
 * intermediate), and KStore runs 4 local KV heads of the model's 8. The convolution is timed at
 * block 4 with one sequence (4 rows) and eight sequences (32 rows). */
const ConvCase kBenchConvCases[] = {
    { "model-C1-M4-half0",    4, 2560, 4,  4, 0, true },
    { "model-C1-M4-half1",    4, 2560, 4,  4, 1, true },
    { "model-C8-M32-half0",  32, 2560, 4, 32, 0, true },
    { "model-C8-M32-half1",  32, 2560, 4, 32, 1, true },
};

/* One context chunk with 2048 distinct slots in a 128-block cache. */
KCase kstore_long_case() {
    KCase c{ "model-prefill-M2048", 2048, 4, 16, 128, 512, {}, {},
             ValueMode::kNormal, false };
    c.pos.reserve(size_t(c.rows));
    c.slot.reserve(size_t(c.rows));
    for (int64_t t = 0; t < c.rows; ++t) {
        c.pos.push_back(int32_t(1024 + t));
        c.slot.push_back(int32_t(t));
    }
    return c;
}

/* Slots stay inside the 4-block, 16-token cache (capacity 64) and are distinct; positions are long
 * and increasing, which is what the rotation sees during decode. */
const KCase kBenchKStoreCases[] = {
    { "model-C1-M4",  4, 4, 16, 4, 512,
      { 1024, 1041, 1058, 1075 }, { 0, 1, 2, 3 }, ValueMode::kNormal, false },
    { "model-C8-M32", 32, 4, 16, 4, 512,
      { 1024, 1041, 1058, 1075, 1092, 1109, 1126, 1143,
        1160, 1177, 1194, 1211, 1228, 1245, 1262, 1279,
        1296, 1313, 1330, 1347, 1364, 1381, 1398, 1415,
        1432, 1449, 1466, 1483, 1500, 1517, 1534, 1551 },
      { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 },
      ValueMode::kNormal, false },
};


const GateCase kBenchGateCases[] = {
    { "model-C1-M4",  4, 7680, 2560 },
    { "model-C8-M32", 32, 7680, 2560 },
};

/* The block has five gate/up matrices per rank; the streaming scene rotates all five so no
 * consecutive launch re-reads a cached weight image. */
constexpr int kGateWeightStreams = 5;

/* ---------------------------------------------------------------------------------- options */

struct Options {
    std::string mine, ref;
    bool help = false;
    bool cases_given = false;
    bool convadd = false, kstore = false, bgate = false;
    bool benchmark = false;
    BenchConfig bench;
};

void usage(FILE* out) {
    std::fprintf(out,
        "usage: dflash2_fusion_gpu_test --mine PATH --ref PATH [--cases LIST] [--benchmark]\n"
        "\n"
        "Device-against-device, byte-exact parity oracle for the DFlash2 local fusions.\n"
        "The candidate row is loaded from plugin 'mine'; every unfused reference step is pinned\n"
        "by kernel name inside plugin 'libr4d'. No tolerance, no fallback, no skip-as-pass.\n"
        "\n"
        "  --mine PATH    the mine plugin .so (candidate rows)\n"
        "  --ref PATH     the libr4d plugin .so (unfused reference rows)\n"
        "  --cases LIST   comma-separated: convadd, kstore, bgate, all.\n"
        "                 Default convadd,kstore; with --benchmark, all.\n"
        "  --benchmark    time the model-sized cases instead of the general parity sweep.\n"
        "                 Each case still runs its byte comparison first; a mismatch aborts\n"
        "                 before any clock is read. Timings are host submission + queue\n"
        "                 execution + final sync, per rank, on synthetic inputs with hot\n"
        "                 buffers -- not engine throughput, and not a global parity claim.\n"
        "                 The gate additionally runs gate/model-C1-M4-stream5, which rotates\n"
        "                 five 39 MB weight buffers per arm (hot-buffers=0) because one matrix\n"
        "                 re-read per launch is a cache best case the block never sees.\n"
        "  --rounds N     paired benchmark rounds (default 7; order alternates every round)\n"
        "  --help         this text; answered without touching a device\n"
        "\n"
        "Exit status is 0 only after every requested candidate ran at least one case and every\n"
        "compared byte matched.\n");
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char* flag) -> std::string {
            require(i + 1 < argc, std::string(flag) + " needs a value");
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { o.help = true; continue; }
        if (a == "--mine") { o.mine = value("--mine"); continue; }
        if (a == "--ref") { o.ref = value("--ref"); continue; }
        if (a == "--cases") {
            o.cases_given = true;
            const std::string list = value("--cases");
            size_t start = 0;
            while (start <= list.size()) {
                const size_t comma = list.find(',', start);
                const std::string item = list.substr(start, comma == std::string::npos
                                                              ? std::string::npos
                                                              : comma - start);
                if (item == "convadd") o.convadd = true;
                else if (item == "kstore") o.kstore = true;
                else if (item == "bgate") o.bgate = true;
                else if (item == "all") { o.convadd = o.kstore = o.bgate = true; }
                else if (!item.empty())
                    throw std::runtime_error("unknown --cases item '" + item + "'");
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            continue;
        }
        if (a == "--benchmark") { o.benchmark = true; continue; }
        if (a == "--rounds") {
            const std::string v = value("--rounds");
            o.bench.rounds = std::stoi(v);
            require(o.bench.rounds >= 3, "--rounds must be at least 3");
            continue;
        }
        throw std::runtime_error("unknown argument '" + a + "'");
    }
    if (o.help) return o;
    require(!o.mine.empty() && !o.ref.empty(), "--mine and --ref are required (try --help)");
    /* An explicitly empty selection is a caller error, not an implicit default: "no cases ran"
     * must never be reachable through a typo. */
    if (o.cases_given) {
        require(o.convadd || o.kstore || o.bgate,
                "--cases selected no candidate (use convadd, kstore, bgate or all)");
    } else if (o.benchmark) {
        o.convadd = o.kstore = o.bgate = true;
    } else {
        o.convadd = true;
        o.kstore = true;
    }
    return o;
}

/* -------------------------------------------------------------------------------- case lists */

const ConvCase kConvCases[] = {
    /* name, rows, H, block, T band, delta half, in place */
    { "packed-block8",         16, 128, 8, 16, 0, true  },
    { "packed-block4",         16, 128, 4, 16, 1, true  },
    { "packed-block7-band",    21, 128, 7, 24, 1, true  },
    { "band-ceiling",          20, 128, 8, 24, 0, true  },
    { "out-of-place",          14, 128, 7, 14, 1, false },
    { "multi-sequence-phases", 24, 128, 8, 24, 1, true  },
    { "wide-256",              16, 256, 8, 16, 1, true  },
    { "narrow-64",              7,  64, 7,  7, 0, true  },
};

const KCase kKStoreCases[] = {
    { "block-slot-boundary", 6, 1, 16, 3, 128,
      { 0, 1, 15, 16, 2047, 2048 }, { 0, 1, 15, 16, -1, 48 }, ValueMode::kNormal, false },
    { "two-heads-cross-block", 5, 2, 8, 4, 256,
      { 100, 2047, 4096, 131072, 7 }, { 0, 7, 8, 31, 32 }, ValueMode::kNormal, false },
    { "all-skipped", 3, 1, 16, 2, 128,
      { 4096, 8191, 8192 }, { -1, 32, 33 }, ValueMode::kNormal, false },
    { "long-positions-two-heads", 8, 2, 16, 2, 256,
      { 4096, 8191, 8192, 16384, 32768, 65535, 65536, 131071 },
      { 0, 15, 16, 31, 1, 14, 17, -5 }, ValueMode::kNormal, false },

    /* ---- exceptional values. These exist because the previous BF16 helper in this tree truncated
     * instead of rounding, and because the norm's `rsqrtf(ss/n + eps)` has regimes ordinary noise
     * never reaches. Every one of them must agree BYTE FOR BYTE, NaN payloads included. */
    { "exceptional-payloads", 4, 2, 8, 4, 256,
      { 0, 1, 2047, 2048 }, { 0, 7, 8, 15 }, ValueMode::kExceptional, false },
    { "exceptional-gain", 4, 2, 8, 4, 256,
      { 0, 1, 2047, 2048 }, { 0, 7, 8, 15 }, ValueMode::kExceptional, true },
    { "norm-overflow-inf", 4, 2, 8, 4, 256,
      { 0, 3, 2047, 2048 }, { 0, 7, 8, 15 }, ValueMode::kOverflow, false },
    { "norm-underflow-zero", 4, 2, 8, 4, 256,
      { 0, 3, 2047, 2048 }, { 0, 7, 8, 15 }, ValueMode::kUnderflow, false },

    /* ---- eight kv heads: the widest real geometry here. It crosses the reduction's four-wave
     * boundary (128 columns = four waves of 32 lanes) at every head and makes the `[.., h*head_dim]`
     * head walk in the store observable. */
    { "eight-heads", 6, 8, 16, 3, 1024,
      { 0, 1, 15, 16, 2047, 2048 }, { 0, 15, 16, 31, -1, 47 },
      ValueMode::kNormal, false },
    { "eight-heads-exceptional", 5, 8, 16, 3, 1024,
      { 1, 2047, 4096, 131072, 7 }, { 1, 16, 31, 32, 47 },
      ValueMode::kExceptional, true },
};

/* Gate cases. N is the gate_up width (2 * n_ff) and the candidate requires N >= 2048, N % 32 == 0
 * and K % 16 == 0. The drafter's real N is far wider; these are the smallest legal geometries that
 * still exercise the multi-tile, multi-slice and multi-row paths. Both the candidate and the
 * libr4d GEMM require contiguous operands, so no pitched activation appears here. */
const GateCase kGateCases[] = {
    /* name, M, N, K */
    { "single-row",         1, 2048, 256 },
    { "row-tile-16",       16, 2048, 256 },
    { "row-band-64",       64, 2048, 512 },
    { "wide-gate",          8, 4096, 256 },
    { "wide-gate-deep-k",   4, 2048, 1024 },
    { "two-sequences",     32, 2048, 256 },
};

}  /* namespace */

int main(int argc, char** argv) {
    try {
        Options opt = parse(argc, argv);
        if (opt.help) { usage(stdout); return 0; }

        Session session;
        rad::Registry registry;
        registry.set_code_device(0);
        check(registry.load_plugin(opt.mine, 0), "load mine plugin");
        check(registry.load_plugin(opt.ref, 1), "load libr4d reference plugin");
        check(registry.check_complete(), "complete plugin schemas");

        const rad::Plugin* mine = registry.plugin("mine");
        const rad::Plugin* ref = registry.plugin("libr4d");
        require(mine != nullptr, "plugin '" + opt.mine + "' did not identify itself as 'mine'");
        require(ref != nullptr, "plugin '" + opt.ref + "' did not identify itself as 'libr4d'");
        std::printf("PLUGIN role=candidate name=mine file=%s\n", mine->file.c_str());
        std::printf("PLUGIN role=reference name=libr4d file=%s\n", ref->file.c_str());

        size_t cases = 0, bytes = 0, refusals = 0;
        int candidates = 0;

        /* BENCHMARK MODE REPLACES THE PARITY SWEEP, it does not follow it. The general lists include
         * geometries whose parity is not settled (kstore's exceptional gain); letting them run first
         * would abort the process before a single model-sized case was timed. In this mode each
         * benchmark case carries its own byte comparison, and a mismatch ends the run. */
        if (opt.benchmark) {
            if (opt.convadd) {
                const rad::KernelRow* conv = pin_row(registry, "dflash_conv",
                                                     "dflash_conv_t2_g16_bf16", "libr4d");
                const rad::KernelRow* add = pin_row(registry, "add", "add_bf16", "libr4d");
                for (const ConvCase& c : kBenchConvCases)
                    bytes += bench_conv(session, registry, conv, add, c, opt.bench);
                std::printf("BENCH_DONE op=dflash_conv_add cases=%zu\n",
                            sizeof(kBenchConvCases) / sizeof(kBenchConvCases[0]));
                cases += sizeof(kBenchConvCases) / sizeof(kBenchConvCases[0]);
            }
            if (opt.kstore) {
                const rad::KernelRow* rms = pin_row(registry, "rmsnorm", "rmsnorm_bf16", "libr4d");
                const rad::KernelRow* rope = pin_row(registry, "rope", "rope_bf16", "libr4d");
                const rad::KernelRow* store = pin_row(registry, "kv_store", "kv_store_bf16",
                                                      "libr4d");
                for (const KCase& c : kBenchKStoreCases)
                    bytes += bench_kstore(session, registry, rms, rope, store, c, opt.bench);
                bytes += bench_kstore(session, registry, rms, rope, store,
                                      kstore_long_case(), opt.bench);
                const size_t n = sizeof(kBenchKStoreCases) / sizeof(kBenchKStoreCases[0]) + 1;
                std::printf("BENCH_DONE op=dflash_k_norm_rope_store cases=%zu\n", n);
                cases += n;
            }
            if (opt.bgate) {
                const rad::KernelRow* gemm = pin_row(registry, "gemm_nt", "gemm_bf16_nt_m64",
                                                     "libr4d");
                const rad::KernelRow* silu = pin_row(registry, "silu_mul", "silu_mul_bf16",
                                                     "libr4d");
                for (const GateCase& c : kBenchGateCases)
                    bytes += bench_gate(session, registry, gemm, silu, c, opt.bench);
                /* THE STREAMING SCENE, on the production decode shape, rotating the block's five
                 * gate matrices. It is part of `--benchmark` rather than a flag of its own: the
                 * hot-buffer number above is only interpretable next to it. */
                bytes += bench_gate_streaming(session, registry, gemm, silu, kBenchGateCases[0],
                                              kGateWeightStreams, opt.bench);
                std::printf("BENCH_DONE op=dflash_gated_bf16 cases=%zu\n",
                            sizeof(kBenchGateCases) / sizeof(kBenchGateCases[0]) + 1);
                cases += sizeof(kBenchGateCases) / sizeof(kBenchGateCases[0]) + 1;
            }
            require(cases > 0, "benchmark selected but executed zero cases");
            std::printf("DFLASH2_FUSION_BENCH_PASS cases=%zu bytes=%zu rounds=%d "
                        "note=finite-cases-only-not-global-parity\n",
                        cases, bytes, opt.bench.rounds);
            return 0;
        }

        if (opt.convadd) {
            const rad::KernelRow* conv = pin_row(registry, "dflash_conv",
                                                 "dflash_conv_t2_g16_bf16", "libr4d");
            const rad::KernelRow* add = pin_row(registry, "add", "add_bf16", "libr4d");
            size_t n = 0;
            for (const ConvCase& c : kConvCases) { bytes += run_conv_case(session, registry, conv, add, c); ++n; }
            require(n > 0, "convadd selected but executed zero cases");
            const size_t refused = run_conv_refusals(registry);
            refusals += refused;
            std::printf("CANDIDATE_DONE op=dflash_conv_add plugin=mine "
                        "kernel=mine_dflash_conv_add_t2_g16_bf16 cases=%zu refusals=%zu\n",
                        n, refused);
            cases += n;
            ++candidates;
        }

        if (opt.kstore) {
            const rad::KernelRow* rms = pin_row(registry, "rmsnorm", "rmsnorm_bf16", "libr4d");
            const rad::KernelRow* rope = pin_row(registry, "rope", "rope_bf16", "libr4d");
            const rad::KernelRow* store = pin_row(registry, "kv_store", "kv_store_bf16", "libr4d");
            size_t n = 0;
            for (const KCase& c : kKStoreCases) { bytes += run_kstore_case(session, registry, rms, rope, store, c); ++n; }
            require(n > 0, "kstore selected but executed zero cases");
            const size_t refused = run_kstore_refusals(registry);
            refusals += refused;
            std::printf("CANDIDATE_DONE op=dflash_k_norm_rope_store plugin=mine "
                        "kernel=mine_dflash_k_norm_rope_store_hd128 cases=%zu refusals=%zu\n",
                        n, refused);
            cases += n;
            ++candidates;
        }

        if (opt.bgate) {
            /* `gemm_bf16_nt_m64` is asserted for every M. libr4d's selector would serve
             * `gemm_bf16_nt_m16` for M <= 16, and that row forwards to m64
             * (r4d_gemm_bf16_nt_m16.hip:19-21), so the accumulation and the bytes are the same;
             * pinning m64 directly means one reference row covers the whole band and that
             * forwarding claim is not taken on trust. */
            const rad::KernelRow* gemm = pin_row(registry, "gemm_nt", "gemm_bf16_nt_m64", "libr4d");
            const rad::KernelRow* silu = pin_row(registry, "silu_mul", "silu_mul_bf16", "libr4d");
            size_t n = 0;
            for (const GateCase& c : kGateCases) { bytes += run_gate_case(session, registry, gemm, silu, c); ++n; }
            require(n > 0, "bgate selected but executed zero cases");
            const size_t refused = run_gate_refusals(registry);
            refusals += refused;
            std::printf("CANDIDATE_DONE op=dflash_gated_bf16 plugin=mine "
                        "kernel=mine_dflash_gated_bf16 cases=%zu refusals=%zu\n", n, refused);
            cases += n;
            ++candidates;
        }

        require(candidates > 0, "no candidate was selected");
        require(cases > 0, "zero parity cases executed");
        std::printf("DFLASH2_FUSION_PARITY_PASS candidates=%d cases=%zu bytes=%zu refusals=%zu\n",
                    candidates, cases, bytes, refusals);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "DFLASH2_FUSION_PARITY_FAIL: %s\n", e.what());
        return 1;
    }
}
