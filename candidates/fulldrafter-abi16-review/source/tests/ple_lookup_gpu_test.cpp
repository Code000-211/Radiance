/* PLE n-gram lookup regression: the specialised 160-wide walk must return each id's own literal
 * BF16 row, every launch, with no remembered state between launches.
 *
 * Three scenarios over the same row of ops:
 *   collisions -- 32 pairs of distinct ids a stride apart (the id pattern a set-indexed L1 used to
 *                 collide on), each launch repeated, so a lookup that remembered anything by id
 *                 alone would answer the second launch from the first one's payload;
 *   identity   -- the same LOCAL ids through two immutable tables, two vocab shards and three
 *                 scales, launched in turn, so a lookup keyed on `tok - vocab_off` alone returns
 *                 another table's, another shard's or another scale's row;
 *   boundary   -- padding ids, an id at n_vocab, an id past it and the last in-range id.
 *
 * Every expectation is computed here from the literal E4M3 code and the literal scale
 * (`expected_cell`), never from the kernel, so this is an oracle comparison and not a self-check.
 *
 * Usage: ple_lookup_gpu_test [libr4d.so] [rounds], default 1000, requires a free GPU.
 * Exit 0 = exact outputs; 2 = mismatch; 1 = setup/launch failure. */
#include "core/device/device.h"
#include "core/plugin/registry.h"
#include "abi/rad_plugin.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

void check(int status, const char* what) {
    if (status == RAD_OK) return;
    const char* detail = rad_dev_last_error();
    fail(std::string(what) + ": " + rad_strerror(status) +
         (detail ? std::string("; ") + detail : std::string()));
}

constexpr int64_t kWidth  = 160;      /* the width the specialised kernel requires */
constexpr int64_t kM      = 64;       /* its ceiling */
constexpr int64_t kPairs  = kM / 2;
constexpr int64_t kVocab  = 100000;
constexpr int64_t kStride = 65536;    /* the pair distance the removed set index collided on */

/* Exactly representable E4M3 codes and their BF16 images under a scale of 1.0. No NaN or Inf
 * encoding is present, and every value survives f32 -> BF16 without rounding. */
constexpr uint8_t kPaletteCode[16] = {
    0x00, 0x20, 0x22, 0x24, 0x28, 0x2A, 0x2C, 0x30,
    0x34, 0x38, 0x3A, 0x3C, 0x40, 0x42, 0x44, 0x48,
};
constexpr uint16_t kPaletteBf16[16] = {
    0x0000, 0x3E00, 0x3E20, 0x3E40, 0x3E80, 0x3EA0, 0x3EC0, 0x3F00,
    0x3F40, 0x3F80, 0x3FA0, 0x3FC0, 0x4000, 0x4020, 0x4040, 0x4080,
};
constexpr uint16_t kScaleOne = 0x3F80;

/* Encode both nibbles of id % 251 so distinct ids have distinct rows. */
int palette_index(int64_t id, int64_t j) {
    const int64_t residue = id % 251;
    const int64_t rot = j < 144 ? residue % 16 : residue / 16;
    return int((j + rot) % 16);
}

uint16_t row_cell(int64_t id, int64_t j) { return kPaletteBf16[palette_index(id, j)]; }

/* The host's own reading of a literal E4M3 code and a literal scale: the palette entry is the
 * code's exact value, so the expected image is that value times the scale, narrowed to BF16.
 * Every palette/scale pair used here is a power of two, so the product must narrow bit-exactly;
 * a non-exact fixture is a broken test, not a kernel mismatch. */
uint16_t expected_cell(uint16_t palette_bf16, uint16_t scale_bf16) {
    float v = 0.0f, s = 0.0f;
    const uint32_t pb = uint32_t(palette_bf16) << 16, sb = uint32_t(scale_bf16) << 16;
    std::memcpy(&v, &pb, 4);
    std::memcpy(&s, &sb, 4);
    const float p = v * s;
    uint32_t u = 0;
    std::memcpy(&u, &p, 4);
    if ((u & 0xFFFFu) || ((u >> 16) & 0x7F80u) == 0x7F80u)
        fail("fixture palette * scale is not an exact finite BF16 value");
    return uint16_t(u >> 16);
}

struct DeviceBuffer {
    void* data = nullptr;
    explicit DeviceBuffer(size_t bytes) {
        if (bytes) data = rad_dev_alloc(int64_t(bytes), RAD_MEM_DEVICE);
        if (bytes && !data) fail("device allocation failed");
    }
    ~DeviceBuffer() { if (data) rad_dev_free(data, RAD_MEM_DEVICE); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

const rad::KernelRow* pin_row(rad::Registry& reg) {
    const auto* rows = reg.rows_for("embed_lookup_q");
    if (!rows || rows->empty()) fail("no row for op 'embed_lookup_q'");
    const rad::KernelRow* hit = nullptr;
    for (const rad::KernelRow* r : *rows) {
        if (!r->info || !r->info->name || std::strcmp(r->info->name, "embed_lookup_q_bf16") != 0)
            continue;
        if (r->plugin != "libr4d")
            fail(std::string("embed_lookup_q_bf16 is published by '") + r->plugin + "'");
        if (hit) fail("duplicate embed_lookup_q_bf16 row");
        hit = r;
    }
    if (!hit) fail("missing libr4d/embed_lookup_q_bf16");
    if (!hit->info->launch || hit->info->domain != RAD_DOMAIN_DEVICE)
        fail("embed_lookup_q_bf16 is not a device row with a launcher");
    return hit;
}

struct Gpu {
    rad::Registry registry;
    const rad::KernelRow* row = nullptr;
    RadStream stream = nullptr;
    DeviceBuffer d_tok{size_t(kM * 4)};
    DeviceBuffer d_x{size_t(kM * kWidth * 2)};
};

/* One launch of the device row: uploads `ids`, runs the kernel and returns the m x kWidth rows.
 * Every operand is per-call, so no state survives a return and the caller compares against its
 * own literal oracle. */
std::vector<uint16_t> launch_rows(Gpu& g, const DeviceBuffer& wte, const DeviceBuffer& scale,
                                  const int32_t* ids, int64_t m, int64_t n_vocab, int32_t voff) {
    if (m <= 0 || m > kM) fail("launch row count outside the fixture's buffer");
    check(rad_memcpy_async(g.d_tok.data, ids, int64_t(m * 4), g.stream), "upload ids");
    check(rad_stream_sync(g.stream), "finish id upload");

    RadTensor t[5]{};
    t[0].dtype = RAD_I32;    t[0].rank = 1;
    t[0].shape[0] = m;       t[0].stride[0] = 1;      t[0].data = g.d_tok.data;
    t[1].dtype = RAD_F8E4M3; t[1].rank = 2;
    t[1].shape[0] = n_vocab; t[1].shape[1] = kWidth;
    t[1].stride[0] = kWidth; t[1].stride[1] = 1;      t[1].data = wte.data;
    t[2].dtype = RAD_BF16;   t[2].rank = 1;
    t[2].shape[0] = 1;       t[2].stride[0] = 1;      t[2].data = scale.data;
    t[3].dtype = RAD_BF16;   t[3].rank = 2;
    t[3].shape[0] = m;       t[3].shape[1] = kWidth;
    t[3].stride[0] = kWidth; t[3].stride[1] = 1;      t[3].data = g.d_x.data;
    /* `ahead` is operand 4 and the device kernel ignores it. */
    t[4].dtype = RAD_I32;    t[4].rank = 1;
    t[4].shape[0] = m;       t[4].stride[0] = 1;      t[4].data = nullptr;

    const RadParam params[] = { RAD_INT("M", m), RAD_INT("n_embd", kWidth),
                                RAD_INT("n_vocab", n_vocab), RAD_INT("vocab_offset", voff),
                                RAD_STR("dtype", "bf16") };
    RadArgs args{};
    args.t = t;  args.n_t = 5;
    args.p = params; args.n_p = 5;
    args.rank = 0; args.world_size = 1;

    std::string verr;
    check(g.registry.validate("embed_lookup_q", args.p, args.n_p, &verr),
          verr.empty() ? "parameters rejected" : verr.c_str());

    const int rc = g.row->info->launch(&args, g.stream);
    if (rc != RAD_OK) fail(std::string("launch refused: ") + rad_strerror(rc));
    check(rad_stream_sync(g.stream), "launch");

    std::vector<uint16_t> got(size_t(m * kWidth));
    check(rad_memcpy_async(got.data(), g.d_x.data, int64_t(got.size() * 2), g.stream),
          "download x");
    check(rad_stream_sync(g.stream), "finish download");
    return got;
}

/* Compares one launch against `want(m, j)`; counts, prints the first mismatches, returns the
 * number of differing cells. */
template <typename Want>
size_t compare(const std::vector<uint16_t>& got, int64_t m, const char* stage, int round,
               const int32_t* ids, Want want, size_t& compared, int& reported,
               int& first_bad_round) {
    size_t bad = 0;
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t j = 0; j < kWidth; ++j) {
            const uint16_t w = want(row, j);
            const uint16_t g = got[size_t(row * kWidth + j)];
            ++compared;
            if (g == w) continue;
            ++bad;
            if (reported < 16) {
                ++reported;
                std::printf("MISMATCH stage=%s round=%d row=%lld tok=%d j=%lld "
                            "got=0x%04x want=0x%04x\n",
                            stage, round, (long long)row, ids[row], (long long)j, g, w);
            }
        }
    }
    if (bad && first_bad_round < 0) first_bad_round = round;
    return bad;
}

/* ---- collisions ---------------------------------------------------------------------------
 * 32 pairs of distinct ids a stride apart, launched twice per round: the second launch is the
 * probe, and any lookup that answers a repeated id from remembered state must still produce the
 * byte-identical row, because the expected row is the literal dequant of the table cell. */
size_t run_collisions(Gpu& g, const DeviceBuffer& d_wte, const DeviceBuffer& d_scale, int rounds,
                      size_t& compared, int& reported, int& first_bad_round) {
    std::array<int32_t, size_t(kM)> ids{};
    size_t bad_total = 0;

    for (int round = 0; round < rounds; ++round) {
        for (int64_t p = 0; p < kPairs; ++p) {
            const int32_t base = int32_t(1000 + round * kPairs + p);
            ids[size_t(2 * p)]     = base;
            ids[size_t(2 * p + 1)] = base + int32_t(kStride);
        }
        for (int pass = 0; pass < 2; ++pass) {
            const std::vector<uint16_t> got =
                launch_rows(g, d_wte, d_scale, ids.data(), kM, kVocab, 0);
            const char* stage = pass ? "repeat" : "fill";
            bad_total += compare(got, kM, stage, round, ids.data(),
                                 [&](int64_t row, int64_t j) {
                                     return row_cell(ids[size_t(row)], j);
                                 },
                                 compared, reported, first_bad_round);
        }
    }
    return bad_total;
}

/* ---- table and scale identity -------------------------------------------------------------
 * Three cases over the SAME local ids 0..kIdLocal-1, launched in turn:
 *   A  table A, vocab_off 0,   scale 0.5   -- distinct table, shard and scale
 *   A2 table A, vocab_off 0,   scale 4.0   -- same token and table, distinct immutable scale
 *   B  table B, vocab_off 512, scale 2.0   -- same local ids, other table, other shard, other scale
 */
constexpr int64_t  kIdLocal   = 8;
constexpr int32_t  kIdVoffB   = 512;     /* shard B's first GLOBAL id */
constexpr uint16_t kIdScaleA  = 0x3F00;  /* 0.5 */
constexpr uint16_t kIdScaleA2 = 0x4080;  /* 4.0 */
constexpr uint16_t kIdScaleB  = 0x4000;  /* 2.0 */

/* Table B rotates the palette by half, so every cell it stores differs from table A's. */
int palette_index_for(bool second, int64_t local, int64_t j) {
    const int64_t idx = (local * 7 + j) % 16;
    return int(second ? (idx + 8) % 16 : idx);
}
uint8_t table_code(bool second, int64_t local, int64_t j) {
    return kPaletteCode[palette_index_for(second, local, j)];
}
uint16_t table_want(bool second, int64_t local, int64_t j, uint16_t scale) {
    return expected_cell(kPaletteBf16[palette_index_for(second, local, j)], scale);
}

size_t run_identity(Gpu& g, int rounds, size_t& compared, int& reported, int& first_bad_round) {
    for (int64_t local = 0; local < kIdLocal; ++local)
        for (int64_t j = 0; j < kWidth; ++j)
            if (table_code(false, local, j) == table_code(true, local, j))
                fail("fixture tables agree at a cell; the identity case would prove nothing");

    std::vector<uint8_t> tA(size_t(kIdLocal * kWidth)), tB(tA.size());
    for (int64_t local = 0; local < kIdLocal; ++local) {
        for (int64_t j = 0; j < kWidth; ++j) {
            tA[size_t(local * kWidth + j)] = table_code(false, local, j);
            tB[size_t(local * kWidth + j)] = table_code(true, local, j);
        }
    }
    DeviceBuffer d_tA(tA.size()), d_tB(tB.size());
    DeviceBuffer d_sA(2), d_sA2(2), d_sB(2);
    check(rad_memcpy_async(d_tA.data, tA.data(), int64_t(tA.size()), g.stream), "upload table A");
    check(rad_memcpy_async(d_tB.data, tB.data(), int64_t(tB.size()), g.stream), "upload table B");
    check(rad_memcpy_async(d_sA.data, &kIdScaleA, 2, g.stream), "upload scale A");
    check(rad_memcpy_async(d_sA2.data, &kIdScaleA2, 2, g.stream), "upload scale A2");
    check(rad_memcpy_async(d_sB.data, &kIdScaleB, 2, g.stream), "upload scale B");
    check(rad_stream_sync(g.stream), "finish identity upload");

    struct IdCase {
        const char* name;
        bool second;
        int32_t voff;
        uint16_t scale;
        const DeviceBuffer* wte;
        const DeviceBuffer* scale_dev;
    };
    const IdCase cases[3] = {
        { "A/off0/0.5", false, 0,        kIdScaleA,  &d_tA, &d_sA  },
        { "A/off0/4",   false, 0,        kIdScaleA2, &d_tA, &d_sA2 },
        { "B/off512/2", true,  kIdVoffB, kIdScaleB,  &d_tB, &d_sB  },
    };

    std::array<int32_t, size_t(kIdLocal)> ids{};
    size_t bad_total = 0;

    for (int round = 0; round < rounds; ++round) {
        for (const IdCase& c : cases) {
            for (int64_t m = 0; m < kIdLocal; ++m) ids[size_t(m)] = c.voff + int32_t(m);
            const std::vector<uint16_t> got =
                launch_rows(g, *c.wte, *c.scale_dev, ids.data(), kIdLocal, kIdLocal, c.voff);
            bad_total += compare(got, kIdLocal, c.name, round, ids.data(),
                                 [&](int64_t row, int64_t j) {
                                     return table_want(c.second, row, j, c.scale);
                                 },
                                 compared, reported, first_bad_round);
        }
    }
    return bad_total;
}

/* ---- boundary -----------------------------------------------------------------------------
 * The padding convention (negative id), an id at n_vocab, and an id past it write a zero row;
 * the first and last in-range ids write their table rows. All in one launch, so the boundary
 * rows sit next to the valid ones in the same wave-width walk. */
size_t run_boundary(Gpu& g, const DeviceBuffer& d_wte, const DeviceBuffer& d_scale,
                    size_t& compared, int& reported, int& first_bad_round) {
    constexpr int64_t kB = 8;
    const std::array<int32_t, size_t(kB)> ids = {
        -1, 0, int32_t(kVocab) - 1, int32_t(kVocab),
        int32_t(kVocab) + 7, -1000, 17, int32_t(kVocab) * 2
    };

    const std::vector<uint16_t> got = launch_rows(g, d_wte, d_scale, ids.data(), kB, kVocab, 0);
    return compare(got, kB, "boundary", 0, ids.data(),
                   [&](int64_t row, int64_t j) {
                       const int32_t id = ids[size_t(row)];
                       return (id < 0 || id >= kVocab) ? uint16_t(0) : row_cell(id, j);
                   },
                   compared, reported, first_bad_round);
}

int run(const char* plugin_path, int rounds) {
    if (rounds <= 0) fail("zero rounds is a failure, not a pass");
    if (1000 + kPairs * (rounds - 1) + (kPairs - 1) + kStride >= kVocab)
        fail("round count needs ids past the table; lower it");

    check(rad_dev_set(0), "select GPU 0");
    RadDeviceProps props{};
    check(rad_dev_props(0, &props), "GPU properties");
    if (props.is_host_backend) fail("a host backend is not the device row under test");

    Gpu g;
    check(rad_stream_create(&g.stream, 0), "create stream");
    g.registry.set_code_device(0);
    const int loaded = g.registry.load_plugin(plugin_path, 0);
    if (loaded < 0) fail("load libr4d plugin: " + std::string(rad_strerror(loaded)));
    if (loaded > 0) fail("libr4d declined this machine; nothing to run");
    if (!g.registry.load_errors().empty())
        fail("plugin load reported: " + g.registry.load_errors().front());
    g.row = pin_row(g.registry);

    std::vector<uint8_t> table(size_t(kVocab * kWidth));
    for (int64_t id = 0; id < kVocab; ++id)
        for (int64_t j = 0; j < kWidth; ++j)
            table[size_t(id * kWidth + j)] = kPaletteCode[palette_index(id, j)];

    DeviceBuffer d_wte(table.size());
    DeviceBuffer d_scale(2);
    check(rad_memcpy_async(d_wte.data, table.data(), int64_t(table.size()), g.stream),
          "upload table");
    check(rad_memcpy_async(d_scale.data, &kScaleOne, 2, g.stream), "upload scale");
    check(rad_stream_sync(g.stream), "finish upload");

    size_t compared = 0;
    int reported = 0, first_bad_round = -1;

    const size_t collision_bad =
        run_collisions(g, d_wte, d_scale, rounds, compared, reported, first_bad_round);
    std::printf("PLE_LOOKUP_COLLISIONS rounds=%d compared=%zu mismatches=%zu first_bad_round=%d "
                "verdict=%s\n",
                rounds, compared, collision_bad, first_bad_round,
                collision_bad ? "MISMATCH" : "OK");

    compared = 0;
    const size_t identity_bad = run_identity(g, rounds, compared, reported, first_bad_round);
    std::printf("PLE_LOOKUP_IDENTITY rounds=%d compared=%zu mismatches=%zu first_bad_round=%d "
                "verdict=%s\n",
                rounds, compared, identity_bad, first_bad_round,
                identity_bad ? "MISMATCH" : "OK");
    if (compared != size_t(rounds) * size_t(kIdLocal * kWidth) * 3u)
        fail("identity comparison count does not match the round count");

    compared = 0;
    const size_t boundary_bad =
        run_boundary(g, d_wte, d_scale, compared, reported, first_bad_round);
    std::printf("PLE_LOOKUP_BOUNDARY compared=%zu mismatches=%zu verdict=%s\n",
                compared, boundary_bad, boundary_bad ? "MISMATCH" : "OK");

    std::printf("PLE_LOOKUP_RESULT rounds=%d mismatches=%zu verdict=%s\n",
                rounds, collision_bad + identity_bad + boundary_bad,
                (collision_bad || identity_bad || boundary_bad) ? "MISMATCH" : "OK");

    rad_stream_destroy(g.stream);
    return (collision_bad || identity_bad || boundary_bad) ? 2 : 0;
}

}  /* namespace */

int main(int argc, char** argv) {
    std::string plugin;
    if (argc > 1) {
        plugin = argv[1];
    } else {
        const char* home = std::getenv("RADIANCE_HOME");
        plugin = home ? std::string(home) + "/kernels/libr4d.so"
                      : std::string("radiance_home/kernels/libr4d.so");
    }
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 1000;
    try {
        return run(plugin.c_str(), rounds);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "PLE_LOOKUP_ERROR %s\n", e.what());
        return 1;
    }
}
