/* format_test.cpp -- the .rad container: write one, read it back, and prove the invariants the
 * hot path relies on.
 *
 * The interesting cases here are the refusals. A container is a file somebody was handed, and the
 * engine mmaps it and then addresses into it with arithmetic and no bounds test -- so every claim
 * the arithmetic rests on has to be checked at open, and "checked at open" has to mean RAD_E_FORMAT
 * with a message rather than a SIGSEGV three components later.
 */
#include "rad_test.h"
#include "format/radfile.h"
#include "format/table_copy.h"
#include "format/tunecache.h"
#include "format/imatrix.h"
#include "format/safetensors.h"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <functional>
#include <signal.h>
#include <string>
#include <sys/wait.h>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>

using namespace rad;

namespace {

std::string tmp_path(const char* stem) {
    const char* dir = std::getenv("TMPDIR");
    return std::string(dir && *dir ? dir : "/tmp") + "/radfmt_" +
           std::to_string((long)::getpid()) + "_" + stem;
}

std::vector<uint8_t> ramp(int64_t n, uint8_t seed) {
    std::vector<uint8_t> v((size_t)n);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (uint8_t)((i * 31 + seed) & 0xFF);
    return v;
}

std::vector<uint8_t> slurp(const std::string& path) {
    std::vector<uint8_t> v;
    if (FILE* f = std::fopen(path.c_str(), "rb")) {
        uint8_t buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
        std::fclose(f);
    }
    return v;
}

bool exists(const std::string& path) { return ::access(path.c_str(), F_OK) == 0; }

/* The two encodings the test container is made of: a bf16 weight stored as it is, and int4 codes
 * with an f16 scale a [1 x 32] group. The second has two planes, and over the expert shape below
 * its codes plane ends off a 256-byte boundary, so the scale plane's placement is a real
 * alignment rather than one that falls out of the arithmetic. */
RadEncoding plain_bf16() { return rad_enc_plain(RAD_BF16); }
RadEncoding int4_g32() { return rad_enc_affine(RAD_I4, RAD_F16, 1, 32); }

RadWriter::Weight mkw(const char* name, const RadEncoding& enc, int64_t rows, int64_t cols,
                      int32_t layer, int32_t expert, int32_t slot) {
    RadWriter::Weight w;
    w.name     = name;
    w.enc      = enc;
    w.rank     = 2;
    w.shape[0] = rows;
    w.shape[1] = cols;
    w.layer    = layer;
    w.expert   = expert;
    w.slot     = slot;
    return w;
}

/* A weight's planes as add_weight(w, planes) takes them: plane k a ramp of its exact
 * rad_enc_plane_bytes(), seeded per plane so two planes of one weight never hold the same bytes. */
struct Planes {
    std::vector<std::vector<uint8_t>> bytes;
    std::vector<const void*>          ptr;
};

Planes planes_of(const RadWriter::Weight& w, uint8_t seed) {
    Planes p;
    int64_t rows = 0, cols = 0;
    rad_enc_view(w.rank, w.shape, &rows, &cols);
    for (int k = 0; k < w.enc.n_planes; ++k)
        p.bytes.push_back(ramp(rad_enc_plane_bytes(&w.enc.plane[k], rows, cols),
                               (uint8_t)(seed + 16 * k)));
    for (const auto& b : p.bytes) p.ptr.push_back(b.data());
    return p;
}

/* The expert shape: [36, 64] of int4 is 1152 bytes of codes, so the 144-byte scale plane starts at
 * 1280. Seeds per weight kind, the same for every expert of every layer -- which is what lets the
 * stride test compare two experts' bytes. */
constexpr int64_t EXP_ROWS = 36, EXP_COLS = 64;
enum : uint8_t { SEED_EMB = 1, SEED_OUT = 2, SEED_QKV = 3, SEED_GATE = 4, SEED_UP = 5,
                 SEED_DOWN = 6 };
const char* const kRecipe = "blk.*.ffn_(gate|up).*=rtn:codes=i4,group=32,scale=f16\n";
const char* const kOptions = "codes=i4,group=32,scale=f16";

/* One small but complete container: two model weights, one layer's dense weight, and two layers
 * of four experts with three slots each -- which is what the fixed-stride addressing is for. The
 * gate is reserved and then written a plane at a time, scale first and codes in two halves, the
 * way a quantiser produces it; everything else goes in whole. */
struct Built {
    std::string path;
};

Built write_model(const char* stem) {
    Built b;
    b.path = tmp_path(stem);

    RadWriter w;
    CHECK_OK(w.begin(b.path.c_str()));
    w.set_arch("radtest");
    w.set_model_name("format-test");
    w.set_quant("fp8_block");
    w.set_recipe(kRecipe);
    w.set_created_by("format_test");
    w.meta_i("n_layers", 2);
    w.meta_i("n_embd", 64);
    w.meta_f("rms_eps", 1e-6);
    w.meta_s("note", "written by format_test");

    auto whole = [&](const RadWriter::Weight& x, uint8_t seed) {
        const Planes p = planes_of(x, seed);
        CHECK(w.add_weight(x, p.ptr.data()) >= 0);
    };
    whole(mkw("token_embd.weight", plain_bf16(), 128, 64, -1, -1, 0), SEED_EMB);
    whole(mkw("output.weight", plain_bf16(), 128, 64, -1, -1, 0), SEED_OUT);

    /* Two layers, each: one dense weight, then four experts of three slots. The dense weight
     * comes first so the expert run is contiguous in the directory, which is what
     * index_experts() requires and what makes the stride derivable. */
    for (int32_t layer = 0; layer < 2; ++layer) {
        const std::string blk = "blk." + std::to_string(layer);
        whole(mkw((blk + ".attn_qkv.weight").c_str(), plain_bf16(), 64, 64, layer, -1, 0),
              SEED_QKV);
        for (int32_t e = 0; e < 4; ++e) {
            const std::string x = "." + std::to_string(e);
            RadWriter::Weight g = mkw((blk + ".ffn_gate" + x).c_str(), int4_g32(), EXP_ROWS,
                                      EXP_COLS, layer, e, 0);
            g.quantizer = "rtn";
            g.options = kOptions;
            const Planes gp = planes_of(g, SEED_GATE);
            const int64_t gi = w.add_weight(g);
            CHECK(gi >= 0);
            const int64_t half = (int64_t)gp.bytes[0].size() / 2;
            CHECK_OK(w.write_plane(gi, 1, 0, gp.bytes[1].data(), (int64_t)gp.bytes[1].size()));
            CHECK_OK(w.write_plane(gi, 0, half, gp.bytes[0].data() + half,
                                   (int64_t)gp.bytes[0].size() - half));
            CHECK_OK(w.write_plane(gi, 0, 0, gp.bytes[0].data(), half));

            RadWriter::Weight u = mkw((blk + ".ffn_up" + x).c_str(), int4_g32(), EXP_ROWS,
                                      EXP_COLS, layer, e, 1);
            u.quantizer = "rtn";
            u.options = kOptions;
            whole(u, SEED_UP);
            whole(mkw((blk + ".ffn_down" + x).c_str(), plain_bf16(), 64, 32, layer, e, 2),
                  SEED_DOWN);
        }
        w.add_profile(layer, 0, 0.4f);
        w.add_profile(layer, 1, 0.3f);
        w.add_profile(layer, 2, 0.2f);
        w.add_profile(layer, 3, 0.1f);
    }

    CHECK_OK(w.finish());
    return b;
}

/* Every plane of `e` holds what planes_of() made for `w` at `seed`. */
bool planes_match(const RadFile& f, const RadFileEntry& e, const RadWriter::Weight& w,
                  uint8_t seed) {
    const Planes want = planes_of(w, seed);
    if (e.n_planes != want.bytes.size()) return false;
    for (uint32_t k = 0; k < e.n_planes; ++k) {
        const RadFilePlane& p = f.plane(e, (int)k);
        if (p.bytes != want.bytes[k].size() ||
            std::memcmp(f.data(p), want.bytes[k].data(), want.bytes[k].size()) != 0)
            return false;
    }
    return true;
}

/* A copy of `src` with `edit` applied to its bytes, beside it; the copy's path. The edit is handed
 * the header as written so it can find the table it means to damage. */
using Edit = std::function<void(std::vector<uint8_t>&, const RadFileHeader&)>;

std::string patched(const std::string& src, const char* tag, const Edit& edit) {
    std::vector<uint8_t> all = slurp(src);
    RadFileHeader h{};
    std::memcpy(&h, all.data(), sizeof h);
    edit(all, h);
    const std::string dst = src + "." + tag;
    FILE* f = std::fopen(dst.c_str(), "wb");
    if (f) {
        std::fwrite(all.data(), 1, all.size(), f);
        std::fclose(f);
    }
    return dst;
}

template <typename T> T get(const std::vector<uint8_t>& f, uint64_t off) {
    T v;
    std::memcpy(&v, f.data() + off, sizeof v);
    return v;
}
template <typename T> void put(std::vector<uint8_t>& f, uint64_t off, const T& v) {
    std::memcpy(f.data() + off, &v, sizeof v);
}

/* What `fn` printed to stderr, where the reader logs why it refused a file. */
std::string stderr_of(const std::function<void()>& fn) {
    const std::string p = tmp_path("stderr");
    std::fflush(stderr);
    const int saved = ::dup(2);
    const int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (saved < 0 || fd < 0) {
        if (saved >= 0) ::close(saved);
        if (fd >= 0) ::close(fd);
        fn();
        return std::string();
    }
    ::dup2(fd, 2);
    ::close(fd);
    fn();
    std::fflush(stderr);
    ::dup2(saved, 2);
    ::close(saved);
    const std::vector<uint8_t> v = slurp(p);
    ::unlink(p.c_str());
    return std::string(v.begin(), v.end());
}

}  /* namespace */

/* ------------------------------------------------------------------ round trip */
TEST(write_and_read_back) {
    Built b = write_model("roundtrip");

    RadFile f;
    REQUIRE(f.open(b.path.c_str()) >= 0);

    CHECK_EQ(std::string(f.str(f.header().arch_id)), std::string("radtest"));
    CHECK_EQ(std::string(f.str(f.header().model_name)), std::string("format-test"));
    CHECK_EQ(std::string(f.str(f.header().quant)), std::string("fp8_block"));
    CHECK_EQ(std::string(f.str(f.header().recipe)), std::string(kRecipe));
    CHECK_EQ(std::string(f.str(f.header().created_by)), std::string("format_test"));
    CHECK_EQ((int)f.header().version, (int)RAD_FORMAT_VER);
    CHECK_EQ((int)f.header().magic, (int)RAD_MAGIC);

    int64_t n = 0;
    CHECK(f.get_i("n_layers", &n));
    CHECK_EQ(n, (int64_t)2);
    double eps = 0;
    CHECK(f.get_f("rms_eps", &eps));
    CHECK_NEAR(eps, 1e-6, 1e-12);
    CHECK_EQ(std::string(f.get_s("note", "")), std::string("written by format_test"));

    /* 2 model + 2 layers * (1 dense + 4 experts * 3 slots) = 2 + 26 = 28. */
    CHECK_EQ(f.entry_count(), (int64_t)28);
    CHECK_EQ(f.profile_count(), (int64_t)8);
    /* 28 entries, of which 16 are two-plane experts. */
    CHECK_EQ((int64_t)f.header().plane_count, (int64_t)(28 + 16));

    const RadFileEntry* e = f.find("token_embd.weight");
    REQUIRE(e != nullptr);
    const RadEncoding plain = plain_bf16();
    CHECK(rad_enc_equal(&f.encoding(*e), &plain));
    CHECK_EQ((int)e->rank, 2);
    CHECK_EQ(e->shape[0], (int64_t)128);
    CHECK_EQ(e->shape[1], (int64_t)64);
    CHECK_EQ((int)e->n_planes, 1);
    CHECK_EQ(std::string(f.str(e->quantizer)), std::string());
    CHECK_EQ((int64_t)f.plane(*e, 0).bytes, (int64_t)(128 * 64 * 2));
    /* The first plane opens the entry, and the bytes came back, not just the directory row. */
    CHECK_EQ((const void*)f.data(f.plane(*e, 0)), (const void*)f.data(*e));
    CHECK(planes_match(f, *e, mkw("", plain, 128, 64, -1, -1, 0), SEED_EMB));

    /* An expert: two planes, and the provenance the quantiser left. */
    const RadFileEntry* g = f.find("blk.1.ffn_gate.3");
    REQUIRE(g != nullptr);
    const RadEncoding i4 = int4_g32();
    CHECK(rad_enc_equal(&f.encoding(*g), &i4));
    CHECK_EQ((int)g->n_planes, 2);
    CHECK_EQ(std::string(f.str(g->quantizer)), std::string("rtn"));
    CHECK_EQ(std::string(f.str(g->options)), std::string(kOptions));
    CHECK_EQ((int)g->layer, 1);
    CHECK_EQ((int)g->expert, 3);
    CHECK_EQ((int)g->slot, 0);
    /* Written a piece at a time, out of order, and read back whole. */
    CHECK(planes_match(f, *g, mkw("", i4, EXP_ROWS, EXP_COLS, 1, 3, 0), SEED_GATE));
    const RadFileEntry* u = f.find("blk.0.ffn_up.0");
    REQUIRE(u != nullptr);
    CHECK(planes_match(f, *u, mkw("", i4, EXP_ROWS, EXP_COLS, 0, 0, 1), SEED_UP));

    CHECK(f.find("no.such.weight") == nullptr);
    f.close();
    ::unlink(b.path.c_str());
}

/* ------------------------------------------------------------------ the encoding table */
TEST(an_encoding_is_stored_once_however_many_weights_share_it) {
    Built b = write_model("encs");
    RadFile f;
    REQUIRE(f.open(b.path.c_str()) >= 0);

    /* 28 weights, two encodings: a table row per distinct encoding, and every entry naming one.
     * Equality is byte-for-byte, so the row is the struct the writer was handed and the reader
     * hands a kernel exactly what the quantiser described. */
    REQUIRE_EQ(f.encoding_count(), (int64_t)2);
    CHECK(!rad_enc_equal(&f.encoding_at(0), &f.encoding_at(1)));
    const RadEncoding plain = plain_bf16(), i4 = int4_g32();
    for (int64_t i = 0; i < f.entry_count(); ++i) {
        const RadFileEntry& e = f.entry(i);
        CHECK(e.enc < 2);
        const bool expert_codes = e.expert >= 0 && e.slot < 2;
        CHECK(rad_enc_equal(&f.encoding(e), expert_codes ? &i4 : &plain));
        CHECK_EQ(e.n_planes, (uint32_t)f.encoding(e).n_planes);
    }
    f.close();
    ::unlink(b.path.c_str());
}

/* ------------------------------------------------------------------ alignment */
TEST(alignment_invariants) {
    Built b = write_model("align");
    RadFile f;
    REQUIRE(f.open(b.path.c_str()) >= 0);

    /* Pool 2 MiB (huge-page mappable), entry 4 KiB (the movement unit), plane 256 B (spec §4.1). */
    CHECK_EQ((int64_t)(f.header().data_off % RAD_ALIGN_POOL), (int64_t)0);

    for (int64_t i = 0; i < f.entry_count(); ++i) {
        const RadFileEntry& e = f.entry(i);
        CHECK_EQ((int64_t)(e.offset % RAD_ALIGN_UNIT), (int64_t)0);
        /* Every entry lies inside the data blob, which is what makes data() safe. */
        CHECK(e.offset >= f.header().data_off);
        CHECK(e.offset + e.bytes <= f.header().data_off + f.header().data_bytes);

        /* Its planes: in the encoding's order, each exactly its encoding's size, 256-aligned,
         * none overlapping the one before, and the entry ending where its last plane does --
         * so the entry is one span a mover can copy and a plane is a slice by arithmetic. */
        int64_t rows = 0, cols = 0;
        rad_enc_view(e.rank, e.shape, &rows, &cols);
        uint64_t end = e.offset;
        for (uint32_t k = 0; k < e.n_planes; ++k) {
            const RadFilePlane& p = f.plane(e, (int)k);
            CHECK_EQ((int64_t)p.bytes, rad_enc_plane_bytes(&f.encoding(e).plane[k], rows, cols));
            CHECK_EQ((int64_t)(p.offset % RAD_ALIGN_SUB), (int64_t)0);
            CHECK(p.offset >= end);
            end = p.offset + p.bytes;
        }
        CHECK_EQ(f.plane(e, 0).offset, e.offset);
        CHECK_EQ(e.offset + e.bytes, end);
    }

    /* The expert's scale plane does not follow its codes directly: 1152 bytes of codes round up
     * to a plane at 1280. */
    const RadFileEntry* g = f.find("blk.0.ffn_gate.0");
    REQUIRE(g != nullptr);
    CHECK_EQ((int64_t)f.plane(*g, 0).bytes, (int64_t)1152);
    CHECK_EQ((int64_t)(f.plane(*g, 1).offset - g->offset), (int64_t)1280);
    CHECK_EQ((int64_t)f.plane(*g, 1).bytes, (int64_t)(EXP_ROWS * 2 * 2));

    /* The string blob is NUL-terminated at its end, which is what lets str() hand back a bare
     * pointer instead of scanning. */
    CHECK_EQ((int)((const char*)f.base())[f.header().str_off + f.header().str_bytes - 1], 0);

    f.close();
    ::unlink(b.path.c_str());
}

/* ------------------------------------------------------------------ the whole point */
TEST(fixed_stride_expert_addressing) {
    Built b = write_model("stride");
    RadFile f;
    REQUIRE(f.open(b.path.c_str()) >= 0);

    CHECK_EQ(f.group_count(), (int64_t)2);

    for (int32_t layer = 0; layer < 2; ++layer) {
        const RadFile::ExpertGroup* g = f.group_for_layer(layer);
        CHECK(g != nullptr);
        if (!g) continue;

        CHECK_EQ((int)g->n_expert, 4);
        CHECK_EQ((int)g->n_slot, 3);
        CHECK(g->stride > 0);
        CHECK(g->stride >= g->unit_bytes);
        CHECK_EQ((int64_t)(g->stride % RAD_ALIGN_UNIT), (int64_t)0);

        /* base + id*stride must land exactly on the directory's own offset for every expert and
         * every slot. This is the property the mover and the expert kernels use WITHOUT a bounds
         * test, so it is the one the test exists for. */
        for (int32_t e = 0; e < g->n_expert; ++e)
            for (int32_t s = 0; s < g->n_slot; ++s) {
                const RadFileEntry& row = f.expert_entry(*g, e, s);
                CHECK_EQ((int)row.layer, (int)layer);
                CHECK_EQ((int)row.expert, (int)e);
                CHECK_EQ((int)row.slot, (int)s);
                CHECK_EQ((const void*)f.expert_ptr(*g, e, s), (const void*)f.data(row));
                CHECK_EQ((int64_t)(row.offset - g->base - (uint64_t)e * g->stride),
                         (int64_t)g->slot_off[s]);
                /* A plane sits at the same displacement inside every expert's slot, so a scale
                 * plane is addressed by the same arithmetic as the codes it scales. */
                const RadFileEntry& row0 = f.expert_entry(*g, 0, s);
                for (uint32_t k = 0; k < row.n_planes; ++k)
                    CHECK_EQ(f.plane(row, (int)k).offset - row.offset,
                             f.plane(row0, (int)k).offset - row0.offset);
            }

        /* And the bytes really are the same for every expert, since the writer was handed the
         * same payload -- so a wrong stride would show up here and not only as a wrong pointer. */
        const uint8_t* a = f.expert_ptr(*g, 0, 0);
        const uint8_t* c = f.expert_ptr(*g, 3, 0);
        CHECK(std::memcmp(a, c, (size_t)g->slot_bytes[0]) == 0);
    }

    f.close();
    ::unlink(b.path.c_str());
}

/* --ngram-placement ram: A TABLE COPIED OUT OF THE CONTAINER INTO LOCKED MEMORY is the mapping's
 * bytes, every one of them, at an address no file backs; and a copy read from the wrong file offset
 * is refused by the check against the mapping rather than handed out. Small enough to lock under a
 * default RLIMIT_MEMLOCK, and an odd size so the copy's 2 MiB rounding is exercised. */
TEST(a_table_copied_into_locked_memory_is_the_mapping_byte_for_byte) {
    const std::string p = tmp_path("tcopy");
    RadWriter w;
    REQUIRE(w.begin(p.c_str()) >= 0);
    w.set_arch("radtest");
    const RadWriter::Weight x = mkw("ngram.weight", plain_bf16(), 1500, 999, -1, -1, 0);
    const Planes want = planes_of(x, 5);
    REQUIRE(w.add_weight(x, want.ptr.data()) >= 0);
    /* A weight after it, so a read shifted past the table's end is still inside the file. */
    const RadWriter::Weight y = mkw("tail.weight", plain_bf16(), 64, 64, -1, -1, 0);
    const Planes ywant = planes_of(y, 6);
    REQUIRE(w.add_weight(y, ywant.ptr.data()) >= 0);
    REQUIRE(w.finish() >= 0);

    RadFile f;
    REQUIRE(f.open(p.c_str()) >= 0);
    const RadFileEntry* e = f.find("ngram.weight");
    REQUIRE(e != nullptr);
    const RadFilePlane& pl = f.plane(*e, 0);
    const uint8_t* src = f.data(pl);
    const int64_t n = (int64_t)pl.bytes;
    CHECK_EQ(n, (int64_t)1500 * 999 * 2);

    TableCopy c;
    std::string why;
    const int rc = table_copy_make(f, src, n, (int64_t)pl.offset, &c, &why);
    if (rc == RAD_E_NOMEM && why.find("mlock2") != std::string::npos) {
        std::fprintf(stderr, "  SKIP %s: %s\n", ::radtest::current(), why.c_str());
    } else {
        CHECK_OK(rc);
        CHECK(c.base != nullptr && c.base != (const void*)src);
        CHECK_EQ(c.bytes % ((size_t)2 << 20), (size_t)0);
        CHECK(c.bytes >= (size_t)n);
        if (c.base) CHECK(std::memcmp(c.base, src, (size_t)n) == 0);
        table_copy_free(&c);
        CHECK(c.base == nullptr);

        /* A sixteen-byte shift in the file offset reads real bytes from the wrong place. */
        TableCopy bad;
        std::string bwhy;
        CHECK_EQ(table_copy_make(f, src, n, (int64_t)pl.offset + 16, &bad, &bwhy), (int)RAD_E_IO);
        CHECK(bad.base == nullptr);
        CHECK(bwhy.find("does not match") != std::string::npos);
    }
    table_copy_free(&c);                  /* a copy never made, or freed already: a no-op */
    f.close();
    ::unlink(p.c_str());
}

/* ------------------------------------------------------------------ reserving and writing */
TEST(a_weight_is_written_in_pieces_in_any_order) {
    const std::string p = tmp_path("pieces");
    RadWriter w;
    REQUIRE(w.begin(p.c_str()) >= 0);
    w.set_arch("radtest");
    const RadWriter::Weight x = mkw("blk.0.ffn_up.weight", int4_g32(), EXP_ROWS, EXP_COLS, 0,
                                    -1, 0);
    const Planes want = planes_of(x, 9);
    const int64_t idx = w.add_weight(x);
    REQUIRE(idx >= 0);
    CHECK_EQ(w.plane_bytes(idx, 0), (int64_t)1152);
    CHECK_EQ(w.plane_bytes(idx, 1), (int64_t)144);
    CHECK(w.plane_bytes(idx, 2) < 0);
    CHECK(w.plane_bytes(idx + 1, 0) < 0);

    /* Refused without writing a byte: past the plane, before it, a plane the encoding does not
     * have, a weight that was never reserved, and data that is not there. */
    const uint8_t* c = want.bytes[0].data();
    CHECK_EQ(w.write_plane(idx, 0, 1100, c, 100), (int)RAD_E_INVAL);
    CHECK_EQ(w.write_plane(idx, 0, -1, c, 1), (int)RAD_E_INVAL);
    CHECK_EQ(w.write_plane(idx, 2, 0, c, 1), (int)RAD_E_INVAL);
    CHECK_EQ(w.write_plane(idx + 1, 0, 0, c, 1), (int)RAD_E_INVAL);
    CHECK_EQ(w.write_plane(-1, 0, 0, c, 1), (int)RAD_E_INVAL);
    CHECK_EQ(w.write_plane(idx, 0, 0, nullptr, 1), (int)RAD_E_INVAL);

    /* The scale first, then the codes a third at a time, last third first -- the order a
     * producer that emits a block of rows of every plane at once writes them in. */
    CHECK_OK(w.write_plane(idx, 1, 0, want.bytes[1].data(), 144));
    CHECK_OK(w.write_plane(idx, 0, 768, c + 768, 384));
    CHECK_OK(w.write_plane(idx, 0, 0, c, 384));
    CHECK_OK(w.write_plane(idx, 0, 384, c + 384, 384));
    REQUIRE(w.finish() >= 0);

    RadFile f;
    REQUIRE(f.open(p.c_str()) >= 0);
    const RadFileEntry* e = f.find("blk.0.ffn_up.weight");
    REQUIRE(e != nullptr);
    CHECK(planes_match(f, *e, x, 9));
    f.close();
    ::unlink(p.c_str());
}

/* A SEALED WRITER IS THE SAME FILE. seal() fixes the layout once every table is in, and the planes
 * then go straight to their final offsets -- in any order -- with no spill and no splice; the
 * container has to come out byte for byte the one the spilling writer makes. A table changed after
 * the seal would move the blob, so finish() refuses it, and so does add_weight(). */
TEST(a_sealed_writer_writes_the_same_file_with_no_spill) {
    struct One { RadWriter::Weight x; uint8_t seed; };
    std::vector<One> ws;
    ws.push_back({ mkw("token_embd.weight", plain_bf16(), 128, 64, -1, -1, 0), SEED_EMB });
    for (int32_t layer = 0; layer < 2; ++layer)
        for (int32_t e = 0; e < 4; ++e) {
            const std::string n = "blk." + std::to_string(layer) + ".ffn_up." + std::to_string(e);
            RadWriter::Weight u = mkw(n.c_str(), int4_g32(), EXP_ROWS, EXP_COLS, layer, e, 0);
            u.quantizer = "rtn";
            u.options = kOptions;
            ws.push_back({ u, (uint8_t)(SEED_UP + layer * 4 + e) });
        }

    auto build = [&](const std::string& p, bool sealed, bool late) -> int {
        RadWriter w;
        if (w.begin(p.c_str()) < 0) return RAD_E_IO;
        w.set_arch("radtest");
        w.set_model_name("seal-test");
        w.meta_i("n_layers", 2);
        std::vector<int64_t> idx;
        for (const One& o : ws) idx.push_back(w.add_weight(o.x));
        w.add_profile(0, 1, 0.5f);
        if (sealed) {
            CHECK_OK(w.seal());
            CHECK(::access((p + ".blob.tmp").c_str(), F_OK) != 0);
            CHECK_EQ(w.add_weight(ws[0].x), (int64_t)RAD_E_STATE);
        }
        for (size_t i = ws.size(); i-- > 0;) {
            const Planes pl = planes_of(ws[i].x, ws[i].seed);
            for (size_t k = 0; k < pl.bytes.size(); ++k)
                CHECK_OK(w.write_plane(idx[i], (int)k, 0, pl.bytes[k].data(),
                                       (int64_t)pl.bytes[k].size()));
        }
        if (late) w.meta_s("note", "after the seal");
        return w.finish();
    };

    const std::string a = tmp_path("seal-spilled"), b = tmp_path("seal-sealed");
    REQUIRE(build(a, false, false) >= 0);
    REQUIRE(build(b, true, false) >= 0);
    const std::vector<uint8_t> fa = slurp(a), fb = slurp(b);
    CHECK(!fa.empty());
    CHECK(fa == fb);

    RadFile f;
    REQUIRE(f.open(b.c_str()) >= 0);
    for (const One& o : ws) {
        const RadFileEntry* e = f.find(o.x.name.c_str());
        REQUIRE(e != nullptr);
        CHECK(planes_match(f, *e, o.x, o.seed));
    }
    f.close();

    const std::string c = tmp_path("seal-late");
    CHECK_EQ(build(c, true, true), (int)RAD_E_STATE);
    CHECK(::access(c.c_str(), F_OK) != 0);
    ::unlink(a.c_str());
    ::unlink(b.c_str());
}

TEST(a_plane_left_short_is_refused_at_finish) {
    /* A hole in a plane is zeros the reader cannot tell from codes: a quantiser that returned
     * early, a block loop with an off-by-one. The writer refuses the file rather than write it,
     * and leaves nothing behind -- neither the target nor the spill. */
    const RadWriter::Weight x = mkw("w", int4_g32(), EXP_ROWS, EXP_COLS, -1, -1, 0);
    const Planes want = planes_of(x, 3);

    for (int which = 0; which < 2; ++which) {
        const std::string p = tmp_path(which ? "short_never" : "short_scale");
        RadWriter w;
        REQUIRE(w.begin(p.c_str()) >= 0);
        w.set_arch("radtest");
        const RadWriter::Weight ok = mkw("whole", plain_bf16(), 4, 64, -1, -1, 0);
        const Planes okp = planes_of(ok, 1);
        CHECK(w.add_weight(ok, okp.ptr.data()) >= 0);
        const int64_t idx = w.add_weight(x);
        REQUIRE(idx >= 0);
        if (which == 0) {
            /* every plane begun, the last two bytes of the scale never written */
            CHECK_OK(w.write_plane(idx, 0, 0, want.bytes[0].data(), 1152));
            CHECK_OK(w.write_plane(idx, 1, 0, want.bytes[1].data(), 142));
        }
        /* which == 1: reserved and never written at all */
        CHECK_EQ(w.finish(), (int)RAD_E_STATE);
        CHECK(!exists(p));
        CHECK(!exists(p + ".blob.tmp"));
    }
}

TEST(a_weight_the_writer_cannot_place_is_refused_and_leaves_no_trace) {
    const std::string p = tmp_path("refuse");
    RadWriter w;
    REQUIRE(w.begin(p.c_str()) >= 0);
    w.set_arch("radtest");

    RadWriter::Weight no_enc = mkw("a", plain_bf16(), 4, 64, -1, -1, 0);
    no_enc.enc = RadEncoding{};                       /* no planes, no scheme */
    CHECK(w.add_weight(no_enc) < 0);
    RadWriter::Weight bad_dtype = mkw("b", plain_bf16(), 4, 64, -1, -1, 0);
    bad_dtype.enc.plane[0].dtype = RAD_DT_PLUGIN_BASE;   /* a dtype the core cannot size */
    CHECK(w.add_weight(bad_dtype) < 0);
    RadWriter::Weight no_rank = mkw("c", plain_bf16(), 4, 64, -1, -1, 0);
    no_rank.rank = 0;
    CHECK(w.add_weight(no_rank) < 0);
    CHECK(w.add_weight(mkw("d", plain_bf16(), 0, 64, -1, -1, 0)) < 0);
    CHECK(w.add_weight(mkw("", plain_bf16(), 4, 64, -1, -1, 0)) < 0);

    /* The refusals reserved nothing: the one weight that is placed is the file's only entry,
     * with the only plane, at the start of the blob. */
    const RadWriter::Weight ok = mkw("ok", plain_bf16(), 4, 64, -1, -1, 0);
    const Planes okp = planes_of(ok, 2);
    CHECK_EQ(w.add_weight(ok, okp.ptr.data()), (int64_t)0);
    REQUIRE(w.finish() >= 0);

    RadFile f;
    REQUIRE(f.open(p.c_str()) >= 0);
    CHECK_EQ(f.entry_count(), (int64_t)1);
    CHECK_EQ((int64_t)f.header().plane_count, (int64_t)1);
    CHECK_EQ(f.encoding_count(), (int64_t)1);
    CHECK_EQ(f.entry(0).offset, f.header().data_off);
    CHECK(planes_match(f, f.entry(0), ok, 2));
    f.close();
    ::unlink(p.c_str());
}

/* ------------------------------------------------------------------ the vocab section */
TEST(the_vocab_section_is_rebased_to_where_it_lands) {
    /* Serialised with section-relative offsets, because the writer alone knows where the section
     * will sit; finish() adds that position, and the reader then reads it in place. */
    const std::string p = tmp_path("vocab");
    RadWriter w;
    REQUIRE(w.begin(p.c_str()) >= 0);
    w.set_arch("radtest");

    std::vector<uint8_t> blob(sizeof(RadVocabHeader) + 3 * sizeof(rad_stroff), 0);
    RadVocabHeader h{};
    h.kind = RAD_TOK_BPE;
    h.n_tokens = 3;
    h.tok_text_off = sizeof(RadVocabHeader);
    h.bos = h.eos = h.eot = h.pad_id = h.unk = h.sep = -1;
    h.chat_template = w.intern_string("{{ messages }}");
    const rad_stroff text[3] = { w.intern_string("a"), w.intern_string("bc"),
                                 w.intern_string("<eos>") };
    std::memcpy(blob.data(), &h, sizeof h);
    std::memcpy(blob.data() + sizeof h, text, sizeof text);

    /* Absolute offsets do not fit the section and are refused before anything is placed. */
    RadVocabHeader abs = h;
    abs.tok_text_off = 1u << 20;
    std::vector<uint8_t> bad = blob;
    std::memcpy(bad.data(), &abs, sizeof abs);
    CHECK_EQ(w.set_vocab_section(bad.data(), (int64_t)bad.size()), (int)RAD_E_INVAL);
    CHECK_EQ(w.set_vocab_section(blob.data(), 16), (int)RAD_E_INVAL);

    CHECK_OK(w.set_vocab_section(blob.data(), (int64_t)blob.size()));
    REQUIRE(w.finish() >= 0);

    RadFile f;
    REQUIRE(f.open(p.c_str()) >= 0);
    const VocabSection v = f.vocab();
    REQUIRE(bool(v));
    CHECK_EQ((int)v.h->n_tokens, 3);
    REQUIRE(v.tok_text != nullptr);
    CHECK_EQ((const void*)v.tok_text,
             (const void*)(f.base() + f.header().vocab_off + sizeof(RadVocabHeader)));
    CHECK_EQ(std::string(f.str(v.tok_text[1])), std::string("bc"));
    CHECK_EQ(std::string(f.str(v.tok_text[2])), std::string("<eos>"));
    CHECK_EQ(std::string(f.str(v.h->chat_template)), std::string("{{ messages }}"));
    CHECK(v.tok_score == nullptr);
    CHECK(v.merges == nullptr);
    f.close();
    ::unlink(p.c_str());
}

/* ------------------------------------------------------------------ append in place */
namespace {

RadWriter::Weight weight_of(const RadFile& f, const RadFileEntry& e) {
    RadWriter::Weight w;
    w.name      = f.str(e.name);
    w.quantizer = f.str(e.quantizer);
    w.options   = f.str(e.options);
    w.enc       = f.encoding(e);
    w.rank      = e.rank;
    for (uint32_t d = 0; d < RAD_MAX_RANK; ++d) w.shape[d] = e.shape[d];
    w.layer     = e.layer;
    w.expert    = e.expert;
    w.slot      = e.slot;
    return w;
}

/* Everything `old` holds, through add_existing, in its own order. */
void carry_over(RadWriter& w, const RadFile& old) {
    w.set_arch("radtest");
    w.set_model_name("format-test");
    w.set_quant("fp8_block");
    w.set_recipe(kRecipe);
    w.set_created_by("format_test, extended");
    w.meta_i("n_layers", 2);
    w.meta_s("note", "extended in place");
    for (int64_t i = 0; i < old.entry_count(); ++i)
        CHECK(w.add_existing(weight_of(old, old.entry(i)), old, old.entry(i)) >= 0);
    for (int64_t i = 0; i < old.profile_count(); ++i)
        w.add_profile(old.profile()[i].layer, old.profile()[i].expert, old.profile()[i].share);
}

}  /* namespace */

TEST(a_container_extended_in_place_keeps_every_old_weight_where_it_was) {
    Built b = write_model("append");
    const std::vector<uint8_t> before = slurp(b.path);
    REQUIRE(before.size() > sizeof(RadFileHeader));

    RadFile old;
    REQUIRE((old.open(b.path.c_str())) >= 0);
    const int64_t n_old = old.entry_count();
    std::vector<uint64_t> offs;
    std::vector<RadFilePlane> planes;
    for (int64_t i = 0; i < n_old; ++i) {
        offs.push_back(old.entry(i).offset);
        for (uint32_t k = 0; k < old.entry(i).n_planes; ++k)
            planes.push_back(old.plane(old.entry(i), (int)k));
    }

    RadWriter w;
    REQUIRE((w.begin_append(b.path.c_str(), old)) >= 0);
    carry_over(w, old);
    /* A carried-over weight's bytes are the old file's, and not the writer's to change. */
    const uint8_t one = 1;
    CHECK_EQ(w.write_plane(0, 0, 0, &one, 1), (int)RAD_E_INVAL);
    /* One weight in an encoding the container does not have yet, one in one it does. */
    const RadWriter::Weight wv = mkw("v.patch.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, 64),
                                     48, 64, -1, -1, 0);
    const Planes pv = planes_of(wv, 9);
    CHECK(w.add_weight(wv, pv.ptr.data()) >= 0);
    RadWriter::Weight wb = mkw("v.patch.bias", plain_bf16(), 1, 64, -1, -1, 0);
    wb.rank = 1;
    wb.shape[0] = 64;
    wb.shape[1] = 0;
    const Planes pb = planes_of(wb, 10);
    CHECK(w.add_weight(wb, pb.ptr.data()) >= 0);
    w.meta_s("radiance.encoder", "vision");
    REQUIRE((w.finish()) >= 0);
    old.close();

    /* Nothing between the header and the old end moved: the old tables are dead but intact,
     * and every old weight's bytes are where they were. */
    const std::vector<uint8_t> after = slurp(b.path);
    REQUIRE(after.size() > before.size());
    CHECK(std::memcmp(after.data() + sizeof(RadFileHeader), before.data() + sizeof(RadFileHeader),
                      before.size() - sizeof(RadFileHeader)) == 0);

    RadFile f;
    REQUIRE((f.open(b.path.c_str())) >= 0);
    CHECK_EQ(f.entry_count(), n_old + 2);
    size_t pi = 0;
    for (int64_t i = 0; i < n_old; ++i) {
        CHECK_EQ(f.entry(i).offset, offs[(size_t)i]);
        for (uint32_t k = 0; k < f.entry(i).n_planes; ++k, ++pi) {
            REQUIRE(pi < planes.size());
            CHECK_EQ(f.plane(f.entry(i), (int)k).offset, planes[pi].offset);
            CHECK_EQ(f.plane(f.entry(i), (int)k).bytes, planes[pi].bytes);
        }
    }
    CHECK_EQ(f.group_count(), (int64_t)2);
    const RadFile::ExpertGroup* g = f.group_for_layer(1);
    REQUIRE(g != nullptr);
    CHECK_EQ((int)g->n_expert, 4);
    const RadFileEntry* e0 = f.find("blk.1.ffn_gate.2");
    REQUIRE(e0 != nullptr);
    CHECK_EQ(std::string(f.str(e0->quantizer)), std::string("rtn"));
    CHECK(planes_match(f, *e0, weight_of(f, *e0), SEED_GATE));

    /* The old encodings, carried over, and the one new encoding beside them: the bias shares
     * the embedding's. */
    CHECK_EQ(f.encoding_count(), (int64_t)3);
    const RadFileEntry* ev = f.find("v.patch.weight");
    REQUIRE(ev != nullptr);
    CHECK((int64_t)ev->offset >= (int64_t)before.size());
    CHECK_EQ((int64_t)(ev->offset % RAD_ALIGN_UNIT), (int64_t)0);
    CHECK_EQ((int64_t)(f.plane(*ev, 1).offset % RAD_ALIGN_SUB), (int64_t)0);
    CHECK(planes_match(f, *ev, wv, 9));
    const RadFileEntry* eb = f.find("v.patch.bias");
    REQUIRE(eb != nullptr);
    CHECK_EQ(eb->enc, f.find("token_embd.weight")->enc);
    CHECK(planes_match(f, *eb, wb, 10));

    CHECK_EQ(std::string(f.get_s("note", "")), std::string("extended in place"));
    CHECK_EQ(std::string(f.get_s("radiance.encoder", "")), std::string("vision"));
    CHECK_EQ(std::string(f.str(f.header().created_by)), std::string("format_test, extended"));
    CHECK_EQ(std::string(f.str(f.header().recipe)), std::string(kRecipe));
    CHECK_EQ(f.profile_count(), (int64_t)8);
    CHECK_EQ((int64_t)(f.header().data_off % RAD_ALIGN_POOL), (int64_t)0);
    f.close();

    /* The way back: the saved header and length restore the file byte for byte. */
    const std::vector<uint8_t> bak = slurp(b.path + ".pre-append");
    REQUIRE_EQ(bak.size(), sizeof(RadFileHeader) + sizeof(int64_t));
    int64_t len = 0;
    std::memcpy(&len, bak.data() + sizeof(RadFileHeader), sizeof len);
    CHECK_EQ(len, (int64_t)before.size());
    const int fd = ::open(b.path.c_str(), O_WRONLY);
    REQUIRE(fd >= 0);
    CHECK_EQ(::pwrite(fd, bak.data(), sizeof(RadFileHeader), 0), (ssize_t)sizeof(RadFileHeader));
    CHECK_EQ(::ftruncate(fd, (off_t)len), 0);
    ::close(fd);
    CHECK(slurp(b.path) == before);

    ::unlink((b.path + ".pre-append").c_str());
    ::unlink(b.path.c_str());
}

TEST(an_in_place_extension_that_fails_leaves_the_container_as_it_was) {
    Built b = write_model("append_fail");
    const std::vector<uint8_t> before = slurp(b.path);

    /* A new expert cannot join its layer from past the end, and the refusal cuts the file back. */
    {
        RadFile old;
        REQUIRE((old.open(b.path.c_str())) >= 0);
        RadWriter w;
        REQUIRE((w.begin_append(b.path.c_str(), old)) >= 0);
        carry_over(w, old);
        const RadWriter::Weight wv = mkw("v.patch.weight", plain_bf16(), 64, 64, -1, -1, 0);
        const Planes pv = planes_of(wv, 9);
        CHECK(w.add_weight(wv, pv.ptr.data()) >= 0);
        const RadWriter::Weight wg = mkw("blk.1.ffn_gate.4", int4_g32(), EXP_ROWS, EXP_COLS, 1,
                                         4, 0);
        const Planes pg = planes_of(wg, SEED_GATE);
        CHECK(w.add_weight(wg, pg.ptr.data()) < 0);
        w.abort();
    }
    CHECK(slurp(b.path) == before);

    /* Another process holding the file: refused before a byte is written. */
    int ready[2];
    REQUIRE_EQ(::pipe(ready), 0);
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        const int fd = ::open(b.path.c_str(), O_RDONLY);
        char c = fd >= 0 ? 'y' : 'n';
        (void)!::write(ready[1], &c, 1);
        ::pause();
        ::_exit(0);
    }
    char c = 0;
    CHECK_EQ(::read(ready[0], &c, 1), (ssize_t)1);
    CHECK_EQ(c, 'y');
    {
        RadFile old;
        REQUIRE((old.open(b.path.c_str())) >= 0);
        RadWriter w;
        CHECK_EQ(w.begin_append(b.path.c_str(), old), (int)RAD_E_STATE);
    }
    ::kill(child, SIGKILL);
    ::waitpid(child, nullptr, 0);
    ::close(ready[0]);
    ::close(ready[1]);
    CHECK(slurp(b.path) == before);

    /* And with it gone, the same file extends. */
    {
        RadFile old;
        REQUIRE((old.open(b.path.c_str())) >= 0);
        RadWriter w;
        CHECK_OK(w.begin_append(b.path.c_str(), old));
        w.abort();
    }
    CHECK(slurp(b.path) == before);
    ::unlink(b.path.c_str());
}

/* ------------------------------------------------------------------ refusals */
TEST(truncated_file_is_refused_not_crashed) {
    Built b = write_model("trunc");

    /* Cut the file to a third: the header still claims the full length and every table offset
     * still points past the end. Nothing may be dereferenced. */
    FILE* fp = std::fopen(b.path.c_str(), "rb");
    CHECK(fp != nullptr);
    std::fseek(fp, 0, SEEK_END);
    const long full = std::ftell(fp);
    std::fclose(fp);

    const std::string cut = b.path + ".cut";
    for (long keep : { (long)16, full / 3, full - 4096 }) {
        FILE* in = std::fopen(b.path.c_str(), "rb");
        FILE* outf = std::fopen(cut.c_str(), "wb");
        CHECK(in && outf);
        std::vector<uint8_t> buf((size_t)keep);
        CHECK_EQ((long)std::fread(buf.data(), 1, (size_t)keep, in), keep);
        std::fwrite(buf.data(), 1, (size_t)keep, outf);
        std::fclose(in);
        std::fclose(outf);

        RadFile f;
        const int s = f.open(cut.c_str());
        CHECK_EQ(s, (int)RAD_E_FORMAT);
        CHECK(!f.is_open());
    }
    ::unlink(cut.c_str());

    /* Not a .rad at all. */
    const std::string junk = b.path + ".junk";
    FILE* j = std::fopen(junk.c_str(), "wb");
    CHECK(j != nullptr);
    const char noise[512] = { 'n', 'o', 't', 'r', 'a', 'd' };
    std::fwrite(noise, 1, sizeof noise, j);
    std::fclose(j);
    RadFile f;
    CHECK_EQ(f.open(junk.c_str()), (int)RAD_E_FORMAT);
    ::unlink(junk.c_str());

    /* A path that is not there is RAD_E_IO, not RAD_E_FORMAT: the two are different problems and
     * the operator has to be told which. */
    CHECK_EQ(f.open((b.path + ".absent").c_str()), (int)RAD_E_IO);

    ::unlink(b.path.c_str());
}

TEST(a_container_of_another_version_is_refused_with_the_way_out) {
    /* One version is read. A file of any other is not a different dialect to be guessed at: it
     * is refused, and the message names the tool that makes a readable one. */
    Built b = write_model("version");
    for (uint32_t v : { 1u, RAD_FORMAT_VER + 1 }) {
        const std::string bad = patched(b.path, "ver", [&](std::vector<uint8_t>& all,
                                                           const RadFileHeader& h) {
            RadFileHeader x = h;
            x.version = v;
            put(all, 0, x);
        });
        RadFile f;
        int s = RAD_OK;
        const std::string log = stderr_of([&] { s = f.open(bad.c_str()); });
        CHECK_EQ(s, (int)RAD_E_FORMAT);
        CHECK(!f.is_open());
        CHECK(log.find("rad-convert") != std::string::npos);
        ::unlink(bad.c_str());
    }
    ::unlink(b.path.c_str());
}

TEST(a_table_count_that_overflows_its_span_is_refused) {
    /* PINS: a table count that cannot be trusted as a byte span is still refused. Checking a table
     * as `count * sizeof(element)` is a uint64 product of a count the file controls, and every
     * element type here has an EVEN size, so a count of 2^63 makes the product exactly 2^64 --
     * zero -- which a span check reads as an empty table and waves through. The loops that follow
     * iterate the COUNT, not the byte span, so a validate that accepts this walks 2^63 rows off
     * the end of the mapping, inside the very function whose contract (radfile.h) is that a
     * corrupt file is RAD_E_FORMAT and not a SIGSEGV three components later. Each table gets its
     * own copy of the file: one bad count at a time, so a refusal here is attributable to the
     * field under test. */
    Built b = write_model("ovf");
    for (int which = 0; which < 5; ++which) {
        const std::string bad = patched(b.path, "ovf", [&](std::vector<uint8_t>& all,
                                                           const RadFileHeader& h) {
            RadFileHeader x = h;
            const uint64_t wrap = 1ULL << 63;
            switch (which) {
                case 0: x.meta_count  = wrap; break;
                case 1: x.dir_count   = wrap; break;
                case 2: x.plane_count = wrap; break;
                case 3: x.enc_count   = wrap; break;
                default: x.prof_count = wrap; break;
            }
            put(all, 0, x);
        });
        RadFile f;
        CHECK_EQ(f.open(bad.c_str()), (int)RAD_E_FORMAT);
        CHECK(!f.is_open());
        ::unlink(bad.c_str());
    }
    ::unlink(b.path.c_str());
}

TEST(an_entry_whose_planes_are_not_its_encodings_is_refused) {
    /* A reader slices a plane by arithmetic and hands it to a kernel's relayout as exactly what
     * the encoding says it is. Every way the directory, the plane table and the encoding table can
     * disagree is refused at open -- each on its own copy of a well-formed file, so the refusal is
     * the edit's and not some other damage's. Entry 0 is the bf16 embedding (one plane); entry 3
     * is layer 0's first expert gate (codes at +0, scale at +1280). */
    Built b = write_model("planes");
    {
        RadFile ok;
        REQUIRE(ok.open(b.path.c_str()) >= 0);
        REQUIRE(ok.encoding(ok.entry(3)).n_planes == 2);
        REQUIRE(ok.entry(3).expert == 0);
    }

    auto entry = [](const std::vector<uint8_t>& all, const RadFileHeader& h, uint64_t i) {
        return get<RadFileEntry>(all, h.dir_off + i * sizeof(RadFileEntry));
    };
    auto set_entry = [](std::vector<uint8_t>& all, const RadFileHeader& h, uint64_t i,
                        const RadFileEntry& e) {
        put(all, h.dir_off + i * sizeof(RadFileEntry), e);
    };
    auto plane_at = [](const RadFileHeader& h, uint64_t i) {
        return h.plane_off + i * sizeof(RadFilePlane);
    };

    struct Case { const char* what; Edit edit; };
    const Case cases[] = {
        { "an encoding index past the table", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = entry(all, h, 0);
              e.enc = (uint32_t)h.enc_count;
              set_entry(all, h, 0, e);
          } },
        { "a plane count that is not its encoding's", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = entry(all, h, 0);
              e.n_planes = 2;
              set_entry(all, h, 0, e);
          } },
        { "planes running off the plane table", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = entry(all, h, 3);
              e.plane_first = h.plane_count - 1;
              set_entry(all, h, 3, e);
          } },
        { "a plane index that wraps", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = entry(all, h, 0);
              e.plane_first = ~0ULL;
              set_entry(all, h, 0, e);
          } },
        { "a plane two bytes short of its encoding", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              const RadFileEntry e = entry(all, h, 0);
              RadFilePlane p = get<RadFilePlane>(all, plane_at(h, e.plane_first));
              p.bytes -= 2;
              put(all, plane_at(h, e.plane_first), p);
          } },
        /* -16 rather than +16: the scale plane then still ends inside its entry and after the
         * codes, so the alignment is the only thing wrong with it. */
        { "a plane off its 256-byte boundary", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              const RadFileEntry e = entry(all, h, 3);
              RadFilePlane p = get<RadFilePlane>(all, plane_at(h, e.plane_first + 1));
              p.offset -= 16;
              put(all, plane_at(h, e.plane_first + 1), p);
          } },
        { "a plane before its own entry", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              const RadFileEntry e0 = entry(all, h, 0), e1 = entry(all, h, 1);
              put(all, plane_at(h, e1.plane_first), get<RadFilePlane>(all, plane_at(h, e0.plane_first)));
          } },
        { "an encoding with no planes", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadEncoding x = get<RadEncoding>(all, h.enc_off);
              x.n_planes = 0;
              put(all, h.enc_off, x);
          } },
        { "an encoding with a dtype the core cannot size", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadEncoding x = get<RadEncoding>(all, h.enc_off);
              x.plane[0].dtype = RAD_DT_PLUGIN_BASE + 1;
              put(all, h.enc_off, x);
          } },
        /* A scheme the core does not define is read as a quantiser's own, which rad_enc_valid()
         * accepts; only the missing terminator is wrong with it. */
        { "an unterminated scheme", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadEncoding x = get<RadEncoding>(all, h.enc_off);
              std::memset(x.scheme, 'q', sizeof x.scheme);
              put(all, h.enc_off, x);
          } },
        { "an entry off its 4 KiB boundary", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = entry(all, h, 1);
              e.offset -= 256;
              set_entry(all, h, 1, e);
          } },
    };

    for (const Case& c : cases) {
        const std::string bad = patched(b.path, "planes", c.edit);
        RadFile f;
        const int s = f.open(bad.c_str());
        if (s != RAD_E_FORMAT) std::fprintf(stderr, "    not refused: %s\n", c.what);
        CHECK_EQ(s, (int)RAD_E_FORMAT);
        CHECK(!f.is_open());
        ::unlink(bad.c_str());
    }
    ::unlink(b.path.c_str());
}

TEST(a_layer_with_no_fixed_stride_is_refused) {
    /* Experts within a layer must share a shape and a format, or `base + id*stride` is a lie and
     * the hot path -- which does that arithmetic with no bounds test -- reads the wrong bytes.
     * Both halves refuse it, and they refuse for different reasons worth stating. */
    const std::string p = tmp_path("nostride");
    const RadWriter::Weight small = mkw("e", plain_bf16(), 32, 64, 0, 0, 0);
    const Planes ps = planes_of(small, 1);

    /* (a) THE WRITER, at the weight that breaks it -- naming the weight, while the operator still
     * has the checkpoint open and can fix the manifest. Expert 0 defines the shape of a unit and
     * expert 1 does not match it slot for slot.
     *
     * The size check is not redundant with the offset check above it. An oversized expert still
     * lands at the right base and displaces only the NEXT one, so an oversized LAST expert passes
     * every offset test and still leaves a group whose base + id*stride runs off the end. */
    {
        RadWriter w;
        CHECK_OK(w.begin(p.c_str()));
        w.set_arch("radtest");
        CHECK(w.add_weight(mkw("e0", plain_bf16(), 32, 64, 0, 0, 0), ps.ptr.data()) >= 0);
        const RadWriter::Weight big = mkw("e1", plain_bf16(), 128, 64, 0, 1, 0);
        CHECK(w.add_weight(big, planes_of(big, 2).ptr.data()) < 0);
        w.abort();
    }

    /* (b) Out of order, which no stride addresses at all. Refused on the first weight. */
    {
        RadWriter w;
        CHECK_OK(w.begin(p.c_str()));
        w.set_arch("radtest");
        CHECK(w.add_weight(mkw("e1", plain_bf16(), 32, 64, 0, 1, 0), ps.ptr.data()) < 0);
        w.abort();
    }

    /* (c) THE READER, on a container RadWriter did not produce. The writer's checks bind only
     * what the writer wrote; a .rad arrives from wherever the operator got it, and the hot path
     * does `base + id*stride` with no bounds test. So the reader RE-DERIVES the stride at open
     * from the directory it was handed and refuses a group that has none -- which is why the
     * check exists on both sides rather than one.
     *
     * Written as a well-formed file whose DIRECTORY is then edited, because that is the shape of
     * the thing being defended against: bytes that parse, with an entry that lies. Expert 1 and
     * its plane move one 4 KiB unit along, onto expert 2: every per-entry check still holds --
     * aligned, inside the file, its plane its encoding's size inside it -- and the three experts
     * now sit at base, base + 8K and base + 8K, which no single stride describes. */
    {
        RadWriter w;
        CHECK_OK(w.begin(p.c_str()));
        w.set_arch("radtest");
        for (int e = 0; e < 3; ++e) {
            /* DISTINCT NAMES. A weight is stored once per name, so three experts called "e" are
             * refused by that rule before the stride is ever reached -- and this case is about the
             * stride. The names carry the expert id for the same reason the diagnostic does. */
            const std::string nm = "e" + std::to_string(e);
            CHECK(w.add_weight(mkw(nm.c_str(), plain_bf16(), 32, 64, 0, e, 0), ps.ptr.data()) >= 0);
        }
        CHECK_OK(w.finish());

        /* It opens as written. */
        { RadFile f; CHECK_OK(f.open(p.c_str())); }

        const std::string bad = patched(p, "moved", [&](std::vector<uint8_t>& all,
                                                        const RadFileHeader& h) {
            const uint64_t at = h.dir_off + sizeof(RadFileEntry);
            RadFileEntry e = get<RadFileEntry>(all, at);
            e.offset += RAD_ALIGN_UNIT;
            put(all, at, e);
            const uint64_t pat = h.plane_off + e.plane_first * sizeof(RadFilePlane);
            RadFilePlane pl = get<RadFilePlane>(all, pat);
            pl.offset += RAD_ALIGN_UNIT;
            put(all, pat, pl);
        });
        RadFile f;
        CHECK_EQ(f.open(bad.c_str()), (int)RAD_E_FORMAT);
        CHECK(!f.is_open());
        ::unlink(bad.c_str());
    }
    ::unlink(p.c_str());
}

TEST(empty_container_still_opens) {
    /* A container with no weights is legal -- a vocab-only or metadata-only file -- and must not
     * be a special case anywhere. Nothing about the layout should depend on there being data. */
    const std::string p = tmp_path("empty");
    RadWriter w;
    CHECK_OK(w.begin(p.c_str()));
    w.set_arch("radtest");
    w.meta_i("n_layers", 0);
    CHECK_OK(w.finish());

    RadFile f;
    CHECK_OK(f.open(p.c_str()));
    CHECK_EQ(f.entry_count(), (int64_t)0);
    CHECK_EQ(f.encoding_count(), (int64_t)0);
    CHECK_EQ(f.group_count(), (int64_t)0);
    CHECK(!f.vocab());
    CHECK_EQ(f.profile_count(), (int64_t)0);
    int64_t n = -1;
    CHECK(f.get_i("n_layers", &n));
    CHECK_EQ(n, (int64_t)0);
    f.close();
    ::unlink(p.c_str());
}

/* ------------------------------------------------------------------ the tuning cache */
TEST(tune_cache_round_trip) {
    const std::string home = tmp_path("home");
    ::mkdir(home.c_str(), 0755);

    TuneCache c;
    c.set_machine("gfx1201-x2-test");

    Geometry g;
    g.set_i("M", 64);
    g.set_i("N", 8192);
    g.set_i("K", 5120);
    g.set_s("dtype", "w4a8");
    const std::string key = TuneCache::canon_geometry(g);
    /* Sorted by key, so the same shape spells the same however the plugin ordered its params. */
    CHECK_EQ(key, std::string("K=5120 M=64 N=8192 dtype=w4a8"));

    Geometry g2;
    g2.set_s("dtype", "w4a8");
    g2.set_i("K", 5120);
    g2.set_i("N", 8192);
    g2.set_i("M", 64);
    CHECK_EQ(TuneCache::canon_geometry(g2), key);

    TuneRow r;
    r.kernel = "gemm_w4a8_nt_m64";
    r.plugin = "libr4d";
    r.choice = "bm=128,bn=64,stages=3";
    r.geom = key;
    r.us = 41.203;
    r.measured = TuneCache::now_iso8601();
    c.put(r);

    TuneRow r2 = r;
    r2.choice = "";             /* a row with no declared axes */
    r2.geom = "K=1 M=1 N=1";
    r2.us = 1.5;
    c.put(r2);

    CHECK_OK(c.save(home));

    TuneCache back;
    CHECK_OK(back.load(home, "gfx1201-x2-test"));
    CHECK_EQ(back.rows().size(), (size_t)2);

    const TuneRow* got = back.find("gemm_w4a8_nt_m64", "libr4d", key);
    CHECK(got != nullptr);
    if (got) {
        CHECK_EQ(got->choice, std::string("bm=128,bn=64,stages=3"));
        CHECK_NEAR(got->us, 41.203, 1e-6);
        CHECK_EQ(got->measured, r.measured);
    }
    /* A row that is not there is absent, not a default. */
    CHECK(back.find("gemm_w4a8_nt_m64", "libr4d", "K=1") == nullptr);
    CHECK(back.find("nope", "libr4d", key) == nullptr);

    /* A missing cache is the normal state of a fresh install, not a failure. */
    TuneCache none;
    CHECK_OK(none.load(home, "no-such-machine"));
    CHECK(none.empty());

    ::unlink(TuneCache::path_for(home, "gfx1201-x2-test").c_str());
    ::rmdir((home + "/tune").c_str());
    ::rmdir(home.c_str());
}

/* A hand-edited cache is read at every start, so a line of nothing but blanks -- before the magic
 * line or among the rows -- is a blank line and not a malformed row or a refused file. */
TEST(a_tune_cache_line_of_blanks_is_a_blank_line) {
    const std::string home = tmp_path("home_blank");
    ::mkdir(home.c_str(), 0755);
    ::mkdir((home + "/tune").c_str(), 0755);
    const std::string path = TuneCache::path_for(home, "blank-test");
    const char* bodies[] = {
        /* among the rows */
        "radtune 2\n"
        "   \n"
        "machine blank-test\n"
        "\t\r\n"
        "gemm_x libr4d mb=2 3.5 2026-01-01T00:00:00Z K=1 M=1 N=1\n"
        "  \n",
        /* ahead of the magic line */
        " \t \n"
        "radtune 2\n"
        "gemm_x libr4d mb=2 3.5 2026-01-01T00:00:00Z K=1 M=1 N=1\n",
    };
    for (const char* body : bodies) {
        FILE* f = std::fopen(path.c_str(), "wb");
        CHECK(f != nullptr);
        if (!f) return;
        std::fputs(body, f);
        std::fclose(f);

        TuneCache c;
        CHECK_OK(c.load(home, "blank-test"));
        CHECK_EQ(c.rows().size(), (size_t)1);
        CHECK(c.find("gemm_x", "libr4d", "K=1 M=1 N=1") != nullptr);
    }

    ::unlink(path.c_str());
    ::rmdir((home + "/tune").c_str());
    ::rmdir(home.c_str());
}

/* ------------------------------------------------------------------ the safetensors reader */

/* A CHECKPOINT NAME IS MATCHED, NOT DISPLAYED, so how it is decoded decides whether a weight is
 * found at all. JSON cannot write a codepoint above U+FFFF except as a surrogate pair, and a
 * reader that encodes the two halves separately produces CESU-8 -- six bytes that round-trip
 * through itself and compare unequal to the four bytes the same name has everywhere else. The
 * failure is a lookup miss on one tensor, reported as a missing weight.
 *
 * The other three cases are here because they are the boundaries either side of it: the last
 * two-byte codepoint, the first three-byte one, and an unpaired high surrogate, which is malformed
 * input this reader passes through rather than rejecting -- a name is data to be matched. */
static std::string st_write(const std::string& header_json) {
    std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                       "/rad_st_test.safetensors";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return std::string();
    uint64_t n = header_json.size();
    std::fwrite(&n, 8, 1, f);
    std::fwrite(header_json.data(), 1, header_json.size(), f);
    const char pad[4] = { 0, 0, 0, 0 };            /* one f32 of data section */
    std::fwrite(pad, 1, 4, f);
    std::fclose(f);
    return path;
}

TEST(safetensors_decodes_a_surrogate_pair_as_one_codepoint) {
    /* U+1F600, written the only way JSON can write it. */
    const std::string path = st_write(
        "{\"a\\uD83D\\uDE00b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}");
    CHECK(!path.empty());
    if (path.empty()) return;

    SafeTensorsFile st;
    CHECK_OK(st.open(path));
    CHECK_EQ(st.tensors().size(), (size_t)1);
    if (st.tensors().size() == 1) {
        const std::string& got = st.tensors()[0].name;
        CHECK_EQ(got, std::string("a\xF0\x9F\x98\x80" "b"));
        CHECK_EQ(got.size(), (size_t)6);           /* 'a' + four bytes + 'b', not 'a' + six + 'b' */
        CHECK(st.find("a\xF0\x9F\x98\x80" "b") != nullptr);
    }
    st.close();
    ::unlink(path.c_str());
}

TEST(safetensors_decodes_the_escape_widths_either_side_of_it) {
    const std::string path = st_write(
        "{\"\\u007F\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
        " \"\\u07FF\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
        " \"\\u0800\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
        " \"\\uD800x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}");
    CHECK(!path.empty());
    if (path.empty()) return;

    SafeTensorsFile st;
    CHECK_OK(st.open(path));
    CHECK(st.find("\x7F") != nullptr);                     /* one byte */
    CHECK(st.find("\xDF\xBF") != nullptr);                 /* two */
    CHECK(st.find("\xE0\xA0\x80") != nullptr);             /* three */
    CHECK(st.find("\xED\xA0\x80" "x") != nullptr);         /* lone surrogate, passed through */
    st.close();
    ::unlink(path.c_str());
}

/* Every reader walks a tensor by shape and dtype, so a shape that disagrees with the span its
 * data_offsets give is refused at open. The data section st_write lays down is one f32. */
TEST(safetensors_refuses_a_shape_that_is_not_its_span) {
    const char* bad[] = {
        /* two f32 claimed over four bytes */
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}",
        /* 2^62 x 4 elements, a product that wraps to zero */
        "{\"t\":{\"dtype\":\"U8\",\"shape\":[4611686018427387904,4],\"data_offsets\":[0,0]}}",
        /* a negative extent */
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[-1,-1],\"data_offsets\":[0,4]}}",
        /* fewer elements than the span */
        "{\"t\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,4]}}",
    };
    for (const char* h : bad) {
        const std::string path = st_write(h);
        CHECK(!path.empty());
        if (path.empty()) return;
        SafeTensorsFile st;
        CHECK_EQ(st.open(path), RAD_E_FORMAT);
        ::unlink(path.c_str());
    }

    const char* good[] = {
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]}}",        /* a scalar */
        "{\"t\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[0,4]}}",
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[0,7],\"data_offsets\":[4,4]}}",     /* empty */
        /* a dtype this reader cannot size is left to its consumers, which refuse it by name */
        "{\"t\":{\"dtype\":\"F8_E8M0\",\"shape\":[9],\"data_offsets\":[0,4]}}",
    };
    for (const char* h : good) {
        const std::string path = st_write(h);
        CHECK(!path.empty());
        if (path.empty()) return;
        SafeTensorsFile st;
        CHECK_OK(st.open(path));
        CHECK_EQ(st.tensors().size(), (size_t)1);
        ::unlink(path.c_str());
    }
}


/* ================================================================== the shard index
 *
 * A SHARDED CHECKPOINT'S weight_map IS THE ONLY THING THAT SAYS WHICH FILE A TENSOR IS IN, and
 * rad-convert reads it before it opens anything. The failure it has to avoid is not a crash: an
 * index that parsed to FEWER entries than it holds makes the converter report a tensor as missing
 * from a checkpoint that contains it, which reads as a broken download.
 *
 * Absent is not an error -- a single-file checkpoint has no index and the caller falls back to
 * model.safetensors -- so "not found" and "malformed" have to be different answers.
 */
namespace {

std::string write_index(const char* stem, const std::string& body) {
    const std::string p = tmp_path(stem);
    FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) return std::string();
    std::fwrite(body.data(), 1, body.size(), f);
    std::fclose(f);
    return p;
}

}  /* namespace */

TEST(the_shard_index_reads_every_entry_of_a_real_weight_map) {
    /* The shape HuggingFace writes: a metadata block first, then the map. */
    const std::string p = write_index("index_ok", R"({
  "metadata": { "total_size": 123456789 },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
    "model.layers.47.mlp.down_proj.weight": "model-00003-of-00003.safetensors",
    "lm_head.weight": "model-00003-of-00003.safetensors"
  }
})");
    CHECK(!p.empty());
    std::unordered_map<std::string, std::string> m;
    CHECK_OK(st_load_index(p, &m));
    CHECK_EQ(m.size(), 4u);
    CHECK_EQ(m["model.embed_tokens.weight"], std::string("model-00001-of-00003.safetensors"));
    CHECK_EQ(m["lm_head.weight"], std::string("model-00003-of-00003.safetensors"));
    CHECK_EQ(m.count("model.layers.47.mlp.down_proj.weight"), 1u);
    std::remove(p.c_str());

    /* The out map is CLEARED first: a second read that failed halfway must not leave the previous
       checkpoint's shards behind for the converter to open. */
    const std::string q = write_index("index_one", R"({"weight_map":{"a":"s0.safetensors"}})");
    CHECK_OK(st_load_index(q, &m));
    CHECK_EQ(m.size(), 1u);
    CHECK_EQ(m.count("lm_head.weight"), 0u);
    std::remove(q.c_str());

    /* An empty map is a valid index that names nothing, not a parse failure. */
    const std::string e = write_index("index_empty", R"({"weight_map":{}})");
    CHECK_OK(st_load_index(e, &m));
    CHECK_EQ(m.size(), 0u);
    std::remove(e.c_str());
}

TEST(a_missing_index_is_not_found_and_a_broken_one_is_a_format_error) {
    std::unordered_map<std::string, std::string> m;
    /* NOT FOUND IS THE ORDINARY CASE: a single-file checkpoint has no index, and the caller
       distinguishes it from a corrupt one to decide between falling back and refusing. */
    CHECK_EQ(st_load_index(tmp_path("does_not_exist"), &m), RAD_E_NOTFOUND);

    const std::string a = write_index("index_nomap", R"({"metadata":{"total_size":1}})");
    CHECK_EQ(st_load_index(a, &m), RAD_E_FORMAT);
    std::remove(a.c_str());

    const std::string b = write_index("index_notobj", R"({"weight_map": "a string"})");
    CHECK_EQ(st_load_index(b, &m), RAD_E_FORMAT);
    std::remove(b.c_str());

    const std::string c = write_index("index_trunc", R"({"weight_map":{"a":)");
    CHECK_EQ(st_load_index(c, &m), RAD_E_FORMAT);
    std::remove(c.c_str());

    const std::string d = write_index("index_empty_file", "");
    CHECK_EQ(st_load_index(d, &m), RAD_E_FORMAT);
    std::remove(d.c_str());

    /* A null out pointer is refused rather than dereferenced. */
    CHECK_EQ(st_load_index(tmp_path("whatever"), nullptr), RAD_E_INVAL);
}

/* ESCAPES IN A TENSOR NAME ARE RARE AND THE CONVERTER STILL HAS TO GET THEM RIGHT: a name read
 * with its backslashes intact does not match the name the safetensors header carries, and the
 * tensor is reported missing from a file that has it. */
TEST(the_shard_index_decodes_escaped_names) {
    const std::string p = write_index("index_esc",
        "{\"weight_map\":{\"a\\/b.weight\":\"s0.safetensors\","
        "\"q\\\"uote\":\"s1.safetensors\"}}");
    std::unordered_map<std::string, std::string> m;
    CHECK_OK(st_load_index(p, &m));
    CHECK_EQ(m.size(), 2u);
    CHECK_EQ(m.count("a/b.weight"), 1u);
    CHECK_EQ(m.count("q\"uote"), 1u);
    std::remove(p.c_str());
}

/* ================================================================== the importance matrix
 *
 * AN IMATRIX IS A FILE SOMEBODY ELSE HANDED YOU, and the quantiser that consumes it then runs
 * unattended for an hour. Both readers walk a length-prefixed format out of a mapping, so a
 * malformed one is an out-of-bounds read rather than a wrong answer -- and a misparse that stays
 * in bounds is worse, because the only symptom is slightly worse quantisation quality, which
 * nothing measures per tensor.
 *
 * Two formats because llama.cpp has written two, and the per-expert dimension exists in only one
 * of them -- so the join between `<name>.in_sum2` and `<name>.counts` is the part a MoE
 * quantisation depends on and the part a legacy file cannot express.
 */
namespace {

/* A little-endian byte builder. Both formats are length-prefixed binary, so a test that writes
 * one has to write the lengths too -- which is most of the point, since that is what the readers
 * are walking. */
struct Bin {
    std::vector<uint8_t> b;
    void raw(const void* p, size_t n) {
        const uint8_t* q = (const uint8_t*)p;
        b.insert(b.end(), q, q + n);
    }
    void i32(int32_t v)  { raw(&v, 4); }
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void f32(float v)    { raw(&v, 4); }
    void gstr(const std::string& s) { u64(s.size()); raw(s.data(), s.size()); }
    void lstr(const std::string& s) { i32((int32_t)s.size()); raw(s.data(), s.size()); }
    void pad_to(size_t a) { while (b.size() % a) b.push_back(0); }
};

std::string write_bin(const char* leaf, const std::vector<uint8_t>& bytes) {
    const char* dir = std::getenv("TMPDIR");
    const std::string path =
        std::string(dir && *dir ? dir : "/tmp") + "/radiance_imatrix_test_" + leaf;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return {};
    if (!bytes.empty()) std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    return path;
}

/* One legacy entry: name, ncall, and the raw sums. */
struct LegacyEntry { std::string name; int32_t ncall; std::vector<float> vals; };

std::vector<uint8_t> legacy_bytes(const std::vector<LegacyEntry>& es) {
    Bin w;
    w.i32((int32_t)es.size());
    for (const LegacyEntry& e : es) {
        w.lstr(e.name);
        w.i32(e.ncall);
        w.i32((int32_t)e.vals.size());
        for (float v : e.vals) w.f32(v);
    }
    return w.b;
}

/* One GGUF tensor: its name, its dims (dim 0 fastest) and its payload. Offsets and the data
 * section are computed here so a case only says what it means. */
struct GgufTensor { std::string name; std::vector<uint64_t> dims; std::vector<float> data; };

struct GgufKv { std::string key; uint32_t type; uint32_t u; std::string s; };

std::vector<uint8_t> gguf_bytes(const std::vector<GgufTensor>& ts,
                                const std::vector<GgufKv>& kvs,
                                uint32_t version = 3, uint64_t alignment = 32) {
    Bin w;
    w.raw("GGUF", 4);
    w.u32(version);
    w.u64(ts.size());
    w.u64(kvs.size());
    for (const GgufKv& kv : kvs) {
        w.gstr(kv.key);
        w.u32(kv.type);
        if (kv.type == 4)      w.u32(kv.u);                        /* u32 */
        else if (kv.type == 8) w.gstr(kv.s);                       /* string */
        else if (kv.type == 9) { w.u32(8); w.u64(1); w.gstr(kv.s); }  /* array of one string */
        else if (kv.type == 12) { double d = (double)kv.u; w.raw(&d, 8); }  /* f64 */
    }
    uint64_t off = 0;
    for (const GgufTensor& t : ts) {
        w.gstr(t.name);
        w.u32((uint32_t)t.dims.size());
        for (uint64_t d : t.dims) w.u64(d);
        w.u32(0);                                 /* GGML_TYPE_F32 */
        w.u64(off);
        off += t.data.size() * sizeof(float);
    }
    w.pad_to((size_t)alignment);
    for (const GgufTensor& t : ts) for (float v : t.data) w.f32(v);
    return w.b;
}

/* One float of an entry's payload. A GGUF payload is pointed at in the mapping at whatever offset
 * the file's alignment gave it, so it is read the way the reader itself reads it: by memcpy. */
float f32_at(const float* p, size_t i) {
    float v = 0.0f;
    std::memcpy(&v, (const char*)p + i * sizeof(float), sizeof v);
    return v;
}

double mean_of(const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += x;
    return v.empty() ? 0.0 : s / (double)v.size();
}

}  /* namespace */

TEST(a_legacy_imatrix_reads_every_entry_it_declares) {
    const std::string path = write_bin("legacy.imatrix", legacy_bytes({
        { "blk.0.attn_q.weight",    10, { 1.0f, 2.0f, 3.0f, 4.0f } },
        { "blk.0.ffn_down.weight",   4, { 8.0f, 8.0f } },
    }));
    CHECK(!path.empty());

    ImatrixFile im;
    CHECK_OK(im.open(path));
    CHECK(!im.is_gguf());
    CHECK_EQ(im.entries().size(), (size_t)2);
    CHECK_EQ(im.path(), path);

    const ImatrixEntry* e = im.find("blk.0.attn_q.weight");
    CHECK(e != nullptr);
    if (e) {
        CHECK_EQ(e->cols, (int64_t)4);
        CHECK_EQ(e->ncall, (int64_t)10);
        /* The legacy format has no per-expert dimension at all, so every tensor is dense and
         * every expert in a layer is weighted by the layer's aggregate. */
        CHECK_EQ(e->n_expert, (int64_t)1);
        CHECK(e->counts == nullptr);
        CHECK(e->sum2 != nullptr);
        if (e->sum2) CHECK_EQ(f32_at(e->sum2, 2), 3.0f);
    }
    CHECK(im.find("blk.0.ffn_up.weight") == nullptr);
    std::remove(path.c_str());
}

/* NORMALISED TO MEAN 1 is the contract, and it is what makes a weighted quantisation error
 * comparable with an unweighted one -- so the quantiser's error report means the same thing with
 * and without an imatrix. */
TEST(importance_is_normalised_to_mean_one) {
    const std::string path = write_bin("norm.imatrix", legacy_bytes({
        { "t", 7, { 1.0f, 3.0f, 4.0f, 8.0f } },
    }));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    std::vector<float> w;
    CHECK(im.importance("t", 0, &w));
    CHECK_EQ(w.size(), (size_t)4);
    CHECK_NEAR(mean_of(w), 1.0, 1e-5);
    /* The shape of the row survives: column 3 is eight times column 0. */
    if (w.size() == 4) CHECK_NEAR(w[3] / w[0], 8.0, 1e-4);
    std::remove(path.c_str());
}

/* A UNIFORM SCALE OF THE ROW CANNOT CHANGE THE ANSWER, because mean-1 divides it out. This is
 * the property the per-expert division relies on being harmless, and it is also why an imatrix
 * collected over ten times the calibration set weights a tensor identically. */
TEST(importance_is_invariant_to_a_uniform_scale_of_the_row) {
    const std::string a = write_bin("scale_a.imatrix", legacy_bytes({
        { "t", 1, { 1.0f, 2.0f, 5.0f, 8.0f } },
    }));
    const std::string b = write_bin("scale_b.imatrix", legacy_bytes({
        { "t", 1000, { 1000.0f, 2000.0f, 5000.0f, 8000.0f } },
    }));
    ImatrixFile ia, ib;
    CHECK_OK(ia.open(a));
    CHECK_OK(ib.open(b));
    std::vector<float> wa, wb;
    CHECK(ia.importance("t", 0, &wa));
    CHECK(ib.importance("t", 0, &wb));
    CHECK_EQ(wa.size(), wb.size());
    for (size_t i = 0; i < wa.size() && i < wb.size(); ++i) CHECK_NEAR(wa[i], wb[i], 1e-4);
    std::remove(a.c_str());
    std::remove(b.c_str());
}

/* A tensor the calibration set never activated has an all-zero row. Weighting by it is a
 * division by zero, so it falls back to uniform -- which is the same thing as quantising that
 * tensor without an imatrix, and is the only answer that is not NaN. */
TEST(a_degenerate_row_falls_back_to_uniform_weights) {
    const std::string path = write_bin("zero.imatrix", legacy_bytes({
        { "t", 3, { 0.0f, 0.0f, 0.0f } },
    }));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    std::vector<float> w;
    CHECK(im.importance("t", 0, &w));
    CHECK_EQ(w.size(), (size_t)3);
    for (float x : w) CHECK_EQ(x, 1.0f);
    std::remove(path.c_str());
}

TEST(importance_declines_a_tensor_or_an_expert_it_does_not_have) {
    const std::string path = write_bin("absent.imatrix", legacy_bytes({
        { "t", 1, { 1.0f, 2.0f } },
    }));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    std::vector<float> w;
    CHECK(!im.importance("not_here", 0, &w));
    CHECK(!im.importance("t", 1, &w));      /* the legacy format has one expert */
    CHECK(!im.importance("t", -1, &w));
    /* And the query with no output buffer answers whether it COULD, for a caller deciding
     * whether to weight at all. */
    CHECK(im.importance("t", 0, nullptr));
    std::remove(path.c_str());
}

/* ------------------------------------------------------------------ gguf */

TEST(a_gguf_imatrix_joins_the_sums_with_the_counts) {
    const std::string path = write_bin("moe.imatrix", gguf_bytes(
        {
            /* Two experts of four columns, expert-major: expert 0 is the first four. */
            { "blk.0.ffn_down.in_sum2", { 4, 2 },
              { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f } },
            { "blk.0.ffn_down.counts",  { 1, 2 }, { 100.0f, 7.0f } },
            { "blk.0.attn_q.in_sum2",   { 3 },    { 2.0f, 4.0f, 6.0f } },
        },
        { { "imatrix.chunk_count", 4, 128, "" },
          { "imatrix.chunk_size",  4, 512, "" },
          { "imatrix.datasets",    9, 0,   "wiki.train.raw" } }));
    CHECK(!path.empty());

    ImatrixFile im;
    CHECK_OK(im.open(path));
    CHECK(im.is_gguf());
    CHECK_EQ(im.chunk_count(), (int64_t)128);
    CHECK_EQ(im.chunk_size(), (int64_t)512);
    CHECK_EQ(im.dataset(), std::string("wiki.train.raw"));
    CHECK_EQ(im.entries().size(), (size_t)2);

    const ImatrixEntry* e = im.find("blk.0.ffn_down");
    CHECK(e != nullptr);
    if (e) {
        CHECK_EQ(e->cols, (int64_t)4);
        CHECK_EQ(e->n_expert, (int64_t)2);
        CHECK(e->counts != nullptr);
    }
    /* A dense tensor in the same file is one expert, and the suffix is stripped from both. */
    const ImatrixEntry* q = im.find("blk.0.attn_q");
    CHECK(q != nullptr);
    if (q) {
        CHECK_EQ(q->cols, (int64_t)3);
        CHECK_EQ(q->n_expert, (int64_t)1);
        CHECK(q->counts == nullptr);
    }
    std::remove(path.c_str());
}

/* EACH EXPERT IS WEIGHTED BY ITS OWN ROW. Reading the layer's aggregate for every expert is what
 * the legacy format forces and what this format exists to stop -- so a per-expert importance that
 * quietly read expert 0 would be indistinguishable from the legacy format's aggregate. */
TEST(a_per_expert_importance_comes_from_that_experts_own_row) {
    const std::string path = write_bin("expert.imatrix", gguf_bytes(
        {
            { "t.in_sum2", { 4, 2 }, { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f } },
            { "t.counts",  { 1, 2 }, { 100.0f, 7.0f } },
        },
        {}));
    ImatrixFile im;
    CHECK_OK(im.open(path));

    std::vector<float> w0, w1;
    CHECK(im.importance("t", 0, &w0));
    CHECK(im.importance("t", 1, &w1));
    CHECK_EQ(w0.size(), (size_t)4);
    CHECK_EQ(w1.size(), (size_t)4);
    /* Expert 0 saw a flat row, so every column is equally important. */
    for (float x : w0) CHECK_NEAR(x, 1.0, 1e-5);
    /* Expert 1 did not, and the ratios are its own. */
    CHECK_NEAR(mean_of(w1), 1.0, 1e-5);
    if (w1.size() == 4) {
        CHECK_NEAR(w1[3] / w1[0], 4.0, 1e-4);
        CHECK(w1[0] != w1[3]);
    }
    CHECK(!im.importance("t", 2, &w1));
    std::remove(path.c_str());
}

/* THE POPULARITY SIGNAL IS THE RAW COUNT AND MUST STAY RAW. The placement planner warm-starts the
 * expert profile from it, and that is a question about how OFTEN an expert ran -- normalising it
 * the way importance is normalised would make every expert equally popular. */
TEST(the_activation_count_is_reported_unnormalised) {
    const std::string path = write_bin("count.imatrix", gguf_bytes(
        {
            { "t.in_sum2", { 2, 3 }, { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f } },
            { "t.counts",  { 1, 3 }, { 900.0f, 30.0f, 1.0f } },
        },
        {}));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    CHECK_NEAR(im.count("t", 0), 900.0, 1e-6);
    CHECK_NEAR(im.count("t", 1), 30.0, 1e-6);
    CHECK_NEAR(im.count("t", 2), 1.0, 1e-6);
    CHECK_EQ(im.count("t", 3), 0.0);
    CHECK_EQ(im.count("absent", 0), 0.0);
    std::remove(path.c_str());
}

TEST(a_gguf_entry_with_counts_and_no_sums_is_dropped) {
    const std::string path = write_bin("orphan.imatrix", gguf_bytes(
        {
            { "good.in_sum2", { 2 }, { 1.0f, 3.0f } },
            { "orphan.counts", { 1 }, { 5.0f } },
        },
        {}));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    CHECK_EQ(im.entries().size(), (size_t)1);
    CHECK(im.find("good") != nullptr);
    CHECK(im.find("orphan") == nullptr);
    std::remove(path.c_str());
}

/* general.alignment moves where the data section starts, and reading it from the wrong place is
 * a whole file of plausible-looking wrong numbers rather than an error. */
TEST(the_declared_alignment_places_the_data_section) {
    for (uint64_t a : { (uint64_t)32, (uint64_t)64, (uint64_t)1 }) {
        const std::string path = write_bin("align.imatrix", gguf_bytes(
            { { "t.in_sum2", { 3 }, { 2.0f, 4.0f, 6.0f } } },
            { { "general.alignment", 4, (uint32_t)a, "" } }, 3, a));
        ImatrixFile im;
        CHECK_OK(im.open(path));
        const ImatrixEntry* e = im.find("t");
        CHECK(e != nullptr);
        if (e && e->sum2) {
            CHECK_EQ(f32_at(e->sum2, 0), 2.0f);
            CHECK_EQ(f32_at(e->sum2, 2), 6.0f);
        }
        std::remove(path.c_str());
    }
}

/* A KV the walker does not care about still has to be STEPPED OVER correctly, or every tensor
 * info after it is read from the middle of a string. */
TEST(an_unrelated_kv_is_stepped_over_rather_than_read) {
    const std::string path = write_bin("kv.imatrix", gguf_bytes(
        { { "t.in_sum2", { 2 }, { 1.0f, 3.0f } } },
        { { "general.architecture", 8, 0, "llama" },
          { "some.f64",             12, 42, "" },
          { "imatrix.chunk_count",  4, 9, "" } }));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    CHECK_EQ(im.chunk_count(), (int64_t)9);
    const ImatrixEntry* e = im.find("t");
    CHECK(e != nullptr);
    if (e && e->sum2) CHECK_EQ(f32_at(e->sum2, 1), 3.0f);
    std::remove(path.c_str());
}

/* ------------------------------------------------------------------ refusals
 *
 * Each of these is a file a user was handed. The reader walks lengths out of a mapping, so the
 * alternative to refusing is reading past the end of it. */

TEST(a_file_that_is_not_there_is_an_io_error) {
    ImatrixFile im;
    CHECK_EQ(im.open("/nonexistent/radiance/imatrix.dat"), RAD_E_IO);
}

TEST(a_file_too_short_to_be_an_imatrix_is_refused) {
    const std::string path = write_bin("short.imatrix", { 1, 2, 3, 4 });
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
    std::remove(path.c_str());
}

TEST(a_legacy_entry_that_runs_past_the_end_is_refused) {
    std::vector<uint8_t> bytes = legacy_bytes({ { "t", 1, { 1.0f, 2.0f, 3.0f } } });
    bytes.resize(bytes.size() - 6);          /* half of the last float, and part of the one before */
    const std::string path = write_bin("trunc.imatrix", bytes);
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
    std::remove(path.c_str());
}

TEST(a_legacy_entry_count_that_is_not_one_is_refused) {
    Bin w;
    w.i32(0);
    w.i32(0);
    const std::string path = write_bin("zeroentries.imatrix", w.b);
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
    std::remove(path.c_str());
}

TEST(a_gguf_version_this_reader_does_not_know_is_refused) {
    const std::string path = write_bin("v2.imatrix", gguf_bytes(
        { { "t.in_sum2", { 2 }, { 1.0f, 2.0f } } }, {}, /*version=*/2));
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
    std::remove(path.c_str());
}

TEST(a_gguf_tensor_that_runs_past_the_end_is_refused) {
    std::vector<uint8_t> bytes =
        gguf_bytes({ { "t.in_sum2", { 64 }, { 1.0f, 2.0f } } }, {});
    const std::string path = write_bin("over.imatrix", bytes);
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
    std::remove(path.c_str());
}

/* 2^62 x 4 floats is 2^64 elements, which wraps to zero: a zero-byte tensor passes any bounds
 * test, and the entry would then claim 2^62 columns to read. */
TEST(a_gguf_tensor_whose_dims_wrap_is_refused) {
    const std::string path = write_bin("wrap.imatrix", gguf_bytes(
        { { "t.in_sum2", { (uint64_t)1 << 62, 4 }, { 1.0f, 2.0f } } }, {}));
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
    std::remove(path.c_str());
}

/* A tensor count is an allocation size before it is anything else. */
TEST(a_gguf_tensor_count_the_file_cannot_hold_is_refused) {
    Bin w;
    w.raw("GGUF", 4);
    w.u32(3);
    w.u64((uint64_t)1 << 60);   /* tensors */
    w.u64(0);                   /* kvs */
    w.u64(0);
    const std::string path = write_bin("manytensors.imatrix", w.b);
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
    std::remove(path.c_str());
}

/* count() reads one value per expert of the .in_sum2 dims, so a .counts that holds fewer is not
 * read at all: the importance stands, and the activation count reads as absent. */
TEST(a_gguf_counts_shorter_than_the_expert_count_is_ignored) {
    const std::string path = write_bin("shortcounts.imatrix", gguf_bytes(
        { { "t.in_sum2", { 2, 3 }, { 1.f, 1.f, 1.f, 1.f, 2.f, 6.f } },
          { "t.counts",  { 1, 2 }, { 900.0f, 30.0f } } }, {}));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    const ImatrixEntry* e = im.find("t");
    CHECK(e != nullptr);
    if (e) {
        CHECK_EQ(e->n_expert, (int64_t)3);
        CHECK(e->counts == nullptr);
    }
    CHECK_EQ(im.count("t", 2), 0.0);
    std::vector<float> imp;
    CHECK(im.importance("t", 2, &imp));
    CHECK_EQ(imp.size(), (size_t)2);
    if (imp.size() == 2) CHECK_NEAR(imp[1] / imp[0], 3.0f, 1e-6);
    std::remove(path.c_str());
}

/* The trailing dims of an .in_sum2 can multiply to less than cols x n_expert -- a zero third dim
 * makes the tensor empty -- and every query indexes by cols x n_expert. */
TEST(a_gguf_sums_smaller_than_their_claimed_extent_are_dropped) {
    const std::string path = write_bin("emptysums.imatrix", gguf_bytes(
        { { "t.in_sum2", { 4, 2, 0 }, {} },
          { "u.in_sum2", { 2 },       { 1.0f, 3.0f } } }, {}));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    CHECK(im.find("t") == nullptr);
    CHECK(im.find("u") != nullptr);
    std::remove(path.c_str());
}

TEST(an_alignment_that_is_not_a_power_of_two_is_refused) {
    Bin w;
    w.raw("GGUF", 4);
    w.u32(3);
    w.u64(0);           /* no tensors */
    w.u64(1);
    w.gstr("general.alignment");
    w.u32(4);
    w.u32(24);          /* not a power of two */
    const std::string path = write_bin("badalign.imatrix", w.b);
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
    std::remove(path.c_str());
}

TEST(reopening_replaces_everything_the_previous_file_left) {
    const std::string a = write_bin("first.imatrix", legacy_bytes({
        { "only_in_a", 1, { 1.0f, 2.0f } },
    }));
    const std::string b = write_bin("second.imatrix", gguf_bytes(
        { { "only_in_b.in_sum2", { 2 }, { 3.0f, 4.0f } } },
        { { "imatrix.chunk_count", 4, 5, "" } }));
    ImatrixFile im;
    CHECK_OK(im.open(a));
    CHECK(im.find("only_in_a") != nullptr);
    CHECK_EQ(im.chunk_count(), (int64_t)0);

    CHECK_OK(im.open(b));
    CHECK(im.find("only_in_a") == nullptr);
    CHECK(im.find("only_in_b") != nullptr);
    CHECK(im.is_gguf());
    CHECK_EQ(im.chunk_count(), (int64_t)5);
    CHECK_EQ(im.entries().size(), (size_t)1);
    std::remove(a.c_str());
    std::remove(b.c_str());
}

RAD_TEST_MAIN()
