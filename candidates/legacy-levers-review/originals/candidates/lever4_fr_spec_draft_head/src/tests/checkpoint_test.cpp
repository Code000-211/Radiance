/* checkpoint_test.cpp -- a checkpoint as a source of logical weights (core/format/checkpoint.h).
 *
 * rad-convert writes a container from a checkpoint and the loader serves a trivial one directly,
 * and both go through ckpt_resolve: which tensors make a declared weight, what encoding they
 * already are, and the rows of it. A wrong answer there is a model that loads and is wrong -- an
 * expert that is every expert, a scale applied to the next piece's rows -- so each case below is
 * one of the ways a checkpoint is laid out in practice, built small and read back.
 */
#include "rad_test.h"
#include "format/checkpoint.h"
#include "rad_plugin.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

using namespace rad;

namespace {

struct T {
    std::string name, dtype;          /* safetensors spelling: BF16, F32, F8_E4M3 */
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

std::string dir_path() {
    const char* d = std::getenv("TMPDIR");
    return std::string(d && *d ? d : "/tmp") + "/radiance_checkpoint_test_" +
           std::to_string((long long)getpid());
}

void write_safetensors_file(const std::string& path, const std::vector<T>& ts) {
    std::string hdr = "{";
    uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        std::string shape;
        for (size_t k = 0; k < ts[i].shape.size(); ++k)
            shape += (k ? "," : "") + std::to_string(ts[i].shape[k]);
        hdr += (i ? "," : "") + std::string("\"") + ts[i].name + "\":{\"dtype\":\"" + ts[i].dtype +
               "\",\"shape\":[" + shape + "],\"data_offsets\":[" + std::to_string(off) + "," +
               std::to_string(off + ts[i].bytes.size()) + "]}";
        off += ts[i].bytes.size();
    }
    hdr += "}";
    while (hdr.size() % 8) hdr += ' ';
    FILE* f = std::fopen(path.c_str(), "wb");
    const uint64_t n = hdr.size();
    std::fwrite(&n, 8, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    for (const T& t : ts) std::fwrite(t.bytes.data(), 1, t.bytes.size(), f);
    std::fclose(f);
}

/* One model.safetensors and a config.json in a fresh directory. */
std::string write_checkpoint(const std::vector<T>& ts, const std::string& config) {
    const std::string dir = dir_path();
    ::mkdir(dir.c_str(), 0755);
    write_safetensors_file(dir + "/model.safetensors", ts);
    FILE* f = std::fopen((dir + "/config.json").c_str(), "wb");
    std::fwrite(config.data(), 1, config.size(), f);
    std::fclose(f);
    return dir;
}

void remove_checkpoint(const std::string& dir) {
    std::remove((dir + "/model.safetensors").c_str());
    std::remove((dir + "/mtp-lm-head-reduced.safetensors").c_str());
    std::remove((dir + "/config.json").c_str());
    ::rmdir(dir.c_str());
}

/* Element i of a tensor is a value that names it: tensor tag + i/1000, exact in bf16 for small i. */
T bf16(const std::string& name, std::vector<int64_t> shape, float tag) {
    T t{ name, "BF16", shape, {} };
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    t.bytes.resize((size_t)n * 2);
    for (int64_t i = 0; i < n; ++i) rad_store_f32(t.bytes.data(), RAD_BF16, i, tag + (float)(i % 64));
    return t;
}

T fp8(const std::string& name, std::vector<int64_t> shape) {
    T t{ name, "F8_E4M3", shape, {} };
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    t.bytes.resize((size_t)n);
    for (int64_t i = 0; i < n; ++i) rad_store_f32(t.bytes.data(), RAD_F8E4M3, i, (float)(i % 7) - 3.0f);
    return t;
}

RadNameMap map(const char* declared, int mode, std::vector<const char*> src,
               std::vector<int> index = {}, int concat_dim = 0) {
    RadNameMap m{};
    m.declared = declared;
    m.mode = mode;
    m.n_src = (int)src.size();
    m.concat_dim = concat_dim;
    for (int i = 0; i < 8; ++i) m.src_index[i] = -1;
    for (size_t i = 0; i < src.size(); ++i) m.src[i] = src[i];
    for (size_t i = 0; i < index.size(); ++i) m.src_index[i] = index[i];
    return m;
}

const char* kConfig =
    "{\"model_type\":\"qwen3_5\",\"hidden_size\":256,\"num_hidden_layers\":2,"
    "\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\","
    "\"weight_block_size\":[128,128]}}";

}  /* namespace */

TEST(the_metadata_names_the_architecture_and_the_quantisation) {
    const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK_EQ(ck.meta().arch, std::string("qwen35"));
    CHECK_EQ(ck.meta().quant, std::string("fp8_e4m3"));
    CHECK_EQ(ck.meta().m.n_embd, (int64_t)256);
    CHECK_EQ(ck.fp8_block_rows(), (int64_t)128);
    CHECK_EQ(ck.dir(), dir);
    remove_checkpoint(dir);
}

TEST(a_plain_tensor_is_its_own_encoding_and_reads_back_as_itself) {
    const std::string dir = write_checkpoint({ bf16("a.weight", { 4, 8 }, 1.0f) }, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("a", RAD_MAP_COPY, { "a.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "a", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    CHECK(rad_enc_is(&w.enc, "plain"));
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(w.rows, (int64_t)4);
    CHECK_EQ(w.cols, (int64_t)8);
    std::vector<float> f(8);
    REQUIRE_EQ(ckpt_rows_f32(w, 2, 1, f.data(), &why), RAD_OK);
    for (int j = 0; j < 8; ++j) CHECK_EQ(f[(size_t)j], 1.0f + (float)((16 + j) % 64));
    std::vector<uint8_t> raw(8 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 2, 1, raw.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(raw.data(), (const uint8_t*)ck.find("a.weight")->data + 32, 16), 0);
    remove_checkpoint(dir);
}

/* BLOCK-SCALED FP8 IS ONE WEIGHT OF TWO PLANES, and its value is the code times its block's
 * scale -- the scale of the block the element is in, along both axes. */
TEST(block_fp8_is_the_codes_and_their_scale_plane) {
    const int64_t N = 256, K = 384;
    T s = bf16("p.weight_scale_inv", { 2, 3 }, 0.0f);
    for (int i = 0; i < 6; ++i) rad_store_f32(s.bytes.data(), RAD_BF16, i, 0.5f * (float)(i + 1));
    const std::string dir = write_checkpoint({ fp8("p.weight", { N, K }), s }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("p", RAD_MAP_COPY, { "p.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "p", false, &w, &why), RAD_OK);
    REQUIRE(w.trivial);
    CHECK(rad_enc_is(&w.enc, "affine"));
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)128);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)128);
    CHECK(rad_enc_valid(&w.enc));
    std::vector<float> f((size_t)(2 * K));
    REQUIRE_EQ(ckpt_rows_f32(w, 129, 2, f.data(), &why), RAD_OK);
    for (int64_t r = 0; r < 2; ++r)
        for (int64_t c : { (int64_t)0, (int64_t)130, (int64_t)383 }) {
            const int64_t i = (129 + r) * K + c;
            const float code = (float)(i % 7) - 3.0f;
            const float scale = 0.5f * (float)(3 + c / 128 + 1);   /* block row 1 */
            CHECK_EQ(f[(size_t)(r * K + c)], code * scale);
        }
    std::vector<uint8_t> sp(3 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 1, 1, sp.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(sp.data(), s.bytes.data() + 6, 6), 0);
    remove_checkpoint(dir);
}

/* A FUSION JOINS THE PIECES -- and their scale planes -- along dim 0, and refuses a piece that ends
 * inside a scale block, whose scales could not follow it. */
TEST(a_fusion_joins_pieces_and_their_scales) {
    const std::string dir = write_checkpoint(
        { fp8("g.weight", { 256, 128 }), bf16("g.weight_scale_inv", { 2, 1 }, 10.0f),
          fp8("u.weight", { 128, 128 }), bf16("u.weight_scale_inv", { 1, 1 }, 20.0f),
          fp8("odd.weight", { 100, 128 }), bf16("odd.weight_scale_inv", { 1, 1 }, 30.0f) },
        kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    prog.name_map.push_back(map("bad", RAD_MAP_CONCAT, { "odd.weight", "u.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    CHECK_EQ(w.shape[0], (int64_t)384);
    std::vector<uint8_t> sp(3 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 3, sp.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 0), 10.0f);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 1), 11.0f);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 2), 20.0f);
    std::vector<float> f(128);
    REQUIRE_EQ(ckpt_rows_f32(w, 300, 1, f.data(), &why), RAD_OK);   /* u's row 44, u's scale */
    CHECK_EQ(f[5], ((float)((44 * 128 + 5) % 7) - 3.0f) * 20.0f);

    REQUIRE_EQ(ckpt_resolve(ck, prog, "bad", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("inside a scale block") != std::string::npos);
    remove_checkpoint(dir);
}

/* A join along dim 1 interleaves the pieces row by row. */
TEST(a_join_along_columns_interleaves_rows) {
    const std::string dir = write_checkpoint({ bf16("l", { 3, 2 }, 0.0f),
                                               bf16("r", { 3, 4 }, 100.0f) }, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("lr", RAD_MAP_CONCAT, { "l", "r" }, {}, 1));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "lr", false, &w, &why), RAD_OK);
    CHECK_EQ(w.shape[1], (int64_t)6);
    std::vector<float> f(6);
    REQUIRE_EQ(ckpt_rows_f32(w, 1, 1, f.data(), &why), RAD_OK);
    const float want[6] = { 2, 3, 104, 105, 106, 107 };
    for (int j = 0; j < 6; ++j) CHECK_EQ(f[(size_t)j], want[j]);
    std::vector<uint8_t> raw(6 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 2, 1, raw.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(raw.data(), RAD_BF16, 0), 4.0f);
    CHECK_EQ(rad_load_f32(raw.data(), RAD_BF16, 2), 108.0f);
    remove_checkpoint(dir);
}

/* AN EXPERT IS ITS OWN SLICE OF A STACKED TENSOR -- never the whole stack -- and a stacked FP8
 * tensor's scales are sliced at the same index. */
TEST(an_expert_is_its_slice_of_a_stack) {
    const std::string dir = write_checkpoint(
        { bf16("exps", { 4, 2, 3 }, 0.0f), fp8("q.weight", { 2, 128, 128 }),
          bf16("q.weight_scale_inv", { 2, 1, 1 }, 7.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("e2", RAD_MAP_COPY, { "exps" }, { 2 }));
    prog.name_map.push_back(map("q1", RAD_MAP_COPY, { "q.weight" }, { 1 }));
    prog.name_map.push_back(map("e9", RAD_MAP_COPY, { "exps" }, { 9 }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "e2", false, &w, &why), RAD_OK);
    CHECK_EQ(w.rank, 2u);
    std::vector<float> f(3);
    REQUIRE_EQ(ckpt_rows_f32(w, 1, 1, f.data(), &why), RAD_OK);
    CHECK_EQ(f[0], (float)(2 * 6 + 3));                       /* stack index 2, row 1, col 0 */
    REQUIRE_EQ(ckpt_resolve(ck, prog, "q1", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    std::vector<uint8_t> sp(2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 1, sp.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 0), 8.0f);    /* the second expert's scale */
    CHECK(ckpt_resolve(ck, prog, "e9", false, &w, &why) < 0);
    remove_checkpoint(dir);
}

/* A CHANGED TENSOR IS NAMED AS THE CHECKPOINT NAMES IT. An expert is a slice "exps[2]" of its
 * stack, and it reads "exps"; a block-FP8 weight reads its scale plane as well as its codes, so a
 * changed scale changes the weight. */
TEST(a_weight_reads_the_checkpoint_tensors_its_pieces_and_scales_come_from) {
    const std::string dir = write_checkpoint(
        { bf16("exps", { 4, 2, 3 }, 0.0f), fp8("q.weight", { 2, 128, 128 }),
          bf16("q.weight_scale_inv", { 2, 1, 1 }, 7.0f), bf16("a.weight", { 2, 3 }, 0.0f),
          bf16("b.weight", { 2, 3 }, 0.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("e2", RAD_MAP_COPY, { "exps" }, { 2 }));
    prog.name_map.push_back(map("q1", RAD_MAP_COPY, { "q.weight" }, { 1 }));
    prog.name_map.push_back(map("ab", RAD_MAP_CONCAT, { "a.weight", "b.weight" }));
    CkptWeight e2, q1, ab;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "e2", false, &e2, &why), RAD_OK);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "q1", false, &q1, &why), RAD_OK);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "ab", false, &ab, &why), RAD_OK);
    CHECK_EQ(ckpt_tensor_of(*e2.pieces[0]), std::string("exps"));

    std::unordered_set<std::string> hit;
    CHECK(ckpt_reads_any(e2, { "exps" }, &hit));
    CHECK(!ckpt_reads_any(e2, { "exps[2]" }));        /* a slice is not a checkpoint name */
    CHECK(ckpt_reads_any(q1, { "q.weight_scale_inv" }, &hit));
    CHECK(ckpt_reads_any(ab, { "b.weight" }, &hit));   /* one changed piece of a join */
    CHECK(!ckpt_reads_any(ab, { "q.weight", "exps" }));
    CHECK_EQ(hit.size(), (size_t)3);
    remove_checkpoint(dir);
}

/* A COPY's sources are alternatives, first present wins: a tied head reads the embedding where
 * the checkpoint ships no head of its own. */
TEST(alternatives_take_the_first_tensor_present) {
    const std::string dir = write_checkpoint({ bf16("embed", { 2, 2 }, 5.0f) }, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("out", RAD_MAP_COPY, { "lm_head", "embed" }));
    prog.name_map.push_back(map("need", RAD_MAP_COPY, { "missing" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "out", false, &w, &why), RAD_OK);
    REQUIRE_EQ(w.pieces.size(), (size_t)1);
    CHECK_EQ(w.pieces[0]->name, std::string("embed"));
    CHECK_EQ(ckpt_resolve(ck, prog, "need", false, &w, &why), RAD_E_NOTFOUND);
    CHECK(why.find("missing") != std::string::npos);
    CHECK_EQ(ckpt_resolve(ck, prog, "need", true, &w, &why), RAD_E_NOTFOUND);
    CHECK_EQ(ckpt_resolve(ck, prog, "unmapped", false, &w, &why), RAD_E_NOTFOUND);
    remove_checkpoint(dir);
}

/* NOT TRIVIAL, AND SAID WHY: FP8 with no scale beside it, and pieces of different dtypes. */
TEST(what_is_not_served_as_it_is_says_why) {
    const std::string dir = write_checkpoint({ fp8("bare", { 128, 128 }),
                                               bf16("b16", { 128, 128 }, 0.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("bare", RAD_MAP_COPY, { "bare" }));
    prog.name_map.push_back(map("mixed", RAD_MAP_CONCAT, { "bare", "b16" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "bare", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("no scale plane") != std::string::npos);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "mixed", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    remove_checkpoint(dir);
}

/* REDUCED VOCABULARY DRAFT HEAD (Q4 affine with companion scale and zero, and vocab_ids). */
TEST(reduced_draft_head_q4_loads_and_resolves) {
    const std::string dir = write_checkpoint({ bf16("embed", { 2, 2 }, 1.0f) }, kConfig);

    std::vector<uint8_t> v_ids(4 * sizeof(int64_t));
    int64_t* ids = (int64_t*)v_ids.data();
    ids[0] = 100; ids[1] = 200; ids[2] = 300; ids[3] = 400;

    std::vector<uint8_t> q4_bytes(2 * 64, 0x55);
    std::vector<uint8_t> sc_bytes(2 * 1 * sizeof(uint16_t));
    uint16_t* sc = (uint16_t*)sc_bytes.data();
    sc[0] = rad_f32_to_f16(1.5f); sc[1] = rad_f32_to_f16(2.5f);

    std::vector<uint8_t> z_bytes(2 * 1 * sizeof(int16_t));
    int16_t* z = (int16_t*)z_bytes.data();
    z[0] = 8; z[1] = 8;

    T t_ids{ "mtp.lm_head_reduced.vocab_ids", "I64", { 4 }, std::move(v_ids) };
    T t_q4{ "mtp.lm_head_reduced.weight_q4", "U8", { 2, 64 }, std::move(q4_bytes) };
    T t_sc{ "mtp.lm_head_reduced.weight_scale", "F16", { 2, 1 }, std::move(sc_bytes) };
    T t_z{ "mtp.lm_head_reduced.weight_zero", "I16", { 2, 1 }, std::move(z_bytes) };

    write_safetensors_file(dir + "/mtp-lm-head-reduced.safetensors", { t_ids, t_q4, t_sc, t_z });

    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);

    const SrcTensor* vt = ck.find("mtp.lm_head_reduced.vocab_ids");
    REQUIRE(vt != nullptr);
    CHECK_EQ(vt->numel(), (int64_t)4);
    CHECK_EQ(vt->rad_dtype, (uint32_t)RAD_I64);

    Program prog;
    prog.name_map.push_back(map("mtp.draft_head_reduced", RAD_MAP_COPY, { "mtp.lm_head_reduced.weight_q4" }));

    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "mtp.draft_head_reduced", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    CHECK_EQ(w.rows, (int64_t)2);
    CHECK_EQ(w.cols, (int64_t)128);
    CHECK_EQ(w.enc.n_planes, 3);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_U4);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_F16);
    CHECK_EQ(w.enc.plane[2].dtype, (uint32_t)RAD_I16);

    std::vector<uint8_t> p0(2 * 64, 0);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 2, p0.data(), &why), RAD_OK);
    CHECK_EQ(p0[0], (uint8_t)0x55);
    CHECK_EQ(p0[127], (uint8_t)0x55);

    std::vector<uint16_t> p1(2, 0);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 2, p1.data(), &why), RAD_OK);
    CHECK_EQ(p1[0], sc[0]);
    CHECK_EQ(p1[1], sc[1]);

    std::vector<int16_t> p2(2, 0);
    REQUIRE_EQ(ckpt_plane_rows(w, 2, 0, 2, p2.data(), &why), RAD_OK);
    CHECK_EQ(p2[0], z[0]);
    CHECK_EQ(p2[1], z[1]);

    std::unordered_set<std::string> hit;
    CHECK(ckpt_reads_any(w, { "mtp.lm_head_reduced.weight_zero" }, &hit));
    CHECK_EQ(hit.size(), (size_t)1);

    remove_checkpoint(dir);
}

TEST(block_fp8_with_f32_scale_resolves_and_converts_to_bf16) {
    std::vector<uint8_t> f32_scale_bytes(1 * 1 * sizeof(float));
    float* f32_sc = (float*)f32_scale_bytes.data();
    f32_sc[0] = 0.05f;

    T t_codes = fp8("blk.0.attn.weight", { 128, 128 });
    T t_scale{ "blk.0.attn.weight_scale_inv", "F32", { 1, 1 }, std::move(f32_scale_bytes) };

    const std::string dir = write_checkpoint({ t_codes, t_scale }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);

    Program prog;
    prog.name_map.push_back(map("blk.0.attn", RAD_MAP_COPY, { "blk.0.attn.weight" }));

    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "blk.0.attn", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    CHECK_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_F8E4M3);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_BF16);

    std::vector<uint16_t> sc_out(1, 0);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 1, sc_out.data(), &why), RAD_OK);
    CHECK_EQ(sc_out[0], rad_f32_to_bf16(0.05f));

    remove_checkpoint(dir);
}

RAD_TEST_MAIN()

