/* container_source_test.cpp -- extending a .rad in place with a second checkpoint's weights.
 *
 * `rad-convert TARGET.rad --in-place --draft-model DIR` is the route that puts a drafter into a
 * container that was converted before the drafter existed, without reading, decoding or rewriting
 * a single target weight. What makes it safe is entirely in two places, and both are exercised
 * here rather than inferred from a successful run:
 *
 *   Checkpoint::open_container / ckpt_resolve -- a container entry IS the logical weight, in the
 *   encoding the file already stores, and a name it does not hold falls through to the merged
 *   safetensors of the same run.
 *
 *   the carry-over loop in rad-convert -- every entry is recorded through add_existing() at the
 *   offset it has, so entries, encodings, plane bytes and movement units are the old file's, and
 *   the new weights are appended past its end.
 *
 * The refusals are the part worth testing hardest: a replacement that would change an encoding, a
 * second entry of one name, and a recipe rule matching a carried weight must each fail BEFORE the
 * header is rewritten, and leave the file byte for byte as it was. That last property is what makes
 * the route usable on a container whose source checkpoint no longer exists.
 *
 * The draft side is a tiny synthetic safetensors directory, not the 1.2 GB DFlash2 checkpoint: the
 * name map, the config merge and the bf16 head path are the same code for either, and a test that
 * needs the real one is not a test.
 */
#include "rad_test.h"
#include "format/checkpoint.h"
#include "format/radfile.h"
#include "text/tokenizer.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace rad;

namespace {

std::string tmp_dir() {
    const char* d = std::getenv("TMPDIR");
    return std::string(d && *d ? d : "/tmp") + "/radiance_container_source_" +
           std::to_string((long long)getpid());
}

/* CREATE THE FIXTURE DIRECTORY, once per case, through the filesystem rather than a shell: TMPDIR
 * is an operator's value and may hold anything a path may, which is exactly what string-built
 * `rm -rf '<dir>'` cannot survive. EEXIST is fine -- the name is this process's pid, so it is only
 * ever this test's own earlier case. */
int make_dir(const std::string& dir) {
    if (::mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST) return 0;
    return -errno;
}

/* Remove the files this test creates, by name. No recursive shell delete: the fixture's contents
 * are known, and a wrong or empty path must never become `rm -rf` something. */
void clean_dir(const std::string& dir) {
    static const char* const kFiles[] = {
        "/target.rad", "/target.rad.pre-append", "/clash/model.safetensors",
        "/teacher/teacher-matrices.safetensors"
    };
    for (const char* f : kFiles) ::unlink((dir + f).c_str());
    ::rmdir((dir + "/clash").c_str());
    ::rmdir((dir + "/teacher").c_str());
    ::rmdir(dir.c_str());
}

std::string slurp(const std::string& p) {
    std::string s;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return s;
    char b[1 << 16];
    size_t n;
    while ((n = std::fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
    std::fclose(f);
    return s;
}

/* A safetensors file of plain bf16 tensors, and a config.json beside it. */
int write_safetensors(const std::string& dir, const std::string& file,
                      const std::vector<std::pair<std::string, std::vector<int64_t>>>& ts,
                      const std::string& config, bool with_config) {
    if (make_dir(dir) < 0) return -1;
    std::string hdr = "{";
    uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        int64_t n = 1;
        for (int64_t d : ts[i].second) n *= d;
        const uint64_t bytes = (uint64_t)n * 2;      /* bf16 */
        std::string shape;
        for (size_t k = 0; k < ts[i].second.size(); ++k)
            shape += (k ? "," : "") + std::to_string(ts[i].second[k]);
        hdr += (i ? "," : "") + std::string("\"") + ts[i].first + "\":{\"dtype\":\"BF16\","
               "\"shape\":[" + shape + "],\"data_offsets\":[" + std::to_string(off) + "," +
               std::to_string(off + bytes) + "]}";
        off += bytes;
    }
    hdr += "}";
    while (hdr.size() % 8) hdr += ' ';
    FILE* f = std::fopen((dir + "/" + file).c_str(), "wb");
    if (!f) return -1;
    const uint64_t n = hdr.size();
    std::fwrite(&n, 8, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    std::vector<uint16_t> zeros(1 << 16, 0x3c00);     /* 1.0 */
    uint64_t left = off;
    while (left) {
        const size_t m = (size_t)std::min<uint64_t>(left, zeros.size() * 2);
        std::fwrite(zeros.data(), 1, m, f);
        left -= m;
    }
    std::fclose(f);
    if (!with_config) return 0;
    f = std::fopen((dir + "/config.json").c_str(), "wb");
    if (!f) return -1;
    std::fwrite(config.data(), 1, config.size(), f);
    std::fclose(f);
    return 0;
}

RadWriter::Weight mk_weight(const std::string& name, const RadEncoding& enc, int64_t rows,
                           int64_t cols, int32_t layer = -1, int32_t expert = -1,
                           int32_t slot = 0) {
    RadWriter::Weight w;
    w.name = name;
    w.enc = enc;
    w.rank = 2;
    w.shape[0] = rows;
    w.shape[1] = cols;
    w.layer = layer;
    w.expert = expert;
    w.slot = slot;
    return w;
}

std::vector<std::vector<uint8_t>> plane_bytes(const RadWriter::Weight& w, uint8_t seed) {
    int64_t rows = 0, cols = 0;
    rad_enc_view(w.rank, w.shape, &rows, &cols);
    std::vector<std::vector<uint8_t>> out((size_t)w.enc.n_planes);
    for (int k = 0; k < w.enc.n_planes; ++k) {
        const int64_t n = rad_enc_plane_bytes(&w.enc.plane[k], rows, cols);
        out[(size_t)k].assign((size_t)n, seed);
        for (int64_t i = 0; i < n; ++i) out[(size_t)k][(size_t)i] = (uint8_t)(seed + k + i);
    }
    return out;
}

/* A container with a quantised target weight, a plain bf16 embedding and a vocabulary.
 *
 * It RETURNS A STATUS and the path separately: REQUIRE leaves the enclosing case, which in a
 * helper that returns a value expands to a bare `return;` and does not compile. A failing step is
 * REPORTED at the call site through REQUIRE_BUILT(), which leaves the case, so a caller never
 * proceeds with a half-written fixture. */
struct Built {
    std::string path;
    int         status = RAD_OK;
};

/* Stop the case unless the fixture was built. A macro rather than a function because leaving the
 * case is `return`, which only the enclosing void test can do -- and the alternative is to keep
 * going against a half-written container. */
#define REQUIRE_BUILT(b)                                                              \
    do {                                                                              \
        ++::radtest::checks();                                                        \
        if ((b).status < 0) {                                                         \
            ::radtest::fail(__FILE__, __LINE__,                                       \
                            std::string("build_container(" #b "): ") +                \
                                rad_strerror((b).status));                            \
            return;                                                                   \
        }                                                                             \
    } while (0)

Built build_container(const std::string& dir) {
    Built b;
    b.path = dir + "/target.rad";
    if ((b.status = make_dir(dir)) < 0) return b;
    RadWriter w;
    if ((b.status = w.begin(b.path.c_str())) < 0) return b;
    w.set_arch("radtest");
    w.set_model_name("container-source-test");
    w.set_quant("fp8_block");
    w.set_created_by("container_source_test");
    w.meta_i("n_layers", 1);
    w.meta_i("n_embd", 8);
    w.meta_i("n_vocab", 4);
    w.meta_s("note", "original");
    /* A FREE-FORM INT ROW AND A FREE-FORM F64 ROW, the kinds a checkpoint's own config carries
     * (`text_config.hidden_size`, a scale factor). They are what an extension has to write back at
     * the same TYPE: `MetaOwn` holds both as text, so a writer that emits them as strings leaves a
     * file whose `get_i`/`get_f` no longer find them. The F64 value is chosen to be inexact in
     * decimal and exact in %.17g, which is how the container stores it. */
    w.meta_i("text_config.hidden_size", 2560);
    w.meta_f("text_config.test_scale", 0.12345678901234567);

    const RadEncoding q = rad_enc_affine(RAD_I8, RAD_BF16, 1, 4);
    const RadWriter::Weight tgt = mk_weight("output.weight", q, 4, 8);
    const RadWriter::Weight emb = mk_weight("token_embd.weight", rad_enc_plain(RAD_BF16), 4, 8);
    const std::vector<std::vector<uint8_t>> pt = plane_bytes(tgt, 7);
    const std::vector<std::vector<uint8_t>> pe = plane_bytes(emb, 11);
    {
        const int64_t i = w.add_weight(tgt);
        if (i < 0) { b.status = (int)i; w.abort(); return b; }
        for (int k = 0; k < tgt.enc.n_planes; ++k) {
            const int st = w.write_plane(i, k, 0, pt[(size_t)k].data(),
                                         (int64_t)pt[(size_t)k].size());
            if (st < 0) { b.status = st; w.abort(); return b; }
        }
        const int64_t j = w.add_weight(emb);
        if (j < 0) { b.status = (int)j; w.abort(); return b; }
        for (int k = 0; k < emb.enc.n_planes; ++k) {
            const int st = w.write_plane(j, k, 0, pe[(size_t)k].data(),
                                         (int64_t)pe[(size_t)k].size());
            if (st < 0) { b.status = st; w.abort(); return b; }
        }
    }

    /* A vocabulary: four tokens, no merges, BPE. */
    VocabBuild vb;
    vb.kind = RAD_TOK_BPE;
    vb.tokens = { "a", "b", "ab", "c" };
    vb.bos = -1; vb.eos = 2; vb.eot = -1; vb.pad = -1; vb.unk = -1; vb.sep = -1;
    vb.chat_template = "template";
    struct Sink : StringSink {
        RadWriter* w = nullptr;
        rad_stroff intern(std::string_view s) override { return w->intern_string(s); }
    } sink;
    sink.w = &w;
    std::vector<uint8_t> blob;
    if ((b.status = vocab_serialize(vb, sink, 0, blob)) < 0) { w.abort(); return b; }
    if ((b.status = w.set_vocab_section(blob.data(), (int64_t)blob.size())) < 0) {
        w.abort();
        return b;
    }
    if ((b.status = w.finish()) < 0) return b;
    return b;
}

}  /* namespace */

TEST(a_container_opens_as_a_source_and_its_entries_are_its_logical_weights) {
    const std::string dir = tmp_dir();
    const Built b = build_container(dir);
    REQUIRE_BUILT(b);

    Checkpoint ck;
    REQUIRE((ck.open(b.path)) >= 0);
    CHECK(ck.is_container());
    CHECK(!ck.is_gguf());
    CHECK_EQ(ck.meta().arch, std::string("radtest"));
    CHECK(ck.meta().m.n_layers == 1);
    CHECK(ck.meta().m.n_vocab == 4);
    CHECK_EQ(std::string(ck.meta().get("note") ? ck.meta().get("note") : ""),
             std::string("original"));
    /* The typed keys are read into RadModelMeta and are NOT duplicated into the free-form list:
     * a second row of the same key would make a later lookup read a stale value. */
    CHECK(ck.meta().get("n_layers") == nullptr);

    const RadFileEntry* e = ck.container_entry("output.weight");
    REQUIRE(e != nullptr);
    const RadEncoding& enc = ck.container().encoding(*e);
    CHECK(rad_enc_is(&enc, "affine"));
    CHECK_EQ(enc.n_planes, 2);
    CHECK_EQ(enc.plane[0].dtype, (uint32_t)RAD_I8);
    CHECK_EQ(enc.plane[1].block[1], (int64_t)4);

    /* An entry resolves as itself, carrying the file it came from -- which is what says "do not
     * decode me", and what the writer's add_existing() path needs. */
    Program prog;
    CkptWeight cw;
    std::string why;
    REQUIRE((ckpt_resolve(ck, prog, "output.weight", false, &cw, &why)) == RAD_OK);
    CHECK(cw.entry != nullptr);
    CHECK(cw.file == &ck.container());
    CHECK_EQ(cw.rows, (int64_t)4);
    CHECK_EQ(cw.cols, (int64_t)8);
    CHECK(cw.pieces.empty());
    CHECK(rad_enc_equal(&cw.enc, &enc));

    /* ...and reading it as f32 is refused rather than decoded: a conversion that quantises a
     * container entry again is the requantisation this route exists to avoid. */
    std::vector<float> f(32);
    CHECK((ckpt_rows_f32(cw, 0, 4, f.data(), &why)) < 0);
    CHECK(why.find("carried") != std::string::npos);

    /* A name the container does not hold is not an entry. */
    CkptWeight none;
    CHECK((ckpt_resolve(ck, prog, "dflash.norm.weight", false, &none, &why)) == RAD_E_NOTFOUND);

    clean_dir(dir);
}

TEST(a_draft_head_merges_unprefixed_and_a_taken_name_is_refused) {
    const std::string dir = tmp_dir();
    const Built b = build_container(dir);
    REQUIRE_BUILT(b);
    const std::string head_dir = dir + "/teacher";
    /* The head file a bf16 drafter reads: the target's lm_head at its original precision. */
    REQUIRE_EQ(write_safetensors(head_dir, "teacher-matrices.safetensors",
                                 { { "lm_head.weight", { 4, 8 } } }, "", false), 0);

    Checkpoint ck;
    REQUIRE((ck.open(b.path)) >= 0);
    REQUIRE((ck.add_tensors(head_dir + "/teacher-matrices.safetensors", "")) >= 0);
    REQUIRE_EQ(ck.merged().size(), (size_t)1);
    CHECK_EQ(ck.merged()[0], std::string("lm_head.weight"));

    /* The merged tensor is found where a plugin looks for it, and it is plain bf16. */
    const SrcTensor* t = ck.find("lm_head.weight");
    REQUIRE(t != nullptr);
    CHECK_EQ(t->rad_dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(t->shape.size(), (size_t)2);
    CHECK_EQ(t->shape[0], (int64_t)4);

    /* A name the container already serves is refused: the entry would shadow the merged tensor
     * (or be shadowed by it), and either way one of the two is a weight nobody reads. */
    const std::string clash = dir + "/clash";
    REQUIRE_EQ(write_safetensors(clash, "model.safetensors",
                                 { { "token_embd.weight", { 4, 8 } } }, "", false), 0);
    CHECK((ck.add_tensors(clash + "/model.safetensors", "")) == RAD_E_DUPLICATE);

    clean_dir(dir);
}

/* THE INTEGRITY CONTRACT. Extending the container carries every original entry at its own offset,
 * adds the new ones past the end, and changes no byte of the old blob. */
TEST(an_in_place_extension_preserves_every_original_entry_vocab_and_header_rollback) {
    const std::string dir = tmp_dir();
    const Built b = build_container(dir);
    REQUIRE_BUILT(b);
    const std::string before = slurp(b.path);
    REQUIRE(before.size() > sizeof(RadFileHeader));

    RadFile old;
    REQUIRE((old.open(b.path.c_str())) >= 0);
    const int64_t n_old = old.entry_count();
    std::vector<RadFileEntry> entries;
    std::vector<RadFilePlane> planes;
    std::vector<RadEncoding> encs;
    std::vector<std::string> names, quants, opts;
    for (int64_t i = 0; i < n_old; ++i) {
        entries.push_back(old.entry(i));
        encs.push_back(old.encoding(old.entry(i)));
        /* The STRINGS are copied now: finish_append rewrites the header in place, so a live
         * `old.str()` after it reads through an offset the new header no longer points at. */
        names.push_back(old.str(old.entry(i).name));
        quants.push_back(old.str(old.entry(i).quantizer));
        opts.push_back(old.str(old.entry(i).options));
        for (uint32_t k = 0; k < old.entry(i).n_planes; ++k)
            planes.push_back(old.plane(old.entry(i), (int)k));
    }
    const RadVocabHeader vh = *old.vocab().h;
    const uint64_t data_off  = old.header().data_off;
    /* The vocabulary is deserialised NOW, for the same reason the strings are copied above:
     * finish_append rewrites the header in place, and vocab_deserialize reads the string blob's
     * extent from that header -- after the append the offsets it would compute name the new
     * tables, not the old ones. */
    VocabBuild vb_orig;
    REQUIRE((vocab_deserialize(old.base(), (uint64_t)old.size(), old.vocab().h, vb_orig)) >= 0);
    REQUIRE_EQ(vb_orig.tokens.size(), (size_t)vh.n_tokens);

    /* Exactly what rad-convert's carry-over loop does, then one new bf16 weight. */
    RadWriter w;
    REQUIRE((w.begin_append(b.path.c_str(), old)) >= 0);
    w.set_arch("radtest");
    w.set_model_name("container-source-test");
    w.set_quant("fp8_block");
    w.set_created_by("container_source_test, extended");
    /* THE METADATA GOES THROUGH THE PRODUCTION METHOD, from a Checkpoint opened on the container
     * being extended. That is the point: a hand-rolled copy of the converter's loop would pass
     * whether or not the tool preserves anything, and would not notice a regression in the code
     * that actually runs. */
    Checkpoint src;
    REQUIRE((src.open(b.path)) >= 0);
    REQUIRE(src.is_container());
    src.write_metadata(w, &old);
    /* What this run adds on top, as the converter does after the call. */
    w.meta_s("draft.architectures", "DFlash2DraftModel");
    w.meta_i("draft.bf16", 1);
    for (int64_t i = 0; i < n_old; ++i) {
        const RadFileEntry& e = old.entry(i);
        RadWriter::Weight ew;
        ew.name = old.str(e.name);
        ew.quantizer = old.str(e.quantizer);
        ew.options = old.str(e.options);
        ew.enc = old.encoding(e);
        ew.rank = e.rank;
        for (uint32_t k = 0; k < RAD_MAX_RANK; ++k) ew.shape[k] = e.shape[k];
        ew.layer = e.layer;
        ew.expert = e.expert;
        ew.slot = e.slot;
        REQUIRE((w.add_existing(ew, old, e)) >= 0);
    }
    /* A second entry of a carried name is refused, not silently shadowed. */
    {
        RadWriter::Weight dup;
        dup.name = "output.weight";
        dup.enc = rad_enc_plain(RAD_BF16);
        dup.rank = 2;
        dup.shape[0] = 4;
        dup.shape[1] = 8;
        CHECK((w.add_weight(dup)) == RAD_E_DUPLICATE);
    }
    const RadWriter::Weight draft = mk_weight("dflash.norm.weight", rad_enc_plain(RAD_BF16), 1, 8);
    const std::vector<std::vector<uint8_t>> pd = plane_bytes(draft, 21);
    const int64_t di = w.add_weight(draft);
    REQUIRE(di >= 0);
    REQUIRE((w.write_plane(di, 0, 0, pd[0].data(), (int64_t)pd[0].size())) >= 0);
    {
        /* The vocabulary is carried by re-serialising the one the container already holds, so it
         * goes through the same code a fresh convert uses. */
        struct Sink : StringSink {
            RadWriter* w = nullptr;
            rad_stroff intern(std::string_view s) override { return w->intern_string(s); }
        } sink;
        sink.w = &w;
        std::vector<uint8_t> blob;
        REQUIRE((vocab_serialize(vb_orig, sink, 0, blob)) >= 0);
        REQUIRE((w.set_vocab_section(blob.data(), (int64_t)blob.size())) >= 0);
    }
    REQUIRE((w.finish()) >= 0);
    /* `old` stays mapped on purpose: it is the pre-extension view, and its tables and blob are
     * still where they were, so every comparison below reads the ORIGINAL bytes rather than a
     * second reader of the extended file. The writer's other-user check skips this process. */

    /* NOTHING IN FRONT OF THE OLD END MOVED. Only the header changed -- it names new tables, past
     * the old file's end -- so everything after sizeof(RadFileHeader) is byte-identical. */
    const std::string after = slurp(b.path);
    REQUIRE(after.size() > before.size());
    CHECK_EQ(std::memcmp(after.data() + sizeof(RadFileHeader),
                         before.data() + sizeof(RadFileHeader),
                         before.size() - sizeof(RadFileHeader)), 0);

    RadFile f;
    REQUIRE((f.open(b.path.c_str())) >= 0);
    /* THE ORIGINAL ENTRIES: same names, same encodings, same planes, same offsets. */
    CHECK_EQ(f.entry_count(), n_old + 1);
    size_t pi = 0;
    for (int64_t i = 0; i < n_old; ++i) {
        const RadFileEntry& e = f.entry(i);
        CHECK_EQ(std::string(f.str(e.name)), names[(size_t)i]);
        CHECK_EQ(std::string(f.str(e.quantizer)), quants[(size_t)i]);
        CHECK_EQ(std::string(f.str(e.options)), opts[(size_t)i]);
        CHECK(rad_enc_equal(&f.encoding(e), &encs[(size_t)i]));
        CHECK_EQ(e.offset, entries[(size_t)i].offset);
        CHECK_EQ(e.bytes, entries[(size_t)i].bytes);
        CHECK_EQ(e.layer, entries[(size_t)i].layer);
        CHECK_EQ(e.expert, entries[(size_t)i].expert);
        CHECK_EQ(e.slot, entries[(size_t)i].slot);
        for (uint32_t k = 0; k < e.n_planes; ++k, ++pi) {
            REQUIRE(pi < planes.size());
            CHECK_EQ(f.plane(e, (int)k).offset, planes[pi].offset);
            CHECK_EQ(f.plane(e, (int)k).bytes, planes[pi].bytes);
        }
    }
    /* THE OLD BLOB, BYTE FOR BYTE. Everything past the header is untouched by the extension
     * (checked above against the pre-extension file), so this reads the same bytes the old
     * mapping did -- without depending on a header that has since been rewritten. */
    for (int64_t i = 0; i < n_old; ++i)
        for (uint32_t k = 0; k < f.entry(i).n_planes; ++k) {
            const RadFilePlane& p = f.plane(f.entry(i), (int)k);
            REQUIRE((int64_t)p.offset + (int64_t)p.bytes <= (int64_t)before.size());
            CHECK_EQ(std::memcmp(f.data(p), before.data() + p.offset, (size_t)p.bytes), 0);
        }
    CHECK_EQ(f.header().data_off, data_off);

    /* THE NEW WEIGHT, past the old end and aligned. */
    const RadFileEntry* nd = f.find("dflash.norm.weight");
    REQUIRE(nd != nullptr);
    CHECK((int64_t)nd->offset >= (int64_t)before.size());
    CHECK_EQ((int64_t)(nd->offset % RAD_ALIGN_UNIT), (int64_t)0);
    CHECK_EQ(std::memcmp(f.data(*nd), pd[0].data(), (size_t)pd[0].size()), 0);

    /* THE VOCABULARY: token by token, and the template. */
    VocabBuild vb;
    REQUIRE((vocab_deserialize(f.base(), (uint64_t)f.size(), f.vocab().h, vb)) >= 0);
    CHECK_EQ(vb.tokens.size(), (size_t)vh.n_tokens);
    CHECK(vb.tokens == std::vector<std::string>({ "a", "b", "ab", "c" }));
    CHECK_EQ(vb.chat_template, std::string("template"));
    CHECK_EQ(vb.eos, 2);

    /* THE METADATA: the original keys survive beside the new ones, as the rows they were. */
    CHECK_EQ(std::string(f.get_s("note", "")), std::string("original"));
    CHECK_EQ(std::string(f.get_s("draft.architectures", "")), std::string("DFlash2DraftModel"));
    int64_t bf16 = 0;
    CHECK(f.get_i("draft.bf16", &bf16));
    CHECK_EQ(bf16, (int64_t)1);
    /* THE FREE-FORM ROWS KEPT THEIR TYPES AND THEIR VALUES. `get_i` matches only an INT row and
     * `get_f` reads an F64 row, so these are the direct test of the row typing an in-place
     * extension has to preserve -- a writer that emitted MetaOwn's text as strings fails both
     * lookups while the file still parses. The f64 is the value that survives only when the
     * container's %.17g spelling is read back exactly. */
    int64_t hidden = 0;
    CHECK(f.get_i("text_config.hidden_size", &hidden));
    CHECK_EQ(hidden, (int64_t)2560);
    double scale = 0;
    CHECK(f.get_f("text_config.test_scale", &scale));
    CHECK_EQ(scale, 0.12345678901234567);
    /* AND THE ROW IS AN F64 ROW, not an INT one `get_f` would also narrow: an integer row cannot
     * hold this value, but the type is what a later reader keys on, so it is checked directly. */
    bool f64_row = false;
    for (int64_t i = 0; i < f.meta_count(); ++i)
        if (std::strcmp(f.str(f.meta_at(i).key), "text_config.test_scale") == 0)
            f64_row = f.meta_at(i).type == RAD_KV_F64;
    CHECK(f64_row);
    /* The typed model fields are INT/F64 rows as well, not just present. */
    int64_t n_embd = 0;
    CHECK(f.get_i("n_embd", &n_embd));
    CHECK_EQ(n_embd, (int64_t)8);
    CHECK_EQ(std::string(f.str(f.header().arch_id)), std::string("radtest"));

    /* THE WAY BACK: writing the saved header and truncating restores the file byte for byte. */
    const std::string bak = slurp(b.path + ".pre-append");
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
    old.close();

    clean_dir(dir);
}

RAD_TEST_MAIN()
