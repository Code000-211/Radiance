/* shipped_plugins_test.cpp -- the plugins this tree ships, loaded through the engine's loader.
 *
 * rad-avx-check keeps its own copy of the loader's schema comparison, and only rad-kbench and the
 * out-of-tree test load the shipped libraries through the Registry, both slowly. A change to the
 * loader's rules, or to a shipped plugin's schemas or selection key, fails here in milliseconds.
 */
#include "rad_test.h"

#include "plugin/registry.h"

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace rad;

namespace {

std::string dir_of(const char* sub) { return home() + "/" + sub; }

std::string errors(const Registry& r) {
    std::string s;
    for (const auto& e : r.load_errors()) s += "    " + e + "\n";
    return s;
}

}  /* namespace */

/* The CPU-only engine's kernels: libavx ahead of libref, as `--kernels libavx,libref` orders them.
 * They load with no refusal (the loader refuses two plugins that disagree on an op's schema),
 * libref is the reference and ranks below, and every op libavx serves libref serves too, so a
 * libavx row that declines always has a reference behind it. */
TEST(the_host_kernel_libraries_load_together_through_the_loader) {
    Registry r;
    const int st = r.load_dir(dir_of("kernels"), { "libavx", "libref" });
    if (st == RAD_E_IO && r.plugins().empty())
        ARTIFACT_MISSING("the kernel libraries under " + dir_of("kernels"));
    if (!r.load_errors().empty()) fprintf(stderr, "%s", errors(r).c_str());
    CHECK_OK(st);
    CHECK(r.load_errors().empty());
    const Plugin* avx = r.plugin("libavx");
    const Plugin* ref = r.plugin("libref");
    REQUIRE(avx != nullptr && ref != nullptr);
    CHECK(avx->order < ref->order);
    CHECK(ref->reference);
    CHECK(!avx->reference);
    CHECK(avx->n_kernels > 0 && ref->n_kernels > 0);
    std::set<std::string> avx_ops, ref_ops;
    for (const KernelRow& k : r.rows()) {
        if (k.plugin == avx->name) avx_ops.insert(k.info->op);
        if (k.plugin == ref->name) ref_ops.insert(k.info->op);
    }
    int orphans = 0;
    for (const auto& op : avx_ops) {
        if (!ref_ops.count(op)) {
            fprintf(stderr, "    libavx serves '%s' and libref does not\n", op.c_str());
            ++orphans;
        }
        CHECK(r.schema(op) != nullptr);
    }
    CHECK_EQ(orphans, 0);
    CHECK(avx_ops.size() > 50);
    /* Every op has one schema, fixed by the first library to declare it. */
    for (const auto& op : r.op_names()) {
        const Plugin* owner = r.schema_owner(op);
        REQUIRE(owner != nullptr);
        if (avx_ops.count(op)) CHECK_EQ(owner->name, avx->name);
    }
}

/* Every shipped architecture plugin loads, exports declare and step, and claims its own
 * (architecture id, quantisation descriptor) pair, the key a container selects it by: two
 * plugins on one pair would make the choice arbitrary. */
TEST(the_shipped_architectures_load_with_distinct_selection_keys) {
    Registry r;
    const int st = r.load_dir(dir_of("architectures"), {});
    if (st == RAD_E_IO && r.plugins().empty())
        ARTIFACT_MISSING("the architecture plugins under " + dir_of("architectures"));
    if (!r.load_errors().empty()) fprintf(stderr, "%s", errors(r).c_str());
    CHECK_OK(st);
    std::map<std::pair<std::string, std::string>, std::string> keys;
    for (const Plugin& p : r.plugins()) {
        if (p.kind != RAD_PLUGIN_ARCH) continue;
        CHECK(p.arch_declare != nullptr);
        CHECK(p.arch_step != nullptr);
        CHECK(keys.emplace(std::make_pair(p.arch_id, p.arch_quant), p.name).second);
    }
    const std::set<std::pair<std::string, std::string>> want = {
        { "anima", "" }, { "breeze", "" }, { "embeddinggemma2", "" }, { "exl3import", "exl3" },
        { "flux3", "" },
        { "flux3distilled", "" },
        { "gemma3", "" }, { "gemma4", "" },
        { "gemma4", "compressed-tensors" }, { "glm5next", "" }, { "lfm2", "" }, { "llama", "" },
        { "llama", "fp8_e4m3" }, { "qwen35", "" },
        { "qwen35", "fp8_e4m3" }, { "qwen35", "w8a8_fwht128" }, { "qwen35moe", "fp8_e4m3" },
        { "qwen3asr", "" }, { "krea2", "" }, { "ltx2", "" }, { "minimaxh3", "" },
        { "minimaxmusic3", "" }, { "qwen4exp", "" }, { "qwenimage21", "" }, { "sdxl", "" } };
    std::set<std::pair<std::string, std::string>> got;
    for (const auto& [k, name] : keys) got.insert(k);
    if (got != want)
        for (const auto& [k, name] : keys)
            fprintf(stderr, "    %s claims (%s, '%s')\n", name.c_str(), k.first.c_str(),
                    k.second.c_str());
    CHECK(got == want);
}

/* A checkpoint whose quantisation was imported weight by weight (Quark, ParoQuant, GPTQ, NVFP4)
 * names a tool no plugin claims, so the family's one quantised plugin serves it, and two are
 * refused (startup.h). So each family has at most one quantised plugin that takes imports:
 * qwen35_w8a8 declines them (rad_arch_serves_imports), leaving Qwen3.8 imports to qwen35_fp8. */
TEST(every_shipped_family_has_at_most_one_plugin_for_an_imported_checkpoint) {
    Registry r;
    const int st = r.load_dir(dir_of("architectures"), {});
    if (st == RAD_E_IO && r.plugins().empty())
        ARTIFACT_MISSING("the architecture plugins under " + dir_of("architectures"));
    CHECK_OK(st);
    std::map<std::string, std::vector<std::string>> takers;
    for (const Plugin& p : r.plugins())
        if (p.kind == RAD_PLUGIN_ARCH && !p.arch_quant.empty() && p.arch_serves_imports)
            takers[p.arch_id].push_back(p.name);
    for (const auto& [id, names] : takers) {
        if (names.size() > 1)
            for (const auto& n : names)
                fprintf(stderr, "    '%s' would take an imported %s checkpoint\n", n.c_str(),
                        id.c_str());
        CHECK_EQ(names.size(), 1u);
    }
    REQUIRE(takers.count("qwen35"));
    CHECK_EQ(takers["qwen35"][0], std::string("qwen35_fp8"));
    const Plugin* w8 = r.plugin("qwen35_w8a8");
    REQUIRE(w8 != nullptr);
    CHECK(!w8->arch_serves_imports);
}

RAD_TEST_MAIN()
