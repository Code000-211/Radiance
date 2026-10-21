/* Copyright 2026 Dylan Johnston and tcclaviger
 * SPDX-License-Identifier: Apache-2.0 */
/* ridgefill_interop_test.cpp -- unit test for RidgeFill projector runtime interop,
 * dynamic kernel symbols binding, safetensors parsing, and bundle management.
 */
#include "rad_test.h"
#include "runtime/rad_ridgefill_wrapper.h"

#include <cstring>
#include <fstream>
#include <vector>

using namespace rad;
using namespace rad::ridgefill;

/* ================================================================== Mode & Flag Normalization */

TEST(ridgefill_mode_parsing_and_env_normalization) {
    CHECK_EQ(parse_mode("speed"), Mode::Speed);
    CHECK_EQ(parse_mode("SPEED"), Mode::Speed);
    CHECK_EQ(parse_mode("1"), Mode::Speed);
    CHECK_EQ(parse_mode("on"), Mode::Speed);
    CHECK_EQ(parse_mode("true"), Mode::Speed);

    CHECK_EQ(parse_mode("quality"), Mode::Quality);
    CHECK_EQ(parse_mode("QUALITY"), Mode::Quality);

    CHECK_EQ(parse_mode("plumb"), Mode::Plumb);
    CHECK_EQ(parse_mode("PLUMB"), Mode::Plumb);

    CHECK_EQ(parse_mode("off"), Mode::Off);
    CHECK_EQ(parse_mode("OFF"), Mode::Off);
    CHECK_EQ(parse_mode("0"), Mode::Off);
    CHECK_EQ(parse_mode("false"), Mode::Off);
    CHECK_EQ(parse_mode(""), Mode::Off);
    CHECK_EQ(parse_mode(nullptr), Mode::Off);
    CHECK_EQ(parse_mode("unknown_mode"), Mode::Off);

    /* Test normalization from env and flags */
    setenv("RADIANCE_RIDGEFILL", "1", 1);
    setenv("RADIANCE_RIDGEFILL_PROJECTOR", "/custom/projector/path", 1);
    InteropConfig cfg = InteropConfig::from_env_and_flags();
    CHECK_EQ(cfg.mode, Mode::Speed);
    CHECK_EQ(cfg.projector_dir, std::string("/custom/projector/path"));

    /* Verify apply_to_env() normalizes "1" to "speed" in process environment */
    cfg.apply_to_env();
    const char* env_after = std::getenv("RADIANCE_RIDGEFILL");
    CHECK(env_after != nullptr);
    CHECK_EQ(std::string(env_after), std::string("speed"));
}

/* ================================================================== Kernel Plugin Dynamic Symbols */

TEST(ridgefill_kernel_symbols_binding) {
    InteropConfig cfg;
    std::string so_path = cfg.find_kernel_library();
    if (so_path.empty() || access(so_path.c_str(), R_OK) != 0) {
        /* Fallback check in /tmp/rf_kernels_build */
        so_path = "/tmp/rf_kernels_build/kernels/ridgefill.so";
    }

    if (access(so_path.c_str(), R_OK) != 0) {
        printf("  SKIP ridgefill.so not found at %s\n", so_path.c_str());
        return;
    }

    KernelSymbols syms;
    std::string err;
    bool ok = syms.bind(so_path, &err);
    if (!ok) {
        printf("  FAIL bind error: %s\n", err.c_str());
    }
    CHECK(ok);

    /* Verify standard Radiance plugin ABI lifecycle symbols */
    CHECK(syms.abi_version != nullptr);
    CHECK_EQ((int)syms.abi_version(), (int)RAD_ABI_VERSION);

    CHECK(syms.plugin_info != nullptr);
    const RadPluginInfo* info = syms.plugin_info();
    CHECK(info != nullptr);
    CHECK_EQ(std::string(info->name), std::string("ridgefill"));

    CHECK(syms.kernel_count != nullptr);
    CHECK(syms.kernel_count() >= 14);

    CHECK(syms.schema_count != nullptr);
    CHECK_EQ(syms.schema_count(), 9);

    CHECK(syms.kernel_concurrent != nullptr);

    /* Forwarded engine GEMM entrypoints */
    CHECK(syms.forward_count != nullptr);
    CHECK(syms.forward_at != nullptr);
    CHECK(syms.forward_concurrent != nullptr);
    CHECK_EQ(syms.kernel_count(), 14 + syms.forward_count());

    /* Verify all 7 op device/host/parse entrypoints are resolved */
    CHECK(syms.mask_device != nullptr);
    CHECK(syms.mask_host != nullptr);
    CHECK(syms.mask_parse != nullptr);

    CHECK(syms.select_device != nullptr);
    CHECK(syms.select_host != nullptr);
    CHECK(syms.select_parse != nullptr);

    CHECK(syms.drop_device != nullptr);
    CHECK(syms.drop_host != nullptr);
    CHECK(syms.drop_parse != nullptr);

    CHECK(syms.rho_device != nullptr);
    CHECK(syms.rho_host != nullptr);
    CHECK(syms.rho_parse != nullptr);

    CHECK(syms.correct_device != nullptr);
    CHECK(syms.correct_host != nullptr);
    CHECK(syms.correct_parse != nullptr);

    CHECK(syms.state_read_device != nullptr);
    CHECK(syms.state_read_host != nullptr);
    CHECK(syms.state_read_parse != nullptr);

    CHECK(syms.hazard_device != nullptr);
    CHECK(syms.hazard_host != nullptr);
    CHECK(syms.hazard_parse != nullptr);
}

/* ================================================================== Safetensors File Mapping */

TEST(ridgefill_safetensors_file_mapping) {
    const std::string path = "/tmp/rf_interop_fixtures/sample.safetensors";
    if (access(path.c_str(), R_OK) != 0) {
        printf("  SKIP sample fixture not present\n");
        return;
    }

    SafetensorsFile sf;
    std::string err;
    bool ok = sf.open_and_map(path, &err);
    if (!ok) {
        printf("  FAIL open error: %s\n", err.c_str());
    }
    CHECK(ok);

    /* Verify metadata */
    const auto& meta = sf.metadata();
    auto it_auth = meta.find("author");
    CHECK(it_auth != meta.end());
    CHECK_EQ(it_auth->second, std::string("radiance-test"));

    /* Verify weight_bf16 tensor */
    const MappedTensor* t_bf16 = sf.find("weight_bf16");
    CHECK(t_bf16 != nullptr);
    CHECK_EQ((int)t_bf16->dtype, (int)RAD_BF16);
    CHECK_EQ(t_bf16->shape.size(), (size_t)2);
    CHECK_EQ(t_bf16->shape[0], (int64_t)4);
    CHECK_EQ(t_bf16->shape[1], (int64_t)8);
    CHECK_EQ(t_bf16->numel, (int64_t)32);
    CHECK_EQ(t_bf16->bytes, (int64_t)64);
    CHECK(t_bf16->data != nullptr);

    /* Verify code_i8 tensor */
    const MappedTensor* t_i8 = sf.find("code_i8");
    CHECK(t_i8 != nullptr);
    CHECK_EQ((int)t_i8->dtype, (int)RAD_I8);
    CHECK_EQ(t_i8->shape.size(), (size_t)2);
    CHECK_EQ(t_i8->shape[0], (int64_t)4);
    CHECK_EQ(t_i8->shape[1], (int64_t)16);
    CHECK_EQ(t_i8->numel, (int64_t)64);
    CHECK_EQ(t_i8->bytes, (int64_t)64);

    /* Verify scale_f32 tensor */
    const MappedTensor* t_f32 = sf.find("scale_f32");
    CHECK(t_f32 != nullptr);
    CHECK_EQ((int)t_f32->dtype, (int)RAD_F32);
    CHECK_EQ(t_f32->shape.size(), (size_t)1);
    CHECK_EQ(t_f32->shape[0], (int64_t)4);
    CHECK_EQ(t_f32->numel, (int64_t)4);
    CHECK_EQ(t_f32->bytes, (int64_t)16);

    /* Non-existent tensor lookup */
    CHECK(sf.find("non_existent_tensor") == nullptr);
}

/* ================================================================== Safetensors Error Handling */

TEST(ridgefill_safetensors_error_handling) {
    SafetensorsFile sf;
    std::string err;

    /* Non-existent file */
    CHECK(!sf.open_and_map("/tmp/does_not_exist_safetensors.bin", &err));
    CHECK(!err.empty());

    /* Truncated / malformed file */
    const std::string bad_file = "/tmp/bad_safetensors.bin";
    {
        std::ofstream out(bad_file, std::ios::binary);
        uint64_t fake_len = 999999999;
        out.write(reinterpret_cast<const char*>(&fake_len), 8);
    }
    CHECK(!sf.open_and_map(bad_file, &err));
    CHECK(!err.empty());
    unlink(bad_file.c_str());
}

/* ================================================================== Projector Bundle */

TEST(ridgefill_projector_bundle_loading) {
    const std::string bundle_dir = "/tmp/rf_interop_fixtures/bundle";
    if (access((bundle_dir + "/ridgefill.json").c_str(), R_OK) != 0) {
        printf("  SKIP bundle fixture not present\n");
        return;
    }

    ProjectorBundle bundle;
    std::string err;
    bool ok = bundle.open(bundle_dir, &err);
    if (!ok) {
        printf("  FAIL bundle open error: %s\n", err.c_str());
    }
    CHECK(ok);

    CHECK_EQ(bundle.name(), std::string("test-projector-bundle"));
    CHECK(bundle.is_int8());
    CHECK_EQ(bundle.split(), (int64_t)24);

    /* Test layer 24 int8 retrieval */
    const int8_t* codes = nullptr;
    const uint16_t* scale = nullptr;
    const uint16_t* bias = nullptr;
    int64_t n = 0, k = 0;
    bool got_l24 = bundle.get_layer_i8(24, &codes, &scale, &bias, &n, &k);
    CHECK(got_l24);
    CHECK(codes != nullptr);
    CHECK(scale != nullptr);
    CHECK(bias != nullptr);
    CHECK_EQ(n, (int64_t)2560);
    CHECK_EQ(k, (int64_t)10240);

    /* Test correction retrieval */
    const MappedTensor* t_corr = bundle.get_correction(24);
    CHECK(t_corr != nullptr);
    CHECK_EQ((int)t_corr->dtype, (int)RAD_F32);
    CHECK_EQ(t_corr->numel, (int64_t)(48 * 128 * 128));

    /* Test rowsel retrieval */
    const MappedTensor* t_rowsel = bundle.get_rowsel("score");
    CHECK(t_rowsel != nullptr);
    CHECK_EQ((int)t_rowsel->dtype, (int)RAD_F32);
    CHECK_EQ(t_rowsel->numel, (int64_t)248320);

    /* Requesting layer below split */
    CHECK(!bundle.get_layer_i8(23, &codes, &scale, &bias, &n, &k));
}

RAD_TEST_MAIN()
