/* fr_spec_test.cpp -- test reduced vocabulary draft head (FR-Spec) from mtp-lm-head-reduced.safetensors. */
#include "rad_test.h"
#include "format/checkpoint.h"
#include "sched/advance.h"
#include "sched/advance_impl.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace rad;

namespace {

RadNameMap map(const char* declared, int mode, std::vector<const char*> src) {
    RadNameMap m{};
    m.declared = declared;
    m.mode = mode;
    m.n_src = (int)src.size();
    for (int i = 0; i < 8; ++i) m.src_index[i] = -1;
    for (size_t i = 0; i < src.size(); ++i) m.src[i] = src[i];
    return m;
}

}  // namespace

TEST(fr_spec_model_reduced_head_loads_from_disk) {
    const std::string model_dir = "/root/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ-v2";
    if (::access((model_dir + "/mtp-lm-head-reduced.safetensors").c_str(), F_OK) != 0) {
        // Model not present in environment, skip gracefully
        return;
    }

    Checkpoint ck;
    REQUIRE_EQ(ck.open(model_dir), RAD_OK);

    // 1. Check vocab_ids tensor
    const SrcTensor* vt = ck.find("mtp.lm_head_reduced.vocab_ids");
    REQUIRE(vt != nullptr);
    CHECK_EQ(vt->numel(), (int64_t)52416);
    CHECK_EQ(vt->rad_dtype, (uint32_t)RAD_I64);

    const int64_t* ids = static_cast<const int64_t*>(vt->data);
    CHECK(ids[0] >= (int64_t)0);
    CHECK(ids[0] < (int64_t)250000);

    // 2. Check Q4 weights and companion scale/zero
    const SrcTensor* wq = ck.find("mtp.lm_head_reduced.weight_q4");
    REQUIRE(wq != nullptr);
    CHECK_EQ(wq->shape.size(), (size_t)2);
    CHECK_EQ(wq->shape[0], (int64_t)52416);
    CHECK_EQ(wq->shape[1], (int64_t)1280);

    const SrcTensor* ws = ck.find("mtp.lm_head_reduced.weight_scale");
    REQUIRE(ws != nullptr);
    CHECK_EQ(ws->shape.size(), (size_t)2);
    CHECK_EQ(ws->shape[0], (int64_t)52416);
    CHECK_EQ(ws->shape[1], (int64_t)20);

    const SrcTensor* wz = ck.find("mtp.lm_head_reduced.weight_zero");
    REQUIRE(wz != nullptr);
    CHECK_EQ(wz->shape.size(), (size_t)2);
    CHECK_EQ(wz->shape[0], (int64_t)52416);
    CHECK_EQ(wz->shape[1], (int64_t)20);

    // 3. Resolve mtp.draft_head_reduced logical weight
    Program prog;
    prog.name_map.push_back(map("mtp.draft_head_reduced", RAD_MAP_COPY, { "mtp.lm_head_reduced.weight_q4" }));

    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "mtp.draft_head_reduced", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    CHECK_EQ(w.rows, (int64_t)52416);
    CHECK_EQ(w.cols, (int64_t)2560);
    CHECK_EQ(w.enc.n_planes, 3);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_U4);
    CHECK_EQ(w.enc.plane[0].block[0], (int64_t)1);
    CHECK_EQ(w.enc.plane[0].block[1], (int64_t)1);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_F16);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)1);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)128);
    CHECK_EQ(w.enc.plane[2].dtype, (uint32_t)RAD_I16);
    CHECK_EQ(w.enc.plane[2].block[0], (int64_t)1);
    CHECK_EQ(w.enc.plane[2].block[1], (int64_t)128);

    // 4. Test reading plane rows
    std::vector<uint8_t> r_codes(2 * 1280);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 2, r_codes.data(), &why), RAD_OK);
    std::vector<uint16_t> r_scales(2 * 20);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 2, r_scales.data(), &why), RAD_OK);
    std::vector<int16_t> r_zeroes(2 * 20);
    REQUIRE_EQ(ckpt_plane_rows(w, 2, 0, 2, r_zeroes.data(), &why), RAD_OK);

    // Check zeroes are in valid range 1..14 (center 8)
    for (int i = 0; i < 40; ++i) {
        CHECK(r_zeroes[i] >= 1);
        CHECK(r_zeroes[i] <= 15);
    }
}

TEST(fr_spec_advance_vocab_translation) {
    // Test that adv_row correctly maps draft token indices to vocab_ids in ADV_ROUND and ADV_TAKE
    std::vector<int32_t> mock_vocab_ids = { 1000, 2000, 3000, 4000, 5000 };
    StepSlot slots[4]{};

    // Create slab buffer for advance args
    // Layout: row_off, tok_off, pos_off, ctx_off, sid_off, cu_off, q_off, acc_off, out_off
    int32_t slab_data[256]{};
    // row_pitch = 4
    // rows[0..3] = flag (1), rtok, rout, aux
    slab_data[0] = 1; // row 0 active
    slab_data[4] = 0; // rtok
    slab_data[8] = 0; // rout
    slab_data[12] = 1; // aux

    int32_t proposals[4] = { 2, 0, 1, 3 }; // rank proposals (indices into mock_vocab_ids)

    AdvanceArgs a{};
    a.mode = ADV_ROUND;
    a.round = 2;
    a.n_seq = 1;
    a.slab = reinterpret_cast<char*>(slab_data);
    a.row_off = 0;
    a.row_pitch = 4;
    a.tok_off = 16 * sizeof(int32_t);
    a.pos_off = 32 * sizeof(int32_t);
    a.ctx_off = 48 * sizeof(int32_t);
    a.sid_off = 64 * sizeof(int32_t);
    a.cu_off = 80 * sizeof(int32_t);
    a.q_off = 96 * sizeof(int32_t);
    a.st = slots;
    a.src = proposals;
    a.src_pitch = 1;
    a.vocab_ids = mock_vocab_ids.data();
    a.n_vocab_ids = (int32_t)mock_vocab_ids.size();

    // In slot 0: head = 10
    slots[0].head = 10;

    adv_row(a, 0);

    // Row 0 proposed index 2 -> should map to mock_vocab_ids[2] = 3000
    CHECK_EQ(slots[0].tok[1], 3000);

    // Now test ADV_TAKE (e.g. final draft token)
    a.mode = ADV_TAKE;
    a.round = 3;
    adv_row(a, 0);
    CHECK_EQ(slots[0].tok[3], 3000);
}

RAD_TEST_MAIN()
