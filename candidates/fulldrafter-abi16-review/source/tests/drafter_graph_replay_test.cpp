#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include "arch/common/rad_block_fulldrafter.h"
#include "libr4d/r4d_fulldrafter.h"
#include "abi/rad_abi.h"

using namespace radiance;

#define SCR_PER_SEQ         8388608ULL
#define SCR_PREV_TOK_OFF    196608

typedef int (*r4d_fulldrafter_two_stage_fn)(const RadArgs* a, RadStream s);

int main(int argc, char** argv) {
    std::string sidecar_path = "/home/orca/drafter_data/checkpoints/flash_next_k12_exp/drafter_step_92500.safetensors";
    std::string teacher_path = "/home/orca/drafter_data/teacher/teacher-matrices.safetensors";
    std::string r4d_path = "/home/orca/RadianceMine/worktrees/tap_record/build_hip/radiance_home/kernels/libr4d.so";

    std::printf("=================================================================\n");
    std::printf("DRAFTER GRAPH REPLAY REGRESSION TEST: ANCHOR DEVICE-BUFFER UPDATE\n");
    std::printf("=================================================================\n");

    void* lib = dlopen(r4d_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        std::fprintf(stderr, "Failed to load %s: %s\n", r4d_path.c_str(), dlerror());
        return 1;
    }

    auto fn = (r4d_fulldrafter_two_stage_fn)dlsym(lib, "r4d_rad_fulldrafter_two_stage");
    if (!fn) {
        std::fprintf(stderr, "Failed to find r4d_rad_fulldrafter_two_stage: %s\n", dlerror());
        return 1;
    }

    radiance::FullDrafterWeights weights{};
    std::string err;
    if (!radiance::FullDrafterLoader::load_weights(sidecar_path, &weights, nullptr, &err, 2)) {
        std::fprintf(stderr, "Failed to load sidecar: %s\n", err.c_str());
        return 1;
    }
    if (!radiance::FullDrafterLoader::load_teacher_matrices(teacher_path, &weights, &err)) {
        std::fprintf(stderr, "Failed to load teacher: %s\n", err.c_str());
        return 1;
    }

    hipSetDevice(0);
    hipStream_t stream = nullptr;
    (void)hipStreamCreate(&stream);

    const int W = 2560;
    const int M = 1;
    void *d_k0 = nullptr, *d_k1 = nullptr, *d_k2 = nullptr, *d_anc = nullptr;
    void *d_toks = nullptr, *d_out = nullptr, *d_scratch = nullptr;

    (void)hipMalloc(&d_k0, M * W * sizeof(radiance::rad_bf16_t));
    (void)hipMalloc(&d_k1, M * W * sizeof(radiance::rad_bf16_t));
    (void)hipMalloc(&d_k2, M * W * sizeof(radiance::rad_bf16_t));
    (void)hipMalloc(&d_anc, M * W * sizeof(radiance::rad_bf16_t));
    (void)hipMalloc(&d_toks, 7 * sizeof(int32_t));
    (void)hipMalloc(&d_out, 12 * sizeof(uint32_t));
    (void)hipMalloc(&d_scratch, SCR_PER_SEQ);

    (void)hipMemset(d_k0, 0, M * W * sizeof(radiance::rad_bf16_t));
    (void)hipMemset(d_k1, 0, M * W * sizeof(radiance::rad_bf16_t));
    (void)hipMemset(d_k2, 0, M * W * sizeof(radiance::rad_bf16_t));
    (void)hipMemset(d_scratch, 0, SCR_PER_SEQ);

    RadTensor t[7]{};
    t[0].data = d_k0;     t[0].shape[0] = M; t[0].shape[1] = W; t[0].dtype = RAD_BF16;
    t[1].data = d_k1;     t[1].shape[0] = M; t[1].shape[1] = W; t[1].dtype = RAD_BF16;
    t[2].data = d_k2;     t[2].shape[0] = M; t[2].shape[1] = W; t[2].dtype = RAD_BF16;
    t[3].data = d_anc;    t[3].shape[0] = M; t[3].shape[1] = W; t[3].dtype = RAD_BF16;
    t[4].data = d_toks;   t[4].shape[0] = 7; t[4].shape[1] = 1; t[4].dtype = RAD_I32;
    t[5].data = const_cast<FullDrafterGpuPointers*>(weights.ranks[0].d_gpu_pointers);
    t[5].shape[0] = 1;    t[5].shape[1] = 1; t[5].dtype = RAD_BF16;
    t[6].data = d_out;    t[6].shape[0] = 1;  t[6].shape[1] = 12; t[6].dtype = RAD_I32;

    RadParam params[2];
    params[0].key = "M";            params[0].kind = RAD_P_INT; params[0].ival = 1;
    params[1].key = "block_stride"; params[1].kind = RAD_P_INT; params[1].ival = 7;

    RadArgs args{};
    args.t = t;
    args.n_t = 7;
    args.p = params;
    args.n_p = 2;
    args.scratch = d_scratch;
    args.scratch_bytes = SCR_PER_SEQ;

    const auto* host_emb = weights.ranks[0].host_teacher_embedding_table;

    // --- SETUP ANCHOR 1 ---
    int32_t anchor1 = 151644;
    int32_t h_toks1[7] = { anchor1, 151643, 151643, 151643, 151643, 151643, 151643 };
    (void)hipMemcpy(d_toks, h_toks1, sizeof(h_toks1), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_anc, host_emb + (size_t)anchor1 * W, W * sizeof(radiance::rad_bf16_t), hipMemcpyHostToDevice);

    std::printf("[STEP 1] Capturing Graph with Anchor ID = %d...\n", anchor1);
    hipGraph_t graph = nullptr;
    hipGraphExec_t instance = nullptr;

    (void)hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    int rc = fn(&args, (RadStream)stream);
    assert(rc == 0);
    (void)hipStreamEndCapture(stream, &graph);
    assert(graph != nullptr);

    (void)hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    assert(instance != nullptr);
    std::printf("  Graph captured and instantiated successfully.\n");

    // --- REPLAY 1 ---
    std::printf("[STEP 2] Replay 1 (Expecting Anchor = %d)...\n", anchor1);
    (void)hipGraphLaunch(instance, stream);
    (void)hipStreamSynchronize(stream);

    uint32_t prop1[12];
    (void)hipMemcpy(prop1, d_out, sizeof(prop1), hipMemcpyDeviceToHost);

    uint32_t read_anchor1 = 0;
    (void)hipMemcpy(&read_anchor1, (char*)d_scratch + SCR_PREV_TOK_OFF, sizeof(uint32_t), hipMemcpyDeviceToHost);
    std::printf("  Replay 1 proposals: [%u, %u, %u, %u, %u, %u]\n",
                prop1[0], prop1[1], prop1[2], prop1[3], prop1[4], prop1[5]);
    std::printf("  Scratch read anchor: %u (expected %d)\n", read_anchor1, anchor1);
    assert(read_anchor1 == (uint32_t)anchor1);

    // --- UPDATE DEVICE BUFFER FOR REPLAY 2 (DIFFERENT ANCHOR) ---
    int32_t anchor2 = 9906;
    std::printf("[STEP 3] Updating Device Buffer on Stream to Anchor ID = %d (NO Graph Recapture)...\n", anchor2);
    int32_t h_toks2[7] = { anchor2, 151643, 151643, 151643, 151643, 151643, 151643 };
    (void)hipMemcpyAsync(d_toks, h_toks2, sizeof(h_toks2), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(d_anc, host_emb + (size_t)anchor2 * W, W * sizeof(radiance::rad_bf16_t), hipMemcpyHostToDevice, stream);

    // --- REPLAY 2 ---
    std::printf("[STEP 4] Replay 2 (Expecting Anchor = %d)...\n", anchor2);
    (void)hipGraphLaunch(instance, stream);
    (void)hipStreamSynchronize(stream);

    uint32_t prop2[12];
    (void)hipMemcpy(prop2, d_out, sizeof(prop2), hipMemcpyDeviceToHost);

    uint32_t read_anchor2 = 0;
    (void)hipMemcpy(&read_anchor2, (char*)d_scratch + SCR_PREV_TOK_OFF, sizeof(uint32_t), hipMemcpyDeviceToHost);
    std::printf("  Replay 2 proposals: [%u, %u, %u, %u, %u, %u]\n",
                prop2[0], prop2[1], prop2[2], prop2[3], prop2[4], prop2[5]);
    std::printf("  Scratch read anchor: %u (expected %d)\n", read_anchor2, anchor2);
    assert(read_anchor2 == (uint32_t)anchor2);
    assert(prop1[0] != prop2[0]); // Proposal must differ because conditioning changed!

    // --- STEP 5: VERIFY BITWISE PARITY WITH FRESH UNCAPTURED EXECUTION ---
    std::printf("[STEP 5] Comparing Replay 2 against fresh uncaptured run with Anchor = %d...\n", anchor2);
    (void)hipMemset(d_out, 0, 12 * sizeof(uint32_t));
    rc = fn(&args, (RadStream)stream);
    assert(rc == 0);
    (void)hipStreamSynchronize(stream);

    uint32_t prop_fresh[12];
    (void)hipMemcpy(prop_fresh, d_out, sizeof(prop_fresh), hipMemcpyDeviceToHost);

    bool bitwise_match = (std::memcmp(prop2, prop_fresh, sizeof(prop2)) == 0);
    std::printf("  Bitwise match Replay 2 vs Fresh Run: %s\n", bitwise_match ? "YES (PASSED)" : "NO (FAILED)");
    assert(bitwise_match);

    std::printf("\n>>> REGRESSION TEST PASSED: Graph replays correctly propagate device-buffer anchor updates! <<<\n");

    // Cleanup
    (void)hipGraphExecDestroy(instance);
    (void)hipGraphDestroy(graph);
    (void)hipFree(d_k0); (void)hipFree(d_k1); (void)hipFree(d_k2); (void)hipFree(d_anc);
    (void)hipFree(d_toks); (void)hipFree(d_out); (void)hipFree(d_scratch);
    (void)hipStreamDestroy(stream);
    dlclose(lib);

    return 0;
}
