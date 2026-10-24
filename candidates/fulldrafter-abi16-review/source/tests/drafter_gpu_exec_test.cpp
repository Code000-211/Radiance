#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include "arch/common/rad_block_fulldrafter.h"
#include "libr4d/r4d_fulldrafter.h"
#include "abi/rad_abi.h"

using namespace radiance;

int main() {
    setenv("RADIANCE_DLOOP_THETA", "0", 0);
    std::printf("==================================================\n");
    std::printf(" FullDrafter GPU Kernel Execution Test\n");
    std::printf("==================================================\n");

    const char* env_sidecar = std::getenv("RADIANCE_FULLDRAFTER_SIDECAR");
    const char* sidecar_path = env_sidecar ? env_sidecar : "/home/orca/drafter_data/checkpoints/flash_next_k12_exp/drafter_final.safetensors";
    FullDrafterWeights weights;
    std::string err;

    std::printf("[1] Loading weights...\n");
    if (!FullDrafterLoader::load_weights(sidecar_path, &weights, nullptr, &err, 2)) {
        std::fprintf(stderr, "Failed to load weights: %s\n", err.c_str());
        return 1;
    }
    std::printf("[2] Loading teacher matrices...\n");
    if (!FullDrafterLoader::load_teacher_matrices("", &weights, &err)) {
        std::fprintf(stderr, "Failed to load teacher matrices: %s\n", err.c_str());
        return 1;
    }
    std::printf("  Weights and teacher matrices loaded.\n");

    // Test on GPU 0 with local allocation of both shards
    std::printf("[3] Preparing test on GPU 0...\n");
    hipSetDevice(0);
    hipStream_t stream;
    hipStreamCreate(&stream);

    constexpr int M = 1;
    constexpr size_t W = FullDrafterConfig::TARGET_WIDTH;
    void *d_k0 = nullptr, *d_k1 = nullptr, *d_k2 = nullptr, *d_anc = nullptr;
    void *d_toks = nullptr, *d_out = nullptr, *d_scratch = nullptr;

    hipMalloc(&d_k0, M * W * 2);
    hipMalloc(&d_k1, M * W * 2);
    hipMalloc(&d_k2, M * W * 2);
    hipMalloc(&d_anc, M * W * 2);
    hipMalloc(&d_toks, M * 7 * sizeof(int32_t));
    hipMalloc(&d_out, M * 12 * sizeof(uint32_t));
    hipMalloc(&d_scratch, 2 * 8388608LL);

    hipMemset(d_k0, 0, M * W * 2);
    hipMemset(d_k1, 0, M * W * 2);
    hipMemset(d_k2, 0, M * W * 2);
    hipMemset(d_anc, 0, M * W * 2);
    hipMemset(d_toks, 0, M * 7 * sizeof(int32_t));
    hipMemset(d_out, 0, M * 12 * sizeof(uint32_t));
    hipMemset(d_scratch, 0, 2 * 8388608LL);

    RadTensor t[7]{};
    t[0].data = d_k0; t[0].shape[0] = M;
    t[1].data = d_k1; t[1].shape[0] = M;
    t[2].data = d_k2; t[2].shape[0] = M;
    t[3].data = d_anc; t[3].shape[0] = M;
    t[4].data = d_toks; t[4].shape[0] = M * 7;
    t[5].data = const_cast<FullDrafterGpuPointers*>(weights.ranks[0].d_gpu_pointers);
    t[6].data = d_out; t[6].shape[0] = M;

    RadArgs a{};
    a.n_t = 7;
    a.t = t;
    a.scratch = d_scratch;
    a.scratch_bytes = 2 * 8388608LL;

    // Load libr4d.so with fallback search paths
    const char* lib_candidates[] = {
        "./build_hip/radiance_home/kernels/libr4d.so",
        "radiance_home/kernels/libr4d.so",
        "./radiance_home/kernels/libr4d.so",
        "../build_hip/radiance_home/kernels/libr4d.so",
        "/home/orca/RadianceMine/worktrees/fulldrafter-speed-opt/build_hip/radiance_home/kernels/libr4d.so",
        nullptr
    };
    void* lib = nullptr;
    const char* loaded_path = nullptr;
    for (int i = 0; lib_candidates[i] != nullptr; ++i) {
        lib = dlopen(lib_candidates[i], RTLD_NOW);
        if (lib) {
            loaded_path = lib_candidates[i];
            break;
        }
    }
    if (!lib) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    typedef int (*OpFunc)(const RadArgs*, RadStream);
    OpFunc op_fn = (OpFunc)dlsym(lib, "r4d_rad_fulldrafter_two_stage");
    if (!op_fn) {
        std::fprintf(stderr, "dlsym failed: %s\n", dlerror());
        return 1;
    }

    std::printf("  Loaded libr4d.so from: %s\n", loaded_path ? loaded_path : "unknown");
    std::printf("[4] Invoking r4d_rad_fulldrafter_two_stage on GPU 0...\n");
    std::fflush(stdout);
    hipEvent_t ev_start, ev_stop;
    (void)hipEventCreate(&ev_start);
    (void)hipEventCreate(&ev_stop);
    (void)hipEventRecord(ev_start, stream);
    int status = op_fn(&a, (RadStream)stream);
    (void)hipEventRecord(ev_stop, stream);
    std::printf("  op returned status: %d\n", status);
    std::fflush(stdout);

    // Warmup
    for (int it = 0; it < 3; ++it) op_fn(&a, (RadStream)stream);
    hipStreamSynchronize(stream);

    // Benchmark 10 iterations
    float total_lat = 0.0f;
    float min_lat = 1e9f;
    for (int it = 0; it < 10; ++it) {
        hipEventRecord(ev_start, stream);
        status = op_fn(&a, (RadStream)stream);
        hipEventRecord(ev_stop, stream);
        hipStreamSynchronize(stream);
        float lat = 0.0f;
        hipEventElapsedTime(&lat, ev_start, ev_stop);
        total_lat += lat;
        if (lat < min_lat) min_lat = lat;
    }
    std::printf("  GPU 0 Forward pass latency: avg=%.3f ms, min=%.3f ms\n",
                total_lat / 10.0f, min_lat);
    std::fflush(stdout);

    std::vector<uint32_t> h_out(12, 0);
    hipMemcpyDtoH(h_out.data(), d_out, 12 * sizeof(uint32_t));
    std::printf("  Tokens generated: ");
    for (int i = 0; i < 12; ++i) std::printf("%u ", h_out[i]);
    std::printf("\n");

    // [5] Verify same-input M=1 vs M=8 prefix, exit-depth & halted-depth parity on GPU 0 (always-on, NDEBUG-safe)
    {
        const int M8 = 8;
        const bool two_stage_alodlm = weights.is_alodlm_loop && weights.alodlm_max_depth == 1;
        const int expected_k1 = weights.is_alodlm_loop ? (two_stage_alodlm ? 8 : 4) : 12;
        const int expected_k8 = weights.is_alodlm_loop ? (two_stage_alodlm ? 8 : 3) : 12;
        for (int i = 0; i < expected_k1; ++i) {
            if (h_out[i] == 0xFFFFFFFFu) {
                std::fprintf(stderr, "FAIL: M=1 output slot %d is unexpected sentinel!\n", i);
                return 1;
            }
        }
        for (int i = expected_k1; i < 12; ++i) {
            if (h_out[i] != 0xFFFFFFFFu) {
                std::fprintf(stderr, "FAIL: M=1 output slot %d past requested depth is not sentinel!\n", i);
                return 1;
            }
        }
        void *d8_k0 = nullptr, *d8_k1 = nullptr, *d8_k2 = nullptr, *d8_anc = nullptr;
        void *d8_toks = nullptr, *d8_out = nullptr, *d8_scr = nullptr;
        hipMalloc(&d8_k0, M8 * W * 2); hipMemset(d8_k0, 0, M8 * W * 2);
        hipMalloc(&d8_k1, M8 * W * 2); hipMemset(d8_k1, 0, M8 * W * 2);
        hipMalloc(&d8_k2, M8 * W * 2); hipMemset(d8_k2, 0, M8 * W * 2);
        hipMalloc(&d8_anc, M8 * W * 2); hipMemset(d8_anc, 0, M8 * W * 2);
        hipMalloc(&d8_toks, M8 * 7 * sizeof(int32_t)); hipMemset(d8_toks, 0, M8 * 7 * sizeof(int32_t));
        hipMalloc(&d8_out, M8 * 12 * sizeof(uint32_t)); hipMemset(d8_out, 0, M8 * 12 * sizeof(uint32_t));
        hipMalloc(&d8_scr, (size_t)M8 * 8388608ULL); hipMemset(d8_scr, 0, (size_t)M8 * 8388608ULL);
        RadTensor t8[7]{};
        t8[0].data = d8_k0; t8[0].shape[0] = M8;
        t8[1].data = d8_k1; t8[1].shape[0] = M8;
        t8[2].data = d8_k2; t8[2].shape[0] = M8;
        t8[3].data = d8_anc; t8[3].shape[0] = M8;
        t8[4].data = d8_toks; t8[4].shape[0] = M8 * 7;
        t8[5].data = const_cast<FullDrafterGpuPointers*>(weights.ranks[0].d_gpu_pointers);
        t8[6].data = d8_out; t8[6].shape[0] = M8;
        RadArgs a8{};
        a8.n_t = 7; a8.t = t8; a8.scratch = d8_scr; a8.scratch_bytes = (int64_t)M8 * 8388608LL;
        int st8 = op_fn(&a8, (RadStream)stream);
        hipStreamSynchronize(stream);
        if (st8 != 0) {
            std::fprintf(stderr, "FAIL: M=8 forward pass returned status %d\n", st8);
            return 1;
        }
        std::vector<uint32_t> h_out8(M8 * 12, 0);
        hipMemcpyDtoH(h_out8.data(), d8_out, M8 * 12 * sizeof(uint32_t));

        constexpr size_t kExitsOff  = 232000ULL + 280672ULL;
        constexpr size_t kDepthsOff = 232000ULL + 280816ULL;
        uint32_t exits1[12] = {0}, depth1 = 0;
        if (weights.is_alodlm_loop) {
            hipMemcpyDtoH(exits1, (char*)d_scratch + kExitsOff, sizeof(exits1));
            hipMemcpyDtoH(&depth1, (char*)d_scratch + kDepthsOff, sizeof(depth1));
        }
        for (int s = 0; s < M8; ++s) {
            for (int i = 0; i < expected_k8; ++i) {
                uint32_t v8 = h_out8[s * 12 + i];
                if (v8 == 0xFFFFFFFFu || v8 != h_out[i]) {
                    std::fprintf(stderr, "FAIL: M=1 vs M=8 token mismatch at seq %d slot %d: M1=%u M8=%u\n",
                                 s, i, h_out[i], v8);
                    return 1;
                }
            }
            for (int i = expected_k8; i < 12; ++i) {
                if (h_out8[s * 12 + i] != 0xFFFFFFFFu) {
                    std::fprintf(stderr, "FAIL: M=8 output slot %d past requested depth is not sentinel at seq %d!\n", i, s);
                    return 1;
                }
            }
            if (weights.is_alodlm_loop) {
                uint32_t exits8[12] = {0}, depth8 = 0;
                char* s8_base = (char*)d8_scr + (size_t)s * 8388608ULL;
                hipMemcpyDtoH(exits8, s8_base + kExitsOff, sizeof(exits8));
                hipMemcpyDtoH(&depth8, s8_base + kDepthsOff, sizeof(depth8));
                if (depth8 != depth1) {
                    std::fprintf(stderr, "FAIL: M=1 vs M=8 halted depth mismatch at seq %d: M1=%u M8=%u\n",
                                 s, depth1, depth8);
                    return 1;
                }
                for (int i = 0; i < expected_k8; ++i) {
                    if (exits8[i] != exits1[i]) {
                        std::fprintf(stderr, "FAIL: M=1 vs M=8 slot exit depth mismatch at seq %d slot %d: M1=%u M8=%u\n",
                                     s, i, exits1[i], exits8[i]);
                        return 1;
                    }
                }
            }
        }
        std::printf("  PASS: 100%% prefix (%d slots) & exit-depth (halted_depth=%u) parity between M=1 and M=8 across all 8 batch sequences!\n",
                    expected_k8, depth1);
        if (two_stage_alodlm) {
            for (int k : {3, 4, 7}) {
                for (RadArgs* args : {&a, &a8}) {
                    RadTensor* tensors = (args == &a) ? t : t8;
                    tensors[5].shape[0] = k;
                    const int st = op_fn(args, (RadStream)stream);
                    hipStreamSynchronize(stream);
                    if (st != 0) {
                        std::fprintf(stderr, "FAIL: requested depth %d returned status %d\n", k, st);
                        return 1;
                    }
                    const int seqs = (args == &a) ? 1 : M8;
                    std::vector<uint32_t> tokens(seqs * 12);
                    hipMemcpyDtoH(tokens.data(), args->t[6].data, tokens.size() * sizeof(uint32_t));
                    for (int s = 0; s < seqs; ++s) {
                        for (int i = 0; i < 12; ++i) {
                            const uint32_t expected = (i < k) ? h_out[i] : 0xFFFFFFFFu;
                            if (tokens[s * 12 + i] != expected) {
                                std::fprintf(stderr, "FAIL: requested depth %d prefix/sentinel mismatch at M=%d seq %d slot %d\n",
                                             k, seqs, s, i);
                                return 1;
                            }
                        }
                    }
                    tensors[5].shape[0] = 0;
                }
            }
        }
        if (weights.is_alodlm_loop) {
            // Time the local TP2 compute geometry only, after full-vocabulary parity passes.
            const auto check_hip = [](hipError_t error, const char* operation) {
                if (error == hipSuccess) return true;
                std::fprintf(stderr, "FAIL: TP2 single-shard %s: %s (%d)\n",
                             operation, hipGetErrorString(error), (int)error);
                return false;
            };
            FullDrafterGpuPointers h_gpu{};
            void* gpu_params = t[5].data;
            if (!check_hip(hipMemcpyDtoH(&h_gpu, gpu_params, sizeof(h_gpu)), "read GPU pointers")) return 1;
            const void* saved_rank1 = h_gpu.teacher_lm_head_rank1;
            const int8_t* saved_q8_rank1 = h_gpu.teacher_lm_head_q8_rank1;
            const float* saved_scale_rank1 = h_gpu.teacher_lm_head_scale_rank1;
            h_gpu.teacher_lm_head_rank1 = nullptr;
            h_gpu.teacher_lm_head_q8_rank1 = nullptr;
            h_gpu.teacher_lm_head_scale_rank1 = nullptr;
            if (!check_hip(hipMemcpyHtoD(gpu_params, &h_gpu, sizeof(h_gpu)), "set single-shard pointers")) return 1;

            for (RadArgs* args : {&a, &a8}) {
                RadTensor* tensors = (args == &a) ? t : t8;
                const int seqs = (args == &a) ? M : M8;
                const int64_t saved_k = tensors[5].shape[0];
                for (int k : {3, 4}) {
                    if (seqs == M8 && k == 4) continue;
                    tensors[5].shape[0] = k;
                    const auto run_once = [&]() {
                        const int st = op_fn(args, (RadStream)stream);
                        if (st == 0) return true;
                        std::fprintf(stderr, "FAIL: TP2 single-shard M=%d req_k=%d returned status %d\n",
                                     seqs, k, st);
                        return false;
                    };
                    for (int it = 0; it < 3; ++it) {
                        if (!run_once()) return 1;
                    }
                    if (!check_hip(hipStreamSynchronize(stream), "warmup sync")) return 1;

                    float shard_total_lat = 0.0f;
                    float shard_min_lat = 1e9f;
                    for (int it = 0; it < 10; ++it) {
                        if (!check_hip(hipEventRecord(ev_start, stream), "record start")) return 1;
                        if (!run_once()) return 1;
                        if (!check_hip(hipEventRecord(ev_stop, stream), "record stop")) return 1;
                        if (!check_hip(hipStreamSynchronize(stream), "timing sync")) return 1;
                        float lat = 0.0f;
                        if (!check_hip(hipEventElapsedTime(&lat, ev_start, ev_stop), "elapsed time")) return 1;
                        shard_total_lat += lat;
                        if (lat < shard_min_lat) shard_min_lat = lat;
                    }
                    std::printf("  TP2 single-shard (vocab=124160) M=%d req_k=%d latency: avg=%.3f ms, min=%.3f ms\n",
                                seqs, k, shard_total_lat / 10.0f, shard_min_lat);
                    std::fflush(stdout);
                }
                tensors[5].shape[0] = saved_k;
            }
            h_gpu.teacher_lm_head_rank1 = saved_rank1;
            h_gpu.teacher_lm_head_q8_rank1 = saved_q8_rank1;
            h_gpu.teacher_lm_head_scale_rank1 = saved_scale_rank1;
            if (!check_hip(hipMemcpyHtoD(gpu_params, &h_gpu, sizeof(h_gpu)), "restore full-vocabulary pointers")) return 1;
        }
        hipFree(d8_k0); hipFree(d8_k1); hipFree(d8_k2); hipFree(d8_anc);
        hipFree(d8_toks); hipFree(d8_out); hipFree(d8_scr);
    }
    // Clean up GPU 0
    hipFree(d_k0); hipFree(d_k1); hipFree(d_k2); hipFree(d_anc);
    hipFree(d_toks); hipFree(d_out); hipFree(d_scratch);
    hipStreamDestroy(stream);

    // Test on GPU 1
    std::printf("[6] Preparing test on GPU 1...\n");
    hipSetDevice(1);
    hipStream_t stream1;
    hipStreamCreate(&stream1);

    hipMalloc(&d_k0, M * W * 2);
    hipMalloc(&d_k1, M * W * 2);
    hipMalloc(&d_k2, M * W * 2);
    hipMalloc(&d_anc, M * W * 2);
    hipMalloc(&d_toks, M * 7 * sizeof(int32_t));
    hipMalloc(&d_out, M * 12 * sizeof(uint32_t));
    hipMalloc(&d_scratch, 2 * 8388608LL);

    hipMemset(d_k0, 0, M * W * 2);
    hipMemset(d_k1, 0, M * W * 2);
    hipMemset(d_k2, 0, M * W * 2);
    hipMemset(d_anc, 0, M * W * 2);
    hipMemset(d_toks, 0, M * 7 * sizeof(int32_t));
    hipMemset(d_out, 0, M * 12 * sizeof(uint32_t));
    hipMemset(d_scratch, 0, 2 * 8388608LL);

    t[0].data = d_k0; t[0].shape[0] = M;
    t[1].data = d_k1; t[1].shape[0] = M;
    t[2].data = d_k2; t[2].shape[0] = M;
    t[3].data = d_anc; t[3].shape[0] = M;
    t[4].data = d_toks; t[4].shape[0] = M * 7;
    t[5].data = const_cast<FullDrafterGpuPointers*>(weights.ranks[1].d_gpu_pointers);
    t[6].data = d_out; t[6].shape[0] = M;

    a.scratch = d_scratch;

    std::printf("[7] Invoking r4d_rad_fulldrafter_two_stage on GPU 1...\n");
    std::fflush(stdout);
    status = op_fn(&a, (RadStream)stream1);
    std::printf("  op returned status: %d\n", status);
    std::fflush(stdout);

    std::printf("[8] Synchronizing GPU 1...\n");
    std::fflush(stdout);
    hipError_t sync_err = hipStreamSynchronize(stream1);
    std::printf("  GPU 1 stream sync: %s (%d)\n", hipGetErrorString(sync_err), sync_err);
    std::fflush(stdout);

    std::vector<uint32_t> h_out1(12, 0);
    hipMemcpyDtoH(h_out1.data(), d_out, 12 * sizeof(uint32_t));
    std::printf("  GPU 1 Tokens generated: ");
    for (int i = 0; i < 12; ++i) std::printf("%u ", h_out1[i]);
    std::printf("\n");

    // Verify tokens on GPU 0 and GPU 1 match exactly (always-on, NDEBUG-safe)
    for (int i = 0; i < 12; ++i) {
        if (h_out[i] != h_out1[i]) {
            std::fprintf(stderr, "FAIL: GPU 0 and GPU 1 token mismatch at slot %d: GPU0=%u GPU1=%u\n",
                         i, h_out[i], h_out1[i]);
            return 1;
        }
    }
    std::printf("  PASS: 100%% token equivalence between GPU 0 and GPU 1!\n");

    hipFree(d_k0); hipFree(d_k1); hipFree(d_k2); hipFree(d_anc);
    hipFree(d_toks); hipFree(d_out); hipFree(d_scratch);
    hipStreamDestroy(stream1);

    std::printf("==================================================\n");
    std::printf(" DUAL GPU KERNEL EXECUTION TEST COMPLETE\n");
    std::printf("==================================================\n");
    return 0;
}
