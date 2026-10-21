#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "arch/common/rad_block_fulldrafter.h"
#include "libr4d/r4d_fulldrafter.h"
#include "abi/rad_abi.h"

using namespace radiance;

struct EvalSequence {
    int doc_idx;
    int p;
    int anchor_token;
    int row_idx;
    std::vector<int> targets; // up to 12 targets
    std::vector<bool> valid_targets; // greedy-consistent
};

int main(int argc, char** argv) {
    std::string sidecar_path = "/home/orca/drafter_data/checkpoints/flash_next_k12_exp/drafter_step_92500.safetensors";
    std::string teacher_path = "/home/orca/drafter_data/teacher/teacher-matrices.safetensors";
    std::string rec_dir = "/home/orca/drafter_data/rec_eval";
    std::string out_json = "";
    int num_seqs = 50;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sidecar" && i + 1 < argc) sidecar_path = argv[++i];
        else if (arg == "--teacher" && i + 1 < argc) teacher_path = argv[++i];
        else if (arg == "--rec" && i + 1 < argc) rec_dir = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_json = argv[++i];
        else if (arg == "--num" && i + 1 < argc) num_seqs = std::atoi(argv[++i]);
    }

    std::printf("=================================================================\n");
    std::printf(" FullDrafter GPU rec_eval Parity Benchmark\n");
    std::printf(" Sidecar: %s\n", sidecar_path.c_str());
    std::printf(" Teacher: %s\n", teacher_path.c_str());
    std::printf("=================================================================\n");

    // 1. Load weights
    FullDrafterWeights weights;
    std::string err;
    if (!FullDrafterLoader::load_weights(sidecar_path, &weights, nullptr, &err, 2)) {
        std::fprintf(stderr, "Failed to load weights: %s\n", err.c_str());
        return 1;
    }
    if (!FullDrafterLoader::load_teacher_matrices(teacher_path, &weights, &err)) {
        std::fprintf(stderr, "Failed to load teacher matrices: %s\n", err.c_str());
        return 1;
    }
    std::printf("[OK] Weights loaded (lm_head_dim=%zu, is_fused=%d)\n",
                weights.lm_head_dim, weights.is_fused ? 1 : 0);

    // 2. Load recording files (mmap)
    std::string kld_path = rec_dir + "/kld.json";
    std::ifstream kld_file(kld_path);
    if (!kld_file.good()) {
        std::fprintf(stderr, "Failed to open %s\n", kld_path.c_str());
        return 1;
    }
    nlohmann::json kld_meta;
    kld_file >> kld_meta;

    int taps_fd = ::open((rec_dir + "/taps.bf16").c_str(), O_RDONLY);
    int toks_fd = ::open((rec_dir + "/tokens.i32").c_str(), O_RDONLY);
    int top_fd  = ::open((rec_dir + "/top.bin").c_str(), O_RDONLY);
    if (taps_fd < 0 || toks_fd < 0 || top_fd < 0) {
        std::fprintf(stderr, "Failed to open taps/tokens/top files in %s\n", rec_dir.c_str());
        return 1;
    }

    struct stat st_taps{}, st_toks{}, st_top{};
    ::fstat(taps_fd, &st_taps); ::fstat(toks_fd, &st_toks); ::fstat(top_fd, &st_top);
    const rad_bf16_t* taps_mmap = (const rad_bf16_t*)::mmap(nullptr, st_taps.st_size, PROT_READ, MAP_SHARED, taps_fd, 0);
    const int32_t* toks_mmap = (const int32_t*)::mmap(nullptr, st_toks.st_size, PROT_READ, MAP_SHARED, toks_fd, 0);
    const uint8_t* top_mmap  = (const uint8_t*)::mmap(nullptr, st_top.st_size, PROT_READ, MAP_SHARED, top_fd, 0);

    int top_k = kld_meta.value("top", 32);
    // top.bin struct: { int32 id, float lp } * top_k
    size_t top_stride = top_k * 8;

    // 3. Collect first num_seqs evaluation positions (prefer golden JSON if present)
    std::vector<EvalSequence> sequences;
    std::string golden_json = "/home/orca/drafter_data/rec_eval_50_golden.json";
    std::ifstream gj_file(golden_json);
    if (gj_file.good()) {
        nlohmann::json gj;
        gj_file >> gj;
        auto rows = gj["rows"].get<std::vector<int>>();
        auto anchors = gj["anchors"].get<std::vector<int>>();
        auto targets = gj["targets"].get<std::vector<std::vector<int>>>();
        size_t n_load = std::min((size_t)num_seqs, rows.size());
        for (size_t i = 0; i < n_load; ++i) {
            EvalSequence seq;
            seq.doc_idx = 0;
            seq.p = 0;
            seq.row_idx = rows[i];
            seq.anchor_token = anchors[i];
            seq.targets = targets[i];
            seq.valid_targets.resize(targets[i].size());
            for (size_t k = 0; k < targets[i].size(); ++k) {
                seq.valid_targets[k] = (targets[i][k] != -100);
            }
            sequences.push_back(seq);
        }
        std::printf("[OK] Loaded %zu golden sequences from %s\n", sequences.size(), golden_json.c_str());
    } else {
        const auto& docs = kld_meta["docs"];
        int collected = 0;
        int off = 0;
        for (size_t d = 0; d < docs.size() && collected < num_seqs; ++d) {
            int n          = docs[d]["tokens"].get<int>();
            int score_from = docs[d]["score_from"].get<int>();
            int row0       = docs[d]["row0"].get<int>();

            for (int p = score_from; p < n - 13 && collected < num_seqs; ++p) {
                int row = row0 + (p - score_from);
                const int32_t* row_top = reinterpret_cast<const int32_t*>(top_mmap + row * top_stride);
                int anchor_id = toks_mmap[off + p + 1];
                if (row_top[0] != anchor_id) continue;

                EvalSequence seq;
                seq.doc_idx = (int)d;
                seq.p = p;
                seq.anchor_token = anchor_id;
                seq.row_idx = row;

                bool chain_valid = true;
                for (int k = 1; k <= 12; ++k) {
                    int target_id = toks_mmap[off + p + 1 + k];
                    seq.targets.push_back(target_id);

                    int next_row = row0 + (p + k - score_from);
                    const int32_t* next_top = reinterpret_cast<const int32_t*>(top_mmap + next_row * top_stride);
                    if (chain_valid && next_top[0] == target_id) {
                        seq.valid_targets.push_back(true);
                    } else {
                        chain_valid = false;
                        seq.valid_targets.push_back(false);
                    }
                }

                sequences.push_back(seq);
                collected++;
            }
            off += n;
        }
        std::printf("[OK] Collected %zu evaluation sequences\n", sequences.size());
    }

    // 4. Initialize GPU 0
    hipSetDevice(0);
    hipStream_t stream;
    hipStreamCreate(&stream);

    constexpr int M = 1;
    constexpr size_t W = FullDrafterConfig::TARGET_WIDTH;
    constexpr size_t H_DIM = FullDrafterConfig::HIDDEN_SIZE;
    void *d_k0 = nullptr, *d_k1 = nullptr, *d_k2 = nullptr, *d_anc = nullptr;
    void *d_toks = nullptr, *d_out = nullptr, *d_scratch = nullptr;

    hipMalloc(&d_k0, W * sizeof(rad_bf16_t));
    hipMalloc(&d_k1, W * sizeof(rad_bf16_t));
    hipMalloc(&d_k2, W * sizeof(rad_bf16_t));
    hipMalloc(&d_anc, W * sizeof(rad_bf16_t));
    hipMalloc(&d_toks, 7 * sizeof(int32_t));
    hipMalloc(&d_out, 12 * sizeof(uint32_t));
    hipMalloc(&d_scratch, 8388608LL);

    hipMemset(d_toks, 0, 7 * sizeof(int32_t));

    RadTensor t[7]{};
    t[0].data = d_k0; t[0].shape[0] = M;
    t[1].data = d_k1; t[1].shape[0] = M;
    t[2].data = d_k2; t[2].shape[0] = M;
    t[3].data = d_anc; t[3].shape[0] = M;
    t[4].data = d_toks; t[4].shape[0] = 7;
    t[5].data = const_cast<FullDrafterGpuPointers*>(weights.ranks[0].d_gpu_pointers);
    t[6].data = d_out; t[6].shape[0] = M;

    RadArgs a{};
    a.n_t = 7;
    a.t = t;
    a.scratch = d_scratch;
    a.scratch_bytes = 8388608LL;

    // Load libr4d.so
    void* lib = dlopen("/home/orca/RadianceMine/worktrees/tap_record/build_hip/radiance_home/kernels/libr4d.so", RTLD_NOW);
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

    const rad_bf16_t* host_emb_table = weights.ranks[0].host_teacher_embedding_table;

    // 5. Warmup (5 iterations on sequence 0)
    if (!sequences.empty()) {
        const auto& seq0 = sequences[0];
        const rad_bf16_t* row_taps = taps_mmap + (size_t)seq0.row_idx * 7680;
        hipMemcpyHtoD(d_k0, row_taps + 0, W * sizeof(rad_bf16_t));
        hipMemcpyHtoD(d_k1, row_taps + 2560, W * sizeof(rad_bf16_t));
        hipMemcpyHtoD(d_k2, row_taps + 5120, W * sizeof(rad_bf16_t));
        hipMemcpyHtoD(d_anc, host_emb_table + (size_t)seq0.anchor_token * 2560, W * sizeof(rad_bf16_t));
        int32_t h_toks0[7] = { seq0.anchor_token, 151643, 151643, 151643, 151643, 151643, 151643 };
        hipMemcpyHtoD(d_toks, h_toks0, 7 * sizeof(int32_t));
        for (int w = 0; w < 5; ++w) {
            op_fn(&a, (RadStream)stream);
        }
        hipStreamSynchronize(stream);
    }

    // 6. Evaluate on all sequences
    std::vector<std::vector<uint32_t>> all_proposals;
    all_proposals.reserve(sequences.size());
    std::vector<int> accepted_lengths_k12;
    accepted_lengths_k12.reserve(sequences.size());
    std::vector<int> accepted_lengths_k3;
    accepted_lengths_k3.reserve(sequences.size());

    int total_evaluated_slots = 0;
    int total_valid_targets = 0;
    int total_target_hits = 0;
    int total_accepted_prefix_k12 = 0;
    int total_accepted_prefix_k3 = 0;
    int slot_valid[12] = {0};
    int slot_hits[12] = {0};
    float total_gpu_ms = 0.0f;

    hipEvent_t ev_start, ev_stop;
    hipEventCreate(&ev_start);
    hipEventCreate(&ev_stop);

    for (size_t i = 0; i < sequences.size(); ++i) {
        const auto& seq = sequences[i];
        const rad_bf16_t* row_taps = taps_mmap + (size_t)seq.row_idx * 7680;
        const rad_bf16_t* p_k0 = row_taps + 0;
        const rad_bf16_t* p_k1 = row_taps + 2560;
        const rad_bf16_t* p_k2 = row_taps + 5120;
        const rad_bf16_t* p_anc = host_emb_table + (size_t)seq.anchor_token * 2560;

        hipMemcpyHtoD(d_k0, p_k0, W * sizeof(rad_bf16_t));
        hipMemcpyHtoD(d_k1, p_k1, W * sizeof(rad_bf16_t));
        hipMemcpyHtoD(d_k2, p_k2, W * sizeof(rad_bf16_t));
        hipMemcpyHtoD(d_anc, p_anc, W * sizeof(rad_bf16_t));
        int32_t h_toks[7] = { seq.anchor_token, 151643, 151643, 151643, 151643, 151643, 151643 };
        hipMemcpyHtoD(d_toks, h_toks, 7 * sizeof(int32_t));

        hipEventRecord(ev_start, stream);
        op_fn(&a, (RadStream)stream);
        hipEventRecord(ev_stop, stream);
        hipStreamSynchronize(stream);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev_start, ev_stop);
        total_gpu_ms += ms;

        std::vector<uint32_t> prop(12, 0);
        hipMemcpyDtoH(prop.data(), d_out, 12 * sizeof(uint32_t));
        all_proposals.push_back(prop);

        total_evaluated_slots += 12;
        int accepted_prefix_k12 = 0;
        bool chain_intact_k12 = true;
        for (int k = 0; k < 12; ++k) {
            if (seq.valid_targets[k]) {
                total_valid_targets++;
                slot_valid[k]++;
                if (prop[k] == static_cast<uint32_t>(seq.targets[k])) {
                    total_target_hits++;
                    slot_hits[k]++;
                    if (chain_intact_k12) accepted_prefix_k12++;
                } else {
                    chain_intact_k12 = false;
                }
            } else {
                chain_intact_k12 = false;
            }
        }
        total_accepted_prefix_k12 += accepted_prefix_k12;
        accepted_lengths_k12.push_back(accepted_prefix_k12);

        int accepted_prefix_k3 = 0;
        bool chain_intact_k3 = true;
        for (int k = 0; k < 3; ++k) {
            if (seq.valid_targets[k]) {
                if (prop[k] == static_cast<uint32_t>(seq.targets[k])) {
                    if (chain_intact_k3) accepted_prefix_k3++;
                } else {
                    chain_intact_k3 = false;
                }
            } else {
                chain_intact_k3 = false;
            }
        }
        total_accepted_prefix_k3 += accepted_prefix_k3;
        accepted_lengths_k3.push_back(accepted_prefix_k3);
    }

    float avg_gpu_ms = total_gpu_ms / sequences.size();
    float hit_rate = (total_valid_targets > 0) ? (100.0f * total_target_hits / total_valid_targets) : 0.0f;
    float tau_k12 = 1.0f + (float)total_accepted_prefix_k12 / sequences.size();
    float tau_k3  = 1.0f + (float)total_accepted_prefix_k3  / sequences.size();

    std::printf("-----------------------------------------------------------------\n");
    std::printf(" Evaluated Sequences         : %zu\n", sequences.size());
    std::printf(" Total Evaluated Slots       : %d (N_slots = 50 * 12 = %d)\n", total_evaluated_slots, total_evaluated_slots);
    std::printf(" Valid Target Positions      : %d (N_valid)\n", total_valid_targets);
    std::printf(" Ground Truth Hits           : %d / %d (%.4f%%)\n", total_target_hits, total_valid_targets, hit_rate);
    std::printf(" K=12 Accepted Prefix Tokens : %d (Gross Tau = %.4f)\n", total_accepted_prefix_k12, tau_k12);
    std::printf(" K=3  Accepted Prefix Tokens : %d (Gross Tau = %.4f)\n", total_accepted_prefix_k3, tau_k3);
    std::printf(" Avg Drafter Pass Latency    : %.3f ms (Total: %.2f ms)\n", avg_gpu_ms, total_gpu_ms);
    std::printf(" Per-slot Hits [0..11]       :\n   ");
    for (int k = 0; k < 12; ++k) {
        std::printf(" s%d:%d/%d", k, slot_hits[k], slot_valid[k]);
    }
    std::printf("\n-----------------------------------------------------------------\n");

    // Output JSON if requested
    if (!out_json.empty()) {
        nlohmann::json j;
        j["sidecar"] = sidecar_path;
        j["teacher"] = teacher_path;
        j["is_fused"] = weights.is_fused;
        j["lm_head_dim"] = weights.lm_head_dim;
        j["total_evaluated_slots"] = total_evaluated_slots;
        j["total_valid_targets"] = total_valid_targets;
        j["total_target_hits"] = total_target_hits;
        j["hit_rate_pct"] = hit_rate;
        j["total_accepted_prefix_tokens"] = total_accepted_prefix_k12;
        j["tau_empirical"] = tau_k12;
        j["total_accepted_prefix_tokens_k12"] = total_accepted_prefix_k12;
        j["tau_empirical_k12"] = tau_k12;
        j["total_accepted_prefix_tokens_k3"] = total_accepted_prefix_k3;
        j["tau_empirical_k3"] = tau_k3;
        j["avg_gpu_ms"] = avg_gpu_ms;
        j["total_gpu_ms"] = total_gpu_ms;
        j["proposals"] = all_proposals;
        j["accepted_lengths"] = accepted_lengths_k12;
        j["accepted_lengths_k12"] = accepted_lengths_k12;
        j["accepted_lengths_k3"] = accepted_lengths_k3;
        std::vector<int> sh(slot_hits, slot_hits + 12);
        std::vector<int> sv(slot_valid, slot_valid + 12);
        j["slot_hits"] = sh;
        j["slot_valid"] = sv;

        std::ofstream ofs(out_json);
        ofs << j.dump(2) << std::endl;
        std::printf("Saved results to %s\n", out_json.c_str());
    }

    // Cleanup
    hipFree(d_k0); hipFree(d_k1); hipFree(d_k2); hipFree(d_anc);
    hipFree(d_toks); hipFree(d_out); hipFree(d_scratch);
    hipStreamDestroy(stream);
    hipEventDestroy(ev_start); hipEventDestroy(ev_stop);
    dlclose(lib);

    return 0;
}
