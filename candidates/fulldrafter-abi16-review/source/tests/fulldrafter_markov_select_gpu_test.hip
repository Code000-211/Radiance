/* Real gfx1201 selector regression, not a reimplementation of its GPU reductions.
 * Including the production translation unit gives the test its actual kernel and scratch
 * layout without exporting test hooks. Only the selector launch wrapper is executed.
 * Build/run: see libr4d/README.md, "FullDrafter selector regression".
 */
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../libr4d/r4d_fulldrafter.hip"

namespace {
constexpr int kSequences = 2;
constexpr int kPrevCount = 16;
constexpr uint32_t kUnwrittenId = 0xdeadbeefu;
constexpr float kUnwrittenScore = -8192.0f;
constexpr uint32_t kAnchors[kSequences] = {11, 13};
constexpr uint32_t kCollisionIds[K_TOP] = {
    7, 263, 519, 775, 1031, 1287, 1543, 1799,
    2055, 2311, 2567, 2823, 3079, 3335, 3591, 3847,
};
constexpr uint32_t kMixedIds[] = {
    31, 287, 543, 799, 1055, 1311, 1567, 1823,
    4007, 27415, 58424, 120343, 124177, 155295, 194191,
    240815, 244503, 248076,
};
static_assert(M2_GRID_BLOCKS == 64 && M2_THREADS == 256 && K_TOP == 16);
static_assert(V_TOTAL == 248320 && V_ACT == 248077 && V_TOTAL / M2_GRID_BLOCKS == 3880);
static_assert(M_RANK == 64 && S_RANK == 64 && H_DIM == 768);
static_assert(BLOCK_K == 6 && FULLDRAFTER_TOTAL_K == 12);
static_assert(kCollisionIds[15] < V_TOTAL / M2_GRID_BLOCKS);
static_assert(kCollisionIds[15] % M2_THREADS == kCollisionIds[0] % M2_THREADS);

size_t checked_values = 0;

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

void hip_check(hipError_t error, const char* expression, int line) {
    if (error != hipSuccess) {
        std::fprintf(stderr, "FAIL: line %d: %s: %s\n", line, expression,
                     hipGetErrorString(error));
        std::exit(1);
    }
}
#define HIP_CHECK(expression) hip_check((expression), #expression, __LINE__)

template <class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { HIP_CHECK(hipFree(data)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

uint16_t exact_bf16(int value) {
    const float f = static_cast<float>(value);
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    require((bits & 0xffffu) == 0, "fixture value must be exactly BF16-representable");
    return static_cast<uint16_t>(bits >> 16);
}

// All nonzero coordinates are integers, and every intermediate is far below 2^24.
// Integer scalar arithmetic is thus an independent, zero-tolerance float/BF16 oracle.
int anchor_a0(uint32_t token) { return static_cast<int>(token % 5) - 2; }
int anchor_a63(uint32_t token) { return static_cast<int>(token % 3) - 1; }
int pred0(uint32_t token) { return 1 + static_cast<int>(token % 2); }
int pred63(uint32_t token) { return 1 + static_cast<int>(token % 3); }
int hidden0(int sequence, int stage, int slot) { return 16 * sequence + 4 * stage + slot; }
int hidden767(int sequence, int stage, int slot) { return sequence + stage + slot; }

struct Features {
    int sequence = -1;
    int b0 = 0, b63 = 0;
    int succ0 = 0, succ63 = 0;
};

int base_logit(const Features& f, uint32_t token, int sequence, int stage, int slot) {
    if (token >= V_ACT) return 1 << 20; // Padding would win every slot if scanned.
    if (f.sequence != sequence) return -4096;
    // Both sequence-1 anchors have pred0 == 2. Give the higher-ID tied winner a
    // unary lead exactly canceled by its smaller successor code, so it is visited first.
    return 128 + 32 * stage + 2 * slot + 8 * sequence + (token == V_ACT - 1 ? 8 : 0)
           + (token == 124177 ? 2 * hidden0(sequence, stage, slot) : 0);
}

struct Candidate { uint32_t id; int unary; };
struct Oracle {
    std::array<uint32_t, M2_GRID_BLOCKS * K_TOP> block_ids;
    std::array<float, M2_GRID_BLOCKS * K_TOP> block_unary;
    std::array<uint32_t, K_TOP> ids;
    std::array<float, K_TOP> scores;
    uint32_t winner = kUnwrittenId;
};

Oracle scalar_oracle(const std::vector<Features>& features, int sequence, int stage,
                     int slot, uint32_t anchor) {
    std::vector<Candidate> vocabulary(V_ACT);
    for (uint32_t v = 0; v < V_ACT; ++v) {
        const Features& f = features[v];
        vocabulary[v] = {v, base_logit(f, v, sequence, stage, slot) +
                            anchor_a0(anchor) * f.b0 + anchor_a63(anchor) * f.b63};
    }
    const auto order = [](const Candidate& a, const Candidate& b) {
        return a.unary != b.unary ? a.unary > b.unary : a.id < b.id;
    };
    Oracle result;
    // Sort the complete scalar vocabulary, never emulate threads, shuffles or feed/pop.
    // Block outputs are also observable, so check them separately before the global sort.
    const int per_block = (V_TOTAL + M2_GRID_BLOCKS - 1) / M2_GRID_BLOCKS;
    for (int b = 0; b < M2_GRID_BLOCKS; ++b) {
        auto begin = vocabulary.begin() + b * per_block;
        auto end = vocabulary.begin() + std::min((b + 1) * per_block, V_ACT);
        std::partial_sort(begin, begin + K_TOP, end, order);
        for (int k = 0; k < K_TOP; ++k) {
            result.block_ids[b * K_TOP + k] = begin[k].id;
            result.block_unary[b * K_TOP + k] = static_cast<float>(begin[k].unary);
        }
    }
    std::partial_sort(vocabulary.begin(), vocabulary.begin() + K_TOP, vocabulary.end(), order);
    int best_score = -1000000;
    for (int k = 0; k < K_TOP; ++k) {
        const Candidate& c = vocabulary[k];
        const Features& f = features[c.id];
        const int score = c.unary + hidden0(sequence, stage, slot) * pred0(anchor) * f.succ0
                                 - 2 * hidden767(sequence, stage, slot) * pred63(anchor) * f.succ63;
        result.ids[k] = c.id;
        result.scores[k] = static_cast<float>(score);
        if (score > best_score || (score == best_score && c.id < result.winner)) {
            best_score = score;
            result.winner = c.id;
        }
    }
    return result;
}

void expect_ids(const char* label, const uint32_t* expected, const uint32_t* actual,
                size_t count, int sequence, int stage, int slot) {
    for (size_t i = 0; i < count; ++i) {
        if (expected[i] != actual[i]) {
            std::fprintf(stderr, "FAIL: %s seq=%d stage=%d slot=%d index=%zu: expected %u, got %u\n",
                         label, sequence, stage, slot, i, expected[i], actual[i]);
            std::exit(1);
        }
        ++checked_values;
    }
}

void expect_scores(const char* label, const float* expected, const float* actual,
                   size_t count, int sequence, int stage, int slot) {
    for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(expected + i, actual + i, sizeof(float)) != 0) {
            std::fprintf(stderr, "FAIL: %s seq=%d stage=%d slot=%d index=%zu: expected %a, got %a\n",
                         label, sequence, stage, slot, i, expected[i], actual[i]);
            std::exit(1);
        }
        ++checked_values;
    }
}

struct Published {
    std::array<uint32_t, FULLDRAFTER_TOTAL_K * K_TOP> ids;
    std::array<float, FULLDRAFTER_TOTAL_K * K_TOP> scores;
    std::array<uint32_t, FULLDRAFTER_TOTAL_K> tokens;
    std::array<uint32_t, kPrevCount> prev;
};

void run_regression() {
    std::vector<Features> features(V_TOTAL);
    for (uint32_t id : kCollisionIds) features[id] = {0, 1, 2, 0, 1};
    for (uint32_t id : kMixedIds) features[id] = {1, 1, 2, 0, 1};
    // These two cross the Top16 cutoff for anchor 13, but not for the winning token 1823.
    for (uint32_t id : {240815u, 244503u}) {
        features[id].b0 = 4;
        features[id].b63 = -2;
    }
    // Equal reranked scores: the lower ID must win, including when visited later.
    for (uint32_t id : {3591u, 3847u, 1823u, 124177u}) features[id].succ0 = 8;
    features[124177].succ0 = 7;

    const size_t code_count = static_cast<size_t>(V_TOTAL) * M_RANK;
    DeviceBuffer<uint16_t> a(code_count), b(code_count), pred(code_count), succ(code_count);
    DeviceBuffer<uint16_t> projection(static_cast<size_t>(S_RANK) * H_DIM);
    std::vector<uint16_t> weights(code_count, 0);
    for (uint32_t v = 0; v < V_TOTAL; ++v) {
        weights[static_cast<size_t>(v) * M_RANK] = exact_bf16(anchor_a0(v));
        weights[static_cast<size_t>(v) * M_RANK + 63] = exact_bf16(anchor_a63(v));
    }
    HIP_CHECK(hipMemcpy(a.data, weights.data(), weights.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    std::fill(weights.begin(), weights.end(), 0);
    for (uint32_t v = 0; v < V_TOTAL; ++v) {
        weights[static_cast<size_t>(v) * M_RANK] = exact_bf16(features[v].b0);
        weights[static_cast<size_t>(v) * M_RANK + 63] = exact_bf16(features[v].b63);
    }
    HIP_CHECK(hipMemcpy(b.data, weights.data(), weights.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    std::fill(weights.begin(), weights.end(), 0);
    for (uint32_t v = 0; v < V_TOTAL; ++v) {
        weights[static_cast<size_t>(v) * S_RANK] = exact_bf16(pred0(v));
        weights[static_cast<size_t>(v) * S_RANK + 63] = exact_bf16(pred63(v));
    }
    HIP_CHECK(hipMemcpy(pred.data, weights.data(), weights.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    std::fill(weights.begin(), weights.end(), 0);
    for (uint32_t v = 0; v < V_TOTAL; ++v) {
        weights[static_cast<size_t>(v) * S_RANK] = exact_bf16(features[v].succ0);
        weights[static_cast<size_t>(v) * S_RANK + 63] = exact_bf16(features[v].succ63);
    }
    HIP_CHECK(hipMemcpy(succ.data, weights.data(), weights.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    weights.assign(static_cast<size_t>(S_RANK) * H_DIM, 0);
    weights[0] = exact_bf16(1);
    weights[63 * H_DIM + H_DIM - 1] = exact_bf16(-2);
    HIP_CHECK(hipMemcpy(projection.data, weights.data(), weights.size() * sizeof(uint16_t), hipMemcpyHostToDevice));

    FullDrafterGpuPointers pointers{};
    pointers.markov_matrix_a = a.data;
    pointers.markov_matrix_b = b.data;
    pointers.selector_pred_codebook = pred.data;
    pointers.selector_succ_codebook = succ.data;
    pointers.selector_hidden_proj = projection.data;
    // The selector only tests this pointer for non-null to choose the full, padded vocabulary;
    // it never reads a teacher head. No teacher GEMV/backbone is part of this focused test.
    pointers.teacher_lm_head_rank1 = a.data;
    DeviceBuffer<FullDrafterGpuPointers> params(1);
    HIP_CHECK(hipMemcpy(params.data, &pointers, sizeof(pointers), hipMemcpyHostToDevice));
    DeviceBuffer<char> scratch(kSequences * SCR_PER_SEQ);
    DeviceBuffer<uint32_t> tokens(kSequences * FULLDRAFTER_TOTAL_K);
    HIP_CHECK(hipMemset(scratch.data, 0, kSequences * SCR_PER_SEQ));
    std::array<Published, kSequences> expected;
    for (int s = 0; s < kSequences; ++s) {
        Published& e = expected[s];
        e.ids.fill(kUnwrittenId);
        e.scores.fill(kUnwrittenScore);
        e.tokens.fill(kUnwrittenId);
        e.prev.fill(kUnwrittenId);
        e.prev[0] = kAnchors[s];
        char* seq = scratch.data + s * SCR_PER_SEQ;
        HIP_CHECK(hipMemcpy(seq + SCR_PROP_ID_OFF, e.ids.data(), sizeof(e.ids), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(seq + SCR_PROP_VAL_OFF, e.scores.data(), sizeof(e.scores), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(seq + SCR_PREV_TOK_OFF, e.prev.data(), sizeof(e.prev), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(tokens.data + s * FULLDRAFTER_TOTAL_K, e.tokens.data(), sizeof(e.tokens), hipMemcpyHostToDevice));
        for (int stage = 1; stage <= 2; ++stage) {
            std::vector<float> hidden(BLOCK_K * H_DIM, 0.0f);
            for (int slot = 0; slot < BLOCK_K; ++slot) {
                hidden[slot * H_DIM] = static_cast<float>(hidden0(s, stage, slot));
                hidden[slot * H_DIM + H_DIM - 1] = static_cast<float>(hidden767(s, stage, slot));
            }
            HIP_CHECK(hipMemcpy(seq + (stage == 1 ? SCR_H1_OFF : SCR_H2_OFF), hidden.data(),
                                hidden.size() * sizeof(float), hipMemcpyHostToDevice));
        }
    }

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    for (int stage = 1; stage <= 2; ++stage) {
        for (int s = 0; s < kSequences; ++s) {
            std::vector<float> logits(static_cast<size_t>(BLOCK_K) * V_TOTAL);
            for (int slot = 0; slot < BLOCK_K; ++slot)
                for (uint32_t v = 0; v < V_TOTAL; ++v)
                    logits[static_cast<size_t>(slot) * V_TOTAL + v] =
                        static_cast<float>(base_logit(features[v], v, s, stage, slot));
            HIP_CHECK(hipMemcpy(scratch.data + s * SCR_PER_SEQ + SCR_BASE_LOGITS_OFF,
                                logits.data(), logits.size() * sizeof(float), hipMemcpyHostToDevice));
        }
        for (int slot = 0; slot < BLOCK_K; ++slot) {
            const int global_slot = (stage - 1) * BLOCK_K + slot;
            std::array<Oracle, kSequences> oracle;
            for (int s = 0; s < kSequences; ++s) {
                Published& e = expected[s];
                oracle[s] = scalar_oracle(features, s, stage, slot, e.prev[global_slot]);
                const Oracle& o = oracle[s];
                if (s == 0) {
                    expect_ids("16 winners from thread 7", kCollisionIds, o.ids.data(), K_TOP, s, stage, slot);
                    require(o.winner == 3591, "reranker must choose the fifteenth same-thread candidate");
                } else {
                    require(o.winner == 1823, "reranker must resolve the cross-block winner tie by low ID");
                    require(o.ids[0] == 124177 && o.ids[1] == V_ACT - 1,
                            "higher-ID tied winner must lead; last active row must survive padding exclusion");
                    const auto low = std::find(o.ids.begin(), o.ids.end(), 1823u);
                    require(low != o.ids.end() && o.scores[low - o.ids.begin()] == o.scores[0],
                            "lower-ID winner must appear later with exactly the same reranked score");
                    for (uint32_t id : {240815u, 244503u}) {
                        const bool present = std::find(o.ids.begin(), o.ids.end(), id) != o.ids.end();
                        require(present == (global_slot == 0), "anchor conditioning must change shortlist membership");
                    }
                }
                std::copy(o.ids.begin(), o.ids.end(), e.ids.begin() + global_slot * K_TOP);
                std::copy(o.scores.begin(), o.scores.end(), e.scores.begin() + global_slot * K_TOP);
                e.tokens[global_slot] = o.winner;
                e.prev[global_slot + 1] = o.winner;
            }
            require(r4d_fulldrafter_slot_markov_select(stage, slot, params.data, scratch.data,
                                                      tokens.data, kSequences, stream) == 0,
                    "production selector launch wrapper must accept the fixture");
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipStreamSynchronize(stream));
            for (int s = 0; s < kSequences; ++s) {
                const char* seq = scratch.data + s * SCR_PER_SEQ;
                Published actual;
                HIP_CHECK(hipMemcpy(actual.ids.data(), seq + SCR_PROP_ID_OFF, sizeof(actual.ids), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(actual.scores.data(), seq + SCR_PROP_VAL_OFF, sizeof(actual.scores), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(actual.prev.data(), seq + SCR_PREV_TOK_OFF, sizeof(actual.prev), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(actual.tokens.data(), tokens.data + s * FULLDRAFTER_TOTAL_K,
                                    sizeof(actual.tokens), hipMemcpyDeviceToHost));
                const Published& e = expected[s];
                // Compare all slots after each launch, including untouched sentinels and earlier
                // stage output: a correct value written to the wrong global slot must fail.
                expect_ids("published Top16", e.ids.data(), actual.ids.data(), e.ids.size(), s, stage, slot);
                expect_scores("reranked scores", e.scores.data(), actual.scores.data(), e.scores.size(), s, stage, slot);
                expect_ids("winning tokens", e.tokens.data(), actual.tokens.data(), e.tokens.size(), s, stage, slot);
                expect_ids("Markov chain publication", e.prev.data(), actual.prev.data(), e.prev.size(), s, stage, slot);
                std::array<uint32_t, M2_GRID_BLOCKS * K_TOP> block_ids;
                std::array<float, M2_GRID_BLOCKS * K_TOP> block_unary;
                HIP_CHECK(hipMemcpy(block_ids.data(), seq + SCR_BLK_CANDS_OFF, sizeof(block_ids), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(block_unary.data(), seq + SCR_BLK_UNARY_OFF, sizeof(block_unary), hipMemcpyDeviceToHost));
                expect_ids("block Top16", oracle[s].block_ids.data(), block_ids.data(), block_ids.size(), s, stage, slot);
                expect_scores("block unary scores", oracle[s].block_unary.data(), block_unary.data(), block_unary.size(), s, stage, slot);
                uint32_t arrival;
                HIP_CHECK(hipMemcpy(&arrival, seq + SCR_ARRIVAL_CTR_OFF, sizeof(arrival), hipMemcpyDeviceToHost));
                require(arrival == 0, "last workgroup must reset arrival counter before the next slot");
            }
        }
    }
    HIP_CHECK(hipStreamDestroy(stream));
}
} // namespace

int main() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int device = -1;
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, i));
        if (std::strncmp(properties.gcnArchName, "gfx1201", 7) == 0 &&
            (properties.gcnArchName[7] == '\0' || properties.gcnArchName[7] == ':') &&
            properties.warpSize == 32) {
            device = i;
            break;
        }
    }
    if (device < 0) {
        std::puts("SKIP: fulldrafter_markov_select_gpu_test requires a visible gfx1201 Wave32 device");
        return 77;
    }
    HIP_CHECK(hipSetDevice(device));
    run_regression();
    std::printf("PASS: actual selector, 2 sequences x 2 stages x 6 slots; %zu exact ID/score checks\n", checked_values);
    return 0;
}
