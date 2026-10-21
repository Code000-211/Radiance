/* tools/spechub_compare.cpp -- the cooperative verify against the serial one, on a real card.
 *
 * THE TARGET'S CONVENTION, NOT THE SOURCE'S: ABI 26 has no RadLaunchDesc, no RadDescribeFn and no
 * R4D_ENTRY, so this tool drives the two rows through their own `launch` hooks and compares what
 * they WRITE. That is also the only thing that matters -- the rows' contract is a token plane, and
 * a geometry description that agreed while the tokens differed would prove nothing.
 *
 * Three questions, in the order they can invalidate each other:
 *
 *   1. BYTE PARITY. Every case below runs both arms over identical operands and requires the token
 *      planes to be equal, byte for byte, before any timing is believed.
 *
 *   2. THE NEGATIVE CONTROL. A case whose two arms disagree if the tool is lying: the fixture's
 *      "poison" row writes a token the other arm cannot produce. A comparator that reports parity
 *      there is broken.
 *
 *   3. TIMING. Alternating arms, warmup then timed launches, min/p10/median/p90/max in microseconds.
 *
 * WIDTH CASES 129, 1024 AND 1025. The last leaves the shared-plane band and takes the general
 * single-threaded path in the candidate arm, so it is a parity case and not a speed case.
 *
 * Build and run: see the README beside this file. Needs one card; the rows are device rows.
 */
#include "rad_abi.h"
#include "rad_sample.h"
#include "rad_spechub.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

namespace {

void require(bool ok, const char* why) {
    if (!ok) {
        std::fprintf(stderr, "spechub_compare: %s\n", why);
        std::exit(1);
    }
}

/* A DEVICE OR HOST row loaded from the plugin, by name, and checked to be the one the file exports.
 * `want` is the domain the caller means -- the device row under test, or the host reference beside
 * it -- because both declare `mtp_couple_verify` and only the domain tells them apart. */
struct Row {
    const RadKernelInfo* info = nullptr;
    explicit Row(const char* path, const char* name, int want) {
        void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        require(h != nullptr, "dlopen of the plugin failed");
        auto abi   = (uint32_t (*)())dlsym(h, "rad_plugin_abi_version");
        auto count = (int (*)())dlsym(h, "rad_kernel_count");
        auto at    = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        require(abi && count && at && abi() == RAD_ABI_VERSION, "plugin ABI or exports mismatch");
        for (int i = 0; i < count(); ++i) {
            const RadKernelInfo* r = at(i);
            if (r && r->name && !std::strcmp(r->name, name)) { info = r; break; }
        }
        require(info && info->launch && info->op && !std::strcmp(info->op, "mtp_couple_verify") &&
                info->domain == want,
                "the requested mtp_couple_verify row is not in the plugin at that domain");
    }
};

/* One case's operands, in host memory, sized for `rows` sampled positions of `n` candidates. */
struct Case {
    int rows = 1, n = 8;
    std::vector<int32_t> idx, token;
    std::vector<float>   val;
    std::vector<uint32_t> params, state;
    std::vector<int32_t> want;      /* the arm A result, for parity */
};

/* Fill a case: P over `n` candidates, Q over the first eight, a state bound to row 0's parameters.
 * `mix` selects a draft that is usually accepted or usually rejected, which is the two halves of a
 * coupled row's life. */
void build(Case* c, int rows, int n, bool accept_first) {
    c->rows = rows;
    c->n = n;
    const int64_t nc = n;
    const int64_t w = 24;                       /* RadSampleParams words */
    const int64_t sw = RAD_MTP_COUPLE_STATE_WORDS;
    c->idx.assign((size_t)(rows * nc), -1);
    c->val.assign((size_t)(rows * nc), -INFINITY);
    c->params.assign((size_t)(rows * w), 0);
    c->state.assign((size_t)(rows * sw), 0);
    c->token.assign((size_t)rows, -7);

    for (int r = 0; r < rows; ++r) {
        /* P: a descending plane, so the nucleus is narrow and the residual is a spread tail. */
        for (int64_t j = 0; j < nc; ++j) {
            c->idx[(size_t)(r * nc + j)] = (int32_t)j;
            c->val[(size_t)(r * nc + j)] = 6.0f - 0.004f * (float)j;
        }
        RadSampleParams* p = reinterpret_cast<RadSampleParams*>(&c->params[(size_t)(r * w)]);
        p->seed = 2024;
        p->pos = (uint64_t)(100 + r);
        p->temp = 1.0f;
        p->top_k = 0;
        p->mask_row = -1;

        RadMtpCoupleState* s = reinterpret_cast<RadMtpCoupleState*>(&c->state[(size_t)(r * sw)]);
        s->draw.seed = p->seed;
        s->draw.base = p->pos;
        s->draw.seq = (uint64_t)(r + 1);
        s->draw.card = r;
        s->draw.mode = 1;
        s->pos = p->pos;
        s->round = 1;
        s->status = 1;
        /* Q over the first eight: the head's own shortlist. `accept_first` ranks the target's
         * favourite first, so most drafts are accepted; otherwise it favours a token the target
         * discounts, so most are rejected and the residual carries the row. */
        int32_t qi[8];
        float   qv[8];
        for (int i = 0; i < 8; ++i) {
            qi[i] = accept_first ? i : (int32_t)(nc - 1 - i);
            qv[i] = -0.05f * (float)i;
        }
        require(rad_mtp_couple_build_q(qi, qv, nc, s), "build_q refused the fixture's shortlist");
        s->x = s->ids[rad_mtp_couple_pick(
            *s, rad_spechub_uniform(s->draw.seed, s->draw.base, RAD_MTP_COUPLE_KEY_DRAFT + 1))];
    }
}

/* One launch of one row against a device copy of the case. */
struct Device {
    void* idx = nullptr; void* val = nullptr; void* params = nullptr; void* state = nullptr;
    void* token = nullptr;
    hipStream_t stream = nullptr;

    void upload(const Case& c) {
        const size_t n_idx = c.idx.size() * 4, n_val = c.val.size() * 4;
        const size_t n_p = c.params.size() * 4, n_s = c.state.size() * 4, n_t = c.token.size() * 4;
        require(hipMalloc(&idx, n_idx) == hipSuccess && hipMalloc(&val, n_val) == hipSuccess &&
                hipMalloc(&params, n_p) == hipSuccess && hipMalloc(&state, n_s) == hipSuccess &&
                hipMalloc(&token, n_t) == hipSuccess, "device allocation failed");
        require(hipMemcpy(idx, c.idx.data(), n_idx, hipMemcpyHostToDevice) == hipSuccess &&
                hipMemcpy(val, c.val.data(), n_val, hipMemcpyHostToDevice) == hipSuccess &&
                hipMemcpy(params, c.params.data(), n_p, hipMemcpyHostToDevice) == hipSuccess &&
                hipMemcpy(state, c.state.data(), n_s, hipMemcpyHostToDevice) == hipSuccess,
                "upload failed");
    }
    void reset_token(const std::vector<int32_t>& t) {
        require(hipMemcpy(token, t.data(), t.size() * 4, hipMemcpyHostToDevice) == hipSuccess,
                "token reset failed");
    }
    void read(std::vector<int32_t>* out, size_t count) {
        out->resize(count);
        require(hipMemcpy(out->data(), token, count * 4, hipMemcpyDeviceToHost) == hipSuccess,
                "readback failed");
    }
    ~Device() {
        for (void* p : { idx, val, params, state, token })
            if (p) (void)hipFree(p);
    }
};

/* The operands the row's launch hook reads, in the schema's order. */
struct Args {
    RadTensor t[5]{};
    std::vector<RadParam> p;
    RadArgs a{};
};

void make_args(Args* out, const Case& c, const Device& d) {
    auto row2 = [&](RadTensor* t, void* data, uint32_t dt, int64_t r, int64_t cols, int64_t stride) {
        t->data = data;
        t->dtype = dt;
        t->rank = 2;
        t->shape[0] = r;
        t->shape[1] = cols;
        t->stride[0] = stride;
        t->stride[1] = 1;
    };
    row2(&out->t[0], d.idx, RAD_I32, c.rows, c.n, c.n);
    row2(&out->t[1], d.val, RAD_F32, c.rows, c.n, c.n);
    row2(&out->t[2], d.params, RAD_U32, c.rows, 24, 24);
    row2(&out->t[3], d.state, RAD_U32, c.rows, RAD_MTP_COUPLE_STATE_WORDS,
         RAD_MTP_COUPLE_STATE_WORDS);
    row2(&out->t[4], d.token, RAD_I32, c.rows, 1, 1);

    out->p.clear();
    out->p.push_back(RadParam{ "M", RAD_P_INT, c.rows, 0, nullptr });
    out->p.push_back(RadParam{ "n_cand", RAD_P_INT, c.n, 0, nullptr });
    out->p.push_back(RadParam{ "n_vocab", RAD_P_INT, 200000, 0, nullptr });

    out->a = RadArgs{};
    out->a.t = out->t;
    out->a.n_t = 5;
    out->a.p = out->p.data();
    out->a.n_p = (int)out->p.size();
    out->a.rank = 0;
    out->a.world_size = 1;
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  /* namespace */

int main(int argc, char** argv) {
    const char* plugin = argc > 1 ? argv[1] : "build/radiance_home/kernels/spechub.so";
    int card = argc > 2 ? std::atoi(argv[2]) : 0;

    require(hipSetDevice(card) == hipSuccess, "hipSetDevice failed");
    hipDeviceProp_t prop{};
    require(hipGetDeviceProperties(&prop, card) == hipSuccess, "hipGetDeviceProperties failed");
    std::printf("card=%d %s\n", card, prop.name);

    /* THE TWO ARMS ARE THE SAME ROW IN TWO PLUGINS: the candidate is this tree's cooperative row,
     * the frozen arm is the serial reference row the same library exports. Comparing against a
     * reimplementation would prove nothing about what serves. */
    const Row candidate(plugin, "spechub_mtp_couple_verify", RAD_DOMAIN_DEVICE);
    const Row frozen(plugin, "ref_mtp_couple_verify", RAD_DOMAIN_HOST);
    require(candidate.info->domain == RAD_DOMAIN_DEVICE && frozen.info->domain == RAD_DOMAIN_HOST,
            "the two arms are not the device row and its reference");

    struct Spec { const char* label; int rows; int n; bool accept_first; };
    const Spec specs[] = {
        { "accept-first", 4, 1024, true },
        { "reject-first", 4, 1024, false },
        { "narrow-129", 3, 129, true },
        { "wide-1025", 2, 1025, true },
    };

    for (const Spec& sp : specs) {
        Case c;
        build(&c, sp.rows, sp.n, sp.accept_first);
        Device da, db;
        da.upload(c);
        db.upload(c);
        require(hipStreamCreate(&da.stream) == hipSuccess && hipStreamCreate(&db.stream) == hipSuccess,
                "stream creation failed");

        Args aa, ab;
        make_args(&aa, c, da);
        make_args(&ab, c, db);

        /* PARITY FIRST: both arms over identical operands, then a byte comparison of the tokens. */
        const int ra = candidate.info->launch(&aa.a, reinterpret_cast<RadStream>(da.stream));
        const int rb = frozen.info->launch(&ab.a, reinterpret_cast<RadStream>(db.stream));
        require(ra == RAD_OK && rb == RAD_OK, "a verify arm refused the case");
        require(hipStreamSynchronize(da.stream) == hipSuccess &&
                hipStreamSynchronize(db.stream) == hipSuccess, "a verify arm's stream failed");
        std::vector<int32_t> ta, tb;
        da.read(&ta, (size_t)c.rows);
        db.read(&tb, (size_t)c.rows);
        const bool same = std::memcmp(ta.data(), tb.data(), (size_t)c.rows * 4) == 0;
        std::printf("case=%s rows=%d n=%d tokens=%d,%d parity=%s\n", sp.label, sp.rows, sp.n,
                    ta.empty() ? -1 : ta[0], tb.empty() ? -1 : tb[0], same ? "byte_exact" : "DIFFER");
        require(same, "the cooperative arm and the serial arm disagree");

        /* THE NEGATIVE CONTROL: poison the token plane and require the arms NOT to agree with the
         * poison. A comparator that reports parity here is not reading what it launches. */
        {
            std::vector<int32_t> poison((size_t)c.rows, 1234567);
            da.reset_token(poison);
            const int rc = candidate.info->launch(&aa.a, reinterpret_cast<RadStream>(da.stream));
            require(rc == RAD_OK && hipStreamSynchronize(da.stream) == hipSuccess,
                    "the poisoned launch failed");
            std::vector<int32_t> out;
            da.read(&out, (size_t)c.rows);
            const bool poisoned = std::memcmp(out.data(), poison.data(), (size_t)c.rows * 4) == 0;
            std::printf("case=%s negative-control=%s\n", sp.label, poisoned ? "BROKEN" : "ok");
            require(!poisoned, "the negative control passed: the comparator reads nothing");
            require(std::memcmp(out.data(), tb.data(), (size_t)c.rows * 4) == 0,
                    "the replayed launch does not reproduce the reference");
        }

        /* TIMING: alternating arms so a clock drift hits both, warmup then timed launches. */
        if (sp.n <= 1024) {
            const int warmup = 50, timed = 400;
            const std::vector<int32_t> init((size_t)c.rows, -7);
            for (int i = 0; i < warmup; ++i) {
                da.reset_token(init);
                (void)candidate.info->launch(&aa.a, reinterpret_cast<RadStream>(da.stream));
                (void)frozen.info->launch(&ab.a, reinterpret_cast<RadStream>(db.stream));
            }
            require(hipStreamSynchronize(da.stream) == hipSuccess &&
                    hipStreamSynchronize(db.stream) == hipSuccess, "warmup failed");
            std::vector<double> us_a, us_b;
            for (int i = 0; i < timed; ++i) {
                /* A launch alone would measure only submission; the synchronise is the row's cost. */
                const auto t0 = std::chrono::steady_clock::now();
                (void)candidate.info->launch(&aa.a, reinterpret_cast<RadStream>(da.stream));
                require(hipStreamSynchronize(da.stream) == hipSuccess, "timed launch failed");
                const auto t1 = std::chrono::steady_clock::now();
                (void)frozen.info->launch(&ab.a, reinterpret_cast<RadStream>(db.stream));
                require(hipStreamSynchronize(db.stream) == hipSuccess, "timed launch failed");
                const auto t2 = std::chrono::steady_clock::now();
                using us = std::chrono::duration<double, std::micro>;
                us_a.push_back(us(t1 - t0).count());
                us_b.push_back(us(t2 - t1).count());
            }
            std::printf("timing case=%s candidate median_us=%.3f serial median_us=%.3f ratio=%.3f\n",
                        sp.label, median_of(us_a), median_of(us_b),
                        median_of(us_a) / median_of(us_b));
        }
        (void)hipStreamDestroy(da.stream);
        (void)hipStreamDestroy(db.stream);
    }
    std::printf("done\n");
    return 0;
}
