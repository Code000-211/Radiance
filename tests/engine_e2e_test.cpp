/* engine_e2e_test.cpp -- the engine end to end on the CPU: the radiance binary serving tiny
 * synthetic checkpoints (tests/tiny_model.h) on the host backend.
 *
 * libref computes every row on its own, with no reduction whose order depends on how many rows a
 * step carries. On it, how a request is scheduled must not change one bit of its logits: the
 * chunks its prompt is split into, the sequences batched beside it, a preemption that recomputes
 * it, a prefix-cache hit that resumes it from a linear-state checkpoint, or the speculative window
 * its tokens are verified in. A card cannot offer that (split-K depends on M), so these invariants
 * are only checkable here. The one path difference is a gated delta net's: its decode recurrence
 * and its chunked prefill round differently, so a decoded token replayed by prefill (preemption
 * mid-decode) agrees within rounding, not to the bit.
 *
 * Two instruments, both through the real binary in a child process:
 *   - KL mode (core/kld.h). `--kld-record` writes every scored position's log-probabilities, and
 *     two recordings of one corpus are compared byte for byte. `--kld-ref` scores a run whose
 *     arithmetic legitimately differs (two ranks, libavx) against a tolerance.
 *   - HTTP on loopback. The server binds port 0 and the test reads the bound port from its log;
 *     requests go to llama-server's native /completion with token-id prompts and return_tokens.
 *
 * Every child gets ROCR_VISIBLE_DEVICES=99, so a machine with a card still runs the host backend.
 */
#include "rad_test.h"
#include "tiny_golden.h"
#include "tiny_model.h"

#include "mm/media.h"
#include "server/oai.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>

#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

using json = nlohmann::json;

namespace {

/* ------------------------------------------------------------------ the children */

/* The host backend with a fixed device size, and OpenMP that sleeps between parallel regions, so
 * concurrent test binaries do not starve each other. */
std::vector<std::string> engine_env(const std::vector<std::string>& more = {}) {
    std::vector<std::string> e = { "ROCR_VISIBLE_DEVICES=99", "RADIANCE_HOST_VRAM_MIB=512",
                                   "OMP_WAIT_POLICY=passive", "OMP_NUM_THREADS=2",
                                   std::string("RADIANCE_HOME=") + RAD_E2E_HOME };
    e.insert(e.end(), more.begin(), more.end());
    return e;
}

/* libref alone unless the arguments name other libraries. An op resolving to a reference library
 * is refused without the debug flag, so it is passed whenever libref is loaded; a list without
 * libref runs as a deployment does, and the gate holds it to that. */
std::vector<std::string> engine_argv(const std::string& model,
                                     const std::vector<std::string>& args) {
    std::vector<std::string> a = { RAD_E2E_SERVER, "--model", model };
    bool kernels = false, ref = true;
    for (size_t i = 0; i < args.size(); ++i)
        if (args[i] == "--kernels") {
            kernels = true;
            ref = i + 1 < args.size() && args[i + 1].find("libref") != std::string::npos;
        }
    if (!kernels) { a.push_back("--kernels"); a.push_back("libref"); }
    if (ref) a.push_back("--debug-accept-reference-kernels");
    a.insert(a.end(), args.begin(), args.end());
    return a;
}

/* As a deployment runs: the arguments as given, so a reference library serves nothing. */
std::vector<std::string> serving_argv(const std::string& model,
                                      const std::vector<std::string>& args) {
    std::vector<std::string> a = { RAD_E2E_SERVER, "--model", model };
    a.insert(a.end(), args.begin(), args.end());
    return a;
}

/* The shape the preemption cases run in: four sequences of at most 512 tokens, 256 a step. Their
 * small KV pool comes from a small device, whose size small_device() finds. */
std::vector<std::string> small_pool_args() {
    return { "--max-num-seqs", "4", "--max-num-batched-tokens", "256", "--max-model-len", "512" };
}
std::vector<std::string> device_env(int device_mib) {
    return { "RADIANCE_HOST_VRAM_MIB=" + std::to_string(device_mib) };
}

/* The KV pool a startup report states ("kv cache ... a card: N tokens"), or -1. */
int64_t pool_tokens(const std::string& log) {
    static const std::string kTag = " a card: ";
    const size_t at = log.rfind(kTag);
    return at == std::string::npos ? -1 : std::atoll(log.c_str() + at + kTag.size());
}

/* A device size, in MiB, whose KV pool lies in [lo, hi): at least the longest request, fewer
 * tokens than the requests reach together. `probe(mib)` starts the engine on a device of that size
 * and returns its pool, or -1 when it does not start; bisection finds the smallest size with a
 * pool of at least `lo`, or, with `largest`, the largest with fewer than `hi`. The largest suits
 * decoding, where a smaller pool preempts again and again and each preemption recomputes a whole
 * request; a prefill-only run finishes its documents quickly and needs the smallest to preempt at
 * all. Found rather than fixed because the budget's reserves and rounding decide what a size leaves
 * the cache; a change to them then fails here, by name, instead of quietly running a case that no
 * longer preempts. 0, with `why`, when no size gives such a pool. */
int small_device(const std::function<int64_t(int)>& probe, int64_t lo, int64_t hi, bool largest,
                 int64_t* pool, std::string* why) {
    std::map<int, int64_t> seen;
    auto pool_at = [&](int mib) {
        auto it = seen.find(mib);
        return it != seen.end() ? it->second : (seen[mib] = probe(mib));
    };
    const int64_t edge = largest ? hi : lo;
    int a = 64, b = 1024;            /* pool_at(a) < edge <= pool_at(b) */
    if (pool_at(b) < edge || pool_at(a) >= edge) {
        *why = "pools of " + std::to_string(pool_at(a)) + " and " + std::to_string(pool_at(b)) +
               " tokens at " + std::to_string(a) + " and " + std::to_string(b) +
               " MiB do not bracket " + std::to_string(edge);
        return 0;
    }
    while (b - a > 1) {
        const int m = (a + b) / 2;
        if (pool_at(m) >= edge) b = m; else a = m;
    }
    const int mib = largest ? a : b;
    *pool = pool_at(mib);
    if (*pool < lo || *pool >= hi) {
        *why = "the pool steps from " + std::to_string(pool_at(a)) + " tokens at " +
               std::to_string(a) + " MiB to " + std::to_string(pool_at(b)) + " at " +
               std::to_string(b) + ", past the window [" + std::to_string(lo) + ", " +
               std::to_string(hi) + "): the cache's share grows in steps too coarse for this model";
        return 0;
    }
    return mib;
}

struct Ran {
    int status = -1;
    std::string log;
};

Ran radiance(const std::string& model, const std::vector<std::string>& args,
             const std::vector<std::string>& env = {}, int timeout_s = 0) {
    Ran r;
    r.status = tiny::run(engine_argv(model, args), engine_env(env), &r.log, timeout_s);
    return r;
}

uint64_t fnv1a(const std::string& bytes) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : bytes) h = (h ^ c) * 0x100000001b3ull;
    return h;
}

/* The last `n` lines of a child's log, for a failure message. */
std::string tail(const std::string& log, int n = 12) {
    size_t at = log.size();
    for (int i = 0; i <= n && at != std::string::npos && at > 0; ++i) at = log.rfind('\n', at - 1);
    return at == std::string::npos || at >= log.size() ? log : log.substr(at + 1);
}

#define REQUIRE_RAN(r)                                                                  \
    do { if ((r).status != 0) fprintf(stderr, "    child exited %d:\n%s\n", (r).status, \
                                      tail((r).log).c_str());                         \
         REQUIRE_EQ((r).status, 0); } while (0)

/* ------------------------------------------------------------------ the checkpoints */

/* `n` bytes of word salad from a fixed seed: printable, so one byte is one token. */
std::string words(tiny::Rng& r, size_t n) {
    static const char* w[] = { "the", "a", "cat", "dog", "sat", "ran", "on", "under", "mat",
                               "house", "tree", "river", "stone", "light", "dark", "quick",
                               "slow", "red", "blue", "and" };
    std::string s;
    while (s.size() < n) { s += w[r.next() % 20]; s += ' '; }
    s.resize(n);
    return s;
}

/* Every checkpoint and corpus the cases share, written once per process. */
struct World {
    /* RAD_E2E_KEEP keeps the checkpoints for tests/mktinygolden.py, which reads them. */
    tiny::TempDir dir{ "e2e", std::getenv("RAD_E2E_KEEP") != nullptr };
    std::string qwen35, lfm2, gemma4, llama, llama_rad;
    /* llama.rad with its lm head block fp8, and int8 per (row, 128 of K). */
    std::string llama_fp8head, llama_i8head;
    /* The llama as an int4 W4A16 release with zero points (tiny_model.h, write_llama_ct_int4),
     * and its container through the import rule. */
    std::string llama_int4, llama_int4_rad;
    std::string llama_mxfp4;   /* llama's trunk as OCP-MXFP4 */
    /* qwen35 and gemma4 with 4 KiB of KV a token (wider KV heads), for the preemption cases: a MiB
     * of device is then about 256 tokens of pool, fine enough for small_device() to land between
     * one request and four. */
    std::string qwen35_kv, gemma4_kv;
    /* qwen35 with a weaker residual branch (0.33, see tiny_model.h Gains), for the speculation
     * cases: its text on the repetitive prompt repeats n-grams, so prompt lookup drafts and the
     * model accepts some drafts and rejects others, where the default checkpoint's text repeats
     * none and nothing is drafted. */
    std::string qwen35_spec;
    /* Qwen3-ASR: an audio tower over a Qwen3 decoder, served from the checkpoint. */
    std::string qwen3asr;
    std::string corpus;        /* six documents, two sharing a 300-token head; 1820 tokens */
    std::string corpus_short;  /* four documents, 760 tokens */
    std::string corpus_tiny;   /* two documents, 300 tokens, for libref's slow fp8 GEMMs */
    std::string corpus_probe;  /* one 40-token document, for a startup that only reports */
    std::string convert_log;
    bool ok = false;

    World() {
        if (!dir.ok()) return;
        if (std::getenv("RAD_E2E_KEEP"))
            fprintf(stderr, "    checkpoints kept in %s\n", dir.path.c_str());
        qwen35 = dir / "qwen35";
        lfm2 = dir / "lfm2";
        gemma4 = dir / "gemma4";
        llama = dir / "llama";
        llama_rad = dir / "llama.rad";
        llama_fp8head = dir / "llama_fp8head.rad";
        llama_i8head = dir / "llama_i8head.rad";
        llama_int4 = dir / "llama_int4";
        llama_int4_rad = dir / "llama_int4.rad";
        llama_mxfp4 = dir / "llama_mxfp4.rad";
        qwen35_kv = dir / "qwen35_kv";
        gemma4_kv = dir / "gemma4_kv";
        qwen35_spec = dir / "qwen35_spec";
        qwen3asr = dir / "qwen3asr";
        for (const std::string* d : { &qwen35, &lfm2, &gemma4, &llama, &qwen35_kv, &gemma4_kv,
                                      &qwen35_spec, &qwen3asr, &llama_int4 })
            if (mkdir(d->c_str(), 0755) != 0) return;
        tiny::Qwen35 qkv;
        qkv.head_dim = 128;
        tiny::Qwen35 qspec;
        qspec.branch = 0.33f;
        tiny::Gemma4 gkv;
        gkv.kv_heads = 4;
        gkv.head_dim = 64;
        gkv.global_kv_heads = 2;
        gkv.global_head_dim = 128;
        if (!tiny::write_qwen35(qwen35, {}) || !tiny::write_lfm2(lfm2, {}) ||
            !tiny::write_gemma4(gemma4, {}) || !tiny::write_llama(llama, {}) ||
            !tiny::write_qwen35(qwen35_kv, qkv) || !tiny::write_gemma4(gemma4_kv, gkv) ||
            !tiny::write_qwen35(qwen35_spec, qspec) || !tiny::write_qwen3asr(qwen3asr, {}) ||
            !tiny::write_text(dir / "llama.recipe", tiny::kLlamaFp8Recipe) ||
            !tiny::write_text(dir / "llama_fp8head.recipe",
                              std::string(tiny::kLlamaFp8Recipe) + tiny::kLlamaFp8HeadRule) ||
            !tiny::write_text(dir / "llama_i8head.recipe",
                              std::string(tiny::kLlamaFp8Recipe) + tiny::kLlamaI8HeadRule) ||
            !tiny::write_llama_ct_int4(llama_int4, {}, true) ||
            !tiny::write_text(dir / "llama_int4.recipe", tiny::kInt4ImportRecipe) ||
            !tiny::write_text(dir / "llama_mxfp4.recipe", tiny::kLlamaMxfp4Recipe))
            return;

        tiny::Rng r(11);
        const std::string head = words(r, 300);
        std::vector<std::string> docs = { head + words(r, 100), head + words(r, 120) };
        for (size_t n : { 150, 220, 280, 350 }) docs.push_back(words(r, n));
        std::vector<std::string> shorts;
        for (size_t n : { 130, 170, 210, 250 }) shorts.push_back(words(r, n));
        corpus = dir / "corpus.jsonl";
        corpus_short = dir / "corpus_short.jsonl";
        corpus_tiny = dir / "corpus_tiny.jsonl";
        corpus_probe = dir / "corpus_probe.jsonl";
        if (!write_corpus(corpus, docs) || !write_corpus(corpus_short, shorts) ||
            !write_corpus(corpus_tiny, { shorts[0], shorts[1] }) ||
            !write_corpus(corpus_probe, { words(r, 40) }))
            return;

        const struct { std::string out; const char* recipe; const std::string* ckpt; }
            kConverts[] = { { llama_rad, "llama.recipe", &llama },
                            { llama_fp8head, "llama_fp8head.recipe", &llama },
                            { llama_i8head, "llama_i8head.recipe", &llama },
                            { llama_int4_rad, "llama_int4.recipe", &llama_int4 },
                            { llama_mxfp4, "llama_mxfp4.recipe", &llama } };
        for (const auto& [out, recipe, ckpt] : kConverts) {
            const int st = tiny::run({ RAD_E2E_CONVERT, *ckpt, "-o", out, "--recipe",
                                       dir / recipe },
                                     engine_env(), &convert_log);
            if (st != 0) {
                fprintf(stderr, "    rad-convert exited %d:\n%s\n", st, tail(convert_log).c_str());
                return;
            }
        }
        ok = true;
    }

    static bool write_corpus(const std::string& path, const std::vector<std::string>& docs) {
        std::string body;
        for (size_t i = 0; i < docs.size(); ++i)
            body += json{ { "prompt", docs[i] }, { "source", "d" + std::to_string(i) } }.dump() +
                    "\n";
        return tiny::write_text(path, body);
    }

    /* The reference recording of `model` over `corpus`: one sequence a step, a step of 512
     * tokens. Recorded once and reused by every case that compares with it. */
    std::map<std::string, std::string> refs;
    std::string reference(const std::string& model, const std::string& corp, std::string* log) {
        const std::string key = model + "|" + corp;
        auto it = refs.find(key);
        if (it != refs.end()) return it->second;
        const std::string out = dir / ("ref" + std::to_string(refs.size()));
        Ran r = radiance(model, { "--kld-record", out, "--kld-corpus", corp,
                                  "--max-num-batched-tokens", "512", "--max-num-seqs", "1" });
        *log = r.log;
        if (r.status != 0) return std::string();
        refs[key] = out;
        return out;
    }
};

World& world() {
    static World w;
    return w;
}

/* small_device()'s probe for the KL mode: record the 40-token probe document on a device of `mib`
 * MiB in the preemption shape, each probe into its own directory under `d`, and read the pool from
 * the startup report. */
std::function<int64_t(int)> kl_pool_at(const std::string& model, const tiny::TempDir& d) {
    auto n = std::make_shared<int>(0);
    return [model, &d, n](int mib) -> int64_t {
        std::vector<std::string> a = { "--kld-record", d / ("p" + std::to_string((*n)++)),
                                       "--kld-corpus", world().corpus_probe };
        for (const std::string& s : small_pool_args()) a.push_back(s);
        const Ran r = radiance(model, a, device_env(mib));
        return r.status == 0 ? pool_tokens(r.log) : -1;
    };
}

/* The files of two recordings that hold numbers. The manifest names the run, so it differs. */
bool same_recording(const std::string& a, const std::string& b, std::string* why) {
    for (const char* f : { "tokens.i32", "logp.f16", "top.bin", "rows.f32" }) {
        const std::string x = tiny::read_text(a + "/" + f), y = tiny::read_text(b + "/" + f);
        if (x.empty() || x != y) {
            *why = std::string(f) + (x.empty() ? " is missing or empty" : " differs");
            return false;
        }
    }
    return true;
}

/* Records `model` over `corp` with `args` and requires every number to equal the reference's. */
void check_recording_matches(const std::string& model, const std::string& corp,
                             const std::vector<std::string>& args,
                             const std::vector<std::string>& env = {}, std::string* log = nullptr) {
    World& w = world();
    REQUIRE(w.ok);
    std::string rlog;
    const std::string ref = w.reference(model, corp, &rlog);
    if (ref.empty()) fprintf(stderr, "    reference run:\n%s\n", tail(rlog).c_str());
    REQUIRE(!ref.empty());
    tiny::TempDir out("e2e_rec");
    std::vector<std::string> a = { "--kld-record", out / "rec", "--kld-corpus", corp };
    a.insert(a.end(), args.begin(), args.end());
    Ran r = radiance(model, a, env);
    REQUIRE_RAN(r);
    std::string why;
    const bool same = same_recording(ref, out / "rec", &why);
    if (!same) fprintf(stderr, "    %s\n", why.c_str());
    CHECK(same);
    if (log) *log = r.log;
}

/* Scores `model` against the reference and returns the --kld-out report. */
json score_against_reference(const std::string& model, const std::string& corp,
                             const std::vector<std::string>& args,
                             const std::vector<std::string>& env = {}) {
    World& w = world();
    std::string rlog;
    const std::string ref = w.reference(model, corp, &rlog);
    if (ref.empty()) return json();
    tiny::TempDir out("e2e_kld");
    std::vector<std::string> a = { "--kld-ref", ref, "--kld-out", out / "kld.json" };
    a.insert(a.end(), args.begin(), args.end());
    Ran r = radiance(model, a, env);
    if (r.status != 0) {
        fprintf(stderr, "    child exited %d:\n%s\n", r.status, tail(r.log).c_str());
        return json();
    }
    return json::parse(tiny::read_text(out / "kld.json"), nullptr, false);
}

/* Agreement within rounding: a different summation order or another library's bf16 rounding gives
 * these models a mean KL of 1e-5 to 3e-4 and top-1 agreement of 98% or more. A wrong vocabulary
 * column, row or state gives KL in nats and top-1 near chance. */
void check_within_rounding(const char* what, const json& rep) {
    REQUIRE(rep.is_object() && rep.contains("all"));
    const double mean = rep["all"]["kld"]["mean"].get<double>();
    const double top1 = rep["all"]["top1_agreement"].get<double>();
    fprintf(stderr, "    %s: mean KL %.2g, top-1 %.4f\n", what, mean, top1);
    CHECK(mean < 0.005);
    CHECK(top1 > 0.95);
}

/* The preemptions a KL run reports in its closing line. */
long long preemptions_in(const std::string& log) {
    const size_t at = log.find(" preemption(s)");
    if (at == std::string::npos) return -1;
    size_t b = log.rfind(' ', at - 1);
    return b == std::string::npos ? -1 : std::atoll(log.c_str() + b + 1);
}

/* ------------------------------------------------------------------ the server */

/* A server on a free loopback port. Its output is drained on a thread so it never blocks on a full
 * pipe; start() returns once the transport reports the port it bound. */
struct Server {
    tiny::Child child;
    std::thread drain;
    std::mutex mu;
    std::condition_variable cv;
    std::string log;
    bool eof = false;
    int port = 0;
    bool quiet = false;   /* a start that fails is expected: print nothing */
    bool serving = false; /* serving_argv: no reference library unless the arguments name one */

    bool start(const std::string& model, const std::vector<std::string>& args,
               const std::vector<std::string>& env = {}) {
        std::vector<std::string> a = { "--host", "127.0.0.1", "--port", "0",
                                       "--http-threads", "4" };
        a.insert(a.end(), args.begin(), args.end());
        if (!child.start(serving ? serving_argv(model, a) : engine_argv(model, a), engine_env(env)))
            return false;
        drain = std::thread([this] {
            char buf[4096];
            for (;;) {
                const ssize_t n = ::read(child.fd, buf, sizeof buf);
                if (n < 0 && errno == EINTR) continue;
                std::lock_guard<std::mutex> lk(mu);
                if (n <= 0) { eof = true; cv.notify_all(); return; }
                log.append(buf, (size_t)n);
                cv.notify_all();
            }
        });
        static const std::string kTag = "listening on http://127.0.0.1:";
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return eof || log.find(kTag) != std::string::npos; });
        const size_t at = log.find(kTag);
        if (at != std::string::npos) port = std::atoi(log.c_str() + at + kTag.size());
        if (port <= 0 && !quiet)
            fprintf(stderr, "    the server did not start:\n%s\n", tail(log).c_str());
        return port > 0;
    }

    /* SIGTERM drains and exits 0. The exit status; a second call returns it again. Afterwards
     * output() holds everything the server wrote. */
    int exit_status = -1;
    int stop() {
        if (child.pid <= 0) return exit_status;
        kill(child.pid, SIGTERM);
        if (drain.joinable()) drain.join();
        close(child.fd);
        child.fd = -1;
        std::string rest;
        exit_status = child.wait(&rest);
        return exit_status;
    }

    std::string output() {
        std::lock_guard<std::mutex> lk(mu);
        return log;
    }

    ~Server() {
        if (child.pid > 0) kill(child.pid, SIGKILL);
        if (drain.joinable()) drain.join();
    }
};

/* One request on its own loopback connection, sent and not yet read, so several can be in flight
 * at once. A request that gets no response says why in `why`. */
struct Call {
    int fd = -1;
    std::string why;
    bool fail(const char* what) {
        why = std::string(what) + ": " + std::strerror(errno);
        return false;
    }
    bool send(int port, const char* method, const std::string& path, const std::string& body) {
        fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0) return fail("socket");
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t)port);
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(fd, (sockaddr*)&a, sizeof a) != 0) return fail("connect");
        std::string req = std::string(method) + " " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\n" +
                          "Connection: close\r\nContent-Type: application/json\r\n" +
                          "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
        for (size_t off = 0; off < req.size();) {
            const ssize_t n = ::send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
            if (n <= 0) return fail("send");
            off += (size_t)n;
        }
        return true;
    }
    /* The status and the body; the server closes the connection after it. */
    int read(std::string* body) {
        std::string raw;
        char buf[16384];
        int err = 0;
        for (;;) {
            const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) err = errno;
            if (n <= 0) break;
            raw.append(buf, (size_t)n);
        }
        close(fd);
        fd = -1;
        const size_t h = raw.find("\r\n\r\n");
        if (raw.compare(0, 9, "HTTP/1.1 ") != 0 || h == std::string::npos) {
            why = "no response: " + std::to_string(raw.size()) + " byte(s) before the server " +
                  (err ? std::string("reset the connection (") + std::strerror(err) + ")"
                       : std::string("closed it"));
            return -1;
        }
        std::string head = raw.substr(0, h), rest = raw.substr(h + 4);
        for (char& c : head) c = (char)std::tolower((unsigned char)c);
        if (head.find("transfer-encoding: chunked") != std::string::npos) {
            std::string out;
            for (size_t at = 0;;) {
                const size_t eol = rest.find("\r\n", at);
                if (eol == std::string::npos) break;
                const size_t n = std::strtoul(rest.c_str() + at, nullptr, 16);
                if (n == 0) break;
                out += rest.substr(eol + 2, n);
                at = eol + 2 + n + 2;
            }
            rest = out;
        }
        *body = rest;
        return std::atoi(raw.c_str() + 9);
    }
    ~Call() { if (fd >= 0) close(fd); }
};

json post(int port, const std::string& path, const json& body, int* status = nullptr,
          std::string* why = nullptr) {
    Call c;
    std::string out;
    const int st = c.send(port, "POST", path, body.dump()) ? c.read(&out) : -1;
    if (status) *status = st;
    if (why) *why = c.why;
    return json::parse(out, nullptr, false);
}

/* A counter from /metrics, or -1. */
double metric(int port, const std::string& name) {
    Call c;
    std::string out;
    if (!c.send(port, "GET", "/metrics", "") || c.read(&out) != 200) return -1;
    for (size_t at = 0; (at = out.find(name, at)) != std::string::npos; at += name.size()) {
        if (at > 0 && out[at - 1] != '\n') continue;
        const size_t sp = out.find(' ', at);
        if (sp == std::string::npos) return -1;
        const std::string key = out.substr(at, sp - at);
        if (key != name && key.rfind(name + "{", 0) != 0) continue;
        return std::atof(out.c_str() + sp + 1);
    }
    return -1;
}

/* How many draft positions the run evaluated, or -1: /metrics exports one
 * radiance:draft_position_reached_total line per position the engine drafted to, so the line count
 * is the deepest verify window the run executed. */
int draft_positions(int port) {
    Call c;
    std::string out;
    if (!c.send(port, "GET", "/metrics", "") || c.read(&out) != 200) return -1;
    const std::string tag = "radiance:draft_position_reached_total{";
    int n = 0;
    for (size_t at = 0; (at = out.find(tag, at)) != std::string::npos; at += tag.size())
        if (at == 0 || out[at - 1] == '\n') ++n;
    return n;
}

/* A completion of `n` tokens from token ids: greedy unless `temp` is set, end-of-text ignored so
 * every reply has the same length. */
json completion(const std::vector<int32_t>& prompt, int n, double temp = 0.0, int64_t seed = 0,
                int n_probs = 0) {
    json b = { { "prompt", prompt }, { "n_predict", n }, { "temperature", temp },
               { "ignore_eos", true }, { "return_tokens", true } };
    if (seed) b["seed"] = seed;
    if (n_probs) b["n_probs"] = n_probs;
    return b;
}

std::vector<int32_t> tokens_of(const json& reply) {
    if (!reply.is_object() || !reply.contains("tokens")) return {};
    return reply["tokens"].get<std::vector<int32_t>>();
}

std::string show(const std::vector<int32_t>& t) {
    std::string s;
    for (int32_t v : t) s += std::to_string(v) + " ";
    return s;
}

/* The prompts the HTTP cases send: distinct word salad, as token ids. */
std::vector<int32_t> prompt_ids(uint64_t seed, size_t n) {
    tiny::Rng r(seed);
    return tiny::byte_ids(words(r, n));
}

}  /* namespace */

/* ================================================================== KL mode, qwen35 */

/* The baseline the cases below lean on: the same configuration run twice writes the same numbers.
 * Without it a scheduling-invariance failure could be plain nondeterminism. */
TEST(qwen35_a_rerun_writes_the_same_recording) {
    check_recording_matches(world().qwen35, world().corpus,
                            { "--max-num-batched-tokens", "512", "--max-num-seqs", "1" });
}

/* A smaller prefill chunk splits every document at other positions: the gated delta net's chunked
 * scan restarts mid-tile, the conv window carries across more boundaries and attention reads more
 * of its keys from the paged cache. None of it may change a logit. */
TEST(qwen35_a_smaller_prefill_chunk_changes_no_logit) {
    check_recording_matches(world().qwen35, world().corpus,
                            { "--max-num-batched-tokens", "64", "--max-num-seqs", "1" });
}

/* Under --prefill-span a document alone in the engine is prefilled four checkpoint intervals a
 * step from its second interval on (the startup line and the steps say so), and no logit changes:
 * the engine states the shared budget to the plugin, adds its arena level and plans the wide steps
 * end to end. */
TEST(qwen35_a_prefill_span_changes_no_logit) {
    std::string log;
    check_recording_matches(world().qwen35, world().corpus,
                            { "--max-num-batched-tokens", "256", "--max-num-seqs", "1",
                              "--checkpoint-interval", "64", "--prefill-span", "4" },
                            { "RADIANCE_LOG_STEPS=1" }, &log);
    CHECK(log.find("spans up to 4 intervals from its second interval on; every other step is "
                   "planned at 64 tokens") != std::string::npos);
    CHECK(log.find("prefill n_seq=1 n_tok=256") != std::string::npos);
}

/* Documents batched together share every step's GEMMs and occupy other state slots and KV
 * blocks; each one's logits stay exactly those it gets alone. */
TEST(qwen35_batching_documents_together_changes_no_logit) {
    check_recording_matches(world().qwen35, world().corpus,
                            { "--max-num-batched-tokens", "512", "--max-num-seqs", "4" });
}

/* A KV pool smaller than the documents running at once forces preemption by recompute: the victim's
 * blocks and state slots go back and it is prefilled again from its first token. The recomputed
 * rows must equal the first computation, and the run must actually have preempted. The pool holds
 * the longest document (420 tokens) and less than the first four, admitted together (1190). */
TEST(qwen35_preemption_by_recompute_changes_no_logit) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("e2e_probe");
    int64_t pool = 0;
    std::string why;
    const int mib = small_device(kl_pool_at(w.qwen35_kv, d), 420, 420 + 400 + 150 + 220, false,
                                 &pool, &why);
    if (!mib) fprintf(stderr, "    no small pool: %s\n", why.c_str());
    REQUIRE(mib > 0);
    std::string log;
    check_recording_matches(w.qwen35_kv, w.corpus, small_pool_args(), device_env(mib), &log);
    const long long n = preemptions_in(log);
    if (n <= 0)
        fprintf(stderr, "    %d MiB, a pool of %lld tokens, preempted %lld times\n%s\n", mib,
                (long long)pool, n, tail(log, 40).c_str());
    CHECK(n > 0);
}

/* A document longer than the whole KV pool could never be admitted: admission takes only a chunk
 * the pool can hold, so it would wait at the head of the queue forever. The run is refused before
 * anything is scored, naming the document's length and the pool's. Bounded by a timeout, so the
 * wedge it guards against fails the case instead of hanging it. */
TEST(a_kl_document_longer_than_the_kv_pool_is_refused_up_front) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("e2e_probe");
    int64_t pool = 0;
    std::string why;
    /* The largest pool shorter than the longest document of the short corpus (250 tokens). */
    const int mib = small_device(kl_pool_at(w.qwen35_kv, d), 1, 250, true, &pool, &why);
    if (!mib) fprintf(stderr, "    no pool shorter than the document: %s\n", why.c_str());
    REQUIRE(mib > 0);
    std::vector<std::string> a = { "--kld-record", d / "rec", "--kld-corpus", w.corpus_short };
    for (const std::string& s : small_pool_args()) a.push_back(s);
    const Ran r = radiance(w.qwen35_kv, a, device_env(mib), 120);
    const std::string want =
        "document 3 is 250 tokens and the KV pool holds " + std::to_string(pool);
    if (r.status == 0 || r.status == 124 || !has(r.log, want))
        fprintf(stderr, "    %d MiB, a pool of %lld tokens: exited %d\n%s\n", mib, (long long)pool,
                r.status, tail(r.log, 20).c_str());
    CHECK(r.status != 0);
    CHECK(r.status != 124);
    CHECK(has(r.log, want));
    CHECK(!has(r.log, "document 2 is"));   /* 210 tokens: it fits, and is not named */
}

/* A checkpoint served directly whose quantization_config is a tool's form the import does not
 * read keeps the tool's descriptor, which no plugin claims: the refusal says why, naming the
 * form, not only that no plugin claims the quantisation. */
TEST(a_checkpoint_in_a_quantisation_the_import_refuses_says_why) {
    tiny::TempDir d("e2e_unread");
    REQUIRE(d.ok());
    REQUIRE(tiny::write_qwen35(d.path, {}));
    json c = json::parse(tiny::read_text(d / "config.json"));
    c["quantization_config"] = {
        { "quant_method", "compressed-tensors" }, { "format", "int-quantized" },
        { "config_groups", { { "group_0", { { "format", "int-quantized" },
                                            { "weights", { { "num_bits", 8 }, { "type", "int" } } } } } } } };
    REQUIRE(tiny::write_text(d / "config.json", c.dump(2)));
    const Ran r = radiance(d.path, {}, {}, 120);
    if (r.status == 0 || !has(r.log, "int-quantized"))
        fprintf(stderr, "    exited %d:\n%s\n", r.status, tail(r.log, 20).c_str());
    CHECK(r.status != 0);
    CHECK(r.status != 124);
    CHECK(has(r.log, "quantisation 'compressed-tensors' cannot be imported"));
    CHECK(has(r.log, "config group 'group_0' is in the format 'int-quantized'"));
}

/* A checkpoint store larger than the host memory left is refused at startup by name, with its size
 * and the flag that sets it, rather than pinned into reclaim. Bounded by a timeout: a server that
 * came up would listen until it. */
TEST(a_checkpoint_store_the_host_cannot_hold_is_refused_at_startup) {
    World& w = world();
    REQUIRE(w.ok);
    const Ran r = radiance(w.qwen35, { "--host", "127.0.0.1", "--port", "0",
                                       "--checkpoint-slots", "1000000000" }, {}, 120);
    const bool named = has(r.log, "checkpoint slots: 1000000000 linear-state snapshot(s) of ") &&
                       has(r.log, "Lower --checkpoint-slots");
    if (r.status == 0 || r.status == 124 || !named)
        fprintf(stderr, "    exited %d:\n%s\n", r.status, tail(r.log, 20).c_str());
    CHECK(r.status != 0);
    CHECK(r.status != 124);
    CHECK(named);
}

/* RADIANCE_MTP_COUPLED is the coupling's only switch, and these are the engine's answers to it
 * before any request: a value that is not 0/1 refuses the start (never a silent uncoupled run),
 * and so does a model whose drafter has no per-round draw to couple. Both are checked through the
 * real binary because both are bringup paths, not helper functions; the drafter shape is the
 * declaration the engine has in hand, which no environment variable describes. The same model with
 * the flag off is the control: it starts, which is what makes the two refusals attributable to the
 * switch and not to the fixture. */
TEST(the_coupling_switch_refuses_what_it_cannot_serve_at_startup) {
    World& w = world();
    REQUIRE(w.ok);
    /* qwen35's tiny checkpoint declares no draft head: no serial chain to couple. The spechub
     * kernels are loaded beside libref, so the refusal is the drafter's shape and not a missing
     * coupling kernel. */
    const Ran on = radiance(w.qwen35, { "--kernels", "libref,spechub" },
                            { "RADIANCE_MTP_COUPLED=1" }, 120);
    const std::string want = "RADIANCE_MTP_COUPLED requires the linear serial draft chain "
                             "(RAD_DRAFT_SERIAL with depth > 0)";
    if (on.status == 0 || on.status == 124 || !has(on.log, want))
        fprintf(stderr, "    RADIANCE_MTP_COUPLED=1 exited %d:\n%s\n", on.status,
                tail(on.log, 20).c_str());
    CHECK(on.status != 0);
    CHECK(on.status != 124);
    CHECK(has(on.log, want));
    CHECK(has(on.log, "declares drafter '(none)'"));

    /* THE GATE IS NOT RELAXED EITHER WAY: the same start as a deployment runs it (no
     * --debug-accept-reference-kernels), with a production library for the model's ops and the
     * spechub library beside it. On this card-less machine the plugin's device rows are left out
     * of selection (no card is visible), so the only coupled row left is the host oracle -- and
     * the gate must still refuse it, naming the coupled op. Marking the library, or the oracle
     * rows, production would pass this start on a machine that cannot run the coupling at all. */
    if (::access(RAD_E2E_HOME "/kernels/libavx.so", R_OK) == 0) {
        std::string gate_log;
        const int gate = tiny::run(serving_argv(w.qwen35, { "--kernels", "libavx,spechub" }),
                                   engine_env({ "RADIANCE_MTP_COUPLED=1" }), &gate_log, 120);
        const std::string gate_ref = "1 declared op(s) resolve only to a reference implementation "
                                     "(spechub)";
        if (gate == 0 || gate == 124 || !has(gate_log, gate_ref) ||
            !has(gate_log, "1 distinct op(s): mtp_couple_verify"))
            fprintf(stderr, "    serving start exited %d:\n%s\n", gate,
                    tail(gate_log, 20).c_str());
        CHECK(gate != 0);
        CHECK(gate != 124);
        CHECK(has(gate_log, gate_ref));
        CHECK(has(gate_log, "distinct op(s): mtp_couple_verify"));
    }

    /* A typo is refused, not read as off. */
    const Ran typo = radiance(w.qwen35, {}, { "RADIANCE_MTP_COUPLED=yes" }, 120);
    const std::string refused = "RADIANCE_MTP_COUPLED accepts 0|1 (or unset)";
    if (typo.status == 0 || typo.status == 124 || !has(typo.log, refused))
        fprintf(stderr, "    RADIANCE_MTP_COUPLED=yes exited %d:\n%s\n", typo.status,
                tail(typo.log, 20).c_str());
    CHECK(typo.status != 0);
    CHECK(typo.status != 124);
    CHECK(has(typo.log, refused));

    /* The control on the same checkpoint: flag off starts and declares, so both refusals above are
     * the switch's answer and not the fixture's. --debug-graph runs declare and plan and exits, so
     * it is a start, not a serve. */
    const Ran off = radiance(w.qwen35, { "--debug-graph" }, { "RADIANCE_MTP_COUPLED=0" }, 120);
    if (off.status != 0)
        fprintf(stderr, "    RADIANCE_MTP_COUPLED=0 exited %d:\n%s\n", off.status,
                tail(off.log, 20).c_str());
    CHECK_EQ(off.status, 0);
    CHECK(!has(off.log, "MTP coupled LIVE"));
}

/* Two ranks split heads, the FFN and the vocabulary, and sum partial products in an all-reduce,
 * so the rounding differs from one rank's; anything more than rounding is a wrong column or row
 * (a rank's logits read at another rank's vocabulary offset score near chance). */
TEST(qwen35_two_ranks_agree_with_one_within_rounding) {
    REQUIRE(world().ok);
    check_within_rounding("qwen35 at two ranks", score_against_reference(
        world().qwen35, world().corpus,
        { "--tp", "2", "--max-num-batched-tokens", "512", "--max-num-seqs", "1" },
        { "RADIANCE_HOST_DEVICES=2" }));
}

/* ================================================================== runtime LoRA */

/* A PEFT adapter over the qwen35 checkpoint on every linear runtime LoRA follows (the gated
 * attention, the delta net's in_proj_qkv, in_proj_z and out_proj, the MLP), and the checkpoint with
 * it merged in. Written once, by the first case that needs them. */
const std::vector<std::string> kLoraTargets = { "q_proj",      "k_proj",    "v_proj",
                                                "o_proj",      "in_proj_qkv", "in_proj_z",
                                                "out_proj",    "gate_proj", "up_proj",
                                                "down_proj" };
struct LoraWorld {
    tiny::TempDir dir{ "e2e_lora" };
    std::string adapter, merged;
    bool ok = false;
    LoraWorld() {
        if (!dir.ok() || !world().ok) return;
        adapter = dir / "adapter";
        merged = dir / "merged";
        ok = tiny::write_lora(adapter, world().qwen35, kLoraTargets, 8, 16.0, 21, 0.25f, merged);
    }
};
LoraWorld& lora_world() {
    static LoraWorld l;
    return l;
}

/* `mixed` against two recordings of the same corpus, document by document: those whose source
 * starts with `prefix` must be `a`'s rows byte for byte, the others `b`'s. */
bool spliced_recording_matches(const std::string& mixed, const std::string& a,
                               const std::string& b, const std::string& prefix, std::string* why) {
    const json m = json::parse(tiny::read_text(mixed + "/kld.json"), nullptr, false);
    if (!m.is_object() || !m.contains("docs")) { *why = "no manifest"; return false; }
    if (tiny::read_text(mixed + "/tokens.i32") != tiny::read_text(a + "/tokens.i32")) {
        *why = "tokens.i32 differs";
        return false;
    }
    const int64_t n_vocab = m["n_vocab"].get<int64_t>(), top = m["top"].get<int64_t>();
    const std::pair<const char*, int64_t> files[] = {
        { "logp.f16", n_vocab * 2 }, { "top.bin", top * 8 }, { "rows.f32", 8 } };
    for (const auto& [f, row_bytes] : files) {
        const std::string x = tiny::read_text(mixed + "/" + f);
        const std::string ya = tiny::read_text(a + "/" + f), yb = tiny::read_text(b + "/" + f);
        if (x.empty() || x.size() != ya.size() || x.size() != yb.size()) {
            *why = std::string(f) + " is missing or of another size";
            return false;
        }
        for (const json& d : m["docs"]) {
            const std::string src = d["source"].get<std::string>();
            const int64_t row0 = d["row0"].get<int64_t>();
            const int64_t rows = d["tokens"].get<int64_t>() - 1 - d["score_from"].get<int64_t>();
            const std::string& want = src.rfind(prefix, 0) == 0 ? ya : yb;
            const size_t at = (size_t)(row0 * row_bytes), len = (size_t)(rows * row_bytes);
            if (x.compare(at, len, want, at, len) != 0) {
                *why = std::string(f) + ": document " + src + " is not " +
                       (&want == &ya ? "the adapter" : "the base") + " recording's";
                return false;
            }
        }
    }
    return true;
}

/* The adapter served at runtime scores as the merged checkpoint does, within rounding, at one rank
 * and at two (where row-parallel GEMMs carry the delta through their all-reduce); the base model
 * against the same reference is well outside it, so a delta that went missing fails here. */
TEST(runtime_lora_scores_as_the_merged_checkpoint_does) {
    World& w = world();
    LoraWorld& L = lora_world();
    REQUIRE(L.ok);
    tiny::TempDir d("e2e_lora_ref");
    const std::string ref = d / "merged";
    const std::vector<std::string> one = { "--max-num-batched-tokens", "512", "--max-num-seqs", "1" };
    std::vector<std::string> rec = { "--kld-record", ref, "--kld-corpus", w.corpus_short };
    rec.insert(rec.end(), one.begin(), one.end());
    const Ran merged = radiance(L.merged, rec);
    REQUIRE_RAN(merged);
    auto score = [&](const std::vector<std::string>& args, const std::vector<std::string>& env) {
        tiny::TempDir o("e2e_lora_kld");
        std::vector<std::string> a = { "--kld-ref", ref, "--kld-out", o / "kld.json" };
        a.insert(a.end(), one.begin(), one.end());
        a.insert(a.end(), args.begin(), args.end());
        const Ran r = radiance(w.qwen35, a, env);
        if (r.status != 0) {
            fprintf(stderr, "    child exited %d:\n%s\n", r.status, tail(r.log).c_str());
            return json();
        }
        return json::parse(tiny::read_text(o / "kld.json"), nullptr, false);
    };
    const std::vector<std::string> ad = { "--lora", "ad=" + L.adapter, "--kld-lora", "ad" };
    check_within_rounding("runtime LoRA against the merged checkpoint", score(ad, {}));
    std::vector<std::string> tp2 = { "--tp", "2" };
    tp2.insert(tp2.end(), ad.begin(), ad.end());
    check_within_rounding("runtime LoRA at two ranks against the merged checkpoint",
                          score(tp2, { "RADIANCE_HOST_DEVICES=2" }));
#if defined(__x86_64__)
    /* libavx alone serves the adapter ops too, as it serves every other op. */
    if (::access(RAD_E2E_HOME "/kernels/libavx.so", R_OK) == 0) {
        std::vector<std::string> avx = { "--kernels", "libavx" };
        avx.insert(avx.end(), ad.begin(), ad.end());
        check_within_rounding("runtime LoRA on libavx against the merged checkpoint",
                              score(avx, {}));
    }
#endif
    const json base = score({}, {});
    REQUIRE(base.is_object() && base.contains("all"));
    const double kl = base["all"]["kld"]["mean"].get<double>();
    fprintf(stderr, "    the base against the merged checkpoint: mean KL %.2g\n", kl);
    CHECK(kl > 0.05);
}

/* An MXFP4 trunk: its GEMM's `group` is the weight's 32-wide exponent block, while the
 * activation it reads is scaled per 128, and the adapter's shrink reads that activation at the
 * activation's grid. Against the merged checkpoint, the adapter takes the container from the
 * adapter's whole effect down to the MXFP4 rounding, on libref and on libavx. (The tiny llama's
 * MXFP4 container declares at one rank only.) */
TEST(runtime_lora_reads_an_mxfp4_trunks_activation_at_its_own_grid) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("e2e_lora_mx");
    const std::string adapter = d / "adapter", merged = d / "merged", ref = d / "ref";
    const std::vector<std::string> targets = { "q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj" };
    REQUIRE(tiny::write_lora(adapter, w.llama, targets, 8, 16.0, 23, 0.25f, merged));
    const std::vector<std::string> one = { "--max-num-batched-tokens", "512", "--max-num-seqs", "1" };
    std::vector<std::string> rec = { "--kld-record", ref, "--kld-corpus", w.corpus_short };
    rec.insert(rec.end(), one.begin(), one.end());
    const Ran recorded = radiance(merged, rec);
    REQUIRE_RAN(recorded);
    auto score = [&](const std::vector<std::string>& args, const std::vector<std::string>& env) {
        tiny::TempDir o("e2e_lora_mx_kld");
        std::vector<std::string> a = { "--kld-ref", ref, "--kld-out", o / "kld.json" };
        a.insert(a.end(), one.begin(), one.end());
        a.insert(a.end(), args.begin(), args.end());
        const Ran r = radiance(w.llama_mxfp4, a, env);
        if (r.status != 0) {
            fprintf(stderr, "    child exited %d:\n%s\n", r.status, tail(r.log).c_str());
            return -1.0;
        }
        const json j = json::parse(tiny::read_text(o / "kld.json"), nullptr, false);
        return j.is_object() && j.contains("all") ? j["all"]["kld"]["mean"].get<double>() : -1.0;
    };
    const double base = score({}, {});
    fprintf(stderr, "    the MXFP4 base against the merged checkpoint: mean KL %.3g\n", base);
    REQUIRE(base > 0.05);
    const std::vector<std::string> ad = { "--lora", "ad=" + adapter, "--kld-lora", "ad" };
    struct Arm { const char* what; std::vector<std::string> args, env; };
    std::vector<Arm> arms = { { "libref", ad, {} } };
#if defined(__x86_64__)
    if (::access(RAD_E2E_HOME "/kernels/libavx.so", R_OK) == 0) {
        std::vector<std::string> avx = { "--kernels", "libavx" };
        avx.insert(avx.end(), ad.begin(), ad.end());
        arms.push_back({ "libavx", avx, {} });
    }
#endif
    for (const Arm& a : arms) {
        const double kl = score(a.args, a.env);
        fprintf(stderr, "    the MXFP4 container with the adapter, %s: mean KL %.3g\n", a.what, kl);
        CHECK(kl >= 0);
        CHECK(kl < base / 4);
    }
}

/* Off is off: without --lora or --max-loras the declared graph names no adapter op, buffer or
 * declined GEMM, and with an adapter it is the same graph plus the adapter ops and their buffers
 * over the same weights. */
TEST(runtime_lora_off_declares_no_adapter_op) {
    World& w = world();
    LoraWorld& L = lora_world();
    REQUIRE(L.ok);
    const Ran off = radiance(w.qwen35, { "--debug-graph" });
    const Ran on = radiance(w.qwen35, { "--debug-graph", "--lora", "ad=" + L.adapter });
    REQUIRE_RAN(off);
    REQUIRE_RAN(on);
    /* The adapter ops, their buffers (`lora.<weight>.a`, `lora.t.r16.lane0`) and a declined fused
     * GEMM's reason; a path may well contain "lora". */
    for (const char* needle : { "lora_shrink", "lora_expand", "lora.", "runtime-LoRA" }) {
        if (has(off.log, needle)) fprintf(stderr, "    the default graph names '%s'\n", needle);
        CHECK(!has(off.log, needle));
    }
    auto counts = [](const std::string& log, long long* ops, long long* weights,
                     long long* buffers) {
        const size_t at = log.find("declare: ");
        return at != std::string::npos &&
               std::sscanf(log.c_str() + at, "declare: %lld ops, %lld weights, %lld buffers", ops,
                           weights, buffers) == 3;
    };
    long long o0 = 0, w0 = 0, b0 = 0, o1 = 0, w1 = 0, b1 = 0;
    REQUIRE(counts(off.log, &o0, &w0, &b0));
    REQUIRE(counts(on.log, &o1, &w1, &b1));
    CHECK_EQ(w1, w0);
    CHECK(o1 > o0 && (o1 - o0) % 2 == 0);
    CHECK(b1 > b0);
    CHECK(has(on.log, "lora_shrink"));
}

/* Base rows are exact: documents run as the base with the adapter loaded are byte for byte what an
 * engine without adapters writes. Mixed steps are exact: with one document through the adapter
 * and the rest as the base, batched four at a time, every document's rows equal those of the run
 * that served all documents its way. */
TEST(runtime_lora_base_rows_and_mixed_steps_are_exact) {
    World& w = world();
    LoraWorld& L = lora_world();
    REQUIRE(L.ok);
    tiny::TempDir d("e2e_lora_mix");
    auto record = [&](const char* name, const std::vector<std::string>& args) {
        std::vector<std::string> a = { "--kld-record", d / name, "--kld-corpus", w.corpus_short,
                                       "--max-num-batched-tokens", "512", "--max-num-seqs", "4" };
        a.insert(a.end(), args.begin(), args.end());
        const Ran r = radiance(w.qwen35, a);
        if (r.status != 0) fprintf(stderr, "    %s exited %d:\n%s\n", name, r.status,
                                   tail(r.log).c_str());
        return r.status == 0;
    };
    const std::string lora = "ad=" + L.adapter;
    REQUIRE(record("off", {}));
    REQUIRE(record("base", { "--lora", lora }));
    REQUIRE(record("adapter", { "--lora", lora, "--kld-lora", "ad" }));
    REQUIRE(record("mixed", { "--lora", lora, "--kld-lora", "ad", "--kld-lora-sources", "d1" }));
    std::string why;
    const bool base_exact = same_recording(d / "off", d / "base", &why);
    if (!base_exact) fprintf(stderr, "    base with the adapter loaded: %s\n", why.c_str());
    CHECK(base_exact);
    CHECK(!same_recording(d / "off", d / "adapter", &why));
    const bool mixed_exact = spliced_recording_matches(d / "mixed", d / "adapter", d / "off",
                                                       "d1", &why);
    if (!mixed_exact) fprintf(stderr, "    mixed steps: %s\n", why.c_str());
    CHECK(mixed_exact);
}

/* What would serve an adapter that cannot land, or one no request could reach, is refused at
 * startup by name: a target module the trunk does not have (or has only as the vocabulary head),
 * and an adapter that has the served model's own name. */
TEST(runtime_lora_refuses_targets_it_cannot_serve_at_startup) {
    World& w = world();
    LoraWorld& L = lora_world();
    REQUIRE(L.ok);
    tiny::TempDir d("e2e_lora_no");
    const std::string lora = "ad=" + L.adapter;
    const struct { const char* targets; const char* why; } kCases[] = {
        { "q_proj,qq_proj", "the target module 'qq_proj' is no trunk linear" },
        { "q_proj,lm_head", "the target module 'lm_head' is no trunk linear" },
    };
    int k = 0;
    for (const auto& c : kCases) {
        const Ran r = radiance(w.qwen35, { "--lora", lora, "--lora-target-modules", c.targets,
                                           "--kld-record", d / ("r" + std::to_string(k++)),
                                           "--kld-corpus", w.corpus_probe }, {}, 120);
        if (r.status == 0 || !has(r.log, c.why))
            fprintf(stderr, "    %s: exited %d:\n%s\n", c.targets, r.status, tail(r.log).c_str());
        CHECK(r.status != 0);
        CHECK(has(r.log, c.why));
    }
    const Ran s = radiance(w.qwen35, { "--served-model-name", "ad", "--lora", lora,
                                       "--host", "127.0.0.1", "--port", "0" }, {}, 120);
    if (s.status == 0 || !has(s.log, "the served model's own name"))
        fprintf(stderr, "    served as 'ad': exited %d:\n%s\n", s.status, tail(s.log).c_str());
    CHECK(s.status != 0);
    CHECK(has(s.log, "has the served model's own name"));
}

/* ================================================================== against transformers */

/* Each family's forward pass against transformers' own, run in float32 over the same weights
 * (tests/tiny_golden.h, from tests/mktinygolden.py): the gated delta net with its query scale and
 * (1 + w) norms, the gated and partial-rotary attention, LFM2's convolutions, Gemma 4's sliding and
 * K-equals-V global layers, soft cap and layer scalars. bf16 activations put this engine within
 * about 0.08 of transformers' log-probabilities; a wrong rotary width, window or epsilon moves
 * them by 0.5 or more and changes the top token at a tenth of the positions. So at every position
 * the top token must agree, unless transformers' own top two are within the bound, and the
 * log-probability of an agreed token must be within 0.2. */
TEST(each_family_computes_what_transformers_computes) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("e2e_gold");
    REQUIRE(World::write_corpus(d / "corpus.jsonl", { kTinyGoldPrompt }));
    for (const TinyGold& g : kTinyGold) {
        const std::string family = g.family;
        const std::string model = family == "qwen35" ? w.qwen35
                                : family == "lfm2"   ? w.lfm2
                                                     : w.gemma4;
        const uint64_t h = fnv1a(tiny::read_text(model + "/model.safetensors"));
        if (h != g.weights_fnv)
            fprintf(stderr, "    %s: the checkpoint is not the one the golden was made from; "
                            "regenerate tests/tiny_golden.h with tests/mktinygolden.py\n",
                    g.family);
        REQUIRE_EQ(h, g.weights_fnv);
        const std::string out = d / family;
        Ran r = radiance(model, { "--kld-record", out, "--kld-corpus", d / "corpus.jsonl" });
        REQUIRE_RAN(r);
        const std::string top = tiny::read_text(out + "/top.bin");
        struct Top { int32_t id; float lp; };
        const size_t row = 16 * sizeof(Top);
        REQUIRE_EQ(top.size(), (size_t)(kTinyGoldPositions - 1) * row);
        int bad = 0;
        double worst = 0;
        for (int p = 0; p + 1 < kTinyGoldPositions; ++p) {
            Top t;
            std::memcpy(&t, top.data() + (size_t)p * row, sizeof t);
            if (t.id == g.id[p][0]) {
                worst = std::max(worst, (double)std::fabs(t.lp - g.logp[p][0]));
                bad += std::fabs(t.lp - g.logp[p][0]) > 0.2f;
                continue;
            }
            /* A near tie in transformers' own ranking may fall either way. */
            float lp_mine = -1e30f;
            for (int k = 0; k < kTinyGoldTop; ++k) if (g.id[p][k] == t.id) lp_mine = g.logp[p][k];
            if (g.logp[p][0] - lp_mine > 0.2f) {
                ++bad;
                fprintf(stderr, "    %s at %d: top token %d, transformers' is %d\n", g.family, p,
                        t.id, g.id[p][0]);
            }
        }
        if (bad || worst > 0.2)
            fprintf(stderr, "    %s: %d position(s) off, largest log-probability gap %g\n",
                    g.family, bad, worst);
        CHECK_EQ(bad, 0);
    }
}

/* ================================================================== KL mode, other families */

/* LFM2's short convolutions keep a rolling window of conv_L_cache - 1 inputs per sequence: a
 * chunk boundary or a different slot must hand it over exactly. */
TEST(lfm2_chunking_and_batching_change_no_logit) {
    check_recording_matches(world().lfm2, world().corpus_short,
                            { "--max-num-batched-tokens", "64", "--max-num-seqs", "1" });
    check_recording_matches(world().lfm2, world().corpus_short,
                            { "--max-num-batched-tokens", "512", "--max-num-seqs", "4" });
}

/* Gemma 4's sliding layers attend over a 32-token window, which every document crosses, and its
 * global layers read V from K at a wider head. */
TEST(gemma4_chunking_and_batching_change_no_logit) {
    check_recording_matches(world().gemma4, world().corpus_short,
                            { "--max-num-batched-tokens", "64", "--max-num-seqs", "1" });
    check_recording_matches(world().gemma4, world().corpus_short,
                            { "--max-num-batched-tokens", "512", "--max-num-seqs", "4" });
}

/* A container rad-convert made block fp8 (llama_dense_fp8): the fp8 activation quantisers and
 * GEMMs are per row too, so the invariants hold through them. */
TEST(llama_fp8_container_chunking_and_batching_change_no_logit) {
    REQUIRE(world().ok);
    check_recording_matches(world().llama_rad, world().corpus_tiny,
                            { "--max-num-batched-tokens", "64", "--max-num-seqs", "1" });
    check_recording_matches(world().llama_rad, world().corpus_tiny,
                            { "--max-num-batched-tokens", "512", "--max-num-seqs", "4" });
}

/* libavx alone, with no libref loaded and no leave to run reference kernels, against libref alone,
 * for every family: the whole-model form of rad-avx-check, through the selection a CPU-only server
 * makes. An op libavx lacks leaves a hole and the server does not start. The build makes libavx on
 * every x86-64 and on no other machine. */
TEST(libavx_alone_serves_every_family_within_rounding_of_libref) {
#if defined(__x86_64__)
    NEED_ARTIFACT(RAD_E2E_HOME "/kernels/libavx.so");
#else
    SKIP("libavx is x86-64 only");
#endif
    REQUIRE(world().ok);
    World& w = world();
    const struct { const char* name; const std::string* model; const std::string* corpus; }
        kRuns[] = { { "qwen35 on libavx", &w.qwen35, &w.corpus },
                    { "lfm2 on libavx", &w.lfm2, &w.corpus_short },
                    { "gemma4 on libavx", &w.gemma4, &w.corpus_short },
                    { "llama_fp8 on libavx", &w.llama_rad, &w.corpus_tiny },
                    { "qwen3asr's decoder on libavx", &w.qwen3asr, &w.corpus_short } };
    for (const auto& run : kRuns)
        check_within_rounding(run.name, score_against_reference(
            *run.model, *run.corpus,
            { "--kernels", "libavx", "--max-num-batched-tokens", "512",
              "--max-num-seqs", "1" }));
}

/* A 16 kHz mono 16-bit WAV of `n` samples: two tones, so every mel band the tower reads carries
 * something. */
std::string wav16(int64_t n) {
    std::string d;
    for (int64_t i = 0; i < n; ++i) {
        const double t = (double)i / 16000.0;
        const double v = 0.4 * std::sin(2.0 * M_PI * 440.0 * t) + 0.2 * std::sin(2.0 * M_PI * 2500.0 * t);
        const int16_t x = (int16_t)std::lround(v * 32767.0);
        d.push_back((char)(x & 0xff));
        d.push_back((char)((x >> 8) & 0xff));
    }
    auto u32 = [](uint32_t v) {
        std::string b;
        for (int i = 0; i < 4; ++i) b.push_back((char)((v >> (8 * i)) & 0xff));
        return b;
    };
    auto u16 = [](uint16_t v) { return std::string{ (char)(v & 0xff), (char)(v >> 8) }; };
    return "RIFF" + u32((uint32_t)(36 + d.size())) + "WAVEfmt " + u32(16) + u16(1) + u16(1) +
           u32(16000) + u32(32000) + u16(2) + u16(16) + "data" + u32((uint32_t)d.size()) + d;
}

/* POST /v1/audio/transcriptions as OpenAI's client sends it: multipart/form-data. */
json transcribe(int port, const std::string& wav, int* status) {
    const std::string b = "e2eBoundary";
    std::string body = "--" + b + "\r\nContent-Disposition: form-data; name=\"file\"; "
                       "filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n" + wav + "\r\n";
    for (const auto& [k, v] : { std::pair<const char*, const char*>{ "language", "en" },
                                { "response_format", "verbose_json" } })
        body += "--" + b + "\r\nContent-Disposition: form-data; name=\"" + k + "\"\r\n\r\n" + v +
                "\r\n";
    body += "--" + b + "--\r\n";
    Call c;
    c.fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    *status = -1;
    if (c.fd < 0 || connect(c.fd, (sockaddr*)&a, sizeof a) != 0) return json();
    const std::string req = "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                            "Connection: close\r\nContent-Type: multipart/form-data; boundary=" +
                            b + "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" +
                            body;
    for (size_t off = 0; off < req.size();) {
        const ssize_t n = ::send(c.fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
        if (n <= 0) return json();
        off += (size_t)n;
    }
    std::string out;
    *status = c.read(&out);
    return json::parse(out, nullptr, false);
}

/* The tiny Qwen3-ASR tower computed apart from both kernel libraries, in double, from its
 * checkpoint and the features the engine staged (`pix`, [frames, mels]): each second convolved on
 * its own (a whole one staged as 104 frames, its last four alignment), the 1 x f3 projection,
 * positions restarting every second, pre-norm layers attending in blocks of 104 rows, the final
 * norm and the projector. The [rows, hidden] rows the decoder's placeholders take. */
std::vector<double> qwen3asr_tower(const std::string& dir, const tiny::Qwen3Asr& s,
                                   const std::vector<float>& pix) {
    std::map<std::string, std::vector<double>> W;
    for (const tiny::Tensor& t : tiny::read_safetensors(dir + "/model.safetensors")) {
        std::vector<double>& v = W[t.name];
        v.resize(t.bytes.size() / 2);
        for (size_t i = 0; i < v.size(); ++i) {
            uint16_t u;
            std::memcpy(&u, t.bytes.data() + 2 * i, 2);
            v[i] = rad_bf16_to_f32(u);
        }
    }
    auto w = [&](const std::string& n) -> const std::vector<double>& { return W.at(n); };
    auto gelu = [](double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); };
    const int64_t C = s.a_channels, A = s.a_dim, nm = s.mels, H = s.hidden;
    /* x [ci][mel][t] -> [co][mel'][t'], 3x3, stride 2, pad 1, gelu. */
    auto conv = [&](const std::vector<double>& x, int64_t ci, int64_t hm, int64_t wt,
                    const std::string& name, int64_t* ho, int64_t* wo) {
        *ho = (hm - 1) / 2 + 1;
        *wo = (wt - 1) / 2 + 1;
        const std::vector<double>& k = w(name + ".weight");
        const std::vector<double>& b = w(name + ".bias");
        std::vector<double> y((size_t)(C * *ho * *wo));
        for (int64_t co = 0; co < C; ++co)
            for (int64_t i = 0; i < *ho; ++i)
                for (int64_t j = 0; j < *wo; ++j) {
                    double acc = b[(size_t)co];
                    for (int64_t c = 0; c < ci; ++c)
                        for (int64_t di = 0; di < 3; ++di)
                            for (int64_t dj = 0; dj < 3; ++dj) {
                                const int64_t r = 2 * i - 1 + di, q = 2 * j - 1 + dj;
                                if (r < 0 || r >= hm || q < 0 || q >= wt) continue;
                                acc += k[(size_t)(((co * ci + c) * 3 + di) * 3 + dj)] *
                                       x[(size_t)((c * hm + r) * wt + q)];
                            }
                    y[(size_t)((co * *ho + i) * *wo + j)] = gelu(acc);
                }
        return y;
    };
    auto rows_of = [](int64_t L) {
        for (int i = 0; i < 3; ++i) L = L > 0 ? (L - 1) / 2 + 1 : 0;
        return L;
    };
    const int64_t frames = (int64_t)pix.size() / nm;
    std::vector<double> x;
    const std::string T = "model.audio_tower.";
    for (int64_t at = 0; at < frames;) {
        const int64_t L = frames - at >= 104 ? 100 : frames - at;
        std::vector<double> in((size_t)(nm * 100), 0.0);
        for (int64_t t = 0; t < L; ++t)
            for (int64_t m = 0; m < nm; ++m) in[(size_t)(m * 100 + t)] = pix[(size_t)((at + t) * nm + m)];
        at += L == 100 ? 104 : L;
        int64_t h1, w1, h2, w2, h3, w3;
        const std::vector<double> y1 = conv(in, 1, nm, 100, T + "conv2d1", &h1, &w1);
        const std::vector<double> y2 = conv(y1, C, h1, w1, T + "conv2d2", &h2, &w2);
        const std::vector<double> y3 = conv(y2, C, h2, w2, T + "conv2d3", &h3, &w3);
        const int64_t keep = L == 100 ? w3 : rows_of(L);
        const std::vector<double>& wo = w(T + "conv_out.weight");
        for (int64_t r = 0; r < keep; ++r)
            for (int64_t o = 0; o < A; ++o) {
                double acc = 0.0;
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t f = 0; f < h3; ++f)
                        acc += wo[(size_t)(o * C * h3 + c * h3 + f)] * y3[(size_t)((c * h3 + f) * w3 + r)];
                const int64_t i = o % (A / 2);
                const double inv = std::exp(-(double)i * std::log(10000.0) / (double)(A / 2 - 1));
                acc += o < A / 2 ? std::sin((double)r * inv) : std::cos((double)r * inv);
                x.push_back(acc);
            }
    }
    const int64_t R = (int64_t)x.size() / A;
    auto norm = [&](const std::vector<double>& v, int64_t n, const std::string& name) {
        std::vector<double> o(v.size());
        const std::vector<double>& g = w(name + ".weight");
        const std::vector<double>& b = w(name + ".bias");
        for (size_t r = 0; r < v.size() / (size_t)n; ++r) {
            double mu = 0, var = 0;
            for (int64_t i = 0; i < n; ++i) mu += v[r * n + i];
            mu /= (double)n;
            for (int64_t i = 0; i < n; ++i) var += (v[r * n + i] - mu) * (v[r * n + i] - mu);
            var /= (double)n;
            for (int64_t i = 0; i < n; ++i)
                o[r * n + i] = (v[r * n + i] - mu) / std::sqrt(var + 1e-5) * g[(size_t)i] + b[(size_t)i];
        }
        return o;
    };
    auto linear = [&](const std::vector<double>& v, int64_t k, int64_t n, const std::string& name,
                      bool bias, bool act) {
        const std::vector<double>& m = w(name + ".weight");
        std::vector<double> o((v.size() / (size_t)k) * (size_t)n);
        for (size_t r = 0; r < v.size() / (size_t)k; ++r)
            for (int64_t j = 0; j < n; ++j) {
                double acc = bias ? w(name + ".bias")[(size_t)j] : 0.0;
                for (int64_t i = 0; i < k; ++i) acc += m[(size_t)(j * k + i)] * v[r * k + i];
                o[r * n + j] = act ? gelu(acc) : acc;
            }
        return o;
    };
    const int64_t nh = s.a_heads, hd = A / nh;
    for (int64_t l = 0; l < s.a_layers; ++l) {
        const std::string p = T + "layers." + std::to_string(l) + ".";
        const std::vector<double> h = norm(x, A, p + "self_attn_layer_norm");
        const std::vector<double> q = linear(h, A, A, p + "self_attn.q_proj", true, false);
        const std::vector<double> k = linear(h, A, A, p + "self_attn.k_proj", true, false);
        const std::vector<double> v = linear(h, A, A, p + "self_attn.v_proj", true, false);
        std::vector<double> att((size_t)(R * A), 0.0);
        for (int64_t hh = 0; hh < nh; ++hh)
            for (int64_t i = 0; i < R; ++i) {
                const int64_t b0 = i / 104 * 104, b1 = std::min<int64_t>(R, b0 + 104);
                std::vector<double> sc((size_t)(b1 - b0));
                double mx = -1e300;
                for (int64_t j = b0; j < b1; ++j) {
                    double d = 0;
                    for (int64_t e = 0; e < hd; ++e)
                        d += q[(size_t)(i * A + hh * hd + e)] * k[(size_t)(j * A + hh * hd + e)];
                    sc[(size_t)(j - b0)] = d / std::sqrt((double)hd);
                    mx = std::max(mx, sc[(size_t)(j - b0)]);
                }
                double sum = 0;
                for (double& e : sc) sum += (e = std::exp(e - mx));
                for (int64_t j = b0; j < b1; ++j)
                    for (int64_t e = 0; e < hd; ++e)
                        att[(size_t)(i * A + hh * hd + e)] +=
                            sc[(size_t)(j - b0)] / sum * v[(size_t)(j * A + hh * hd + e)];
            }
        const std::vector<double> o = linear(att, A, A, p + "self_attn.out_proj", true, false);
        for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
        const std::vector<double> h2 = norm(x, A, p + "final_layer_norm");
        const std::vector<double> f = linear(h2, A, s.a_ffn, p + "fc1", true, true);
        const std::vector<double> o2 = linear(f, s.a_ffn, A, p + "fc2", true, false);
        for (size_t i = 0; i < x.size(); ++i) x[i] += o2[i];
    }
    const std::vector<double> hn = norm(x, A, T + "ln_post");
    const std::vector<double> p1 = linear(hn, A, A, "model.multi_modal_projector.linear_1", true, true);
    return linear(p1, A, H, "model.multi_modal_projector.linear_2", true, false);
}

/* Rel L2 of bf16 rows against an answer in double. */
double rel_to(const std::string& rows, const std::vector<double>& want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size() && 2 * i + 1 < rows.size(); ++i) {
        uint16_t u;
        std::memcpy(&u, rows.data() + 2 * i, 2);
        const double d = (double)rad_bf16_to_f32(u) - want[i];
        num += d * d;
        den += want[i] * want[i];
    }
    return rows.size() == 2 * want.size() && den > 0.0 ? std::sqrt(num / den) : 1.0;
}

/* Qwen3-ASR from its checkpoint on libref and on libavx alone, as a CPU-only server runs it: a
 * transcription comes back, both stage the same features, and the encoder's rows
 * (RADIANCE_DEBUG_MM_DUMP) are the tower computed apart in double, within bf16 rounding. A
 * comparison of the two libraries alone passes when both compute the same wrong thing. The clip
 * is 10.45 s: ten whole seconds and a short one, so attention runs in two 104-row blocks and the
 * last second is staged short. The decoder is scored against libref in the case above. */
TEST(qwen3asr_transcribes_on_libavx_alone_as_on_libref) {
#if defined(__x86_64__)
    NEED_ARTIFACT(RAD_E2E_HOME "/kernels/libavx.so");
#else
    SKIP("libavx is x86-64 only");
#endif
    if (!rad::mm::decoder_available()) SKIP("no media decoder in this build");
    REQUIRE(world().ok);
    const std::string wav = wav16(167200);
    std::map<std::string, std::string> pix[2], rows[2];
    for (int k = 0; k < 2; ++k) {
        tiny::TempDir dump("e2e_asr");
        Server srv;
        srv.serving = k == 1;
        REQUIRE(srv.start(world().qwen3asr, { "--kernels", k ? "libavx" : "libref",
                                              "--max-num-seqs", "2" },
                          { "RADIANCE_DEBUG_MM_DUMP=" + dump.path }));
        int status = 0;
        const json j = transcribe(srv.port, wav, &status);
        CHECK_EQ(status, 200);
        if (status != 200) fprintf(stderr, "    %s\n", tail(srv.output()).c_str());
        REQUIRE(j.is_object());
        CHECK(j["text"].is_string());
        CHECK_EQ(j.value("language", ""), std::string("english"));
        CHECK(j["usage"]["input_tokens"].get<int64_t>() > 0);
        CHECK(j["usage"]["output_tokens"].get<int64_t>() > 0);
        CHECK_EQ(srv.stop(), 0);
        for (const auto& e : std::filesystem::directory_iterator(dump.path)) {
            const std::string name = e.path().filename().string();
            const std::string bytes = tiny::read_text(e.path().string());
            if (name.size() > 9 && name.compare(name.size() - 9, 9, ".pix.bf16") == 0)
                pix[k][name] = bytes;
            else if (name.size() > 5 && name.compare(name.size() - 5, 5, ".bf16") == 0)
                rows[k][name] = bytes;
        }
    }
    REQUIRE_EQ(pix[0].size(), (size_t)1);
    CHECK(pix[0] == pix[1]);
    REQUIRE_EQ(rows[0].size(), (size_t)1);
    REQUIRE_EQ(rows[1].size(), (size_t)1);
    const std::string& staged = pix[0].begin()->second;
    std::vector<float> frames(staged.size() / 2);
    for (size_t i = 0; i < frames.size(); ++i) {
        uint16_t u;
        std::memcpy(&u, staged.data() + 2 * i, 2);
        frames[i] = rad_bf16_to_f32(u);
    }
    const tiny::Qwen3Asr shape;
    const std::vector<double> want = qwen3asr_tower(world().qwen3asr, shape, frames);
    CHECK_EQ(want.size() / (size_t)shape.hidden, (size_t)136);
    for (int k = 0; k < 2; ++k) {
        const double rel = rel_to(rows[k].begin()->second, want);
        fprintf(stderr, "    encoder rows on %s against the tower computed apart: rel_l2 %.3g\n",
                k ? "libavx" : "libref", rel);
        CHECK(rel < 2e-2);
    }
}

/* A container whose lm head is block fp8 or int8, served by libavx alone (no libref loaded) and
 * scored against libref alone: both join the head's codes and scale on the host (abi/rad_planes.h)
 * and read it through logits_gemm's joined form, libref as the core's decoder gives it. */
TEST(libavx_alone_serves_a_quantised_head_within_rounding_of_libref) {
#if defined(__x86_64__)
    NEED_ARTIFACT(RAD_E2E_HOME "/kernels/libavx.so");
#else
    SKIP("libavx is x86-64 only");
#endif
    REQUIRE(world().ok);
    World& w = world();
    const struct { const char* name; const std::string* model; } kRuns[] = {
        { "llama, fp8 head, on libavx alone", &w.llama_fp8head },
        { "llama, int8 head, on libavx alone", &w.llama_i8head } };
    for (const auto& run : kRuns)
        check_within_rounding(run.name, score_against_reference(
            *run.model, w.corpus_tiny,
            { "--kernels", "libavx", "--max-num-batched-tokens", "512", "--max-num-seqs", "1" }));
}

/* An int4 W4A16 release with zero points, imported, runs its linears on the 4-bit GEMMs: at or
 * below 64 rows a step the f16 one over its cast input, above that the int8 one over
 * quant_act_i8g's codes. libavx alone agrees with libref within rounding on each: the prefill in
 * 512-row steps, and in 64-row steps against libref's own run in 64-row steps. */
TEST(libavx_alone_serves_an_int4_release_within_rounding_of_libref) {
#if defined(__x86_64__)
    NEED_ARTIFACT(RAD_E2E_HOME "/kernels/libavx.so");
#else
    SKIP("libavx is x86-64 only");
#endif
    REQUIRE(world().ok);
    World& w = world();
    check_within_rounding("int4 llama on libavx alone, steps of 512",
                          score_against_reference(w.llama_int4_rad, w.corpus_tiny,
                                                  { "--kernels", "libavx",
                                                    "--max-num-batched-tokens", "512",
                                                    "--max-num-seqs", "1" }));
    tiny::TempDir d("e2e_int4");
    const std::vector<std::string> step64 = { "--max-num-batched-tokens", "64",
                                              "--max-num-seqs", "1" };
    std::vector<std::string> rec = { "--kld-record", d / "ref64", "--kld-corpus", w.corpus_tiny };
    rec.insert(rec.end(), step64.begin(), step64.end());
    const Ran r_ref = radiance(w.llama_int4_rad, rec);
    REQUIRE_RAN(r_ref);
    std::vector<std::string> sc = { "--kernels", "libavx", "--kld-ref", d / "ref64",
                                    "--kld-out", d / "kld.json" };
    sc.insert(sc.end(), step64.begin(), step64.end());
    const Ran r_avx = radiance(w.llama_int4_rad, sc);
    REQUIRE_RAN(r_avx);
    check_within_rounding("int4 llama on libavx alone, steps of 64",
                          json::parse(tiny::read_text(d / "kld.json"), nullptr, false));
}

/* ================================================================== HTTP */

namespace {

/* A server decoding one token a step, and one verifying up to four prompt-lookup drafts a step;
 * neither caches prefixes, so every request runs cold. */
const std::vector<std::string> kPlain = { "--num-speculative-tokens", "0", "--no-prefix-cache" };
const std::vector<std::string> kSpec = { "--num-speculative-tokens", "4", "--no-prefix-cache" };

int64_t timing(const json& reply, const char* field) {
    return reply.value(json::json_pointer(std::string("/timings/") + field), (int64_t)-1);
}

struct Family {
    const char* name;
    std::string World::*model;
};
const Family kSpecFamilies[] = { { "lfm2", &World::lfm2 }, { "gemma4", &World::gemma4 },
                                 { "qwen35", &World::qwen35_spec } };

const std::vector<int32_t>& repetitive_prompt() {
    static const std::vector<int32_t> p =
        tiny::byte_ids("one two three four one two three four one two three four one two");
    return p;
}

}  /* namespace */

/* Prompt lookup drafts from n-grams the sequence already holds, and the model's own continuation
 * rejects some of them. Verifying a window must give exactly the tokens one-at-a-time greedy
 * decoding gives, and exactly as many: the conv window (lfm2), the sliding window (gemma4) and
 * the recurrent state (qwen35) each roll back what was rejected. */
TEST(speculation_by_prompt_lookup_returns_the_greedy_tokens) {
    REQUIRE(world().ok);
    for (const Family& f : kSpecFamilies) {
        Server plain, spec;
        REQUIRE(plain.start(world().*f.model, kPlain));
        REQUIRE(spec.start(world().*f.model, kSpec));
        int status = 0;
        const json a = post(plain.port, "/completion", completion(repetitive_prompt(), 48));
        const json b = post(spec.port, "/completion", completion(repetitive_prompt(), 48), &status);
        CHECK_EQ(status, 200);
        const std::vector<int32_t> ta = tokens_of(a), tb = tokens_of(b);
        CHECK_EQ(ta.size(), (size_t)48);
        if (ta != tb)
            fprintf(stderr, "    %s plain %s\n    %s spec  %s\n", f.name, show(ta).c_str(), f.name,
                    show(tb).c_str());
        CHECK(ta == tb);
        /* The check means something only when drafts were made and some were wrong. */
        const int64_t drafted = timing(b, "draft_n");
        const int64_t accepted = timing(b, "draft_n_accepted");
        if (!(drafted > accepted && accepted > 0))
            fprintf(stderr, "    %s drafted %lld, accepted %lld\n", f.name, (long long)drafted,
                    (long long)accepted);
        CHECK(drafted > accepted);
        CHECK(accepted > 0);
        CHECK_EQ(plain.stop(), 0);
        CHECK_EQ(spec.stop(), 0);
    }
}

/* Sampling draws each position's random number from (seed, position), so a seeded request repeats
 * exactly, and speculation, which accepts a draft only where the sampler at that coordinate would
 * have drawn it, gives the same tokens as plain decoding. Another seed must give other tokens, or
 * the check is vacuous. */
TEST(seeded_sampling_repeats_and_is_unchanged_by_speculation) {
    REQUIRE(world().ok);
    for (const Family& f : kSpecFamilies) {
        Server plain, spec;
        REQUIRE(plain.start(world().*f.model, kPlain));
        REQUIRE(spec.start(world().*f.model, kSpec));
        const json req = completion(repetitive_prompt(), 64, 0.5, 1234);
        const std::vector<int32_t> a = tokens_of(post(plain.port, "/completion", req));
        const std::vector<int32_t> a2 = tokens_of(post(plain.port, "/completion", req));
        const std::vector<int32_t> other =
            tokens_of(post(plain.port, "/completion",
                           completion(repetitive_prompt(), 64, 0.5, 99)));
        const std::vector<int32_t> greedy =
            tokens_of(post(plain.port, "/completion", completion(repetitive_prompt(), 64)));
        int status = 0;
        const json sb = post(spec.port, "/completion", req, &status);
        CHECK_EQ(status, 200);
        const std::vector<int32_t> b = tokens_of(sb);
        CHECK_EQ(a.size(), (size_t)64);
        CHECK(a == a2);
        if (a != b) fprintf(stderr, "    %s plain %s\n    %s spec  %s\n", f.name, show(a).c_str(),
                            f.name, show(b).c_str());
        CHECK(a == b);
        CHECK(a != other);
        CHECK(a != greedy);
        const int64_t drafted = timing(sb, "draft_n");
        if (drafted <= 0) fprintf(stderr, "    %s: nothing drafted\n", f.name);
        CHECK(drafted > 0);
        CHECK_EQ(plain.stop(), 0);
        CHECK_EQ(spec.stop(), 0);
    }
}

/* Four distinct prompts in flight at once, each against its reply alone on a fresh server: every
 * step batches them in other slots and other rows, and with speculation each row's window differs
 * as well. */
TEST(concurrent_requests_return_what_each_returns_alone) {
    REQUIRE(world().ok);
    std::vector<std::vector<int32_t>> prompts;
    for (uint64_t i = 0; i < 4; ++i) prompts.push_back(prompt_ids(100 + i, 60 + 37 * i));
    const struct { const char* name; std::string World::*model; const char* n_spec; } kRuns[] = {
        { "qwen35", &World::qwen35, "4" }, { "lfm2", &World::lfm2, "4" },
        { "gemma4", &World::gemma4, "4" } };
    for (const auto& run : kRuns) {
        std::vector<std::vector<int32_t>> alone;
        {
            Server s;
            REQUIRE(s.start(world().*run.model, kPlain));
            for (const auto& p : prompts)
                alone.push_back(tokens_of(post(s.port, "/completion", completion(p, 40))));
            CHECK_EQ(s.stop(), 0);
        }
        Server s;
        REQUIRE(s.start(world().*run.model, { "--num-speculative-tokens", run.n_spec,
                                              "--no-prefix-cache", "--max-num-seqs", "4" }));
        Call calls[4];
        for (int i = 0; i < 4; ++i)
            REQUIRE(calls[i].send(s.port, "POST", "/completion",
                                  completion(prompts[i], 40).dump()));
        for (int i = 0; i < 4; ++i) {
            std::string body;
            CHECK_EQ(calls[i].read(&body), 200);
            const std::vector<int32_t> t = tokens_of(json::parse(body, nullptr, false));
            CHECK_EQ(t.size(), (size_t)40);
            if (t != alone[(size_t)i])
                fprintf(stderr, "    %s n_spec %s prompt %d\n    alone    %s\n    together %s\n",
                        run.name, run.n_spec, i, show(alone[(size_t)i]).c_str(), show(t).c_str());
            CHECK(t == alone[(size_t)i]);
        }
        CHECK_EQ(s.stop(), 0);
    }
}

/* A prompt that shares a head with an earlier one resumes from the prefix cache: attention from
 * the cached blocks, the gated delta net from the linear-state checkpoint the earlier request wrote
 * at the end of the shared head, never from that request's later state. The reply, and every
 * probability in it, must equal a cold run's; so must a repeat of the prompt itself. */
TEST(a_prompt_resumed_from_the_prefix_cache_returns_what_a_cold_run_does) {
    REQUIRE(world().ok);
    const std::vector<int32_t> head = prompt_ids(7, 200);
    std::vector<int32_t> a = head, b = head;
    for (int32_t t : prompt_ids(8, 70)) a.push_back(t);
    for (int32_t t : prompt_ids(9, 90)) b.push_back(t);
    json cold;
    {
        Server s;
        REQUIRE(s.start(world().qwen35, kPlain));
        cold = post(s.port, "/completion", completion(b, 24, 0.0, 0, 3));
        CHECK_EQ(s.stop(), 0);
    }
    Server s;
    REQUIRE(s.start(world().qwen35,
                    { "--num-speculative-tokens", "0", "--checkpoint-interval", "64" }));
    (void)post(s.port, "/completion", completion(a, 24, 0.0, 0, 3));
    const json shared = post(s.port, "/completion", completion(b, 24, 0.0, 0, 3));
    const json again = post(s.port, "/completion", completion(b, 24, 0.0, 0, 3));
    const int64_t c1 = timing(shared, "cache_n"), c2 = timing(again, "cache_n");
    /* The head ends at 200, so the shared hit ends at a checkpoint (every 64 tokens) at or before
     * it; the repeat reaches further into the prompt. */
    if (!(c1 > 0 && c1 <= 200 && c2 > c1))
        fprintf(stderr, "    cached: shared head %lld, repeat %lld\n", (long long)c1,
                (long long)c2);
    CHECK(c1 > 0);
    CHECK(c1 <= 200);
    CHECK(c2 > c1);
    REQUIRE(cold.contains("completion_probabilities"));
    CHECK_EQ(tokens_of(cold).size(), (size_t)24);
    for (const json* r : { &shared, &again }) {
        CHECK(tokens_of(*r) == tokens_of(cold));
        CHECK((*r)["completion_probabilities"] == cold["completion_probabilities"]);
    }
    CHECK_EQ(s.stop(), 0);
}

/* The row a running server's /diagnostics reports for `option`, or null. */
json config_row(int port, const std::string& option) {
    Call c;
    std::string out;
    if (!c.send(port, "GET", "/diagnostics", "") || c.read(&out) != 200) return json();
    const json d = json::parse(out, nullptr, false);
    if (!d.is_object() || !d.contains("configuration")) return json();
    for (const json& r : d["configuration"])
        if (r.value("option", "") == option) return r;
    return json();
}

/* --checkpoint-slots auto, the default, sizes the snapshot store from --max-num-seqs and what the
 * retention policy keeps for a --max-model-len session (the tiny hybrid's 4096 tokens are two
 * 2048-token intervals: 2 snapshots, 3 a sequence), says so on its startup line and reports the
 * count it chose with the reason. A count given on the command line is taken as it stands; a model
 * with nothing to snapshot, or a server without a prefix cache, resolves to none. */
TEST(the_snapshot_store_is_sized_by_auto_unless_a_count_is_given) {
    REQUIRE(world().ok);
    const struct Case {
        std::string World::*model;
        std::vector<std::string> args;
        const char* value;
        const char* source;
        const char* why;
        const char* line;
    } kCases[] = {
        { &World::qwen35, {}, "768", "auto", "3 a sequence for --max-num-seqs 256",
          "checkpoint slots: auto: 768 linear-state snapshot(s) of " },
        { &World::qwen35, { "--max-num-seqs", "4" }, "12", "auto",
          "geometric-backoff keeps 2 for a 4096-token session at a 2048-token interval",
          "checkpoint slots: auto: 12 linear-state snapshot(s) of " },
        { &World::qwen35, { "--max-num-seqs", "4", "--checkpoint-slots", "auto" }, "12",
          "command line", "3 a sequence", "checkpoint slots: auto: 12 " },
        { &World::qwen35,
          { "--max-num-seqs", "4", "--checkpoint-policy", "keep-all", "--checkpoint-interval", "256" },
          "68", "auto", "keep-all keeps 16 for a 4096-token session at a 256-token interval",
          "checkpoint slots: auto: 68 " },
        { &World::qwen35, { "--max-num-seqs", "4", "--checkpoint-slots", "5" }, "5", "command line",
          "", "checkpoint slots: 5 linear-state snapshot(s) of " },
        { &World::qwen35, { "--max-num-seqs", "4", "--no-prefix-cache" }, "0", "auto",
          "prefix caching is off", "" },
        { &World::llama, { "--max-num-seqs", "4" }, "0", "auto",
          "the model keeps no linear or sliding-window state", "" },
        { &World::lfm2, { "--max-num-seqs", "4" }, "12", "auto", "3 a sequence for --max-num-seqs 4",
          "checkpoint slots: auto: 12 linear-state snapshot(s) of 512 B " },
    };
    for (const Case& k : kCases) {
        Server s;
        REQUIRE(s.start(world().*k.model, k.args));
        const json row = config_row(s.port, "--checkpoint-slots");
        const std::string log = s.output();
        std::string args;
        for (const std::string& a : k.args) args += " " + a;
        const bool ok = row.is_object() && row.value("value", "") == k.value &&
                        row.value("source", "") == k.source &&
                        row.value("why", "").find(k.why) != std::string::npos &&
                        has(log, k.line);
        if (!ok)
            fprintf(stderr, "    %s:%s\n    row %s\n%s\n", (world().*k.model).c_str(), args.c_str(),
                    row.dump().c_str(), tail(log, 12).c_str());
        CHECK(ok);
        CHECK_EQ(s.stop(), 0);
    }
}

/* A snapshot store that cannot be had stops the start, naming --checkpoint-slots, the size and
 * what to change; served without it, every prefix hit on a hybrid model would prefill from the
 * start. The largest count the flag takes is more bytes than a 64-bit size says, whatever a slot's
 * size, so this needs no host short of memory. */
TEST(a_snapshot_store_that_cannot_be_had_stops_the_start) {
    REQUIRE(world().ok);
    const std::string n = "9223372036854775807";
    const Ran r = radiance(world().qwen35, { "--host", "127.0.0.1", "--port", "0",
                                             "--checkpoint-slots", n }, {}, 60);
    const std::string want = "checkpoint slots: " + n + " linear-state snapshot(s) of ";
    if (r.status == 0 || r.status == 124 || !has(r.log, want))
        fprintf(stderr, "    exited %d:\n%s\n", r.status, tail(r.log, 20).c_str());
    CHECK(r.status != 0);
    CHECK(r.status != 124);
    CHECK(has(r.log, want));
    CHECK(has(r.log, "more than 8 EiB of pinned host memory, which no host has"));
    CHECK(has(r.log, "pass --checkpoint-slots 0"));
}

/* Without a drafter the scheduler issues each step's forward pass behind the previous one, before
 * that one commits (Scheduler::step_next), so the decode in the cases above ran through it. This
 * pins that it did, and that prompt lookup, which reads the committed sequence, turns it off. */
TEST(decoding_without_a_drafter_issues_steps_ahead) {
    REQUIRE(world().ok);
    Server plain, spec;
    REQUIRE(plain.start(world().lfm2, kPlain));
    REQUIRE(spec.start(world().lfm2, kSpec));
    const std::vector<int32_t> p = prompt_ids(5, 80);
    CHECK(tokens_of(post(plain.port, "/completion", completion(p, 32))) ==
          tokens_of(post(spec.port, "/completion", completion(p, 32))));
    const double ahead = metric(plain.port, "radiance:engine_steps_ahead_total");
    if (!(ahead > 0)) fprintf(stderr, "    steps ahead %g\n", ahead);
    CHECK(ahead > 0);
    CHECK_EQ(metric(spec.port, "radiance:engine_steps_ahead_total"), 0.0);
    CHECK_EQ(plain.stop(), 0);
    CHECK_EQ(spec.stop(), 0);
}

/* --draft-profile-out end to end: a cost serving run with a serial assistant head measures the
 * draft cycles one real request runs -- the start cycle included -- exports them on the orderly
 * SIGTERM shutdown, and a fresh start of the same deployment -- which derives its own runtime
 * identity -- binds the file and prices its own request from it. Nothing here writes a bucket by
 * hand: the reader, the writer and the recording ring have their own fixtures (engine_profile_test,
 * sched_draft_test); this case proves that the numbers in the file came out of this run, and that a
 * reloaded deployment decides with them.
 *
 * The calibration runs the Cost policy with a record-only table: it issues the named fallback at
 * its ceiling (2), because it has no prices to decide with, and that is exactly the baseline it
 * measures. The ladder is cut to 0,2 for it: a tier above the ceiling is refused by the planner, and
 * the default ladder ends at 6. This is the only way to file the start cycle: under Cost a cycle
 * prices the verify window the previous plan published, and a request's first cycle has nothing to
 * verify, so its shape is keyed verify_k = 0 -- a cycle legacy drafting, which verifies what it
 * drafts in the same step, never executes.
 *
 * The reload keeps the default tier ladder and a ceiling of 6. Its first cycle finds the measured
 * start bucket (verify_k = 0 at draft_k = 2) and takes it; every later cycle verifies the 2-token
 * window it published, which the calibration also measured. Nothing can name 2 but the loaded
 * profile -- the fallback would draft 6 -- so the executed window is the proof. The export lists one
 * position per drafted slot, so the priced depth 2 is two of them and the fallback would be six. */
TEST(a_serving_run_exports_the_cycles_a_request_measured_and_a_second_run_binds_them) {
    /* A calibration run needs its own checkpoint (the assistant head lives in draft/), and the
     * shared World has none with a drafter. */
    tiny::TempDir dir("profile_out");
    REQUIRE(dir.ok());
    tiny::Gemma4 s;
    s.assistant = true;
    REQUIRE(tiny::write_gemma4(dir.path, s));
    const std::string out = dir / "profile.json";

    Server calib;
    {
        const std::vector<std::string> a = { "--num-speculative-tokens", "2",
                                             "--draft-depth-tiers", "0,2",
                                             "--draft-window", "fixed",
                                             "--draft-policy", "cost",
                                             "--draft-profile-out", out };
        REQUIRE(calib.start(dir.path, a));
    }
    const std::vector<int32_t> prompt = prompt_ids(9, 64);
    int status = 0;
    std::string why;
    const json reply = post(calib.port, "/completion", completion(prompt, 24), &status, &why);
    if (status != 200)
        fprintf(stderr, "    the request failed: %s\n%s\n", why.c_str(),
                tail(calib.output()).c_str());
    REQUIRE_EQ(status, 200);
    REQUIRE_EQ(tokens_of(reply).size(), (size_t)24);

    /* SIGTERM is the orderly shutdown: the request drains, serve() returns RAD_OK and the export
     * runs after the step threads joined. A run that cannot price a bucket from its own cycles
     * fails the export here, with the reason in the log. */
    const int calib_exit = calib.stop();
    const std::string calib_log = calib.output();
    if (calib_exit != 0)
        fprintf(stderr, "    the calibration run exited %d:\n%s\n", calib_exit,
                tail(calib_log).c_str());
    REQUIRE_EQ(calib_exit, 0);
    CHECK(calib_log.find("bucket(s), fingerprint") != std::string::npos);

    const json profile = json::parse(tiny::read_text(out), nullptr, false);
    REQUIRE(profile.is_object());
    const json buckets = profile.value("buckets", json::array());
    REQUIRE(buckets.is_array());
    REQUIRE(!buckets.empty());
    /* One bucket the table may price from: no disturbing flag (ahead is not one) and a positive
     * mean over at least one measurement -- the rule the export and estimate() share. */
    bool usable = false;
    for (const json& b : buckets) {
        bool clean = true;
        for (const json& f : b.value("flags", json::array())) clean = clean && f == "ahead";
        const json t = b.value("time_ms", json::object());
        usable = usable || (clean && t.value("mean", 0.0) > 0.0 && t.value("n", 0) >= 1);
    }
    CHECK(usable);

    /* The reload. A fresh process derives the identity from the same deployment and must bind the
     * file; the log names the match. */
    Server cost;
    {
        const std::vector<std::string> a = { "--num-speculative-tokens", "6",
                                             "--draft-window", "fixed",
                                             "--draft-policy", "cost",
                                             "--draft-profile", out };
        REQUIRE(cost.start(dir.path, a));
    }
    const std::string cost_log = cost.output();
    if (cost_log.find("buckets match this deployment") == std::string::npos)
        fprintf(stderr, "    the second run did not bind the profile:\n%s\n", tail(cost_log).c_str());
    CHECK(cost_log.find("buckets match this deployment") != std::string::npos);
    CHECK(cost_log.find("is not used for this deployment") == std::string::npos);

    /* The start cycle the reload's first plan has to price was measured: nothing preceded it, so
     * its shape is keyed verify_k = 0 with verify_tokens = 1 and history_tokens = 1, at the depth
     * the calibration drafted. A reload from a file without it cannot price its first cycle at
     * all. */
    bool first_cycle_known = false;
    for (const json& b : buckets) {
        const json shape = b.value("shape", json::object());
        if (shape.value("verify_k", -1) != 0) continue;
        first_cycle_known = true;
        CHECK_EQ(shape.value("verify_tokens", -1), 1);
        CHECK_EQ(shape.value("history_tokens", -1), 1);
        CHECK_EQ(shape.value("draft_k", -1), 2);
    }
    CHECK(first_cycle_known);

    /* The reload serves a real request and decides on it. Its first cycle finds the start bucket
     * (verify_k = 0 at draft_k = 2) and every later cycle verifies the window that plan published,
     * which the calibration measured too. Only the loaded profile can name a depth of 2: the
     * fallback would draft the ceiling, 6. The exported positions are the deepest window the run
     * drafted, so two of them (0, 1) is the priced depth and six would be the fallback. */
    const json served = post(cost.port, "/completion", completion(prompt, 24), &status, &why);
    if (status != 200)
        fprintf(stderr, "    the second run failed the request: %s\n%s\n", why.c_str(),
                tail(cost.output()).c_str());
    REQUIRE_EQ(status, 200);
    REQUIRE_EQ(tokens_of(served).size(), (size_t)24);
    const int positions = draft_positions(cost.port);
    if (positions != 2)
        fprintf(stderr, "    the reloaded run drafted %d positions, not the priced depth 2 "
                        "(6 positions = the fallback ceiling):\n%s\n",
                positions, tail(cost.output()).c_str());
    CHECK_EQ(positions, 2);
    CHECK_EQ(cost.stop(), 0);
}

/* Decoding a token runs the recurrent kernels over one row; prefilling it runs the chunked ones
 * over the whole prompt. Each decoded position's probabilities must be what a cold prefill of the
 * prompt and the tokens before it gives: exactly where both paths compute each row the same way
 * (attention, the short convolution, the fp8 linears), and within rounding for the gated delta
 * net, whose chunked scan is another algorithm than the recurrence and keeps its within-chunk
 * inverse in bf16 (up to about 0.03 in a log-probability on this model). A state never cleared,
 * written to another slot or read before its chunk boundary is off by far more. */
TEST(a_decoded_token_has_the_probabilities_prefilling_it_gives) {
    REQUIRE(world().ok);
    const struct { const char* name; std::string World::*model; double tol; } kRuns[] = {
        { "lfm2", &World::lfm2, 0.0 }, { "gemma4", &World::gemma4, 0.0 },
        { "llama_fp8", &World::llama_rad, 0.0 }, { "qwen35", &World::qwen35, 0.1 } };
    const std::vector<int32_t> p = prompt_ids(31, 120);
    for (const auto& run : kRuns) {
        Server s;
        REQUIRE(s.start(world().*run.model, kPlain));
        const json solo = post(s.port, "/completion", completion(p, 24, 0.0, 0, 2));
        const std::vector<int32_t> out = tokens_of(solo);
        REQUIRE_EQ(out.size(), (size_t)24);
        for (size_t k : { 1, 9, 23 }) {
            std::vector<int32_t> q = p;
            q.insert(q.end(), out.begin(), out.begin() + (long)k);
            const json cold = post(s.port, "/completion", completion(q, 1, 0.0, 0, 2));
            const json& a = solo["completion_probabilities"][k]["top_logprobs"];
            const json& b = cold["completion_probabilities"][0]["top_logprobs"];
            REQUIRE(a.size() == 2 && b.size() == 2);
            const double d =
                std::fabs(a[0]["logprob"].get<double>() - b[0]["logprob"].get<double>());
            const bool same_top = a[0]["id"] == b[0]["id"];
            const double gap = a[0]["logprob"].get<double>() - a[1]["logprob"].get<double>();
            if (!(d <= run.tol && (same_top || gap < 2 * run.tol)))
                fprintf(stderr, "    %s at %zu: decoded %s, prefilled %s\n", run.name, k,
                        a.dump().c_str(), b.dump().c_str());
            CHECK(d <= run.tol);
            CHECK(same_top || gap < 2 * run.tol);
        }
        CHECK_EQ(s.stop(), 0);
    }
}

/* A pool that holds the longest request but fewer tokens than four reach together preempts by
 * recompute: the victim is prefilled again from prompt ++ its output so far and continues. Its
 * stream must neither repeat nor skip a token, and no request may fail while the pool could hold
 * it alone. Gemma 4, because there the prefill that replays decoded tokens computes them exactly
 * as decoding did (see a_decoded_token_has_the_probabilities_prefilling_it_gives). */
TEST(preemption_by_recompute_continues_each_stream_exactly) {
    REQUIRE(world().ok);
    std::vector<std::vector<int32_t>> prompts;
    for (uint64_t i = 0; i < 4; ++i) prompts.push_back(prompt_ids(200 + i, 220 + 20 * i));
    std::vector<std::vector<int32_t>> alone;
    {
        Server s;
        REQUIRE(s.start(world().gemma4_kv, kPlain));
        for (const auto& p : prompts)
            alone.push_back(tokens_of(post(s.port, "/completion", completion(p, 48))));
        CHECK_EQ(s.stop(), 0);
    }
    std::vector<std::string> args = kPlain;
    for (const std::string& a : small_pool_args()) args.push_back(a);
    auto pool_at = [&](int mib) -> int64_t {
        Server probe;
        probe.quiet = true;
        if (!probe.start(world().gemma4_kv, args, device_env(mib))) return -1;
        return pool_tokens(probe.output());   /* killed on scope exit; it served nothing */
    };
    int64_t longest = 0, together = 0;
    for (const auto& p : prompts) {
        longest = std::max<int64_t>(longest, (int64_t)p.size() + 48);
        together += (int64_t)p.size() + 48;
    }
    int64_t pool = 0;
    std::string why;
    const int mib = small_device(pool_at, longest, together, true, &pool, &why);
    if (!mib) fprintf(stderr, "    no small pool: %s\n", why.c_str());
    REQUIRE(mib > 0);
    Server s;
    REQUIRE(s.start(world().gemma4_kv, args, device_env(mib)));
    Call calls[4];
    for (int i = 0; i < 4; ++i)
        REQUIRE(calls[i].send(s.port, "POST", "/completion", completion(prompts[i], 48).dump()));
    for (int i = 0; i < 4; ++i) {
        std::string body;
        CHECK_EQ(calls[i].read(&body), 200);
        const std::vector<int32_t> t = tokens_of(json::parse(body, nullptr, false));
        if (t != alone[(size_t)i])
            fprintf(stderr, "    prompt %d\n    alone    %s\n    together %s\n", i,
                    show(alone[(size_t)i]).c_str(), show(t).c_str());
        CHECK(t == alone[(size_t)i]);
    }
    const double n = metric(s.port, "vllm:num_preemptions_total");
    if (!(n > 0))
        fprintf(stderr, "    %d MiB, a pool of %lld tokens, preempted %g times\n", mib,
                (long long)pool, n);
    CHECK(n > 0);
    CHECK_EQ(s.stop(), 0);
}

/* ================================================================== image generation */

namespace {

/* The tiny Qwen-Image pipeline (tiny_model.h: QwenImage) as rad-convert makes it containers, the
 * bf16 one and the one whose DiT linears are block fp8, written once per process. Empty when a
 * step fails. */
struct QwenImageContainers {
    tiny::TempDir dir{ "e2e_qimg", std::getenv("RAD_E2E_KEEP") != nullptr };
    std::string pipeline, bf16, fp8;   /* the diffusers folder, and its two containers */
    QwenImageContainers() {
        const std::string p = dir / "qwen_image";
        if (!dir.ok() || mkdir(p.c_str(), 0755) != 0 || !tiny::write_qwen_image(p, {}) ||
            !tiny::write_text(dir / "fp8.recipe", tiny::kQwenImageFp8Recipe))
            return;
        for (const bool q : { false, true }) {
            const std::string rad = dir / (q ? "qwen_image_fp8.rad" : "qwen_image.rad");
            std::vector<std::string> argv = { RAD_E2E_CONVERT, p, "-o", rad };
            if (q) { argv.push_back("--recipe"); argv.push_back(dir / "fp8.recipe"); }
            std::string log;
            const int st = tiny::run(argv, engine_env(), &log);
            if (st != 0) {
                fprintf(stderr, "    rad-convert exited %d:\n%s\n", st, tail(log).c_str());
                return;
            }
            (q ? fp8 : bf16) = rad;
        }
        pipeline = p;
    }
};

const QwenImageContainers& qwen_image() {
    static QwenImageContainers c;
    return c;
}

/* One picture through the images API, and its pixels. */
struct Picture {
    int status = -1;
    std::string why;
    std::string png;
    std::vector<uint8_t> rgba;
    std::string log;
};

/* A 64 x 64 picture of `prompt` from a server on `kernels`: as a deployment runs it (no reference
 * library) unless `reference`, on a host device large enough for the buffers the plugin sizes for
 * its largest picture. */
Picture generate(const std::string& model, const std::vector<std::string>& kernels,
                 bool reference, uint64_t seed, int steps, double cfg = 0.0) {
    Picture p;
    Server s;
    s.serving = !reference;
    std::vector<std::string> args = { "--kernels" };
    std::string ks;
    for (const std::string& k : kernels) ks += (ks.empty() ? "" : ",") + k;
    args.push_back(ks);
    if (!s.start(model, args, device_env(2048))) {
        p.log = s.output();
        return p;
    }
    json req = { { "prompt", "a red cat on a blue mat" }, { "size", "64x64" }, { "seed", seed },
                 { "num_inference_steps", steps }, { "output_format", "png" } };
    if (cfg > 0) {
        req["negative_prompt"] = "a dark river";
        req["true_cfg_scale"] = cfg;
    }
    const json r = post(s.port, "/v1/images/generations", req, &p.status, &p.why);
    if (p.status == 200 && r.is_object() && r.contains("data") && r["data"].size() == 1) {
        std::vector<uint8_t> bytes;
        if (rad::server::base64_decode(r["data"][0].value("b64_json", std::string()), &bytes)) {
            p.png.assign(bytes.begin(), bytes.end());
            rad::mm::FrameRGBA f;
            std::string why;
            if (rad::mm::decode_image_rgba(bytes.data(), bytes.size(), 1 << 20, &f, &why) == RAD_OK &&
                f.w == 64 && f.h == 64)
                p.rgba = std::move(f.rgba);
        }
    }
    s.stop();
    p.log = s.output();
    if (p.status != 200 || p.rgba.empty())
        fprintf(stderr, "    %s: status %d%s%s, %zu pixel bytes:\n%s\n", ks.c_str(), p.status,
                p.why.empty() ? "" : " -- ", p.why.c_str(), p.rgba.size(), tail(p.log, 20).c_str());
    return p;
}

/* Peak signal to noise of two pictures' bytes, in dB (infinite when equal). */
double psnr(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    double se = 0;
    for (size_t i = 0; i < a.size(); ++i) se += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
    return se == 0 ? INFINITY : 10.0 * std::log10(255.0 * 255.0 * (double)a.size() / se);
}

}  /* namespace */

/* Qwen-Image on the host backend with libavx alone, no reference library loaded: every op the
 * generator declares has a libavx row, the picture comes back, a seed repeats it byte for byte, and
 * it agrees with libref's within rounding (the two libraries round the DiT's and the decoder's
 * products in different orders). Another seed gives another picture, so the agreement is not two
 * constant images. Guidance runs a second, negative conditioning beside the first; the fp8
 * container takes the fused fp8 norm, the fp8 GEMMs and the int8 attention scores. */
TEST(qwen_image_generates_on_libavx_alone_as_on_libref) {
#if defined(__x86_64__)
    NEED_ARTIFACT(RAD_E2E_HOME "/kernels/libavx.so");
#else
    SKIP("libavx is x86-64 only");
#endif
    if (!rad::mm::decoder_available()) SKIP("this build has no image codec");
    const QwenImageContainers& q = qwen_image();
    REQUIRE(!q.bf16.empty() && !q.fp8.empty());
    for (const std::string* model : { &q.bf16, &q.fp8 })
        for (const double cfg : { 0.0, 3.0 }) {
            const Picture a = generate(*model, { "libavx" }, false, 7, 3, cfg);
            const Picture b = generate(*model, { "libavx" }, false, 7, 3, cfg);
            const Picture c = generate(*model, { "libavx" }, false, 8, 3, cfg);
            const Picture ref = generate(*model, { "libref" }, true, 7, 3, cfg);
            REQUIRE(!a.rgba.empty() && !b.rgba.empty() && !c.rgba.empty() && !ref.rgba.empty());
            CHECK(a.png == b.png);
            const double same = psnr(a.rgba, ref.rgba), other = psnr(a.rgba, c.rgba);
            fprintf(stderr, "    %s, cfg %.0f: libavx against libref %.1f dB, against another "
                    "seed %.1f dB\n", model == &q.fp8 ? "fp8" : "bf16", cfg, same, other);
            CHECK(same > 40.0);
            CHECK(other < 30.0);
        }
}

/* A diffusers pipeline served as it is, unconverted: its tokenizer sits in the pipeline's own
 * folder, not beside the model's config, and the picture is the bf16 container's byte for byte,
 * since the container holds the same bf16 weights the loader reads from the pipeline. */
TEST(an_unconverted_qwen_image_pipeline_serves_the_containers_picture) {
#if defined(__x86_64__)
    NEED_ARTIFACT(RAD_E2E_HOME "/kernels/libavx.so");
#else
    SKIP("libavx is x86-64 only");
#endif
    if (!rad::mm::decoder_available()) SKIP("this build has no image codec");
    const QwenImageContainers& q = qwen_image();
    REQUIRE(!q.pipeline.empty() && !q.bf16.empty());
    const Picture direct = generate(q.pipeline, { "libavx" }, false, 7, 3);
    const Picture conv = generate(q.bf16, { "libavx" }, false, 7, 3);
    REQUIRE(!direct.rgba.empty() && !conv.rgba.empty());
    CHECK(direct.png == conv.png);
}

RAD_TEST_MAIN()
