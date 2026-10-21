/* kld.cpp -- see kld.h. */
#include "kld.h"

#include "rad_internal.h"
#include "rad_plugin.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <map>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rad {

using json = nlohmann::ordered_json;

namespace {

constexpr const char* kFormat = "radiance-kld-1";

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

bool pwrite_all(int fd, const void* p, size_t n, int64_t off) {
    const auto* b = static_cast<const char*>(p);
    while (n) {
        const ssize_t w = ::pwrite(fd, b, n, (off_t)off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        b += w; n -= (size_t)w; off += w;
    }
    return true;
}

bool pread_all(int fd, void* p, size_t n, int64_t off) {
    auto* b = static_cast<char*>(p);
    while (n) {
        const ssize_t r = ::pread(fd, b, n, (off_t)off);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        b += r; n -= (size_t)r; off += r;
    }
    return true;
}

bool write_file(const std::string& path, const void* p, size_t n) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;
    const bool ok = pwrite_all(fd, p, n, 0);
    return ::close(fd) == 0 && ok;
}

/* A file of exactly `bytes`, sparse until its rows are written. */
int create_sized(const std::string& path, int64_t bytes) {
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    if (::ftruncate(fd, (off_t)bytes) != 0) { ::close(fd); return -1; }
    return fd;
}

int open_sized(const std::string& path, int64_t bytes) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return -1;
    struct stat st{};
    if (::fstat(fd, &st) != 0 || (int64_t)st.st_size != bytes) { ::close(fd); return -2; }
    return fd;
}

std::string now_iso() {
    char buf[32];
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

int worker_count() {
    const unsigned h = std::thread::hardware_concurrency();
    return (int)std::clamp<unsigned>(h > 2 ? h - 2 : 1, 1, 32);
}

/* The q-quantile by linear interpolation between order statistics, numpy's default. */
double quantile(const std::vector<float>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double x = q * (double)(sorted.size() - 1);
    const size_t i = (size_t)x;
    if (i + 1 >= sorted.size()) return sorted.back();
    return sorted[i] + (x - (double)i) * (sorted[i + 1] - sorted[i]);
}

struct Summary {
    int64_t n = 0;
    double mean = 0, median = 0, p90 = 0, p95 = 0, p99 = 0, p999 = 0, p9999 = 0, max = 0;
    double top1 = 0, ppl_q = 0, ppl_p = 0;
};

Summary summarise(std::vector<float> kl, int64_t top1, double nll_q, double nll_p) {
    Summary s;
    s.n = (int64_t)kl.size();
    if (!s.n) return s;
    std::sort(kl.begin(), kl.end());
    double sum = 0;
    for (float v : kl) sum += v;
    s.mean   = sum / (double)s.n;
    s.median = quantile(kl, 0.5);
    s.p90    = quantile(kl, 0.90);
    s.p95    = quantile(kl, 0.95);
    s.p99    = quantile(kl, 0.99);
    s.p999   = quantile(kl, 0.999);
    s.p9999  = quantile(kl, 0.9999);
    s.max    = kl.back();
    s.top1   = (double)top1 / (double)s.n;
    s.ppl_q  = std::exp(nll_q / (double)s.n);
    s.ppl_p  = std::exp(nll_p / (double)s.n);
    return s;
}

json summary_json(const Summary& s) {
    json j;
    j["positions"] = s.n;
    j["kld"] = { {"mean", s.mean}, {"median", s.median}, {"p90", s.p90}, {"p95", s.p95},
                 {"p99", s.p99}, {"p99_9", s.p999}, {"p99_99", s.p9999}, {"max", s.max} };
    j["top1_agreement"] = s.top1;
    j["ppl"] = { {"candidate", s.ppl_q}, {"reference", s.ppl_p},
                 {"ratio", s.ppl_p > 0 ? s.ppl_q / s.ppl_p : 0.0} };
    return j;
}

/* The logit of global token `id`, from whichever shard holds it. */
inline float logit_at(const std::vector<KldShard>& sh, int64_t at, int64_t id) {
    for (const KldShard& s : sh)
        if (id >= s.vocab0 && id < s.vocab0 + s.width) return s.rows[at * s.width + (id - s.vocab0)];
    return -std::numeric_limits<float>::infinity();
}

struct TopEntry { int32_t id; float lp; };

}  /* namespace */

Kld::Kld() {
    for (uint32_t h = 0; h < 65536; ++h) f16_lut_[h] = rad_f16_to_f32((uint16_t)h);
}

Kld::~Kld() {
    for (int fd : { fd_logp_, fd_top_, fd_rows_, fd_taps_ }) if (fd >= 0) ::close(fd);
}

/* After add_doc is fine; after record_into is not, and the open top.bin is what says that. */
int Kld::set_taps(const std::string& layers, int64_t parts, int64_t part_width) {
    if (parts <= 0 || part_width <= 0 || fd_top_ >= 0) return RAD_E_INVAL;
    taps_layers_ = layers;
    taps_parts_  = parts;
    taps_pw_     = part_width;
    top_n_       = kKldTopTaps;
    return RAD_OK;
}

std::vector<int32_t> Kld::tokens(size_t d) const {
    const Doc& doc = docs_[d];
    return std::vector<int32_t>(tokens_.begin() + doc.off, tokens_.begin() + doc.off + doc.n);
}

int Kld::open(const Config& cfg, std::shared_ptr<const Vocab> vocab, int64_t n_vocab,
              const std::string& model) {
    if (cfg.kld_record.empty()) return score_against(cfg.kld_ref, cfg.kld_out, n_vocab, model);
    RAD_TRY(load_corpus(cfg, vocab));
    return record_into(cfg.kld_record, n_vocab, model);
}

int Kld::score_against(const std::string& dir, const std::string& out, int64_t n_vocab,
                       const std::string& model) {
    record_  = false;
    dir_     = dir;
    out_     = out;
    model_   = model;
    n_vocab_ = n_vocab;
    RAD_TRY(load_ref(dir_));
    size_rows();
    return RAD_OK;
}

void Kld::size_rows() {
    seen_.assign((size_t)n_rows_, 0);
    if (record_) return;
    kl_.assign((size_t)n_rows_, 0.0f);
    nll_q_.assign((size_t)n_rows_, 0.0f);
    nll_p_.assign((size_t)n_rows_, 0.0f);
    top1_.assign((size_t)n_rows_, 0);
}

int Kld::add_doc(const std::string& source, const std::vector<int32_t>& ids, int64_t score_from) {
    Doc d;
    d.source     = source;
    d.off        = (int64_t)tokens_.size();
    d.n          = (int64_t)ids.size();
    d.score_from = score_from;
    if (d.score_from < 0 || d.rows() < 1) return RAD_E_INVAL;
    d.row0 = n_rows_;
    n_rows_ += d.rows();
    tokens_.insert(tokens_.end(), ids.begin(), ids.end());
    docs_.push_back(std::move(d));
    return RAD_OK;
}

/* ------------------------------------------------------------------ the corpus (record) */
int Kld::load_corpus(const Config& cfg, const std::shared_ptr<const Vocab>& vocab) {
    corpus_ = cfg.kld_corpus;
    std::ifstream in(corpus_, std::ios::binary);
    if (!in) { RAD_ERR("kld: cannot read --kld-corpus %s", corpus_.c_str()); return RAD_E_IO; }
    /* Built at the first "prompt" line: a corpus of "ids" alone needs no tokenizer. */
    Tokenizer tok;
    bool tok_ready = false;
    std::string line;
    int64_t lineno = 0;
    corpus_hash_ = fnv1a(nullptr, 0);
    while (std::getline(in, line)) {
        ++lineno;
        corpus_hash_ = fnv1a(line.data(), line.size(), corpus_hash_);
        if (line.empty()) continue;
        json j;
        try { j = json::parse(line); }
        catch (const std::exception& e) {
            RAD_ERR("kld: %s line %lld is not JSON: %s", corpus_.c_str(), (long long)lineno, e.what());
            return RAD_E_FORMAT;
        }
        const bool has_prompt = j.contains("prompt"), has_ids = j.contains("ids");
        if (has_prompt == has_ids || (has_prompt && !j["prompt"].is_string()) ||
            (has_ids && !j["ids"].is_array())) {
            RAD_ERR("kld: %s line %lld needs exactly one of a \"prompt\" string or an \"ids\" array",
                    corpus_.c_str(), (long long)lineno);
            return RAD_E_FORMAT;
        }
        std::vector<int32_t> ids;
        if (has_ids) {
            /* THE TOKENS THEMSELVES, for a document whose tokens are known exactly -- a prompt the
             * server rendered and tokenised, followed by what the model generated. Re-encoding the
             * generated text can split it differently from how it was sampled, and a drafter's
             * targets are the sampled tokens. */
            const int64_t nv = vocab ? (int64_t)vocab->n_tokens() : 0;
            for (const json& t : j["ids"]) {
                if (!t.is_number_integer() || t.get<int64_t>() < 0 ||
                    (nv > 0 && t.get<int64_t>() >= nv)) {
                    RAD_ERR("kld: %s line %lld: \"ids\" holds something other than a token id of "
                            "this vocabulary", corpus_.c_str(), (long long)lineno);
                    return RAD_E_FORMAT;
                }
                ids.push_back((int32_t)t.get<int64_t>());
            }
        } else {
            /* THE TEXT AS THE SERVER WOULD SEE A RENDERED TEMPLATE: control tokens in it are
             * themselves, and nothing is added in front. */
            if (!tok_ready) { RAD_TRY(tok.init(vocab)); tok_ready = true; }
            RAD_TRY(tok.encode(j["prompt"].get<std::string>(), ids, false, true));
        }
        const int64_t from = j.value("score_from", (int64_t)0);
        if (add_doc(j.value("source", std::string()), ids, from) < 0) {
            RAD_ERR("kld: %s line %lld: %zu tokens leave nothing to score from position %lld",
                    corpus_.c_str(), (long long)lineno, ids.size(), (long long)from);
            return RAD_E_FORMAT;
        }
    }
    if (docs_.empty()) { RAD_ERR("kld: %s holds no documents", corpus_.c_str()); return RAD_E_FORMAT; }
    return RAD_OK;
}

int Kld::record_into(const std::string& dir, int64_t n_vocab, const std::string& model) {
    record_  = true;
    dir_     = dir;
    model_   = model;
    n_vocab_ = n_vocab;
    if (docs_.empty()) return RAD_E_INVAL;
    size_rows();
    /* A REFERENCE IS NEVER OVERWRITTEN. Everything measured against it would silently change
     * meaning; a directory with no manifest is a run that died, and is written over. */
    ::mkdir(dir_.c_str(), 0755);
    struct stat st{};
    if (::stat((dir_ + "/kld.json").c_str(), &st) == 0) {
        RAD_ERR("kld: %s already holds a reference; record into a new directory", dir_.c_str());
        return RAD_E_INVAL;
    }
    if (!write_file(dir_ + "/tokens.i32", tokens_.data(), tokens_.size() * sizeof(int32_t))) {
        RAD_ERR("kld: cannot write %s/tokens.i32: %s", dir_.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    /* A taps recording keeps the hidden states INSTEAD of the whole distribution (kld.h). */
    const bool taps = taps_parts_ > 0;
    if (!taps) fd_logp_ = create_sized(dir_ + "/logp.f16", n_rows_ * n_vocab_ * 2);
    else       fd_taps_ = create_sized(dir_ + "/taps.bf16", n_rows_ * taps_width() * 2);
    fd_top_  = create_sized(dir_ + "/top.bin", n_rows_ * top_n_ * (int64_t)sizeof(TopEntry));
    fd_rows_ = create_sized(dir_ + "/rows.f32", n_rows_ * 2 * 4);
    if ((taps ? fd_taps_ : fd_logp_) < 0 || fd_top_ < 0 || fd_rows_ < 0) {
        RAD_ERR("kld: cannot create the reference files in %s: %s", dir_.c_str(),
                std::strerror(errno));
        return RAD_E_IO;
    }
    const int64_t row_bytes = taps ? taps_width() * 2 : n_vocab_ * 2;
    RAD_INFO("kld: recording %lld positions over %zu documents (%zu tokens) into %s, %.1f GiB of %s",
             (long long)n_rows_, docs_.size(), tokens_.size(), dir_.c_str(),
             (double)(n_rows_ * row_bytes) / (1024.0 * 1024.0 * 1024.0),
             taps ? "taps" : "log-probabilities");
    return RAD_OK;
}

/* ------------------------------------------------------------------ the reference (ref) */
int Kld::load_ref(const std::string& dir) {
    std::ifstream in(dir + "/kld.json");
    if (!in) {
        RAD_ERR("kld: %s has no kld.json -- not a reference, or its recording did not finish",
                dir.c_str());
        return RAD_E_IO;
    }
    json m;
    try { in >> m; }
    catch (const std::exception& e) {
        RAD_ERR("kld: %s/kld.json: %s", dir.c_str(), e.what());
        return RAD_E_FORMAT;
    }
    if (m.value("format", std::string()) != kFormat) {
        RAD_ERR("kld: %s/kld.json is not %s", dir.c_str(), kFormat);
        return RAD_E_FORMAT;
    }
    if (!m.value("logp", true)) {
        RAD_ERR("kld: %s is a taps recording and holds no log-probabilities to score against",
                dir.c_str());
        return RAD_E_INVAL;
    }
    if (m.value("n_vocab", (int64_t)0) != n_vocab_ || m.value("top", 0) != kKldTop) {
        RAD_ERR("kld: the reference in %s has a %lld-token vocabulary and this model %lld",
                dir.c_str(), (long long)m.value("n_vocab", (int64_t)0), (long long)n_vocab_);
        return RAD_E_INVAL;
    }
    ref_model_ = m.value("model", std::string());
    corpus_    = m.value("corpus", std::string());
    int64_t n_tok = 0;
    for (const json& jd : m["docs"]) {
        Doc d;
        d.source     = jd.value("source", std::string());
        d.off        = n_tok;
        d.n          = jd.at("tokens").get<int64_t>();
        d.score_from = jd.at("score_from").get<int64_t>();
        d.row0       = jd.at("row0").get<int64_t>();
        if (d.row0 != n_rows_ || d.rows() < 1) {
            RAD_ERR("kld: %s/kld.json: document %zu does not follow on", dir.c_str(), docs_.size());
            return RAD_E_FORMAT;
        }
        n_rows_ += d.rows();
        n_tok   += d.n;
        docs_.push_back(std::move(d));
    }
    if (n_rows_ != m.value("rows", (int64_t)-1)) {
        RAD_ERR("kld: %s/kld.json: its documents add up to %lld rows, not %lld", dir.c_str(),
                (long long)n_rows_, (long long)m.value("rows", (int64_t)-1));
        return RAD_E_FORMAT;
    }
    tokens_.resize((size_t)n_tok);
    const int fd = open_sized(dir + "/tokens.i32", n_tok * 4);
    const bool ok = fd >= 0 && pread_all(fd, tokens_.data(), (size_t)n_tok * 4, 0);
    if (fd >= 0) ::close(fd);
    fd_logp_ = open_sized(dir + "/logp.f16", n_rows_ * n_vocab_ * 2);
    fd_top_  = open_sized(dir + "/top.bin", n_rows_ * kKldTop * (int64_t)sizeof(TopEntry));
    fd_rows_ = open_sized(dir + "/rows.f32", n_rows_ * 2 * 4);
    if (!ok || fd_logp_ < 0 || fd_top_ < 0 || fd_rows_ < 0) {
        RAD_ERR("kld: the files in %s are missing or not the size kld.json says", dir.c_str());
        return RAD_E_FORMAT;
    }
    RAD_INFO("kld: scoring %lld positions over %zu documents against %s (%s)",
             (long long)n_rows_, docs_.size(), dir.c_str(), ref_model_.c_str());
    return RAD_OK;
}

/* ------------------------------------------------------------------ a step's rows */
int Kld::score(const std::vector<KldSpan>& spans, const std::vector<KldShard>& shards,
               const KldTaps* taps) {
    /* A taps recording without its taps would leave rows of zeros that look like data, and taps
     * handed to anything else would be dropped without a word. */
    if ((record_ && taps_parts_ > 0) != (taps != nullptr)) {
        RAD_ERR("kld: a step's taps %s", taps ? "were handed to a run that records none"
                                              : "are missing from a taps recording");
        return RAD_E_STATE;
    }
    if (taps && ((int64_t)taps->parts.size() != taps_parts_ || taps->part_width != taps_pw_)) {
        RAD_ERR("kld: a step handed %zu tap parts of %lld and the recording takes %lld of %lld",
                taps->parts.size(), (long long)taps->part_width, (long long)taps_parts_,
                (long long)taps_pw_);
        return RAD_E_STATE;
    }
    std::vector<Job> jobs;
    for (const KldSpan& s : spans) {
        if (s.id == 0 || s.id > docs_.size()) continue;
        const Doc& d = docs_[(size_t)(s.id - 1)];
        for (int64_t j = 0; j < s.n; ++j) {
            const int64_t p = s.pos0 + j;
            if (p < d.score_from || p >= d.n - 1) continue;
            jobs.push_back({ d.row0 + p - d.score_from, s.row0 + j, tokens_[(size_t)(d.off + p + 1)] });
        }
    }
    if (jobs.empty()) return RAD_OK;

    std::atomic<size_t> next{0};
    auto work = [&] {
        std::vector<uint16_t> f16((size_t)n_vocab_);
        std::vector<uint16_t> tap_row((size_t)taps_width());
        for (size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < jobs.size();) {
            const Job& j = jobs[i];
            if (record_) record_row(j, shards, f16, taps, tap_row);
            else         score_row(j, shards, f16);
            if (!seen_[(size_t)j.row]) {
                seen_[(size_t)j.row] = 1;
                done_n_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    const int nt = (int)std::min<size_t>((size_t)worker_count(), jobs.size());
    std::vector<std::thread> th;
    th.reserve((size_t)nt);
    for (int t = 0; t < nt; ++t) th.emplace_back(work);
    for (auto& t : th) t.join();
    if (io_err_.load()) {
        RAD_ERR("kld: reading or writing the reference in %s failed", dir_.c_str());
        return RAD_E_IO;
    }
    return RAD_OK;
}

/* THE REFERENCE ROW: log-softmax over the whole vocabulary, in double where it sums. A taps
 * recording keeps the head, the NLL and the entropy the same way and writes the tap row in place
 * of the f16 one. */
void Kld::record_row(const Job& j, const std::vector<KldShard>& sh, std::vector<uint16_t>& f16,
                     const KldTaps* taps, std::vector<uint16_t>& tap_row) {
    const bool keep_logp = fd_logp_ >= 0;
    float m = -std::numeric_limits<float>::infinity();
    for (const KldShard& s : sh) {
        const float* l = s.rows + j.at * s.width;
        for (int64_t i = 0; i < s.width; ++i) m = std::max(m, l[i]);
    }
    double sum = 0;
    for (const KldShard& s : sh) {
        const float* l = s.rows + j.at * s.width;
        for (int64_t i = 0; i < s.width; ++i) sum += std::exp(l[i] - m);
    }
    const double lse = (double)m + std::log(sum);

    /* The head kept exactly: a min-heap on log p, so an entry leaves only for a strictly larger
     * one and a tie keeps the lower id, which is the candidate's argmax rule. */
    TopEntry top[kKldTopTaps] = {};
    const int cap = top_n_;
    int n_top = 0;
    auto worse = [](const TopEntry& a, const TopEntry& b) {
        return a.lp > b.lp || (a.lp == b.lp && a.id < b.id);
    };
    double ent = 0;
    for (const KldShard& s : sh) {
        const float* l = s.rows + j.at * s.width;
        for (int64_t i = 0; i < s.width; ++i) {
            const float lp = (float)((double)l[i] - lse);
            const int64_t id = s.vocab0 + i;
            if (keep_logp) f16[(size_t)id] = rad_f32_to_f16(lp);
            const double p = std::exp((double)lp);
            if (p > 0) ent -= p * (double)lp;
            if (n_top < cap) {
                top[n_top++] = { (int32_t)id, lp };
                std::push_heap(top, top + n_top, worse);
            } else if (lp > top[0].lp) {
                std::pop_heap(top, top + n_top, worse);
                top[n_top - 1] = { (int32_t)id, lp };
                std::push_heap(top, top + n_top, worse);
            }
        }
    }
    std::sort(top, top + n_top, worse);
    const float rows[2] = { (float)(lse - (double)logit_at(sh, j.at, j.next)), (float)ent };
    const int64_t r = j.row;
    if (taps) {
        const int64_t pw = taps->part_width;
        for (size_t k = 0; k < taps->parts.size(); ++k)
            std::memcpy(tap_row.data() + (int64_t)k * pw, taps->parts[k] + j.at * pw,
                        (size_t)pw * 2);
        if (!pwrite_all(fd_taps_, tap_row.data(), tap_row.size() * 2,
                        r * (int64_t)tap_row.size() * 2))
            io_err_.store(1);
    }
    if ((keep_logp && !pwrite_all(fd_logp_, f16.data(), (size_t)n_vocab_ * 2, r * n_vocab_ * 2)) ||
        !pwrite_all(fd_top_, top, (size_t)cap * sizeof(TopEntry), r * cap * (int64_t)sizeof(TopEntry)) ||
        !pwrite_all(fd_rows_, rows, sizeof rows, r * (int64_t)sizeof rows))
        io_err_.store(1);
}

/* THE CANDIDATE ROW against the reference's:
 *
 *   KL(P || Q) = sum_i p_i (log p_i - log q_i)
 *
 * with log q from this model's logits at full f32 precision and log p from the reference: f16
 * for the whole row, then the kKldTop most likely tokens replaced by their exact values. p is
 * renormalised over the row as stored -- the f16 rounding leaves sum p a few 1e-5 off one -- so
 * with S = sum e^{lp} and A = sum e^{lp} (lp - lq), KL = A / S - log S. */
void Kld::score_row(const Job& j, const std::vector<KldShard>& sh, std::vector<uint16_t>& f16) {
    float m = -std::numeric_limits<float>::infinity();
    int64_t arg = 0;
    for (const KldShard& s : sh) {
        const float* l = s.rows + j.at * s.width;
        for (int64_t i = 0; i < s.width; ++i)
            if (l[i] > m) { m = l[i]; arg = s.vocab0 + i; }
    }
    double sum = 0;
    for (const KldShard& s : sh) {
        const float* l = s.rows + j.at * s.width;
        for (int64_t i = 0; i < s.width; ++i) sum += std::exp(l[i] - m);
    }
    const double lse = (double)m + std::log(sum);

    TopEntry top[kKldTop];
    float rows[2];
    const int64_t r = j.row;
    if (!pread_all(fd_logp_, f16.data(), (size_t)n_vocab_ * 2, r * n_vocab_ * 2) ||
        !pread_all(fd_top_, top, sizeof top, r * (int64_t)sizeof top) ||
        !pread_all(fd_rows_, rows, sizeof rows, r * (int64_t)sizeof rows)) {
        io_err_.store(1);
        return;
    }

    double S = 0, A = 0;
    for (const KldShard& s : sh) {
        const float* l = s.rows + j.at * s.width;
        const uint16_t* h = f16.data() + s.vocab0;
        for (int64_t i = 0; i < s.width; ++i) {
            const float lp = f16_lut_[h[i]];
            const double p = std::exp((double)lp);
            if (!(p > 0)) continue;
            S += p;
            A += p * ((double)lp - ((double)l[i] - lse));
        }
    }
    for (const TopEntry& t : top) {
        const double lq  = (double)logit_at(sh, j.at, t.id) - lse;
        const float lp16 = f16_lut_[f16[(size_t)t.id]];
        const double p16 = std::exp((double)lp16), px = std::exp((double)t.lp);
        if (p16 > 0) { S -= p16; A -= p16 * ((double)lp16 - lq); }
        if (px > 0)  { S += px;  A += px * ((double)t.lp - lq); }
    }
    kl_[(size_t)r]    = (float)(A / S - std::log(S));
    top1_[(size_t)r]  = arg == (int64_t)top[0].id;
    nll_q_[(size_t)r] = (float)(lse - (double)logit_at(sh, j.at, j.next));
    nll_p_[(size_t)r] = rows[0];
}

/* ------------------------------------------------------------------ the end */
int Kld::finish() {
    const int64_t missing = n_rows_ - done_n_.load();
    if (missing) {
        RAD_ERR("kld: %lld of %lld positions never came back from the engine", (long long)missing,
                (long long)n_rows_);
        return RAD_E_STATE;
    }
    if (!record_) return report();

    for (int fd : { fd_logp_, fd_top_, fd_rows_, fd_taps_ })
        if (fd >= 0 && ::fsync(fd) != 0) {
            RAD_ERR("kld: fsync: %s", std::strerror(errno));
            return RAD_E_IO;
        }
    json m;
    m["format"]         = kFormat;
    m["model"]          = model_;
    m["corpus"]         = corpus_;
    m["corpus_fnv1a64"] = fmt("%016llx", (unsigned long long)corpus_hash_);
    m["created"]        = now_iso();
    m["n_vocab"]        = n_vocab_;
    m["top"]            = top_n_;
    m["rows"]           = n_rows_;
    m["logp"]           = fd_logp_ >= 0;
    if (taps_parts_ > 0)
        m["taps"] = { {"file", "taps.bf16"}, {"dtype", "bf16"}, {"layers", taps_layers_},
                      {"parts", taps_parts_}, {"part_width", taps_pw_},
                      {"width", taps_width()} };
    json docs = json::array();
    for (const Doc& d : docs_)
        docs.push_back({ {"source", d.source}, {"tokens", d.n}, {"score_from", d.score_from},
                         {"row0", d.row0} });
    m["docs"] = std::move(docs);
    const std::string tmp = dir_ + "/kld.json.tmp", path = dir_ + "/kld.json";
    {
        std::ofstream o(tmp);
        o << m.dump(1) << "\n";
        if (!o) { RAD_ERR("kld: cannot write %s", tmp.c_str()); return RAD_E_IO; }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        RAD_ERR("kld: cannot rename %s: %s", tmp.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    RAD_INFO("kld: reference recorded: %lld positions in %s", (long long)n_rows_, dir_.c_str());
    return RAD_OK;
}

int Kld::report() {
    auto group = [&](const std::vector<size_t>& docs) {
        std::vector<float> kl;
        int64_t t1 = 0;
        double nq = 0, np = 0;
        for (size_t di : docs) {
            const Doc& d = docs_[di];
            for (int64_t r = d.row0; r < d.row0 + d.rows(); ++r) {
                kl.push_back(kl_[(size_t)r]);
                t1 += top1_[(size_t)r];
                nq += nll_q_[(size_t)r];
                np += nll_p_[(size_t)r];
            }
        }
        return summarise(std::move(kl), t1, nq, np);
    };
    std::vector<size_t> all(docs_.size());
    std::map<std::string, std::vector<size_t>> by_src;
    for (size_t i = 0; i < docs_.size(); ++i) { all[i] = i; by_src[docs_[i].source].push_back(i); }
    const Summary s = group(all);

    RAD_INFO("kld: %s against %s, %lld positions over %zu documents", model_.c_str(),
             ref_model_.c_str(), (long long)s.n, docs_.size());
    RAD_INFO("kld: KLD mean %.6f  median %.6f  p99 %.5f  p99.9 %.4f  p99.99 %.4f  max %.4f",
             s.mean, s.median, s.p99, s.p999, s.p9999, s.max);
    RAD_INFO("kld: top-1 agreement %.3f%%   PPL %.4f vs reference %.4f (x%.4f)",
             100.0 * s.top1, s.ppl_q, s.ppl_p, s.ppl_p > 0 ? s.ppl_q / s.ppl_p : 0.0);
    json rep;
    rep["model"]     = model_;
    rep["reference"] = ref_model_;
    rep["ref_dir"]   = dir_;
    rep["corpus"]    = corpus_;
    rep["created"]   = now_iso();
    rep["docs"]      = (int64_t)docs_.size();
    rep["percentile"] = "linear interpolation between order statistics";
    rep["all"]       = summary_json(s);
    json src = json::object();
    for (const auto& [name, docs] : by_src) {
        const Summary g = group(docs);
        RAD_INFO("kld:   %-24s %6lld pos  mean %.5f  median %.6f  p99 %.4f  p99.9 %.4f  top-1 %.2f%%",
                 name.empty() ? "(no source)" : name.c_str(), (long long)g.n, g.mean, g.median,
                 g.p99, g.p999, 100.0 * g.top1);
        src[name] = summary_json(g);
        src[name]["docs"] = (int64_t)docs.size();
    }
    rep["by_source"] = std::move(src);

    if (out_.empty()) return RAD_OK;
    {
        std::ofstream o(out_);
        o << rep.dump(1) << "\n";
        if (!o) { RAD_ERR("kld: cannot write %s", out_.c_str()); return RAD_E_IO; }
    }
    /* Every position, in reference row order: KL, the candidate's and the reference's NLL of the
     * next token, and whether the argmaxes agreed. */
    std::vector<float> rows((size_t)n_rows_ * 4);
    for (int64_t r = 0; r < n_rows_; ++r) {
        rows[(size_t)r * 4 + 0] = kl_[(size_t)r];
        rows[(size_t)r * 4 + 1] = nll_q_[(size_t)r];
        rows[(size_t)r * 4 + 2] = nll_p_[(size_t)r];
        rows[(size_t)r * 4 + 3] = (float)top1_[(size_t)r];
    }
    if (!write_file(out_ + ".rows", rows.data(), rows.size() * sizeof(float))) {
        RAD_ERR("kld: cannot write %s.rows", out_.c_str());
        return RAD_E_IO;
    }
    RAD_INFO("kld: report in %s, every position in %s.rows", out_.c_str(), out_.c_str());
    return RAD_OK;
}

}  /* namespace rad */
