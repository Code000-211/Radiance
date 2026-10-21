/* kld.h -- the KL divergence of this model against a recorded reference, at every scored position
 * of a fixed corpus.
 *
 * TWO RUNS, ONE YARDSTICK. `--kld-record DIR` runs the corpus through the model that defines the
 * answer -- the bf16 checkpoint -- and writes its log-probabilities over the whole vocabulary at
 * every scored position, once. `--kld-ref DIR` runs a candidate over the same tokens and compares
 * its own logits against what DIR holds, row by row, so a candidate costs one prefill of the
 * corpus and the reference is never computed again.
 *
 * WHAT IS REPORTED is the distribution of the per-position KL(P_ref || Q_candidate): its mean,
 * median, 99th, 99.9th and 99.99th percentiles and maximum, the top-1 agreement (the candidate's
 * argmax is the reference's), and both models' perplexity on the corpus's own next tokens. The
 * tail is the point: a quantised model whose mean KL is small can still be confidently wrong at
 * one position in ten thousand, and that is the position a user sees.
 *
 * THE WHOLE VOCABULARY, NOT A TOP-K. A KL over a truncated support is a lower bound that is
 * tightest where it matters least: a peaked distribution keeps its mass in the top few tokens and
 * a flat one -- the high-KL tail -- does not.
 *
 * THE REFERENCE FILES (DIR):
 *   kld.json     the manifest: the model, the corpus, every document's source, token count, first
 *                scored position and first row. Written LAST, so a run that dies leaves a
 *                directory that no --kld-ref will read.
 *   tokens.i32   every document's tokens, concatenated: a candidate is scored on the tokens the
 *                reference saw, not on a re-tokenisation that could differ.
 *   logp.f16     [rows, n_vocab] log-probabilities, f16. Near log p = 0 an f16 has its finest
 *                resolution, and the tail tokens it rounds coarsely carry almost no mass.
 *   top.bin      [rows, top] {int32 id, f32 log p} (top = kKldTop; kKldTopTaps in a taps recording,
 *                as kld.json's "top" says): the most likely tokens exactly, which is
 *                where the KL of a confident position lives and where f16 would cost ~1e-4 nats.
 *   rows.f32     [rows, 2] the reference's own next-token NLL and entropy.
 *
 * Document d scores positions [score_from, n - 1): the logits at p predict token p + 1, and the
 * last token predicts nothing in the corpus. Its row r is row0 + p - score_from.
 *
 * A TAPS RECORDING (--kld-taps, set_taps) is the same run with the model's hidden states beside
 * each row instead of the full log-probabilities: a drafter is trained on what the trunk computed
 * at position p and on the tokens that follow p, and 485 KB of f16 a position over a 248K
 * vocabulary is what makes millions of positions impossible to keep. So:
 *   taps.bf16    [rows, parts * part_width] bf16: the plugin's tap buffers side by side, part k at
 *                columns [k * part_width, (k + 1) * part_width), row r beside top.bin's row r.
 *   logp.f16     NOT WRITTEN. kld.json says "logp": false, and --kld-ref refuses the directory:
 *                a reference without the whole distribution cannot be scored against.
 * tokens.i32, top.bin and rows.f32 are written exactly as above, so a reader can still check every
 * row's next token against the model's own argmax (top.bin's first entry).
 */
#pragma once
#include "rad_core.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rad {

class Vocab;

/* One scoring entry's rows of a step: request `id`'s prompt positions [pos0, pos0 + n) are rows
 * [row0, row0 + n) of the block the engine copied back. */
struct KldSpan {
    uint64_t id   = 0;
    int64_t  pos0 = 0;
    int64_t  n    = 0;
    int64_t  row0 = 0;
};

/* One rank's part of that block: its rows of `width` f32 logits, the vocabulary from `vocab0`. */
struct KldShard {
    const float* rows   = nullptr;
    int64_t      vocab0 = 0;
    int64_t      width  = 0;
};

/* A step's tap rows, rank 0's: `parts[k]` is the plugin's k-th tap buffer as copied back, rows of
 * `part_width` bf16 in the same order as the logits block. Only a taps recording passes one. */
struct KldTaps {
    std::vector<const uint16_t*> parts;
    int64_t part_width = 0;
    const uint16_t* anchor = nullptr;
    int64_t anchor_width = 0;
};

/* The exact head kept beside every f16 row. */
static constexpr int kKldTop = 16;
/* ...and beside every taps row, where it is the ONLY distribution kept: 32 covers a top_k 20
 * sampling profile with room, so the probabilities a sampled verify draws from can be rebuilt
 * exactly from top.bin wherever top_p cuts inside them. */
static constexpr int kKldTopTaps = 32;

class Kld {
public:
    Kld();
    ~Kld();

    /* Reads the corpus and creates DIR (record), or reads DIR's manifest and opens its rows (ref).
     * `model` names this run's model in the manifest and the report. */
    int open(const Config& cfg, std::shared_ptr<const Vocab> vocab, int64_t n_vocab,
             const std::string& model);

    /* The two halves of open() for documents already tokenised, which is what a test drives:
     * add_doc() each document, then record_into() creates DIR. score_against() is the ref mode
     * whole. */
    int add_doc(const std::string& source, const std::vector<int32_t>& ids, int64_t score_from);
    /* Makes record_into() a taps recording: `parts` tap buffers of `part_width` bf16 a row, which
     * every score() then has to hand over, and no logp.f16. `layers` is what the manifest says the
     * parts are. Before record_into(); a ref run takes none. */
    int set_taps(const std::string& layers, int64_t parts, int64_t part_width);
    int record_into(const std::string& dir, int64_t n_vocab, const std::string& model);
    int score_against(const std::string& dir, const std::string& out, int64_t n_vocab,
                      const std::string& model);

    size_t n_docs() const { return docs_.size(); }
    /* Document d's tokens, and the request id the engine submits it under: d + 1. */
    std::vector<int32_t> tokens(size_t d) const;
    int64_t n_rows() const { return n_rows_; }
    int64_t rows_done() const { return done_n_.load(std::memory_order_relaxed); }

    /* A step's scored rows. Rows outside a document's scored range are skipped; a row seen twice
     * -- a document re-prefilled after preemption -- is simply computed again. Blocks until every
     * row is written (record) or scored (ref): the engine reuses the block at the next step.
     * `taps` is required of a taps recording and refused anywhere else. */
    int score(const std::vector<KldSpan>& spans, const std::vector<KldShard>& shards,
              const KldTaps* taps = nullptr);

    /* Record: checks every row was written and writes the manifest. Ref: the report, printed and,
     * with --kld-out, written. */
    int finish();

private:
    struct Doc {
        std::string source;
        int64_t     off = 0;          /* into tokens_ */
        int64_t     n = 0;
        int64_t     score_from = 0;
        int64_t     row0 = 0;
        int64_t     rows() const { return n - 1 - score_from; }
    };
    struct Job {
        int64_t row;                  /* the reference's row */
        int64_t at;                   /* the row in the copied block */
        int32_t next;                 /* the corpus's next token */
    };

    int  load_corpus(const Config& cfg, const std::shared_ptr<const Vocab>& vocab);
    int  load_ref(const std::string& dir);
    void size_rows();
    void record_row(const Job& j, const std::vector<KldShard>& sh, std::vector<uint16_t>& f16,
                    const KldTaps* taps, std::vector<uint16_t>& tap_row);
    void score_row(const Job& j, const std::vector<KldShard>& sh, std::vector<uint16_t>& f16);
    int  report();

    bool record_ = false;
    std::string dir_, out_, model_, ref_model_, corpus_;
    uint64_t corpus_hash_ = 0;
    int64_t n_vocab_ = 0;
    std::vector<int32_t> tokens_;
    std::vector<Doc> docs_;
    int64_t n_rows_ = 0;
    int fd_logp_ = -1, fd_top_ = -1, fd_rows_ = -1, fd_taps_ = -1;

    /* A taps recording (set_taps): zero parts is an ordinary reference. */
    std::string taps_layers_;
    int64_t taps_parts_ = 0, taps_pw_ = 0;
    int     top_n_ = kKldTop;          /* entries a top.bin row holds */
    int64_t taps_width() const { return taps_parts_ * taps_pw_; }

    /* Per reference row. `seen_` marks a row written or scored; the rest is the ref mode's. */
    std::vector<uint8_t> seen_;
    std::atomic<int64_t> done_n_{0};
    std::vector<float>   kl_, nll_q_, nll_p_;
    std::vector<uint8_t> top1_;
    std::atomic<int>     io_err_{0};
    float f16_lut_[65536];
};

}  /* namespace rad */
