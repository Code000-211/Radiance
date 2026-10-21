/* tokenizer.h -- the tokeniser, driven by the step chain baked into the .rad.
 *
 * spec §12: "The tokenizer is the one place llama.cpp's answer is Python." Their vocab is
 * GGUF-shaped, the tokenizer.json -> vocab mapping lives in convert_hf_to_gguf.py, and at RUNTIME
 * they identify the pre-tokeniser by HASHING tokenizer.json against a hardcoded enum. An
 * unrecognised model therefore gets the default split, and the symptom of that is plausible text
 * with wrong token boundaries -- the hardest kind of wrong to notice, because nothing crashes and
 * the perplexity moves by a few percent.
 *
 * So the chain is DECLARED rather than recognised. rad-convert reads tokenizer.json's
 * normalizer / pre_tokenizer / decoder sections and emits them as RadVocabStep rows; this file
 * executes those rows in order. A model nobody has ever seen tokenises correctly if its
 * tokenizer.json is honest, and a model whose chain we cannot express fails at CONVERT time with
 * the step named, which is a diagnosis rather than a silent 2% regression.
 *
 * A `Split` step is a regex, and it is COMPILED at load (tokenizer_regex.h) into a DFA that
 * reproduces what HuggingFace tokenizers' Oniguruma reports for it, over Oniguruma's own Unicode
 * tables. A pattern outside the dialect that compiler implements does not load: the refusal names
 * the construct, and there is no second matcher behind it.
 *
 * What is lifted from llama.cpp (06938ac12, MIT -- see vendor/UPSTREAM):
 *   - the byte-level alphabet and the unicode category tables the non-regex steps read
 *                                                                        (vendor/llama/src/unicode*)
 *   - the shape of the BPE bigram queue and the UGM Viterbi              (src/llama-vocab.cpp)
 * The last of those is re-expressed here rather than vendored: llama-vocab.cpp is welded to
 * llama_model_loader, GGML_ASSERT and the llama_vocab class, so it is a file we would have had
 * to modify heavily, and spec §12 says such a file belongs in core/ under our own name.
 */
#pragma once
#include "rad_internal.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rad {

class GgufFile;
struct TokEncoder;

/* ================================================================== the mutable vocab */

/* One row of the declared chain, in the form rad-convert builds and the .rad bakes.
 *
 * `kind` is a RAD_NORM_* / RAD_PRE_* / RAD_DEC_* value from rad_format.h and its numeric range
 * says which of the three chains the row belongs to, so the three are ONE list in declared order
 * and no reader has to reassemble them. Per-kind meaning of flags/args is documented at
 * `step_describe()` in tokenizer.cpp; it is the contract rad-convert writes against. */
struct VocabStep {
    uint32_t    kind = 0;
    uint32_t    flags = 0;
    std::string arg0, arg1;
    int64_t     iarg = 0;
};

/* Kind ranges, matching the enum blocks in rad_format.h. */
inline bool step_is_norm(uint32_t k) { return k >= 1  && k < 32; }
inline bool step_is_pre (uint32_t k) { return k >= 32 && k < 64; }
inline bool step_is_dec (uint32_t k) { return k >= 64 && k < 96; }

/* Everything the tokeniser needs, in a form that can be built, inspected, written and read.
 *
 * The engine could bind directly to the mmapped .rad and save copying 248K token strings, and
 * deliberately does not: one construction path means the tokeniser that rad-convert tests with is
 * byte-for-byte the tokeniser the server runs. The cost is a few megabytes and a few tens of
 * milliseconds at load, against a model measured in gigabytes -- which is not a cost. */
struct VocabBuild {
    uint32_t kind = RAD_TOK_BPE;

    std::vector<std::string> tokens;          /* id -> text, in the vocab's own encoding */
    std::vector<float>       scores;          /* unigram only; empty otherwise */
    std::vector<uint8_t>     types;           /* RAD_TT_*; empty means all RAD_TT_NORMAL */
    std::vector<std::pair<uint32_t, uint32_t>> merges;   /* BPE, in rank order */

    std::vector<VocabStep> steps;             /* the whole declared chain, in order */

    int32_t bos = -1, eos = -1, eot = -1, pad = -1, unk = -1, sep = -1;
    bool    add_bos = false, add_eos = false;

    std::string chat_template;

    /* Diagnostics only: what rad-convert thought it was reading. Not baked. */
    std::string source;
};

/* ================================================================== the loaded vocab */

class Vocab {
public:
    int  load(VocabBuild b);          /* takes ownership; RAD_OK or negative */
    bool ready() const { return ready_; }

    uint32_t kind() const { return b_.kind; }
    int32_t  n_tokens() const { return (int32_t)b_.tokens.size(); }

    const std::string& text(int32_t id) const;
    uint8_t            type(int32_t id) const;
    float              score(int32_t id) const;
    bool               is_eog(int32_t id) const;       /* eos, eot, or a declared extra */
    /* Every one of them, by id. The server needs the SPELLINGS: a chat format can fold the turn
     * terminator into its last marker, so the decoder has to render it for the reply parser --
     * and then whatever the parser did not consume has to come back off the content, because a
     * terminator is never something the user asked to read. */
    const std::vector<int32_t>& eog() const { return eog_; }

    /* -1 when the exact text is not a token. */
    int32_t find(std::string_view text) const;
    /* BPE merge rank of the pair, or -1. */
    int32_t merge_rank(std::string_view left, std::string_view right) const;

    int32_t bos() const { return b_.bos; }
    int32_t eos() const { return b_.eos; }
    int32_t eot() const { return b_.eot; }
    int32_t unk() const { return b_.unk; }
    int32_t pad() const { return b_.pad; }
    bool    add_bos() const { return b_.add_bos; }
    bool    add_eos() const { return b_.add_eos; }
    const std::string& chat_template() const { return b_.chat_template; }

    const std::vector<VocabStep>& steps() const { return b_.steps; }
    const std::vector<int32_t>&   special_ids() const { return special_; }
    const VocabBuild&             build() const { return b_; }

    /* Pre-decoded piece for this token id (FF05). */
    const std::string& decoded_piece(int32_t id) const;

    /* The byte a RAD_TT_BYTE token stands for, or -1. Both spellings are recognised: SPM's
     * "<0xF0>" and byte-level BPE's single mapped codepoint. */
    int32_t token_byte(int32_t id) const;

private:
    VocabBuild b_;
    bool       ready_ = false;
    std::unordered_map<std::string, int32_t> text_to_id_;
    std::unordered_map<uint64_t, int32_t>    merge_rank_;   /* (lhs id << 32 | rhs id) -> rank */
    std::vector<int32_t> special_;                          /* ids the pre-splitter must isolate */
    std::vector<int32_t> byte_of_;                          /* id -> byte, or -1 */
    std::vector<std::string> decoded_pieces_;               /* id -> pre-decoded piece (FF05) */
    /* Every token that ends a generation, not just the one the header had room for. Built in
     * load(); see the comment there for why a container's `eos` alone is not the whole set. */
    std::vector<int32_t> eog_;
    /* The chain compiled for encoding: the Split DFAs, the flat token and merge tables, the
     * special-token matcher and the segment cache. Built in load(), shared by every Tokenizer
     * over this vocab and by every copy of it. */
    std::shared_ptr<TokEncoder> enc_;
    friend class Tokenizer;
    friend class Detokenizer;
};

/* ================================================================== encode */

class Tokenizer {
public:
    int init(std::shared_ptr<const Vocab> v);

    /* `add_special` prepends BOS / appends EOS as the vocab declares. `parse_special` lets
     * control tokens in the raw text be recognised as themselves -- which the server wants for a
     * rendered chat template and must NOT have for user content, because otherwise a user can
     * type "<|im_start|>system" and reach the system role. */
    int encode(std::string_view text, std::vector<int32_t>& out,
               bool add_special, bool parse_special) const;

    /* Non-streaming detokenise. The streaming one is Detokenizer below and it is the one the
     * server uses; this exists for tests, logging and the /v1/completions echo path. */
    std::string decode(const std::vector<int32_t>& ids, bool render_special = false) const;

    /* One token's text after the decoder chain. Does not resolve partial UTF-8 -- that is
     * exactly what Detokenizer is for. */
    std::string piece(int32_t id, bool render_special = false) const;

    const Vocab& vocab() const { return *v_; }

    /* THE SEGMENT CACHE. encode() cuts its text at special tokens before anything else touches
     * it, and every piece between two of them is normalised, pre-tokenised and merged on its own:
     * no step of the chain reads across a special token, and none reads anything but the piece
     * (a Metaspace "first" prepend, a ByteLevel prefix space and every normaliser apply to each
     * piece as if it were the whole text). A piece's ids are therefore a function of its bytes
     * and the vocab alone, and a cache from the bytes to the ids returns exactly what encoding
     * them again would. That is what makes a re-sent conversation cost only what is new in it: an
     * agent's history arrives as the same pieces between the same `<|im_start|>` markers, turn
     * after turn.
     *
     * The cache belongs to the vocab, so every Tokenizer over it shares it. It is keyed on the
     * whole piece (the hash only picks the bucket), bounded in bytes with least-recently-used
     * eviction, and locked for concurrent encodes. Pieces under kSegmentCacheMin bytes are not
     * cached: re-encoding them costs less than the lookup. `set_segment_cache(false)` makes this
     * Tokenizer bypass it, which is how the tests and rad-tokcheck compare the two. */
    static constexpr size_t kSegmentCacheMin     = 64;
    static constexpr size_t kSegmentCacheDefault = size_t(64) << 20;
    void set_segment_cache(bool on) { use_cache_ = on; }
    struct CacheStats {
        uint64_t hits = 0, misses = 0;
        size_t   entries = 0, bytes = 0, limit = 0;
    };
    CacheStats cache_stats() const;
    void       set_cache_limit(size_t bytes) const;
    void       clear_cache() const;

private:
    std::shared_ptr<const Vocab> v_;
    bool                         use_cache_ = true;
};

/* ================================================================== streaming decode */

/* Replace every byte that is not part of a well-formed UTF-8 sequence with U+FFFD, and leave
 * everything else alone. A byte-fallback vocab can emit a byte that no continuation completes --
 * `<0xC1>` is a lead byte for an encoding that is overlong and therefore never legal -- and the
 * generated text goes straight into a JSON string, where an invalid byte is not a mojibake
 * character but a 500: nlohmann::json::dump throws type_error.316 on it and the client gets an
 * engine error instead of the model's answer. Sanitising is the only option that keeps the
 * response: the bytes cannot be held back (nothing will complete them) and dropping them silently
 * would hide that the model produced something undecodable. Well-formed input is returned byte for
 * byte, so this is a no-op on every normal completion. */
std::string utf8_sanitize(std::string_view s);


/* Byte-level BPE emits partial UTF-8 sequences -- a single token can be one byte of a
 * three-byte codepoint -- and a stop string can straddle a chunk boundary. Both are buffered
 * until they resolve (spec §12). Emitting either one early is a visible bug: the first puts a
 * replacement character in the user's stream, the second leaks "</to" before the server notices
 * it was "</tool_call>".
 *
 * The contract: push() returns text that is FINAL. It will never be retracted, it is always
 * WELL-FORMED UTF-8, and it is never a prefix of a stop string. Whatever is still ambiguous stays
 * inside until a later push resolves it or flush() gives up on it. A sequence that turns out to be
 * malformed rather than merely unfinished -- a byte-fallback vocab can emit one -- leaves as
 * U+FFFD (utf8_sanitize above) instead of as raw bytes, since the raw bytes cannot be put into a
 * JSON string and would cost the client the whole response. */
class Detokenizer {
public:
    Detokenizer() = default;
    /* `preserved` is the tokens that must reach the caller as TEXT even though they are CONTROL.
     * A tool-calling format is MADE of them -- MiniCPM5 writes its calls with `<function`,
     * `<param`, `</param>`, `</function>`, every one a control token -- so suppressing them the
     * way a control token is normally suppressed hands the reply parser the arguments with the
     * structure removed, and the call comes back to the client as prose. The chat template says
     * which ones it needs (common_chat_params::preserved_tokens); nothing else is unsuppressed. */
    Detokenizer(std::shared_ptr<const Vocab> v, std::vector<std::string> stops = {},
                bool render_special = false, const std::vector<std::string>& preserved = {});

    std::string push(int32_t token);
    std::string flush();

    /* True once a stop string has been seen. The text before it has already been returned; the
     * stop string itself and anything after it never is. */
    bool               stopped() const { return stopped_; }
    const std::string& stop_hit() const { return stop_hit_; }

    /* Everything pushed so far, stop string included. The scheduler wants this to decide whether
     * a request finished on a stop or on a length limit. */
    const std::string& text() const { return all_; }

    void reset();

private:
    std::shared_ptr<const Vocab> v_;
    std::vector<std::string>     stops_;
    bool        render_special_ = false;
    std::unordered_set<int32_t>  preserved_;   /* control ids this stream renders anyway */
    std::string pending_;      /* decoded but not yet safe to emit */
    std::string all_;          /* everything, for the caller's record */
    bool        stopped_ = false;
    std::string stop_hit_;
    size_t      max_stop_ = 0;
    bool        first_ = true;

    size_t safe_prefix() const;   /* how much of pending_ can be released */
};

/* ================================================================== sources and sinks */

/* rad-convert owns the .rad's single string blob; the vocab section only needs to intern into
 * it. Passing the interner rather than a blob keeps the two writers from disagreeing about who
 * owns offset 0. */
struct StringSink {
    virtual ~StringSink() = default;
    virtual rad_stroff intern(std::string_view s) = 0;
};

/* Build a VocabBuild from a HuggingFace tokenizer.json.
 *
 * This is the function spec §12 is about: it INTERPRETS the declared normalizer / pre_tokenizer /
 * decoder chain instead of hashing the file. Returns RAD_OK, or a negative RAD_E_*; in
 * particular RAD_E_UNSUPPORTED when the file declares a component we cannot express as a
 * RadVocabStep, with the component named in the log. Failing there is the point -- a chain we
 * cannot express must not become a chain we silently approximate. */
int parse_tokenizer_json(const char* path, VocabBuild& out);
/* Same, from an in-memory buffer, for a tokenizer.json that arrived some other way. */
int parse_tokenizer_json_buf(std::string_view json, const char* origin, VocabBuild& out);

/* Build a VocabBuild from a GGUF's tokenizer.ggml.* metadata.
 *
 * GGUF has no declared chain -- it has `tokenizer.ggml.pre`, a name llama.cpp derived by hashing
 * the original tokenizer.json. So this reconstructs the chain from that name against a table of
 * the ones we know, and returns RAD_E_UNSUPPORTED for a name we do not, rather than falling back
 * to a default split. That refusal is the whole difference from llama.cpp's behaviour, and it is
 * why converting from the original tokenizer.json is the preferred path. */
int vocab_from_gguf(const GgufFile& g, VocabBuild& out);

/* Serialise into the .rad's vocab section. `section_file_off` is where rad-convert will place
 * the section; every offset written into the RadVocabHeader is absolute from the start of the
 * file, per rad_format.h's rule that "every offset is from the start of the file". */
int vocab_serialize(const VocabBuild& in, StringSink& strings, uint64_t section_file_off,
                    std::vector<uint8_t>& section_out);
/* Read it back. `file_base` is the mmapped .rad; `h` points inside it. */
int vocab_deserialize(const void* file_base, uint64_t file_bytes,
                      const RadVocabHeader* h, VocabBuild& out);

/* Human-readable one-liner for a step, for --debug-graph and for the convert log. */
std::string step_describe(const VocabStep& s);

}  /* namespace rad */
