#include "text/tokenizer.h"
#include "text/tokenizer_bpe.h"
#include "text/tokenizer_nfc.h"
#include "text/tokenizer_regex.h"
#include "text/unicode_norm.h"
#include "llama/src/unicode.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>

namespace rad {

/* ================================================================== byte level
 *
 * GPT-2's byte alphabet, exactly. The printable ASCII range, Latin-1 punctuation and the
 * accented Latin-1 letters map to themselves; the remaining 68 bytes map to U+0100 upward in
 * ascending byte order. Getting this wrong is not a crash -- it is a vocab lookup that misses
 * for one byte in four, which shows up as a 20% longer prompt and nothing else. It is
 * reproduced from llama.cpp's unicode_byte_to_utf8_map(), which is itself HuggingFace's table.
 *
 * We do not call llama.cpp's version because its inverse throws std::out_of_range on an
 * unmapped string, and in this engine errors are values (docs/IMPLEMENTATION.md). */
namespace {

struct ByteLevelAlphabet {
    std::string to_utf8[256];                         /* byte -> its 1-2 byte UTF-8 spelling */
    std::unordered_map<std::string, uint8_t> from;    /* and back */
    uint32_t cpt_of[256];
    int16_t  byte_of_cpt[0x144];                      /* codepoint -> byte, or -1 */

    ByteLevelAlphabet() {
        bool taken[256] = { false };
        for (int16_t& b : byte_of_cpt) b = -1;
        auto give = [&](int b, uint32_t cpt) {
            to_utf8[b] = unicode_cpt_to_utf8(cpt);
            cpt_of[b]  = cpt;
            from[to_utf8[b]] = (uint8_t)b;
            byte_of_cpt[cpt] = (int16_t)b;
            taken[b] = true;
        };
        for (int b = 0x21; b <= 0x7E; ++b) give(b, (uint32_t)b);   /* '!' .. '~'  */
        for (int b = 0xA1; b <= 0xAC; ++b) give(b, (uint32_t)b);   /* U+00A1..AC  */
        for (int b = 0xAE; b <= 0xFF; ++b) give(b, (uint32_t)b);   /* U+00AE..FF  */
        uint32_t n = 0;
        for (int b = 0; b < 256; ++b) if (!taken[b]) give(b, 256 + n++);
    }
};

const ByteLevelAlphabet& alphabet() {
    static const ByteLevelAlphabet a;
    return a;
}

std::string byte_encode(std::string_view raw) {
    const ByteLevelAlphabet& a = alphabet();
    std::string out;
    out.reserve(raw.size() * 2);
    for (unsigned char c : raw) out += a.to_utf8[c];
    return out;
}

/* Inverse. An unmapped codepoint is passed through as its own UTF-8, which is what a
 * user-defined token spelt in real text needs; it cannot be confused with a mapped one because
 * the mapped set is closed. */
std::string byte_decode(const std::string& enc) {
    const ByteLevelAlphabet& a = alphabet();
    std::string out;
    out.reserve(enc.size());
    for (size_t i = 0; i < enc.size(); ) {
        const size_t n = std::min(unicode_len_utf8(enc[i]), enc.size() - i);
        const std::string one = enc.substr(i, n);
        auto it = a.from.find(one);
        if (it != a.from.end()) out += (char)it->second;
        else                    out += one;
        i += n;
    }
    return out;
}

/* The raw bytes a byte-level token stands for, when EVERY character of it is in the alphabet.
 * Only such a token can equal a byte-level word -- mapping puts every byte of a word into the
 * alphabet -- so a table of these by their raw bytes answers every lookup the mapped text would,
 * and the encoder can merge raw bytes and never build the mapped text at all. */
bool byte_decode_strict(const std::string& enc, std::string* out) {
    const ByteLevelAlphabet& a = alphabet();
    out->clear();
    const uint8_t* p = (const uint8_t*)enc.data();
    for (size_t i = 0; i < enc.size(); ) {
        uint32_t c;
        if (p[i] < 0x80)                                          { c = p[i]; i += 1; }
        else if (p[i] >= 0xC2 && p[i] < 0xE0 && i + 1 < enc.size() && (p[i + 1] & 0xC0) == 0x80) {
            c = ((p[i] & 0x1F) << 6) | (p[i + 1] & 0x3F);
            i += 2;
        } else {
            return false;
        }
        if (c >= 0x144 || a.byte_of_cpt[c] < 0) return false;
        *out += (char)a.byte_of_cpt[c];
    }
    return true;
}

constexpr const char* k_metaspace = "\xe2\x96\x81";   /* U+2581 LOWER ONE EIGHTH BLOCK */

void replace_all(std::string& s, std::string_view from, std::string_view to) {
    if (from.empty()) return;
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    for (;;) {
        size_t j = s.find(from, i);
        if (j == std::string::npos) { out.append(s, i, std::string::npos); break; }
        out.append(s, i, j - i);
        out.append(to);
        i = j + from.size();
    }
    s.swap(out);
}

/* The GPT-2 pattern HuggingFace's ByteLevel pre-tokeniser carries inside itself when
 * use_regex is set. */
constexpr const char* k_gpt2_pattern =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

}  /* namespace */

/* ================================================================== the compiled encoder
 *
 * Everything encode() needs that can be decided once per vocab: each Split pattern compiled to
 * its DFA, the token and merge tables flattened into open-addressed arrays keyed the way the merge
 * loop asks, the special-token matcher, and the segment cache. Built by Vocab::load and shared by
 * every Tokenizer over the vocab. Two things in it change afterwards and both are safe to share:
 * the segment cache, which locks, and the memo of which tokens merge back to themselves, whose
 * entries are written once with relaxed atomics. */

namespace {

/* --- special tokens ---------------------------------------------------------------------- */

struct SpecialFrag { int32_t id; uint32_t off, len; };

/* The text is cut at every special token before any other step sees it. WHICH occurrences are
 * cut is decided by priority, not position: longest token first (then lowest id), and a token
 * only claims occurrences that no higher-priority token has already claimed and that do not
 * overlap one of its own claimed to their left. That is the result of splitting the text once
 * per token, longest first, and it is why "<|im_start|>" is never cut in half by a shorter token
 * that is a prefix of it. Finding the occurrences is one pass over the text with a
 * trie of every special token, started only at a byte some token begins with. */
class SpecialMatcher {
public:
    void build(const std::vector<int32_t>& special, const std::vector<std::string>& tokens,
               const std::vector<uint8_t>& types) {
        nodes_.assign(1, Node{});
        for (size_t prio = 0; prio < special.size(); ++prio) {
            const int32_t id = special[prio];
            const std::string& t = tokens[id];
            toks_.push_back({ id, (uint32_t)t.size(),
                              types[id] == RAD_TT_CONTROL || types[id] == RAD_TT_UNKNOWN });
            uint32_t n = 0;
            for (unsigned char c : t) {
                uint32_t next = UINT32_MAX;
                for (const auto& k : nodes_[n].kids) if (k.first == c) { next = k.second; break; }
                if (next == UINT32_MAX) {
                    next = (uint32_t)nodes_.size();
                    nodes_[n].kids.push_back({ c, next });
                    nodes_.push_back(Node{});
                }
                n = next;
            }
            nodes_[n].ends.push_back((uint32_t)prio);
            first_[(unsigned char)t[0]] = true;
        }
        int n_first = 0;
        for (int c = 0; c < 256; ++c) if (first_[c]) { ++n_first; only_first_ = c; }
        if (n_first != 1) only_first_ = -1;
    }

    void split(std::string_view text, bool parse_special, std::vector<SpecialFrag>& out) const {
        out.clear();
        const uint8_t* s = (const uint8_t*)text.data();
        const size_t n = text.size();
        struct Occ { uint32_t prio, at; };
        std::vector<Occ> occ;
        auto walk = [&](size_t i) {
            uint32_t node = 0;
            for (size_t j = i; j < n; ++j) {
                uint32_t next = UINT32_MAX;
                for (const auto& k : nodes_[node].kids) if (k.first == s[j]) { next = k.second; break; }
                if (next == UINT32_MAX) return;
                node = next;
                for (uint32_t prio : nodes_[node].ends)
                    if (parse_special || !toks_[prio].control) occ.push_back({ prio, (uint32_t)i });
            }
        };
        if (only_first_ >= 0) {
            for (const uint8_t* p = s; (p = (const uint8_t*)std::memchr(p, only_first_, (s + n) - p)); ++p)
                walk((size_t)(p - s));
        } else if (!toks_.empty()) {
            for (size_t i = 0; i < n; ++i) if (first_[s[i]]) walk(i);
        }
        if (occ.empty()) {
            out.push_back({ -1, 0, (uint32_t)n });
            return;
        }
        std::sort(occ.begin(), occ.end(), [](const Occ& a, const Occ& b) {
            return a.prio != b.prio ? a.prio < b.prio : a.at < b.at;
        });
        std::map<uint32_t, std::pair<uint32_t, int32_t>> taken;     /* start -> (end, id) */
        for (const Occ& o : occ) {
            const uint32_t b = o.at, e = o.at + toks_[o.prio].len;
            auto it = taken.lower_bound(b);
            if (it != taken.end() && it->first < e) continue;
            if (it != taken.begin() && std::prev(it)->second.first > b) continue;
            taken.emplace(b, std::make_pair(e, toks_[o.prio].id));
        }
        uint32_t pos = 0;
        for (const auto& t : taken) {
            if (t.first > pos) out.push_back({ -1, pos, t.first - pos });
            out.push_back({ t.second.second, t.first, t.second.first - t.first });
            pos = t.second.first;
        }
        if (pos < n) out.push_back({ -1, pos, (uint32_t)n - pos });
    }

private:
    struct Tok { int32_t id; uint32_t len; bool control; };
    struct Node {
        std::vector<std::pair<uint8_t, uint32_t>> kids;
        std::vector<uint32_t> ends;            /* priorities of the tokens that end here */
    };
    std::vector<Tok>  toks_;                   /* in priority order */
    std::vector<Node> nodes_;
    bool first_[256] = {};
    int  only_first_ = -1;
};

/* --- the segment cache ------------------------------------------------------------------- */

/* Segment bytes -> ids. See Tokenizer's header for why a hit is exact. Entries are keyed by the
 * whole segment: the hash picks the bucket and the bytes decide the match. The bound is on bytes
 * held (key, ids and bookkeeping), and eviction is least recently used.
 *
 * The lock covers the index and nothing else. An entry is built before it is locked in, and a hit
 * takes a reference to the entry's ids and copies them after the lock is released, so a
 * 100,000-id segment copied into one admission never holds up another. */
class SegmentCache {
public:
    using Ids = std::shared_ptr<const std::vector<int32_t>>;

    bool lookup(std::string_view key, uint64_t h, std::vector<int32_t>& out) {
        Ids ids;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto range = index_.equal_range(h);
            for (auto it = range.first; it != range.second; ++it) {
                const Entry& e = *it->second;
                if (e.key.size() != key.size() ||
                    std::memcmp(e.key.data(), key.data(), key.size()) != 0)
                    continue;
                ids = e.ids;
                lru_.splice(lru_.begin(), lru_, it->second);
                break;
            }
            ++(ids ? hits_ : misses_);
        }
        if (!ids) return false;
        out.insert(out.end(), ids->begin(), ids->end());
        return true;
    }

    void insert(std::string_view key, uint64_t h, const int32_t* ids, size_t n) {
        const size_t cost = key.size() + n * sizeof(int32_t) + kOverhead;
        if (cost > limit_.load(std::memory_order_relaxed)) return;
        std::list<Entry> node;
        node.push_back(Entry{ std::string(key), std::make_shared<const std::vector<int32_t>>(ids, ids + n),
                              h, cost });
        std::list<Entry> evicted;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto range = index_.equal_range(h);
            for (auto it = range.first; it != range.second; ++it) {
                const Entry& e = *it->second;
                if (e.key.size() == key.size() && std::memcmp(e.key.data(), key.data(), key.size()) == 0)
                    return;                              /* another encode got here first */
            }
            lru_.splice(lru_.begin(), node);
            index_.emplace(h, lru_.begin());
            bytes_ += cost;
            evict(evicted);
        }
    }

    void set_limit(size_t bytes) {
        std::list<Entry> evicted;
        std::lock_guard<std::mutex> lk(mu_);
        limit_.store(bytes, std::memory_order_relaxed);
        evict(evicted);
    }

    void clear() {
        std::list<Entry> gone;
        std::lock_guard<std::mutex> lk(mu_);
        gone.swap(lru_);
        index_.clear();
        bytes_ = 0;
        hits_ = misses_ = 0;
    }

    Tokenizer::CacheStats stats() {
        std::lock_guard<std::mutex> lk(mu_);
        Tokenizer::CacheStats s;
        s.hits = hits_;
        s.misses = misses_;
        s.entries = lru_.size();
        s.bytes = bytes_;
        s.limit = limit_.load(std::memory_order_relaxed);
        return s;
    }

private:
    struct Entry {
        std::string key;
        Ids         ids;
        uint64_t    hash;
        size_t      cost;
    };
    /* A list node, a hash node, the shared vector's control block and the heap headers, rounded
     * up. */
    static constexpr size_t kOverhead = 160;

    /* Moves the least recently used entries into `out` until the bound holds; the caller frees
     * them after it unlocks. */
    void evict(std::list<Entry>& out) {
        while (bytes_ > limit_.load(std::memory_order_relaxed) && !lru_.empty()) {
            auto last = std::prev(lru_.end());
            auto range = index_.equal_range(last->hash);
            for (auto it = range.first; it != range.second; ++it) {
                if (it->second == last) { index_.erase(it); break; }
            }
            bytes_ -= last->cost;
            out.splice(out.begin(), lru_, last);
        }
    }

    std::mutex mu_;
    std::list<Entry> lru_;                                    /* front is most recent */
    std::unordered_multimap<uint64_t, std::list<Entry>::iterator> index_;
    size_t   bytes_ = 0;
    std::atomic<size_t> limit_{ Tokenizer::kSegmentCacheDefault };
    uint64_t hits_ = 0, misses_ = 0;
};

}  /* namespace */

struct TokEncoder {
    struct Step {
        VocabStep                 step;
        std::unique_ptr<TokRegex> re;        /* Split, a regex Replace, ByteLevel with use_regex */
    };
    std::vector<Step> norm, pre;
    uint32_t kind = RAD_TOK_BPE;
    bool     bytelevel = false;              /* the chain has a ByteLevel pre-tokeniser */
    bool     byte_fallback = false;
    /* The ByteLevel step at this index is the last step that rewrites text, so the mapping is
     * never materialised: the BPE merges raw bytes against `raw` instead. -1 when that does not
     * hold (no ByteLevel, or a step after it that reads the mapped text). */
    int      defer_bytelevel = -1;

    TokTable   text;                         /* token text -> lowest id */
    TokTable   raw;                          /* raw bytes a byte-level token stands for -> id */
    int32_t    byte_sym[256];                /* raw byte -> id of its byte-level character */
    int32_t    byte_tok[256];                /* raw byte -> id of "<0xNN>" */
    MergeTable merges;
    std::vector<uint8_t> types;
    int32_t    unk = -1;

    /* Per token: whether its own text merges back to exactly itself -- 0 not yet known, 1 yes,
     * 2 no. Most words of real text ARE a token, and the merge loop's answer for a token's text
     * never changes, so once it has been worked out for one word it is looked up for every later
     * one, by every thread, for the life of the vocab. Relaxed atomics: two threads that race
     * store the same value. */
    std::unique_ptr<std::atomic<uint8_t>[]> self_merges;

    SpecialMatcher       specials;
    mutable SegmentCache cache;

    int build(const VocabBuild& b, const std::vector<int32_t>& special);
};

int TokEncoder::build(const VocabBuild& b, const std::vector<int32_t>& special) {
    kind = b.kind;
    types = b.types;
    unk = b.unk;

    auto compile = [&](const std::string& pat, const char* what, std::unique_ptr<TokRegex>* re) {
        std::string why;
        if (TokRegex::compile(pat, re, &why) < 0) {
            RAD_ERR("vocab: the %s pattern '%s' is outside the supported regex dialect: %s. "
                    "There is no fallback matcher; a pattern that does not compile does not load.",
                    what, pat.c_str(), why.c_str());
            return RAD_E_UNSUPPORTED;
        }
        return RAD_OK;
    };

    for (const VocabStep& s : b.steps) {
        if (step_is_norm(s.kind)) {
            Step st{ s, nullptr };
            if (s.kind == RAD_NORM_REPLACE && (s.flags & 1)) RAD_TRY(compile(s.arg0, "Replace", &st.re));
            norm.push_back(std::move(st));
        } else if (step_is_pre(s.kind)) {
            Step st{ s, nullptr };
            if (s.kind == RAD_PRE_SPLIT) RAD_TRY(compile(s.arg0, "Split", &st.re));
            if (s.kind == RAD_PRE_BYTELEVEL) {
                bytelevel = true;
                if (s.flags & 2) RAD_TRY(compile(k_gpt2_pattern, "ByteLevel", &st.re));
            }
            if (s.kind == RAD_PRE_BYTE_FALLBACK) byte_fallback = true;
            pre.push_back(std::move(st));
        }
    }
    if (kind == RAD_TOK_BPE) {
        for (size_t k = pre.size(); k-- > 0; ) {
            const uint32_t sk = pre[k].step.kind;
            if (sk == RAD_PRE_BYTE_FALLBACK) continue;
            if (sk == RAD_PRE_BYTELEVEL) defer_bytelevel = (int)k;
            break;
        }
    }

    const size_t n = b.tokens.size();
    text.reserve(n);
    for (size_t i = 0; i < n; ++i) text.insert(b.tokens[i], (int32_t)i);

    const ByteLevelAlphabet& a = alphabet();
    for (int c = 0; c < 256; ++c) {
        static const char* hex = "0123456789ABCDEF";
        const char buf[7] = { '<', '0', 'x', hex[c >> 4], hex[c & 15], '>', 0 };
        byte_tok[c] = text.find(buf);
        byte_sym[c] = text.find(a.to_utf8[c]);
    }
    if (defer_bytelevel >= 0) {
        raw.reserve(n);
        std::string r;
        for (size_t i = 0; i < n; ++i)
            if (!b.tokens[i].empty() && byte_decode_strict(b.tokens[i], &r)) raw.insert(r, (int32_t)i);
    }

    /* Keyed by the pair exactly as the vocab lists it, the lowest rank winning a duplicate. A
     * symbol's id is always the LOWEST id spelling its text, so a merge listed against a higher
     * id with the same text is never looked up. */
    merges.reserve(b.merges.size());
    std::string cat;
    for (size_t r = 0; r < b.merges.size(); ++r) {
        const uint32_t l = b.merges[r].first, rr = b.merges[r].second;
        if (l >= n || rr >= n) continue;
        cat.assign(b.tokens[l]);
        cat += b.tokens[rr];
        merges.insert((int32_t)l, (int32_t)rr, (uint32_t)r, text.find(cat));
    }

    specials.build(special, b.tokens, b.types);
    self_merges.reset(new std::atomic<uint8_t>[n]());
    /* A byte-level BPE never looks text up by its mapped spelling once `raw` exists. */
    if (defer_bytelevel >= 0) text = TokTable();
    return RAD_OK;
}

/* ================================================================== Vocab */

int Vocab::load(VocabBuild b) {
    b_ = std::move(b);
    ready_ = false;
    text_to_id_.clear();
    merge_rank_.clear();
    special_.clear();
    byte_of_.clear();

    if (b_.tokens.empty()) {
        RAD_ERR("vocab: no tokens");
        return RAD_E_FORMAT;
    }
    if (b_.kind != RAD_TOK_BPE && b_.kind != RAD_TOK_UNIGRAM && b_.kind != RAD_TOK_WORDPIECE) {
        RAD_ERR("vocab: tokenizer kind %u is not implemented (BPE, unigram and wordpiece are)",
                b_.kind);
        return RAD_E_UNSUPPORTED;
    }
    if (b_.types.empty()) b_.types.assign(b_.tokens.size(), (uint8_t)RAD_TT_NORMAL);
    if (b_.types.size() != b_.tokens.size()) {
        RAD_ERR("vocab: %zu tokens but %zu token types", b_.tokens.size(), b_.types.size());
        return RAD_E_FORMAT;
    }
    if (b_.kind == RAD_TOK_UNIGRAM && b_.scores.size() != b_.tokens.size()) {
        RAD_ERR("vocab: unigram needs a score per token; have %zu for %zu tokens",
                b_.scores.size(), b_.tokens.size());
        return RAD_E_FORMAT;
    }

    /* A SPECIAL ID IS EMITTED AS IT STANDS -- bos and eos by encode, unk by every encoder on text
     * no piece covers -- so one that names no token puts an id outside the vocabulary into a
     * sequence. Every source of a vocabulary arrives here, and a GGUF or a container carries these
     * ids as plain integers nothing else checks, so one out of range is treated as absent, which
     * every reader of these fields already handles, and said so. */
    {
        const struct { const char* name; int32_t* id; } specials[] = {
            { "bos", &b_.bos }, { "eos", &b_.eos }, { "eot", &b_.eot },
            { "pad", &b_.pad }, { "unk", &b_.unk }, { "sep", &b_.sep },
        };
        for (const auto& s : specials) {
            if (*s.id >= -1 && (int64_t)*s.id < (int64_t)b_.tokens.size()) continue;
            RAD_WARN("vocab: the %s id %d names no token in a %zu-token vocabulary; treating "
                     "the vocabulary as having none", s.name, *s.id, b_.tokens.size());
            *s.id = -1;
        }
    }

    text_to_id_.reserve(b_.tokens.size() * 2);
    for (size_t i = 0; i < b_.tokens.size(); ++i) {
        /* First spelling wins. A duplicate is a producer bug, but it is not ours to fail on:
         * llama.cpp's own vocabs contain them, and the lower id is the one the merge table and
         * every published token count refer to. */
        text_to_id_.emplace(b_.tokens[i], (int32_t)i);
    }

    /* Merge ranks are stored as an id pair rather than a string pair: the string pair would be
     * a 248K-entry map of concatenated strings hashed on every bigram push, and the bigram push
     * is the tokeniser's inner loop. */
    merge_rank_.reserve(b_.merges.size() * 2);
    for (size_t r = 0; r < b_.merges.size(); ++r) {
        const uint64_t key = ((uint64_t)b_.merges[r].first << 32) | (uint64_t)b_.merges[r].second;
        merge_rank_.emplace(key, (int32_t)r);
    }

    byte_of_.assign(b_.tokens.size(), -1);
    const ByteLevelAlphabet& a = alphabet();
    for (size_t i = 0; i < b_.tokens.size(); ++i) {
        const std::string& t = b_.tokens[i];
        if (b_.types[i] == RAD_TT_BYTE) {
            /* Two spellings in the wild: SPM's "<0xF0>" and byte-level BPE's single mapped
             * codepoint. Both are byte tokens and both have to decode to one byte. */
            if (t.size() == 6 && t.compare(0, 3, "<0x") == 0 && t.back() == '>') {
                const int hi = t[3], lo = t[4];
                auto hex = [](int c) { return c >= '0' && c <= '9' ? c - '0'
                                            : c >= 'a' && c <= 'f' ? c - 'a' + 10
                                            : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
                if (hex(hi) >= 0 && hex(lo) >= 0) byte_of_[i] = hex(hi) * 16 + hex(lo);
            } else {
                auto it = a.from.find(t);
                if (it != a.from.end()) byte_of_[i] = it->second;
            }
        }
        if (b_.types[i] == RAD_TT_CONTROL || b_.types[i] == RAD_TT_UNKNOWN ||
            b_.types[i] == RAD_TT_USER_DEFINED) {
            if (!t.empty()) special_.push_back((int32_t)i);
        }
    }

    /* Longest first. Special-token partitioning scans the text once per special token, and a
     * shorter token that is a prefix of a longer one would otherwise cut the longer one in half
     * -- "<|im_start|>" losing to "<|im" is the classic version of this bug. */
    std::sort(special_.begin(), special_.end(), [&](int32_t x, int32_t y) {
        const size_t nx = b_.tokens[x].size(), ny = b_.tokens[y].size();
        return nx != ny ? nx > ny : x < y;
    });

    /* ------------------------------------------------------------------ end of generation
     *
     * A MODEL ENDS ITS TURN ON MORE THAN ONE TOKEN AND THE CONTAINER CARRIES ROOM FOR ONE. The
     * header has `eos` and `eot` and nothing else, but a Qwen container declares
     * eos_token_id: [<|im_end|>, <|endoftext|>] and will end a turn on either -- and its
     * `<|endoftext|>` lands in `pad_id`, which nothing treats as terminal. The engine then
     * generates straight past the end of the turn into a hallucinated next one, which does not
     * read as a bug: it reads as a model that rambles.
     *
     * So the set is completed here, from the ONE piece of evidence in the container that is not a
     * guess -- a CONTROL token whose spelling is a known end-of-generation marker. Both halves of
     * that matter. Restricting it to control tokens means ordinary text can never become
     * terminal, and matching an explicit list of spellings rather than a pattern means a token
     * named `<|end_of_thinking|>` does not silently end generations. This is the same list, and
     * the same reasoning, that llama.cpp's vocabulary loader applies.
     *
     * WHAT IS ADDED IS LOGGED. A token that silently gained the power to end every generation is
     * exactly the kind of change that should be visible in a startup log, so it is. */
    eog_.clear();
    if (b_.eos >= 0) eog_.push_back(b_.eos);
    if (b_.eot >= 0 && b_.eot != b_.eos) eog_.push_back(b_.eot);
    {
        static const char* const kEogNames[] = {
            "<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>",
            "<|end|>", "<|EOT|>", "<|eom_id|>", "<|return|>", "<eos>", "</s>",
        };
        std::string added;
        for (int32_t id : special_) {
            if (b_.types[id] != RAD_TT_CONTROL) continue;
            if (std::find(eog_.begin(), eog_.end(), id) != eog_.end()) continue;
            bool known = false;
            for (const char* n : kEogNames) if (b_.tokens[id] == n) { known = true; break; }
            if (!known) continue;
            eog_.push_back(id);
            added += (added.empty() ? "" : " ") + b_.tokens[id] + "=" + std::to_string(id);
        }
        if (!added.empty())
            RAD_INFO("vocab: end-of-generation also on %s (a control token the container declares "
                     "neither as eos nor eot; without this a turn ending on it runs on)",
                     added.c_str());
    }

    /* The encoder: the chain compiled and the tables flattened. A pattern the regex compiler
     * refuses refuses the vocab, with the construct named -- there is no slower matcher to hand
     * it to instead. */
    enc_ = std::make_shared<TokEncoder>();
    RAD_TRY(enc_->build(b_, special_));

    /* Precompute decoded pieces (FF05) */
    decoded_pieces_.resize(b_.tokens.size());
    for (size_t i = 0; i < b_.tokens.size(); ++i) {
        const int32_t b = (i < byte_of_.size()) ? byte_of_[i] : -1;
        if (b >= 0) {
            decoded_pieces_[i] = std::string(1, (char)b);
            continue;
        }
        std::string s = b_.tokens[i];
        const uint8_t t = b_.types[i];
        if (t == RAD_TT_CONTROL || t == RAD_TT_UNKNOWN || t == RAD_TT_USER_DEFINED) {
            decoded_pieces_[i] = std::move(s);
            continue;
        }
        for (const VocabStep& st : b_.steps) {
            if (!step_is_dec(st.kind)) continue;
            switch (st.kind) {
                case RAD_DEC_BYTELEVEL: s = byte_decode(s); break;
                case RAD_DEC_METASPACE:
                    replace_all(s, st.arg0.empty() ? std::string(k_metaspace) : st.arg0, " ");
                    break;
                case RAD_DEC_REPLACE: replace_all(s, st.arg0, st.arg1); break;
                case RAD_DEC_STRIP: {
                    const std::string c = st.arg0.empty() ? std::string(" ") : st.arg0;
                    int32_t start = (int32_t)(st.iarg & 0xFFFFFFFF);
                    int32_t stop  = (int32_t)(st.iarg >> 32);
                    while (start-- > 0 && s.size() >= c.size() && s.compare(0, c.size(), c) == 0)
                        s.erase(0, c.size());
                    while (stop-- > 0 && s.size() >= c.size() &&
                           s.compare(s.size() - c.size(), c.size(), c) == 0)
                        s.erase(s.size() - c.size());
                    break;
                }
                default: break;
            }
        }
        decoded_pieces_[i] = std::move(s);
    }

    ready_ = true;
    RAD_DEBUG("vocab: %zu tokens, %zu merges, %zu special, %zu eog, %zu chain steps, kind %u",
              b_.tokens.size(), b_.merges.size(), special_.size(), eog_.size(), b_.steps.size(),
              b_.kind);
    return RAD_OK;
}

const std::string& Vocab::decoded_piece(int32_t id) const {
    static const std::string empty;
    if (id < 0 || id >= (int32_t)decoded_pieces_.size()) return empty;
    return decoded_pieces_[(size_t)id];
}

const std::string& Vocab::text(int32_t id) const {
    static const std::string empty;
    if (id < 0 || id >= (int32_t)b_.tokens.size()) return empty;
    return b_.tokens[id];
}

uint8_t Vocab::type(int32_t id) const {
    if (id < 0 || id >= (int32_t)b_.types.size()) return RAD_TT_UNKNOWN;
    return b_.types[id];
}

float Vocab::score(int32_t id) const {
    if (id < 0 || id >= (int32_t)b_.scores.size()) return 0.0f;
    return b_.scores[id];
}


bool Vocab::is_eog(int32_t id) const {
    return id >= 0 && std::find(eog_.begin(), eog_.end(), id) != eog_.end();
}

int32_t Vocab::find(std::string_view t) const {
    auto it = text_to_id_.find(std::string(t));
    return it == text_to_id_.end() ? -1 : it->second;
}

int32_t Vocab::merge_rank(std::string_view left, std::string_view right) const {
    const int32_t l = find(left), r = find(right);
    if (l < 0 || r < 0) return -1;
    auto it = merge_rank_.find(((uint64_t)(uint32_t)l << 32) | (uint32_t)r);
    return it == merge_rank_.end() ? -1 : it->second;
}

int32_t Vocab::token_byte(int32_t id) const {
    if (id < 0 || id >= (int32_t)byte_of_.size()) return -1;
    return byte_of_[id];
}

/* ================================================================== the step chain */

std::string step_describe(const VocabStep& s) {
    switch (s.kind) {
        case RAD_NORM_NFC:  return "NFC";
        case RAD_NORM_NFKC: return "NFKC";
        case RAD_NORM_NFD:  return "NFD";
        case RAD_NORM_NFKD: return "NFKD";
        case RAD_NORM_LOWERCASE: return "Lowercase";
        case RAD_NORM_STRIP:
            return fmt("Strip(left=%d right=%d)", (s.flags & 1) ? 1 : 0, (s.flags & 2) ? 1 : 0);
        case RAD_NORM_REPLACE:
            return fmt("Replace(%s '%s' -> '%s')", (s.flags & 1) ? "regex" : "literal",
                       s.arg0.c_str(), s.arg1.c_str());
        case RAD_NORM_PREPEND: return fmt("Prepend('%s')", s.arg0.c_str());
        case RAD_PRE_BYTELEVEL:
            return fmt("ByteLevel(add_prefix_space=%d use_regex=%d)",
                       (s.flags & 1) ? 1 : 0, (s.flags & 2) ? 1 : 0);
        case RAD_PRE_SPLIT: {
            /* What the pattern compiled to is part of the description: the size of its DFA, or
             * the construct that refused it. */
            std::unique_ptr<TokRegex> re;
            std::string why;
            const std::string dfa = TokRegex::compile(s.arg0, &re, &why) < 0
                ? "[refused: " + why + "]"
                : fmt("[dfa: %zu states, %zu classes]", re->n_states(), re->n_classes());
            return fmt("Split(%s behavior=%lld invert=%d) %s",
                       s.arg0.c_str(), (long long)s.iarg, (s.flags & 1) ? 1 : 0, dfa.c_str());
        }
        case RAD_PRE_METASPACE:
            return fmt("Metaspace('%s' prepend=%lld split=%d)", s.arg0.c_str(),
                       (long long)s.iarg, (s.flags & 1) ? 1 : 0);
        case RAD_PRE_WHITESPACE:   return "Whitespace";
        case RAD_PRE_PUNCTUATION:  return fmt("Punctuation(behavior=%lld)", (long long)s.iarg);
        case RAD_PRE_DIGITS:
            return fmt("Digits(individual=%d)", (s.flags & 1) ? 1 : 0);
        case RAD_PRE_BYTE_FALLBACK: return "ByteFallback";
        case RAD_DEC_BYTELEVEL:     return "decode ByteLevel";
        case RAD_DEC_METASPACE:
            return fmt("decode Metaspace('%s' prepend=%lld)", s.arg0.c_str(), (long long)s.iarg);
        case RAD_DEC_REPLACE:
            return fmt("decode Replace('%s' -> '%s')", s.arg0.c_str(), s.arg1.c_str());
        case RAD_DEC_STRIP:
            return fmt("decode Strip('%s' start=%lld stop=%lld)", s.arg0.c_str(),
                       (long long)(s.iarg & 0xFFFFFFFF), (long long)(s.iarg >> 32));
        case RAD_DEC_FUSE:          return "decode Fuse";
        case RAD_DEC_BYTE_FALLBACK: return "decode ByteFallback";
        default: return fmt("<unknown step %u>", s.kind);
    }
}

namespace {

/* --- normalisers ------------------------------------------------------------------------- */

void apply_norm(std::string& text, const TokEncoder::Step& st) {
    const VocabStep& s = st.step;
    switch (s.kind) {
        case RAD_NORM_NFC: tok_nfc(text); break;
        case RAD_NORM_NFKC: text = unorm_utf8(text, true,  true);  break;
        case RAD_NORM_NFD:  text = unorm_utf8(text, false, false); break;
        case RAD_NORM_NFKD: text = unorm_utf8(text, true,  false); break;
        case RAD_NORM_LOWERCASE: {
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text);
            std::string out;
            out.reserve(text.size());
            for (uint32_t c : cpts) out += unicode_cpt_to_utf8(unicode_tolower(c));
            text.swap(out);
            break;
        }
        case RAD_NORM_STRIP: {
            /* Whitespace by the unicode definition, not isspace(): a non-breaking space in a
             * pasted prompt is whitespace to HuggingFace and would not be to isspace(). */
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text);
            size_t a = 0, b = cpts.size();
            if (s.flags & 1) while (a < b && unicode_cpt_flags_from_cpt(cpts[a]).is_whitespace) ++a;
            if (s.flags & 2) while (b > a && unicode_cpt_flags_from_cpt(cpts[b-1]).is_whitespace) --b;
            std::string out;
            for (size_t i = a; i < b; ++i) out += unicode_cpt_to_utf8(cpts[i]);
            text.swap(out);
            break;
        }
        case RAD_NORM_REPLACE:
            if (s.flags & 1) {
                /* HuggingFace's Replace: every match of the compiled pattern becomes the content,
                 * verbatim -- the content is text, not a substitution template. */
                std::string canon, out;
                std::string_view v = text;
                if (!utf8_is_canonical(v)) { utf8_canonicalize(v, canon); v = canon; }
                tok_replace(*st.re, v, s.arg1, out);
                text.swap(out);
            } else {
                replace_all(text, s.arg0, s.arg1);
            }
            break;
        case RAD_NORM_PREPEND: text = s.arg0 + text; break;
        default:
            RAD_WARN("tokenizer: normaliser step %u is not implemented; text passed through",
                     s.kind);
            break;
    }
}

/* --- pre-tokenisers ----------------------------------------------------------------------
 *
 * The pieces are spans of one buffer rather than a string each: a 500 KB prompt is 150,000
 * pieces, and a string each would be 150,000 allocations. A step that has to rewrite text
 * appends the new text to the buffer (or builds a new buffer) and points its spans there. */

struct Pieces {
    std::string          buf;
    std::vector<TokSpan> spans;
    bool                 canonical = false;   /* every piece is known to be canonical UTF-8 */
    std::string_view at(const TokSpan& s) const {
        return std::string_view(buf).substr(s.begin, s.end - s.begin);
    }
};

struct EncodeScratch {
    Pieces               p;
    std::string          canon;
    std::vector<TokSpan> next, found, match;
    std::vector<BpeSym>  syms;
};

/* One Split step over every piece. The pieces it produces are canonical UTF-8: a piece that is
 * not is decoded and re-encoded first -- invalid bytes to U+FFFD, overlong forms shortest -- piece
 * by piece, by the rules of the decoder the other steps use (utf8_canonicalize). */
void apply_split(Pieces& p, const TokRegex& re, int behavior, bool invert, EncodeScratch& sc) {
    if (!p.canonical) p.canonical = utf8_is_canonical(p.buf);
    const bool whole = p.canonical;
    sc.next.clear();
    for (size_t k = 0; k < p.spans.size(); ++k) {
        const TokSpan s = p.spans[k];
        if (s.end == s.begin) continue;
        uint32_t base = s.begin;
        std::string_view v = p.at(s);
        if (!whole && !utf8_is_canonical(v)) {
            sc.canon.clear();
            utf8_canonicalize(v, sc.canon);
            base = (uint32_t)p.buf.size();
            p.buf += sc.canon;
            v = std::string_view(p.buf).substr(base, sc.canon.size());
        }
        sc.found.clear();
        tok_split(re, v, behavior, invert, sc.found, sc.match);
        for (const TokSpan& f : sc.found) sc.next.push_back({ base + f.begin, base + f.end });
    }
    p.spans.swap(sc.next);
    p.canonical = true;
}

void rebuild(Pieces& p, const std::vector<std::string>& words) {
    std::string buf;
    std::vector<TokSpan> spans;
    for (const std::string& w : words) {
        spans.push_back({ (uint32_t)buf.size(), (uint32_t)(buf.size() + w.size()) });
        buf += w;
    }
    p.buf.swap(buf);
    p.spans.swap(spans);
}

std::vector<std::string> words_of(const Pieces& p) {
    std::vector<std::string> w;
    w.reserve(p.spans.size());
    for (const TokSpan& s : p.spans) w.emplace_back(p.at(s));
    return w;
}

void apply_bytelevel(Pieces& p, const TokEncoder::Step& st, bool defer, EncodeScratch& sc) {
    const bool add_prefix_space = (st.step.flags & 1) != 0;
    if (add_prefix_space && !p.spans.empty() && p.spans[0].end > p.spans[0].begin &&
        p.buf[p.spans[0].begin] != ' ') {
        const std::string first = " " + std::string(p.at(p.spans[0]));
        p.spans[0] = { (uint32_t)p.buf.size(), (uint32_t)(p.buf.size() + first.size()) };
        p.buf += first;
    }
    /* HuggingFace's ByteLevel carries the GPT-2 pattern inside it. rad-convert normally splits
     * that out into an explicit Split step so the chain reads as what it does, but a chain that
     * declares it inline still has to work. */
    if (st.re) apply_split(p, *st.re, TOK_SPLIT_ISOLATED, false, sc);
    if (defer) return;
    std::vector<std::string> w = words_of(p);
    for (std::string& x : w) x = byte_encode(x);
    rebuild(p, w);
}

void apply_metaspace(std::vector<std::string>& words, const VocabStep& s) {
    const std::string rep = s.arg0.empty() ? std::string(k_metaspace) : s.arg0;
    const int64_t prepend = s.iarg;            /* 0 never, 1 first, 2 always */
    const bool    split   = (s.flags & 1) != 0;

    for (size_t i = 0; i < words.size(); ++i) {
        std::string& w = words[i];
        if (prepend == 2 || (prepend == 1 && i == 0)) {
            if (w.empty() || w.compare(0, 1, " ") != 0) w.insert(0, " ");
        }
        replace_all(w, " ", rep);
    }
    if (!split) return;
    std::vector<std::string> out;
    for (const std::string& w : words) {
        size_t i = 0;
        while (i < w.size()) {
            size_t j = w.find(rep, i + (w.compare(i, rep.size(), rep) == 0 ? rep.size() : 0));
            if (j == std::string::npos) { out.push_back(w.substr(i)); break; }
            out.push_back(w.substr(i, j - i));
            i = j;
        }
        if (w.empty()) out.push_back(w);
    }
    words.swap(out);
}

/* Whitespace pre-tokeniser: HuggingFace's is the regex \w+|[^\w\s]+ with Isolated behaviour,
 * written here over the category flags. */
void apply_whitespace(std::vector<std::string>& words) {
    std::vector<std::string> out;
    for (const std::string& w : words) {
        const std::vector<uint32_t> cpts = unicode_cpts_from_utf8(w);
        size_t i = 0;
        while (i < cpts.size()) {
            const auto f = unicode_cpt_flags_from_cpt(cpts[i]);
            if (f.is_whitespace) { ++i; continue; }
            const bool word = f.is_letter || f.is_number || cpts[i] == '_';
            size_t j = i;
            while (j < cpts.size()) {
                const auto g = unicode_cpt_flags_from_cpt(cpts[j]);
                if (g.is_whitespace) break;
                const bool wj = g.is_letter || g.is_number || cpts[j] == '_';
                if (wj != word) break;
                ++j;
            }
            std::string piece;
            for (size_t k = i; k < j; ++k) piece += unicode_cpt_to_utf8(cpts[k]);
            out.push_back(std::move(piece));
            i = j;
        }
    }
    words.swap(out);
}

void apply_punctuation(std::vector<std::string>& words, const VocabStep& s) {
    const int64_t behavior = s.iarg;
    std::vector<std::string> out;
    for (const std::string& w : words) {
        const std::vector<uint32_t> cpts = unicode_cpts_from_utf8(w);
        std::string acc;
        for (uint32_t c : cpts) {
            const auto f = unicode_cpt_flags_from_cpt(c);
            const bool punct = f.is_punctuation || f.is_symbol;
            if (punct) {
                if (!acc.empty()) { out.push_back(acc); acc.clear(); }
                if (behavior != TOK_SPLIT_REMOVED) out.push_back(unicode_cpt_to_utf8(c));
            } else {
                acc += unicode_cpt_to_utf8(c);
            }
        }
        if (!acc.empty()) out.push_back(acc);
    }
    words.swap(out);
}

void apply_digits(std::vector<std::string>& words, const VocabStep& s) {
    const bool individual = (s.flags & 1) != 0;
    std::vector<std::string> out;
    for (const std::string& w : words) {
        const std::vector<uint32_t> cpts = unicode_cpts_from_utf8(w);
        std::string acc;
        bool acc_digit = false;
        for (uint32_t c : cpts) {
            const bool d = unicode_cpt_flags_from_cpt(c).is_number;
            if (acc.empty()) { acc_digit = d; acc += unicode_cpt_to_utf8(c); continue; }
            if (d != acc_digit || (d && individual)) {
                out.push_back(acc);
                acc.clear();
                acc_digit = d;
            }
            acc += unicode_cpt_to_utf8(c);
        }
        if (!acc.empty()) out.push_back(acc);
    }
    words.swap(out);
}

/* --- models ------------------------------------------------------------------------------ */

/* Word -> ids within one encode() call. Prose and code repeat their words heavily, so most words
 * are merged once per call and then copied; the ids depend on nothing but the word, so the copy
 * is exact. */
class WordCache {
public:
    static constexpr size_t kMaxKey = 128;

    bool find(std::string_view w, uint64_t h, std::vector<int32_t>& out) const {
        if (slots_.empty()) return false;
        for (size_t i = h & mask_;; i = (i + 1) & mask_) {
            const Slot& s = slots_[i];
            if (s.klen == kEmpty) return false;
            if (s.hash == h && s.klen == w.size() &&
                std::memcmp(keys_.data() + s.koff, w.data(), w.size()) == 0) {
                out.insert(out.end(), ids_.begin() + s.ioff, ids_.begin() + s.ioff + s.ilen);
                return true;
            }
        }
    }

    void insert(std::string_view w, uint64_t h, const int32_t* ids, size_t n) {
        if ((n_ + 1) * 2 > slots_.size()) grow();
        for (size_t i = h & mask_;; i = (i + 1) & mask_) {
            Slot& s = slots_[i];
            if (s.klen != kEmpty) continue;
            s = { h, (uint32_t)keys_.size(), (uint32_t)w.size(), (uint32_t)ids_.size(), (uint32_t)n };
            keys_.append(w.data(), w.size());
            ids_.insert(ids_.end(), ids, ids + n);
            ++n_;
            return;
        }
    }

private:
    static constexpr uint32_t kEmpty = UINT32_MAX;
    struct Slot { uint64_t hash; uint32_t koff, klen, ioff, ilen; };
    std::vector<Slot>    slots_;
    std::string          keys_;
    std::vector<int32_t> ids_;
    size_t n_ = 0, mask_ = 0;

    void grow() {
        std::vector<Slot> old;
        old.swap(slots_);
        const size_t cap = old.empty() ? 1024 : old.size() * 2;
        slots_.assign(cap, Slot{ 0, 0, kEmpty, 0, 0 });
        mask_ = cap - 1;
        for (const Slot& s : old) {
            if (s.klen == kEmpty) continue;
            size_t i = s.hash & mask_;
            while (slots_[i].klen != kEmpty) i = (i + 1) & mask_;
            slots_[i] = s;
        }
    }
};

/* One word through the merge. The ids a symbol carries are the LOWEST id spelling its text, which
 * is what the merge table is keyed on and what the word finally emits. */
void bpe_word(const TokEncoder& e, std::string_view w, std::vector<int32_t>& out,
              EncodeScratch& sc) {
    const bool raw = e.defer_bytelevel >= 0;
    const TokTable& table = raw ? e.raw : e.text;

    /* A user-defined token is taken whole rather than re-split. Added tokens are exact strings
     * the vocabulary promises to round-trip; decomposing one into codepoints and merging back
     * can land on a different sequence, because the merge table was never fitted over them.
     *
     * ORDINARY VOCABULARY ENTRIES STILL GO THROUGH THE MERGE LOOP, which is HuggingFace's
     * behaviour at `ignore_merges: false` and is what every checkpoint in scope declares. A
     * vocabulary setting it TRUE wants the same short-circuit for any word that is already a
     * token, and this engine does not honour the flag: core/text/tokenizer_json.cpp only warns
     * when it is set, and the vocabulary has nowhere to carry it. Such a model would tokenise
     * differently here than it does in transformers, mostly on rare words. Supporting it is a
     * vocabulary field, not a change to this loop. */
    const int32_t whole = table.find(w);
    if (whole >= 0 && e.types[whole] == RAD_TT_USER_DEFINED) {
        out.push_back(whole);
        return;
    }
    const uint8_t known = whole >= 0 ? e.self_merges[whole].load(std::memory_order_relaxed) : 0;
    if (known == 1) {
        out.push_back(whole);
        return;
    }

    std::vector<BpeSym>& syms = sc.syms;
    syms.clear();
    const size_t n = w.size();
    if (raw) {
        for (size_t i = 0; i < n; ++i)
            syms.push_back({ (uint32_t)i, 1, e.byte_sym[(unsigned char)w[i]] });
    } else {
        for (size_t i = 0; i < n; ) {
            const size_t len = std::min(unicode_len_utf8(w[i]), n - i);
            syms.push_back({ (uint32_t)i, (uint32_t)len, table.find(w.substr(i, len)) });
            i += len;
        }
    }
    bpe_merge(syms, e.merges);
    if (whole >= 0 && known == 0) {
        const bool itself = syms.size() == 1 && syms[0].id == whole;
        e.self_merges[whole].store(itself ? 1 : 2, std::memory_order_relaxed);
    }

    for (const BpeSym& s : syms) {
        if (s.id >= 0) { out.push_back(s.id); continue; }
        const std::string_view str = w.substr(s.start, s.len);

        /* Miss. Decompose into single-byte tokens; how a byte is SPELT depends on the chain.
         * Under ByteLevel every byte is already its own mapped codepoint, so the piece splits at
         * codepoint boundaries; under byte fallback it is "<0xNN>". */
        if (raw) {
            for (unsigned char c : str) {
                const int32_t b = e.byte_sym[c];
                if (b >= 0) out.push_back(b);
                else RAD_WARN("tokenizer: no token for byte-level piece '%s'",
                              alphabet().to_utf8[c].c_str());
            }
        } else if (e.bytelevel) {
            for (size_t k = 0; k < str.size(); ) {
                const size_t m = std::min(unicode_len_utf8(str[k]), str.size() - k);
                const int32_t b = e.text.find(str.substr(k, m));
                if (b >= 0) out.push_back(b);
                else RAD_WARN("tokenizer: no token for byte-level piece '%s'",
                              std::string(str.substr(k, m)).c_str());
                k += m;
            }
        } else if (e.byte_fallback) {
            for (unsigned char c : str) {
                const int32_t b = e.byte_tok[c];
                if (b >= 0) out.push_back(b);
                else if (e.unk >= 0) out.push_back(e.unk);
            }
        } else if (e.unk >= 0) {
            out.push_back(e.unk);
        } else {
            RAD_WARN("tokenizer: piece '%s' is not in the vocab and the chain declares neither "
                     "ByteLevel nor ByteFallback; it is dropped", std::string(str).c_str());
        }
    }
}

void bpe(const TokEncoder& e, const Pieces& p, std::vector<int32_t>& out, EncodeScratch& sc,
         WordCache& cache) {
    const bool raw = e.defer_bytelevel >= 0;
    for (const TokSpan& s : p.spans) {
        const std::string_view w = p.at(s);
        if (w.empty()) continue;
        /* A one-byte word of a byte-level vocab is the token of that byte: the whole-word lookup
         * and the merge both land on byte_sym, and a missing one is the fallback's warning. */
        if (raw && w.size() == 1 && e.byte_sym[(unsigned char)w[0]] >= 0) {
            out.push_back(e.byte_sym[(unsigned char)w[0]]);
            continue;
        }
        if (w.size() > WordCache::kMaxKey) { bpe_word(e, w, out, sc); continue; }
        const uint64_t h = tok_hash(w);
        if (cache.find(w, h, out)) continue;
        const size_t start = out.size();
        bpe_word(e, w, out, sc);
        cache.insert(w, h, out.data() + start, out.size() - start);
    }
}

/* Unigram, by Viterbi over a prefix trie -- SentencePiece's algorithm, in shape from
 * llama.cpp's llm_tokenizer_ugm_session. Scores are summed in double precision because the
 * reference implementation does and the argmax genuinely differs in float on long inputs. */
void unigram(const Vocab& v, const TokEncoder& e, const std::string& text,
             std::vector<int32_t>& out) {
    if (text.empty()) return;
    const size_t n = text.size();

    struct Best { int32_t id; size_t from; double score; };
    std::vector<Best> best(n + 1, { v.unk(), 0, -DBL_MAX });
    best[0] = { v.unk(), 0, 0.0 };

    /* The unknown-token penalty: SentencePiece uses (min score - 10). Recomputing it here rather
     * than storing it keeps the .rad from carrying a derived number that could disagree. */
    double unk_penalty = 0.0;
    if (!v.build().scores.empty()) {
        unk_penalty = *std::min_element(v.build().scores.begin(), v.build().scores.end()) - 10.0;
    }

    for (size_t i = 0; i < n; ) {
        const size_t cp_len = std::min(unicode_len_utf8(text[i]), n - i);
        bool matched_whole_cp = false;
        /* Longest-prefix scan. A trie would beat this on a large vocab; the substring lookup is
         * O(len) with a bounded len because no vocab entry is longer than the longest token, and
         * that bound is what keeps it linear. */
        const size_t max_len = std::min(n - i, (size_t)256);
        for (size_t l = 1; l <= max_len; ++l) {
            const int32_t id = e.text.find(std::string_view(text.data() + i, l));
            if (id < 0) continue;
            if (l == cp_len) matched_whole_cp = true;
            const double sc = v.type(id) == RAD_TT_USER_DEFINED ? 0.0 : (double)v.score(id);
            const double cand = best[i].score + sc;
            if (cand > best[i + l].score) best[i + l] = { id, i, cand };
        }
        if (!matched_whole_cp) {
            const double cand = best[i].score + unk_penalty;
            if (cand > best[i + cp_len].score) best[i + cp_len] = { v.unk(), i, cand };
        }
        i += cp_len;
    }

    /* A VOCABULARY WITH NO UNK TOKEN HAS NO ID FOR TEXT NO PIECE COVERS, and the Viterbi above
     * still routes such text through the unk transition -- whose id is then -1, which is not a
     * token and must never reach the output. The span is spelled as byte tokens when the chain
     * declares byte fallback and dropped otherwise, as bpe() does with a piece it cannot match.
     * Built back to front like the rest of the backtrack, so the bytes go in reversed. */
    const int32_t unk = v.unk();
    const bool byte_fallback = unk < 0 && e.byte_fallback;

    const size_t start = out.size();
    size_t dropped = 0;
    bool prev_unk = false;
    for (size_t pos = n; ; ) {
        const Best& b = best[pos];
        const bool is_unk = (b.id == unk);
        if (is_unk && unk < 0) {
            for (size_t k = pos; k-- > b.from; ) {
                const unsigned char c = (unsigned char)text[k];
                const int32_t id = byte_fallback ? e.byte_tok[c] : -1;
                if (id >= 0) out.push_back(id);
                else         ++dropped;
            }
        } else if (!(prev_unk && is_unk)) {
            out.push_back(b.id);
        }
        if (b.from == 0) break;
        prev_unk = is_unk;
        pos = b.from;
    }
    std::reverse(out.begin() + start, out.end());
    if (dropped)
        RAD_WARN("tokenizer: %zu byte(s) of this text have no piece in the vocab, which has no unk "
                 "token and %s; they are dropped", dropped,
                 byte_fallback ? "lacks a byte token for them" : "declares no byte fallback");
}

/* WordPiece: greedy longest-match-first with a continuation prefix. Kept because BERT-family
 * embedding models are a real deployment and they are all WordPiece; it is 30 lines. */
void wordpiece(const Vocab& v, const TokEncoder& e, const Pieces& p, std::vector<int32_t>& out) {
    for (const TokSpan& s : p.spans) {
        const std::string w(p.at(s));
        if (w.empty()) continue;
        size_t i = 0;
        std::vector<int32_t> piece;
        bool ok = true;
        while (i < w.size()) {
            size_t j = w.size();
            int32_t id = -1;
            for (; j > i; --j) {
                std::string sub = w.substr(i, j - i);
                if (i > 0) sub = "##" + sub;
                id = e.text.find(sub);
                if (id >= 0) break;
            }
            if (id < 0) { ok = false; break; }
            piece.push_back(id);
            i = j;
        }
        if (ok) out.insert(out.end(), piece.begin(), piece.end());
        else if (v.unk() >= 0) out.push_back(v.unk());
    }
}

/* One segment -- the text between two special tokens -- through the whole chain. */
int encode_segment(const Vocab& v, const TokEncoder& e, std::string_view seg,
                   std::vector<int32_t>& out, EncodeScratch& sc, WordCache& words) {
    Pieces& p = sc.p;
    p.buf.assign(seg.data(), seg.size());
    for (const TokEncoder::Step& st : e.norm) apply_norm(p.buf, st);
    if (p.buf.size() > (size_t)(UINT32_MAX / 8)) {
        RAD_ERR("tokenizer: a %zu-byte segment is past what one encode can index", p.buf.size());
        return RAD_E_INVAL;
    }
    p.spans.assign(1, TokSpan{ 0, (uint32_t)p.buf.size() });
    p.canonical = false;

    for (size_t k = 0; k < e.pre.size(); ++k) {
        const TokEncoder::Step& st = e.pre[k];
        switch (st.step.kind) {
            case RAD_PRE_SPLIT:
                apply_split(p, *st.re, (int)st.step.iarg, (st.step.flags & 1) != 0, sc);
                break;
            case RAD_PRE_BYTELEVEL:
                apply_bytelevel(p, st, (int)k == e.defer_bytelevel, sc);
                break;
            case RAD_PRE_METASPACE: {
                std::vector<std::string> w = words_of(p);
                apply_metaspace(w, st.step);
                rebuild(p, w);
                break;
            }
            case RAD_PRE_WHITESPACE: {
                std::vector<std::string> w = words_of(p);
                apply_whitespace(w);
                rebuild(p, w);
                break;
            }
            case RAD_PRE_PUNCTUATION: {
                std::vector<std::string> w = words_of(p);
                apply_punctuation(w, st.step);
                rebuild(p, w);
                break;
            }
            case RAD_PRE_DIGITS: {
                std::vector<std::string> w = words_of(p);
                apply_digits(w, st.step);
                rebuild(p, w);
                break;
            }
            case RAD_PRE_BYTE_FALLBACK:
                /* A declaration about what the MODEL does with an unknown piece, not a split.
                 * The fallback itself lives in bpe_word, where the miss happens. */
                break;
            default:
                RAD_ERR("tokenizer: pre-tokeniser step %u is not implemented", st.step.kind);
                return RAD_E_UNSUPPORTED;
        }
    }

    switch (e.kind) {
        case RAD_TOK_BPE:       bpe(e, p, out, sc, words); break;
        case RAD_TOK_WORDPIECE: wordpiece(v, e, p, out); break;
        case RAD_TOK_UNIGRAM: {
            /* Unigram runs Viterbi over the whole normalised segment; the pre-tokeniser's job
             * there is only to insert the metaspace, so the pieces are rejoined. */
            std::string joined;
            for (const TokSpan& s : p.spans) joined += p.at(s);
            unigram(v, e, joined, out);
            break;
        }
        default:
            RAD_ERR("tokenizer: kind %u has no encoder", e.kind);
            return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

}  /* namespace */

/* ================================================================== encode */

int Tokenizer::init(std::shared_ptr<const Vocab> v) {
    if (!v || !v->ready()) {
        RAD_ERR("tokenizer: vocab is not loaded");
        return RAD_E_STATE;
    }
    v_ = std::move(v);
    if (log_level() >= Log::Debug) {
        for (const VocabStep& s : v_->steps())
            RAD_DEBUG("tokenizer: step %s", step_describe(s).c_str());
    }
    return RAD_OK;
}

int Tokenizer::encode(std::string_view text, std::vector<int32_t>& out,
                      bool add_special, bool parse_special) const {
    if (!v_ || !v_->ready()) return RAD_E_STATE;
    const TokEncoder& e = *v_->enc_;

    /* Special-token partitioning. The text is cut into (raw, token, raw, token, ...) around any
     * declared special token that appears literally, before any other step touches it, because a
     * special token is not something the pre-tokeniser is allowed to split.
     *
     * `parse_special == false` means a control token typed by a user is ordinary text. That is
     * not a nicety: without it, user content containing "<|im_start|>system" reaches the model
     * as an actual role switch. */
    std::vector<SpecialFrag> frags;
    e.specials.split(text, parse_special, frags);

    /* A text that already opens with BOS gets no second one. A client that renders its own chat
     * template writes the template's BOS into the prompt, and a doubled BOS is a sequence the
     * model never saw in training. */
    if (add_special && v_->add_bos() && v_->bos() >= 0) {
        int32_t first = -1;
        for (const SpecialFrag& f : frags) {
            if (f.id < 0 && f.len == 0) continue;
            first = f.id;
            break;
        }
        if (first != v_->bos()) out.push_back(v_->bos());
    }

    EncodeScratch sc;
    WordCache words;
    for (const SpecialFrag& f : frags) {
        if (f.id >= 0) { out.push_back(f.id); continue; }
        if (f.len == 0) continue;
        const std::string_view seg = text.substr(f.off, f.len);
        if (!use_cache_ || seg.size() < kSegmentCacheMin) {
            RAD_TRY(encode_segment(*v_, e, seg, out, sc, words));
            continue;
        }
        const uint64_t h = tok_hash(seg);
        if (e.cache.lookup(seg, h, out)) continue;
        const size_t start = out.size();
        RAD_TRY(encode_segment(*v_, e, seg, out, sc, words));
        e.cache.insert(seg, h, out.data() + start, out.size() - start);
    }

    if (add_special && v_->add_eos() && v_->eos() >= 0) out.push_back(v_->eos());
    return RAD_OK;
}

Tokenizer::CacheStats Tokenizer::cache_stats() const {
    return v_ && v_->enc_ ? v_->enc_->cache.stats() : CacheStats{};
}

void Tokenizer::set_cache_limit(size_t bytes) const {
    if (v_ && v_->enc_) v_->enc_->cache.set_limit(bytes);
}

void Tokenizer::clear_cache() const {
    if (v_ && v_->enc_) v_->enc_->cache.clear();
}

/* ================================================================== decode */

std::string Tokenizer::piece(int32_t id, bool render_special) const {
    if (!v_ || !v_->ready()) return {};
    if (id < 0 || id >= v_->n_tokens()) return {};

    const uint8_t t = v_->type(id);
    if (!render_special && (t == RAD_TT_CONTROL || t == RAD_TT_UNKNOWN)) return {};

    return v_->decoded_piece(id);
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool render_special) const {
    std::string out;
    for (int32_t id : ids) out += piece(id, render_special);
    return out;
}

/* ================================================================== Detokenizer */

Detokenizer::Detokenizer(std::shared_ptr<const Vocab> v, std::vector<std::string> stops,
                         bool render_special, const std::vector<std::string>& preserved)
    : v_(std::move(v)), stops_(std::move(stops)), render_special_(render_special) {
    /* RESOLVED BY TOKENISING THE MARKER, NOT BY LOOKING IT UP. What the chat format hands over
     * are the MARKERS its parser scans for -- `<function name="`, `</param>`, `<think>` -- and a
     * marker is not in general a token: MiniCPM5 spells that first one as the control token
     * `<function` followed by ordinary text. Looking the whole marker up finds nothing and
     * preserves nothing. Tokenise each marker with parse_special on and take every CONTROL token
     * it is built from; those are exactly the ones the decoder would otherwise drop, and no
     * other control token is unsuppressed by this. */
    if (v_ && v_->ready() && !preserved.empty()) {
        Tokenizer t;
        if (t.init(v_) >= 0) {
            std::vector<int32_t> ids;
            for (const std::string& s : preserved) {
                if (s.empty()) continue;
                ids.clear();
                if (t.encode(s, ids, /*add_special=*/false, /*parse_special=*/true) < 0) continue;
                for (int32_t id : ids) {
                    /* THE TERMINATOR IS PRESERVED TOO, and it has to be. A derived format folds
                     * the turn terminator into its last marker -- MiniCPM5's argument-value
                     * suffix comes out as `</param></function><|im_end|>` -- so a decoder that
                     * drops `<|im_end|>` hands the parser a string its grammar can never
                     * complete, and every tool call comes back as prose. The parser consumes it
                     * on a match; core/server strips it on a miss. */
                    if (v_->type(id) == RAD_TT_CONTROL) preserved_.insert(id);
                }
            }
        }
    }
    /* Empty stops would make every position a match and hold the whole stream. */
    stops_.erase(std::remove_if(stops_.begin(), stops_.end(),
                                [](const std::string& s) { return s.empty(); }),
                 stops_.end());
    for (const std::string& s : stops_) max_stop_ = std::max(max_stop_, s.size());
}

void Detokenizer::reset() {
    pending_.clear();
    all_.clear();
    stopped_ = false;
    stop_hit_.clear();
    first_ = true;
}

/* How many bytes of pending_ are final.
 *
 * Two separate reasons to hold bytes back and they compose, so the answer is the smaller of the
 * two cut points:
 *   1. a trailing INCOMPLETE UTF-8 sequence. Byte-level BPE hands out one byte at a time, so a
 *      three-byte CJK codepoint arrives across three pushes. Emitting a lone continuation byte
 *      puts U+FFFD in the user's stream and there is no way to take it back.
 *   2. a trailing PROPER PREFIX of a stop string. "</tool_call>" arrives as "</", "tool", "_call",
 *      ">"; emitting "</" and then discovering the stop means the client already rendered it. */
size_t Detokenizer::safe_prefix() const {
    size_t cut = pending_.size();

    /* (1) walk back at most 3 bytes to find the start of a trailing partial sequence. */
    size_t i = pending_.size();
    size_t back = 0;
    while (i > 0 && back < 4) {
        --i; ++back;
        const unsigned char c = (unsigned char)pending_[i];
        if ((c & 0xC0) == 0x80) continue;              /* continuation, keep walking back */
        const size_t need = unicode_len_utf8((char)c);
        if (need > pending_.size() - i) cut = i;       /* the sequence is not all here yet */
        break;
    }

    /* (2) the longest suffix of pending_ that is a proper prefix of some stop string. */
    if (!stops_.empty()) {
        const size_t limit = std::min(pending_.size(), max_stop_ - 1);
        for (size_t n = limit; n > 0; --n) {
            const char* tail = pending_.data() + pending_.size() - n;
            bool is_prefix = false;
            for (const std::string& s : stops_) {
                if (s.size() > n && std::memcmp(s.data(), tail, n) == 0) { is_prefix = true; break; }
            }
            if (is_prefix) { cut = std::min(cut, pending_.size() - n); break; }
        }
    }
    return cut;
}

/* WHATWG's replacement policy: one U+FFFD per maximal subpart that cannot begin a well-formed
 * sequence, which is what browsers and every other inference server do. The alternative -- one
 * U+FFFD per bad byte -- turns a truncated three-byte character into three replacement
 * characters, which reads as three lost characters rather than the one that was actually lost. */
std::string utf8_sanitize(std::string_view s) {
    /* The common case is a string that is already valid, and it must not be copied byte by byte. */
    std::string out;
    size_t i = 0, good = 0;                  /* [good, i) is verified and not yet appended */
    const size_t n = s.size();
    auto flush_good = [&] { out.append(s.data() + good, i - good); good = i; };

    while (i < n) {
        const unsigned char c = (unsigned char)s[i];
        if (c < 0x80) { ++i; continue; }

        /* Length, and the range the SECOND byte is restricted to. The second-byte bounds are what
         * reject overlong encodings, surrogates and anything past U+10FFFF without decoding. */
        size_t len; unsigned char lo, hi;
        if      (c >= 0xC2 && c <= 0xDF) { len = 2; lo = 0x80; hi = 0xBF; }
        else if (c == 0xE0)              { len = 3; lo = 0xA0; hi = 0xBF; }
        else if (c >= 0xE1 && c <= 0xEC) { len = 3; lo = 0x80; hi = 0xBF; }
        else if (c == 0xED)              { len = 3; lo = 0x80; hi = 0x9F; }  /* no surrogates */
        else if (c >= 0xEE && c <= 0xEF) { len = 3; lo = 0x80; hi = 0xBF; }
        else if (c == 0xF0)              { len = 4; lo = 0x90; hi = 0xBF; }
        else if (c >= 0xF1 && c <= 0xF3) { len = 4; lo = 0x80; hi = 0xBF; }
        else if (c == 0xF4)              { len = 4; lo = 0x80; hi = 0x8F; }  /* <= U+10FFFF */
        else                             { len = 0; lo = 0; hi = 0; }        /* 0x80-0xC1, 0xF5+ */

        size_t k = 1;
        if (len) {
            for (; k < len && i + k < n; ++k) {
                const unsigned char d = (unsigned char)s[i + k];
                const unsigned char l = (k == 1) ? lo : 0x80, h = (k == 1) ? hi : 0xBF;
                if (d < l || d > h) break;
            }
        }
        if (len && k == len) { i += len; continue; }

        flush_good();
        out += "\xEF\xBF\xBD";               /* U+FFFD */
        i += k;                              /* the maximal subpart, never zero */
        good = i;
    }
    flush_good();
    return out;
}

std::string Detokenizer::push(int32_t token) {
    if (!v_ || !v_->ready() || stopped_) return {};

    const uint8_t t = v_->type(token);
    const bool render = render_special_ || preserved_.count(token) != 0;
    if (!render && (t == RAD_TT_CONTROL || t == RAD_TT_UNKNOWN)) return {};

    std::string p = v_->decoded_piece(token);

    /* HuggingFace's decoders strip the leading space that Metaspace's prepend added. Only the
     * first piece of a stream can carry it, which is why this is a member flag and not a
     * property of the piece. */
    if (first_) {
        first_ = false;
        for (const VocabStep& s : v_->steps()) {
            if (s.kind == RAD_DEC_METASPACE && s.iarg != 0 && !p.empty() && p[0] == ' ') {
                p.erase(0, 1);
                break;
            }
        }
    }

    pending_ += p;

    /* A complete stop string inside pending_ ends the stream. Everything before it is emitted;
     * the stop string and anything after it never is. */
    if (!stops_.empty()) {
        size_t best = std::string::npos;
        const std::string* which = nullptr;
        for (const std::string& s : stops_) {
            const size_t m = pending_.find(s);
            if (m != std::string::npos && m < best) { best = m; which = &s; }
        }
        if (which) {
            std::string out = pending_.substr(0, best);
            pending_.clear();
            stopped_ = true;
            stop_hit_ = *which;
            return utf8_sanitize(out);
        }
    }

    const size_t cut = safe_prefix();
    if (cut == 0) return {};
    std::string out = pending_.substr(0, cut);
    pending_.erase(0, cut);
    return utf8_sanitize(out);
}

std::string Detokenizer::flush() {
    /* End of stream: whatever is held back will never resolve, so it goes out. An incomplete
     * UTF-8 tail here means the model stopped mid-codepoint, which is a real thing a length limit
     * does; it leaves as one U+FFFD rather than as the raw bytes, because the raw bytes do not
     * survive being put into a JSON string and the client would get an error in place of the
     * whole answer. */
    std::string out;
    out.swap(pending_);
    return utf8_sanitize(out);
}

}  /* namespace rad */
