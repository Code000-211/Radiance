/* core/server/adapters.h -- bind the real components onto the server's interfaces.
 *
 * iface.h knows nothing about core/sched or core/text, so the HTTP layer can be built and tested
 * against fakes. This file joins the two sides and is included only by the translation unit that
 * wires the engine (core/engine.cpp); nothing in core/server/ includes it.
 *
 *   rad::server::SchedulerBridge  sched(&engine_scheduler);
 *   rad::server::TokenizerBridge  tok(vocab);
 *   rad::server::Deps d; d.sched = &sched; d.tok = &tok; ...
 *
 * Mismatches between what the server wants and what the components provide are resolved here
 * rather than as special cases inside either side.
 */
#pragma once
#include "iface.h"
#include "mem/kvtier.h"
#include "sink.h"

#if !defined(RAD_SERVER_NO_ENGINE_BRIDGE) && \
    __has_include("sched/scheduler.h") && __has_include("text/tokenizer.h")
#define RAD_SERVER_HAVE_ENGINE_BRIDGE 1

#include "sample/gbnf.h"
#include "sample/grammar.h"
#include "sample/sampler.h"
#include "sched/scheduler.h"
#include "rad_builder.h"
#include "text/chat.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"
#include "text/tokenizer.h"
#include "mm/processor.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <iterator>
#include <memory>

namespace rad {
namespace server {

/* ================================================================== scheduler */

/* The scheduler's snapshot in the server's terms (SchedulerBridge::metrics). */
inline SchedMetrics bridge_metrics(const rad::SchedMetrics& m) {
    SchedMetrics o;
    o.waiting = m.queue_depth;
    o.running = m.running;
    o.preempted = m.preempted_waiting;

    /* KV utilisation is the maximum over groups, not the mean: a hybrid model's attention
     * and linear-state groups have different geometries (spec §7.2) and exhausting either
     * one preempts.
     *
     * Token capacity is not a sum either. Paged groups share one token axis: each context
     * token takes a slot in every paged group (e.g. Qwen4-Exp's `kv_attn` and `kv_qsa_bkey`),
     * so summing overcounts by the number of paged groups. The token figures are the binding
     * group's: the paged group nearest full, which decides the next preemption. A windowed group
     * never binds: it recycles blocks below its window, so its fill bounds lookback, not how much
     * context fits (a drafter's window sits full for the server's whole life). */
    o.kv_util = 0.0;
    int bind = -1;
    for (int i = 0; i < m.n_kv_groups; ++i) {
        if (m.kv[i].block_tokens <= 0) continue;     /* a linear state has no token axis */
        if (m.kv[i].window) continue;
        if (bind < 0 || m.kv[i].utilisation > m.kv[bind].utilisation) bind = i;
    }
    for (int i = 0; i < m.n_kv_groups; ++i)
        o.kv_util = std::max(o.kv_util, (double)m.kv[i].utilisation);
    if (bind >= 0) {
        const int64_t bt = m.kv[bind].block_tokens;
        o.kv_tokens_total = m.kv[bind].blocks_total * bt;
        o.kv_tokens_used  = (m.kv[bind].blocks_total - m.kv[bind].blocks_free) * bt;
    }

    /* Bytes, unlike tokens, do sum: each group has its own extent of VRAM, so every group
     * (recurrent ones included) is added. */
    for (int i = 0; i < m.n_kv_groups; ++i) {
        o.kv_bytes_total += m.kv[i].bytes_total;
        o.kv_bytes_used  += m.kv[i].bytes_used;
        o.kv_bytes_carved += m.kv[i].bytes_carved;
        if (m.kv[i].block_tokens <= 0 || m.kv[i].window) continue;
        const int64_t t = m.kv[i].blocks_total * m.kv[i].block_tokens;
        if (o.kv_tokens_max == 0 || t < o.kv_tokens_max) o.kv_tokens_max = t;
    }

    /* vLLM's gpu_prefix_cache_hit_rate means attention caching. The linear-checkpoint hit
     * rate has a different granularity (spec §7.3) and is reported separately, not averaged
     * in. */
    o.prefix_hit_rate = m.attn_hit_rate;
    o.prefix_evictions = m.prefix_evictions;
    /* What only the cache holds (SchedMetrics::prefix_held_tokens), clamped to the pool's
     * used figure: the two are counted at different moments in a step, and cached > used
     * would show a negative live share. */
    o.kv_tokens_cached = std::min(m.prefix_held_tokens, o.kv_tokens_used);
    o.prefix_ckpt_evictions = m.prefix_ckpt_evictions;
    o.prefix_ckpt_retired = m.prefix_ckpt_retired;
    o.prefix_ckpt_lru = m.prefix_ckpt_lru;
    o.prefix_ckpt_refused = m.prefix_ckpt_refused;
    o.hit_cut_no_snapshot = m.hit_cut_no_snapshot;
    o.hit_cut_window = m.hit_cut_window;
    o.hit_cut_adopt = m.hit_cut_adopt;
    o.hit_cut_tokens = m.hit_cut_tokens;
    o.prefix_expired = m.prefix_expired;
    o.prefix_ckpt_expired = m.prefix_ckpt_expired;

    o.prefill_tps = m.prefill_tps;
    o.decode_tps = m.decode_tps;
    o.draft_acceptance = m.draft_acceptance;
    o.prefill_tokens = m.prefill_tokens;   o.decode_tokens = m.decode_tokens;
    o.draft_tokens = m.draft_tokens;       o.draft_accepted = m.draft_accepted;
    o.draft_refills = m.draft_refills;     o.draft_refill_tokens = m.draft_refill_tokens;
    o.linear_hit_rate = m.linear_hit_rate;
    o.attn_cached_tokens = m.attn_cached_tokens;
    o.linear_cached_tokens = m.linear_cached_tokens;
    o.checkpoints_written = m.checkpoints_written;
    o.draft_depth_seen = m.draft_depth_seen;
    if (o.draft_depth_seen > SchedMetrics::MAX_DRAFT_POS)
        o.draft_depth_seen = SchedMetrics::MAX_DRAFT_POS;
    for (int j = 0; j < o.draft_depth_seen; ++j) {
        o.draft_pos_reached [j] = m.draft_pos_reached [j];
        o.draft_pos_accepted[j] = m.draft_pos_accepted[j];
    }
    o.draft_adaptive = m.draft_adaptive;
    o.draft_depth_adaptive = m.draft_depth_adaptive;
    for (size_t d = 0;
         d < std::size(m.draft_verify_depth) && d < std::size(o.draft_verify_depth); ++d)
        o.draft_verify_depth[d] = m.draft_verify_depth[d];
    /* The observation ledger: three row counts and one cycle count, renamed to keep the unit in
     * the name. */
    o.draft_obs_rows_attempted   = m.draft_obs_attempted;
    o.draft_obs_rows_applied     = m.draft_obs_applied;
    o.draft_obs_rows_skipped     = m.draft_obs_skipped;
    o.draft_obs_cycles_discarded = m.draft_obs_discarded;
    o.preemptions = m.preemptions;
    o.total_requests = m.admitted;
    o.steps = m.steps;
    o.steps_prefill = m.steps_prefill;  o.steps_mixed = m.steps_mixed;
    o.steps_decode = m.steps_decode;
    o.steps_ahead = m.steps_ahead;
    o.prefill_chunks_capped = m.prefill_chunks_capped;
    o.steps_held = m.steps_held;
    o.decode_rows_unspeculated = m.decode_rows_unspeculated;
    o.step_ns = m.step_ns;  o.step_ns_decode = m.step_ns_decode;
    o.step_ns_decode_best = m.step_ns_decode_best;
    o.step_ns_decode_recent = m.step_ns_decode_recent;
    return o;
}

class SchedulerBridge final : public IScheduler {
public:
    /* `sampler` is the rank-0 sampler; optional only for callers with no engine. Every rank
     * declares the same chain against the same hierarchy, so rank 0 answers for all.
     * `tiers` is rank 0's IdleTiers, passed even when no tier can move anything: it holds the
     * session table either way (without --prefix-cache-mib every session is in VRAM). Null only
     * with no engine. */
    explicit SchedulerBridge(Scheduler* s, const Sampler* sampler = nullptr,
                             const IdleTiers* tiers = nullptr)
        : s_(s), sampler_(sampler), tiers_(tiers) {}

    int admit(const SamplingParams& sp, std::string* why) const override {
        if (!sampler_) return RAD_OK;
        return sampler_->check_request(sp, why);
    }

    uint64_t submit(Request&& r) override {
        std::vector<Request> one;
        one.push_back(std::move(r));
        std::vector<uint64_t> ids;
        submit_all(one, &ids);
        return ids[0];
    }

    /* The server allocates request ids (Scheduler::add refuses duplicates and mints none), so
     * the id exists before submission and a connection dropped mid-submission can cancel it.
     * A refused request gets id 0 and a failed sink: it never entered the queue, so nothing
     * else would finish the sink, and the HTTP thread waiting on it would hang. */
    void submit_all(std::vector<Request>& rs, std::vector<uint64_t>* ids) override {
        const size_t n = rs.size();
        std::vector<std::unique_ptr<Request>> ps(n);
        std::vector<void*> sinks(n);
        std::vector<int> rc(n, RAD_OK);
        ids->assign(n, 0);
        for (size_t i = 0; i < n; ++i) {
            ps[i] = std::make_unique<Request>(std::move(rs[i]));
            ps[i]->id = next_.fetch_add(1, std::memory_order_relaxed);
            (*ids)[i] = ps[i]->id;
            sinks[i] = ps[i]->sink;
        }
        s_->add_all(ps.data(), n, rc.data());
        for (size_t i = 0; i < n; ++i) {
            if (rc[i] >= 0) continue;
            RAD_WARN("server: scheduler refused request %llu: %s",
                     (unsigned long long)(*ids)[i], rad_strerror(rc[i]));
            (*ids)[i] = 0;
            /* A stopped engine has already failed the sink with its own reason. */
            Sink* k = static_cast<Sink*>(sinks[i]);
            if (k && !k->done()) k->fail(rc[i], "the scheduler refused the request");
        }
    }

    void cancel(uint64_t id) override { s_->cancel(id); }
    void reap(uint64_t id) override { s_->reap(id); }

    int64_t reset_prefix_cache() override { return s_->reset_prefix_cache(); }

    /* The stored conversations. The tiers keep monotonic timestamps; idle and age are converted
     * to "seconds ago" here, so only the monotonic clock is authoritative. */
    TierCounters tier_counters() const override {
        TierCounters c;
        if (!tiers_) return c;
        const IdleTiers::Stats& s = tiers_->stats();
        c.to_host = s.to_host; c.dropped = s.dropped; c.promoted = s.promoted;
        c.host_drops = s.host_drops; c.failures = s.failures;
        c.held_back = s.held_back;
        c.host_bytes = s.host_bytes;
        c.restores = s.restores; c.restore_ns = s.restore_ns;
        c.flushes = s.flushes; c.flush_ns = s.flush_ns;
        c.ck_bytes = tiers_->ckpt_bytes();
        c.ck_slots = tiers_->ckpt_slots();
        c.host_on = tiers_->host_on();
        return c;
    }

    int sessions(std::vector<SessionStat>* out, SessionOccupancy* occ) const override {
        out->clear();
        if (!tiers_) return 0;
        IdleTiers::Occupancy held;
        const std::vector<IdleTiers::SessionInfo> v = tiers_->sessions(occ ? &held : nullptr);
        if (occ) {
            occ->host_bytes = held.host_bytes;
            occ->host_cap_bytes = held.host_cap_bytes;
        }
        const uint64_t now = rad_mono_ns();
        out->reserve(v.size());
        for (const IdleTiers::SessionInfo& s : v) {
            SessionStat o;
            o.id        = s.id;
            o.model     = s.model;
            o.label     = s.label;
            o.n_tokens  = s.n_tokens;
            o.bytes     = s.bytes;
            o.idle_s    = now > s.last_used_ns  ? (double)(now - s.last_used_ns)  / 1e9 : 0.0;
            o.age_s     = now > s.first_seen_ns ? (double)(now - s.first_seen_ns) / 1e9 : 0.0;
            o.on_device = s.on_device;
            o.on_host   = s.on_host;
            o.moving    = s.moving;
            o.snapshots   = s.snapshots;
            o.snap_bytes  = s.snap_bytes;
            o.resume_tokens = s.resume_tokens;
            out->push_back(std::move(o));
        }
        return (int)out->size();
    }

    /* The state name is derived: ReqState has no `Prefill`, since to the scheduler prefill is a
     * running request whose prompt is not yet all in the KV. The table reports it separately. */
    int requests(std::vector<RequestStat>* out) const override {
        std::vector<Scheduler::ReqStat> raw(256);
        const int n = s_->observed_requests(raw.data(), (int)raw.size());
        out->clear();
        out->reserve((size_t)n);
        for (int i = 0; i < n; ++i) {
            const Scheduler::ReqStat& r = raw[(size_t)i];
            RequestStat o;
            o.id = r.id;
            switch ((ReqState)r.state) {
                case ReqState::Waiting:   o.state = "waiting";   break;
                case ReqState::Preempted: o.state = "preempted"; break;
                case ReqState::Running:
                    o.state = r.computed_tokens < r.prompt_tokens ? "prefill" : "decode";
                    break;
                default:                  o.state = "done";      break;
            }
            o.prompt_tokens   = r.prompt_tokens;
            o.computed_tokens = r.computed_tokens;
            o.cached_tokens   = r.cached_tokens;
            o.output_tokens   = r.output_tokens;
            o.max_tokens      = r.max_tokens;
            o.ctx_tokens      = r.ctx_tokens;
            o.kv_blocks       = r.kv_blocks;
            o.kv_tokens       = r.kv_tokens;
            o.n_draft         = r.n_draft;
            o.slot            = r.slot;
            o.age_s           = r.age_s;
            o.ttft_s          = r.ttft_s;
            o.decode_tps      = r.decode_tps;
            out->push_back(std::move(o));
        }
        return n;
    }

    /* Both read the copy the engine thread publishes (Scheduler::publish), never the
     * scheduler's live state: every caller is an observer. */
    SchedMetrics metrics() const override {
        return bridge_metrics(s_->observed_metrics());
    }

private:
    /* Declared in the constructor's initialiser order (-Wreorder). */
    Scheduler*            s_;
    const Sampler*        sampler_ = nullptr;
    const IdleTiers*      tiers_ = nullptr;
    /* Ids start at 1: zero is the "no request" value the sink and the metrics use. */
    std::atomic<uint64_t> next_{1};
};

/* ================================================================== text */

class DetokenizerBridge final : public IDetokenizer {
public:
    DetokenizerBridge(std::shared_ptr<const Vocab> v, std::vector<std::string> stops,
                      const std::vector<std::string>& preserved)
        : d_(std::move(v), std::move(stops), /*render_special=*/false, preserved) {}
    std::string push(int32_t t) override { return d_.push(t); }
    std::string flush() override { return d_.flush(); }
    bool stopped() const override { return d_.stopped(); }
    std::string stop_hit() const override { return d_.stop_hit(); }
    void defer_stops_until(const std::string& m) override { d_.defer_stops_until(m); }
private:
    Detokenizer d_;
};

class TokenizerBridge final : public ITokenizer {
public:
    explicit TokenizerBridge(std::shared_ptr<const Vocab> v) : vocab_(std::move(v)) {
        tok_.init(vocab_);
    }

    std::vector<int32_t> encode(const std::string& text, bool add_special,
                                bool parse_special) const override {
        std::vector<int32_t> out;
        int st = tok_.encode(text, out, add_special, parse_special);
        if (st < 0) {
            /* The caller answers 400 on an empty result. A partial tokenisation would silently
             * serve a different prompt. */
            RAD_WARN("server: tokenise failed: %s", rad_strerror(st));
            out.clear();
        }
        return out;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        /* Sanitised like Detokenizer::push: byte-fallback ids need not decode to valid UTF-8,
         * and one malformed byte in a JSON string breaks the whole response. */
        return utf8_sanitize(tok_.decode(ids, /*render_special=*/false));
    }
    int32_t eos() const override { return vocab_->eos(); }
    int32_t bos() const override { return vocab_->bos(); }

    VocabInfo vocab_info() const override {
        VocabInfo i;
        i.n_tokens = vocab_->n_tokens();
        i.bos = vocab_->bos(); i.eos = vocab_->eos(); i.eot = vocab_->eot();
        i.unk = vocab_->unk(); i.pad = vocab_->pad();
        i.add_bos = vocab_->add_bos(); i.add_eos = vocab_->add_eos();
        switch (vocab_->kind()) {
            case RAD_TOK_BPE:       i.kind = "bpe";       break;
            case RAD_TOK_UNIGRAM:   i.kind = "unigram";   break;
            case RAD_TOK_WORDPIECE: i.kind = "wordpiece"; break;
            case RAD_TOK_RWKV:      i.kind = "rwkv";      break;
            default:                i.kind = "unknown";   break;
        }
        return i;
    }

    /* Vocab::text, sanitised: a byte-fallback vocabulary's piece need not be valid UTF-8 (a
     * byte-level BPE piece always is), and this goes into a JSON string. */
    std::string piece(int32_t id) const override {
        if (id < 0 || id >= vocab_->n_tokens()) return std::string();
        return utf8_sanitize(vocab_->text(id));
    }
    std::string token_bytes(int32_t id) const override {
        if (id < 0 || id >= vocab_->n_tokens()) return std::string();
        return tok_.decode({ id }, /*render_special=*/true);
    }
    std::string reply_bytes(int32_t id) const override {
        if (id < 0 || id >= vocab_->n_tokens()) return std::string();
        return tok_.decode({ id }, /*render_special=*/false);
    }

    std::unique_ptr<IDetokenizer> detokenizer(
        const std::vector<std::string>& stops,
        const std::vector<std::string>& preserved = {}) const override {
        return std::unique_ptr<IDetokenizer>(new DetokenizerBridge(vocab_, stops, preserved));
    }

private:
    std::shared_ptr<const Vocab> vocab_;
    Tokenizer                    tok_;
};

/* ================================================================== chat
 *
 * core/text's ChatTemplate renders the prompt and generation prompt. The model's reply format
 * (chatfmt.h) says how the reply is spelled, and everything depending on the spelling (grammar,
 * trigger words, tokens the decoder must render, the reply parser) is derived from it here.
 *
 *   * `preserved_tokens` lists control tokens the decoder must render as text. The decoder
 *     suppresses control tokens, and some formats are built from them: MiniCPM5 writes
 *     `<function`, `<param` and their closers as control tokens, so without this the parser
 *     sees no tags and the call reaches the client as prose. (Upstream the field protects them
 *     from a sampler ban stage; this sampler has none, core/sample/sampler.h.)
 *
 *   * The parser is built per render, behind IReplyParser: which arguments are JSON depends on
 *     this request's tool schemas, and whether the reply starts inside a reasoning block on this
 *     generation prompt.
 */
class ReplyReaderBridge final : public IReplyReader {
public:
    ReplyReaderBridge(std::unique_ptr<ReplyStream> s, const std::string* format_name)
        : s_(std::move(s)), name_(format_name),
          keep_(log_level() >= Log::Trace) {}

    void feed(std::string_view text, std::vector<ReplyDelta>* out) override {
        if (keep_) raw_.append(text);
        s_->feed(text, out ? &ev_ : nullptr);
        move_out(out);
    }

    void finish(std::vector<ReplyDelta>* out) override {
        s_->finish(out ? &ev_ : nullptr);
        move_out(out);
        /* A reply outside the format (a call to an unknown function, text inside a call, a call
         * after closing prose) is still returned whole, but logged once: either the model writes
         * its format loosely or the format is described wrongly. */
        if (!s_->strict() && !warned_) {
            warned_ = true;
            RAD_WARN("server: a reply left the '%s' reply format and was read by its recovery "
                     "rules; at trace level the reply is logged", name_->c_str());
            if (keep_) RAD_TRACE("server: the reply was <<<%s>>>", raw_.c_str());
        }
    }

    ParsedMessage message() const override {
        const ReplyMessage& r = s_->message();
        ParsedMessage m;
        m.content   = r.content;
        m.reasoning = r.reasoning;
        m.tool_calls.reserve(r.calls.size());
        for (const ReplyCall& c : r.calls) m.tool_calls.push_back({ std::string(), c.name, c.arguments });
        m.open_call = !r.calls.empty() && !r.calls.back().closed;
        return m;
    }

    std::string content_with_open_call(std::string_view reply) const override {
        return s_->content_with_open_call(reply);
    }

private:
    void move_out(std::vector<ReplyDelta>* out) {
        if (!out) return;
        for (ReplyEvent& e : ev_) {
            ReplyDelta d;
            switch (e.kind) {
            case ReplyEvent::Reasoning: d.kind = ReplyDelta::Reasoning; break;
            case ReplyEvent::Content:   d.kind = ReplyDelta::Content;   break;
            case ReplyEvent::CallBegin: d.kind = ReplyDelta::CallBegin; break;
            case ReplyEvent::CallArgs:  d.kind = ReplyDelta::CallArgs;  break;
            case ReplyEvent::CallEnd:   d.kind = ReplyDelta::CallEnd;   break;
            }
            d.call = e.call;
            d.text = std::move(e.text);
            out->push_back(std::move(d));
        }
        ev_.clear();
    }

    std::unique_ptr<ReplyStream> s_;
    const std::string*           name_;
    std::vector<ReplyEvent>      ev_;
    bool                         keep_ = false, warned_ = false;
    std::string                  raw_;   /* the reply, kept only when trace logging is on */
};

class ReplyParserBridge final : public IReplyParser {
public:
    explicit ReplyParserBridge(std::shared_ptr<const ReplyParser> p) : p_(std::move(p)) {}
    std::unique_ptr<IReplyReader> open() const override {
        return std::unique_ptr<IReplyReader>(new ReplyReaderBridge(p_->open(), &p_->format().name));
    }
private:
    std::shared_ptr<const ReplyParser> p_;
};

class ChatTemplateBridge final : public IChatTemplate {
public:
    /* Returns RAD_OK, or ChatTemplate::load_from_vocab's status. A server whose template did not
     * compile must not wire it: a 501 beats a 200 with ChatML-shaped nonsense.
     *
     * `declared` is the architecture plugin's reply format (rad_arch_chat_format), or null to
     * derive it from the template. A declaration the parser cannot run is refused at startup. */
    int init(const Vocab& v, const RadChatFormat* declared = nullptr,
             const std::string& plugin = std::string()) {
        /* The spellings, resolved once. See ReplyOptions::terminators. */
        terminators_.clear();
        for (int32_t id : v.eog()) {
            const std::string s = v.text(id);
            if (!s.empty()) terminators_.push_back(s);
        }
        const int st = t_.load_from_vocab(v);
        if (st < 0) return st;

        ChatFormat f;
        std::string why;
        declared_ = declared != nullptr;
        if (declared) {
            if (chat_format_from_abi(declared, &f, &why) < 0 || chat_format_check(f, &why) < 0) {
                RAD_ERR("chat: plugin '%s' declares a reply format the parser cannot run: %s",
                        plugin.c_str(), why.c_str());
                return RAD_E_INVAL;
            }
            if (f.name.empty()) f.name = "plugin:" + plugin;
        } else {
            const std::string bos = v.bos() >= 0 ? v.text(v.bos()) : std::string();
            const std::string eos = v.eos() >= 0 ? v.text(v.eos()) : std::string();
            std::string refused;
            if (chat_format_from_template(v.chat_template(), bos, eos, &f, &refused, &why) < 0) {
                RAD_WARN("chat: this model's reply format cannot be described (%s). Replies are "
                         "returned as content with no reasoning split, and requests with tools "
                         "are refused.", why.c_str());
                f = ChatFormat{};
                f.name = "content";
                tool_refusal_ = why;
            } else if (!refused.empty()) {
                tool_refusal_ = refused;
            } else if (t_.caps().tools && !f.has_tools()) {
                tool_refusal_ = "the template analysis found no tool-call format in it";
            }
        }
        if (ReplyFormat::compile(f, &fmt_, &why) < 0) {
            RAD_ERR("chat: the reply format '%s' cannot be run: %s", f.name.c_str(), why.c_str());
            return RAD_E_INVAL;
        }
        RAD_INFO("chat: reply format %s", chat_format_describe(f).c_str());
        if (t_.caps().tools && !tool_refusal_.empty())
            RAD_WARN("chat: tool calls are refused for this model: %s", tool_refusal_.c_str());
        return RAD_OK;
    }

    bool ready() const { return t_.ready() && fmt_ != nullptr; }
    bool supports_tools() const override { return t_.caps().tools; }
    bool supports_reasoning() const override { return t_.caps().preserve_reasoning; }
    std::string tool_refusal() const override { return tool_refusal_; }
    std::string source() const override { return t_.source(); }
    /* Under llama-server's names (its chat_template_caps), from the same probe. */
    std::string caps() const override {
        const ChatCaps& c = t_.caps();
        json o;
        o["supports_tools"]               = c.tools;
        o["supports_tool_calls"]          = c.tool_calls;
        o["supports_system_role"]         = c.system_role;
        o["supports_parallel_tool_calls"] = c.parallel_tool_calls;
        o["supports_preserve_reasoning"]  = c.preserve_reasoning;
        o["supports_string_content"]      = c.string_content;
        o["supports_typed_content"]       = c.typed_content;
        o["supports_object_arguments"]    = c.object_arguments;
        return o.dump();
    }

    int apply(const json& messages, const json& tools, const ChatRenderOptions& opt,
              ChatRender* out, std::string* why) const override {
        ChatOptions o;
        o.add_generation_prompt = opt.add_generation_prompt;
        o.enable_thinking       = opt.enable_thinking;
        o.parallel_tool_calls   = opt.parallel_tool_calls;
        o.tool_choice           = opt.tool_choice;
        o.reasoning_format      = opt.reasoning_format;
        o.continue_final        = opt.continue_final;
        o.grammar               = opt.grammar;
        o.json_schema           = opt.json_schema;
        o.template_kwargs       = opt.template_kwargs;
        o.quiet                 = opt.quiet;

        ChatPrompt p;
        const int st = t_.apply(messages, tools, o, &p);
        if (st < 0) {
            /* Prefer the template's own message: a jinja `raise_exception` names the field and
             * its accepted values, which the generic lines below do not. */
            if (why)
                *why = !p.error.empty()
                           ? p.error
                           : st == RAD_E_UNSUPPORTED
                                 ? "the model's chat template refused this request"
                                 : "the messages could not be rendered by this model's chat "
                                   "template";
            return st;
        }

        const ChatFormat& f = fmt_->format();
        /* Calls are parsed only when the request offered tools to call. With none, or with
         * `tool_choice: "none"` (which arrives here with the tools already cleared), call syntax
         * in the reply is prose to this request. */
        const bool calls = opt.tool_choice != "none" && tools.is_array() && !tools.empty();
        const bool doc = !opt.json_schema.empty();

        out->prompt            = std::move(p.prompt);
        out->additional_stops  = std::move(p.additional_stops);
        out->generation_prompt = p.generation_prompt;
        out->preserved_tokens  = declared_ ? chat_format_markers(f) : p.preserved_tokens;
        out->grammar_triggers.clear();

        if (calls || doc) {
            /* The grammar, and the words that trigger it, come from the reply format: the same
             * description the parser reads the reply back with. */
            const nlohmann::ordered_json schema =
                doc ? nlohmann::ordered_json::parse(opt.json_schema, nullptr, false)
                    : nlohmann::ordered_json();
            ChatGrammarRequest gq;
            gq.tools               = calls ? &tools : nullptr;
            gq.json_schema         = doc && schema.is_object() ? &schema : nullptr;
            gq.generation_prompt   = p.generation_prompt;
            gq.parallel_tool_calls = opt.parallel_tool_calls;
            gq.extract_reasoning   = opt.reasoning_format != "none";
            gq.require_call        = calls && opt.tool_choice == "required";
            gq.only_tool           = opt.tool_name;
            ChatGrammar g;
            std::string gerr;
            if (chat_format_grammar(f, gq, &g, &gerr) < 0) {
                if (why) *why = "the grammar for this request could not be built: " + gerr;
                return RAD_E_INVAL;
            }
            out->grammar      = std::move(g.text);
            out->grammar_lazy = g.lazy;
            for (const std::string& w : g.trigger_words)
                out->grammar_triggers.push_back({ kTriggerWord, w });
        } else {
            out->grammar      = std::move(p.grammar);
            out->grammar_lazy = p.grammar_lazy;
            for (const auto& t : p.grammar_triggers)
                out->grammar_triggers.push_back({ t.type, t.value });
        }

        ReplyOptions ro;
        ro.generation_prompt   = p.generation_prompt;
        ro.extract_reasoning   = opt.reasoning_format != "none";
        ro.tools               = calls ? &tools : nullptr;
        ro.parallel_tool_calls = opt.parallel_tool_calls;
        ro.response_format     = doc;
        ro.terminators         = terminators_;
        std::shared_ptr<const ReplyParser> rp;
        std::string perr;
        const int ps = ReplyParser::create(fmt_, ro, &rp, &perr);
        if (ps < 0) {
            if (why) *why = perr;
            return ps;
        }
        const ReasoningMarkers rm = rp->reasoning();
        out->think_open        = rm.open;
        out->think_close       = rm.close;
        out->think_starts_open = rm.starts_open;
        out->parser = std::make_shared<ReplyParserBridge>(std::move(rp));
        return RAD_OK;
    }

private:
    /* A lazy grammar's trigger is a literal word; SamplingParams::GrammarTrigger numbers the
     * kinds the way llama.cpp's common_grammar_trigger_type does. */
    static constexpr int kTriggerWord = 1;

    ChatTemplate t_;
    std::vector<std::string> terminators_;
    std::shared_ptr<const ReplyFormat> fmt_;
    bool declared_ = false;
    std::string tool_refusal_;
};

/* ================================================================== grammar
 *
 * Compiles at admission, not at first token, so an uncompilable schema is a 400 rather than a
 * request that fails on the step path, and the scheduler thread never compiles.
 */
class GrammarBridge final : public IGrammarCompiler {
public:
    explicit GrammarBridge(const VocabView* v) : v_(v) {}

    int from_json_schema(const json& schema, std::string& out, std::string& err) const override {
        return grammar_from_json_schema(schema.dump(), "root", &out, &err);
    }

    /* The whole compile (parse, checks, mask plan) runs on the admission thread. The program
     * rides on the request; the sampler finds it by its text while the request holds it. */
    int compile(const std::string& gbnf, std::shared_ptr<const GbnfProgram>* out,
                std::string& err) const override {
        if (gbnf.empty()) return RAD_OK;
        if (!v_) { err = "no vocabulary attached"; return RAD_E_STATE; }
        return gbnf_compile(gbnf, "root", v_, out, &err);
    }

private:
    const VocabView* v_;
};

/* Media, through the container's processor (core/mm). Present only when the program declares an
 * encoder; `accepts` follows the modalities that encoder serves, so e.g. video against an
 * image-only tower is refused with a reason instead of processed into unencodable patches. */
class MultimodalBridge final : public IMultimodal {
public:
    MultimodalBridge(const mm::Processor* p, uint32_t modalities) : p_(p), mod_(modalities) {}

    bool accepts(MediaKind k) const override {
        if (!p_ || !mm::decoder_available()) return false;
        if (k == MediaKind::Image) return (mod_ & RAD_MM_IMAGE) != 0;
        if (k == MediaKind::Video) return (mod_ & RAD_MM_VIDEO) != 0;
        if (k == MediaKind::Audio) return (mod_ & RAD_MM_AUDIO) != 0 && p_->has_audio();
        return false;
    }

    int prepare(MediaKind k, const std::vector<uint8_t>& bytes, const MediaOpts& opts,
                std::shared_ptr<mm::Item>* out, std::string* why) override {
        if (!accepts(k)) { if (why) *why = "this kind of media is not served"; return RAD_E_UNSUPPORTED; }
        if (k == MediaKind::Audio) return p_->audio(bytes.data(), bytes.size(), out, why);
        return k == MediaKind::Image
                   ? p_->image(bytes.data(), bytes.size(), opts.image_tokens, out, why,
                               opts.max_pixels)
                   : p_->video(bytes.data(), bytes.size(), out, why);
    }

    int32_t audio_rate() const override {
        return accepts(MediaKind::Audio) ? p_->audio_config().rate : 0;
    }
    int64_t audio_max_samples() const override {
        return accepts(MediaKind::Audio) ? p_->audio_config().max_samples : 0;
    }
    int decode_audio(const std::vector<uint8_t>& bytes, int64_t max_samples,
                     std::vector<float>* out, std::string* why) override {
        if (!accepts(MediaKind::Audio)) { if (why) *why = "audio is not served"; return RAD_E_UNSUPPORTED; }
        /* One sample past the bound is decoded, so a longer file is refused rather than cut. */
        RAD_TRY(mm::decode_audio(bytes.data(), bytes.size(), p_->audio_config().rate,
                                 max_samples + 1, out, why));
        if ((int64_t)out->size() > max_samples) {
            if (why) *why = "the recording is longer than this server transcribes";
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }
    int prepare_audio(const std::vector<float>& samples, std::shared_ptr<mm::Item>* out,
                      std::string* why) override {
        if (!accepts(MediaKind::Audio)) { if (why) *why = "audio is not served"; return RAD_E_UNSUPPORTED; }
        return p_->audio_from(samples, out, why);
    }

private:
    const mm::Processor* p_;
    uint32_t             mod_;
};

}  /* namespace server */
}  /* namespace rad */

#endif  /* the engine components are present */
