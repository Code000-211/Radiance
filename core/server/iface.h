/* core/server/iface.h -- everything the server needs from components it does not own.
 *
 * The server is written against pure-virtual interfaces rather than core/sched and core/text, so
 * the HTTP layer has no header dependency on them, and tests can drive every endpoint against a
 * fake scheduler in-process (cancellation and the streaming wire format, with no port or model).
 *
 * The cost is one virtual call per operation, on the admission path and the HTTP thread, never
 * inside rad_arch_step or per token on the step path. adapters.h binds the concrete types
 * (rad::Scheduler, rad::Tokenizer, ...) to these.
 */
#pragma once
#include "rad_core.h"

#include <nlohmann/json_fwd.hpp>

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rad {
namespace gen { struct Request; struct AudioRequest; class AudioTicket; struct VideoRequest; class VideoTicket; }
namespace mm { struct Frame; struct Item; }
namespace server {

/* ordered_json, not json: a tool's parameter schema is rendered into the prompt in the caller's
 * order, and property order in a schema changes what the model emits. */
using json = nlohmann::ordered_json;

/* ================================================================== the scheduler
 *
 * Contract with core/sched: `submit` takes the request by value and returns its assigned id;
 * `cancel` is idempotent and frees the request's KV blocks immediately, not at the next
 * completion (spec §14), since abandoned generations are routine for agent clients.
 */
struct SchedMetrics {
    int64_t waiting = 0;
    int64_t running = 0;
    int64_t preempted = 0;
    double  kv_util = 0.0;            /* 0..1 of the paged KV pool */
    /* The paged pools in tokens: how much context fits, which kv_util does not say. */
    int64_t kv_tokens_used = 0;
    int64_t kv_tokens_total = 0;
    /* The part of `used` no running request reads: the prefix cache's unshared holding of
     * finished turns. It separates a working cache from a pool that never releases. A block a
     * running request also reads counts as that request's, since evicting it frees nothing. */
    int64_t kv_tokens_cached = 0;
    /* Paged and stateful pools together, in bytes: how much VRAM the KV cache holds. Tokens do
     * not sum across groups (paged groups share one token axis, a recurrent group has none), so
     * the token figures above are the binding group's; bytes are disjoint and cover every group.
     *
     * Used means held, including prefix-cache blocks of finished turns; a block stops counting
     * when it reaches the free list (eviction, cancellation or idle-tier demotion). */
    int64_t kv_bytes_used = 0;
    int64_t kv_bytes_total = 0;
    /* What the pool was carved to address, which an elastic pool may not be holding. */
    int64_t kv_bytes_carved = 0;
    /* The most context the cache can hold: every block of its carve, including those lent to the
     * expert slab (`kv_bytes_carved` in bytes). The smallest over the paged groups, which share
     * one token axis; windowed groups are excluded, since their carve bounds lookback, not fit. */
    int64_t kv_tokens_max = 0;

    double  prefix_hit_rate = 0.0;    /* 0..1 */
    /* Cumulative prefix-cache evictions. They rise as reusable prefixes are dropped, while
     * `prefix_hit_rate` falls only afterwards. */
    int64_t prefix_evictions = 0;
    int64_t prefix_ckpt_evictions = 0;
    /* Snapshot evictions by reason (the retention policy; the least recently used of a store
     * retention left full) and reservations that went without a snapshot. */
    int64_t prefix_ckpt_retired = 0, prefix_ckpt_lru = 0, prefix_ckpt_refused = 0;
    /* Admissions whose prefix hit was cut to nothing, by reason, and the checkpoint shortfall:
     * attention-cached prompt tokens computed again (rad::SchedMetrics::hit_cut_tokens). */
    int64_t hit_cut_no_snapshot = 0, hit_cut_window = 0, hit_cut_adopt = 0;
    int64_t hit_cut_tokens = 0;
    /* Removed by --prefix-cache-ttl, cumulative. Not evictions: nothing wanted the memory. */
    int64_t prefix_expired = 0, prefix_ckpt_expired = 0;
    /* Decoupled prefix caching (spec §7.3). `prefix_hit_rate` is the attention hit rate, as in
     * vLLM. The linear side hits at a different granularity (whole checkpoints, not blocks), so
     * it is reported separately, with token counts and checkpoints written. */
    double  linear_hit_rate = 0.0;
    int64_t attn_cached_tokens = 0;
    int64_t linear_cached_tokens = 0;
    int64_t checkpoints_written = 0;
    double  prefill_tps = 0.0;
    double  decode_tps = 0.0;
    /* Exact cumulative counts behind the EWMA rates above, for comparing runs.
     * `vllm:generation_tokens_total` is charged when a request finishes, so it cannot serve. */
    int64_t prefill_tokens = 0;
    int64_t decode_tokens = 0;
    double  draft_acceptance = 0.0;   /* 0..1, spec §10 */
    /* The exact pair the ratio is computed from. */
    int64_t draft_tokens = 0;
    int64_t draft_accepted = 0;
    /* Prefix hits ended early for a windowed block drafter, and the tokens they gave back. */
    int64_t draft_refills = 0;
    int64_t draft_refill_tokens = 0;
    /* Per-position draft acceptance. The pooled `draft_acceptance` cannot tell a drafter that
     * decays with depth from one whose later rounds are broken. Per position j: verifies that
     * reached j (every earlier position accepted) and how many of those took j. */
    static const int MAX_DRAFT_POS = 64;
    int     draft_depth_seen = 0;
    int64_t draft_pos_reached [MAX_DRAFT_POS] = {};
    int64_t draft_pos_accepted[MAX_DRAFT_POS] = {};
    /* Adaptive draft window (--draft-window adaptive), zero while off: the running sequences'
     * mean window, and verifies by the window they carried (index = drafts). */
    bool    draft_adaptive = false;
    double  draft_depth_adaptive = 0.0;
    int64_t draft_verify_depth[MAX_DRAFT_POS + 1] = {};
    /* The cost policy's observation ledger (rad::SchedMetrics::draft_obs_*), cumulative over the
     * server's lifetime like every source counter -- there is no reset. The first three count
     * observation ROWS -- one per delivered request/sequence row of a cycle, not one per token
     * position: a wider verify window adds labels inside a row, not rows, so only more delivered
     * rows raise these -- and the cycle counter counts whole CYCLES dropped before delivery. The
     * cost-observation counter has its own unit again: one per valid observation handed to an
     * inactive table, which is neither a row nor a dropped cycle. The units are not a sum. */
    int64_t draft_obs_rows_attempted = 0;
    int64_t draft_obs_rows_applied = 0;
    int64_t draft_obs_rows_skipped = 0;
    int64_t draft_obs_cycles_discarded = 0;
    int64_t draft_cost_observations_inactive = 0;
    /* The cost policy's plan ledger (rad::SchedMetrics::draft_cost_*): cycles opened, how they
     * ended -- priced, the named fallback, a forced calibration depth, an empty planning point, or
     * no plan at all; they sum to `draft_cost_cycles` -- and the last plan's depth, replay
     * candidate, source and fallback reason. `last_*` describe the last plan **with rows**
     * (`last_rows`), so an idle or prefill step never overwrites the last real decision; a valid
     * all-K0 plan is one of them (`last_k == 0` with `last_rows > 0`). `last_replay`,
     * `last_source` and `last_fallback` are the names draft_types.h spells them by, never
     * numbers. */
    int64_t draft_cost_cycles = 0;
    int64_t draft_cost_priced = 0;
    int64_t draft_cost_fallback = 0;
    int64_t draft_cost_no_plan = 0;
    /* Planning points with no request rows: every step opens a cycle, and a pure prefill or idle
     * step has nothing to plan over. Not a decision, not a measurement (rad::SchedMetrics). */
    int64_t draft_cost_empty = 0;
    int64_t draft_cost_last_k = 0;
    int64_t draft_cost_last_replay = 0;
    int64_t draft_cost_last_source = 0;
    int64_t draft_cost_last_fallback = 0;
    int64_t draft_cost_last_rows = 0;
    /* The last issued Cost plan's priced-tier count: how many of the ladder's depths had a price
     * when it was chosen, so "both depths compared" is distinguishable from "only the deepest
     * was priced" (rad::SchedMetrics::draft_cost_tiers_priced). A search metric: a later resource
     * cut or re-price does not change it. 0 unless the last plan with rows was Cost: an empty
     * planning point leaves the last plan's ledger alone. */
    int64_t draft_cost_tiers_priced = 0;
    /* Forced calibration plans, and of them the ones a plan-time resource cut touched
     * (rad::SchedMetrics::draft_calib_forced / draft_calib_cut). Not cost decisions. */
    int64_t draft_cost_calibration = 0;
    int64_t draft_cost_calibration_cut = 0;
    /* Cycles a no-loan reference table refused to absorb (rad::SchedMetrics::draft_ref_skipped). */
    int64_t draft_cost_reference_skipped = 0;
    int64_t preemptions = 0;
    int64_t total_requests = 0;
    /* Engine steps: one forward pass plus any draft rounds after it. With speculation a step
     * emits more than one token, so step time cannot be derived from tokens per second. */
    int64_t steps = 0;
    /* Steps by shape. A mixed step carries a prefill chunk and decode rows, and its decode rows
     * take drafts like a decode step's (Scheduler::step). */
    int64_t steps_prefill = 0;
    int64_t steps_mixed = 0;
    int64_t steps_decode = 0;
    int64_t steps_ahead = 0;          /* issued behind the step before, ahead of its commit */
    int64_t prefill_chunks_capped = 0; /* held back for other work in the step (--busy-prefill-chunk) */
    int64_t steps_held = 0;           /* decode-only steps --prefill-interleave inserted */
    int64_t decode_rows_unspeculated = 0;
    /* Cumulative nanoseconds inside steps, all shapes and decode-only. Wall clock over a step
     * count is not a step time: the engine idles between turns, and a prefill chunk costs an
     * order of magnitude more than a decode step. `step_ns_decode / steps_decode` is the cost of
     * a decode step regardless of load. */
    int64_t step_ns = 0;
    int64_t step_ns_decode = 0;
    /* The single-sequence decode floor and most recent window (rad::SchedMetrics). */
    int64_t step_ns_decode_best = 0;
    int64_t step_ns_decode_recent = 0;
};


/* One stored conversation: what the engine still holds for a session between turns, which
 * decides whether the next turn costs a copy or a re-prefill.
 *
 * Sizes overlap: sessions sharing a system prompt share its blocks, so a total is not a sum, and
 * anything rendering these must say so. A session is routinely split across tiers (a maintenance
 * pass is bounded), so its location is block counts, not a label. */
struct SessionStat {
    uint64_t    id = 0;
    std::string model;            /* the container it was built against */
    std::string label;            /* the client's name for it (`session_id`), or "" */
    int64_t     n_tokens = 0;
    int64_t     bytes = 0;
    double      idle_s = 0;       /* since its last turn */
    double      age_s = 0;        /* since it was first seen */
    int64_t     on_device = 0, on_host = 0;   /* blocks */
    /* Linear-state snapshots on a hybrid model, counted apart from blocks: a session with its
     * blocks kept but no snapshot re-runs every recurrent layer over its transcript, though the
     * hit rate reports a hit. Zero on a model with no recurrent state. */
    int64_t     snapshots = 0;
    int64_t     snap_bytes = 0;
    /* The prefix a return skips: the deepest snapshot's position (IdleTiers::SessionInfo). */
    int64_t     resume_tokens = 0;
    bool        moving = false;   /* a transfer is outstanding */
};

/* What stored conversations occupy off the card, blocks and snapshots each counted once (the
 * per-session rows cannot be summed into this). Caps are 0 for a tier that is off. */
struct SessionOccupancy {
    int64_t host_bytes = 0;
    int64_t host_cap_bytes = 0;
};

/* One in-flight request, for the live request table: shows which request is stuck, e.g. one
 * waiting on a prompt that will never fit, which no aggregate metric reveals. */
struct RequestStat {
    uint64_t    id = 0;
    std::string state;             /* waiting | prefill | decode | preempted | done */
    int64_t     prompt_tokens = 0;
    int64_t     computed_tokens = 0;
    int64_t     cached_tokens = 0;
    int64_t     output_tokens = 0;
    int64_t     max_tokens = 0;
    int64_t     ctx_tokens = 0;
    int64_t     kv_blocks = 0;    /* summed over the paged groups: they are disjoint allocations */
    /* Context held, on the token axis the paged groups share (not their sum). */
    int64_t     kv_tokens = 0;
    int32_t     n_draft = 0;
    int32_t     slot = -1;
    double      age_s = 0;
    double      ttft_s = -1;       /* negative until the first token */
    double      decode_tps = 0;
};

class IScheduler {
public:
    virtual ~IScheduler() = default;
    virtual uint64_t     submit(Request&& r) = 0;
    /* Admits all sequences of one API request before a step can see any of them; submitted one
     * at a time, a step could start between calls and the batch (and its tie-level rounding)
     * would depend on thread timing. ids[i] is submit()'s id for rs[i]. The default loop suits a
     * scheduler whose steps cannot interleave with submission. */
    virtual void submit_all(std::vector<Request>& rs, std::vector<uint64_t>* ids) {
        ids->clear();
        for (Request& r : rs) ids->push_back(submit(std::move(r)));
    }
    virtual void         cancel(uint64_t id) = 0;
    virtual SchedMetrics metrics() const = 0;

    /* The HTTP thread is done with this request's sink; the scheduler may drop the Request it
     * keeps alive after termination. Called at most once per id, after the response is written.
     * Default no-op, for a scheduler that frees at completion. */
    virtual void reap(uint64_t /*id*/) {}

    /* Whether this request's sampling parameters can be served at all. An unresolved sampler op
     * is not a declare failure (a deployment that never sends `typical_p` needs no typical
     * kernel; core/sample/sampler.h), only a request that asks for it is refused. The step loop
     * treats an unresolved stage as unreachable and relies on this check.
     *
     * Called once per submission on the HTTP thread, before any Request is built. RAD_OK, or a
     * negative status with `why` set. The default admits everything (test fakes); the real
     * answer comes from SchedulerBridge. */
    virtual int admit(const SamplingParams& /*sp*/, std::string* /*why*/) const { return RAD_OK; }

    /* Drops every prefix-cache entry and returns how many, or a negative status (501) when there
     * is no prefix cache. Lets a benchmark start cold, as vLLM's endpoint does. Releasing the
     * cache's references is safe against a running step (blocks a live sequence holds stay
     * allocated), but it takes the scheduler's lock because a step may be inserting into the map. */
    virtual int64_t reset_prefix_cache() { return RAD_E_UNSUPPORTED; }

    /* The live request table, oldest first; returns the count. Default: none kept. */
    virtual int requests(std::vector<RequestStat>* /*out*/) const { return 0; }

    /* The stored conversations and where they are held. Default: no idle tiers, so an empty
     * table (everything the cache holds is in VRAM). */
    virtual int sessions(std::vector<SessionStat>* /*out*/, SessionOccupancy* /*occ*/) const {
        return 0;
    }

    /* What the idle tiers have done. Without it the session table looks the same whether the
     * tiers work, are off, or fail every pass. */
    struct TierCounters {
        /* `to_host` counts copies; `dropped` is VRAM given up after one (a conversation that
         * returns before its VRAM is wanted is copied but never dropped). `host_drops` is entries
         * a full host tier gave up; `promoted` counts restores. */
        int64_t to_host = 0, dropped = 0, promoted = 0;
        int64_t host_drops = 0, failures = 0;
        /* Copies a full host tier did not take, every host entry being worth as much; their
         * entries stay in VRAM only (IdleTiers::Stats::held_back). */
        int64_t held_back = 0;
        int64_t host_bytes = 0;
        /* The snapshot store, in host memory; it moves nothing. */
        int64_t ck_bytes = 0;      /* one snapshot; 0 on a model with no recurrent state */
        int64_t ck_slots = 0;
        int64_t restores = 0, restore_ns = 0, flushes = 0, flush_ns = 0;   /* IdleTiers::Stats */
        /* Whether a host tier exists; an idle tier and an unconfigured one have equal counters. */
        bool host_on = false;
    };
    virtual TierCounters tier_counters() const { return TierCounters{}; }
};

/* ================================================================== text frontend
 *
 * Contract with core/text. Detokenisation is incremental and streaming-safe (spec §12): byte-level
 * BPE emits partial UTF-8, so `push` returns "" until the bytes form a character, and `flush`
 * drains the rest at the end of a generation.
 */
class IDetokenizer {
public:
    virtual ~IDetokenizer() = default;
    virtual std::string push(int32_t token) = 0;
    virtual std::string flush() = 0;
    /* True once the decoder itself swallowed a stop string. core/text's Detokenizer, built with
     * the stop strings, never emits one, so the server cannot find it in the text and relies on
     * this. Default false, for a decoder that leaves stop handling to the server. */
    virtual bool stopped() const { return false; }
    /* The stop string that stopped it, "" when none has. */
    virtual std::string stop_hit() const { return std::string(); }
    /* Match the stop strings only after `marker` has been generated (core/text Detokenizer::
     * defer_stops_until): the reply starts inside its reasoning block. */
    virtual void defer_stops_until(const std::string& /*marker*/) {}
};

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    /* `parse_special` decides whether a control token spelled out in the text is that token or
     * its characters; it has no default so every caller chooses. Whole prompts are parsed (a
     * rendered chat template, or a raw /v1/completions prompt, as vLLM and llama.cpp read it).
     * Text that must reach the model as characters is encoded with false. */
    virtual std::vector<int32_t> encode(const std::string& text, bool add_special,
                                        bool parse_special) const = 0;
    virtual std::string          decode(const std::vector<int32_t>& toks) const = 0;
    virtual int32_t              eos() const = 0;
    virtual int32_t              bos() const = 0;

    /* Vocabulary facts for /tokenize and /get_tokenizer_info; never called during generation.
     * One struct, so the fields describe one vocabulary consistently. The default is "unknown"
     * (test fakes). */
    struct VocabInfo {
        int32_t     n_tokens = 0;
        int32_t     bos = -1, eos = -1, eot = -1, unk = -1, pad = -1;
        bool        add_bos = false, add_eos = false;
        std::string kind;            /* "bpe", "spm", "wpm", ... -- how it splits */
    };
    virtual VocabInfo vocab_info() const { return VocabInfo{}; }

    /* The raw piece an id stands for, not its decoded text: /tokenize's `return_token_strs` shows
     * what the tokeniser chose, and decoding one id at a time would turn a byte-fallback token
     * into a replacement character. "" when the id is out of range. */
    virtual std::string piece(int32_t /*id*/) const { return std::string(); }

    /* The bytes one id decodes to, unsanitised, control tokens rendered (llama-server's
     * `with_pieces`). Not necessarily whole UTF-8, so the caller checks before putting it in a
     * JSON string. */
    virtual std::string token_bytes(int32_t /*id*/) const { return std::string(); }
    /* The bytes one id adds to a reply: token_bytes with control tokens rendered as nothing, as
     * the decoder does. Used for logprobs entries, so their texts add up to the reply, as
     * llama-server's do. */
    virtual std::string reply_bytes(int32_t id) const { return token_bytes(id); }

    /* One incremental decoder per in-flight generation. `stops` lets core/text's Detokenizer
     * suppress a stop string that straddles a token boundary; IDetokenizer::stopped() reports it.
     *
     * `preserved` lists control tokens to decode as text. Some tool-call formats are built from
     * control tokens (MiniCPM5's `<function`, `<param` and closers); suppressing them would hand
     * the reply parser argument values without their structure. Set by the chat render. */
    virtual std::unique_ptr<IDetokenizer> detokenizer(
        const std::vector<std::string>& stops,
        const std::vector<std::string>& preserved = {}) const = 0;
};

class IReplyParser;

/* What a chat template needs beyond the messages.
 *
 * `template_kwargs` values are JSON text, not display text: `"false"`, `"\"high\""`, `"3"`, as
 * the request carries them and the template context wants them. Converting would mean guessing
 * that the string "false" is a boolean, which makes `enable_thinking: "false"` enable thinking. */
struct ChatRenderOptions {
    bool        add_generation_prompt = true;
    bool        enable_thinking = true;
    bool        parallel_tool_calls = true;
    std::string tool_choice = "auto";        /* auto | none | required */
    std::string tool_name;                   /* a named tool_choice's function, or "" */
    std::string grammar;                     /* the caller's GBNF, or "" */
    std::string json_schema;                 /* a response_format schema, as JSON text, or "" */
    std::string reasoning_format = "auto";
    /* Assistant prefill: continue the last (assistant) message rather than open a new turn. ""
     * off, else llama.cpp's "auto", "content" or "reasoning_content". */
    std::string continue_final;
    std::map<std::string, std::string> template_kwargs;
    /* Startup probes, where a refusal is an answer, not an error: do not log it. */
    bool        quiet = false;
};

/* A render's result. A tool-calling template produces more than a prompt: the grammar that keeps
 * the model's call parseable, the triggers that keep it off the prose, and the format's stop
 * strings. Dropping them yields tool calls no client can read. */
struct ChatRender {
    std::string prompt;

    std::string grammar;                     /* the template's, merged with the caller's */
    bool        grammar_lazy = false;
    std::vector<SamplingParams::GrammarTrigger> grammar_triggers;
    std::vector<std::string> additional_stops;

    /* The assistant prefix already inside `prompt`. The grammar describes the whole turn, so it
     * is advanced over this before constraining the first generated token. */
    std::string generation_prompt;

    /* The reply parser for this render: the model's reply format bound to these tools and this
     * generation prompt, hence per request. It reads calls only when the request sent tools.
     * Null when nothing needs parsing. */
    std::shared_ptr<IReplyParser> parser;

    /* Control tokens the parser must see as text. See ITokenizer::detokenizer. */
    std::vector<std::string> preserved_tokens;

    /* The reasoning block the reply faces, as the parser reads it: markers with optional
     * whitespace trimmed, and whether the reply starts inside the block. Empty `think_close`
     * means no reasoning block. The server maps these to token ids (ReasoningSpan) to count the
     * block and enforce a budget. */
    std::string think_open, think_close;
    bool        think_starts_open = false;
};

class IChatTemplate {
public:
    virtual ~IChatTemplate() = default;

    /* Render. RAD_OK, or a negative status with `why` naming what the template refused; `why`
     * becomes the 400's message, so it must be actionable for the caller. */
    virtual int apply(const json& messages, const json& tools, const ChatRenderOptions& opt,
                      ChatRender* out, std::string* why) const = 0;

    virtual bool supports_tools() const = 0;
    virtual bool supports_reasoning() const = 0;

    /* Why this model's tool calls cannot be read, or "" when they can. A template may support
     * tools in a format the reply parser does not know; the model still chats, and a request that
     * sends tools is refused with this sentence. */
    virtual std::string tool_refusal() const { return std::string(); }

    /* The template's source and its probed capabilities, for /props (llama-server's
     * `chat_template` and `chat_template_caps`), so clients can render prompts as this server
     * does. "" and "{}" (a JSON object as text) when unknown. */
    virtual std::string source() const { return std::string(); }
    virtual std::string caps() const { return "{}"; }
};

/* A reply taken apart: what core/text's reply parser gives back once the reply is complete. */
struct ToolCall {
    std::string id;          /* "" from the parser; the server assigns one */
    std::string name;
    std::string arguments;   /* JSON text; unterminated when the reply stopped inside the call */
};

struct ParsedMessage {
    std::string           content;
    std::string           reasoning;    /* "" unless the template supports reasoning */
    std::vector<ToolCall> tool_calls;
    /* The reply ended inside its last call: that call never closed, and is no call to run. */
    bool                  open_call = false;
};

/* One increment of a reply read as it is generated. Delivered increments are never revised: the
 * concatenated increments of each kind are the finished values, so streaming forwards each as it
 * comes. */
struct ReplyDelta {
    enum Kind : uint8_t {
        Reasoning,   /* text: a reasoning delta */
        Content,     /* text: a content delta */
        CallBegin,   /* text: the function name; `call` is the new call's index */
        CallArgs,    /* text: an arguments delta for call `call` */
        CallEnd,     /* call `call` is closed */
    };
    Kind        kind = Content;
    uint32_t    call = 0;
    std::string text;
};

/* One choice's reply, read once, in order, as it is generated. feed() appends generated text and
 * finish() ends it; each appends the increments it completes. Linear in the reply however it is
 * split; the text is never re-read from the top. */
class IReplyReader {
public:
    virtual ~IReplyReader() = default;
    virtual void feed(std::string_view text, std::vector<ReplyDelta>* out) = 0;
    virtual void finish(std::vector<ReplyDelta>* out) = 0;
    /* The whole reply, once finish() has run. */
    virtual ParsedMessage message() const = 0;
    /* After finish(): the content with an open last call (ParsedMessage::open_call) restored as
     * text, i.e. the content before the call plus the call's own bytes from `reply` (the whole
     * text fed). The content alone when no call is open. */
    virtual std::string content_with_open_call(std::string_view reply) const = 0;
};

/* A request's reply format. Each choice opens its own reader. A request that sent no tools gets
 * a parser that reads no calls, so call syntax in its reply is plain text. */
class IReplyParser {
public:
    virtual ~IReplyParser() = default;
    virtual std::unique_ptr<IReplyReader> open() const = 0;
};

/* JSON-schema -> GBNF (spec §13). Errors are values: RAD_OK, or negative with `err` naming the
 * part of the schema that could not be compiled. */
class IGrammarCompiler {
public:
    virtual ~IGrammarCompiler() = default;
    virtual int from_json_schema(const json& schema, std::string& out, std::string& err) const = 0;

    /* Compiles this GBNF on the admitting thread into the program the request carries
     * (SamplingParams::grammar_program). Compiling finds grammars that cannot run (left
     * recursion, constructs the mask engine cannot serve, bounds it outgrows) and builds the mask
     * plan, tens to hundreds of milliseconds of host work. On the scheduler thread a failure
     * could only kill the request mid-answer and the build would stall every step. Failure is a
     * 400 naming why. */
    virtual int compile(const std::string& gbnf, std::shared_ptr<const GbnfProgram>* out,
                        std::string& err) const = 0;
};

/* ================================================================== optional backends */

/* /v1/embeddings, present only for an embedding model (rad_declare_pooled); absent, the endpoint
 * is 501.
 *
 * Vectors come through the scheduler: the server submits one Request per input with
 * Request::embed set, and each finishes after one step with its pooled vector in its sink
 * (Sink::embedding). Batching, admission, media encoding and cancellation work as for a
 * completion. */
class IEmbedder {
public:
    virtual ~IEmbedder() = default;
    /* The width of a pooled vector. */
    virtual int64_t dim() const = 0;
    /* The longest input one step carries: an input is never split, so a longer one is refused. */
    virtual int64_t max_input() const = 0;
    /* The model's named prompts (sentence-transformers' `prompts`): the text a `prompt_name`
     * prepends to an input, e.g. "query" -> "task: search result | query: ". False when the model
     * states no prompt of that name. */
    virtual bool prompt(const std::string& name, std::string* text) const {
        (void)name; (void)text;
        return false;
    }
    /* The prompt an input takes when the request names none, "" for none. */
    virtual std::string default_prompt() const { return std::string(); }
    /* The chat template a `messages` input renders through, or null. */
    virtual IChatTemplate* message_template() const { return nullptr; }
};

/* Multimodal preprocessing (spec §11). The server pulls the bytes out of a content part; the
 * processor behind this decodes them, resizes and cuts them for the encoder, and says which
 * tokens take the part's place in the prompt -- placeholder runs, the markers around them and,
 * for a video, the timestamps between its frames. */
enum class MediaKind { Image, Video, Audio };

/* The placeholder a chat template renders where a media part sat. Template and server must agree
 * on it, so it is declared once, here. It is llama.cpp mtmd's string, since the templates are
 * shared. */
#define RAD_MEDIA_MARKER "<__media__>"

/* Per-request media options beside the bytes. Zero is the server's default. */
struct MediaOpts {
    /* Soft tokens (encoder output rows) an image may become, for a processor that sizes images
     * by a token budget (Gemma 4's: 70, 140, 280, 560 or 1120). */
    int32_t image_tokens = 0;
    /* A picture over this many pixels is first scaled down to that area (PIL's bicubic, sides
     * truncated), as a model's own pipeline may do before its processor; 0 leaves it as sent. */
    int64_t max_pixels = 0;
};

class IMultimodal {
public:
    virtual ~IMultimodal() = default;
    virtual bool accepts(MediaKind k) const = 0;
    /* Decode and process one part. `bytes` is the decoded file, not a URL: the server never
     * fetches a remote URL on a request's behalf. On success `*out` carries the patches and the
     * token run that replaces the marker; on failure `why` says what is wrong with the media. */
    virtual int prepare(MediaKind k, const std::vector<uint8_t>& bytes, const MediaOpts& opts,
                        std::shared_ptr<mm::Item>* out, std::string* why) = 0;

    /* Audio in two halves, for a route that cuts a long recording into clips itself
     * (/v1/audio/transcriptions): the file decoded to mono samples at audio_rate(), at most
     * `max_samples` of them (RAD_E_INVAL past that, never a silent cut), and a clip of at most
     * audio_max_samples() processed as prepare() would. Zero rate: no audio is served. */
    virtual int32_t audio_rate() const { return 0; }
    virtual int64_t audio_max_samples() const { return 0; }
    virtual int decode_audio(const std::vector<uint8_t>& bytes, int64_t max_samples,
                             std::vector<float>* out, std::string* why) {
        (void)bytes; (void)max_samples; (void)out;
        if (why) *why = "this server decodes no audio";
        return RAD_E_UNSUPPORTED;
    }
    virtual int prepare_audio(const std::vector<float>& samples, std::shared_ptr<mm::Item>* out,
                              std::string* why) {
        (void)samples; (void)out;
        if (why) *why = "this server serves no audio";
        return RAD_E_UNSUPPORTED;
    }
};

/* How the model takes audio, as the architecture plugin states it (RadAudioFormat,
 * abi/rad_builder.h), copied out at startup. It is all the speech routes know of the model. */
struct SpeechFormat {
    std::string transcribe_prompt;   /* {context}, {audio}, {language} */
    std::string language_prompt;     /* {language} */
    std::string transcript_open;
    std::string language_lead;
    /* (ISO 639-1 code, the model's name for it) */
    std::vector<std::pair<std::string, std::string>> languages;

    bool transcribes() const { return !transcribe_prompt.empty(); }
    /* The model's name for `lang`, which may be a code or a name in any case; "" when the model
     * names no such language. */
    std::string language_name(const std::string& lang) const;
};


/* The image generator (core/gen), for a model that makes pictures: its declared prompt templates
 * and sizes, and its request queue. The server turns an OpenAI images request into a gen::Request
 * (prompt rendered and tokenised, reference pictures resized for both of the model's readers) and
 * reads the pictures back off its ticket. */
struct ImageGenInfo {
    std::string name, tmpl_text, tmpl_edit, tmpl_image, tmpl_image_sep, drop_prefix,
                tmpl_transparent;
    int64_t latent_scale = 16, size_multiple = 32, pixel_channels = 4, max_tokens = 0;
    int64_t max_refs = 0, ref_area = 0, default_width = 0, default_height = 0;
    int64_t default_steps = 0, max_lanes = 1, max_cond_tokens = 0;
    /* The model samples on a schedule of its own and takes no step count (a distilled model). */
    bool    fixed_schedule = false;
    int64_t fixed_steps = 0;
    /* Guidance in the model card's terms: with `cfg_original` the scale s means cond + s (cond -
     * uncond), off at 0, and the core runs s + 1; otherwise uncond + s (cond - uncond), off at 1. */
    double  default_cfg = 1.0;
    bool    cfg_original = false;
    /* The negative prompt a guided request gets when it names none; without one it must. */
    bool        has_default_negative = false;
    std::string default_negative;
    /* The pictures are not read by the conditioning network: they go to the generator's own
     * encoder only (RAD_GEN_LAYOUT_JOINT_STEP). */
    bool    refs_encoder_only = false;
    /* A reference picture is shrunk to `ref_area` only when larger, then centre-cropped to the
     * size multiple; otherwise resized to `ref_area` at its aspect. */
    bool    ref_shrink_only = false;
    /* A prompt of fewer tokens after the drop prefix is padded with `pad_token` up to this many
     * (RadGeneratorDecl::min_tokens); 0: no minimum. */
    int64_t min_tokens = 0;
    int32_t pad_token = 0;
    /* The prompt is also read through the model's second tokenizer (IImageGen::encode_aux), at
     * most `aux_max_tokens` ids of it. */
    bool    aux_tokens = false;
    int64_t aux_max_tokens = 0;
    /* A discrete-time model (RadGeneratorDecl::train_steps): the samplers it serves (1 << id)
     * and the sampler and scheduler a request gets when it names none. 0 samplers: flow
     * matching, which takes neither field. */
    uint32_t samplers = 0;
    int32_t  default_sampler = 0, default_scheduler = 0;
    /* The prompt in windows (RadGeneratorDecl::chunk_tokens): tokens a window, its markers and
     * its padding token; 0 for a prompt read whole. */
    int32_t  chunk_tokens = 0, chunk_bos = 0, chunk_eos = 0, chunk_pad = 0;
};

class IImageGen {
public:
    virtual ~IImageGen() = default;
    virtual const ImageGenInfo& info() const = 0;
    /* Queue a request; refused at once, with `why`, when it can never run. */
    virtual int submit(std::unique_ptr<gen::Request> r, std::string* why) = 0;
    /* A reference picture as the conditioning network's encoder reads it: RGB at the size it was
     * resized to, which the processor must not change. */
    virtual int picture(const mm::Frame& f, std::shared_ptr<mm::Item>* out, std::string* why) = 0;
    virtual int64_t queued() const = 0;
    virtual int64_t running() const = 0;
    /* The user's prompt through the model's second tokenizer (ImageGenInfo::aux_tokens), with the
     * special tokens that tokenizer adds; RAD_E_UNSUPPORTED for a model without one. */
    virtual int encode_aux(const std::string& text, std::vector<int32_t>* out) const {
        (void)text;
        (void)out;
        return RAD_E_UNSUPPORTED;
    }
};

/* The audio generator (core/gen/audio_driver.h), for a model that makes music or other audio from
 * text: its declared prompt template and limits, the plugin's rewrite of a request's texts into
 * the form it was trained on, and its request queue. The server renders and tokenises the prompt
 * and reads the audio back off the ticket, a window at a time. */
struct AudioGenInfo {
    /* The prompt templates (RadAudioGenDecl): with instructions, without them ("" makes them
     * required), and the unconditional branch's own ("" derives it from the conditional prompt). */
    std::string name, tmpl, tmpl_plain, neg_tmpl;
    int64_t max_prompt = 0, max_frames = 0, frame_rate_num = 1, frame_rate_den = 1;
    int64_t sample_rate = 0, channels = 0;
    int64_t default_steps = 0, max_lanes = 1;   /* default_steps 0: no denoiser */
    double  default_cfg = 1.0;
};

class IAudioGen {
public:
    virtual ~IAudioGen() = default;
    virtual const AudioGenInfo& info() const = 0;
    /* `in` rewritten as the model wants text for `field` (RAD_TEXT_*); the text unchanged when the
     * plugin states no rewrite. RAD_E_INVAL with `why` for text the model cannot take. */
    virtual int prepare(int field, const std::string& in, std::string* out,
                        std::string* why) const = 0;
    /* Queue a request; refused at once, with `why`, when it can never run. */
    virtual int submit(std::unique_ptr<gen::AudioRequest> r, std::string* why) = 0;
    virtual int64_t queued() const = 0;
    virtual int64_t running() const = 0;
};

/* The video generator (core/gen/video_driver.h), for a model that makes a video and its
 * soundtrack from text: its declared prompt template, sizes and lengths, and its request queue.
 * The server renders and tokenises the prompt, queues the request as a job, and turns what the
 * ticket hands back into an MP4 file. */
struct VideoGenInfo {
    std::string name, tmpl;
    int64_t max_text = 0;
    int64_t fps_num = 1, fps_den = 1;
    int64_t frames_per_chunk = 1, frames_tail = 0, min_chunks = 1, max_chunks = 1;
    int64_t size_multiple = 32, min_side = 0, max_area = 0, max_aspect = 1;
    int64_t default_width = 0, default_height = 0, default_chunks = 1;
    int64_t sample_rate = 0, channels = 0;
    int64_t default_steps = 0, max_steps = 0, max_lanes = 1;
    bool    fixed_steps = false;   /* a trained schedule: max_steps grid points and no other */
    /* A latent upscaler: the largest factor (0: none) and the full-size evaluations by default. */
    double  upscale_max = 0.0;
    int64_t upscale_refine = 0;
    /* `steps` counts model evaluations rather than sigma grid points (RadVideoGenDecl form 1). */
    bool    step_evals = false;
    /* Guidance (RadVideoGenDecl::guidance): the terms a request may name, their defaults and the
     * negative prompt a guided request gets when it names none. */
    bool    guidance = false;
    double  guidance_default[8] = {};
    std::string default_negative;
    /* Image to video: a first frame is taken, after an H.264 round trip at image_crf when > 0. */
    bool    image_cond = false;
    int32_t image_crf = 0;
    /* A length head: a request that names no length gets the one the model predicts. */
    bool    length_head = false;
    /* The decoders a request may name, the default first, and each one's largest frame in pixels
     * (RadVideoGenDecl::decoders); empty for a model with one. */
    std::vector<std::string> decoders;
    std::vector<int64_t>     decoder_max_area;
};

class IVideoGen {
public:
    virtual ~IVideoGen() = default;
    virtual const VideoGenInfo& info() const = 0;
    /* Queue a request; refused at once, with `why`, when it can never run. */
    virtual int submit(std::unique_ptr<gen::VideoRequest> r, std::string* why) = 0;
    virtual int64_t queued() const = 0;
    virtual int64_t running() const = 0;
};

}  /* namespace server */
}  /* namespace rad */
