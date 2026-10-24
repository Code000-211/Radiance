/* engine_diagnose.cpp -- the --diagnose driver: a fixed workload the engine runs on itself in
 * place of the HTTP server, and the report it writes. The arithmetic and the report format are
 * core/diag.h's; this file only drives the scheduler and collects what it needs.
 *
 * THE WORKLOAD IS A SERVING WORKLOAD. Prompts go through the model's chat template with the
 * deployment's reasoning default, sampling is the server's default (the same resolve the HTTP
 * path uses) with a fixed seed per request, and pipelining and drafting stay as configured -- so
 * what is measured is the engine that serves, not a benchmark mode of it. No HTTP, no stop
 * strings.
 *
 * NOTHING IS SERVED FROM THE PREFIX CACHE. Every prompt is salted at its head with a tag unique to
 * the request (diag::salt_messages), the cache is emptied before every run, and every request's
 * cached-token count is checked afterwards: a run the cache served any of is reported, and the
 * report's verdict says the run is not a measurement. The cache itself stays on, so what it costs
 * a request that misses -- lookups, publishing, checkpoints -- is measured as a server pays it.
 *
 * PARITY. The tag is a counter, not a clock, so two builds run the same workload on the same
 * prompts with the same seeds, and every request's tokens are in the report. With
 * --diagnose-baseline the run compares them with an earlier report token for token
 * (diag::compare_outputs) and fails on any difference.
 *
 * Points, each run once unmeasured (first launches of a shape record passes and load code) and
 * then --diagnose-runs times, every run from an idle engine:
 *   categories     one stream a prompt of the corpus, ending on the model's EOS or max tokens
 *   concurrency    N streams submitted together, for each N of --diagnose-concurrency;
 *                  ignore_eos, so all N decode for the whole run
 *   prefill        one prompt of each length of --diagnose-prefill, then a short decode at that
 *                  depth
 *   interference   streams decoding, then one prompt arriving: its TTFT under load and the gap it
 *                  puts into the running streams
 */
#include "engine_priv.h"
#include "diag.h"

#include "build/startup.h"
#include "device/device.h"
#include "server/adapters.h"
#include "server/sink.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>


extern char** environ;

namespace rad {

namespace {

using json = diag::json;

int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* Every instrument and knob the environment can set, so a report says what it was measured
 * under; several of these are read once into statics and cannot be seen any other way. */
json env_snapshot() {
    json e = json::object();
    for (char** p = environ; p && *p; ++p) {
        const char* kv = *p;
        if (std::strncmp(kv, "RADIANCE_", 9) && std::strncmp(kv, "R4D_", 4) && std::strncmp(kv, "RAD_", 4) &&
            std::strncmp(kv, "AVX_", 4))
            continue;
        const char* eq = std::strchr(kv, '=');
        if (!eq) continue;
        e[std::string(kv, (size_t)(eq - kv))] = std::string(eq + 1);
    }
    return e;
}

/* One request of a run. */
struct Job {
    std::vector<int32_t> prompt;
    int   max_tokens = 0;
    bool  ignore_eos = true;
    float temp = -1.0f;   /* < 0: the server's default */
};

double loadavg1() {
    FILE* f = std::fopen("/proc/loadavg", "r");
    if (!f) return -1;
    double v = -1;
    if (std::fscanf(f, "%lf", &v) != 1) v = -1;
    std::fclose(f);
    return v;
}

}  /* namespace */

int Engine::run_diagnose() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(cfg_.diagnose, ec);
    if (ec) {
        RAD_ERR("diagnose: cannot create '%s': %s", cfg_.diagnose.c_str(), ec.message().c_str());
        return RAD_E_IO;
    }

    json warnings = json::array();
    json problems = json::array();   /* what makes the run not a measurement: the verdict */
    auto warn = [&](const std::string& w) {
        RAD_WARN("diagnose: %s", w.c_str());
        warnings.push_back(w);
    };
    json not_measured = json::array({
        "per-kernel device time, launch gaps and host joins: needs dispatch timestamps inside the "
        "engine (docs/DIAGNOSE.md, step 3); rocprofv3 is not a substitute here",
        "placement, expert streaming and link counters per phase (docs/DIAGNOSE.md, step 1b)",
        "op correctness (repeat, oracle) over a step window (docs/DIAGNOSE.md, step 4)",
        "the HTTP path, detokenisation and streaming to a client: a client-side benchmark measures "
        "those, and the difference to these numbers is their cost",
    });

    /* ---- prompts: the chat template where the container has one, as the server renders it ---- */
    server::TokenizerBridge tok(vocab_);
    server::ChatTemplateBridge chat;
    const RadChatFormat* fmt = rad_tools_chat_format(plugins_, meta_);
    const bool use_chat = vocab_ && !vocab_->chat_template().empty() &&
                          chat.init(*vocab_, fmt, plugins_.arch_plugin) >= 0 && chat.ready();
    if (!use_chat)
        warn("no usable chat template: prompts are raw text, so draft acceptance and tokens a step "
             "are not those of chat traffic");

    /* A counter, the same in every process, so a baseline and a candidate see the same prompts. A
     * disk tier of the prefix cache outlives the process and could serve a second run its first
     * run's prompts; the cached-token check after every run catches that, and it is warned of. */
    if (cfg_.prefix_cache_disk_mib > 0)
        warn("the prefix cache has a disk tier, which outlives this process: a second diagnose run "
             "would be served from it and its runs refused. Run without --prefix-cache-disk-mib");
    uint64_t salt_n = 0;
    auto render = [&](const json& messages, std::vector<int32_t>* out) -> int {
        char tag[64];
        std::snprintf(tag, sizeof tag, "diagnose request %llu", (unsigned long long)++salt_n);
        const json msgs = diag::salt_messages(messages, tag);
        if (use_chat) {
            server::ChatRenderOptions o;
            const std::string& def = cfg_.reasoning_effort;
            if (def == "none" || def == "minimal") o.enable_thinking = false;
            else if (!def.empty()) o.template_kwargs.emplace("reasoning_effort", json(def).dump());
            server::ChatRender r;
            std::string why;
            const int st = chat.apply(msgs, json::array(), o, &r, &why);
            if (st < 0) {
                RAD_ERR("diagnose: the chat template refused a workload prompt: %s", why.c_str());
                return st;
            }
            *out = tok.encode(r.prompt, false, /*parse_special=*/true);
        } else {
            std::string text;
            for (const json& m : msgs)
                if (m["content"].is_string()) text += m["content"].get<std::string>() + "\n\n";
            *out = tok.encode(text, true, /*parse_special=*/false);
        }
        if (out->empty()) {
            RAD_ERR("diagnose: a workload prompt tokenised to nothing");
            return RAD_E_STATE;
        }
        return RAD_OK;
    };
    auto user = [](const std::string& text) {
        return json::array({ json{ { "role", "user" }, { "content", text } } });
    };

    const int64_t ctx_limit = cfg_.max_ctx > 0 ? (int64_t)cfg_.max_ctx : (int64_t)meta_.n_ctx_train;
    const int dec_tokens = cfg_.diagnose_tokens;
    const int runs = cfg_.diagnose_runs;
    if (cfg_.profile_ops)
        warn("--profile-ops is on: it turns recorded passes off and adds barriers, so every step time "
             "here is of a different engine than the one that serves");
    if (const char* h = std::getenv("RADIANCE_HEAT_RESIDENCY"); h && std::atoi(h) > 0)
        warn("RADIANCE_HEAT_RESIDENCY > 0 forces active experts to zero-copy over the link");
    if (!cfg_.prefix_cache)
        warn("--no-prefix-cache: requests do not pay the cache's lookups and publishing, which a "
             "server with the cache on does");
    const double load0 = loadavg1();
    const unsigned ncpu = std::max(1u, std::thread::hardware_concurrency());
    if (load0 > (double)ncpu / 4)
        warn("the host is busy (load average " + std::to_string(load0) + " on " + std::to_string(ncpu) +
             " CPUs): the scheduler's planning between steps competes with it, and emission gaps show that");

    /* A prompt of about `want` tokens: a document to summarise, sized by a first render and
     * scaled, because the template adds tokens of its own and a vocabulary's characters a token is
     * not known in advance. */
    auto sized_prompt = [&](int64_t want, uint32_t doc_salt, std::vector<int32_t>* out) -> int {
        const std::string ask = "Summarise the document above in detail.";
        int64_t chars = want * 4;
        for (int pass = 0; pass < 4; ++pass) {
            RAD_TRY(render(user(diag::synthetic_document(chars, doc_salt) + "\n\n" + ask), out));
            const int64_t got = (int64_t)out->size();
            if (got <= want && got >= want - std::max<int64_t>(8, want / 50)) return RAD_OK;
            chars = std::max<int64_t>(16, (int64_t)((double)chars * (double)want / (double)got * 0.99));
        }
        if ((int64_t)out->size() > want) {
            RAD_ERR("diagnose: could not fit a prefill prompt in %lld tokens", (long long)want);
            return RAD_E_STATE;
        }
        return RAD_OK;
    };

    const SamplingParams sp0 = resolve_sampling_defaults(meta_, file_ != nullptr, cfg_);
    uint64_t next_id = 1;
    /* TEMPERATURE 1 FOR EVERY REQUEST, whatever the server's default: it is the temperature the
     * model is served at, and draft acceptance -- what an adaptive draft depth is tuned on -- falls
     * with temperature, so a greedy workload would tune it for traffic that never comes. top_k,
     * top_p and min_p stay the server's. Parity does not need greedy: a fixed seed per request
     * makes bit-identical logits sample identical tokens at any temperature. */
    const float kTemp = 1.0f;

    /* ---- one run: submit, wait, collect ----
     * `late` is submitted once every request of `early` has produced `late_after` tokens; the
     * early requests are cancelled once every late one has produced `late_after` tokens too, so a
     * stream started only to be interfered with does not decode to its limit. */
    auto run_set = [&](const std::string& name, diag::PhaseKind kind, const std::vector<Job>& early,
                       const std::vector<Job>& late, int late_after, diag::PhaseInput* in) -> int {
        for (const auto* set : { &early, &late })
            for (const Job& jb : *set)
                if ((int64_t)jb.prompt.size() + jb.max_tokens > ctx_limit) {
                    RAD_ERR("diagnose: %s needs %zu + %d tokens of context and the engine serves %lld",
                            name.c_str(), jb.prompt.size(), jb.max_tokens, (long long)ctx_limit);
                    return RAD_E_INVAL;
                }
        sched_.reset_prefix_cache();
        const SchedMetrics before = sched_.metrics();
        const uint64_t seq0 = sched_.step_record_count();

        std::vector<std::unique_ptr<server::Sink>> sinks;
        auto submit = [&](const std::vector<Job>& jobs) -> int {
            std::vector<std::unique_ptr<Request>> reqs;
            for (const Job& jb : jobs) {
                sinks.push_back(std::make_unique<server::Sink>((size_t)jb.max_tokens + 64));
                auto r = std::make_unique<Request>();
                r->id = next_id++;
                r->prompt = jb.prompt;
                r->max_tokens = jb.max_tokens;
                r->ignore_eos = jb.ignore_eos;
                r->sp = sp0;
                if (jb.temp >= 0.0f) r->sp.temp = jb.temp;
                r->sp.seed = 0x5eedull * 7919ull + r->id;   /* fixed per request, so reruns sample alike */
                r->sink = sinks.back().get();
                sinks.back()->id = r->id;
                reqs.push_back(std::move(r));
            }
            std::vector<int> rc(reqs.size(), RAD_OK);
            /* All at once, under one hold of the queue: the streams are admitted into the same steps. */
            sched_.add_all(reqs.data(), reqs.size(), rc.data());
            for (int c : rc)
                if (c < 0) {
                    RAD_ERR("diagnose: the scheduler refused a %s request: %s", name.c_str(), rad_strerror(c));
                    return c;
                }
            return RAD_OK;
        };
        auto produced_all = [&](size_t from, size_t to, int64_t n) {
            for (size_t i = from; i < to; ++i)
                if (!sinks[i]->done() && sinks[i]->n_produced() < n) return false;
            return true;
        };
        auto wait_until = [&](const std::function<bool()>& ready) -> int {
            while (!ready()) {
                const int ls = loop_status_.load(std::memory_order_acquire);
                if (ls < 0) return ls;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return RAD_OK;
        };

        const int64_t t0 = steady_ns();
        RAD_TRY(submit(early));
        const size_t n_early = sinks.size();
        if (!late.empty()) {
            RAD_TRY(wait_until([&] { return produced_all(0, n_early, late_after); }));
            RAD_TRY(submit(late));
            RAD_TRY(wait_until([&] { return produced_all(n_early, sinks.size(), late_after); }));
            for (size_t i = 0; i < n_early; ++i)
                if (!sinks[i]->done()) sched_.cancel(sinks[i]->id);
        }
        for (size_t first = 0;;) {
            while (first < sinks.size() && sinks[first]->done()) ++first;
            if (first == sinks.size()) break;
            const int ls = loop_status_.load(std::memory_order_acquire);
            if (ls < 0) return ls;
            sinks[first]->wait(200);
        }
        /* A sink finishes before the step that produced its last token is charged to the
         * counters, so wait for the scheduler to go idle and its step count to stop moving. */
        for (uint64_t last = UINT64_MAX;;) {
            const uint64_t c = sched_.step_record_count();
            const SchedMetrics m = sched_.metrics();
            if (c == last && m.running == 0 && m.queue_depth == 0) break;
            last = c;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        *in = diag::PhaseInput{};
        in->name = name;
        in->kind = kind;
        in->streams = (int)sinks.size();
        in->wall_ns = steady_ns() - t0;
        in->before = before;
        in->after = sched_.metrics();
        in->steps_lost = sched_.step_records(seq0, &in->steps);
        int bad = 0;
        for (size_t i = 0; i < sinks.size(); ++i) {
            const server::Sink& s = *sinks[i];
            diag::RequestTiming t;
            const int64_t base = s.t_start_ns();
            t.start_ns = base;
            t.admitted_ns = s.t_admitted_ns() >= 0 ? base + s.t_admitted_ns() : -1;
            t.first_ns = s.t_first_token_ns() >= 0 ? base + s.t_first_token_ns() : -1;
            t.last_ns = s.t_last_token_ns() >= 0 ? base + s.t_last_token_ns() : -1;
            t.tokens = s.n_produced();
            t.prompt = s.n_prompt();
            t.cached = s.n_cached();
            t.drafted = s.n_drafted();
            t.accepted = s.n_draft_accepted();
            for (int32_t d = 0; d <= server::Sink::kMaxDepthRows; ++d) {
                const int64_t n = s.n_rows_at_depth(d);
                if (n > 0) t.depth_rows.push_back({d, n});
            }
            {
                std::vector<server::SinkToken> toks((size_t)std::max<int64_t>(s.n_produced(), 0) + 1);
                const size_t got = sinks[i]->read(toks.data(), toks.size());
                t.output.reserve(got);
                for (size_t k = 0; k < got; ++k) t.output.push_back(toks[k].token);
            }
            const server::Finish f = s.finish_reason();
            const bool cancelled_by_us = i < n_early && !late.empty() && f == server::Finish::Cancelled;
            t.ok = f == server::Finish::Length || f == server::Finish::Stop || cancelled_by_us;
            t.stopped = f == server::Finish::Stop;
            if (!t.ok) {
                RAD_ERR("diagnose: a %s request did not finish: %s", name.c_str(),
                        s.why() ? s.why() : rad_strerror(s.status()));
                ++bad;
            }
            in->requests.push_back(t);
            sched_.reap(s.id);
        }
        int64_t cached = 0;
        for (const auto& t : in->requests) cached += t.cached;
        if (cached > 0)
            problems.push_back(name + ": the prefix cache served " + std::to_string(cached) +
                               " prompt tokens, so this run measured a cache hit");
        if (in->steps_lost > 0)
            problems.push_back(name + ": " + std::to_string(in->steps_lost) +
                               " step records were overwritten before they were read");
        RAD_INFO("diagnose: %s done in %.2f s", name.c_str(), (double)in->wall_ns / 1e9);
        return bad ? RAD_E_STATE : RAD_OK;
    };
    /* A measured point: one unmeasured run, then `runs` measured ones. `make` builds a run's jobs
     * fresh each time, so every run has its own salt. */
    auto measure = [&](const std::string& name, diag::PhaseKind kind,
                       const std::function<int(bool warm, std::vector<Job>*)>& make,
                       std::vector<json>* out) -> int {
        diag::PhaseInput in;
        std::vector<Job> jobs;
        RAD_TRY(make(true, &jobs));
        RAD_TRY(run_set(name + "/warmup", kind, jobs, {}, 0, &in));
        for (int r = 0; r < runs; ++r) {
            jobs.clear();
            RAD_TRY(make(false, &jobs));
            RAD_TRY(run_set(name + "/" + std::to_string(r + 1), kind, jobs, {}, 0, &in));
            out->push_back(diag::phase_report(in));
        }
        return RAD_OK;
    };

    /* ---- categories ---- */
    std::vector<diag::CorpusEntry> corpus;
    if (!cfg_.diagnose_corpus.empty()) {
        std::string why;
        if (diag::load_corpus(cfg_.diagnose_corpus, &corpus, &why) < 0) {
            RAD_ERR("diagnose: %s", why.c_str());
            return RAD_E_INVAL;
        }
    } else {
        corpus = diag::builtin_corpus();
    }
    std::vector<std::string> cats;
    for (const auto& e : corpus)
        if (std::find(cats.begin(), cats.end(), e.category) == cats.end()) cats.push_back(e.category);
    json categories = json::array();
    for (const std::string& cat : cats) {
        std::vector<const diag::CorpusEntry*> es;
        for (const auto& e : corpus)
            if (e.category == cat) es.push_back(&e);
        std::vector<json> rs;
        for (size_t k = 0; k < es.size(); ++k) {
            const diag::CorpusEntry& e = *es[k];
            const int mt = e.max_tokens > 0 ? e.max_tokens : dec_tokens;
            RAD_TRY(measure("category " + cat + " #" + std::to_string(k + 1), diag::PhaseKind::Decode,
                            [&](bool warm, std::vector<Job>* jobs) -> int {
                                Job jb;
                                RAD_TRY(render(e.messages, &jb.prompt));
                                jb.max_tokens = warm ? std::min(mt, 32) : mt;
                                jb.ignore_eos = false;   /* the model's own lengths: that is the category */
                                jb.temp = kTemp;
                                jobs->push_back(std::move(jb));
                                return RAD_OK;
                            },
                            &rs));
        }
        int64_t reqs = 0, stopped = 0;
        for (const json& r : rs) {
            reqs += r["requests"]["count"].get<int64_t>();
            stopped += r["requests"]["stopped_on_eos"].get<int64_t>();
        }
        categories.push_back({ { "category", cat }, { "prompts", (int64_t)es.size() },
                               { "requests", reqs }, { "stopped_on_eos", stopped },
                               { "summary", diag::summarize_runs(rs) },
                               { "draft_by_position", diag::pool_draft_positions(rs) }, { "runs", rs } });
    }

    /* ---- concurrency ---- */
    json concurrency = json::array();
    for (int n : cfg_.diagnose_concurrency) {
        if (n > cfg_.max_seqs) {
            warn("concurrency " + std::to_string(n) + " skipped: --max-num-seqs is " + std::to_string(cfg_.max_seqs));
            continue;
        }
        std::vector<json> rs;
        RAD_TRY(measure("concurrency " + std::to_string(n), diag::PhaseKind::Decode,
                        [&](bool warm, std::vector<Job>* jobs) -> int {
                            for (int s = 0; s < n; ++s) {
                                Job jb;
                                RAD_TRY(render(corpus[(size_t)s % corpus.size()].messages, &jb.prompt));
                                jb.max_tokens = warm ? std::min(dec_tokens, 32) : dec_tokens;
                                jb.ignore_eos = true;   /* every stream decodes the whole run */
                                jb.temp = kTemp;
                                jobs->push_back(std::move(jb));
                            }
                            return RAD_OK;
                        },
                        &rs));
        concurrency.push_back({ { "streams", n }, { "summary", diag::summarize_runs(rs) },
                                { "draft_by_position", diag::pool_draft_positions(rs) }, { "runs", rs } });
    }

    /* ---- prefill ---- */
    const int depth_tokens = std::min(dec_tokens, 64);
    json prefill = json::array();
    uint32_t doc_salt = 1;
    for (int want : cfg_.diagnose_prefill) {
        if ((int64_t)want + depth_tokens > ctx_limit) {
            warn("prefill " + std::to_string(want) + " skipped: the engine serves " + std::to_string(ctx_limit) +
                 " tokens of context");
            continue;
        }
        std::vector<json> rs;
        int64_t got = 0;
        RAD_TRY(measure("prefill " + std::to_string(want), diag::PhaseKind::Prefill,
                        [&](bool warm, std::vector<Job>* jobs) -> int {
                            Job jb;
                            RAD_TRY(sized_prompt(want, doc_salt++, &jb.prompt));
                            got = (int64_t)jb.prompt.size();
                            jb.max_tokens = warm ? 1 : depth_tokens;
                            jb.ignore_eos = true;
                            jb.temp = kTemp;
                            jobs->push_back(std::move(jb));
                            return RAD_OK;
                        },
                        &rs));
        prefill.push_back({ { "requested_tokens", want }, { "prompt_tokens", got },
                            { "summary", diag::summarize_runs(rs) }, { "runs", rs } });
    }

    /* ---- interference: a prompt arriving into running streams ---- */
    json interference = nullptr;
    {
        const int streams = (int)std::min<int64_t>(4, cfg_.max_seqs - 1);
        /* The arriving prompt: the longest measured prefill up to 16k, so its idle TTFT is known. */
        const json* pick = nullptr;
        for (const json& p : prefill)
            if (p["requested_tokens"].get<int>() <= 16384) pick = &p;
        if (!pick && !prefill.empty()) pick = &prefill[0];
        if (streams < 1 || !pick) {
            not_measured.push_back("a prompt arriving into running streams: needs --max-num-seqs of 2 or "
                                   "more and a measured prefill length");
        } else {
            const int want = (*pick)["requested_tokens"].get<int>();
            /* The running streams must outlast the arrival's prefill; they are cancelled once it has
             * decoded a little, so their limit only has to be out of reach. */
            const int run_tokens = (int)std::min<int64_t>(1 << 16, ctx_limit / 2);
            const int after = 16;
            std::vector<json> rs;
            for (int r = -1; r < runs; ++r) {
                std::vector<Job> early, late(1);
                for (int s = 0; s < streams; ++s) {
                    Job jb;
                    RAD_TRY(render(corpus[(size_t)s % corpus.size()].messages, &jb.prompt));
                    jb.max_tokens = run_tokens;
                    jb.ignore_eos = true;
                    jb.temp = kTemp;
                    early.push_back(std::move(jb));
                }
                RAD_TRY(sized_prompt(want, doc_salt++, &late[0].prompt));
                late[0].max_tokens = after;
                late[0].ignore_eos = true;
                late[0].temp = kTemp;
                diag::PhaseInput in;
                const std::string name = "interference/" + (r < 0 ? std::string("warmup") : std::to_string(r + 1));
                RAD_TRY(run_set(name, diag::PhaseKind::Decode, early, late, after, &in));
                if (r < 0) continue;
                const diag::RequestTiming& a = in.requests.back();
                json x;
                x["arrival_ttft_ms"] = a.admitted_ns >= 0 && a.first_ns >= 0
                                           ? json(std::round((double)(a.first_ns - a.admitted_ns) / 1e3) / 1e3)
                                           : json(nullptr);
                /* The running streams' steps only: the arrival's own decode would fill the gap. */
                x["interference"] = diag::interference_report(in.steps, a.admitted_ns, a.first_ns);
                x["phase"] = diag::phase_report(in);
                rs.push_back(x);
            }
            json idle = nullptr;
            const json& ps = (*pick)["summary"];
            if (ps.contains("requests.ttft_ms.p50") && ps["requests.ttft_ms.p50"].is_object())
                idle = ps["requests.ttft_ms.p50"]["median"];
            interference = { { "streams", streams }, { "prompt_tokens", (*pick)["prompt_tokens"] },
                             { "idle_ttft_ms", idle },
                             { "summary", diag::summarize_paths(rs, { "/arrival_ttft_ms",
                                                                      "/interference/typical_gap_ms",
                                                                      "/interference/longest_gap_ms",
                                                                      "/interference/stall_ratio" }) },
                             { "runs", rs } };
        }
    }

    /* ---- the report ---- */
    json rep;
    json eng = json::parse(engine_info(), nullptr, /*allow_exceptions=*/false);
    if (eng.is_discarded()) eng = json::object();
    eng["tp"] = cfg_.tp;
    eng["backend"] = device_is_host() ? "host" : "device";
    eng["plan"] = cfg_.deterministic ? "deterministic (static placement)" : "dynamic placement";
    eng["max_num_seqs"] = cfg_.max_seqs;
    eng["context"] = ctx_limit;
    rep["engine"] = eng;
    json w;
    w["description"] = std::string(use_chat ? "chat-templated" : "raw-text") + " prompts, each salted at its head; " +
                       std::to_string(runs) + " measured run(s) a point after one unmeasured; categories from " +
                       (cfg_.diagnose_corpus.empty() ? std::string("the built-in corpus") : cfg_.diagnose_corpus) +
                       " up to " + std::to_string(dec_tokens) + " tokens, ending on EOS; concurrency streams " +
                       std::to_string(dec_tokens) + " tokens with ignore_eos; prefill then " +
                       std::to_string(depth_tokens) + " tokens of decode; temperature 1 with the server's "
                       "top_k/top_p/min_p and a fixed seed per request; prefix cache emptied before every run";
    w["sampling"] = { { "temp", kTemp }, { "top_k", sp0.top_k }, { "top_p", sp0.top_p }, { "min_p", sp0.min_p } };
    w["reasoning_effort"] = cfg_.reasoning_effort;
    w["runs"] = runs;
    w["tokens"] = dec_tokens;
    {
        /* What decides the prompts and the seeds: two reports with the same fingerprint ran the
         * same requests in the same order. */
        json fp;
        json c = json::array();
        for (const auto& e : corpus) c.push_back({ e.category, e.messages, e.max_tokens });
        fp["corpus"] = c;
        fp["tokens"] = dec_tokens;
        fp["runs"] = runs;
        fp["concurrency"] = cfg_.diagnose_concurrency;
        fp["prefill"] = cfg_.diagnose_prefill;
        fp["max_num_seqs"] = cfg_.max_seqs;
        fp["context"] = ctx_limit;
        fp["chat_template"] = use_chat;
        fp["reasoning_effort"] = cfg_.reasoning_effort;
        fp["sampling"] = w["sampling"];
        w["fingerprint"] = diag::workload_fingerprint(fp);
    }
    rep["workload"] = w;
    json envj;
    envj["variables"] = env_snapshot();
    envj["loadavg_start"] = load0;
    envj["loadavg_end"] = loadavg1();
    envj["cpus"] = ncpu;
    rep["environment"] = envj;
    rep["verdict"] = problems.empty() ? std::string("valid") : std::string("NOT A MEASUREMENT -- see problems");
    rep["problems"] = problems;
    for (const auto& p : problems) warnings.push_back(p);
    rep["warnings"] = warnings;
    rep["categories"] = categories;
    rep["concurrency"] = concurrency;
    rep["prefill"] = prefill;
    rep["interference"] = interference;
    rep["not_measured"] = not_measured;

    bool parity_ok = true;
    if (!cfg_.diagnose_baseline.empty()) {
        std::ifstream f(cfg_.diagnose_baseline);
        const json base = json::parse(f, nullptr, /*allow_exceptions=*/false);
        std::string why;
        json cmp;
        if (!f.is_open() || base.is_discarded()) {
            why = "cannot read " + cfg_.diagnose_baseline + " as JSON";
        } else if (diag::compare_outputs(rep, base, &cmp, &why) == RAD_OK) {
            cmp["baseline"] = cfg_.diagnose_baseline;
            rep["parity"] = cmp;
            parity_ok = cmp["verdict"] == "identical" && cmp["acceptance_verdict"] == "no drop";
            RAD_INFO("diagnose: parity %s, %lld of %lld requests identical; draft acceptance: %s",
                     cmp["verdict"].get<std::string>().c_str(), (long long)cmp["identical"].get<int64_t>(),
                     (long long)cmp["compared"].get<int64_t>(), cmp["acceptance_verdict"].get<std::string>().c_str());
        }
        if (!why.empty()) {
            rep["parity"] = { { "baseline", cfg_.diagnose_baseline }, { "verdict", "not compared: " + why } };
            parity_ok = false;
            RAD_ERR("diagnose: %s", why.c_str());
        }
    }

    const fs::path dir(cfg_.diagnose);
    {
        std::ofstream f(dir / "diagnose.json");
        f << rep.dump(2) << "\n";
        if (!f) {
            RAD_ERR("diagnose: cannot write %s", (dir / "diagnose.json").c_str());
            return RAD_E_IO;
        }
    }
    {
        std::ofstream f(dir / "diagnose.md");
        f << diag::render_markdown(rep);
        if (!f) {
            RAD_ERR("diagnose: cannot write %s", (dir / "diagnose.md").c_str());
            return RAD_E_IO;
        }
    }
    RAD_INFO("diagnose: report in %s", (dir / "diagnose.md").c_str());
    return problems.empty() && parity_ok ? RAD_OK : RAD_E_STATE;
}

}  /* namespace rad */
