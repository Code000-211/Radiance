/* config_test.cpp -- the command line.
 *
 * Every pool is budgeted explicitly (spec §6), and downstream a mistyped budget flag looks like a
 * decision: nothing after the parser can tell a requested value from one that failed to parse and
 * fell back to a default. So the parser refuses by name, and these tests keep it refusing.
 *
 *   Taken     the value reaches the field an operator expects
 *   Refused   a value outside the legal set stops the server instead of being rounded into it
 *   Resolved  what is left unsaid comes from the environment or a stated default
 *
 * No device, no model, no container: argc/argv in, a Config out.
 */
#include "rad_test.h"

#include "rad_core.h"
#include "diag/diag.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <regex>
#include <sstream>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <vector>

using namespace rad;

namespace {

/* Parse with a --model prepended, so the required-argument check does not trip every case. */
int parse(std::initializer_list<const char*> v, Config* c) {
    std::vector<std::string> args{ "--model", "/m/x.rad" };
    args.insert(args.end(), v.begin(), v.end());
    Argv a(std::move(args));
    return config_parse(a.argc(), a.argv.data(), c);
}

/* Parse into a fresh Config, for cases that must not inherit flags from an earlier parse. */
int parse_fresh(std::initializer_list<const char*> v) {
    Config c;
    return parse(v, &c);
}

/* Restores the log level, which `-v` changes and every test in this binary reads. */
struct LogGuard {
    Log saved = log_level();
    ~LogGuard() { log_set_level(saved); }
};

}  /* namespace */

/* ================================================================== the shape of the call */

TEST(help_is_reported_separately_from_an_error) {
    /* 1, not an error: the caller prints usage and exits 0, so `--help` is not a failed run. */
    Config c;
    Argv a({ "--help" });
    CHECK_EQ(config_parse(a.argc(), a.argv.data(), &c), 1);
    Argv b({ "-h" });
    CHECK_EQ(config_parse(b.argc(), b.argv.data(), &c), 1);
    /* It wins from anywhere on the line, even after an otherwise fatal argument. */
    Argv d({ "--tp", "0", "--help" });
    CHECK_EQ(config_parse(d.argc(), d.argv.data(), &c), 1);
}

/* An unknown flag is an error naming it, not a warning: a typo in a budget flag would otherwise
 * yield a plan the operator did not ask for. */
TEST(an_unknown_option_is_refused) {
    Config c;
    CHECK_EQ(parse({ "--not-a-flag" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-num-seq", "8" }, &c), RAD_E_INVAL);   /* near-miss on a real flag */
    CHECK_EQ(parse({ "-x" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "positional" }, &c), RAD_E_INVAL);
}

/* A retired flag is refused, not accepted and ignored. Each names a decision the engine does not
 * take from the operator: the KV cache is elastic and derives its own buffer, the activation arena
 * comes from the buffer plan, and a finished conversation is copied to host memory as soon as
 * possible rather than after an idle time. Starting anyway would serve something other than what
 * the command line expects; a warning would go unread. */
TEST(a_retired_flag_is_refused) {
    Config c;
    CHECK_EQ(parse({ "--kv-flex-mib", "512" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--vram-headroom-mib", "2000" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-idle-to-host", "30" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-idle-to-disk", "1800" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-host-cache-mib", "4096" }, &c), RAD_E_INVAL);
}

/* A flag written last with no argument is refused rather than reading past argv. */
TEST(a_flag_missing_its_argument_is_refused_rather_than_read_past) {
    Config c;
    CHECK_EQ(parse({ "--tp" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--port" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-cache-dtype" }, &c), RAD_E_INVAL);
    Argv a({ "--model" });
    CHECK_EQ(config_parse(a.argc(), a.argv.data(), &c), RAD_E_INVAL);
}

/* KL mode records or scores, never both, and requires that direction's inputs. Guessing between
 * them could make a quantised model the reference. */
TEST(the_kl_mode_records_or_scores_and_says_which_input_it_lacks) {
    Config c;
    CHECK_EQ(parse({ "--kld-record", "/r", "--kld-corpus", "/c.jsonl" }, &c), RAD_OK);
    CHECK_EQ(c.kld_record, std::string("/r"));
    CHECK_EQ(c.kld_corpus, std::string("/c.jsonl"));
    CHECK(c.kld());

    Config d;
    CHECK_EQ(parse({ "--kld-ref", "/r", "--kld-out", "/o.json" }, &d), RAD_OK);
    CHECK_EQ(d.kld_ref, std::string("/r"));
    CHECK_EQ(d.kld_out, std::string("/o.json"));
    CHECK(d.kld());

    Config e;
    CHECK_EQ(parse({}, &e), RAD_OK);
    CHECK(!e.kld());

    Config f;
    CHECK_EQ(parse({ "--kld-record", "/r", "--kld-ref", "/s" }, &f), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kld-record", "/r" }, &f), RAD_E_INVAL);              /* no corpus */
    CHECK_EQ(parse({ "--kld-ref", "/r", "--kld-corpus", "/c" }, &f), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kld-out", "/o.json" }, &f), RAD_E_INVAL);            /* nothing to report */
    CHECK_EQ(parse({ "--kld-ref" }, &f), RAD_E_INVAL);
}

/* Runtime LoRA: the adapters reach their fields in order, the targets as sorted leaf names, and a
 * combination that would size slots no adapter uses, or name an adapter that is not served, is
 * refused by name. A text model's slot count against its adapters, and its targets, are checked
 * at startup, where the engine knows whether the model generates text. */
TEST(the_lora_flags_reach_their_fields_and_refuse_what_serves_no_adapter) {
    Config c;
    CHECK_OK(parse({ "--lora", "b=/ad/b", "--lora", "a=/ad/a", "--max-loras", "3",
                     "--max-lora-rank", "32", "--lora-target-modules", "v_proj, q_proj,v_proj" },
                   &c));
    REQUIRE_EQ(c.lora.adapters.size(), (size_t)2);
    CHECK_EQ(c.lora.adapters[0].first, std::string("b"));
    CHECK_EQ(c.lora.adapters[1].second, std::string("/ad/a"));
    CHECK_EQ(c.lora.slots(), 3);
    CHECK_EQ(c.lora.max_rank, 32);
    CHECK(c.lora.targets == std::vector<std::string>({ "q_proj", "v_proj" }));
    Config d;
    CHECK_OK(parse({}, &d));
    CHECK(!d.lora.enabled());
    CHECK_EQ(d.lora.rank(false), 16);
    CHECK_EQ(d.lora.rank(true), 64);
    Config e;
    const std::string dir = radtest::tmp_path("loradir");
    mkdir(dir.c_str(), 0755);
    CHECK_OK(parse({ "--lora-dir", dir.c_str() }, &e));
    CHECK(e.lora.enabled());
    CHECK_EQ(e.lora.slots(), 2);
    CHECK_EQ(e.lora.rank(true), 64);

    auto refused = [](std::initializer_list<const char*> v, const char* why) {
        int st = RAD_OK;
        const std::string log = stderr_of([&] { st = parse_fresh(v); });
        if (st != RAD_E_INVAL || log.find(why) == std::string::npos)
            std::fprintf(stderr, "    expected a refusal naming '%s', got %d: %s", why, st,
                         log.c_str());
        CHECK_EQ(st, (int)RAD_E_INVAL);
        CHECK(log.find(why) != std::string::npos);
    };
    refused({ "--lora", "/ad/a" }, "NAME=PATH");
    refused({ "--lora", "a=" }, "NAME=PATH");
    refused({ "--lora", "a=/x", "--lora", "a=/y" }, "names the adapter 'a' twice");
    refused({ "--max-lora-rank", "32" }, "--max-lora-rank sizes runtime LoRA slots");
    refused({ "--lora-target-modules", "q_proj" }, "--lora-target-modules sizes");
    refused({ "--lora-dir", "/no/such/dir" }, "is not a directory");
    refused({ "--lora", "a=/x", "--lora-target-modules", "self_attn.q_proj" }, "leaf names");
    refused({ "--lora", "a=/x", "--lora-target-modules", ".*proj" }, "leaf names");
    refused({ "--lora", "a=/x", "--lora-target-modules", " , " }, "names no module");
    refused({ "--lora", "a=/x", "--kld-lora", "a" }, "in KL mode");
    refused({ "--lora", "a=/x", "--kld-lora", "b", "--kld-ref", "/r" }, "'b' is not one of");
    refused({ "--lora", "a=/x", "--kld-ref", "/r", "--kld-lora-sources", "d" }, "give --kld-lora");
}

TEST(the_model_is_required_and_nothing_else_is) {
    Config c;
    Argv none({});
    CHECK_EQ(config_parse(none.argc(), none.argv.data(), &c), RAD_E_INVAL);

    Config ok;
    CHECK_EQ(parse({}, &ok), RAD_OK);
    CHECK_EQ(ok.model, std::string("/m/x.rad"));
}

/* ================================================================== taken */

TEST(the_pool_size_flags_are_gone_and_say_what_replaced_them) {
    /* The card's split belongs to the resolver (core/mem/vram_budget.h); a launch line that sizes
     * a pool is refused by name rather than served with a different split. */
    Config c;
    CHECK_EQ(parse({ "--vram-weights-mib", "20480" }, &c), RAD_E_INVAL);
    Config k;
    CHECK_EQ(parse({ "--vram-kv-mib", "6144" }, &k), RAD_E_INVAL);
}

TEST(the_prefix_cache_disk_flags_are_gone_and_the_host_size_was_renamed) {
    /* The prefix cache has no disk tier. A launch line naming the disk tier, or the host tier's
     * old flag name, is refused. */
    for (const char* f : { "--prefix-cache-host-mib", "--prefix-cache-disk-mib", "--prefix-cache-dir" }) {
        Config c;
        CHECK_EQ(parse({ f, "4096" }, &c), RAD_E_INVAL);
    }
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.prefix_cache_mib, 0ll);
    CHECK_EQ(d.prefix_cache_ttl, 3600ll);              /* an hour */
    Config z;
    CHECK_EQ(parse({ "--prefix-cache-ttl", "0" }, &z), RAD_OK);
    CHECK_EQ(z.prefix_cache_ttl, 0ll);                 /* never */
    Config n;
    CHECK_EQ(parse({ "--prefix-cache-ttl", "-1" }, &n), RAD_E_INVAL);
    Config h;
    CHECK_EQ(parse({ "--prefix-cache-ttl", "1h" }, &h), RAD_E_INVAL);
}

TEST(the_sizes_and_counts_reach_the_fields_they_name) {
    Config c;
    CHECK_EQ(parse({ "--tp", "2",
                     "--max-num-batched-tokens", "512",
                     "--max-num-seqs", "8",
                     "--max-model-len", "200000",
                     "--host-pool-mib", "12288",
                     "--prefix-cache-mib", "4096",
                     "--prefix-cache-ttl", "600",
                     "--checkpoint-interval", "2048",
                     "--checkpoint-slots", "8",
                     "--tp-wire-min-kb", "64",
                     "--port", "8100" }, &c), RAD_OK);
    CHECK_EQ(c.tp, 2);
    CHECK_EQ(c.max_tok, 512ll);
    CHECK_EQ(c.max_seqs, 8ll);
    CHECK_EQ(c.max_ctx, 200000ll);
    CHECK_EQ(c.host_pool_mib, 12288ll);
    CHECK_EQ(c.prefix_cache_mib, 4096ll);
    CHECK_EQ(c.prefix_cache_ttl, 600ll);
    CHECK_EQ(c.checkpoint_interval, 2048ll);
    CHECK_EQ(c.checkpoint_slots, 8ll);
    CHECK_EQ(c.tp_wire_min_kb, 64ll);
    CHECK_EQ(c.port, 8100);

    /* --max-model-len may exceed INT_MAX and must not wrap. */
    Config big;
    CHECK_EQ(parse({ "--max-model-len", "4294967296" }, &big), RAD_OK);
    CHECK_EQ(big.max_ctx, 4294967296ll);
}

TEST(the_paths_and_names_reach_their_fields) {
    Config c;
    CHECK_EQ(parse({ "--radiance-home", "/rh",
                     "--weights-disk-tier",
                     "--placement", "expert_tiered",
                     "--host", "0.0.0.0",
                     "--generation-config", "/g.json",
                     "--reasoning-effort", "medium",
                     "--served-model-name", "qwen-coder",
                     "--override-chat-template", "/t.jinja" }, &c), RAD_OK);
    CHECK_EQ(c.radiance_home, std::string("/rh"));
    CHECK(c.weights_disk_tier);
    CHECK_EQ(c.placement, std::string("expert_tiered"));
    CHECK_EQ(c.host, std::string("0.0.0.0"));
    CHECK_EQ(c.generation_config, std::string("/g.json"));
    CHECK_EQ(c.reasoning_effort, std::string("medium"));
    CHECK_EQ(c.served_model_name, std::string("qwen-coder"));
    CHECK_EQ(c.override_chat_template, std::string("/t.jinja"));
}

/* The reasoning budget is a count or unset (-1); a negative value cannot be given. */
TEST(thinking_token_budget_takes_a_count_and_defaults_to_none) {
    Config c;
    CHECK_EQ(parse({}, &c), RAD_OK);
    CHECK_EQ(c.thinking_token_budget, (int64_t)-1);
    Config d;
    CHECK_EQ(parse({ "--thinking-token-budget", "2048" }, &d), RAD_OK);
    CHECK_EQ(d.thinking_token_budget, (int64_t)2048);
    Config e;
    CHECK_EQ(parse({ "--thinking-token-budget", "0" }, &e), RAD_OK);
    CHECK_EQ(e.thinking_token_budget, (int64_t)0);
    Config f;
    CHECK(parse({ "--thinking-token-budget", "-1" }, &f) < 0);
    Config g;
    CHECK(parse({ "--thinking-token-budget", "lots" }, &g) < 0);
}

TEST(the_boolean_flags_are_off_until_they_are_given) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(!d.deterministic);
    CHECK(d.prefix_cache);                 /* on by default; the flag turns it off */
    CHECK(!d.debug_graph);
    CHECK(!d.debug_selection);
    CHECK(!d.debug_placement);
    CHECK(!d.accept_reference_kernels);
    CHECK(!d.profile_ops);
    CHECK(!d.live_view);

    Config c;
    CHECK_EQ(parse({ "--deterministic", "--no-prefix-cache", "--debug-placement",
                     "--debug-accept-reference-kernels", "--profile-ops", "--live" }, &c), RAD_OK);
    CHECK(c.deterministic);
    CHECK(!c.prefix_cache);
    CHECK(c.debug_placement);
    CHECK(c.accept_reference_kernels);
    CHECK(c.profile_ops);
    CHECK(c.live_view);
}

/* The graph dump implies the selection table: without the kernel chosen for each node, the graph
 * does not show what will run. */
TEST(debug_graph_turns_the_selection_table_on_too) {
    Config c;
    CHECK_EQ(parse({ "--debug-graph" }, &c), RAD_OK);
    CHECK(c.debug_graph);
    CHECK(c.debug_selection);

    Config s;
    CHECK_EQ(parse({ "--debug-selection" }, &s), RAD_OK);
    CHECK(!s.debug_graph);                 /* not the reverse */
    CHECK(s.debug_selection);
}

TEST(the_verbosity_flags_move_the_log_level) {
    LogGuard guard;
    Config c;
    log_set_level(Log::Info);
    CHECK_EQ(parse({ "-v" }, &c), RAD_OK);
    CHECK((int)log_level() == (int)Log::Debug);
    CHECK_EQ(parse({ "-vv" }, &c), RAD_OK);
    CHECK((int)log_level() == (int)Log::Trace);
}

/* ================================================================== refused */

TEST(tp_below_one_is_refused) {
    Config c;
    CHECK_EQ(parse({ "--tp", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--tp", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--tp", "notanumber" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--tp", "1" }, &c), RAD_OK);
}

/* A span is a whole number of intervals, one or more; 1 is the default and plans as before. */
TEST(a_prefill_span_below_one_interval_is_refused) {
    Config c;
    CHECK_EQ(c.prefill_span, 1);
    CHECK_EQ(parse({ "--prefill-span", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--prefill-span", "-2" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--prefill-span", "2x" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--prefill-span", "4" }, &c), RAD_OK);
    CHECK_EQ(c.prefill_span, 4);
}

/* A budget flag takes a whole number; a typo is refused rather than read as 0 or its prefix. */
TEST(a_numeric_flag_refuses_what_is_not_a_number) {
    Config c;
    CHECK_EQ(parse({ "--max-num-seqs", "8x" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-num-batched-tokens", "4k" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--host-pool-mib", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--checkpoint-interval", "" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--port", "70000" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--num-speculative-tokens", "three" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--num-speculative-tokens", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--num-speculative-tokens", "auto" }, &c), RAD_OK);
    CHECK_EQ(c.n_spec, -1);
    CHECK_EQ(parse({ "--num-speculative-tokens", "3" }, &c), RAD_OK);
    CHECK_EQ(c.n_spec, 3);
}

/* --prefill-interleave is a step count, off by default and at most 64. */
TEST(prefill_interleave_takes_a_step_count_up_to_64) {
    Config d;
    REQUIRE_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.prefill_interleave, 0);
    Config c;
    CHECK_EQ(parse({ "--prefill-interleave", "4" }, &c), RAD_OK);
    CHECK_EQ(c.prefill_interleave, 4);
    CHECK_EQ(parse({ "--prefill-interleave", "65" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--prefill-interleave", "-1" }, &c), RAD_E_INVAL);
}

/* ================================================================== the table, the parser and the help agree
 *
 * One table (g_flags) drives the help text and the startup report, and a separate chain of ifs
 * parses. A flag in the table the parser does not take, a number read without the strict parser,
 * or a default the help states by hand that the code does not have, each passes every sampled
 * test. These walk the whole table. */
namespace {

/* Each option and its argument's metavariable as the help prints them ("" for a switch). */
std::vector<std::pair<std::string, std::string>> usage_options() {
    const std::string text = stderr_of([] { config_usage("radiance"); });
    std::vector<std::pair<std::string, std::string>> out;
    size_t at = 0;
    while (at < text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(at, nl - at);
        at = nl + 1;
        if (line.rfind("  -", 0) != 0) continue;
        std::istringstream in(line);
        std::string name, arg;
        in >> name >> arg;
        /* The help starts after the padded column; a metavariable is upper case. */
        const bool is_arg = !arg.empty() && line.find(name + " " + arg) == 2 &&
                            std::all_of(arg.begin(), arg.end(), [](char ch) {
                                return (ch >= 'A' && ch <= 'Z') || ch == '_';
                            });
        out.emplace_back(name, is_arg ? arg : "");
    }
    return out;
}

/* A legal value for every option that takes one, with any option it cannot be given without. */
const std::map<std::string, std::vector<std::string>>& legal_values() {
    static const std::map<std::string, std::vector<std::string>> v = {
        { "--model", { "/m/y.rad" } }, { "--download-dir", { "/d" } },
        { "--radiance-home", { "/h1:/h2" } }, { "--kernels", { "libref" } },
        { "--tp", { "2" } }, { "--tp-wire", { "wht6" } }, { "--tp-wire-min-kb", { "64" } },
        { "--p2p", { "off" } }, { "--max-num-batched-tokens", { "512" } },
        { "--busy-prefill-chunk", { "256" } }, { "--prefill-interleave", { "2" } },
        { "--max-num-seqs", { "4" } },
        { "--max-model-len", { "4096" } }, { "--rope-scaling", { "yarn:4" } },
        { "--gpu-headroom-mib", { "128" } }, { "--kv-cache-dtype", { "fp8" } },
        { "--host-pool-mib", { "1024" } }, { "--host-headroom-mib", { "2048" } },
        { "--embedding-placement", { "vram" } },
        { "--vision-tower-placement", { "ram" } }, { "--ngram-placement", { "disk" } },
        { "--prefix-cache-mib", { "256" } }, { "--prefix-cache-ttl", { "60" } },
        { "--placement", { "all_vram" } }, { "--checkpoint-interval", { "256" } },
        { "--prefill-span", { "2" } },
        { "--checkpoint-slots", { "4" } }, { "--checkpoint-policy", { "tip-only" } },
        { "--num-speculative-tokens", { "2" } }, { "--draft-window", { "fixed" } },
        { "--draft-policy", { "cost" } }, { "--draft-depth-tiers", { "0,2" } },
        { "--draft-calib-k", { "3,2,2,3" } },
        { "--draft-profile", { "/p.json" } },
        { "--draft-profile-out", { "/out.json" } },
        { "--host", { "127.0.0.1" } }, { "--port", { "8001" } }, { "--api-key", { "k" } },
        { "--served-model-name", { "m" } }, { "--mm-max-patches", { "1024" } },
        { "--control-vector", { "off" } },
        { "--generation-config", { "/g.json" } }, { "--override-chat-template", { "/t.jinja" } },
        { "--temp", { "0.7" } }, { "--top-k", { "40" } }, { "--top-p", { "0.9" } },
        { "--min-p", { "0.05" } }, { "--reasoning-effort", { "low" } },
        { "--reasoning-effort-map", { "high=xhigh" } }, { "--thinking-token-budget", { "128" } },
        { "--typical-p", { "0.9" } }, { "--presence-penalty", { "0.5" } },
        { "--frequency-penalty", { "0.5" } }, { "--repetition-penalty", { "1.1" } },
        { "--repeat-last-n", { "64" } }, { "--dry-multiplier", { "0.8" } },
        { "--dry-base", { "1.75" } }, { "--dry-allowed-length", { "2" } },
        { "--dry-penalty-last-n", { "-1" } }, { "--dry-sequence-breakers", { "[\"\\n\"]" } },
        { "--xtc-probability", { "0.5" } }, { "--xtc-threshold", { "0.1" } },
        { "--default-max-tokens", { "256" } }, { "--max-tokens-cap", { "1024" } },
        { "--max-n", { "4" } }, { "--max-queued-requests", { "4096" } },
        { "--max-stop-strings", { "8" } }, { "--max-stop-bytes", { "256" } },
        { "--chat-template-kwargs", { "{\"enable_thinking\": false}" } },
        { "--reasoning-format", { "none" } }, { "--http-threads", { "4" } },
        { "--read-timeout", { "10" } }, { "--write-timeout", { "10" } },
        { "--keep-alive-timeout", { "2" } }, { "--max-body-mib", { "64" } },
        { "--retry-after", { "2" } }, { "--image-min-pixels", { "1024" } },
        { "--image-max-pixels", { "4096" } }, { "--video-min-pixels", { "1024" } },
        { "--video-max-pixels", { "4096" } }, { "--video-fps", { "2" } },
        { "--video-min-frames", { "2" } }, { "--video-max-frames", { "8" } },
        { "--video-max-frame-tokens", { "70" } }, { "--image-tokens", { "280" } },
        { "--max-source-pixels", { "1000000" } },
        { "--kld-record", { "/r", "--kld-corpus", "/c.jsonl" } }, { "--kld-ref", { "/r" } },
        { "--kld-corpus", { "/c.jsonl", "--kld-record", "/r" } },
        { "--kld-out", { "/o.json", "--kld-ref", "/r" } },
        { "--lora", { "a=/adapters/a" } },
        { "--lora-dir", { "/" } },
        { "--max-loras", { "2", "--lora-target-modules", "q_proj" } },
        { "--max-lora-rank", { "32", "--lora", "a=/adapters/a" } },
        { "--lora-target-modules", { "q_proj,v_proj", "--max-loras", "1" } },
        { "--kld-lora", { "a", "--lora", "a=/adapters/a", "--kld-ref", "/r" } },
        { "--kld-lora-sources", { "doc/", "--kld-lora", "a", "--lora", "a=/adapters/a",
                                  "--kld-ref", "/r" } },
    };
    return v;
}

}  /* namespace */

/* Every option the table lists parses with a legal value (a switch with none) and is reported as
 * given on the command line. --draft-model is listed to be refused by name, and --help is not a
 * setting. */
TEST(every_listed_option_parses_and_is_reported_as_given) {
    LogGuard lg;
    const auto opts = usage_options();
    CHECK(opts.size() > 90);
    const std::vector<std::string> names = diag::config_option_names();
    int checked = 0;
    for (const auto& [name, arg] : opts) {
        if (name == "--help" || name == "--draft-model") continue;
        CHECK(std::find(names.begin(), names.end(), name) != names.end());
        std::vector<std::string> args{ "--model", "/m/x.rad", name };
        if (!arg.empty()) {
            auto it = legal_values().find(name);
            if (it == legal_values().end()) {
                std::fprintf(stderr, "    no legal value listed for %s %s\n", name.c_str(), arg.c_str());
                CHECK(false);
                continue;
            }
            args.insert(args.end(), it->second.begin(), it->second.end());
        }
        Config c;
        Argv a(args);
        int st = RAD_OK;
        const std::string log = stderr_of([&] { st = config_parse(a.argc(), a.argv.data(), &c); });
        if (st != RAD_OK) std::fprintf(stderr, "    %s refused: %s", name.c_str(), log.c_str());
        CHECK_EQ(st, (int)RAD_OK);
        auto g = c.given.find(name);
        CHECK(g != c.given.end() && g->second == "command line");
        ++checked;
    }
    CHECK(checked > 90);
}

/* A whole-number option refuses what is not one, and a number option a word: "8x" read as 8, ""
 * as 0 or "12k" as 12 would each be served as a decision. Seconds are whole numbers too. */
TEST(every_numeric_option_refuses_what_is_not_its_kind) {
    const std::vector<std::string> seconds = { "--prefix-cache-ttl", "--read-timeout",
                                               "--write-timeout", "--keep-alive-timeout",
                                               "--retry-after" };
    int n_int = 0, n_num = 0;
    for (const auto& [name, arg] : usage_options()) {
        const bool whole = arg == "N" ||
                           std::find(seconds.begin(), seconds.end(), name) != seconds.end();
        if (!whole && arg != "F") continue;
        (whole ? n_int : n_num)++;
        for (const char* bad : whole ? std::vector<const char*>{ "8x", "", "12k", "1.5", "high" }
                                     : std::vector<const char*>{ "high", "", "0.5x" }) {
            Config c;
            int st = RAD_OK;
            const std::string log = stderr_of([&] { st = parse({ name.c_str(), bad }, &c); });
            if (st != RAD_E_INVAL)
                std::fprintf(stderr, "    %s %s was not refused\n", name.c_str(), bad);
            CHECK_EQ(st, (int)RAD_E_INVAL);
            CHECK(has(log, name));
        }
    }
    std::fprintf(stderr, "    %d whole-number and %d number option(s)\n", n_int, n_num);
    CHECK(n_int >= 35);
    CHECK(n_num >= 12);
}

/* A default the help states as a number is the default a parse with no flags reports. */
TEST(every_default_the_help_states_is_the_one_reported) {
    const std::string text = stderr_of([] { config_usage("radiance"); });
    Config c;
    REQUIRE_EQ(parse({}, &c), RAD_OK);
    const std::vector<diag::ConfigRow> rows = diag::config_rows(c);
    /* The help of one option: its line and the indented lines after it. */
    std::string block, flag;
    int compared = 0;
    auto settle = [&] {
        std::smatch m;
        static const std::regex def(R"(\(default (-?[0-9][0-9.]*)\))");
        if (flag.empty() || !std::regex_search(block, m, def)) return;
        const std::string want = m[1];
        const diag::ConfigRow* row = nullptr;
        for (const diag::ConfigRow& r : rows) if (r.flag == flag) row = &r;
        REQUIRE(row != nullptr);
        const bool ok = row->value == want || row->value.rfind(want + " ", 0) == 0;
        if (!ok) std::fprintf(stderr, "    %s: help says %s, the report %s\n", flag.c_str(),
                              want.c_str(), row->value.c_str());
        CHECK(ok);
        ++compared;
    };
    size_t at = 0;
    while (at < text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(at, nl - at);
        at = nl + 1;
        if (line.rfind("  -", 0) == 0) {
            settle();
            std::istringstream in(line);
            in >> flag;
            block = line;
        } else if (line.rfind("    ", 0) == 0) {
            block += " " + line;
        } else {
            settle();
            flag.clear();
            block.clear();
        }
    }
    settle();
    CHECK(compared >= 8);
}

/* --placement names one of the planner's four strategies, refused at the command line rather than
 * at startup once the model is open. */
TEST(the_placement_is_one_of_the_planners_four) {
    for (const char* ok : { "all_vram", "layer_offload", "expert_tiered", "auto" }) {
        Config c;
        CHECK_EQ(parse({ "--placement", ok }, &c), RAD_OK);
        CHECK_EQ(c.placement, std::string(ok));
    }
    for (const char* bad : { "all-vram", "AUTO", "", "tiered" }) {
        Config c;
        int st = RAD_OK;
        const std::string log = stderr_of([&] { st = parse({ "--placement", bad }, &c); });
        CHECK_EQ(st, (int)RAD_E_INVAL);
        CHECK(has(log, "--placement"));
    }
}

/* Three states, and the absent one is the zero-cost default: off and on load the module, disabled --
 * or no flag -- does not, so no projection op is declared. */
TEST(the_control_vector_flag_is_off_on_or_disabled) {
    for (const char* ok : { "off", "on", "disabled" }) {
        Config c;
        CHECK_EQ(parse({ "--control-vector", ok }, &c), RAD_OK);
        CHECK_EQ(c.control_vector, std::string(ok));
    }
    {
        Config c;
        CHECK_EQ(parse({}, &c), RAD_OK);
        CHECK_EQ(c.control_vector, std::string("disabled"));
    }
    for (const char* bad : { "auto", "OFF", "", "1", "true" }) {
        Config c;
        int st = RAD_OK;
        const std::string log = stderr_of([&] { st = parse({ "--control-vector", bad }, &c); });
        CHECK_EQ(st, (int)RAD_E_INVAL);
        CHECK(has(log, "--control-vector"));
    }
}

TEST(the_wire_mode_is_one_of_two_names) {
    Config c;
    CHECK_EQ(parse({ "--tp-wire", "exact" }, &c), RAD_OK);
    CHECK_EQ(c.tp_wire_exact, 1);
    CHECK_EQ(parse({ "--tp-wire", "wht6" }, &c), RAD_OK);
    CHECK_EQ(c.tp_wire_exact, 0);
    /* A lossy wire is undetectable downstream (a rank just receives a different sum), so an
     * unknown name stops the server instead of falling through to the default mode. */
    CHECK_EQ(parse({ "--tp-wire", "wht8" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--tp-wire", "" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--tp-wire", "EXACT" }, &c), RAD_E_INVAL);
}

TEST(the_p2p_mode_is_one_of_three_names) {
    Config c;
    CHECK_EQ(c.p2p, RAD_P2P_AUTO);
    CHECK_EQ(parse({ "--p2p", "on" }, &c), RAD_OK);
    CHECK_EQ(c.p2p, RAD_P2P_ON);
    CHECK_EQ(parse({ "--p2p", "off" }, &c), RAD_OK);
    CHECK_EQ(c.p2p, RAD_P2P_OFF);
    CHECK_EQ(parse({ "--p2p", "auto" }, &c), RAD_OK);
    CHECK_EQ(c.p2p, RAD_P2P_AUTO);
    /* An unknown name stops the server: `on` guarantees a deployment measured on P2P never
     * silently serves through host memory, which a typo falling through to auto could do. */
    CHECK_EQ(parse({ "--p2p", "yes" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--p2p", "" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--p2p", "OFF" }, &c), RAD_E_INVAL);
}

TEST(the_kv_cache_dtype_is_one_of_four_names) {
    Config c;
    CHECK_EQ(parse({ "--kv-cache-dtype", "bf16" }, &c), RAD_OK);
    CHECK_EQ(c.kv_cache_dtype, std::string("bf16"));
    CHECK_EQ(parse({ "--kv-cache-dtype", "fp8" }, &c), RAD_OK);
    CHECK_EQ(c.kv_cache_dtype, std::string("fp8"));
    CHECK_EQ(parse({ "--kv-cache-dtype", "fp8_v4" }, &c), RAD_OK);
    CHECK_EQ(c.kv_cache_dtype, std::string("fp8_v4"));
    CHECK_EQ(parse({ "--kv-cache-dtype", "k6_v4" }, &c), RAD_OK);
    CHECK_EQ(c.kv_cache_dtype, std::string("k6_v4"));
    /* Four values only: an "auto" resolving to bf16 would be a second name for the default, and
     * each value should be unambiguous in a log. */
    CHECK_EQ(parse({ "--kv-cache-dtype", "auto" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-cache-dtype", "fp16" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-cache-dtype", "v4" }, &c), RAD_E_INVAL);

    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.kv_cache_dtype, std::string("bf16"));
}

/* The embedding takes ram (default) or vram; the n-gram table takes auto (default), vram, ram or
 * disk. The parser records whether the embedding placement was named and leaves resolution to
 * Engine::plan, which can refuse a placement with the model in hand. */
TEST(the_embedding_is_ram_or_vram_and_the_ngram_table_takes_four) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(d.embedding_placement == TablePlacement::Ram);
    CHECK(!d.embedding_placement_given);
    CHECK(d.ngram_placement == TablePlacement::Auto);
    CHECK_EQ(d.embedding_host_mib, 0ll);                     /* set by the resolver */
    Config c;
    CHECK_EQ(parse({ "--embedding-placement", "ram", "--ngram-placement", "disk" }, &c), RAD_OK);
    CHECK(c.embedding_placement == TablePlacement::Ram);
    CHECK(c.embedding_placement_given);
    CHECK(c.ngram_placement == TablePlacement::Disk);
    Config v;
    CHECK_EQ(parse({ "--embedding-placement", "vram", "--ngram-placement", "auto" }, &v), RAD_OK);
    CHECK(v.embedding_placement == TablePlacement::Vram);
    CHECK(v.embedding_placement_given);
    CHECK(v.ngram_placement == TablePlacement::Auto);
    CHECK(v.embedding_resolved == TablePlacement::Vram);     /* untouched until plan */
    CHECK_EQ(parse({ "--ngram-placement", "ram" }, &v), RAD_OK);
    CHECK(v.ngram_placement == TablePlacement::Ram);
    for (const char* bad : { "auto", "disk" })
        CHECK_EQ(parse({ "--embedding-placement", bad }, &c), RAD_E_INVAL);
    for (const char* bad : { "RAM", "memory", "host", "" }) {
        CHECK_EQ(parse({ "--embedding-placement", bad }, &c), RAD_E_INVAL);
        CHECK_EQ(parse({ "--ngram-placement", bad }, &c), RAD_E_INVAL);
    }
    CHECK_EQ(parse({ "--ngram-placement" }, &c), RAD_E_INVAL);
}

/* --draft-window takes exactly two names; the default is kDraftWindowDefault, which the help text
 * also states. Whether the window can adapt (a serial drafter, depth above zero) is decided by the
 * scheduler with the program in hand, not the parser. */
TEST(the_draft_window_is_fixed_or_adaptive) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(d.draft_window == kDraftWindowDefault);
    Config c;
    CHECK_EQ(parse({ "--draft-window", "adaptive" }, &c), RAD_OK);
    CHECK(c.draft_window == DraftWindow::Adaptive);
    CHECK_EQ(parse({ "--draft-window", "fixed" }, &c), RAD_OK);
    CHECK(c.draft_window == DraftWindow::Fixed);
    CHECK_EQ(parse({ "--draft-window", "adaptive", "--num-speculative-tokens", "6" }, &c), RAD_OK);
    CHECK(c.draft_window == DraftWindow::Adaptive);
    CHECK_EQ(c.n_spec, 6);
    for (const char* bad : { "Adaptive", "auto", "dynamic", "1", "" })
        CHECK_EQ(parse({ "--draft-window", bad }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--draft-window" }, &c), RAD_E_INVAL);
    CHECK_EQ(std::string(draft_window_name(DraftWindow::Adaptive)), std::string("adaptive"));
    CHECK_EQ(std::string(draft_window_name(DraftWindow::Fixed)), std::string("fixed"));
}

/* --draft-policy takes exactly two names. Whether the deployment can run the cost policy (a serial
 * drafter, a positive ceiling) is decided by the scheduler with the program in hand. */
TEST(the_draft_policy_is_legacy_or_cost) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(d.draft_policy == kDraftPolicyDefault);
    Config c;
    CHECK_EQ(parse({ "--draft-policy", "cost" }, &c), RAD_OK);
    CHECK(c.draft_policy == DraftPolicyKind::Cost);
    CHECK_EQ(parse({ "--draft-policy", "legacy" }, &c), RAD_OK);
    CHECK(c.draft_policy == DraftPolicyKind::Legacy);
    for (const char* bad : { "Cost", "auto", "adaptive", "1", "" })
        CHECK_EQ(parse({ "--draft-policy", bad }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--draft-policy" }, &c), RAD_E_INVAL);
    CHECK_EQ(std::string(draft_policy_name(DraftPolicyKind::Cost)), std::string("cost"));
    CHECK_EQ(std::string(draft_policy_name(DraftPolicyKind::Legacy)), std::string("legacy"));
}

/* --draft-depth-tiers is the cost policy's ladder: ascending, unique, at most kDraftMaxTiers.
 * "<= ceiling" needs the ceiling, which auto resolves after the container is open, so that check
 * stays with the planner. */
TEST(the_draft_depth_tiers_are_an_ascending_unique_list) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.draft_n_tiers, 5);
    CHECK_EQ(d.draft_tiers[0], 0);
    CHECK_EQ(d.draft_tiers[4], 6);
    Config c;
    CHECK_EQ(parse({ "--draft-depth-tiers", "0,2" }, &c), RAD_OK);
    CHECK_EQ(c.draft_n_tiers, 2);
    CHECK_EQ(c.draft_tiers[1], 2);
    CHECK_EQ(parse({ "--draft-depth-tiers", "1,3,3" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--draft-depth-tiers", "4,2" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--draft-depth-tiers", "0,,2" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--draft-depth-tiers", "-1,2" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--draft-depth-tiers", "high" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--draft-depth-tiers" }, &c), RAD_E_INVAL);
}

/* --draft-calib-k is a sequence, not a ladder: order and duplicates are what it is for, so the
 * parser checks only the count and the values. Whether each depth is one the effective ladder
 * offers is DraftPlanner::configure's check at startup, where the trimmed ladder is known. */
TEST(the_calibration_sequence_keeps_its_order_and_its_duplicates) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.draft_calib_n, 0);
    Config c;
    CHECK_EQ(parse({ "--draft-calib-k", "3,2,2,3" }, &c), RAD_OK);
    CHECK_EQ(c.draft_calib_n, 4);
    CHECK_EQ(c.draft_calib_k[0], 3);
    CHECK_EQ(c.draft_calib_k[1], 2);
    CHECK_EQ(c.draft_calib_k[2], 2);
    CHECK_EQ(c.draft_calib_k[3], 3);
    CHECK_EQ(parse({ "--draft-calib-k", "0,6,0" }, &c), RAD_OK);   /* unsorted, repeated */
    CHECK_EQ(c.draft_calib_n, 3);
    CHECK_EQ(parse({ "--draft-calib-k", "3" }, &c), RAD_E_INVAL);  /* a depth is not a sequence */
    CHECK_EQ(parse({ "--draft-calib-k", "3,2,2,3,2,3,3,2,3" }, &c), RAD_E_INVAL);   /* > 8 */
    CHECK_EQ(parse({ "--draft-calib-k", "3,,2" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--draft-calib-k", "-1,2" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--draft-calib-k" }, &c), RAD_E_INVAL);
}

/* --draft-profile is a path, read by the engine once the deployment is up. Nothing parses the file
 * here: the reader owns that, and a missing or malformed file is a start error there. */
TEST(the_draft_profile_is_a_path_and_optional) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(d.draft_profile.empty());
    Config c;
    CHECK_EQ(parse({ "--draft-profile", "/profiles/r9700-tp2.json" }, &c), RAD_OK);
    CHECK_EQ(c.draft_profile, std::string("/profiles/r9700-tp2.json"));
    CHECK_EQ(parse({ "--draft-profile" }, &c), RAD_E_INVAL);
}

/* An unknown policy name is refused; falling through to the default would make a comparison
 * silently measure the default against itself. */
TEST(the_checkpoint_policy_is_one_of_the_three_that_exist) {
    Config c;
    for (const char* p : { "geometric-backoff", "tip-only", "keep-all" }) {
        CHECK_EQ(parse({ "--checkpoint-policy", p }, &c), RAD_OK);
        CHECK_EQ(c.checkpoint_policy, std::string(p));
    }
    CHECK_EQ(parse({ "--checkpoint-policy", "lru" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--checkpoint-policy", "geometric_backoff" }, &c), RAD_E_INVAL);

    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.checkpoint_policy, std::string("geometric-backoff"));
}

/* --checkpoint-slots is auto by default, resolved by the engine once a snapshot's size is known;
 * a count is taken as given, 0 included, and the report shows which. */
TEST(the_checkpoint_slots_are_auto_or_a_count) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.checkpoint_slots, -1ll);
    auto row = [](const Config& c) {
        for (const diag::ConfigRow& r : diag::config_rows(c))
            if (r.flag == "--checkpoint-slots") return r;
        return diag::ConfigRow{};
    };
    CHECK_EQ(row(d).value, std::string("auto"));
    CHECK_EQ(row(d).source, std::string("default"));

    Config a;
    CHECK_EQ(parse({ "--checkpoint-slots", "auto" }, &a), RAD_OK);
    CHECK_EQ(a.checkpoint_slots, -1ll);
    CHECK_EQ(row(a).value, std::string("auto"));
    CHECK_EQ(row(a).source, std::string("command line"));
    for (const auto& [arg, want] : std::vector<std::pair<const char*, long long>>{
             { "0", 0 }, { "8", 8 }, { "96", 96 }, { "9223372036854775807", INT64_MAX } }) {
        Config c;
        CHECK_EQ(parse({ "--checkpoint-slots", arg }, &c), RAD_OK);
        CHECK_EQ(c.checkpoint_slots, want);
        CHECK_EQ(row(c).value, std::string(arg));
    }
    for (const char* bad : { "-1", "Auto", "auto8", "8 ", "", "0x10", "1e3",
                             "9223372036854775808" }) {
        Config c;
        int st = RAD_OK;
        const std::string log = stderr_of([&] { st = parse({ "--checkpoint-slots", bad }, &c); });
        if (st != RAD_E_INVAL) std::fprintf(stderr, "    '%s' was not refused\n", bad);
        CHECK_EQ(st, (int)RAD_E_INVAL);
        CHECK(has(log, "--checkpoint-slots"));
    }
}

/* The request parser's ranges, enforced at startup: a default outside them would 400 every
 * request omitting the field, from a server that reports healthy. */
TEST(the_sampler_defaults_are_range_checked_at_startup) {
    Config c;
    CHECK_EQ(parse({ "--temp", "0.7", "--top-k", "20", "--top-p", "0.95",
                     "--min-p", "0.05" }, &c), RAD_OK);
    CHECK_NEAR(c.sample_temp, 0.7f, 1e-6);
    CHECK_EQ(c.sample_top_k, 20);
    CHECK_NEAR(c.sample_top_p, 0.95f, 1e-6);
    CHECK_NEAR(c.sample_min_p, 0.05f, 1e-6);

    /* The legal endpoints. Temperature 0 is greedy and top_k 0 is off. */
    CHECK_EQ(parse({ "--temp", "0" }, &c), RAD_OK);
    CHECK_EQ(parse({ "--temp", "2" }, &c), RAD_OK);
    CHECK_EQ(parse({ "--top-k", "0" }, &c), RAD_OK);
    CHECK_EQ(parse({ "--top-p", "1" }, &c), RAD_OK);
    CHECK_EQ(parse({ "--min-p", "0" }, &c), RAD_OK);
    CHECK_EQ(parse({ "--min-p", "1" }, &c), RAD_OK);

    CHECK_EQ(parse({ "--temp", "2.1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--top-p", "0" }, &c), RAD_E_INVAL);   /* empty nucleus */
    CHECK_EQ(parse({ "--top-p", "1.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--min-p", "1.5" }, &c), RAD_E_INVAL);

    /* A negative value is refused: below the parser a negative field means unset, and the
     * container's default would be served instead. */
    CHECK_EQ(parse({ "--temp", "-0.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--top-k", "-5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--top-p", "-0.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--min-p", "-0.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--temp", "nan" }, &c), RAD_E_INVAL);

    /* Unset is negative, not zero, so the container's generation config can be resolved
     * underneath: zero is a real temperature and top_k, indistinguishable from asking for greedy. */
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(d.sample_temp < 0.0f);
    CHECK(d.sample_top_k < 0);
    CHECK(d.sample_top_p < 0.0f);
    CHECK(d.sample_min_p < 0.0f);
}

/* A number must be the whole argument; atof would read "high" as 0, which for --temp is greedy. */
TEST(a_sampler_default_that_is_not_a_number_is_refused) {
    Config c;
    CHECK_EQ(parse({ "--temp", "high" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--temp", "0.7x" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--temp", "" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--top-k", "20.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--top-p", "inf" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--presence-penalty", "nan" }, &c), RAD_E_INVAL);
}

/* ================================================================== request defaults and bounds
 *
 * None of these changes anything until given: each unset value is the server's built-in behaviour
 * or a marker the engine reads as "leave it". */
TEST(the_request_defaults_and_bounds_are_unset_until_given) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(!d.sample_typical_p && !d.sample_rep_penalty && !d.sample_pres_penalty &&
          !d.sample_freq_penalty && !d.sample_dry_multiplier && !d.sample_dry_base &&
          !d.sample_xtc_probability && !d.sample_xtc_threshold);
    CHECK(!d.sample_penalty_last_n && !d.sample_dry_allowed_length &&
          !d.sample_dry_penalty_last_n && !d.sample_dry_seq_breakers);
    CHECK_EQ(d.default_max_tokens, -1ll);
    CHECK_EQ(d.max_tokens_cap, 0ll);
    CHECK_EQ(d.max_n, 0ll);
    CHECK_EQ(d.max_queued_requests, 0ll);
    CHECK_EQ(d.max_stop_strings, 64ll);
    CHECK_EQ(d.max_stop_bytes, 4096ll);
    CHECK(d.chat_template_kwargs.empty());
    CHECK_EQ(d.reasoning_format, std::string("auto"));
    /* The transport's own values (server/http.h TransportOptions). */
    CHECK_EQ(d.http_threads, 0);
    CHECK_EQ(d.http_read_timeout_s, 30);
    CHECK_EQ(d.http_write_timeout_s, 600);
    CHECK_EQ(d.http_keep_alive_timeout_s, 5);
    CHECK_EQ(d.http_max_body_mib, 512ll);
    CHECK_EQ(d.retry_after_s, 1);
    CHECK(d.cors);
    CHECK(!d.media_flags_set());
}

TEST(the_request_defaults_reach_their_fields) {
    Config c;
    CHECK_EQ(parse({ "--typical-p", "0.9", "--presence-penalty", "1.5",
                     "--frequency-penalty", "-0.5", "--repetition-penalty", "1.05",
                     "--repeat-last-n", "-1", "--dry-multiplier", "0.8", "--dry-base", "1.75",
                     "--dry-allowed-length", "3", "--dry-penalty-last-n", "256",
                     "--dry-sequence-breakers", R"(["\n", ":"])",
                     "--xtc-probability", "0.5", "--xtc-threshold", "0.1",
                     "--default-max-tokens", "4096", "--max-tokens-cap", "8192",
                     "--max-n", "4", "--max-queued-requests", "64",
                     "--max-stop-strings", "8", "--max-stop-bytes", "256",
                     "--reasoning-format", "none" }, &c), RAD_OK);
    CHECK_NEAR(*c.sample_typical_p, 0.9f, 1e-6);
    CHECK_NEAR(*c.sample_pres_penalty, 1.5f, 1e-6);
    CHECK_NEAR(*c.sample_freq_penalty, -0.5f, 1e-6);
    CHECK_NEAR(*c.sample_rep_penalty, 1.05f, 1e-6);
    CHECK_EQ(*c.sample_penalty_last_n, -1);
    CHECK_NEAR(*c.sample_dry_multiplier, 0.8f, 1e-6);
    CHECK_NEAR(*c.sample_dry_base, 1.75f, 1e-6);
    CHECK_EQ(*c.sample_dry_allowed_length, 3);
    CHECK_EQ(*c.sample_dry_penalty_last_n, 256);
    CHECK(*c.sample_dry_seq_breakers == std::vector<std::string>({ "\n", ":" }));
    CHECK_NEAR(*c.sample_xtc_probability, 0.5f, 1e-6);
    CHECK_NEAR(*c.sample_xtc_threshold, 0.1f, 1e-6);
    CHECK_EQ(c.default_max_tokens, 4096ll);
    CHECK_EQ(c.max_tokens_cap, 8192ll);
    CHECK_EQ(c.max_n, 4ll);
    CHECK_EQ(c.max_queued_requests, 64ll);
    CHECK_EQ(c.max_stop_strings, 8ll);
    CHECK_EQ(c.max_stop_bytes, 256ll);
    CHECK_EQ(c.reasoning_format, std::string("none"));

    /* `auto` is 0, resolved per request to what the context leaves. */
    Config a;
    CHECK_EQ(parse({ "--default-max-tokens", "auto" }, &a), RAD_OK);
    CHECK_EQ(a.default_max_tokens, 0ll);

    /* An empty breaker list is a stated value (DRY with no breakers), not unset. */
    Config e;
    CHECK_EQ(parse({ "--dry-sequence-breakers", "[]" }, &e), RAD_OK);
    CHECK(e.sample_dry_seq_breakers && e.sample_dry_seq_breakers->empty());

    /* A request may send a negative penalty, so the flag accepts one. */
    Config n;
    CHECK_EQ(parse({ "--presence-penalty", "-2" }, &n), RAD_OK);
    CHECK_NEAR(*n.sample_pres_penalty, -2.0f, 1e-6);
}

/* Request ranges, enforced at startup: a default outside them would 400 every request that
 * omits the field. */
TEST(the_request_defaults_are_range_checked_like_a_request) {
    Config c;
    CHECK_EQ(parse({ "--typical-p", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--typical-p", "1.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--presence-penalty", "2.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--frequency-penalty", "-3" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--repetition-penalty", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--repeat-last-n", "-2" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--dry-base", "0.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--dry-penalty-last-n", "-2" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--xtc-probability", "1.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--xtc-threshold", "-0.1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--dry-allowed-length", "3.5" }, &c), RAD_E_INVAL);
    /* With DRY on, a request whose allowed length is below 1 is refused at admission, so a
     * default below 1 is refused here rather than refusing every such request. */
    CHECK_EQ(parse({ "--dry-allowed-length", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--dry-allowed-length", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--dry-allowed-length", "1" }, &c), RAD_OK);
    CHECK_EQ(parse({ "--dry-sequence-breakers", R"("\n")" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--dry-sequence-breakers", "[1]" }, &c), RAD_E_INVAL);
    /* The sampler takes at most 16 (sample/host_ref.h); more would refuse every DRY request. */
    CHECK_EQ(parse({ "--dry-sequence-breakers",
                     R"(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q"])" },
                   &c), RAD_E_INVAL);

    CHECK_EQ(parse({ "--default-max-tokens", "0" }, &c), RAD_E_INVAL);   /* spell it `auto` */
    CHECK_EQ(parse({ "--default-max-tokens", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--default-max-tokens", "AUTO" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--default-max-tokens", "4294967296" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-tokens-cap", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-n", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-queued-requests", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-stop-strings", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-stop-bytes", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--reasoning-format", "deepseek" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--reasoning-format", "" }, &c), RAD_E_INVAL);
}

/* Contradictory bounds are refused: a default above the cap is never served, and an `n` above the
 * queue size gets 429 however long its caller waits. */
TEST(request_bounds_that_contradict_each_other_are_refused) {
    CHECK_EQ(parse_fresh({ "--default-max-tokens", "4096", "--max-tokens-cap", "1024" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--default-max-tokens", "1024", "--max-tokens-cap", "1024" }), RAD_OK);
    CHECK_EQ(parse_fresh({ "--default-max-tokens", "auto", "--max-tokens-cap", "1024" }), RAD_OK);

    CHECK_EQ(parse_fresh({ "--max-n", "65", "--max-queued-requests", "64" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--max-n", "64", "--max-queued-requests", "64" }), RAD_OK);
    /* The queue's default is 8 x --max-num-seqs, and the check uses it. */
    CHECK_EQ(parse_fresh({ "--max-num-seqs", "2", "--max-n", "17" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--max-num-seqs", "2", "--max-n", "16" }), RAD_OK);
    /* A stated queue smaller than the default n (8, or --max-num-seqs). */
    CHECK_EQ(parse_fresh({ "--max-queued-requests", "4" }), RAD_E_INVAL);
}

TEST(the_template_variables_merge_key_by_key_and_later_wins) {
    Config c;
    CHECK_EQ(parse({ "--chat-template-kwargs", R"({"enable_thinking": false, "custom": "a"})",
                     "--chat-template-kwargs", R"({"custom": "b", "n": 3})" }, &c), RAD_OK);
    CHECK_EQ(c.chat_template_kwargs.size(), 3u);
    CHECK_EQ(c.chat_template_kwargs["enable_thinking"], std::string("false"));
    CHECK_EQ(c.chat_template_kwargs["custom"], std::string("\"b\""));   /* JSON text, quoted */
    CHECK_EQ(c.chat_template_kwargs["n"], std::string("3"));

    /* @FILE reads the object from a file. */
    const std::string path = radtest::tmp_path("kwargs.json");
    FILE* fp = std::fopen(path.c_str(), "w");
    REQUIRE(fp != nullptr);
    const std::string body = R"({"date_string": "01 Jan 2026"})";
    CHECK_EQ(std::fwrite(body.data(), 1, body.size(), fp), body.size());
    std::fclose(fp);
    Config f;
    const std::string at = "@" + path;
    CHECK_EQ(parse({ "--chat-template-kwargs", at.c_str() }, &f), RAD_OK);
    CHECK_EQ(f.chat_template_kwargs["date_string"], std::string("\"01 Jan 2026\""));
    unlink(path.c_str());
    CHECK_EQ(parse({ "--chat-template-kwargs", at.c_str() }, &f), RAD_E_INVAL);   /* gone */
}

TEST(a_template_variable_the_server_owns_or_cannot_read_is_refused) {
    CHECK_EQ(parse_fresh({ "--chat-template-kwargs", "[1]" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--chat-template-kwargs", "{nope" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--chat-template-kwargs", R"("enable_thinking")" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--chat-template-kwargs", R"({"enable_thinking": "no"})" }), RAD_E_INVAL);
    /* The renderer sets these; one here would replace every conversation. */
    for (const char* k : { "messages", "tools", "bos_token", "eos_token", "add_generation_prompt" }) {
        const std::string j = std::string("{\"") + k + "\": 1}";
        CHECK_EQ(parse_fresh({ "--chat-template-kwargs", j.c_str() }), RAD_E_INVAL);
    }
    /* Two flags setting the thinking default, with no precedence between them. */
    CHECK_EQ(parse_fresh({ "--reasoning-effort", "low",
                           "--chat-template-kwargs", R"({"reasoning_effort": "high"})" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--chat-template-kwargs", R"({"enable_thinking": true})",
                           "--reasoning-effort", "none" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--reasoning-effort", "low",
                           "--chat-template-kwargs", R"({"custom": 1})" }), RAD_OK);
}

TEST(the_transport_flags_reach_their_fields_and_hold_their_floors) {
    Config c;
    CHECK_EQ(parse({ "--http-threads", "16", "--read-timeout", "10", "--write-timeout", "120",
                     "--keep-alive-timeout", "30", "--max-body-mib", "64", "--retry-after", "5",
                     "--no-cors" }, &c), RAD_OK);
    CHECK_EQ(c.http_threads, 16);
    CHECK_EQ(c.http_read_timeout_s, 10);
    CHECK_EQ(c.http_write_timeout_s, 120);
    CHECK_EQ(c.http_keep_alive_timeout_s, 30);
    CHECK_EQ(c.http_max_body_mib, 64ll);
    CHECK_EQ(c.retry_after_s, 5);
    CHECK(!c.cors);

    /* The transport keeps at least four workers, so fewer is refused, not ignored. */
    CHECK_EQ(parse({ "--http-threads", "2" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--read-timeout", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--write-timeout", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--keep-alive-timeout", "x" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-body-mib", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--retry-after", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--retry-after", "0" }, &c), RAD_OK);
}

TEST(the_media_flags_reach_their_fields_and_refuse_an_empty_band) {
    Config c;
    CHECK_EQ(parse({ "--image-min-pixels", "3136", "--image-max-pixels", "1003520",
                     "--video-min-pixels", "4096", "--video-max-pixels", "8388608",
                     "--video-fps", "1.5", "--video-min-frames", "2", "--video-max-frames", "64",
                     "--video-max-frame-tokens", "0", "--max-source-pixels", "16777216" }, &c),
             RAD_OK);
    CHECK(c.media_flags_set());
    CHECK_EQ(*c.image_min_pixels, 3136ll);
    CHECK_EQ(*c.image_max_pixels, 1003520ll);
    CHECK_EQ(*c.video_min_pixels, 4096ll);
    CHECK_EQ(*c.video_max_pixels, 8388608ll);
    CHECK_NEAR(*c.video_fps, 1.5, 1e-9);
    CHECK_EQ(*c.video_min_frames, 2);
    CHECK_EQ(*c.video_max_frames, 64);
    CHECK_EQ(*c.video_max_frame_tokens, 0);      /* 0 is "no per-frame cap" */
    CHECK_EQ(*c.max_source_pixels, 16777216ll);

    CHECK_EQ(parse({ "--image-min-pixels", "2000", "--image-max-pixels", "1000" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--video-min-pixels", "2000", "--video-max-pixels", "1000" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--video-min-frames", "8", "--video-max-frames", "4" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--image-max-pixels", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--video-fps", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--video-fps", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--video-min-frames", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--video-max-frame-tokens", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-source-pixels", "0" }, &c), RAD_E_INVAL);
}

/* --draft-model (familiar from vLLM) is refused by name; accepting it and serving the container's
 * own drafter, or none, would silently differ from what was asked. */
TEST(draft_model_is_refused_by_name_rather_than_ignored) {
    Config c;
    CHECK_EQ(parse({ "--draft-model", "/d.rad" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--draft-model" }, &c), RAD_E_INVAL);
}

/* ================================================================== resolved */

TEST(auto_speculation_is_negative_one_and_not_zero) {
    /* The depth is the drafter's, stated by the architecture plugin once the container is open,
     * so "auto" survives as a distinct value here. Zero is off. */
    Config c;
    CHECK_EQ(parse({ "--num-speculative-tokens", "auto" }, &c), RAD_OK);
    CHECK_EQ(c.n_spec, -1);
    CHECK_EQ(parse({ "--num-speculative-tokens", "0" }, &c), RAD_OK);
    CHECK_EQ(c.n_spec, 0);
    CHECK_EQ(parse({ "--num-speculative-tokens", "3" }, &c), RAD_OK);
    CHECK_EQ(c.n_spec, 3);
    /* Only the word means auto; a negative depth would be read as auto downstream. */
    CHECK_EQ(parse({ "--num-speculative-tokens", "-3" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--num-speculative-tokens", "-1" }, &c), RAD_E_INVAL);

    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.n_spec, -1);                /* auto is also the default */
}

/* --gpu-headroom is an alias for --gpu-headroom-mib; both set the same field. */
TEST(gpu_headroom_parses_under_both_spellings) {
    Config a, b, d;
    CHECK_EQ(parse({ "--gpu-headroom-mib", "256" }, &a), RAD_OK);
    CHECK_EQ(parse({ "--gpu-headroom", "256" }, &b), RAD_OK);
    CHECK_EQ(a.gpu_headroom_mib, 256ll);
    CHECK_EQ(b.gpu_headroom_mib, 256ll);
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.gpu_headroom_mib, 96ll);
}

TEST(the_kernel_hierarchy_is_separated_by_commas_or_colons_and_ordered) {
    Config c;
    CHECK_EQ(parse({ "--kernels", "libr4d.so,libref.so" }, &c), RAD_OK);
    CHECK_EQ(c.kernel_hierarchy.size(), 2u);
    CHECK_EQ(c.kernel_hierarchy[0], std::string("libr4d.so"));   /* first wins, so order matters */
    CHECK_EQ(c.kernel_hierarchy[1], std::string("libref.so"));

    /* Empty segments are dropped: a trailing comma is a typo, and a plugin path of "" matches
     * nothing and would warn on every start. */
    Config t;
    CHECK_EQ(parse({ "--kernels", ",a,,b," }, &t), RAD_OK);
    CHECK_EQ(t.kernel_hierarchy.size(), 2u);
    CHECK_EQ(t.kernel_hierarchy[0], std::string("a"));
    CHECK_EQ(t.kernel_hierarchy[1], std::string("b"));

    Config one;
    CHECK_EQ(parse({ "--kernels", "solo" }, &one), RAD_OK);
    CHECK_EQ(one.kernel_hierarchy.size(), 1u);

    /* A colon separates as a comma does: rad-kbench, rad-tune and rad-convert document
     * `--kernels A:B`, and the engine must load the same list from the same spelling. */
    Config col;
    CHECK_EQ(parse({ "--kernels", "mykernels:libr4d,libref" }, &col), RAD_OK);
    REQUIRE_EQ(col.kernel_hierarchy.size(), 3u);
    CHECK_EQ(col.kernel_hierarchy[0], std::string("mykernels"));
    CHECK_EQ(col.kernel_hierarchy[1], std::string("libr4d"));
    CHECK_EQ(col.kernel_hierarchy[2], std::string("libref"));
}

/* There is deliberately no built-in default hierarchy: naming one card's kernel library in the
 * core would rank the installed library behind the reference elsewhere. The loader orders an empty
 * hierarchy. */
TEST(an_unnamed_hierarchy_stays_empty_for_the_loader_to_order) {
    Config c;
    CHECK_EQ(parse({}, &c), RAD_OK);
    CHECK_EQ(c.kernel_hierarchy.size(), 0u);
}

/* The engine honours the same environment variables as the tools, so a plugin validated with
 * rad-kbench or rad-tune serves without editing every launcher's command line. */
TEST(the_environment_supplies_what_the_line_left_unsaid) {
    const char* old_home = getenv("RADIANCE_HOME");
    const char* old_kern = getenv("RADIANCE_KERNELS");
    const std::string save_home = old_home ? old_home : "";
    const std::string save_kern = old_kern ? old_kern : "";

    setenv("RADIANCE_HOME", "/env/home", 1);
    setenv("RADIANCE_KERNELS", "env_a,env_b", 1);

    Config c;
    CHECK_EQ(parse({}, &c), RAD_OK);
    CHECK_EQ(c.radiance_home, std::string("/env/home"));
    CHECK_EQ(c.kernel_hierarchy.size(), 2u);
    CHECK_EQ(c.kernel_hierarchy[0], std::string("env_a"));

    /* The command line wins over the environment. */
    Config f;
    CHECK_EQ(parse({ "--radiance-home", "/flag/home", "--kernels", "flag_only" }, &f), RAD_OK);
    CHECK_EQ(f.radiance_home, std::string("/flag/home"));
    CHECK_EQ(f.kernel_hierarchy.size(), 1u);
    CHECK_EQ(f.kernel_hierarchy[0], std::string("flag_only"));

    /* An empty variable is treated as unset, not as an empty hierarchy. */
    setenv("RADIANCE_KERNELS", "", 1);
    Config e;
    CHECK_EQ(parse({}, &e), RAD_OK);
    CHECK_EQ(e.kernel_hierarchy.size(), 0u);

    /* $RADIANCE_KERNELS splits by the same rule as --kernels. */
    setenv("RADIANCE_KERNELS", "env_a:env_b", 1);
    Config k;
    CHECK_EQ(parse({}, &k), RAD_OK);
    CHECK(k.kernel_hierarchy == (std::vector<std::string>{ "env_a", "env_b" }));

    unsetenv("RADIANCE_HOME");
    Config h;
    CHECK_EQ(parse({}, &h), RAD_OK);
    CHECK(!h.radiance_home.empty());               /* the compiled-in default */

    /* A home set but empty is unset too: as the tools read it, and an empty home would load no
     * plugin. The source names the default, not the environment. */
    setenv("RADIANCE_HOME", "", 1);
    Config eh;
    CHECK_EQ(parse({}, &eh), RAD_OK);
    CHECK_EQ(eh.radiance_home, h.radiance_home);
    CHECK(!eh.given.count("--radiance-home"));

    if (old_home) setenv("RADIANCE_HOME", save_home.c_str(), 1); else unsetenv("RADIANCE_HOME");
    if (old_kern) setenv("RADIANCE_KERNELS", save_kern.c_str(), 1); else unsetenv("RADIANCE_KERNELS");
}

TEST(the_download_dir_is_taken_and_left_empty_until_given) {
    /* Empty means the XDG cache, which hf_model.cpp resolves only when --model is a repo id, so a
     * server started on a local file never reads HOME for it. */
    Config c;
    CHECK_EQ(parse({}, &c), 0);
    CHECK(c.download_dir.empty());
    Config d;
    CHECK_EQ(parse({ "--download-dir", "/big/models" }, &d), 0);
    CHECK_EQ(d.download_dir, std::string("/big/models"));
}

/* The last occurrence wins for every flag with a value: scripts append overrides. */
TEST(a_repeated_flag_takes_the_last_one) {
    Config c;
    CHECK_EQ(parse({ "--port", "1", "--port", "8100" }, &c), RAD_OK);
    CHECK_EQ(c.port, 8100);
    CHECK_EQ(parse({ "--kv-cache-dtype", "bf16", "--kv-cache-dtype", "fp8" }, &c), RAD_OK);
    CHECK_EQ(c.kv_cache_dtype, std::string("fp8"));
    /* A bad value is refused even after a good one. */
    CHECK_EQ(parse({ "--tp-wire", "exact", "--tp-wire", "nope" }, &c), RAD_E_INVAL);
}

TEST(usage_names_every_flag_it_documents) {
    /* config_usage writes to stderr; it must mention the flags an operator most likely looks
     * for, so a flag added without a help line is caught. */
    const std::string text = stderr_of([] { config_usage("radiance"); });
    for (const char* f : { "--model", "--host-pool-mib", "--kv-cache-dtype",
                           "--num-speculative-tokens", "--draft-window", "--checkpoint-policy",
                           "--prefix-cache-mib", "--prefix-cache-ttl", "--tp-wire",
                           "--embedding-placement", "--ngram-placement", "--kld-record",
                           "--kld-ref", "--default-max-tokens", "--max-tokens-cap",
                           "--presence-penalty", "--dry-sequence-breakers",
                           "--chat-template-kwargs", "--reasoning-format", "--max-n",
                           "--max-queued-requests", "--max-stop-strings", "--http-threads",
                           "--max-body-mib", "--no-cors", "--image-max-pixels", "--video-fps",
                           "--max-source-pixels", "--download-dir" })
        CHECK(has(text, f));
    CHECK(has(text, std::string("Default ") + draft_window_name(kDraftWindowDefault)));
    /* top_k 0 is the sampler's candidate width, not the whole vocabulary or "off". */
    CHECK(has(text, "0 is the 1024 most"));
    CHECK(!has(text, "top_k (0 = off)"));
    CHECK(text.size() > 1000);
}

/* --rope-scaling is validated at the command line: a typo is refused rather than serving the
 * trained rotary at a length it cannot reach. */
TEST(rope_scaling_takes_yarn_with_a_factor_above_one_or_none) {
    for (const char* ok : { "yarn:4", "yarn:2.5", "yarn:4:262144", "none" }) {
        Config c;
        CHECK_EQ(parse({ "--rope-scaling", ok }, &c), RAD_OK);
        CHECK_EQ(c.rope_scaling, std::string(ok));
    }
    for (const char* bad : { "yarn:1", "yarn:0.5", "yarn:4x", "yarn:", "yarn:4:", "yarn:4:0",
                             "yarn:4:12k", "linear:2", "yarn" }) {
        Config c;
        CHECK(parse({ "--rope-scaling", bad }, &c) < 0);
    }
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(d.rope_scaling.empty());
}

RAD_TEST_MAIN()
