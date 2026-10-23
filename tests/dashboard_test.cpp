/* tests/dashboard_test.cpp -- what the dashboard and its JSON routes show.
 *
 * The page at `/` polls /stats and /sessions and reads dozens of their fields by name; a field
 * renamed on either side blanks a panel with no error anywhere. So the routes are checked for the
 * values a populated engine gives them, and the page's own script is read for every field it
 * takes from them, each of which a route must emit. The KV figures behind both come from
 * SchedulerBridge's aggregation over the scheduler's groups, checked here on a hybrid model's
 * groups, where summing or picking the wrong group goes wrong silently.
 */
#include "server_fakes.h"
#include "rad_internal.h"   /* rad_mono_ns: a synthetic bringup start */

#include <set>

namespace {

/* A scheduler with a live request table, stored sessions and idle-tier counters. */
struct StatsScheduler : FakeScheduler {
    int requests(std::vector<RequestStat>* out) const override {
        RequestStat r;
        r.id = 41;
        r.state = "decode";
        r.prompt_tokens = 900;
        r.computed_tokens = 900;
        r.cached_tokens = 512;
        r.output_tokens = 33;
        r.max_tokens = 4096;
        r.ctx_tokens = 933;
        r.kv_blocks = 120;
        r.kv_tokens = 960;
        r.n_draft = 3;
        r.slot = 2;
        r.age_s = 4.5;
        r.ttft_s = 0.25;
        r.decode_tps = 61.0;
        out->assign(1, r);
        return 1;
    }
    int sessions(std::vector<SessionStat>* out, SessionOccupancy* occ) const override {
        SessionStat s;
        s.id = 7;
        s.model = "m";
        s.label = "conv-7";
        s.n_tokens = 5000;
        s.bytes = 1 << 20;
        s.idle_s = 12.0;
        s.age_s = 300.0;
        s.on_device = 40;
        s.on_host = 10;
        s.snapshots = 2;
        s.snap_bytes = 4096;
        s.resume_tokens = 4096;
        s.moving = true;
        out->assign(1, s);
        occ->host_bytes = 8 << 20;
        occ->host_cap_bytes = 64 << 20;
        return 1;
    }
    TierCounters tier_counters() const override {
        TierCounters c;
        c.to_host = 3; c.dropped = 2; c.promoted = 1; c.host_drops = 4; c.failures = 0;
        c.host_bytes = 8 << 20; c.ck_bytes = 2048; c.ck_slots = 16;
        c.restores = 5; c.restore_ns = 2000000; c.flushes = 6; c.flush_ns = 3000000;
        c.host_on = true;
        return c;
    }
};

/* Every value a snapshot carries, set, so every optional field is emitted. */
void populated(LiveSnapshot& s) {
    CardStat c;
    c.index = 0;
    c.name = "R9700";
    c.util = 0.5;
    c.vram_used = 10LL << 30;
    c.vram_total = 32LL << 30;
    c.power_w = 200;
    c.temp_c = 60;
    s.cards = { c };
    s.experts.n_layers = 2;
    s.experts.n_experts = 3;
    s.experts.tier = { 0, 1, 2, 0, 0, 1 };
    for (int t = 0; t < kLiveTiers; ++t) { s.experts.units[t] = t + 1; s.experts.bytes[t] = (t + 1) << 20; }
    s.experts.n_units = 6;
    s.experts.total_bytes = 6 << 20;
    s.experts.movable = 4;
    s.experts.layer_units = { 3, 3 };
    s.experts.layer_resident = { 2, 3 };
    s.experts.promotions = 9; s.experts.demotions = 8; s.experts.readmits = 2;
    s.experts.distinct_promoted = 7; s.experts.dispatches = 20; s.experts.refused = 3;
    s.experts.no_candidate = 1; s.experts.routed = 100; s.experts.routed_resident = 80;
    s.experts.flex_bytes = 1 << 20; s.experts.flex_capacity = 2 << 20;
    s.experts.heat_enabled = true;
    s.link.present = true;
    s.link.world = 2;
    s.link.ar_calls = 50; s.link.ar_bytes = 5000; s.link.h2d_bytes = 900; s.link.d2h_bytes = 800;
    s.link.ssd_reads = 4; s.link.ssd_bytes = 400; s.link.stream_bytes = 700;
    s.link.prefetched = 2; s.link.prefetch_bytes = 300; s.link.staged_bytes = 200;
    s.passes.present = true;
    s.passes.played = 10; s.passes.recorded = 2;
}

struct Dash {
    StatsScheduler sched;
    FakeTokenizer  tok;
    std::unique_ptr<Server> srv;
    explicit Dash(bool with_graph) {
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.live = populated;
        ServerOptions o;
        o.model_id = "dash";
        o.max_seqs = 4;
        if (with_graph) o.graph_json = [] { return std::string(R"({"order":[]})"); };
        sched.m.running = 1;
        sched.m.waiting = 2;
        sched.m.kv_tokens_used = 1000;
        sched.m.kv_tokens_total = 4000;
        sched.m.kv_tokens_cached = 300;
        sched.m.kv_tokens_max = 6000;
        sched.m.prefix_ckpt_lru = 3;
        sched.m.hit_cut_no_snapshot = 2;
        sched.m.hit_cut_tokens = 9000;
        sched.m.draft_depth_seen = 2;
        sched.m.draft_pos_reached[0] = 10; sched.m.draft_pos_accepted[0] = 8;
        sched.m.draft_pos_reached[1] = 8;  sched.m.draft_pos_accepted[1] = 2;
        srv.reset(new Server(d, o));
    }
    json get(const char* path) {
        HttpResponse r = srv->router().call("GET", path);
        CHECK_EQ(r.status, 200);
        return json::parse(r.body, nullptr, false);
    }
};

/* The identifiers `var.<name>` that `text` reads, and the string arguments of `calls("<name>")`. */
std::set<std::string> fields(const std::string& text, const std::string& var) {
    std::set<std::string> out;
    const std::string pat = var + ".";
    for (size_t at = text.find(pat); at != std::string::npos; at = text.find(pat, at + 1)) {
        if (at > 0) {
            const char b = text[at - 1];
            if (std::isalnum((unsigned char)b) || b == '_' || b == '.' || b == '$') continue;
        }
        size_t e = at + pat.size();
        while (e < text.size() && (std::isalnum((unsigned char)text[e]) || text[e] == '_')) ++e;
        if (e > at + pat.size()) out.insert(text.substr(at + pat.size(), e - at - pat.size()));
    }
    return out;
}

std::set<std::string> called(const std::string& text, const std::string& fn) {
    std::set<std::string> out;
    const std::string pat = fn + "(\"";
    for (size_t at = text.find(pat); at != std::string::npos; at = text.find(pat, at + 1)) {
        if (at > 0 && (std::isalnum((unsigned char)text[at - 1]) || text[at - 1] == '_')) continue;
        const size_t b = at + pat.size(), e = text.find('"', b);
        out.insert(text.substr(b, e - b));
    }
    return out;
}

/* The page's script from `from` up to the line that closes the function starting there. */
std::string function_body(const std::string& page, const std::string& from) {
    const size_t at = page.find(from);
    if (at == std::string::npos) return std::string();
    const size_t end = page.find("\n}\n", at);
    return page.substr(at, end == std::string::npos ? std::string::npos : end - at);
}

void each_field_is_emitted(const std::set<std::string>& names, const json& o, const char* what) {
    for (const std::string& n : names) {
        const bool there = o.is_object() && o.contains(n);
        CHECK(there);
        if (!there) fprintf(stderr, "    the page reads %s.%s, which the route does not emit\n", what, n.c_str());
    }
}

}  /* namespace */

/* ================================================================== the routes */

/* /stats carries the scheduler's figures, the live request table, the draft positions with their
 * reach, and the engine's snapshot (cards, experts by named tier, link, passes); /graph is the
 * provider's JSON, or a 501 naming the missing provider. */
TEST(stats_carries_the_engine_the_requests_and_the_snapshot) {
    Dash d(false);
    const json s = d.get("/stats");
    CHECK_EQ(s.value("model", ""), std::string("dash"));
    CHECK_EQ(s.value("queue_cap", (int64_t)0), (int64_t)32);
    CHECK_EQ(s.value("in_flight", (int64_t)-1), (int64_t)0);
    CHECK_EQ(s.value("running", (int64_t)0), (int64_t)1);
    CHECK_EQ(s.value("kv_tokens_cached", (int64_t)0), (int64_t)300);
    CHECK_EQ(s.value("kv_tokens_max", (int64_t)0), (int64_t)6000);
    REQUIRE(s["draft_positions"].size() == 2);
    CHECK_NEAR(s["draft_positions"][1]["rate"].get<double>(), 0.25, 1e-12);
    CHECK_EQ(s["draft_positions"][1]["reached"].get<int64_t>(), (int64_t)8);
    REQUIRE(s["requests"].size() == 1);
    const json& r = s["requests"][0];
    CHECK_EQ(r.value("state", ""), std::string("decode"));
    CHECK_EQ(r.value("kv_tokens", (int64_t)0), (int64_t)960);
    CHECK_NEAR(r.value("ttft_s", -1.0), 0.25, 1e-12);
    REQUIRE(s["cards"].size() == 1);
    CHECK_EQ(s["cards"][0].value("name", ""), std::string("R9700"));
    REQUIRE(s["experts"]["tiers"].size() == (size_t)kTierCount);
    for (int t = 0; t < kTierCount; ++t) {
        CHECK_EQ(s["experts"]["tiers"][t].value("name", ""), std::string(tier_name((Tier)t)));
        CHECK_EQ(s["experts"]["tiers"][t].value("units", (int64_t)0), (int64_t)(t + 1));
    }
    CHECK_EQ(s["link"].value("stream_bytes", (int64_t)0), (int64_t)700);
    CHECK_EQ(s["passes"].value("played", (int64_t)0), (int64_t)10);
    CHECK_EQ(s["ckpt_evictions"].value("store_full", (int64_t)0), (int64_t)3);
    CHECK_EQ(s["hit_cuts"].value("no_snapshot", (int64_t)0), (int64_t)2);
    CHECK_EQ(s.value("hit_cut_tokens", (int64_t)0), (int64_t)9000);

    CHECK_EQ(d.srv->router().call("GET", "/graph").status, 501);
    Dash g(true);
    CHECK_EQ(g.get("/graph").value("order", json()).size(), (size_t)0);
}

/* The cost policy's plan ledger. "The policy never priced" and "the ladder priced nothing" are
 * different facts about a run, and a fallback-only run cannot tell them apart from its plans
 * alone, so /stats carries both counts beside what the last plan asked for. The replay is named,
 * never emitted as a number: it is the *candidate* the plan declared (the last bracket's observed
 * state), so a page must not read it as what ran. No cycle yet means no object at all -- a zeroed
 * ledger would look like a measurement of nothing. */
TEST(stats_reports_the_cost_policys_plan_ledger_by_name) {
    Dash d(false);
    CHECK(!d.get("/stats").contains("draft_cost"));

    const char* names[] = { "unknown", "none", "play", "record", "audit" };
    for (int r = 0; r < (int)(sizeof(names) / sizeof(names[0])); ++r) {
        d.sched.m.draft_cost_cycles      = 4;
        d.sched.m.draft_cost_priced      = 3;
        d.sched.m.draft_cost_fallback    = 1;
        d.sched.m.draft_cost_last_k      = 2;
        d.sched.m.draft_cost_last_replay = r;
        const json dc = d.get("/stats")["draft_cost"];
        CHECK_EQ(dc.value("cycles", (int64_t)0), (int64_t)4);
        CHECK_EQ(dc.value("priced", (int64_t)0), (int64_t)3);
        CHECK_EQ(dc.value("fallback", (int64_t)0), (int64_t)1);
        CHECK_EQ(dc.value("last_k", (int64_t)0), (int64_t)2);
        CHECK_EQ(dc.value("last_replay", std::string()), std::string(names[r]));
    }
}

/* /sessions lists each stored conversation with where its blocks are, the pool as the card holds
 * it (each block once), the tiers' counters and the snapshot store. */
TEST(sessions_lists_each_conversation_and_what_the_tiers_did) {
    Dash d(false);
    const json s = d.get("/sessions");
    REQUIRE(s["sessions"].size() == 1);
    const json& e = s["sessions"][0];
    CHECK_EQ(e.value("label", ""), std::string("conv-7"));
    CHECK_EQ(e.value("on_device", (int64_t)0), (int64_t)40);
    CHECK_EQ(e.value("resume_tokens", (int64_t)0), (int64_t)4096);
    CHECK_EQ(s["blocks_sum_overlapping"].value("host", (int64_t)0), (int64_t)10);
    CHECK_EQ(s["pool"].value("tokens_used", (int64_t)0), (int64_t)1000);
    CHECK_EQ(s["pool"].value("unshared_tokens", (int64_t)0), (int64_t)300);
    CHECK_EQ(s["moved"].value("to_host", (int64_t)0), (int64_t)3);
    CHECK_EQ(s["snapshot_store"].value("bytes_each", (int64_t)0), (int64_t)2048);
    CHECK_NEAR(s["stall"].value("restore_ms", 0.0), 2.0, 1e-9);
    CHECK_EQ(s["occupancy"].value("host_cap_bytes", (int64_t)0), (int64_t)(64 << 20));
    CHECK(s.value("enabled", false));
    CHECK_EQ(s["linear"]["evictions"].value("store_full", (int64_t)0), (int64_t)3);
    CHECK_EQ(s["linear"]["hit_cuts"].value("no_snapshot", (int64_t)0), (int64_t)2);
    CHECK_EQ(s["linear"].value("hit_cut_tokens", (int64_t)0), (int64_t)9000);
}

/* ================================================================== the page's field contract */

/* Every field the page's script reads from /stats (in its update and the request table) and from
 * /sessions (its loader and the session rows) is one the routes emit for a populated engine. A
 * rename on either side otherwise blanks a panel and reports nothing. */
/* The price of a restart, measured rather than guessed: whatever decides whether a restart is
 * worth suffering needs this figure, and a guessed cost would decide the answer by itself. The
 * page stopped consuming it on 2026-10-18, when the payback clause it fed was withdrawn -- the
 * field stays because it is the missing half of that decision made downstream, by a human or a
 * tool. A test runs in microseconds, so a real note_ready() would compute 0 -- indistinguishable
 * from "never called". Passing a synthetic start exercises the arithmetic through the real route. */
TEST(stats_reports_how_long_bringup_took_once_the_server_listens) {
    Dash d(false);
    /* Before the server has listened, there is no measured cost to report. */
    CHECK_EQ(d.get("/stats").value("startup_s", (int64_t)-1), (int64_t)0);

    d.srv->note_ready(rad_mono_ns() - 47ull * 1000000000ull);
    CHECK_EQ(d.get("/stats").value("startup_s", (int64_t)-1), (int64_t)47);

    /* A start in the future is not a negative cost: the figure is a duration, and a page that
     * divided by it would otherwise be handed a sign flip. */
    d.srv->note_ready(rad_mono_ns() + 10ull * 1000000000ull);
    CHECK_EQ(d.get("/stats").value("startup_s", (int64_t)-1), (int64_t)47);

    /* 0 means "not yet", so it must not overwrite a measured figure. */
    d.srv->note_ready(0);
    CHECK_EQ(d.get("/stats").value("startup_s", (int64_t)-1), (int64_t)47);
}

/* The pooled ring's element shape is a contract between two functions in the page, and no
 * browser runs here to notice when they disagree. accWindow pushes {ratio, dr, da}; acceptTriple
 * iterates the ring as `d` and reads `last`. When a rename left acceptTriple reading accPrev's
 * {r, a} instead, the pooled figures became NaN and the detail line froze at "recent -" while
 * every C++ test stayed green (Turnstone judge slices 08 and 11, 2026-10-20). So: scan the page
 * text and hold the two sides to the same names. */
TEST(the_pooled_acceptance_ring_reads_the_names_accWindow_writes) {
    Dash d(false);
    const std::string p = d.srv->router().call("GET", "/").body;
    const std::string aw = function_body(p, "function accWindow(");
    const std::string at = function_body(p, "function acceptTriple(");
    REQUIRE(!aw.empty() && !at.empty());
    /* what goes INTO the ring */
    CHECK(aw.find("dr: dr") != std::string::npos);
    CHECK(aw.find("da: da") != std::string::npos);
    /* what comes OUT of it: every `d.<field>` and `last.<field>` must be a ring name */
    const std::set<std::string> ring = {"dr", "da"};
    for (const std::string& n : fields(at, "d")) {
        CHECK(ring.count(n) == 1);
        if (!ring.count(n))
            fprintf(stderr, "    acceptTriple reads d.%s, which the ring does not carry\n", n.c_str());
    }
    for (const std::string& n : fields(at, "last")) {
        CHECK(ring.count(n) == 1);
        if (!ring.count(n))
            fprintf(stderr, "    acceptTriple reads last.%s, which the ring does not carry\n", n.c_str());
    }
    /* and the ring must not be empty by construction: the push is gated on dr > 0 */
    CHECK(aw.find("dr <= 0") != std::string::npos);
}

TEST(every_field_the_page_reads_is_one_the_routes_emit) {
    Dash d(false);
    HttpResponse page = d.srv->router().call("GET", "/");
    REQUIRE(page.status == 200 && has(page.body, "function update(s, dt)"));
    const std::string& p = page.body;
    const json stats = d.get("/stats");
    const json sessions = d.get("/sessions");

    const std::string update = function_body(p, "function update(s, dt)");
    const std::string batch = function_body(p, "function renderBatch(list, s)");
    REQUIRE(!update.empty() && !batch.empty());
    std::set<std::string> top = fields(update, "s");
    for (const std::string& n : fields(batch, "s")) top.insert(n);
    CHECK(top.size() > 15);
    each_field_is_emitted(top, stats, "stats");
    each_field_is_emitted(fields(update, "ex"), stats["experts"], "experts");
    each_field_is_emitted(fields(update, "t"), stats["experts"]["tiers"][0], "experts.tiers[]");
    std::set<std::string> link = fields(update, "lk");
    for (const char* fn : { "d", "pre", "dP" })
        for (const std::string& n : called(update, fn)) link.insert(n);
    each_field_is_emitted(link, stats["link"], "link");
    each_field_is_emitted(fields(update, "c"), stats["cards"][0], "cards[]");
    each_field_is_emitted(fields(update, "p"), stats["draft_positions"][0], "draft_positions[]");

    /* The request table: renderBatch through renderRequests. */
    const size_t rb = p.find("function renderBatch("), up = p.find("function update(s, dt)");
    REQUIRE(rb != std::string::npos && up > rb);
    each_field_is_emitted(fields(p.substr(rb, up - rb), "r"), stats["requests"][0], "requests[]");

    /* The sessions view: its loader, and the rows from reuseOf through sessRow. */
    const std::string load = function_body(p, "async function loadSessions()");
    REQUIRE(!load.empty());
    each_field_is_emitted(fields(load, "s"), sessions, "sessions");
    each_field_is_emitted(fields(load, "mv"), sessions["snapshot_store"], "snapshot_store");
    each_field_is_emitted(fields(load, "ln"), sessions["linear"], "linear");
    each_field_is_emitted(fields(load, "pool"), sessions["pool"], "pool");
    each_field_is_emitted(fields(load, "occ"), sessions["occupancy"], "occupancy");
    const size_t ro = p.find("const reuseOf"), lo = p.find("async function loadSessions()");
    REQUIRE(ro != std::string::npos && lo > ro);
    std::set<std::string> rows = fields(p.substr(ro, lo - ro), "r");
    for (const std::string& n : fields(load, "r")) rows.insert(n);
    each_field_is_emitted(rows, sessions["sessions"][0], "sessions[]");
}

/* ================================================================== the bridge's KV figures */

/* SchedulerBridge's aggregation over a hybrid model's groups: two paged groups share one token
 * axis, so the token figures are the fuller one's, not their sum; a linear state has no token
 * axis and adds only bytes; a windowed group's carve bounds lookback, not fit, so it neither
 * binds the token figures (a drafter's window sits full for the server's life) nor bounds
 * kv_tokens_max. Utilisation is the maximum over every group, bytes their sum, the cache's
 * unshared share is clamped to what is used, and the draft depth to the table's size. */
TEST(the_bridge_reads_the_binding_paged_group_and_sums_only_bytes) {
    rad::SchedMetrics m;
    m.n_kv_groups = 4;
    auto group = [&](int i, int64_t bt, int64_t total, int64_t free_, bool window, int64_t bytes) {
        m.kv[i].block_tokens = bt;
        m.kv[i].blocks_total = total;
        m.kv[i].blocks_free = free_;
        m.kv[i].utilisation = total ? (float)(1.0 - (double)free_ / (double)total) : 0.0f;
        m.kv[i].window = window;
        m.kv[i].bytes_total = bytes;
        m.kv[i].bytes_used = bytes / 2;
        m.kv[i].bytes_carved = bytes * 2;
    };
    group(0, 16, 100, 40, false, 1000);   /* attention, 60% */
    group(1, 16, 100, 10, false, 200);    /* a second paged group on the same axis, 90% */
    group(2, 0, 8, 0, false, 500);        /* linear state, full, no token axis */
    group(3, 16, 4, 0, true, 64);         /* a drafter's window, always full */
    m.prefix_held_tokens = 5000;
    m.draft_depth_seen = 70;
    for (int j = 0; j < 64; ++j) { m.draft_pos_reached[j] = 100 - j; m.draft_pos_accepted[j] = j; }

    const server::SchedMetrics o = bridge_metrics(m);
    CHECK_NEAR(o.kv_util, 1.0, 1e-6);
    CHECK_EQ(o.kv_tokens_total, (int64_t)(100 * 16));
    CHECK_EQ(o.kv_tokens_used, (int64_t)(90 * 16));
    CHECK_EQ(o.kv_tokens_max, (int64_t)(100 * 16));
    CHECK_EQ(o.kv_bytes_total, (int64_t)(1000 + 200 + 500 + 64));
    CHECK_EQ(o.kv_bytes_used, (int64_t)(500 + 100 + 250 + 32));
    CHECK_EQ(o.kv_bytes_carved, (int64_t)(2000 + 400 + 1000 + 128));
    CHECK_EQ(o.kv_tokens_cached, o.kv_tokens_used);
    CHECK_EQ(o.draft_depth_seen, server::SchedMetrics::MAX_DRAFT_POS);
    CHECK_EQ(o.draft_pos_reached[63], (int64_t)37);

    /* The fuller group binds wherever it sits in the list. */
    group(0, 16, 200, 10, false, 1000);
    const server::SchedMetrics p = bridge_metrics(m);
    CHECK_EQ(p.kv_tokens_total, (int64_t)(200 * 16));
    CHECK_EQ(p.kv_tokens_used, (int64_t)(190 * 16));
    CHECK_EQ(p.kv_tokens_max, (int64_t)(100 * 16));
}

/* The observation ledger crosses the bridge with its units intact: the scheduler's three row
 * counters, its cycle counter and its cost-observation counter arrive as such, never converted
 * into each other. */
TEST(the_bridge_carries_the_observation_ledger_with_its_units) {
    rad::SchedMetrics m;
    m.draft_obs_attempted = 96;
    m.draft_obs_applied = 51;
    m.draft_obs_skipped = 45;
    m.draft_obs_discarded = 3;
    m.draft_obs_inactive = 7;

    const server::SchedMetrics o = bridge_metrics(m);
    CHECK_EQ(o.draft_obs_rows_attempted, (int64_t)96);
    CHECK_EQ(o.draft_obs_rows_applied, (int64_t)51);
    CHECK_EQ(o.draft_obs_rows_skipped, (int64_t)45);
    CHECK_EQ(o.draft_obs_cycles_discarded, (int64_t)3);
    CHECK_EQ(o.draft_cost_observations_inactive, (int64_t)7);
}

RAD_TEST_MAIN()
