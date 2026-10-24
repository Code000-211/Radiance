/* scheduler.cpp -- one decision per step. See scheduler.h for the shape and spec §7.1 for why.
 */
#include "scheduler.h"
#include "mm/processor.h"

#include "server/sink.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace rad {

/* ------------------------------------------------------------------ ordering */
/* Priority first, arrival second, and arrival is what makes preemption fair without a special
 * case: a preempted request re-enters the queue carrying its ORIGINAL arrival number, which is
 * smaller than every request admitted after it, so it lands at the front of its priority class
 * rather than at the back of the queue. It has already been served once and its blocks were taken
 * from it; sending it to the back would be charging it twice. */
bool Scheduler::ahead_of(int32_t a, int32_t b) const {
    const SchedReq& x = *reqs_[(size_t)a];
    const SchedReq& y = *reqs_[(size_t)b];
    if (x.req->priority != y.req->priority) return x.req->priority > y.req->priority;
    return x.arrival < y.arrival;
}

void Scheduler::queue_insert(std::vector<int32_t>& q, int32_t slot) {
    auto it = std::lower_bound(q.begin(), q.end(), slot,
                               [&](int32_t a, int32_t b) { return ahead_of(a, b); });
    q.insert(it, slot);   /* capacity is reserved, so this moves and does not allocate */
}

void Scheduler::queue_erase(std::vector<int32_t>& q, int32_t slot) {
    auto it = std::find(q.begin(), q.end(), slot);
    if (it != q.end()) q.erase(it);
}

/* ------------------------------------------------------------------ init */
int Scheduler::init(Program& prog, const Config& cfg, IKVManager* kv, IPrefixCache* pc,
                    const std::vector<BatchBuilder::RankIO>& ranks, const ChunkGeometry* geo) {
    if (!kv || !pc) return RAD_E_INVAL;
    prog_ = &prog;
    cfg_ = cfg;
    kv_ = kv;
    pc_ = pc;

    /* The engine resolves this once and hands it over, because the prefix cache needs the
     * checkpoint interval out of it before this runs. A caller that passes nothing (the tests)
     * resolves its own -- harmless, it is a pure function of prog and cfg. What must not happen is
     * two independent reads of the operator's RAW interval: the cache is given the resolved
     * number, so there is one answer to "where does a chunk split". */
    if (geo) geo_ = *geo;
    else RAD_TRY(chunk_geometry_resolve(prog, cfg, &geo_));

    kvgeom_ = KVGeom::of(prog.kv_groups);
    for (size_t i = 0; i < kvgeom_.g.size(); ++i) {
        const int64_t w = kv->state_index_width((int32_t)i);
        if (w > 0) kvgeom_.g[i].sidx_width = w;
    }
    has_linear_state_ = false;
    for (const auto& g : kvgeom_.g) if (!g.paged) has_linear_state_ = true;

    /* A hit can only leave a drafter's own cache short where a PAGED group is not prefix-cached;
     * see scheduler.h at hit_holes_draft_. With the cache off, cached_groups() is empty and this
     * is true -- which costs nothing, because a run with no cache takes no hits and every
     * sequence's first block pass sits at ctx_len 0 anyway. */
    {
        /* PAGED AS THE CACHE COUNTS IT, which is KVGroupPlan::paged() and reaches here through
         * block_tokens(). NOT kvgeom_'s `paged` flag: that one is also true for an empty
         * placeholder group (0 layers, no plan), which a model is free to declare, so counting it
         * overstates the paged groups and pins this predicate true. That is exactly the drift the
         * comment in scheduler.h warns about. */
        size_t n_paged = 0;
        for (size_t i = 0; i < kvgeom_.g.size(); ++i)
            if (kv->block_tokens((int32_t)i) > 0) ++n_paged;
        hit_holes_draft_ = pc->cached_groups().size() < n_paged;
        RAD_INFO("draft floor: %zu paged group(s), %zu prefix-cached -> a hit %s",
                 n_paged, pc->cached_groups().size(),
                 hit_holes_draft_ ? "LEAVES a hole in a drafter's own KV" : "leaves no hole");
    }

    DraftController::Params dp;
    dp.max_depth = cfg.n_spec;
    const char* env_adapt = std::getenv("RADIANCE_ADAPTIVE_SPEC");
    dp.adaptive = cfg.adaptive_spec || (env_adapt && strcmp(env_adapt, "0") != 0);
    RAD_TRY(draft_.configure(dp));

    RAD_TRY(builder_.init(prog, cfg, geo_, kvgeom_, ranks));
    /* The builder's bound and not a second derivation of it: it is what the block tables were
     * sized for, so it is the one number that says which positions exist. */
    ctx_limit_ = builder_.max_ctx();
    n_deferred_ = 0;
    closed_ = false;
    closed_status_ = RAD_E_STATE;

    /* Everything the step path touches is reserved here. A vector that grows during a step is an
     * allocator call on the path this whole component exists to keep clear. */
    plan_.e.reserve((size_t)cfg.max_seqs);
    running_.reserve((size_t)cfg.max_seqs);
    waiting_.reserve((size_t)cfg.max_seqs * 4 + 64);
    free_slots_.reserve((size_t)cfg.max_seqs);
    free_cards_.resize((size_t)cfg.max_seqs);
    for (int64_t c = 0; c < cfg.max_seqs; ++c)
        free_cards_[(size_t)c] = (int32_t)(cfg.max_seqs - 1 - c);   /* 0 is taken first */
    reqs_.reserve((size_t)cfg.max_seqs);
    completed_.reserve((size_t)cfg.max_seqs);
    by_id_.reserve((size_t)cfg.max_seqs * 2);
    hit_blocks_.resize(prog.kv_groups.size());
    for (auto& v : hit_blocks_) v.reserve(64);
    return RAD_OK;
}

void Scheduler::fini() {
    for (auto& s : reqs_)
        if (s && s->req && (s->running || s->pending_free)) kv_->free_sequence(s->req->id);
    builder_.fini();
    reqs_.clear();
    free_slots_.clear();
    free_cards_.clear();
    by_id_.clear();
    running_.clear();
    waiting_.clear();
}

/* ------------------------------------------------------------------ arrival */
int Scheduler::add(std::unique_ptr<Request> r) {
    std::lock_guard<std::mutex> lk(mu_);
    return add_locked(std::move(r));
}

/* ONE LOCK FOR ALL OF THEM, so the step that admits the first finds every one waiting. Taken one
 * at a time, a step can begin between two of them and admit a prefix of the set, and which prefix
 * depends on thread timing -- so the batch a request's sequences decode in, and with it their
 * tie-level rounding, would vary between identical runs. */
void Scheduler::add_all(std::unique_ptr<Request>* rs, size_t n, int* rc) {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < n; ++i) rc[i] = add_locked(std::move(rs[i]));
}

int Scheduler::add_locked(std::unique_ptr<Request> r) {
    if (!r) return RAD_E_INVAL;
    if (r->prompt.empty()) return RAD_E_INVAL;
    if (by_id_.count(r->id)) return RAD_E_DUPLICATE;
    /* NO ENGINE BEHIND THE QUEUE ANY MORE. Accepting would leave the caller waiting on a step
     * that is never coming, so the sink is failed here -- it is the one thing the caller is
     * listening to -- and the refusal says why. */
    if (closed_) {
        RAD_WARN("request %llu refused: %s", (unsigned long long)r->id, closed_reason_);
        server::sink_fail(r->sink, closed_status_, closed_reason_);
        return RAD_E_STATE;
    }

    int32_t slot;
    if (!free_slots_.empty()) { slot = free_slots_.back(); free_slots_.pop_back(); }
    else { slot = (int32_t)reqs_.size(); reqs_.emplace_back(); }
    if (!reqs_[(size_t)slot]) reqs_[(size_t)slot] = std::make_unique<SchedReq>();

    SchedReq& s = *reqs_[(size_t)slot];
    s.req = std::move(r);
    s.arrival = ++arrival_;
    s.arrived_us = mono_us();
    s.first_tok_us = 0;
    s.slot = slot;
    s.card = -1;
    s.running = false;
    s.drafter_reset();
    if (running_.empty()) {
        draft_.reset_adaptive();
    }
    s.sched_tokens = 0;
    s.entry = -1;
    s.entry_next = -1;
    s.pending_free = false;
    s.reap_pending = false;
    s.ckpt_src = -1;
    s.hit_attn = 0;
    s.hit_owed = false;
    s.req->state = ReqState::Waiting;

    /* The output vector is the one thing the step path appends to, so its capacity is settled
     * here, on the arrival thread, and never again. Bounded by what the context can still hold:
     * max_tokens is the caller's number, and reserving it as given lets one request ask this
     * thread for gigabytes it could never fill. */
    int64_t out_cap = s.req->max_tokens > 0 ? (int64_t)s.req->max_tokens : 1;
    const int64_t ctx_room = ctx_limit_ - (int64_t)s.req->prompt.size();
    if (ctx_limit_ > 0 && out_cap > ctx_room) out_cap = ctx_room > 1 ? ctx_room : 1;
    s.req->output.reserve((size_t)out_cap + (size_t)RAD_SCHED_MAX_SPEC);

    by_id_[s.req->id] = slot;
    /* Keep enough headroom that a step which preempts every running request cannot grow the
     * queue's storage while it holds the lock. */
    if (waiting_.capacity() < waiting_.size() + (size_t)cfg_.max_seqs + 1)
        waiting_.reserve(waiting_.size() + (size_t)cfg_.max_seqs + 1);
    queue_insert(waiting_, slot);
    pub_dirty_.store(true, std::memory_order_relaxed);
    return RAD_OK;
}

Request* Scheduler::find(uint64_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_id_.find(id);
    return it == by_id_.end() ? nullptr : reqs_[(size_t)it->second]->req.get();
}

int Scheduler::set_draft(uint64_t id, const int32_t* tokens, int n) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return RAD_E_NOTFOUND;
    SchedReq& s = *reqs_[(size_t)it->second];
    if (n < 0) return RAD_E_INVAL;

    /* THE NEXT STEP ALREADY VERIFIES THESE. It was planned with the count the drafter issued, and
     * the card already put the tokens in its batch; the host copy is what that step's commit
     * emits the accepted ones from, so it takes exactly those, unclamped. */
    if (next_open_ && s.entry_next >= 0) {
        if (n != s.draft_next) {
            RAD_ERR("request %llu: %d draft(s) published for a step planned to verify %d",
                    (unsigned long long)id, n, s.draft_next);
            return RAD_E_STATE;
        }
        for (int i = 0; i < n; ++i) s.draft[i] = tokens[i];
        s.n_draft = n;
        s.req->n_draft = n;
        s.req->draft = n ? s.draft : nullptr;
        return RAD_OK;
    }

    /* An ineligible request is given no proposal at all rather than a refusal: it decodes one
     * token a step, exactly as it would on a server started without a drafter. See
     * Request::may_speculate for the two reasons, and step() for the gate that catches the n-gram
     * fallback as well. */
    if (!s.req->may_speculate) {
        s.n_draft = 0; s.req->n_draft = 0; s.req->draft = nullptr;
        return RAD_OK;
    }

    int cap = draft_.next_depth((int)running_.size());
    cap = s.suggest_depth(cap, draft_.params().adaptive, running_.size() > 1, draft_.params().max_depth);
    if (s.req->max_tokens > 0) {
        const int64_t rem = (int64_t)s.req->max_tokens - (int64_t)s.req->output.size();
        const int64_t max_d = rem > 1 ? rem - 1 : 0;
        if (n > max_d) n = (int)max_d;
    }
    if (n > cap) n = cap;
    if (n > RAD_SCHED_MAX_SPEC) n = RAD_SCHED_MAX_SPEC;
    for (int i = 0; i < n; ++i) s.draft[i] = tokens[i];
    s.n_draft = n;
    s.req->n_draft = n;
    s.req->draft = s.draft;
    return RAD_OK;
}

/* ------------------------------------------------------------------ cancellation */
/* ------------------------------------------------------------------ the sink
 *
 * THE TOKENS HAVE TO REACH THE CONNECTION, and this is where they leave the engine. `Request::sink`
 * is the server's per-request channel (core/server/sink.h) and it is a `void*` on Request precisely
 * so that core/sched does not depend on the server's types; the inline helpers there are the call
 * sites. A null sink is legal and means nobody is listening -- an offline or benchmark request.
 *
 * Appending to `Request::output` is not enough on its own: without a push the engine generates
 * correctly, step after step, while the HTTP thread waits in `Sink::wait()` until the client times
 * out. Every sink_* call -- push, finish, prompt stats, admission and drafts -- is made from this
 * file and nowhere else.
 *
 * A push returns false ONLY on overflow, and an overflowing sink has already failed the request --
 * a stream missing tokens in the middle is worse than one that stops and says so -- so the return
 * is what tells the scheduler to stop generating for it. */
static server::Finish finish_of(ReqState st, const std::string& reason) {
    if (st == ReqState::Cancelled) return server::Finish::Cancelled;
    if (st == ReqState::Failed)    return server::Finish::Error;
    if (reason == "length")        return server::Finish::Length;
    return server::Finish::Stop;
}

/* THE CACHE IS NOT ALLOWED TO OWN THE WHOLE POOL, and that is a policy decision rather than an
 * optimisation. Every block the cache indexes is a block it holds a reference on, so a cache free
 * to grow without limit drives the pool to full occupancy and keeps it there: admission then pays
 * eviction on every chunk of every prefill, `kv_cache_usage_perc` pins at 1.0, the scheduler
 * spends its time reclaiming rather than running, and a long enough run of distinct large prompts
 * wedges.
 *
 * So a reserve is kept free. It is a fraction of the pool rather than a token count because the
 * thing being protected is the ability to admit ANY sequence, and how big that is depends on the
 * deployment. Trimming happens after a publish, off the step path, where the blocks were just
 * added -- not during admission, where the cost lands on a request that did nothing wrong. */

/* Drop every entry, in VRAM or off it -- IPrefixCache::reset, because an eviction does not
 * mean "forgotten" for an entry the idle tiers hold a copy of. */
int64_t Scheduler::reset_prefix_cache() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!pc_ || !cfg_.prefix_cache) return RAD_E_UNSUPPORTED;
    const int64_t dropped = pc_->reset();
    RAD_INFO("prefix cache: reset, %lld entries dropped", (long long)dropped);
    return dropped;
}

void Scheduler::trim_prefix_cache() {
    if (!pc_ || !cfg_.prefix_cache) return;
    const std::vector<int32_t>& cg = pc_->cached_groups();
    if (cg.empty()) return;
    const int32_t g = cg[0];
    const int64_t total = kv_->total_blocks(g);
    if (total <= 0) return;
    const int64_t reserve = total / 4;              /* a quarter of the pool stays free */
    for (int guard = 0; guard < 4096; ++guard) {
        /* FREE AGAINST THE CARVE, not free_blocks(). On an elastic pool free_blocks() is bounded
         * by the live mark, so a pool that has handed its unused top back reads as nearly full
         * however little anything holds -- and this loop would then drain the whole cache to
         * make room the pool can grow back into on demand. */
        if (total - kv_->held_blocks(g) >= reserve) return;
        if (pc_->evict(256) <= 0) return;           /* nothing left to give back */
    }
    RAD_WARN("prefix cache: could not free %lld of %lld blocks in group %d after 4096 rounds; the "
             "pool is held by live sequences rather than by the cache",
             (long long)reserve, (long long)total, (int)g);
}

void Scheduler::close_request(SchedReq& s, ReqState st, const char* reason, bool defer_free,
                              bool publish, int status, const char* why) {
    /* PUBLISH BEFORE FREEING. This is the only moment the sequence's whole prefix is both
     * computed and still owned, and it is what makes the cache a cache rather than a permanently
     * empty lookup: insert() takes its own KV references, so the blocks survive the free below.
     * A cancelled or failed request is published too -- its prompt was computed either way, and
     * an agent client that hangs up mid-answer is exactly who asks for that prefix again. */
    if (publish && cfg_.prefix_cache && pc_ && kv_->has_sequence(s.req->id) &&
        s.req->n_computed > 0) {
        const std::vector<int32_t>& cg = pc_->cached_groups();
        pub_blocks_.assign(cg.size(), {});
        bool ok = !cg.empty();
        for (size_t i = 0; i < cg.size(); ++i) {
            /* A rebased table does not describe the prefix from position zero, so it cannot be
             * published. cached_groups() already excludes windowed groups, which is where a
             * moving floor comes from; this is the belt to that. */
            if (kv_->first_block_pos(s.req->id, cg[i]) != 0) { ok = false; break; }
            pub_blocks_[i] = kv_->block_table(s.req->id, cg[i]);
        }
        if (ok) {
            const std::vector<int32_t> hot = pc_->current_hot_units();
            const int rc = pc_->insert(*s.req, pub_blocks_, s.req->n_computed, hot);
            if (rc < 0)
                RAD_DEBUG("prefix cache: request %llu not published: %s",
                          (unsigned long long)s.req->id, rad_strerror(rc));
            trim_prefix_cache();
        }
    }

    if (defer_free) {
        if (!s.pending_free) ++n_deferred_;
        s.pending_free = true;
    } else if (kv_->has_sequence(s.req->id)) {
        /* A preempted request has already had its blocks taken but the manager still knows the
         * sequence, and it may still own a state slot. Asking first is what keeps this from
         * depending on free_sequence() being idempotent. */
        kv_->free_sequence(s.req->id);
    }
    if (!defer_free) drop_card(s);
    s.running = false;
    s.req->state = st;
    if (s.req->finish_reason.empty() && reason) s.req->finish_reason = reason;
    queue_erase(running_, s.slot);
    queue_erase(waiting_, s.slot);
    s.n_draft = 0;

    /* Terminal, and idempotent on the sink's side: the first call wins, so a cancel racing a
     * natural stop does not produce two finish reasons. This is the last thing the producer does
     * to the sink, and the HTTP thread is waiting on exactly it. */
    if (st == ReqState::Failed && status < 0) server::sink_fail(s.req->sink, status, why);
    else server::sink_finish(s.req->sink, finish_of(st, s.req->finish_reason));
    pub_dirty_.store(true, std::memory_order_relaxed);
}

int Scheduler::cancel(uint64_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return RAD_E_NOTFOUND;
    SchedReq& s = *reqs_[(size_t)it->second];
    if (s.req->state == ReqState::Finished || s.req->state == ReqState::Cancelled)
        return RAD_OK;

    /* Blocks go back NOW, not when the step that would have finished this request runs. A
     * cancelled request that keeps its pool share until completion is a pool that shrinks under
     * exactly the load that generates cancellations (spec §14).
     *
     * The one exception is a request whose tokens are in the step the device is still executing.
     * Its blocks are released at commit instead -- one step later, not one request later. Handing
     * a live block to a new sequence mid-step is a race, and it is the kind that presents as a
     * model quality problem rather than as a crash.
     *
     * AND A DRAFTER'S SNAPSHOT IS THE OTHER. The drafter builds its passes one at a time after
     * commit, releasing this lock between them, so a cancel can land after one pass was built
     * and before the next. Freeing then leaves the next pass naming a sequence with no blocks:
     * its block-table row is all -1 while its length is not, and attention reads block -1. The
     * blocks are released once the drafter is done with the sequence (sweep_deferred). */
    const bool in_flight = in_step(s) || block_holds(id);
    close_request(s, ReqState::Cancelled, "cancelled", in_flight);
    ++m_.cancelled;
    completed_.push_back(id);
    return RAD_OK;
}

void Scheduler::take_completed(std::vector<uint64_t>* out) {
    if (!out) return;
    out->clear();
    std::lock_guard<std::mutex> lk(mu_);
    if (!next_open_) { out->swap(completed_); return; }
    /* A REQUEST THE NEXT STEP STILL CARRIES IS HELD BACK. That step's sampler is staged for it
     * after this one commits, so its sampler state has to outlive the finish by one step; it is
     * handed out once the step that carries it has committed. */
    size_t keep = 0;
    for (uint64_t id : completed_) {
        auto it = by_id_.find(id);
        const bool held = it != by_id_.end() && reqs_[(size_t)it->second]->entry_next >= 0;
        if (held) completed_[keep++] = id;
        else out->push_back(id);
    }
    completed_.resize(keep);
}

void Scheduler::release_locked(SchedReq& s) {
    if (s.pending_free) { free_kv(s); s.pending_free = false; --n_deferred_; }
    if (s.reap_pending) { s.reap_pending = false; --n_deferred_; }
    by_id_.erase(s.req->id);
    s.req.reset();
    free_slots_.push_back(s.slot);
}

void Scheduler::free_kv(SchedReq& s) {
    kv_->free_sequence(s.req->id);
    drop_card(s);
}

void Scheduler::drop_card(SchedReq& s) {
    if (s.card < 0) return;
    free_cards_.push_back(s.card);
    s.card = -1;
}

bool Scheduler::block_holds(uint64_t id) const {
    for (const BlockSeq& d : block_) if (d.seq == id) return true;
    for (const SerialSeq& m : serial_) if (m.seq == id) return true;
    return false;
}

const Request* Scheduler::live_req(int32_t slot, uint64_t id) const {
    if (slot < 0 || (size_t)slot >= reqs_.size() || !reqs_[(size_t)slot]) return nullptr;
    const SchedReq& s = *reqs_[(size_t)slot];
    return (s.req && s.req->id == id) ? s.req.get() : nullptr;
}

void Scheduler::sweep_deferred() {
    if (n_deferred_ <= 0) return;
    for (auto& up : reqs_) {
        if (!up) continue;
        SchedReq& s = *up;
        if (!s.req || (!s.reap_pending && !s.pending_free)) continue;
        if (in_step(s)) continue;
        if (block_holds(s.req->id)) continue;
        if (s.reap_pending) { release_locked(s); continue; }
        /* A free deferred on a drafter snapshot, for a request nobody has reaped yet. The Request
         * stays until the reap; only the blocks go back. */
        free_kv(s);
        s.pending_free = false;
        --n_deferred_;
    }
}

bool Scheduler::has_deferred() const {
    std::lock_guard<std::mutex> lk(mu_);
    return n_deferred_ > 0;
}

void Scheduler::release_deferred() {
    std::lock_guard<std::mutex> lk(mu_);
    sweep_deferred();
}

void Scheduler::shutdown(const char* reason, int status) {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    closed_reason_ = reason ? reason : "the engine has stopped";
    closed_status_ = status < 0 ? status : RAD_E_STATE;
    /* Copied first: close_request erases from both queues. */
    std::vector<int32_t> live(running_);
    live.insert(live.end(), waiting_.begin(), waiting_.end());
    for (int32_t slot : live) {
        SchedReq& s = *reqs_[(size_t)slot];
        if (!s.req) continue;
        close_request(s, ReqState::Failed, closed_reason_, /*defer_free=*/false,
                      /*publish=*/false, closed_status_, closed_reason_);
        ++m_.failed;
        completed_.push_back(s.req->id);
    }
    /* Nothing will commit again, so no entry of the last step still owns its slot. */
    step_open_ = false;
    next_open_ = false;
    for (auto& up : reqs_) if (up) { up->entry = -1; up->entry_next = -1; }
    block_.clear();
    serial_.clear();
    sweep_deferred();
}

int Scheduler::reap(uint64_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return RAD_E_NOTFOUND;
    int32_t slot = it->second;
    SchedReq& s = *reqs_[(size_t)slot];
    if (s.req->state == ReqState::Running || s.req->state == ReqState::Waiting ||
        s.req->state == ReqState::Preempted)
        return RAD_E_STATE;
    /* NOT WHILE ANYTHING STILL POINTS AT THE SLOT, and two things can. The server reaps as soon as
     * the sink is finished, and cancel() finishes the sink at once, so a client that hangs up
     * mid-generation gets here at an arbitrary point in the step loop.
     *
     *   - a StepEntry, if the device is still running the step this request is in. Destroying the
     *     Request there segfaults commit() on `r.state`; returning the slot there is worse,
     *     because a request admitted before the commit lands would be handed the entry and the
     *     tokens.
     *   - the DRAFTER'S SNAPSHOT. block_begin takes it under the lock and block_context,
     *     block_query and block_end read it again after releasing and retaking the lock; a
     *     Request freed in that window leaves BatchBuilder::build reading tokens off freed
     *     memory. Keeping it alive rather than dropping the row is what
     *     preserves the drafter's row mapping -- the engine already sized its readback from
     *     block_begin's answer, so a query pass with one fewer row would attribute one
     *     sequence's drafts to another.
     *
     * Both are the same one-step window `pending_free` covers, and both close the same way. */
    if (in_step(s) || block_holds(id)) {
        if (!s.reap_pending) ++n_deferred_;
        s.reap_pending = true;
        return RAD_OK;
    }
    release_locked(s);
    return RAD_OK;
}

int Scheduler::queue_snapshot(uint64_t* out, int max) const {
    std::lock_guard<std::mutex> lk(mu_);
    int n = 0;
    for (int32_t slot : waiting_) {
        if (n >= max) break;
        out[n++] = reqs_[(size_t)slot]->req->id;
    }
    return n;
}


/* WHAT EVERY REQUEST IS DOING RIGHT NOW. Copied out under the step lock rather than handed out as
 * pointers: a caller holding a SchedReq* across a step is holding one that commit() may reap.
 *
 * Oldest first. A dashboard's reader is looking for the one that is stuck, and the one that is
 * stuck is the one that has been here longest. */
int Scheduler::requests(ReqStat* out, int max) const {
    std::lock_guard<std::mutex> lk(mu_);
    return requests_locked(out, max);
}

int Scheduler::requests_locked(ReqStat* out, int max) const {
    const int64_t now = mono_us();
    const int32_t n_groups = prog_ ? (int32_t)prog_->kv_groups.size() : 0;

    int n = 0;
    for (const auto& p : reqs_) {
        if (n >= max) break;
        if (!p || !p->req) continue;
        const SchedReq& s = *p;
        const Request&  r = *s.req;
        ReqStat& o = out[n++];
        o.id              = r.id;
        o.state           = (int)r.state;
        o.running         = s.running;
        o.prompt_tokens   = (int64_t)r.prompt.size();
        o.computed_tokens = r.n_computed;
        o.cached_tokens   = r.n_cached;
        o.output_tokens   = (int64_t)r.output.size();
        o.max_tokens      = r.max_tokens;
        o.ctx_tokens      = s.seq_len();
        /* BLOCKS ADD UP AND TOKENS DO NOT. Each paged group was carved its own blocks, so a count
         * of them is a sum; but the paged groups SHARE ONE TOKEN AXIS -- one token of context
         * takes a slot in every one of them -- so summing their tokens reports a request holding
         * as many times its own context as there are paged groups. Against a pool capacity that
         * is correctly one group's, two 100K requests then read as 95% of a pool that is 72%
         * full. The token figure is the largest group's, which is the context the request is
         * holding, and it is comparable with SchedMetrics::kv_tokens_used by construction. */
        o.kv_blocks = o.kv_tokens = 0;
        for (int32_t g = 0; g < n_groups; ++g) {
            const int64_t bt = kv_->block_tokens(g);
            if (bt <= 0) continue;                    /* stateful groups are not paged */
            const int64_t held = (int64_t)kv_->block_table(r.id, g).size();
            o.kv_blocks += held;
            if (held * bt > o.kv_tokens) o.kv_tokens = held * bt;
        }
        o.n_draft         = s.n_draft;
        o.slot            = s.slot;
        o.age_s           = (double)(now - s.arrived_us) / 1e6;
        o.ttft_s          = s.first_tok_us ? (double)(s.first_tok_us - s.arrived_us) / 1e6 : -1.0;
        /* Since its FIRST token, not since admission: a request that waited thirty seconds in the
         * queue did not decode slowly, and dividing by the wait would say it did. */
        const double dec = s.first_tok_us ? (double)(now - s.first_tok_us) / 1e6 : 0.0;
        o.decode_tps      = dec > 1e-3 ? (double)r.output.size() / dec : 0.0;
    }
    std::sort(out, out + n, [](const ReqStat& a, const ReqStat& b) { return a.id < b.id; });
    return n;
}

/* ------------------------------------------------------------------ drafting */
/* PROMPT-LOOKUP DRAFTING, and it needs no second model. Take the last few committed tokens of a
 * sequence, find where that pattern last occurred in the SAME sequence, and propose whatever
 * followed it. On the workloads this engine is for -- a coder model quoting the file it was given,
 * a model rewriting a diff, JSON with repeated keys -- that guess is right often enough to pay,
 * and when it is wrong the verify step still emits the target's own token, so it can only cost
 * bandwidth. It is here rather than behind the drafter plugin interface because it is not a model:
 * there is nothing to load, nothing to place, and no second forward pass.
 *
 * Longest pattern first: a longer match is a stronger prediction, and dropping to a shorter one
 * only when the longer finds nothing is what keeps the acceptance rate from collapsing on common
 * bigrams. Most recent occurrence first, for the same reason.
 *
 * A draft head, when one is declared, replaces this by calling set_draft() before step() -- which
 * is why this only fills a proposal that is still empty. */
int Scheduler::draft_ngram(SchedReq& s, int depth) const {
    /* RADIANCE_NO_DRAFT separates the two halves of speculation. With it set the deployment is
     * still speculative -- the state pool is laid out for it and the linear kernels take the
     * path that defers the state -- but no step ever carries a draft, so every step is an
     * ordinary one-token decode and the output must match a non-speculative run exactly. It
     * answers 'is the deferred state right' without 'is the rollback right' on top of it. */
    static const bool none = [] { const char* e = getenv("RADIANCE_NO_DRAFT"); return e && *e == '1'; }();
    if (none) return 0;
    if (depth <= 0 || !s.req) return 0;
    const Request& r = *s.req;
    const int64_t np = (int64_t)r.prompt.size(), no = (int64_t)r.output.size();
    const int64_t n  = np + no;
    if (n < 2) return 0;
    auto at = [&](int64_t i) -> int32_t {
        return i < np ? r.prompt[(size_t)i] : r.output[(size_t)(i - np)];
    };
    for (int k = kNgramMax; k >= kNgramMin; --k) {
        if (n < (int64_t)k + 1) continue;
        const int64_t min_j = std::max<int64_t>(0, n - 2048);
        for (int64_t j = n - k - 1; j >= min_j; --j) {
            bool hit = true;
            for (int t = 0; t < k && hit; ++t) hit = (at(j + t) == at(n - k + t));
            if (!hit) continue;
            int d = 0;
            while (d < depth && j + k + d < n) { s.draft[d] = at(j + k + d); ++d; }
            return d;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ chunking */
/* A prefill chunk splits at the linear-state checkpoint interval so that a checkpoint is always
 * WRITTEN. vLLM writes one only when a scheduler step happens to end on a boundary, which is why
 * a checkpoint landing in request-unique tokens silently drops caching to zero (spec §7.3).
 *
 * The order of the two clamps matters. The checkpoint clamp comes first because the interval is a
 * multiple of the quantum, so a chunk clamped to an interval is already quantum-aligned and the
 * second clamp is a no-op on it. Reversing them would round a chunk PAST the checkpoint down and
 * then miss the boundary, which is the exact failure this is written to avoid. */
int64_t Scheduler::plan_chunk(int64_t n_computed, int64_t n_target, int64_t budget,
                              int64_t* checkpoint_pos) const {
    if (checkpoint_pos) *checkpoint_pos = -1;
    int64_t remaining = n_target - n_computed;
    if (remaining <= 0 || budget <= 0) return 0;

    int64_t want = remaining < budget ? remaining : budget;

    if (geo_.checkpoints()) {
        const int64_t I = geo_.checkpoint_interval;
        int64_t next = ((n_computed / I) + 1) * I;      /* strictly after n_computed */
        if (n_computed + want > next) want = next - n_computed;
        if (n_computed + want == next && checkpoint_pos) *checkpoint_pos = next;
    }

    if (want < remaining) {
        /* A non-final chunk has to end on a quantum boundary measured from the SEQUENCE origin,
         * not from the start of the step, or the next chunk hands the linear kernel a partial
         * tile and the attention kernel a partial block. The final chunk of a request is allowed
         * to be short: that is the accepted cost of owning the boundaries. */
        const int64_t Q = geo_.quantum;
        int64_t end = n_computed + want;
        int64_t aligned = end / Q * Q;
        if (aligned <= n_computed) {
            if (checkpoint_pos) *checkpoint_pos = -1;
            return 0;                     /* the budget left cannot hold one quantum */
        }
        if (aligned < end) {
            want = aligned - n_computed;
            if (checkpoint_pos) *checkpoint_pos = -1;
        }
    }
    return want;
}

/* ------------------------------------------------------------------ preemption */
void Scheduler::preempt_back() {
    int32_t slot = running_.back();
    SchedReq& s = *reqs_[(size_t)slot];
    Request& r = *s.req;

    /* Preemption by RECOMPUTE: the blocks go back to the pool and the request restarts from zero.
     * Its generated output is simply the tail of its token sequence, so recompute needs no special
     * path -- it re-prefills prompt ++ output through the same chunker. */
    kv_->preempt(r.id);
    drop_card(s);
    r.n_computed = 0;
    r.n_cached = 0;
    r.n_accepted = 0;
    r.state = ReqState::Preempted;
    s.drafter_reset();
    /* AND THE REQUEST'S OWN COPY OF THE WINDOW. The sampler stages 1 + Request::n_draft rows for
     * a request (Sampler::build_step), and a drafter may have set it for the step this preemption
     * just took away. Left standing, the recompute's first sampled step stages rows its batch does
     * not have, and every request after it in the step samples another's logits row. */
    r.n_draft = 0;
    r.draft = nullptr;
    s.running = false;

    running_.pop_back();
    queue_insert(waiting_, slot);   /* original arrival -> the front of its priority class */
    ++m_.preemptions;
}

/* ------------------------------------------------------------------ admission */
/* Admission checks the KV budget BEFORE a request enters running. A request admitted without the
 * blocks to run it is a request that will be preempted on its first step, which costs a full
 * recompute for nothing. */
int Scheduler::admit_one(SchedReq& s, int64_t budget) {
    Request& r = *s.req;

    if (!kv_->has_sequence(r.id)) RAD_TRY(kv_->add_sequence(r.id));

    /* A FRESH ATTEMPT: nothing looked up yet, nothing counted yet. A request that kept an
     * earlier attempt's hit (n_computed > 0) keeps what that attempt found as well. */
    if (r.n_computed == 0) {
        s.ckpt_src = -1;
        s.hit_attn = 0;
        s.hit_owed = true;
    }

    if (r.n_computed == 0 && cfg_.prefix_cache && pc_) {
        /* BEFORE the lookup, because the lookup stops at the first block the tiers hold -- an
         * honest shorter hit, but a shorter one. This is where a cold session gets its
         * context back instead of re-prefilling it: it costs a copy, and the alternative
         * costs a prefill. */
        if (pc_->restore(r) == IPrefixCache::Restore::Wait) return kAdmitWait;
        CacheHit h = pc_->lookup(r, &hit_blocks_);
        int32_t reuse = h.n_attn_tokens;
        if (geo_.checkpoints() && h.n_linear_tokens < reuse) {
            /* A hit landing between checkpoints reuses the attention blocks and must replay the
             * linear layers forward from the last checkpoint (§7.3). RadBatch carries ONE query
             * length for every group, so this build cannot run the linear layers over a longer
             * range than the attention layers: that would need per-group q_lens on
             * RadKVGroupBatch. Without them, the conservative and correct answer is to recompute
             * from the checkpoint. */
            reuse = h.n_linear_tokens;
        }
        int32_t last = (int32_t)r.prompt.size() - 1;   /* one token must remain to produce logits */
        if (reuse > last) reuse = last;
        if (reuse < 0) reuse = 0;

        /* A HIT ON A HYBRID MODEL IS ONLY LEGAL BECAUSE OF THE CHECKPOINT. Reuse skips recomputing
         * the prefix, which the adopted blocks cover for the paged attention groups and NOTHING
         * covers for a recurrent group -- its state lives in a per-sequence slot. So a hit that
         * skips tokens must also restore that state, which is what `ckpt_src` carries into the
         * entry below and what KVManager::checkpoint_restore performs.
         *
         * Refusing to reuse without one is therefore not conservatism, it is the only correct
         * answer: `reuse` is already clamped to `h.n_linear_tokens` above, so a hit that found no
         * valid checkpoint clamps to zero on its own. What must never happen is reuse > 0 with no
         * snapshot to resume from -- that is every recurrent layer continuing from whatever the
         * slot last held, and it reads as fluent text rather than as a failure. */
        if (reuse > 0 && has_linear_state_ && h.checkpoint_slot < 0) reuse = 0;

        /* Claim the cached blocks. Without this the hit is only an accounting entry: the sequence
         * would skip computing those tokens and then attend to blocks it does not own. adopt()
         * takes a whole block multiple, which `reuse` is -- it is a multiple of the checkpoint
         * interval, which geometry.h made a multiple of the quantum, which is a multiple of the
         * block size. */
        if (reuse > 0) {
            const std::vector<int32_t>& cg = pc_->cached_groups();
            for (size_t i = 0; i < cg.size() && i < hit_blocks_.size(); ++i) {
                int64_t nb = reuse / kvgeom_.g[(size_t)cg[i]].block_size;
                if ((int64_t)hit_blocks_[i].size() > nb) hit_blocks_[i].resize((size_t)nb);
                if (kv_->adopt(r.id, cg[i], hit_blocks_[i], reuse) < 0) {
                    /* AND WHAT THE EARLIER GROUPS ADOPTED GOES BACK. Left in the table, those
                     * blocks would be the first blocks of a sequence that now starts from zero,
                     * and the recompute would write its KV into the cache's own blocks. */
                    kv_->preempt(r.id);
                    reuse = 0;
                    break;
                }
            }
        }

        r.n_computed = reuse;
        r.n_cached = reuse;
        s.ckpt_src = (reuse > 0 && reuse == h.n_linear_tokens) ? h.checkpoint_slot : -1;
        s.hit_attn = h.n_attn_tokens;
    }

    int64_t target = s.seq_len();
    int64_t ckpt = -1;
    int64_t want = plan_chunk(r.n_computed, target, budget, &ckpt);
    int rc = RAD_OK;
    if (want <= 0)                                          rc = RAD_E_FULL;
    else if (s.card < 0 && free_cards_.empty())             rc = RAD_E_FULL;
    else if (kv_->ensure(r.id, r.n_computed + want) < 0)    rc = RAD_E_NOMEM;
    if (rc < 0) {
        /* A HIT THAT RESUMES FROM A SNAPSHOT IS NOT KEPT ACROSS A FAILED ATTEMPT. The slot is
         * named by id only, and until this request's first step restores from it nothing stops
         * the cache retiring it and handing it to another sequence's checkpoint -- so a later
         * attempt that trusted the id would restore somebody else's recurrent state, which reads
         * as fluent text about the wrong conversation. Giving the hit back costs the next attempt
         * a lookup; the attention-only hit of a model with no recurrent state holds only
         * refcounted blocks and is kept. */
        if (s.ckpt_src >= 0) undo_hit(s);
        return rc;
    }

    /* Counted once per admission, here, where it is known to have happened. */
    if (s.hit_owed) {
        m_.prompt_tokens += (int64_t)r.prompt.size();
        if (cfg_.prefix_cache) {
            m_.attn_cached_tokens += s.hit_attn;
            m_.linear_cached_tokens += r.n_cached;
        }
        s.hit_owed = false;
    }

    if (s.card < 0) { s.card = free_cards_.back(); free_cards_.pop_back(); }
    r.state = ReqState::Running;
    s.running = true;
    server::sink_admitted(r.sink);
    s.sched_tokens = (int32_t)want;
    s.pending_ckpt = ckpt;
    ++m_.admitted;
    return RAD_OK;
}

void Scheduler::undo_hit(SchedReq& s) {
    Request& r = *s.req;
    /* preempt() rather than free_sequence(): the sequence stays known to the manager, at zero
     * tokens and holding nothing, which is exactly where a fresh attempt starts from. */
    if (kv_->has_sequence(r.id)) kv_->preempt(r.id);
    r.n_computed = 0;
    r.n_cached = 0;
    s.ckpt_src = -1;
    s.hit_attn = 0;
}

/* ------------------------------------------------------------------ the step */
const RadBatch* Scheduler::step() {
    std::lock_guard<std::mutex> lk(mu_);

    /* The drafter's snapshots describe the step before this one and nothing reads them from here
     * on. block_end() and serial_end() clear them already; this covers a drafter that stopped
     * part way through, so a stale snapshot cannot keep deferring a release. */
    block_.clear();
    serial_.clear();
    /* And the snapshots the last step's admissions were handed have been read by its forward,
     * which committed before this was called. */
    if (pc_) pc_->begin_step();

    plan_.reset();
    step_prefill_tok_ = 0;
    step_decode_tok_ = 0;

    int64_t budget = cfg_.max_tok;
    bool preempted_here = false;

    /* A STEP THAT ALSO CARRIES A PREFILL TAKES A DRAFT LIKE ANY OTHER, and that rests on how the
     * linear blocks issue a mixed step.
     *
     * The blocks issue the RECURRENT path over the decode rows of a mixed step and the CHUNKED
     * path over the prefill ones, so a verifying decode row carries num_accepted and rolls back
     * exactly as it does in a pure decode step. Were a mixed step prefill-shaped for every
     * sequence in it, a verifying row would take the chunked path -- which has no num_accepted
     * operand, because a chunk has nothing to reject -- and its recurrent state and conv window
     * would advance over all 1 + n_spec tokens with no way back. The rejected tokens stay folded
     * into the state and the model is then conditioned on text it never emitted: fluent, and
     * drifting, so it surfaces as concurrent requests disagreeing with the same requests run
     * alone rather than as anything failing. */
    int n_running_dec = 0;
    for (size_t k = 0; k < running_.size(); ++k) {
        int32_t sl = running_[k];
        const SchedReq& sr = *reqs_[(size_t)sl];
        const Request& req = *sr.req;
        const int64_t tgt = sr.seq_len();
        const bool is_pf = (req.n_computed < (int64_t)req.prompt.size()) ||
                           (req.n_computed + 1 < tgt);
        if (!is_pf) n_running_dec++;
    }
    int depth = draft_.next_depth((int)running_.size());
    if (running_.size() == 1 && (size_t)running_[0] < reqs_.size() && reqs_[(size_t)running_[0]]) {
        depth = reqs_[(size_t)running_[0]]->suggest_depth(depth, draft_.params().adaptive, false, draft_.params().max_depth);
    }

    /* C1: Reserve token budget for running decode requests so that a long prefill chunk
     * doesn't exhaust the token budget and starve active decoders in concurrent workloads. */
    int64_t decode_reserved = 0;
    for (size_t k = 0; k < running_.size(); ++k) {
        int32_t sl = running_[k];
        const SchedReq& sr = *reqs_[(size_t)sl];
        const Request& req = *sr.req;
        const int64_t tgt = sr.seq_len();
        const bool is_pf = (req.n_computed < (int64_t)req.prompt.size()) ||
                           (req.n_computed + 1 < tgt);
        if (!is_pf) {
            decode_reserved += (1 + (req.may_speculate ? depth : 0));
        }
    }

    /* ---------------------------------------------------------- running requests first */
    /* Continuous batching: a request already holding blocks is served before a new one is let in,
     * because the alternative is admitting work that then preempts work already half done. */
    for (size_t i = 0; i < running_.size(); ) {
        if ((int64_t)plan_.e.size() >= cfg_.max_seqs || budget <= 0) break;

        int32_t slot = running_[i];
        SchedReq& s = *reqs_[(size_t)slot];
        Request& r = *s.req;

        const int64_t target = s.seq_len();
        const bool prefill = (r.n_computed < (int64_t)r.prompt.size()) ||
                             (r.n_computed + 1 < target);

        int64_t ckpt = -1;
        int64_t want;
        if (prefill) {
            int64_t pf_budget = budget;
            if (decode_reserved > 0 && budget > decode_reserved) {
                pf_budget = budget - decode_reserved;
            }
            want = plan_chunk(r.n_computed, target, pf_budget, &ckpt);
        } else {
            /* A verify step is 1 + n_draft query rows, which is why attention kernels take a
             * query length rather than assuming one (spec §10). */
            /* THE LAST GATE, and it has to be here rather than only at set_draft, because the
             * n-gram fallback below fills any proposal a drafter left empty -- so refusing a
             * draft upstream and stopping there hands the request a different draft instead of
             * none. Request::may_speculate says why a request can be ineligible. */
            if (!r.may_speculate) s.n_draft = 0;
            else if (s.n_draft == 0) s.n_draft = (int32_t)draft_ngram(s, depth);
            const bool multi_seq = running_.size() > 1;
            const int32_t seq_depth = (int32_t)s.suggest_depth(depth, draft_.params().adaptive, multi_seq, draft_.params().max_depth);
            if (s.n_draft > seq_depth) s.n_draft = seq_depth;
            /* NO VERIFIED POSITION AT OR PAST THE CONTEXT. The step computes positions n_computed
             * .. n_computed + n_draft, and nothing past ctx_limit_ - 1 has a rotary row or a
             * block-table column. commit() finishes a sequence before its length reaches the
             * limit, so there is always room for the one token itself. */
            const int64_t spare = ctx_limit_ - 1 - (int64_t)r.n_computed;
            if (ctx_limit_ > 0 && s.n_draft > spare) s.n_draft = spare > 0 ? (int32_t)spare : 0;
            want = 1 + s.n_draft;
            if (want > budget) want = 0;
            const int64_t dec_demand = 1 + (r.may_speculate ? seq_depth : 0);
            decode_reserved = std::max<int64_t>(0, decode_reserved - dec_demand);
        }
        if (want <= 0) {
            if (prefill && budget > 0) {
                /* Prefill could not take a legal quantum with remaining budget; continue to see
                 * if subsequent decode requests can use the remaining budget. */
                ++i;
                continue;
            }
            break;   /* the budget is spent; the rest of running_ waits a step */
        }

        /* KV first, and preempt from the BACK on refusal -- the lowest priority, latest arrival
         * running request, which is the one that would be re-admitted last anyway. */
        /* BOUNDED, because this loop is the scheduler thread and an unbounded one is a server that
         * stops stepping with nothing in the log to say why. Reclaiming from the cache is tried
         * first (a cached prefix is a saving; a running request is work in flight), but only so
         * many times -- past that the shortfall is not the cache's to fix and preemption is. */
        static const int kMaxEvictRounds = 256;
        int evict_rounds = 0;
        bool failed = false;
        while (kv_->ensure(r.id, r.n_computed + want) < 0) {
            /* THE CACHE GIVES ITS BLOCKS BACK BEFORE A RUNNING REQUEST DOES. Everything the prefix
             * cache holds is a saving on work that has already been done once; everything a
             * running request holds is work in flight. Evicting first is therefore strictly the
             * cheaper order, and it is also what keeps publishing from turning into a leak -- the
             * cache retains a reference on every block it indexes and nothing else ever releases
             * one. Batched rather than one at a time so a large shortfall does not walk the LRU
             * list a hundred times. */
            if (++evict_rounds <= kMaxEvictRounds && pc_ && cfg_.prefix_cache &&
                pc_->evict(256) > 0)
                continue;
            if (!kv_->can_ever_fit(r.n_computed + want) || running_.size() - 1 <= i) {
                /* Nothing left to take. §17: if the smallest running request cannot fit, it is
                 * failed rather than the engine deadlocking on a pool that cannot grow. */
                /* §17: either an empty pool could not serve this length at all, or there is no
                 * lower-priority request left to take blocks from. Failing the request beats
                 * deadlocking on a pool that cannot grow. */
                RAD_WARN("request %llu failed: the KV pool cannot hold %lld tokens%s",
                         (unsigned long long)r.id, (long long)(r.n_computed + want),
                         kv_->can_ever_fit(r.n_computed + want)
                             ? " and nothing lower-priority is left to preempt"
                             : " at any occupancy");
                close_request(s, ReqState::Failed, "kv_exhausted", false, true, RAD_E_FULL,
                              "the KV pool cannot hold it");
                ++m_.failed;
                completed_.push_back(r.id);
                failed = true;
                break;
            }
            preempt_back();
            preempted_here = true;
        }
        if (failed) continue;   /* close_request removed it from running_; i now indexes the next */

        StepEntry e;
        e.req = &r;
        e.seq = r.id;
        e.slot = slot;
        e.card = s.card;
        e.n_tokens = (int32_t)want;
        e.ctx_len = (int32_t)r.n_computed;
        e.n_spec = prefill ? 0 : s.n_draft;
        /* AND THE REQUEST SAYS THE SAME NUMBER. Sampler::build_step sizes its per-row staging from
         * `Request::n_draft` -- one row per sampled position, so 1 + n_draft of them -- while the
         * batch sizes its logits rows from `n_spec`. The two are the same window and must be kept
         * equal by one writer: setting only the scheduler`s copy leaves the sampler staging one
         * row for a step that produces several, and setting only the request`s leaves it staging
         * rows for a window a later clamp to the controller`s depth took away. Set here, once,
         * where the window is decided. */
        r.n_draft = e.n_spec;
        r.draft   = e.n_spec ? s.draft : nullptr;
        /* ONE-BASED, AND A PREFILL ROW IS 1 AND NOT 0. See StepEntry: the value is "how many
         * tokens the previous step committed", the rolling windows are read at `n_accepted - 1`,
         * and zero is not a value any of them can read. A prefill has nothing to roll back, which
         * is offset zero, which is ONE -- the same thing the kernels assume when the operand is
         * absent, and what ple_conv's contract states for exactly this case.
         *
         * IT IS NOT ONLY THE FIRST CHUNK THAT IS AFFECTED. `has_init` is the context length, so
         * every chunk after the first of a chunked prefill is WARM, and a warm window read at -1
         * is one element before the slot: ple_conv takes the previous channel's last sample and
         * ngram_ids hashes a token the slot never held. A 200K prompt is a hundred chunks. */
        e.n_accepted = prefill ? 1 : r.n_accepted + 1;
        e.conv_cursor = s.conv_cursor;
        e.draft = s.draft;
        e.n_draft = s.n_draft;
        e.is_prefill = prefill;
        e.produces_token = (r.n_computed + want >= target);
        /* The snapshot a hit resumes from, consumed once: the state is in the slot for the whole
         * sequence after this, and restoring it again on a later chunk would rewind it. */
        e.ckpt_restore = s.ckpt_src;
        s.ckpt_src = -1;
        if (ckpt > 0) {
            /* The checkpoint is the state AFTER the last token of the chunk, and that token sits
             * at the interval boundary because the split put it there. */
            int32_t cs = -1;
            if (pc_ && pc_->reserve_checkpoint(r, ckpt, &cs) >= 0 && cs >= 0) {
                e.ckpt_tok = (int32_t)(want - 1);
                e.ckpt_slot = cs;
                ++m_.checkpoints_written;
            }
        }
        s.entry = (int32_t)plan_.e.size();
        s.sched_tokens = (int32_t)want;
        plan_.e.push_back(e);
        plan_.n_tok += want;
        if (!prefill && s.n_draft > plan_.n_spec) plan_.n_spec = s.n_draft;
        budget -= want;
        ++i;
    }

    /* ---------------------------------------------------------- admission */
    /* Skipped entirely in a step that preempted: the pool just refused a request that already had
     * blocks, so it will refuse a new one, and the front of the queue is now the request we just
     * took them from. Probing would cost a lookup per step and admit nothing. */
    if (!preempted_here) {
        /* `at` walks the queue and `took_` is what it admitted: a request waiting on a read off
         * disk stays where it is and the one behind it is asked, so the queue is compacted after
         * the walk rather than cut at its front. */
        took_.clear();
        size_t at = 0;
        while (at < waiting_.size() &&
               (int64_t)plan_.e.size() < cfg_.max_seqs && budget > 0) {
            int32_t slot = waiting_[at];
            SchedReq& s = *reqs_[(size_t)slot];
            /* A SEQUENCE THE CONTEXT CANNOT HOLD IS NEVER RUN. Its prompt alone reaches the last
             * position this deployment can address, so there is no room to generate into, and
             * prefilling it would write positions the block tables and rotary tables do not
             * have. The server refuses such a prompt at the door; this is the scheduler's own
             * bound for any caller that did not. close_request takes it out of waiting_, so the
             * same index now names the next request. */
            if (ctx_limit_ > 0 && s.seq_len() >= ctx_limit_) {
                RAD_WARN("request %llu failed: %lld tokens leave no room in a %lld-token context",
                         (unsigned long long)s.req->id, (long long)s.seq_len(),
                         (long long)ctx_limit_);
                close_request(s, ReqState::Failed, "context_length_exceeded", false, true,
                              RAD_E_FULL, "it leaves no room in the context");
                ++m_.failed;
                completed_.push_back(s.req->id);
                continue;
            }
            /* ITS MEDIA FIRST. A request whose items are not encoded keeps its place and the one
             * behind it is asked, exactly as for a context still being read off disk; the engine
             * runs the encoder before the next step (encodes_pending). */
            if (s.req->media && !media_ready(s)) { ++at; continue; }
            const int as = admit_one(s, budget);
            if (as == kAdmitWait) { ++at; continue; }
            if (as < 0) break;                     /* head of line: do not admit behind it */

            Request& r = *s.req;
            StepEntry e;
            e.req = &r;
            e.seq = r.id;
            e.slot = slot;
            e.card = s.card;
            e.n_tokens = s.sched_tokens;
            e.ctx_len = (int32_t)r.n_computed;
            e.n_spec = 0;
            /* The sampler sizes its staging off the Request's own window, so it is said here as
             * well: a request coming back from preemption may still carry a proposal a drafter
             * set before it was taken off the device. */
            r.n_draft = 0;
            r.draft = nullptr;
            e.n_accepted = 1;               /* one-based; a prefill rolls nothing back */
            e.conv_cursor = s.conv_cursor;
            e.draft = s.draft;
            e.n_draft = 0;
            e.is_prefill = true;
            e.produces_token = (r.n_computed + s.sched_tokens >= s.seq_len());
            /* THE ADMISSION PATH IS THE ONE THAT MATTERS for a restore -- admit_one set ckpt_src
             * moments ago and this is the entry it applies to. Consumed once: the state is in the
             * slot for the rest of the sequence, and restoring it again on a later chunk would
             * rewind the recurrence to where the hit started. */
            e.ckpt_restore = s.ckpt_src;
            s.ckpt_src = -1;
            if (s.pending_ckpt > 0) {
                int32_t cs = -1;
                if (pc_ && pc_->reserve_checkpoint(r, s.pending_ckpt, &cs) >= 0 && cs >= 0) {
                    e.ckpt_tok = s.sched_tokens - 1;
                    e.ckpt_slot = cs;
                    ++m_.checkpoints_written;
                }
            }
            s.entry = (int32_t)plan_.e.size();
            plan_.e.push_back(e);
            plan_.n_tok += s.sched_tokens;
            budget -= s.sched_tokens;
            took_.push_back(at);
            ++at;
        }
        if (!took_.empty()) {
            for (size_t i : took_) queue_insert(running_, waiting_[i]);
            size_t w = 0, k = 0;
            for (size_t i = 0; i < waiting_.size(); ++i) {
                if (k < took_.size() && took_[k] == i) { ++k; continue; }
                waiting_[w++] = waiting_[i];
            }
            waiting_.resize(w);
        }
    }

    if (plan_.e.empty()) return nullptr;

    /* THE DECODE ROWS FIRST, and both halves therefore contiguous.
     *
     * A mixed step is prefill-shaped for every sequence in it only because the two kinds of row
     * are interleaved: the running loop walks `running_` in priority order and a request part way
     * through its prompt sits wherever its priority put it. Sorted, each half is a contiguous run
     * of sequences, and a block can issue the recurrent path over one and the chunked path over
     * the other -- cu_seqlens holds absolute row offsets, so a sub-array of it needs no rebasing
     * and no kernel change (RadBatch::n_seq_decode).
     *
     * Stable, so the order within each half is still the order the scheduler chose. `s.entry` is
     * the index into this vector and every entry's was assigned before the sort, so all of them
     * are rewritten here -- a stale entry index is a request reading another request's result.
     *
     * A pure step sorts to itself and pays one pass over at most max_seqs entries. */
    std::stable_partition(plan_.e.begin(), plan_.e.end(),
                          [](const StepEntry& e) { return !e.is_prefill; });
    for (size_t i = 0; i < plan_.e.size(); ++i) {
        StepEntry& e = plan_.e[i];
        if (e.slot >= 0 && (size_t)e.slot < reqs_.size()) reqs_[(size_t)e.slot]->entry = (int32_t)i;
    }

    /* Phase, from what actually got scheduled. The run phase uses it to index bucket tables, so a
     * step with one prefill chunk in it is MIXED and not DECODE -- calling it decode would pick
     * the skinny-GEMM band for a thousand-row GEMM. */
    bool any_pf = false; int64_t n_dec = 0;
    for (const StepEntry& e : plan_.e) { if (e.is_prefill) any_pf = true; else ++n_dec; }
    plan_.phase = any_pf && n_dec ? RAD_PHASE_MIXED
                : any_pf         ? RAD_PHASE_PREFILL
                                 : RAD_PHASE_DECODE;

    /* WHAT THE STEP SHAPE COSTS, AND WHAT SPECULATION DID NOT REACH. `steps_mixed` over `steps`
     * says how often the load produces a step carrying both kinds of row; an agentic load produces
     * them often enough that it is worth letting such a step draft.
     *
     * `decode_rows_unspeculated` is every decode row that took no draft, whatever the reason:
     * the controller at depth 0, a request that may not speculate, a drafter with nothing to
     * propose. It counts ROWS rather than any one reason, so it stays the answer to "how much of
     * my decode is running unspeculated" whatever the current set of reasons is. */
    switch (plan_.phase) {
        case RAD_PHASE_PREFILL: ++m_.steps_prefill; break;
        case RAD_PHASE_MIXED:   ++m_.steps_mixed;   break;
        default:                ++m_.steps_decode;  break;
    }
    for (const StepEntry& e : plan_.e)
        if (!e.is_prefill && e.n_spec == 0) ++m_.decode_rows_unspeculated;
    plan_.n_seq_decode = n_dec;
    plan_.n_tok_decode = 0;
    for (int64_t i = 0; i < n_dec; ++i) plan_.n_tok_decode += plan_.e[(size_t)i].n_tokens;
    /* THE ROWS THE CARD'S ACCEPTANCE READS (advance.h), in the final order: each entry's token row
     * of its first verified position, its first sampled row, and its draft count. Sampled rows
     * go to the entries that produce a token, 1 + n_spec each, which is the order the builder
     * writes out_ids in and the engine reads them back in. */
    {
        int64_t k = 0;
        int32_t row = 0;
        for (StepEntry& e : plan_.e) {
            e.dev_tok = (int32_t)(k + e.n_tokens - 1 - e.n_spec);
            e.dev_out = e.produces_token ? row : -1;
            e.dev_aux = e.n_spec;
            if (e.produces_token) row += 1 + e.n_spec;
            k += e.n_tokens;
        }
    }
    /* DIAGNOSTIC, not a policy: RADIANCE_FORCE_MIXED=1 labels a pure-decode step MIXED, which is
     * the one thing that changes about it -- every sequence still carries one token. It exists to
     * ask whether the ops an architecture block issues for MIXED are correct at decode lengths,
     * without needing a concurrent workload to produce a real mixed step. */
    static const bool force_mixed = [] {
        const char* v = std::getenv("RADIANCE_FORCE_MIXED");
        const bool on = v && *v && *v != '0';
        /* SAID OUT LOUD. A diagnostic whose only evidence is the output it changes cannot be told
         * apart from one that never took -- and "identical output" is exactly the result this
         * flag is used to look for, so a silent no-op reads as a pass. */
        if (on) RAD_WARN("RADIANCE_FORCE_MIXED is set: every pure-decode step is labelled MIXED");
        return on;
    }();
    if (force_mixed && plan_.phase == RAD_PHASE_DECODE) plan_.phase = RAD_PHASE_MIXED;

    step_start_ = std::chrono::steady_clock::now();
    const RadBatch* b = builder_.build(plan_, *kv_, step_);
    if (!b) {
        /* Staging failed. Nothing was computed, so nothing is committed and n_computed did not
         * advance: the next step re-schedules the same chunks and the KV allocations already made
         * cover them. Leaving the step OPEN would deadlock the engine on a commit that will never
         * come, which is worse than losing a step.
         *
         * THE ONE THING THE RETRY CANNOT RE-SCHEDULE IS A RESTORE. An entry admitted off a
         * checkpoint carried the slot to resume from, and that was consumed here; the next step
         * would run the sequence on as if its recurrent state were in place. Such a request goes
         * back to the queue and gives its hit back, so the next admission looks it up again. */
        for (StepEntry& e : plan_.e) {
            SchedReq& s = *reqs_[(size_t)e.slot];
            s.entry = -1;
            if (e.ckpt_restore < 0 || !s.req) continue;
            undo_hit(s);
            s.running = false;
            s.req->state = ReqState::Waiting;
            queue_erase(running_, s.slot);
            queue_insert(waiting_, s.slot);
        }
        plan_.reset();
        step_open_ = false;
        return nullptr;
    }
    ++step_;
    step_open_ = true;
    return b;
}

/* ------------------------------------------------------------------ media */
bool Scheduler::media_ready(SchedReq& s) {
    bool ready = true;
    for (mm::Placed& p : s.req->media->items) {
        mm::Item& it = *p.item;
        if (it.encoded) continue;
        ready = false;
        if (!it.queued) {
            it.queued = true;
            enc_want_.push_back(p.item);
        }
    }
    return ready;
}

bool Scheduler::encodes_pending() const {
    std::lock_guard<std::mutex> lk(mu_);
    return !enc_want_.empty();
}

void Scheduler::take_encodes(std::vector<std::shared_ptr<mm::Item>>* out) {
    std::lock_guard<std::mutex> lk(mu_);
    out->clear();
    out->swap(enc_want_);
}

/* ------------------------------------------------------------------ the next step */
void Scheduler::expect_draft(uint64_t id, int n) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return;
    reqs_[(size_t)it->second]->draft_next = n > 0 ? n : 0;
}

const RadBatch* Scheduler::step_next() {
    std::lock_guard<std::mutex> lk(mu_);
    const RadBatch* b = step_next_locked();
    if (!b)
        for (const StepEntry& e : plan_.e)
            if (e.slot >= 0 && (size_t)e.slot < reqs_.size() && reqs_[(size_t)e.slot])
                reqs_[(size_t)e.slot]->draft_next = 0;
    return b;
}

const RadBatch* Scheduler::step_next_locked() {
    if (!step_open_ || next_open_ || plan_.e.empty()) return nullptr;
    /* An admission is a decision about the pool and the queue that the full step makes.
     * When capacity is already full (plan_.e.size() >= cfg_.max_seqs), no waiting request
     * can be admitted on the next step anyway, so device lookahead is safe to proceed. */
    if (!waiting_.empty() && (int64_t)plan_.e.size() < cfg_.max_seqs) return nullptr;

    plan_next_.reset();
    std::vector<std::pair<uint64_t, int64_t>> extended_kv;
    auto rollback_extended = [&]() {
        for (const auto& item : extended_kv) {
            kv_->rollback(item.first, item.second);
        }
        extended_kv.clear();
        plan_next_.reset();
    };

    for (const StepEntry& e : plan_.e) {
        if (e.slot < 0 || (size_t)e.slot >= reqs_.size()) { rollback_extended(); return nullptr; }
        SchedReq& s = *reqs_[(size_t)e.slot];
        if (!s.req || s.req->state != ReqState::Running || !s.running) { rollback_extended(); return nullptr; }
        Request& r = *s.req;
        /* A chunk in the middle of a prompt continues as a prefill, from where the host knows it
         * is. And a grammar that cannot rewind decides, when this step commits, whether the next
         * may speculate; one that can -- and a request with no grammar -- answers as it did. */
        if (!e.produces_token || !r.speculation_fixed) { rollback_extended(); return nullptr; }

        const bool surely_terminal = (r.max_tokens > 0) &&
                                     ((int64_t)r.max_tokens - (int64_t)r.output.size() <= 1) &&
                                     e.produces_token;
        if (surely_terminal) {
            s.draft_next = 0;
            s.entry_next = -1;
            continue;
        }

        /* THE FURTHEST CONTEXT THE ROW CAN REACH: every position this step verifies committed,
         * counted from where the host knows the sequence is -- the last commit -- and not from
         * this step's own bound, which would add a window of slack every step it is issued
         * ahead. The card's acceptance says where it really is; this is what the pool, the
         * block tables and every host-side ceiling are sized for. */
        const int64_t ctx_up = r.n_computed + e.n_tokens;
        const int32_t d      = s.draft_next;
        const int64_t want   = 1 + d;
        if (ctx_limit_ > 0 && ctx_up + want > ctx_limit_) { rollback_extended(); return nullptr; }
        if (kv_->ensure(r.id, ctx_up + want) < 0) { rollback_extended(); return nullptr; }
        extended_kv.push_back({r.id, ctx_up});

        StepEntry n;
        n.req            = &r;
        n.seq            = r.id;
        n.slot           = e.slot;
        n.card           = e.card;
        n.n_tokens       = (int32_t)want;
        n.ctx_len        = (int32_t)ctx_up;
        n.n_spec         = d;
        n.n_accepted     = 1;          /* the card's: StepSlot::n_acc */
        n.conv_cursor    = s.conv_cursor;
        n.draft          = s.draft;    /* published before this step commits (set_draft) */
        n.n_draft        = d;
        n.is_prefill     = false;
        n.produces_token = true;
        n.dev            = true;
        plan_next_.e.push_back(n);
        plan_next_.n_tok += want;
        if (d > plan_next_.n_spec) plan_next_.n_spec = d;
    }
    if (plan_next_.e.empty()) { rollback_extended(); return nullptr; }
    plan_next_.phase        = RAD_PHASE_DECODE;
    plan_next_.n_seq_decode = (int64_t)plan_next_.e.size();
    plan_next_.n_tok_decode = plan_next_.n_tok;
    {
        int64_t k = 0;
        int32_t row = 0;
        for (StepEntry& e : plan_next_.e) {
            e.dev_tok = (int32_t)(k + e.n_tokens - 1 - e.n_spec);
            e.dev_out = row;
            e.dev_aux = e.n_spec;
            row += 1 + e.n_spec;
            k += e.n_tokens;
        }
    }
    const RadBatch* b = builder_.build(plan_next_, *kv_, step_);
    if (!b) { rollback_extended(); return nullptr; }
    for (size_t i = 0; i < plan_next_.e.size(); ++i)
        reqs_[(size_t)plan_next_.e[i].slot]->entry_next = (int32_t)i;
    next_open_ = true;
    return b;
}

void Scheduler::promote_next() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!next_open_) return;
    std::swap(plan_, plan_next_);
    plan_next_.reset();
    step_prefill_tok_ = 0;
    step_decode_tok_  = 0;
    for (size_t i = 0; i < plan_.e.size(); ++i) {
        SchedReq& s = *reqs_[(size_t)plan_.e[i].slot];
        s.entry      = (int32_t)i;
        s.entry_next = -1;
        s.draft_next = 0;
        s.sched_tokens = plan_.e[i].n_tokens;
        if (s.req) { s.req->n_draft = plan_.e[i].n_spec; s.req->draft = s.draft; }
    }
    ++m_.steps_decode;
    ++m_.steps_ahead;
    for (const StepEntry& e : plan_.e)
        if (e.n_spec == 0) ++m_.decode_rows_unspeculated;
    step_start_ = std::chrono::steady_clock::now();
    ++step_;
    step_open_ = true;
    next_open_ = false;
}

/* ------------------------------------------------------------------ commit */
int Scheduler::commit(const StepResult& res) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!step_open_) return RAD_E_STATE;
    if (res.n_seq != (int)plan_.e.size()) {
        RAD_ERR("commit of %d sequences against a step of %d", res.n_seq, (int)plan_.e.size());
        return RAD_E_INVAL;
    }
    step_open_ = false;

    const int32_t W = conv_window();

    /* A MIXED STEP DOES NOT RUN THE CHUNKED PATH OVER A SEQUENCE THAT WAS ONLY DECODING, so the
     * scratch beside its state is not invalidated here.
     *
     * The blocks issue the recurrent path over the decode rows of a mixed step and the chunked
     * path over the prefill ones (RadBatch::n_seq_decode), so those rows leave exactly the replay
     * factors a decode step leaves, and discarding them would throw away the rollback the next
     * verify needs. Invalidation would only be correct if a mixed step were prefill-shaped for
     * everyone in it: the chunked path commits the state directly and leaves no replay factors,
     * so anything in the scratch would then describe tokens the state already had.
     *
     * The one thing that must stay true for that is the ordering: the recurrent path is issued
     * over sequences [0, n_seq_decode) and nothing else, so a decode row that sorted into the
     * prefill half would silently take the chunked path and leave stale factors behind it. That
     * is what the sort in step() and its test are for. */
    for (int i = 0; i < res.n_seq; ++i) {
        const StepEntry& e = plan_.e[(size_t)i];
        SchedReq& s = *reqs_[(size_t)e.slot];
        Request& r = *s.req;
        if (r.state != ReqState::Running) {
            /* Cancelled between step and commit. Its blocks were held until the device finished
             * with them; that is now. Its SLOT may also be owed back, if the server reaped it in
             * the same window -- the sweep at the end of this function does that, once every
             * entry has been retired. */
            s.entry = -1;
            /* Unless the drafter's snapshot still names it, or the next step carries it: the
             * sweep below, or that step's commit, releases it once nothing does. */
            if (s.pending_free && !block_holds(r.id) && !(next_open_ && s.entry_next >= 0)) {
                free_kv(s);
                s.pending_free = false;
                --n_deferred_;
            }
            continue;
        }

        /* `ignore_eos` drops the token's termination and nothing else: max_tokens and the stop
         * strings below still end the request. See Request::ignore_eos. */
        bool eos = res.eos && res.eos[i] && !r.ignore_eos;

        bool overflowed = false;

        /* THE SNAPSHOT THE ENGINE JUST TOOK IS NOW READABLE. Marking it here rather than at
         * reservation is the difference between a checkpoint and a promise: reserve_checkpoint
         * runs when the chunk is PLANNED, and a lookup between then and now would adopt a slot
         * still holding the previous tenant's state. Stream ordering does the rest -- a later
         * restore is enqueued on the same stream behind this save, so validity does not have to
         * wait on the copy landing, only on it being issued. */
        if (e.ckpt_slot >= 0 && e.ckpt_tok >= 0 && pc_)
            pc_->commit_checkpoint(r, (int64_t)e.ctx_len + e.ckpt_tok + 1);

        if (e.is_prefill) {
            r.n_computed += e.n_tokens;
            step_prefill_tok_ += e.n_tokens;
            /* Once, when the prompt is fully placed: the admission accounting the usage block
             * reports. `n_cached` is what the prefix cache supplied and is not recomputed here. */
            if (r.n_computed >= (int64_t)r.prompt.size())
                server::sink_prompt_stats(r.sink, (int32_t)r.prompt.size(), r.n_cached);
            if (e.produces_token && res.token) {
                if (!s.first_tok_us) s.first_tok_us = mono_us();
                r.output.push_back(res.token[i]);
                overflowed = !server::sink_push(r.sink, res.token[i]);
                /* The conv window absorbed this chunk; the cursor advances by one committed
                 * position, exactly as it does in decode. */
                s.conv_cursor = (int32_t)((s.conv_cursor + 1) % (W ? W : 1));
            }
        } else {
            int32_t a = res.n_accepted ? res.n_accepted[i] : 0;
            if (a < 0) a = 0;
            if (a > e.n_spec) a = e.n_spec;
            /* A ROW ISSUED AHEAD was staged at the furthest context it could reach; it really
             * continued from the committed end, which is where the host's sequence is. */
            const int64_t ctx0 = e.dev ? r.n_computed : (int64_t)e.ctx_len;

            /* PAGED KV rolls back: free the blocks past the accepted position, because the KV
             * written for the rejected tokens is garbage a later step would attend to.
             *
             * LINEAR STATE DOES NOT. The recurrent state and the conv window have already absorbed
             * the rejected tokens and there is nothing to undo: libr4d's conv update treats the
             * state cache as a rolling window of width-1 + num_spec entries and reads at the slot
             * the last accepted token left, so a rejection is a change of READ OFFSET rather than
             * a recompute (spec §10). The two kinds roll back differently and conflating them is
             * how you get a model that is subtly wrong only after a rejection. */
            int64_t valid = ctx0 + 1 + a;
            /* NOT WHILE THE NEXT STEP IS WRITING PAST IT. A step already issued behind this one
             * continues the sequence from `valid` and writes its keys and values there and on;
             * its own commit rolls back to the end it reaches. */
            if (a < e.n_spec && !(next_open_ && s.entry_next >= 0)) {
                int st = kv_->rollback(r.id, valid);
                if (st < 0) RAD_WARN("kv rollback for request %llu: %s",
                                     (unsigned long long)r.id, rad_strerror(st));
            }
            s.conv_cursor = (int32_t)((s.conv_cursor + 1 + a) % (W ? W : 1));
            r.n_accepted = a;

            /* The accepted drafts FIRST and then the verified token, in generation order: the
             * sink is a stream and a reordering here is a reordering in the client's text.
             *
             * AND THE BUDGET IS CHECKED PER TOKEN, because a verify step emits 1 + a of them at
             * once. Pushing them all and trimming `output` afterwards leaves the SINK holding the
             * overshoot -- the client has already been streamed tokens it did not ask for, so the
             * same prompt at the same max_tokens comes back longer with speculation than without.
             * The same tokens in the same order, just more of them, which is the one way greedy
             * speculation is allowed to differ and should not. */
            int64_t room = r.max_tokens > 0
                         ? r.max_tokens - (int64_t)r.output.size() : (int64_t)1 << 62;
            /* AND THE CONTEXT BOUNDS IT WHATEVER max_tokens SAYS, including "no limit": a
             * sequence may not grow past the positions this deployment can address. */
            if (ctx_limit_ > 0 && ctx_limit_ - s.seq_len() < room) room = ctx_limit_ - s.seq_len();
            int32_t emit_buf[RAD_SCHED_MAX_SPEC + 2];
            int n_emit = 0;
            auto emit = [&](int32_t t) {
                if (room <= 0) return;
                --room;
                if (!s.first_tok_us) s.first_tok_us = mono_us();
                r.output.push_back(t);
                emit_buf[n_emit++] = t;
            };
            for (int32_t j = 0; j < a; ++j) emit(s.draft[j]);
            if (res.token) emit(res.token[i]);
            if (n_emit > 0) {
                if (!server::sink_push_span(r.sink, emit_buf, (size_t)n_emit))
                    overflowed = true;
            }
            r.n_computed = valid;
            step_decode_tok_ += 1 + a;

            if (e.n_spec > 0) {
                int n_spec_seqs = 0;
                for (int si = 0; si < res.n_seq; ++si) {
                    if (plan_.e[(size_t)si].n_spec > 0) n_spec_seqs++;
                }
                int64_t step_elapsed_us = 0;
                if (step_start_.time_since_epoch().count() != 0) {
                    step_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - step_start_).count();
                }
                draft_.observe(e.n_spec, a, n_spec_seqs);
                s.observe_draft(e.n_spec, a, step_elapsed_us);
                m_.draft_tokens += e.n_spec;
                m_.draft_accepted += a;
                server::sink_drafts(r.sink, e.n_spec, a);
            }
        }
        s.n_draft = 0;
        r.n_draft = 0;
        s.entry = -1;

        /* A REQUEST THE NEXT STEP CARRIES FINISHES NOW AND KEEPS ITS BLOCKS: that step is writing
         * them. Its commit finds the request no longer running and gives them back. */
        const bool carried = next_open_ && s.entry_next >= 0;

        /* Finish. The emit above already stops at max_tokens, so `output` cannot run past it --
         * the resize below is the belt to that braces, kept because a >= test that can only ever
         * be == is the kind of thing a later change makes wrong quietly. */
        if (overflowed) {
            /* The sink has already failed the request -- a consumer that far behind is gone --
             * so all that is left is to stop generating for it and give its blocks back. */
            close_request(s, ReqState::Failed, "sink_overflow", carried);
            ++m_.failed;
            completed_.push_back(r.id);
        } else if (server::sink_cancelled(r.sink)) {
            /* The client hung up. Spec §14: the blocks go back now rather than at the end of a
             * generation nobody will read. */
            close_request(s, ReqState::Cancelled, "cancelled", carried);
            ++m_.cancelled;
            completed_.push_back(r.id);
        } else if (r.max_tokens > 0 && (int64_t)r.output.size() >= r.max_tokens) {
            r.output.resize((size_t)r.max_tokens);
            close_request(s, ReqState::Finished, "length", carried);
            ++m_.finished;
            completed_.push_back(r.id);
        } else if (ctx_limit_ > 0 && s.seq_len() >= ctx_limit_) {
            /* The context is full. The same finish as max_tokens, because to the caller it is
             * the same thing: the generation ran out of room. */
            close_request(s, ReqState::Finished, "length", carried);
            ++m_.finished;
            completed_.push_back(r.id);
        } else if (eos) {
            close_request(s, ReqState::Finished, "stop", carried);
            ++m_.finished;
            completed_.push_back(r.id);
        }
    }

    m_.prefill_tokens += step_prefill_tok_;
    m_.decode_tokens += step_decode_tok_;
    ++m_.steps;

    /* Every entry is retired now, so anything whose reap or free was deferred on THIS step is
     * releasable, and so is anything deferred on the last drafter snapshot, which block_end() or
     * serial_end() has cleared since. Every pass that named them was issued before this step's
     * readback, which waited for the compute stream. */
    sweep_deferred();
    return RAD_OK;
}

/* ------------------------------------------------------------------ the end of a step */
/* THE STEP IS NOT OVER AT COMMIT, and timing it there is wrong by the whole draft pass.
 *
 * The drafter for the NEXT step runs after commit, because that is the first moment both of its
 * inputs exist -- the token the sampler chose, and which drafted position it came from. Its rounds
 * are real forward passes through the draft head. Timed at commit they fall between that commit
 * and the next step(), a gap the engine loop charges to nothing at all: the reported step time
 * comes out short by the draft pass and `decode_tps` -- which divides by the same interval --
 * comes out correspondingly high, which is the direction of error nobody reports.
 *
 * So the engine says when the iteration is genuinely finished and the accounting happens here.
 * Separate from commit() rather than folded into it because they answer different questions:
 * commit() retires a step's tokens, and this closes the engine's loop around it. A caller that
 * only drives commit() -- the scheduler tests -- gets correct counters and no timing, which is
 * the honest answer for a step that never ran on a device.
 *
 * THE RATE IS TOKENS OVER WALL TIME ACROSS A WINDOW, NOT AN AVERAGE OF STEP RATES. A step's
 * interval is host time, and the host does not wait on the card at every step: a prefill chunk
 * that samples nothing is committed while the card is still on it, and a later step's wait pays
 * for it. Averaged as rates, those short intervals dominate -- a 2048-token chunk committed 70 ms
 * after it began reads 30K tok/s -- while the step that absorbed the wait counts once, and the
 * same skew reads decode high. Summed, all of the time is there: each kind's tokens and the wall
 * time since the last step ended -- which also covers a step issued ahead, whose start precedes
 * its predecessor's end, without counting that overlap twice -- or since this step began, when
 * the engine sat idle in between; all three decayed with one time constant.
 *
 * A STEP WITH NO TOKENS OF A KIND STILL ADDS ITS TIME. Skipping it instead would freeze the rate
 * at whatever it last was, so a server that prefilled once long ago would report that prefill
 * throughput for the rest of its life, and a busy engine can read as an idle one.
 *
 * Not a device measurement: enabling per-op timing restores synchronisations that change the
 * mover's slot supply, so a profiled run is not comparable anyway (spec §16). This is the number
 * an operator can always have. */
void Scheduler::step_finished() {
    std::lock_guard<std::mutex> lk(mu_);
    if (step_start_.time_since_epoch().count() == 0) return;   /* no step has run */

    const auto step_end = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(step_end - step_start_).count();
    m_.step_ns += ns;
    if (plan_.phase == RAD_PHASE_DECODE) m_.step_ns_decode += ns;

    const auto   from = step_start_ - last_step_at_ < std::chrono::milliseconds(500)
                            ? last_step_at_ : step_start_;
    const double wall = std::chrono::duration<double>(step_end - from).count();
    /* The window ages by the time that PASSED, idle included, so a phase that ended -- a prefill
     * burst before a decode run -- leaves it at the window's own rate; only the busy part is
     * added as time the tokens took. */
    const double aged = std::chrono::duration<double>(step_end - last_step_at_).count();
    if (wall > 0) {
        constexpr double kWindowS = 1.0;
        const double f = std::exp(-aged / kWindowS);
        tps_win_prefill_ = tps_win_prefill_ * f + (double)step_prefill_tok_;
        tps_win_decode_  = tps_win_decode_  * f + (double)step_decode_tok_;
        tps_win_time_    = tps_win_time_    * f + wall;
        /* AT LEAST HALF A SECOND OF TIME UNDER THE RATE. After an idle spell the window starts
         * over, and a burst's first chunk is committed as soon as it is issued: 2048 tokens over
         * those few milliseconds is exactly the spike this replaces. Held to half a second the
         * first reads are low instead, and the steady window (about a second) is not touched. */
        const double t = std::max(tps_win_time_, 0.5);
        m_.prefill_tps = (float)(tps_win_prefill_ / t);
        m_.decode_tps  = (float)(tps_win_decode_  / t);
        /* An exponential decay approaches zero and never arrives, and 7e-14 tok/s is a number
         * nobody wants to read. Below a twentieth of a token a second it IS zero. */
        if (m_.prefill_tps < 0.05f) m_.prefill_tps = 0.0f;
        if (m_.decode_tps  < 0.05f) m_.decode_tps  = 0.0f;
    }
    last_step_at_ = step_end;
    /* So a second call, or one on a loop iteration that ran no step, charges nothing. */
    step_start_ = {};
}

/* ------------------------------------------------------------------ the serial drafter */
/* Which sequences may take a serial draft after the step just committed, and what each of them
 * needs. Called while plan() still describes that step, because everything here is a coordinate
 * INTO it: the head reads the trunk's hidden states out of the buffer the step just filled.
 *
 * THE HEAD'S INDEXING, which StepPlan::tok_shift carries: head index i reads hidden(i) and the
 * token at i + 1, writes its K and V at KV position i, and predicts i + 2. So after a step that
 * committed `n_computed` positions, the newest index the head can serve is n_computed - 1 --
 * its hidden state is the last row the trunk produced and the token one on is the one the
 * sampler just chose. That is draft round 1, and each round after it steps one position further out.
 */
int Scheduler::serial_begin(int* rounds, SerialRow* out, int max) {
    std::lock_guard<std::mutex> lk(mu_);
    serial_.clear();
    if (!rounds || *rounds <= 0 || !out || max <= 0) return 0;

    /* MAY ANYTHING DRAFT AT ALL? The controller decides depth from measured acceptance, and 0 means
     * it has turned drafting off. That is the whole test: a step carrying a prefill chunk is not
     * itself a reason to refuse, because its decode rows still take the recurrent path.
     *
     * Neither answer stops the HISTORY pass. The head's own K and V have to be filled whether or
     * not this particular step drafts -- a sequence three chunks into a long prompt is exactly the
     * case where nothing can draft yet and the history is exactly what will be needed. */
    int n_decode_seqs = 0;
    int max_req_depth = 0;
    const bool is_multi_seq = (running_.size() > 1);
    for (size_t i = 0; i < plan_.e.size(); ++i) {
        if (plan_.e[i].produces_token) {
            n_decode_seqs++;
            int32_t slot = plan_.e[i].slot;
            if (slot >= 0 && (size_t)slot < reqs_.size() && reqs_[(size_t)slot]) {
                int d = reqs_[(size_t)slot]->suggest_depth(draft_.params().max_depth,
                                                           draft_.params().adaptive,
                                                           is_multi_seq,
                                                           draft_.params().max_depth);
                if (d > max_req_depth) max_req_depth = d;
            }
        }
    }
    int depth = draft_.params().adaptive ? max_req_depth : draft_.next_depth(std::max(n_decode_seqs, (int)running_.size()));
    bool may_draft = depth > 0;
    if (depth > 0 && *rounds > depth) *rounds = depth;
    if (*rounds > RAD_SCHED_MAX_SPEC) *rounds = RAD_SCHED_MAX_SPEC;

    int n_draft_rows = 0;
    int64_t cu = 0;                     /* this entry's first token in the step just run */
    for (size_t i = 0; i < plan_.e.size(); ++i) {
        const StepEntry& e = plan_.e[i];
        const int64_t base = cu;
        cu += e.n_tokens;

        if ((int)serial_.size() >= max) continue;
        if (e.slot < 0 || (size_t)e.slot >= reqs_.size()) continue;
        SchedReq& s = *reqs_[(size_t)e.slot];
        if (!s.req || s.req->state != ReqState::Running || !s.running) continue;
        Request& r = *s.req;

        /* THE VERIFIED POSITIONS: the last 1 + n_spec of the entry. Round 1 continues from the
         * last committed one, which is the first of them plus however many drafts were accepted
         * -- a number the card has and the host does not yet, so the host bounds it by all of
         * them. A chunk in the middle of a prompt verified nothing and continues nothing. */
        const int64_t vstart = (int64_t)e.ctx_len + e.n_tokens - 1 - e.n_spec;
        const int64_t head_max = vstart + e.n_spec;
        const bool surely_terminal = (r.max_tokens > 0) &&
                                     ((int64_t)r.max_tokens - (int64_t)r.output.size() <= 1) &&
                                     e.produces_token;
        bool can_draft = may_draft && r.may_speculate && e.produces_token && !surely_terminal;

        /* THE LOOKAHEAD KV. Round k writes the head's K and V at position head + k - 1, which runs
         * past the committed end, so the blocks have to exist before the round does -- for the
         * furthest head acceptance could leave. They are the same blocks the next verify step
         * writes the trunk's own K and V into, so this reservation is one the next step would
         * have made anyway.
         *
         * A ROW THAT CANNOT HAVE THEM STILL OWES ITS HISTORY. The history pass writes positions
         * the sequence already holds blocks for, so a refused lookahead is a reason not to draft
         * and not a reason to skip the row: skipping it would leave the positions this step added
         * unwritten in the head's layer for good. The same goes for a row too near the end of the
         * context for its rounds, whose last positions do not exist. */
        if (can_draft && ctx_limit_ > 0 && head_max + *rounds > ctx_limit_) can_draft = false;
        if (can_draft && kv_->ensure(r.id, head_max + *rounds) < 0) can_draft = false;

        /* THE HISTORY THIS STEP OWES, in two parts. Before the verified positions, the positions
         * the step computed whose next token the host holds -- a prefill chunk's prompt -- from
         * the watermark on, clamped to the rows this step produced: a prefix-cache hit adopts
         * blocks whose head layer somebody else wrote, and there are no hidden states here to
         * redo those from. Then the verified positions themselves, all 1 + n_spec of them, whose
         * next tokens are the sampler's picks and exist only on the card.
         *
         * ALL OF THEM, AND THE LAST ROW IS DRAFT ROUND 1. The card fills the rows past the last
         * committed position with that position again (advance.h, ADV_HIST), so the final row of
         * the run always holds exactly round 1's inputs -- the committed position, the trunk's
         * hidden state there and the token it chose after it -- and a drafting row takes its
         * round-1 proposal from it (RadBatch::draft_out_ids) instead of from a pass of its own.
         *
         * THE VERIFIED PART HAS A FIXED SHAPE. The repeats write the committed position's key and
         * value again, identical to the first; nothing past it is written, so no query can reach
         * a rejected draft's state. */
        const int64_t owed_end = e.produces_token ? vstart : (int64_t)e.ctx_len + e.n_tokens;
        int64_t hs = s.draft_filled > e.ctx_len ? s.draft_filled : (int64_t)e.ctx_len;
        if (hs > owed_end) hs = owed_end;
        const int64_t lead = owed_end - hs;
        const int64_t verified = e.produces_token ? e.n_spec + 1 : 0;
        const int64_t hist_n = lead + verified;

        if (!can_draft && hist_n <= 0) continue;   /* nothing owed and nothing to draft */

        SerialSeq m{};
        m.seq        = r.id;
        m.slot       = e.slot;
        m.card       = e.card;
        m.hist_row   = (int32_t)(base + (hs - e.ctx_len));
        m.hist_n     = (int32_t)hist_n;
        m.hist_pos   = hs;
        m.dev        = e.produces_token;
        m.dev_tok    = (int32_t)(base + (vstart - e.ctx_len));
        m.dev_out    = e.dev_out;
        m.lead       = (int32_t)lead;
        m.base_pos   = head_max;
        m.draft_idx  = can_draft ? n_draft_rows : -1;
        serial_.push_back(m);

        if (can_draft) {
            out[n_draft_rows].seq  = m.seq;
            out[n_draft_rows].slot = m.slot;
            out[n_draft_rows].card = m.card;
            ++n_draft_rows;
        }
    }
    return n_draft_rows;
}

const RadBatch* Scheduler::serial_batch(int round, int step) {
    /* RADIANCE_DEBUG_STAGE, below. Read before the lock so the timer covers the wait for it. */
    static const bool stg = std::getenv("RADIANCE_DEBUG_STAGE") != nullptr;
    const auto t_stage0 = stg ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
    std::lock_guard<std::mutex> lk(mu_);
    if (round < 0 || serial_.empty()) return nullptr;
    /* Round 1 has no pass of its own: the history pass carries it (RadBatch::draft_out_ids). */
    if (round == 1) return nullptr;

    serial_plan_.reset();
    serial_plan_.phase      = RAD_PHASE_DECODE;
    /* Negative is the history pass; positive is draft round r. The sign is what the plugin reads
     * to decide whether the pass ends in the lm_head (rad_runtime.h). */
    serial_plan_.draft_pass = round == 0 ? -1 : round;
    serial_plan_.tok_shift  = 1;           /* the head's own indexing; see StepPlan::tok_shift */
    serial_plan_.e.reserve(serial_.size());

    /* THE DRAFTING SEQUENCES FIRST, in draft order, then the ones that only owe history. Every
     * round's rows are the drafting sequences in that order, and at depth 1 the history pass is a
     * group's last pass -- so its first rows are where the advance that takes the last proposal
     * (ADV_TAKE) reads them, whichever pass that is. */
    for (int part = 0; part < 2; ++part)
    for (size_t i = 0; i < serial_.size(); ++i) {
        const SerialSeq& m = serial_[i];
        if ((m.draft_idx >= 0) != (part == 0)) continue;
        if (round == 0 && m.hist_n <= 0) continue;      /* nothing owed for this sequence */
        if (round > 0 && m.draft_idx < 0) continue;     /* history only: no token to propose */
        const Request* rq = serial_req(m);
        /* Cannot happen while reap() defers on block_holds(), and this is what says so: a whole
         * round skipped is a lost draft, which costs acceptance for one step and nothing else. */
        if (!rq) return nullptr;

        StepEntry e;
        e.req      = rq;
        e.seq      = m.seq;
        e.slot     = m.slot;
        e.card     = m.card;
        e.n_tokens = round == 0 ? m.hist_n : 1;
        /* A round's position is the card's; the host stages the furthest it can be, which is what
         * every host-side ceiling reads. The history's are known here exactly. */
        e.ctx_len  = (int32_t)(round == 0 ? m.hist_pos : m.base_pos + (round - 1));
        e.n_spec   = 0;
        /* One-based and constant: a head pass commits exactly the tokens it was handed. The head
         * issues no linear-state op, so nothing reads it; it is set rather than left at a value
         * the kernels cannot take. */
        e.n_accepted  = 1;
        e.conv_cursor = 0;
        /* NO DRAFT ARRAY. A history row's tokens past its lead, and every round's token, are the
         * card's (advance.h); token_at() stages a 0 in their place, and the positions before the
         * lead are the prompt, which it stages exactly. */
        e.draft   = nullptr;
        e.n_draft = 0;
        e.is_prefill     = false;
        e.produces_token = round > 0;
        /* The history pass gathers the trunk's hidden states, a whole run of them. Later rounds
         * read the head's own output, which it left in the same buffer. */
        e.hidden_row = round == 0 ? m.hist_row : -1;
        e.draft_out  = round == 0 && m.draft_idx >= 0;
        e.dev      = m.dev;
        e.dev_tok  = m.dev_tok;
        e.dev_out  = m.dev_out;
        e.dev_aux  = round == 0 ? m.lead : 0;
        serial_plan_.e.push_back(e);
        serial_plan_.n_tok += e.n_tokens;
    }
    if (serial_plan_.e.empty()) return nullptr;

    /* RADIANCE_DEBUG_STAGE: THE BATCH HALF, AND IT IS NOT THE COST. Splits a draft pass into the
     * plan loop above and `builder_.build`, which stages the block tables and slots, and reports
     * the average of each. Both are small fractions of what a pass spends ISSUING it
     * (engine_draft.cpp has the five-way split): staging a draft batch is effectively free, and
     * the block-table rectangle it copies -- sized for the worst step, not this one -- is not on
     * any critical path worth shrinking.
     *
     * Off by default and it costs two clock reads when it is on, so it can stay. */
    if (!stg) return builder_.build(serial_plan_, *kv_, step);

    static double acc_plan = 0.0, acc_build = 0.0;
    static long   n_calls = 0;
    const auto t1 = std::chrono::steady_clock::now();
    const RadBatch* b = builder_.build(serial_plan_, *kv_, step);
    const auto t2 = std::chrono::steady_clock::now();
    acc_plan  += std::chrono::duration<double, std::micro>(t1 - t_stage0).count();
    acc_build += std::chrono::duration<double, std::micro>(t2 - t1).count();
    if (++n_calls % 400 == 0)
        std::fprintf(stderr, "D stage: %ld draft batches, plan %.1f us, build %.1f us a call\n",
                     n_calls, acc_plan / (double)n_calls, acc_build / (double)n_calls);
    return b;
}

/* The watermark moves to what actually ran, and it is read off the COMMITTED sequence: the history
 * pass wrote the head's K and V at the last committed index, n_computed - 1, from the trunk's own
 * hidden state and the committed token, so after it the head's history is exact up to
 * n_computed. The rounds past it write SPECULATIVELY -- correct only for the drafts that turn out
 * to be accepted -- and are deliberately not counted: redoing them next step costs one head pass
 * over a couple of tokens and removes the whole question. A row that only owed history is exact
 * up to where the history reached; that is n_computed too when it covered the verified positions,
 * and its host-known end when it did not. */
void Scheduler::serial_end(int rounds_done) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const SerialSeq& m : serial_) {
        if (m.slot < 0 || (size_t)m.slot >= reqs_.size()) continue;
        SchedReq& s = *reqs_[(size_t)m.slot];
        if (!s.req || s.req->id != m.seq) continue;
        const int64_t filled = m.dev ? s.req->n_computed
                                     : m.hist_pos + m.hist_n;
        const bool counts = m.dev ? (rounds_done >= 1 || m.draft_idx < 0) : true;
        if (counts && filled > s.draft_filled) s.draft_filled = filled;
    }
    /* THE SNAPSHOT ENDS HERE. Every pass that names these rows has been built, so nothing reads
     * them again, and holding them any longer would keep deferring the release of a request that
     * was cancelled or reaped while the rounds ran. */
    serial_.clear();
}

/* ------------------------------------------------------------------- the block drafter */
/* Which sequences the drafter runs over after the step just committed, and which of those may also
 * propose. Called while plan() still describes that step, because the context pass MIRRORS it.
 *
 * The mirror is exact, including a verify step's rejected suffix. That is not sloppiness, it is
 * what keeps the pass free of coordinates: the drafter reads the trunk's tapped hidden states
 * positionally out of a buffer of its own, so batch row i must be buffer row i, and trimming
 * one entry would slide every entry after it. The rejected positions are all inside [n_computed,
 * n_computed + block), which the query pass overwrites before anything reads it -- and the one
 * ensure() below covers them, because n_computed + block is at least the trunk step's own end.
 */
int Scheduler::block_begin(int block, int depth, int window, int32_t mask, BlockRow* out, int max) {
    std::lock_guard<std::mutex> lk(mu_);
    block_.clear();
    block_len_ = 0;
    block_depth_ = 0;
    if (block < 2 || depth <= 0 || depth > block || !out || max <= 0) return 0;
    block_len_   = (int32_t)block;
    block_depth_ = (int32_t)depth;
    block_mask_.assign((size_t)(block - 1), mask);

    /* NOT the drafter's declared `depth` above -- this is how many tokens the draft POLICY wants
     * out of this step, which is zero when it has backed off. The two were both called `depth`. */
    const bool may_draft = draft_.next_depth() > 0;
    /* THE QUERY PASS IS `block` TOKENS A DRAFTING SEQUENCE, and the builder's staging arrays are
     * sized at max_tok. A decode step is one token a sequence, so nothing before this could ever
     * put more rows in a batch than sequences -- this pass can, and eight rows a sequence against
     * a small token budget is exactly the shape that would run off the end of h_tok. */
    if ((int64_t)max * block > cfg_.max_tok) max = (int)(cfg_.max_tok / block);
    if (max <= 0) return 0;

    int n_draft_rows = 0;
    for (size_t i = 0; i < plan_.e.size(); ++i) {
        const StepEntry& e = plan_.e[i];

        BlockSeq d{};
        d.seq      = e.seq;
        d.slot     = e.slot;
        d.card     = e.card;
        d.ctx_len  = e.ctx_len;
        d.n_tokens = e.n_tokens;

        /* EVERY ENTRY OF THE TRUNK STEP GETS A ROW, IN ORDER, whether or not its sequence is
         * still running. The drafter reads the trunk's hidden states positionally out of a buffer
         * of its own, so the context pass's row i has to be the trunk's row i; a sequence the
         * commit just finished that was left out would put every sequence after it onto its
         * neighbour's hidden states. It stays as a placeholder row that stores nothing -- see
         * BlockSeq::mirror. */
        SchedReq* sp = (e.slot >= 0 && (size_t)e.slot < reqs_.size()) ? reqs_[(size_t)e.slot].get()
                                                                        : nullptr;
        if (!sp || !sp->req || sp->req->id != e.seq || sp->req->state != ReqState::Running ||
            !sp->running) {
            d.mirror = true;
            block_.push_back(d);
            continue;
        }
        SchedReq& s = *sp;
        Request& r = *s.req;
        d.base_pos = r.n_computed;

        /* MAY IT PROPOSE? It needs a token to anchor on -- which is the one the sampler just
         * chose, so the entry has to have produced one -- and it needs a draft cache that covers
         * what the block's attention will read. The high end is this step's own context pass,
         * which has not run yet and is about to cover everything up to the trunk's end. The low
         * end is draft_lo, which is non-zero only after a prefix-cache hit THAT LEFT A HOLE. */
        const bool have_tok = e.produces_token && !r.output.empty() && r.n_computed >= 1;
        /* Where this sequence's draft cache will start once the context pass about to run has
         * written: its own floor, or -- for a sequence that has never had one -- wherever this
         * step's first row sits, which is 0 for an ordinary prefill and the hit length for a
         * prefix-cache hit. The binding query row is the LOWEST one, at n_computed, and it reads
         * back to n_computed - window + 1. */
        const int64_t lo = draft_floor(s, (int64_t)e.ctx_len);
        const bool reach = lo <= 0 ? true
                                   : (window > 0 && r.n_computed - window + 1 >= lo);

        /* RADIANCE_DEBUG_DRAFT: WHICH GATE DECLINED. A drafter that quietly stops proposing reads
         * downstream as "acceptance collapsed" -- the accept% metric is conditional on having
         * drafted at all, so it stays healthy while tok/step falls to 1. No counter distinguishes
         * the two, and nothing else tells the five gates below apart. */
        const bool ok_room = n_draft_rows < max;
        /* The query block writes positions n_computed .. n_computed + block - 1, and none of them
         * may be at or past the end of the context. */
        const bool ok_ctx  = ctx_limit_ <= 0 || r.n_computed + block <= ctx_limit_;
        const bool ok_kv   = ok_room && ok_ctx && kv_->ensure(r.id, r.n_computed + block) >= 0;
        static const bool dbg = [] { const char* v = std::getenv("RADIANCE_DEBUG_DRAFT");
                                     return v && *v && *v != '0'; }();
        if (dbg) {
            static int64_t n_call = 0, n_depth = 0, n_tok = 0, n_reach = 0, n_room = 0, n_kv = 0, n_ok = 0;
            ++n_call;
            if (!may_draft)      ++n_depth;
            else if (!have_tok)  ++n_tok;
            else if (!reach)     ++n_reach;
            else if (!ok_room)   ++n_room;
            else if (!ok_kv)     ++n_kv;
            else                 ++n_ok;
            if ((n_call % 256) == 0)
                RAD_INFO("draft gates over %lld asks: ok %lld | declined depth %lld tok %lld "
                         "reach %lld room %lld kv %lld  (window %d, floor %lld, ctx_len %lld)",
                         (long long)n_call, (long long)n_ok, (long long)n_depth, (long long)n_tok,
                         (long long)n_reach, (long long)n_room, (long long)n_kv,
                         window, (long long)lo, (long long)e.ctx_len);
        }
        if (may_draft && have_tok && reach && ok_kv) {
            d.can_draft = true;
            out[n_draft_rows].seq  = d.seq;
            out[n_draft_rows].slot = d.slot;
            out[n_draft_rows].card = d.card;
            ++n_draft_rows;
        }
        block_.push_back(d);
    }
    return n_draft_rows;
}

const Request* Scheduler::block_req(const BlockSeq& d) const { return live_req(d.slot, d.seq); }
const Request* Scheduler::serial_req(const SerialSeq& m) const { return live_req(m.slot, m.seq); }

const RadBatch* Scheduler::block_context(int step) {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_.empty()) return nullptr;

    block_plan_.reset();
    block_plan_.phase      = RAD_PHASE_DECODE;
    block_plan_.draft_pass = -1;           /* the sign is what the plugin reads (rad_runtime.h) */
    block_plan_.e.reserve(block_.size());

    for (const BlockSeq& d : block_) {
        const Request* rq = block_req(d);
        /* Cannot happen while reap() defers on block_holds(), and this is what says so: a whole
         * round skipped is a lost draft, which costs acceptance for one step and nothing else.
         * A placeholder row may have lost its Request before block_begin ran -- the server reaps
         * a finished request as soon as its sink is done -- and needs none: this pass reads no
         * token ids, and its KV is gone, so the builder stages it as padding. */
        if (!rq && !d.mirror) return nullptr;
        StepEntry e;
        e.req        = rq;
        e.seq        = d.seq;
        e.slot       = d.slot;
        e.card       = d.card;
        e.n_tokens   = d.n_tokens;
        e.ctx_len    = d.ctx_len;
        e.n_spec     = 0;
        e.n_accepted = 1;               /* one-based; nothing in this pass reads it */
        /* The context pass touches no token ids at all -- it starts from the tapped hidden states
         * -- so there is nothing to shift and no draft to reach into. */
        e.produces_token = false;
        block_plan_.e.push_back(e);
        block_plan_.n_tok += e.n_tokens;
    }
    if (block_plan_.e.empty()) return nullptr;
    return builder_.build(block_plan_, *kv_, step);
}

/* One entry a drafting sequence, `block` tokens: the anchor at the sequence's next position, then
 * `block - 1` masks. `out_ids` names THE LAST `depth` ROWS.
 *
 * AT block = depth + 1 that is rows 1..depth, the mask rows: the anchor row predicts nothing and
 * each mask row predicts the token at ITS OWN position (vLLM's _prepare_dflash_inputs_kernel:
 * is_sample = query_off >= 1, sample_pos = query_pos). AT block = depth it is every row, anchor
 * included, and each predicts the token AFTER it -- the ordinary next-token relation, which is
 * what a drafter whose reference reads `hidden[:, -block_size:, :]` was trained in. One rule,
 * both conventions; rad_builder.h argues why the plugin is what states which. */
const RadBatch* Scheduler::block_query(int step) {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_.empty() || block_len_ < 2 || block_depth_ <= 0) return nullptr;
    const int32_t block = block_len_, steps = block - 1, depth = block_depth_;

    block_plan_.reset();
    block_plan_.phase      = RAD_PHASE_DECODE;
    block_plan_.draft_pass = 1;
    block_plan_.tok_shift  = 0;            /* the anchor sits at its own row; see StepPlan */
    block_plan_.e.reserve(block_.size());

    int32_t row = 0;
    for (const BlockSeq& d : block_) {
        if (!d.can_draft) continue;
        const Request* rq = block_req(d);
        if (!rq) return nullptr;   /* see block_context */
        StepEntry e;
        e.req        = rq;
        e.seq        = d.seq;
        e.slot       = d.slot;
        e.card       = d.card;
        e.n_tokens   = block;
        e.ctx_len    = (int32_t)d.base_pos;
        e.n_spec     = 0;
        e.n_accepted = 1;
        /* token_at() walks prompt ++ output ++ draft: index base_pos is the token the sampler just
         * chose (the anchor) and everything past it falls into the mask array. */
        e.draft      = block_mask_.data();
        e.n_draft    = steps;
        e.produces_token = false;
        /* THE LAST `depth` ROWS OF THE BLOCK. At block = depth + 1 that skips the anchor; at
         * block = depth it starts on it, because there the anchor row is a predicting row.
         *
         * RADIANCE_DFLASH_ANCHOR=1 slides the head one row EARLIER, onto the anchor, and so does
         * nothing at block = depth where it already sits there. The anchor is an ordinary row: a
         * real token, embedded, with the whole context behind it -- so its hidden state predicts
         * the token at base_pos + 1, which is exactly what draft[0] is verified against. A healthy
         * backbone therefore accepts at position 0 most of the time, and a broken one does not. It
         * is the same measurement the hazard counter already makes, pointed at a row whose answer
         * is known. Debug-only: it drafts `depth` tokens for `depth` positions starting one too
         * early, so positions 1.. are deliberately misaligned and only position 0 means anything. */
        static const bool anchor_head = std::getenv("RADIANCE_DFLASH_ANCHOR") != nullptr;
        int32_t head0 = block - depth;
        if (anchor_head && head0 > 0) head0 = 0;
        e.hidden_row = row + head0;
        e.hidden_n   = depth;
        block_plan_.e.push_back(e);
        block_plan_.n_tok += block;
        row += block;
    }
    if (block_plan_.e.empty()) return nullptr;
    return builder_.build(block_plan_, *kv_, step);
}

/* The filled range moves over what the context pass wrote. The query pass's own K and V are
 * SPECULATIVE -- correct only for the positions that turn out to be accepted -- and are
 * deliberately not counted: the next step's context pass rewrites them from the trunk's real
 * hidden states, which is one pass over a handful of rows and removes the whole question. */
void Scheduler::block_end() {
    std::lock_guard<std::mutex> lk(mu_);
    for (const BlockSeq& d : block_) {
        if (d.mirror) continue;                /* finished; it has no draft cache to track */
        if (d.slot < 0 || (size_t)d.slot >= reqs_.size() || !reqs_[(size_t)d.slot]) continue;
        SchedReq& s = *reqs_[(size_t)d.slot];
        if (!s.req || s.req->id != d.seq) continue;
        if (s.draft_hi == 0 && s.draft_lo == 0) s.draft_lo = draft_floor(s, d.ctx_len);
        const int64_t hi = (int64_t)d.ctx_len + d.n_tokens;
        if (hi > s.draft_hi) s.draft_hi = hi;
    }
    /* THE SNAPSHOT ENDS HERE. Both passes are built, so nothing reads these rows again, and
     * holding them any longer would keep deferring the release of a request that was cancelled
     * or reaped while the passes ran. */
    block_.clear();
}

int32_t Scheduler::conv_window() const {
    int32_t W = 1;
    for (const KVGroupInfo& g : prog_->kv_groups)
        if (g.decl.kind == RAD_KV_CONV && g.decl.conv_width > 0)
            W = (int32_t)conv_window_entries(g.decl, cfg_.n_spec);
    return W;
}

/* ------------------------------------------------------------------ metrics */
SchedMetrics Scheduler::metrics() const {
    std::lock_guard<std::mutex> lk(mu_);
    return metrics_locked();
}

void Scheduler::publish(bool idle) {
    const int64_t now = mono_us();
    const bool demand = observer_demand_.exchange(false, std::memory_order_relaxed);
    const bool dirty  = pub_dirty_.load(std::memory_order_relaxed);
    if (!idle && !demand && !dirty && (now - pub_at_us_ < 20000)) {
        if (!pub_mu_.try_lock()) return;
        pub_m_.steps          = m_.steps;
        pub_m_.decode_tokens  = m_.decode_tokens;
        pub_m_.prefill_tokens = m_.prefill_tokens;
        pub_m_.finished       = m_.finished;
        pub_m_.cancelled      = m_.cancelled;
        pub_m_.failed         = m_.failed;
        pub_m_.draft_tokens   = m_.draft_tokens;
        pub_m_.draft_accepted = m_.draft_accepted;
        pub_m_.step           = step_;
        pub_mu_.unlock();
        return;
    }
    if (idle && now - pub_at_us_ < 20000) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        scratch_m_ = metrics_locked();
        scratch_reqs_.resize(reqs_.size());
        scratch_reqs_.resize((size_t)requests_locked(scratch_reqs_.data(), (int)scratch_reqs_.size()));
        pub_dirty_.store(false, std::memory_order_relaxed);
    }
    if (!pub_mu_.try_lock()) return;
    pub_m_ = scratch_m_;
    pub_reqs_.swap(scratch_reqs_);
    pub_mu_.unlock();
    pub_at_us_ = now;
}

SchedMetrics Scheduler::observed_metrics() const {
    observer_demand_.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(pub_mu_);
    return pub_m_;
}

int Scheduler::observed_requests(ReqStat* out, int max) const {
    observer_demand_.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(pub_mu_);
    const int n = std::min(max, (int)pub_reqs_.size());
    std::copy(pub_reqs_.begin(), pub_reqs_.begin() + n, out);
    return n;
}

SchedMetrics Scheduler::metrics_locked() const {
    SchedMetrics m = m_;
    m.step = step_;
    m.running = (int64_t)running_.size();
    m.queue_depth = (int64_t)waiting_.size();
    m.preempted_waiting = 0;
    for (int32_t slot : waiting_)
        if (reqs_[(size_t)slot]->req->state == ReqState::Preempted) ++m.preempted_waiting;

    /* AND AN IDLE ENGINE RUNS NO STEPS AT ALL, so the window never moves and the last rate would
     * stand forever. Decayed on the way out rather than mutated in place: the stored window is a
     * property of the steps that ran, and this is the honest reading of "tokens a second over the
     * recent past" when the recent past contains no tokens -- to 37% after half a second of
     * silence, to nothing after a few.
     *
     * NOT WHILE A STEP IS OPEN: a 2048-token prefill chunk runs for a third of a second, and a
     * read halfway through it is not a read of an idle engine. */
    const bool   open = step_start_.time_since_epoch().count() != 0;
    const double idle = open ? 0.0
        : std::chrono::duration<double>(std::chrono::steady_clock::now() - last_step_at_).count();
    if (idle > 0.05) {
        const float f = (float)std::exp(-idle / 0.5);
        m.prefill_tps *= f;
        m.decode_tps  *= f;
        if (m.prefill_tps < 0.05f) m.prefill_tps = 0.0f;
        if (m.decode_tps  < 0.05f) m.decode_tps  = 0.0f;
    }


    const std::vector<KVGroupInfo>& gs = prog_->kv_groups;
    m.n_kv_groups = (int)gs.size();
    if (m.n_kv_groups > RAD_SCHED_MAX_KV_GROUPS) m.n_kv_groups = RAD_SCHED_MAX_KV_GROUPS;
    for (int i = 0; i < m.n_kv_groups; ++i) {
        /* AGAINST THE CARVE AND WHAT IS HELD OF IT, not against the live mark. An elastic pool
         * grows back on demand, so a block it has released is not occupied and is not
         * unavailable; counting it as either makes an idle server read almost full and makes
         * every decision downstream that watches kv_util read the pool's own arbitration as
         * pressure from the workload. */
        const int64_t tot = kv_->total_blocks(i);
        const int64_t fr  = tot - kv_->held_blocks(i);
        m.kv[i].name = gs[(size_t)i].name.c_str();
        m.kv[i].blocks_total = tot;
        m.kv[i].blocks_free = fr;
        m.kv[i].utilisation = tot > 0 ? (float)(1.0 - (double)fr / (double)tot) : 0.0f;
        m.kv[i].block_tokens = kv_->block_tokens(i);
        m.kv[i].bytes_total  = kv_->pool_bytes(i);
        m.kv[i].bytes_used   = kv_->pool_bytes_used(i);
        m.kv[i].bytes_carved = kv_->pool_bytes_carved(i);
    }

    if (pc_) { m.prefix_evictions = pc_->evictions();
               m.prefix_ckpt_evictions = pc_->ckpt_evictions();
               m.prefix_held_tokens = pc_->held_tokens(); }
    m.attn_hit_rate = m.prompt_tokens
        ? (float)((double)m.attn_cached_tokens / (double)m.prompt_tokens) : 0.0f;
    m.linear_hit_rate = m.prompt_tokens
        ? (float)((double)m.linear_cached_tokens / (double)m.prompt_tokens) : 0.0f;
    m.draft_acceptance = draft_.acceptance();
    m.draft_depth = draft_.next_depth();
    /* THE LOOP BOUND IS THE ARRAY'S OWN EXTENT, not a constant that happens to equal it. Clamping
     * against RAD_SCHED_MAX_SPEC is correct -- depth_seen() is capped at it and both arrays are
     * that long -- but the bound and the extent merely being the same number leaves gcc unable to
     * rule out the boundary index through the inlined accessors, which guard with
     * `j < RAD_SCHED_MAX_SPEC ? reach_[j] : 0`, and it warns. Writing the cap as the extent makes
     * the two provably the same rather than coincidentally equal. */
    const int kPos = (int)(sizeof m.draft_pos_reached / sizeof m.draft_pos_reached[0]);
    m.draft_depth_seen = draft_.depth_seen();
    if (m.draft_depth_seen > kPos) m.draft_depth_seen = kPos;
    for (int j = 0; j < m.draft_depth_seen && j < kPos; ++j) {
        m.draft_pos_reached [j] = draft_.reached(j);
        m.draft_pos_accepted[j] = draft_.hits(j);
    }
    return m;
}

}  /* namespace rad */
