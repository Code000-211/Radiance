/* kbench_core.h -- rad-kbench's rules, apart from its plumbing.
 *
 * Everything here is a pure function of its arguments or a small file format: how two outputs are
 * compared, which tolerance an op gets, what a cap or cache key is made of, what the exit code is,
 * how much memory a case may take, and how a reference answer is cached. rad_kbench.cpp wires these
 * to plugins, devices and the fixture; tests/kbench_test.cpp checks the rules directly, and
 * tests/kbench_mutant_test.cpp checks the wiring with known-bad kernels.
 */
#pragma once
#include "kfixture.h"
#include "opshapes.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace rad {
namespace kb {

/* ================================================================== elementwise comparison */
struct Err {
    double max_abs = 0;
    double max_rel = 0;
    double rel_l2  = 0;
    int64_t n_nonfinite_mismatch = 0;   /* elements finite on one side and not the other */
    int64_t n = 0;
};

/* rel_l2 over the finite elements both sides agree are finite. A zero reference against a
 * non-zero output is 1, against a zero output 0. */
Err compare(const float* got, const float* ref, int64_t n);

/* Whether an operand's error replaces the worst so far when reporting a case: a failing operand
 * before a passing one, then the larger error. A NaN, once worst, stays worst. Start from
 * worst_rel < 0 so the first operand always takes it. */
bool takes_worst(double rel, bool bad, double worst_rel, bool worst_bad);

/* The larger of two errors, where NaN is the largest. std::max(a, NaN) returns a, which would
 * report a NaN case's worst error as whatever came before it. Negative means "none yet". */
double worst_of(double a, double b);

/* ================================================================== tolerances */
/* Keyed by (op prefix, activation dtype), most specific first, "*" matching anything. Each value
 * covers what a different summation order of the same arithmetic costs, with a margin, and stays
 * far below what a wrong kernel costs. The activation dtype is `dtype`, else attention's
 * `q_dtype`, else `kv_dtype`. */
struct TolRow { const char* op; const char* dtype; double rel_l2; };

struct TolPick {
    double        rel_l2 = 0;
    const TolRow* row = nullptr;      /* the table row it came from; null for an override */
    bool          catch_all = false;  /* the ("*", "*") row: no row was chosen for this op */
};

/* The activation dtype an op's parameters name, or null. */
const char* activation_dtype(const Geometry& g);

/* The tolerance for output `operand` of `op` at `g`. Per-operand overrides come first. */
TolPick tolerance_for(const std::string& op, const Geometry& g, const char* operand);

/* The rows themselves, for a test that walks every op of a fixture through them. */
const std::vector<TolRow>& tolerance_table();

/* ================================================================== index outputs
 *
 * A token id, an expert id or a top-k index is right or wrong, so it is compared exactly. A rel_l2
 * over ids would pass an expert id off by one at M = 1 (rms ~150 over 8 ids gives 2.4e-3).
 *
 * The exception is a tie: where the op's contract does not fix which of two equal inputs wins,
 * an id that indexes a value equal to the reference id's value is accepted. Ops whose contract
 * breaks ties on the lower id (row_topk, row_topk_merge, pair_argmax, logit_rerank) are exact. */

/* Integer dtypes whose values are ids, counts or offsets. i8 and u8 are quantised codes, which a
 * tolerance judges like the values they stand for. */
bool is_index_dtype(uint32_t dt);

/* Where an index output's ties are judged: the input operand its ids index, by row. */
struct TieRule {
    const char* op;
    const char* output;      /* the index operand's name */
    const char* input;       /* the operand its ids index, row by row */
    int         lane_stride; /* 2 for an (id, value) pair plane: only even lanes are ids */
};
const TieRule* tie_rule(const std::string& op, const char* output);

/* What an index output's ids point at, for judging a tie at one position. */
struct TieView {
    const void* in = nullptr;     /* the indexed input, as the kernel read it */
    uint32_t    in_dtype = 0;
    int64_t     in_rows = 0, in_width = 0;
    int64_t     out_width = 1;    /* elements per row of the index output */
    int         lane_stride = 1;
    int64_t     id_base = 0;      /* subtracted from an id before indexing (vocab_off) */
};

/* Whether position `pos` of an index output may hold `got` where the reference holds `want`:
 * equal, or (with a tie view) both ids index equal values in the same row. A lane that is not an
 * id lane must be equal. */
bool index_matches(const TieView* tie, int64_t pos, int64_t got, int64_t want);

/* ================================================================== the replay's verdict
 *
 * A replay has the reference reduced to a digest (kfixture.h); the kernel's output is reduced the
 * same way here and the two compared. Every quantity feeds one rel_l2-scaled error: the sampled
 * positions, the full-tensor norm (a correct first tile followed by zeros) and, in a version-5
 * recording, the projection sketch (a localised error the samples miss and the norm cannot see,
 * such as a permuted or sign-flipped tile). */
struct Digest {
    int64_t             n = 0;
    double              sumsq = 0;
    int64_t             nonfinite = 0;
    std::vector<float>  sample;
    std::vector<double> sketch;
};

/* The kernel side of a recorded operand: the same reduction the recorder ran, with the sketch only
 * when `sketch` (a version-4 recording has none to compare with). */
void digest_of(const float* v, int64_t n, uint64_t seed, bool sketch, Digest* out);

/* The recorded reference of `rec` as a digest. */
Digest digest_of(const KfOperand& rec);

struct OpdVerdict {
    double  rel = 0;            /* the error the verdict uses: the largest of the three below */
    double  sample_rel = 0, norm_rel = 0, sketch_rel = -1;
    double  max_abs = 0, max_rel = 0;
    int64_t nonfinite_gap = 0;  /* sampled finiteness disagreements plus the count difference */
    bool    index = false;      /* compared as ids */
    int64_t index_mismatch = 0, ties = 0;
    bool    vacuous = false;    /* both sides identically zero: nothing was checked */
    bool    bad = false;
};

/* One recorded output against the kernel's digest. `tol` is the operand's tolerance; `exact`
 * judges with tolerance 0 (the reference library replaying its own recording). Index dtypes are
 * compared at the sampled positions id by id, through `tie` where the op allows ties; positions are
 * re-derived from `pos_seed` as the recorder drew them. */
OpdVerdict judge_operand(const KfOperand& rec, const Digest& mine, uint64_t pos_seed, double tol,
                         bool exact, const TieView* tie);

/* ================================================================== keys */
/* The replay's cap key: the op, the kernel and library that serve it, the domain, the recorded
 * parameters, and every operand's dtype and shape. Depth lives in the operand shapes (a KV cache
 * of 1024 blocks against 128), not in the parameters, so a key without them collapses the
 * attention depth sweep into its shallowest case. */
std::string cap_key(const KfCase& c, const std::string& kernel, const std::string& lib, int dom,
                    const Geometry& g);

/* The host weight cache's key: the weight's identity seed, its encoding and its plane selection.
 * Two weights with one seed can differ in encoding (a bf16 embedding against a two-plane fp8
 * lm_head of the same shape). */
uint64_t host_weight_key(const KfOperand& d, size_t k);

/* The device weight cache's key: the host key plus the library and kernel whose layout produced
 * the bytes, and their size. */
std::string device_weight_key(const std::string& lib, const std::string& kernel,
                              const KfOperand& d, size_t k, size_t bytes);

/* ================================================================== memory */
/* What one case may take: three copies of every present operand (the draw, the kernel's copy, a
 * re-laid weight or a device copy). -1 when an extent is not positive or the product overflows,
 * which a loaded fixture refuses anyway. */
int64_t case_bytes(const std::vector<KfOperand>& opd);

/* The default case budget from /proc/meminfo's text: half of MemAvailable, clamped to
 * [512 MiB, 8 GiB]. MemFree is not the budget: page cache counts against it and is reclaimable.
 * `avail` gets MemAvailable in bytes, or 0 when the text has none (then the budget is 8 GiB). */
int64_t default_case_budget(const std::string& meminfo, int64_t* avail);

/* ================================================================== red zones */
/* A guarded allocation's margins after a launch: the furthest byte changed before the start (as a
 * distance from it) and past the end. Zero and zero is clean. */
struct Overrun { int64_t before = 0, after = 0; };
Overrun scan_margins(const uint8_t* lo, const uint8_t* hi, int64_t red_bytes, uint8_t fill);

/* ================================================================== what a run found */
struct Stats {
    int checked = 0, passed = 0, failed = 0, skipped = 0, capped = 0, diverged = 0;
    int overrun = 0, guarded = 0, timed = 0;
    int fuse_failed = 0;
    /* A host kernel that changed an operand it was handed to read. */
    int input_written = 0;
    /* Both sides identically zero: counted apart from passes, since nothing was checked. */
    int vacuous = 0;
    /* Skips that mean the fixture and the libraries no longer agree: an operand count or schema
     * that moved, the reference without a row for a case it recorded, or describing an output
     * differently from its recording. Failures under --strict. */
    int stale = 0;
    /* Cases no library under test serves: named, never a failure. */
    int unserved = 0;
    /* Reference results taken from the cache instead of a launch. */
    int cached = 0;
    double worst = 0;
};

/* 1 when anything failed: a wrong number, an overrun, a fusion that differs from its chain, a
 * kernel that wrote its input, and under `strict` a stale fixture. 0 otherwise. */
int exit_code(const Stats& st, bool strict);

/* Why a `--bench` run may not report its times, or "" when it may: every op the run has a time for
 * must have been checked. A case of that op that was skipped or is stale leaves the op unchecked,
 * and a time for an unchecked op is a time for something nobody has shown to be right -- which is
 * why the gate refuses instead of printing it. `measured` names the ops a time would be reported
 * for; the gaps are the pass's own named skips and stale cases (their keys are `op  (why)`). A
 * --timing-only run measures unchecked launches on purpose and never asks. */
std::string bench_refusal(const Stats& st, const std::set<std::string>& measured,
                          const std::map<std::string, int>& skipped,
                          const std::map<std::string, int>& stale);

/* Why a recorded case is stale, or "" when it can be checked. A case is stale when its recording
 * cannot be answered: the library it names (`recorded`, the fixture's `oracle` without its version)
 * serves no host row for the op now and no loaded library declares a reference for it, or that row
 * no longer describes the recording (`differs`, which non-empty states how). `oracle` is the library
 * the resolver found -- a substitute reference for another library is not stale, and is reported as
 * such. A substitute reference is named in `differs` only when it is the recording itself, which is
 * the point: only the recording's own row is held to the recording. The reason names the library,
 * because the only action it asks for is re-recording from the one that changed. */
std::string stale_reason(const std::string& recorded, const std::string& oracle, bool oracle_row,
                         const std::string& differs);

/* ================================================================== where the time went */
enum Phase {
    PH_LOAD = 0,     /* plugins and fixture */
    PH_DRAW,         /* inputs and weight planes */
    PH_HASH,         /* input hashes for the reference cache */
    PH_LAYOUT,       /* weight relayouts */
    PH_UPLOAD,       /* allocations and host-to-device copies */
    PH_LAUNCH,       /* the checked launches, per library below */
    PH_READBACK,     /* device-to-host copies and margins */
    PH_COMPARE,      /* reductions and verdicts */
    PH_CACHE,        /* reference cache reads and writes */
    PH_TIMING,       /* --bench: calibration, warm-up and the timed batches */
    PH_REPORT,
    PH_N
};

struct Phases {
    double s[PH_N] = {};
    std::map<std::string, double> launch_by_lib;
    void add(Phase p, double sec) { s[p] += sec; }
};

/* Accumulates the time from construction to destruction (or stop()) into one phase. */
class PhaseTimer {
public:
    PhaseTimer(Phases* ph, Phase p) : ph_(ph), p_(p), t0_(std::chrono::steady_clock::now()) {}
    ~PhaseTimer() { stop(); }
    double stop();
private:
    Phases* ph_;
    Phase   p_;
    std::chrono::steady_clock::time_point t0_;
    bool    done_ = false;
};

/* "time: draw 1.2 s, launch 4.0 s (libavx 3.1 s, libref 0.9 s), ... of 6.0 s wall". */
std::string phase_line(const Phases& ph, double wall_s);

/* ================================================================== timing
 *
 * The rules of the timing pass (tools/kbench_timer.cpp), apart from its device calls: statistics
 * over per-launch times and when to stop collecting them, how long a batch runs, how many rotated
 * copies make every launch read its weights from memory, which operands rotate, the clock
 * telemetry's table, the calibration's plausibility check, and the inputs the timing pass draws
 * where the correctness draw is not what the engine sees. */

/* Order statistics of per-launch times in microseconds, one per timed batch. */
struct TimeStats {
    int    n = 0;
    double median = -1, min = -1, max = -1;
    double p90 = -1;          /* nearest rank: the ceil(0.9 n)-th smallest */
    /* A distribution-free confidence interval for the median: the order statistics of ranks
     * floor(n/2 - 0.98 sqrt(n)) and ceil(n/2 + 1 + 0.98 sqrt(n)), clipped to the sample. Up to
     * ten samples that is the whole range, which covers the median with probability
     * 1 - 2^(1-n): 95% from six samples on, less below (94% at five, 50% at two). From eleven on,
     * one outlier on either side leaves it unchanged. None (-1) for fewer than two samples. */
    double ci_lo = -1, ci_hi = -1;
    double ci_pct = -1;       /* half the interval's width, as a percent of the median */
};
TimeStats time_stats(std::vector<double> v);

/* Whether a case has been timed enough: `max_n` batches, or at least `min_n` whose median's
 * interval half-width is within `ci_pct` percent. */
bool timing_done(const std::vector<double>& v, int min_n, int max_n, double ci_pct);

/* The device time one batch should last: a millisecond, and fifty launch floors, so the batch's
 * own start and end (two barrier packets) are a small share of it. */
double batch_target_us(double floor_us);

/* Launches in one batch: `fixed` when positive (--bench-iters), else enough to last `target_us` at
 * `est_us` a launch. At least one, at most `room` (what fits behind the gate). */
int64_t batch_launches(double est_us, double target_us, int64_t room, int64_t fixed);

/* Why a device time is an order of magnitude rather than a measurement, or "" when it is one. A
 * time is a timed batch minus a batch of the in-place restores alone, divided by its launches:
 * restores at `restore_pct` >= 75 of the batch leave the kernel a small difference of two large
 * times, and a median more than 5% under the back-to-back launch floor, which a dependent
 * dispatch cannot beat, shows the difference is biased. -1 for either input means not measured. */
std::string timing_caveat(double median_us, double floor_us, double restore_pct);

/* The distance between rotated copies of an operand: its bytes rounded up to 256, the alignment a
 * guarded allocation gives every operand. */
int64_t rotation_stride(int64_t bytes);

/* How many copies of the rotated operands make every launch read them from memory: between two
 * uses of one copy, the launches in between read at least `footprint` bytes of other copies, which
 * evicts it from every cache smaller than half the footprint. `per_launch` is the bytes one launch
 * reads, one copy of each rotated operand. One when that alone reaches the footprint (it evicts
 * itself) or the footprint is zero (warm timing). */
int64_t rotation_copies(int64_t per_launch, int64_t footprint);

/* The rotation footprint when --bench-flush does not set one: twice `cache_bytes` (L2 plus the last
 * level), and at least 256 MiB, twice the R9700's 8 MiB L2 and 64 MiB last-level cache with room to
 * spare. */
int64_t default_footprint(int64_t cache_bytes);

/* Whether an operand rotates in cold timing. Weights and weight tables always: the engine streams
 * them once a step. An input from `big_input` bytes on: such an input is state the engine also
 * reads once a step (a KV cache, a table), while a smaller one is an activation the previous op
 * left in the cache. Outputs never, and in-place operands are restored instead. */
bool rotated(int role, int64_t bytes, int64_t big_input);

/* The order of a weight table's entries for launch `l`: a seeded permutation of [0, n), the same
 * for every table of the same size. A launch reads the experts its routing names through a pointer
 * table that sends them to these slots, so consecutive launches read different experts' bytes for
 * the same routing (a table the size of the footprint is then cold, though one launch reads only
 * its routed experts). */
void table_permutation(int64_t n, uint64_t seed, int64_t l, std::vector<int64_t>* out);

/* How many pointer tables a weight table rotated over `copies` copies cycles through: at least 64,
 * and a multiple of `copies`, so table l always points into copy l mod copies and launch i, using
 * table i mod count, reads the same copy as the other operands' rotation. */
int64_t table_count(int64_t copies);

/* Pointer table `l` of a weight table: `n` pointers, entry e at the base of copy l mod R (`bases`,
 * one per copy) plus table_permutation(n, seed, l)[e] slots of `entry` bytes. Appended to `out`. */
void table_pointers(const std::vector<uint64_t>& bases, int64_t n, int64_t entry, uint64_t seed,
                    int64_t l, std::vector<uint64_t>* out);

/* Rows of each of `buckets` experts out of `total`, skewed as real routing is: Zipf weights with
 * exponent 0.5 over the experts in a seeded order (the hottest of 128 takes about six times the mean
 * share), at most `cap` each (a token visits an expert once: the op's M), every row placed. */
std::vector<int64_t> skewed_counts(int64_t buckets, int64_t total, int64_t cap, uint64_t seed);

/* An input drawn again for the timing pass, where the correctness draw gives the kernel less (or
 * different) work than the engine does and the kernel's time depends on the values. The checked
 * launch keeps the recorded draw, so fixtures replay byte for byte. Into `out` (in the operand's
 * dtype), true for:
 *   - an expert histogram's offsets (an index prefix): skewed_counts with the op's M as the cap;
 *   - an index operand ranging over the op's n_expert (routing choices): each row's ids distinct,
 *     drawn from the same skewed weights;
 *   - an index table of rank two or more with as many entries as its range (a block table, drawn
 *     with repeats): a permutation, as an allocator hands out distinct blocks.
 * False, `out` untouched, for every other operand. */
bool timing_input(const KfOperand& d, const Geometry& g, uint64_t seed, std::vector<uint8_t>* out);

/* The SMU's metrics table as the driver exports it (gpu_metrics). */
struct GpuMetrics {
    int      format = 0, content = 0;   /* the header's revisions */
    double   gfxclk_mhz = -1;           /* averaged graphics clock */
    double   gfxclk_now_mhz = -1;       /* instantaneous */
    double   hotspot_c = -1;
    double   power_w = -1;
    double   gfx_busy_pct = -1;
    uint64_t throttle = 0;              /* the ASIC-independent throttler bits */
};

/* Reads a gpu_metrics table. Only layout v1.3 (RDNA3 and RDNA4 discrete cards, the layout
 * scripts/cardclock.py reads and checks against hwmon) is read; any other is refused, so another
 * card's table is never misread. A field the firmware reports as 0xFFFF stays -1. */
bool parse_gpu_metrics(const uint8_t* p, size_t n, GpuMetrics* out);

/* "PPT0+TEMP_HOTSPOT" for throttler bits (amdgpu_smu.h SMU_THROTTLER_*_BIT), "" for none. */
std::string throttle_names(uint64_t bits);

/* One size of the calibration's streaming-read probe: GB/s of bytes read, the same buffer reread
 * (warm) and a buffer rotated over the footprint (cold). */
struct BwProbe {
    int64_t bytes = 0;
    double  warm_gbs = -1, cold_gbs = -1;
};

/* Why the probes say cold timings would not be cold, or "" when they are consistent. Cold reads
 * stream from memory at one rate whatever their size, so a smaller cold probe more than 1.5x as
 * fast as the largest means the rotation was served from a cache. The largest probe is as large as
 * the footprint, past every cache warm or cold, so its two rates must agree within 1.5x; otherwise
 * the clock moved during calibration. A probe without a positive rate failed. */
std::string bandwidth_refusal(const std::vector<BwProbe>& probes);

/* ================================================================== hashing */
/* XXH64 (Collet's xxHash, 64-bit), for input hashes: fast enough to hash every drawn input on
 * every run. Not cryptographic. */
uint64_t hash64(const void* p, size_t n, uint64_t seed);
uint64_t hash_combine(uint64_t h, uint64_t v);
uint64_t hash_str(uint64_t h, const std::string& s);

/* A hash of a whole buffer, its 4 MiB blocks hashed on every core and combined in block order:
 * the same value at any thread count. Fingerprints both the reference cache's input hash and the
 * check that a host kernel left its inputs alone. */
uint64_t fingerprint(const void* p, size_t n);

/* The identity of the code a library runs: the ELF build-ids of the library at `lib_path` and of
 * every shared object the process has loaded that is not in `exclude` (other kernel plugins) and
 * not the main program, plus the CPU's model and feature flags (libm picks FMA or non-FMA
 * routines at load). 0 when the library has no build-id, which disables caching for it. */
uint64_t code_identity(const std::string& lib_path, const std::vector<std::string>& exclude);

/* The build-id note of the loaded object at `path`, as hex; "" when it has none. */
std::string build_id_of(const std::string& path);

/* ================================================================== the reference cache
 *
 * libref is deterministic and is most of a replay's CPU time, and its answer to a case changes
 * only when its code, its inputs or the case change. So its reduced outputs are cached under a key
 * of all three: the code identity, a hash of every input byte it was handed (which catches a draw
 * or decode change wherever that code lives) and the case itself (op, parameters, operand
 * descriptions, sample seeds, the row). The comparison with the recording still runs on every
 * hit, so a tolerance or comparison change is never hidden by the cache.
 *
 * One file per entry, written to a temporary name and renamed. An entry that fails its checksum
 * or does not match its key is a miss. */
class RefCache {
public:
    /* `dir` empty disables the cache. */
    void open(const std::string& dir, uint64_t identity);
    bool enabled() const { return on_; }

    bool get(uint64_t key, std::vector<Digest>* outs) const;
    void put(uint64_t key, const std::vector<Digest>& outs);

    int hits() const { return hits_; }
    int writes() const { return writes_; }
    const std::string& dir() const { return dir_; }
    uint64_t identity() const { return id_; }

private:
    std::string path_of(uint64_t key) const;
    std::string dir_;
    uint64_t    id_ = 0;
    bool        on_ = false;
    bool        warned_ = false;
    mutable int hits_ = 0;
    int         writes_ = 0;
};

/* A case's cache key: the code identity, the row, the case's description and the input hash. */
uint64_t ref_cache_key(uint64_t identity, const std::string& kernel, const KfCase& c,
                       const Geometry& g, uint64_t input_hash);

/* ================================================================== row coverage ratchet
 *
 * One file per (fixture, library): the kernel rows a replay of that fixture reached, one per line,
 * sorted. A row listed and not reached fails the ratchet, unless every case that would have reached
 * it was skipped for memory (it is unknown, not lost). Rows reached and not listed are reported, to
 * be added. */
std::set<std::string> read_rows_file(const std::string& path, bool* exists);
int write_rows_file(const std::string& path, const std::set<std::string>& rows);

struct Ratchet {
    std::vector<std::string> lost, gained;
};
Ratchet ratchet(const std::set<std::string>& listed, const std::set<std::string>& reached,
                const std::set<std::string>& unknown);

/* The ratchet file name for a fixture and library: kbench_rows_<fixture stem>_<library>.txt. */
std::string rows_file_name(const std::string& fixture_path, const std::string& lib);

/* ================================================================== synthetic geometry */
/* Moves `g` into a kernel row's constraints where a single parameter can satisfy one: an EQ value,
 * the first IN value, a violated LE or GE edge, the next DIV multiple. Returns true when every
 * constraint then holds. Device facts are never invented. */
bool satisfy_constraints(const RadKernelInfo* ki, Geometry* g);

/* ================================================================== skeleton containers */
/* Writes a sparse copy of the container at `src`: the header and every table byte for byte, the
 * data region a hole. Opening it gives the same metadata, directory and encodings, so the declare
 * phase and a recording (which draws every weight) run on it, while it occupies a few megabytes
 * on disk whatever the model's size. RAD_OK, or an error with `err` set. */
int write_skeleton(const std::string& src, const std::string& dst, std::string* err);

}  /* namespace kb */
}  /* namespace rad */
