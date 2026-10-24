#!/bin/sh
# The paired sealed A/B for a chunked prefill sharing a step: docs/PREFILL-OVERLAP.md.
#
# The question is whether a prompt's chunk is worth carrying beside other people's decode rows, or
# whether the same work is better run as separate steps. Issue #8 asks for that comparison against
# one configuration, reporting decode, prefill, the throughput at eight streams and the draft
# acceptance, because a faster prefill that costs acceptance is not a win. This runs it.
#
# Method, and why each part is there:
#
#   One container, one corpus, one seed per arm. Every arm restarts the server through
#   scripts/fnserve.sh with only its own flags in EXTRA, so the container, the context, the wire
#   and the chunk are the same in all of them. The chunk is part of the configuration
#   (docs/GUIDE.md §4.5): each arm is byte-stable on its own, so what is compared is configurations,
#   not runs.
#
#   Arms alternate, and every arm runs in every pass. Cards drift with temperature over a session
#   and a long MoE prefill is what heats them, so an arm measured first every time carries the
#   temperature of the arm before it. Two passes, the second reversed, is the minimum that
#   separates an arm from its position; the passes are printed apart rather than averaged, so a
#   spread that large stays visible instead of being smoothed away. The first request after a
#   restart is a warm-up and is not measured: `expert_tiered` faults its tier in over the first
#   requests, and charging that to whichever arm ran first is how contention gets mistaken for a
#   speedup (scripts/fnconc.sh).
#
#   The figures come from the engine's own counters, as deltas across a timed window, not from the
#   rate gauges: a rate is a decaying window and covers whatever else the server was doing. The
#   step's own milliseconds are engine_step_seconds_decode_total over engine_steps_decode_total,
#   which is the one figure no wall clock gives.
#
#   Acceptance is the delta of draft_accepted_total over draft_tokens_total across the arm's whole
#   measurement, including the prefill phases, because a shared step changes the batch shape and a
#   shape change moves ties, which is what acceptance reports first. If acceptance moves between
#   arms, their throughput figures describe different text and the comparison means nothing.
#
#   The mixed-step counters are printed as deltas too (docs/PREFILL-OVERLAP.md). They are what
#   attributes a prefill loss: a chunk in a step too small for the prefill stager to arm
#   (prefill_rows_unstaged) is a link cost; a smaller GEMM is not. Without them, an arm that lost
#   prefill throughput and an arm that never staged are the same line in a table.
#
#   Inter-token latency is client-side and is really inter-chunk: the server streams as it decodes
#   and a chunk can carry more than one token, so the percentile is of arrivals, not of tokens. It
#   is here because the sweep this harness replaces reported a p99 ITL and a reader looks for it;
#   compare it only against another run of this script. Timed throughput windows are otherwise
#   unstreamed, so their token counts come from `usage` and not from an arrival count.
#
#   Nothing here is a verdict. docs/PREFILL-OVERLAP.md names what each outcome means; this prints
#   numbers, one sealed JSON line per arm-pass.
#
#   P=8100 M=$HOME/models/rad/w4nl64-i8lm.rad ARMS="off auto 1024 isolated" \
#     PFN=24000 DEC=300 C=8 PASSES=2 scripts/fnmix.sh
#
# Needs a card, the container, curl and jq (git for the sha, rocm-smi for the clocks, both
# optional). Stop the server afterwards with scripts/fnstop.sh.
#
# Environment:
#   ARMS      which arms to run, in order: off, auto, isolated, or a token count
#             (default "off auto 1024 isolated")
#   PASSES    how many times the whole list is run, alternating (default 2)
#   PFN       prompt tokens for the prefill phases (default 24000)
#   DEC       tokens each decoder generates (default 300)
#   C         concurrent decoders for the loaded decode phase (default 8)
#   NPROBE    decoders for the contention phase, 0 to skip it (default 4)
#   MNBT      --max-num-batched-tokens for every arm (default 2048)
#   EXTRA_EXTRA   appended to every arm's flags, e.g. "--tp-wire exact"
#   REPORT    where the JSON lines go (default $HOME/fnmix-<date>.jsonl)
P="${P:-8100}"
H="${H:-127.0.0.1}"
URL="http://$H:$P"
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ARMS="${ARMS:-off auto 1024 isolated}"
PASSES="${PASSES:-2}"
PFN="${PFN:-24000}"
DEC="${DEC:-300}"
C="${C:-8}"
NPROBE="${NPROBE:-4}"
MNBT="${MNBT:-2048}"
BPT="${BPT:-4}"                        # bytes of this corpus per token, about 4.0
M="${M:-$HOME/models/rad/w4nl64-i8lm.rad}"
CORPUS="${CORPUS:-/dev/shm/fnmix-corpus.txt}"
REPORT="${REPORT:-$HOME/fnmix-$(date +%Y%m%d-%H%M).jsonl}"
W=/tmp/fnmix.$$
mkdir -p "$W" || exit 1
trap 'rm -rf "$W"' EXIT INT TERM

arm_flags() {
    case "$1" in
      off)      echo "--busy-prefill-chunk off" ;;
      auto)     echo "--busy-prefill-chunk auto" ;;
      isolated) echo "--busy-prefill-chunk off --prefill-interleave 2" ;;
      *)        echo "--busy-prefill-chunk $1" ;;
    esac
}

# ---------------------------------------------------------------- corpus and prompts
# The repository's own markdown, repeated: random-letter words route through the experts
# differently from real text (scripts/fnpf.sh). Every prompt carries a fresh nonce at the front,
# because the prefix cache matches from the start and a trailing nonce would leave all but the last
# block a hit.
want=$((PFN * BPT + 4096))
if [ "$(wc -c < "$CORPUS" 2>/dev/null || echo 0)" -lt "$want" ]; then
    : > "$CORPUS" || exit 1
    while [ "$(wc -c < "$CORPUS")" -lt "$want" ]; do
        cat "$ROOT"/spec.md "$ROOT"/docs/*.md >> "$CORPUS" 2>/dev/null || exit 1
    done
fi

# $1 tag, $2 max_tokens, $3 stream (true|false), $4 long (1: PFN tokens of corpus, 0: a short
# decoding prompt). The body lands in $W/$1.json. `stream` has to be a JSON boolean, so it is
# merged in a second jq call rather than passed as a string.
make_body() {
    if [ "$4" = 1 ]; then
        { printf '%s %s\n\n' "$1" "$(date +%s%N)"; head -c $((PFN * BPT)) "$CORPUS"; } > "$W/$1.txt"
    else
        printf '%s count upward and do not stop:' "$1" > "$W/$1.txt"
    fi
    jq -Rs --argjson mt "$2" '{prompt:., max_tokens:$mt, temperature:0, ignore_eos:true}' \
        < "$W/$1.txt" > "$W/$1.b" || exit 1
    if [ "$3" = true ]; then
        jq -c '. + {stream:true, stream_options:{include_usage:true}}' < "$W/$1.b" > "$W/$1.json"
    else
        jq -c '. + {stream:false}' < "$W/$1.b" > "$W/$1.json"
    fi || exit 1
}

post() {                                  # $1 tag -> response in $W/$1.sse
    curl -sN --max-time 1800 -X POST "$URL/v1/completions" -H 'Content-Type: application/json' \
        --data-binary @"$W/$1.json" > "$W/$1.sse"
}

# ---------------------------------------------------------------- the engine's counters
# Exact totals, from /metrics. The names carry the namespace's colon and are matched exactly, so a
# prefix cannot answer for a longer name.
COUNTS="prefill_tokens_total decode_tokens_total engine_step_seconds_decode_total
engine_steps_decode_total engine_steps_mixed_total engine_steps_multi_prefill_total
engine_decode_rows_mixed_total engine_prefill_rows_mixed_total engine_prefill_tokens_mixed_total
engine_prefill_rows_unstaged_total engine_prefill_chunks_capped_total engine_steps_held_total
draft_tokens_total draft_accepted_total attn_cached_tokens_total"

mnum() {                                  # $1 metric name -> its value, 0 if absent
    curl -s -m 10 "$URL/metrics" | awk -v n="radiance:$1" '$1 == n { print $2; exit }'
}

snap() {                                  # $1 file: "name value" per counter
    : > "$1"
    for n in $COUNTS; do printf '%s %s\n' "$n" "$(mnum "$n")" >> "$1"; done
}

deltas() {                                # $1 before $2 after -> "name delta" where it moved
    awk 'NR==FNR { a[$1] = $2 + 0; next } { d = ($2 + 0) - a[$1]
          if (d) { if (d == int(d)) printf "%s %d\n", $1, d; else printf "%s %.6f\n", $1, d } }' \
        "$1" "$2"
}

dget() {                                  # $1 name $2 delta file -> delta, 0 if absent
    awk -v n="$1" '$1 == n { print $2; found = 1 } END { if (!found) print 0 }' "$2"
}

# The usage a response carried: for a streamed request, the last chunk that had one (with
# stream_options.include_usage the server appends it); for an unstreamed one, the object itself.
# It is the only token count here that is not an arrival count.
usage_field() {                           # $1 response file $2 field -> number
    if grep -q '^data: {' "$1" 2>/dev/null; then
        grep '^data: {' "$1" | sed 's/^data: //' |
            jq -sr "[.[] | (.usage // empty) | (.$2 // empty)] | if length then last else 0 end" \
            2>/dev/null
    else
        jq -r "(.usage // {}) | (.${2}) // 0" "$1" 2>/dev/null
    fi
}

unum() {                                  # $1 response file $2 field -> integer, 0 if no usage
    v=$(usage_field "$1" "$2" | cut -d. -f1)
    printf '%s' "${v:-0}"
}

card() {
    rocm-smi --showtemp --showclocks --csv 2>/dev/null |
        awk -F, 'NR>1 && $1 ~ /card/ { printf "%s ", $2 }' | head -c 48
}

# ---------------------------------------------------------------- the phases
# One stream, unstreamed: the C1 throughput over a fixed token count. Prints "seconds tokens".
probe_c1() {
    make_body "$1" "$DEC" false 0
    t0=$(date +%s.%N); post "$1"; t1=$(date +%s.%N)
    awk -v a="$t0" -v b="$t1" -v n="$(unum "$W/$1.sse" completion_tokens)" \
        'BEGIN { printf "%.2f %d\n", b - a, n }'
}

# One stream, streamed, arrival times recorded while the stream is read (the clock and the read
# have to be in the same loop). Prints "first_ms p99_ms chunks seconds".
probe_c1_stream() {
    make_body "$1" "$DEC" true 0
    : > "$W/$1.ts"
    t0=$(date +%s.%N)
    curl -sN --max-time 1800 -X POST "$URL/v1/completions" -H 'Content-Type: application/json' \
        --data-binary @"$W/$1.json" |
    while IFS= read -r line; do
        case "$line" in 'data: '{*}) date +%s.%N >> "$W/$1.ts" ;; esac
    done
    t1=$(date +%s.%N)
    awk -v a="$t0" -v b="$t1" -v f="$W/$1.ts" 'BEGIN {
        n = 0
        while ((getline t < f) > 0) { if (n) d[n - 1] = t - prev; else first = t - a; prev = t; n++ }
        for (i = 1; i < n - 1; i++) for (j = i + 1; j < n - 1; j++)
            if (d[j] < d[i]) { x = d[i]; d[i] = d[j]; d[j] = x }
        p99 = 0
        if (n > 3) { k = int(0.99 * (n - 2)); if (k > n - 3) k = n - 3; p99 = d[k] }
        printf "%.0f %.2f %d %.2f\n", first * 1000, p99 * 1000, n, b - a
    }'
}

# C streams at once, unstreamed, started before the clock starts so none of them pays for the
# others being built. Prints "seconds tokens".
probe_cN() {
    pids=
    i=0
    while [ "$i" -lt "$C" ]; do make_body "$1-$i" "$DEC" false 0; i=$((i + 1)); done
    t0=$(date +%s.%N)
    i=0
    while [ "$i" -lt "$C" ]; do post "$1-$i" & pids="$pids $!"; i=$((i + 1)); done
    for p in $pids; do wait "$p"; done
    t1=$(date +%s.%N)
    tok=0
    i=0
    while [ "$i" -lt "$C" ]; do
        tok=$((tok + $(unum "$W/$1-$i.sse" completion_tokens)))
        i=$((i + 1))
    done
    awk -v a="$t0" -v b="$t1" -v n="$tok" 'BEGIN { printf "%.2f %d\n", b - a, n }'
}

# The long prompt, unstreamed: prefill throughput. Prints "seconds prompt_tokens".
probe_prefill() {
    make_body "$1" 1 false 1
    t0=$(date +%s.%N); post "$1"; t1=$(date +%s.%N)
    awk -v a="$t0" -v b="$t1" -v n="$(unum "$W/$1.sse" prompt_tokens)" \
        'BEGIN { printf "%.2f %d\n", b - a, n }'
}

# The same length again, streamed, with its own nonce, for the time to the first token: what the
# isolation arm pays. Prints "ttft_ms".
probe_ttft() {
    make_body "$1" 1 true 1
    t0=$(date +%s.%N)
    curl -sN --max-time 1800 -X POST "$URL/v1/completions" -H 'Content-Type: application/json' \
        --data-binary @"$W/$1.json" |
    { IFS= read -r line
      case "$line" in 'data: '{*) date +%s.%N > "$W/$1.first" ;; esac
      while IFS= read -r line; do :; done; }
    t1=$(date +%s.%N)
    awk -v a="$t0" -v b="$t1" -v f="$W/$1.first" 'BEGIN {
        if ((getline t < f) > 0) printf "%.0f\n", (t - a) * 1000; else printf "0\n" }'
}

# ---------------------------------------------------------------- one arm, one pass
run_arm() {
    arm="$1"; pass="$2"
    flags=$(arm_flags "$arm")
    echo "== pass $pass arm $arm: EXTRA=\"$flags\""
    EXTRA="$flags $EXTRA_EXTRA" MNBT="$MNBT" P="$P" M="$M" ROOT="$ROOT" \
        sh "$ROOT/scripts/fnserve.sh" || { echo "  arm $arm did not start"; return 1; }

    # Warm-up: a request nobody measures, so the tier is faulted in before the first window.
    probe_prefill "warm-$arm-$pass" > /dev/null

    snap "$W/$arm-$pass.before"

    probe_c1         "c1-$arm-$pass"  > "$W/c1";   read -r c1_w c1_tok < "$W/c1"
    probe_c1_stream  "c1s-$arm-$pass" > "$W/c1s";  read -r c1_first c1_p99 c1_chunks c1s_w < "$W/c1s"
    probe_cN         "cN-$arm-$pass"  > "$W/cN";   read -r cN_w cN_tok < "$W/cN"
    probe_prefill    "pf-$arm-$pass"  > "$W/pf";   read -r pf_w pf_tok < "$W/pf"
    probe_ttft       "tt-$arm-$pass"  > "$W/tt";   read -r ttft < "$W/tt"

    cont=""
    if [ "$NPROBE" -gt 0 ]; then
        cont=$(P="$P" C="$NPROBE" DEC="$DEC" PROMPT="$PFN" ROOT="$ROOT" \
               sh "$ROOT/scripts/fnconc.sh" 2>/dev/null | tail -1 | tr -d '"')
    fi

    snap "$W/$arm-$pass.after"
    deltas "$W/$arm-$pass.before" "$W/$arm-$pass.after" > "$W/$arm-$pass.d"

    awk -v c1w="$c1_w" -v c1t="$c1_tok" -v c1f="$c1_first" -v c1p="$c1_p99" \
        -v cnw="$cN_w" -v cnt="$cN_tok" -v C="$C" -v pf="$pf_tok" -v pfw="$pf_w" \
        -v ttft="$ttft" \
        -v sdec="$(dget engine_step_seconds_decode_total "$W/$arm-$pass.d")" \
        -v ndec="$(dget engine_steps_decode_total "$W/$arm-$pass.d")" \
        -v d_pf="$(dget prefill_tokens_total "$W/$arm-$pass.d")" \
        -v acc="$(dget draft_accepted_total "$W/$arm-$pass.d")" \
        -v drf="$(dget draft_tokens_total "$W/$arm-$pass.d")" 'BEGIN {
        printf "  decode C1    %6.1f tok/s   first token %5.0f ms   p99 inter-chunk %6.2f ms\n",
               (c1w > 0 ? c1t / c1w : 0), c1f, c1p;
        printf "  decode C%-3s  %6.1f tok/s   (%d tokens over %.2f s)\n", C,
               (cnw > 0 ? cnt / cnw : 0), cnt, cnw;
        printf "  prefill      %6.1f tok/s   %d prompt tokens over %.2f s (engine counted %d)\n",
               (pfw > 0 ? pf / pfw : 0), pf, pfw, d_pf;
        printf "  long prompt  TTFT %6.0f ms\n", ttft;
        printf "  decode step  %7.3f ms mean over %d decode steps\n",
               (ndec > 0 ? sdec * 1000 / ndec : 0), ndec;
        # mawk will not continue a ternary across a newline after a complete sprintf call.
        if (drf > 0) printf "  acceptance   %.4f of %d drafted\n", acc / drf, drf;
        else printf "  acceptance   no drafts in this window\n";
    }'
    printf '  mixed        '
    awk '{ printf "%s=%s ", $1, $2 } END { if (!NR) printf "nothing moved" }' "$W/$arm-$pass.d"
    printf '\n'
    [ -n "$cont" ] && printf '  contention   %s\n' "$cont"
    printf '  cards        %s\n' "$(card)"

    # One sealed record per arm-pass: the figures, every counter delta behind them, and everything
    # that makes the run reproducible.
    {
        printf '{"arm":"%s","pass":%s,"flags":"%s","model":"%s","mnbt":%s,"sha":"%s","cards":"%s"' \
               "$arm" "$pass" "$flags" "$M" "$MNBT" \
               "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)" "$(card)"
        printf ',"c1":{"seconds":%s,"tokens":%s,"first_ms":%s,"p99_chunk_ms":%s,"chunks":%s}' \
               "$c1_w" "$c1_tok" "$c1_first" "$c1_p99" "$c1_chunks"
        printf ',"cN":{"streams":%s,"seconds":%s,"tokens":%s}' "$C" "$cN_w" "$cN_tok"
        printf ',"prefill":{"seconds":%s,"prompt_tokens":%s,"ttft_ms":%s}' "$pf_w" "$pf_tok" "$ttft"
        printf ',"counters":{'
        awk '{ printf "%s\"%s\":%s", (NR > 1 ? "," : ""), $1, $2 }' "$W/$arm-$pass.d"
        printf '},"contention":"%s"}\n' "$cont"
    } >> "$REPORT" && echo "  sealed       -> $REPORT"
}

echo "arms: $ARMS    passes: $PASSES    prompt $PFN tok    decoders $DEC x C$C    report $REPORT"
: > "$REPORT" || { echo "cannot write $REPORT" >&2; exit 1; }
pass=1
while [ "$pass" -le "$PASSES" ]; do
    # Alternate the order by pass, so no arm is always measured at the same point of the card's
    # thermal cycle.
    if [ $((pass % 2)) -eq 0 ]; then
        order=$(printf '%s\n' $ARMS | awk '{ a[NR] = $0 } END { for (i = NR; i >= 1; i--) print a[i] }' |
                tr '\n' ' ')
    else
        order="$ARMS"
    fi
    for arm in $order; do run_arm "$arm" "$pass" || exit 1; done
    pass=$((pass + 1))
done

echo
echo "Compare the arms on decode C1 and C$C first, then prefill, then acceptance: acceptance moving"
echo "means the arms generated different text, and their throughput is not comparable. What each"
echo "outcome means is docs/PREFILL-OVERLAP.md. scripts/ident.sh gives the noise floor of one"
echo "configuration, which is what any difference between arms has to clear."
