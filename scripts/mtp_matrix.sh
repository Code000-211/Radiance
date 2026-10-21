#!/usr/bin/env bash
# The adaptive-MTP benchmark matrix: one immutable manifest in, raw rows and a comparison out.
#
#   scripts/mtp_matrix.sh --check <manifest>    validate and print the plan, start nothing
#   scripts/mtp_matrix.sh <manifest>            run it, under the manifest's exclusive lock
#
# Method (docs/ADAPTIVE-MTP.md, "Benchmark manifest, version 1"). Per arm the script starts its
# own server on its own port, sends the workload as C concurrent streaming requests, and waits
# until exactly C are running before opening the measurement window -- cstep2.sh's rule, since a
# window with fewer than C at either edge measures no single concurrency. The global counters are
# differenced across the window for throughput and acceptance. Every request also timestamps its
# own chunks, so TTFT, TPOT and the gap distribution come from real streaming arrivals instead of
# a token count divided by a window.
#
# A chunk is one arrival, never one arrival per token it carries: a chunk holding two tokens
# arrives once, and counting it twice would invent a latency sample. Chunks with an empty text are
# not arrivals at all. The chunk count and the server's own usage count are both recorded, so a
# stream that batches tokens shows up in the raw row instead of silently redefining "per token".
#
# TTFT here is send-to-first-chunk under load, so it includes queueing at concurrency C and is not
# a cold-start figure. TPOT is per request over its in-window arrivals; the gap percentiles pool
# the chunk-to-chunk intervals. Both are reported because they answer different questions.
#
# Only the server this script started is stopped: its PID is recorded and reaped, nothing else on
# the box is touched. The whole run holds the manifest's lock, so a second matrix cannot measure
# beside it. A lock someone else holds is reported after 60 s, not waited out -- a long wait puts
# another run's start inside someone's window.
#
# The manifest is read once at startup by python3's standard library; jq is not assumed, and the
# repo's scripts already depend on python3. Nothing on the measurement path runs in Python:
# aggregation is awk, sort and shell.
#
# Workload files hold one prompt per line and must hold at least as many prompts as the largest
# concurrency, so no two concurrent requests share a prompt. Relative prompt and output paths
# resolve against the manifest's directory.
#
# Environment:
#   RADIANCE=   the server binary (default: $BUILD/bin/radiance)
#   BUILD=      the build directory (default: build/ beside the checkout)
#   PORT=       the first port to use (default: 8120); each arm takes the next one
#   WIN=        measurement window in seconds (default: 6)
#   MT=         max_tokens per request (default: 900)
set -u

SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
HERE=$(dirname "$SELF")
BUILD=${BUILD:-$HERE/../build}
RADIANCE_BIN=${RADIANCE:-$BUILD/bin/radiance}
PORT=${PORT:-8120}
WIN=${WIN:-6}
MT=${MT:-900}

die() { printf 'mtp_matrix: %s\n' "$*" >&2; exit 2; }

case ${1:-} in
  -h|--help) sed -n '2,41p' "$SELF" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

CHECK=0
SELFTEST=0
[ "${1:-}" = "--check" ] && { CHECK=1; shift; }
[ "${1:-}" = "--self-test" ] && { SELFTEST=1; shift; }
[ "${1:-}" = "--inside-lock" ] && shift   # the child re-exec'd under flock
M=${1:-}
if [ "$SELFTEST" = 0 ]; then
  [ -n "$M" ] && [ -f "$M" ] || die "usage: $0 [--check|--self-test] <manifest.json>"
  command -v python3 >/dev/null || die "python3 is needed to read the manifest"
fi

PLAN=$(mktemp -d "${TMPDIR:-/tmp}/mtp-plan.XXXXXX") || die "no temp dir"
trap 'rm -rf "$PLAN"' EXIT

# ------------------------------------------------------------------ aggregation (no server needed)
# Split out from the run loop so it can be checked against synthetic arrivals, which is the only
# part of this script that can be verified without a card.

# Nearest-rank percentile of one number per line; 0 for an empty file.
pct() { sort -n "$1" 2>/dev/null | awk -v p="$2" '{a[NR]=$1} END{if(!NR){print 0;exit} i=int(NR*p+0.999999999); if(i<1)i=1; print a[i]}'; }

# One request log -> its TTFT, its TPOT, its chunk gaps. A log holds "send <us>", then one line per
# content chunk arrival (<us>), then "usage <n>". Only arrivals inside [t0, t1] are latency samples;
# the first arrival may predate t0, because TTFT runs from the send.
aggregate() {   # $1 t0, $2 t1, $3 out dir, then the logs; prints "n_req n_chunks usage"
  local t0=$1 t1=$2 dir=$3; shift 3
  : > "$dir/ttft.txt"; : > "$dir/tpot.txt"; : > "$dir/gaps.txt"
  local f m n u n_req=0 n_chunks=0 usage=0
  for f in "$@"; do
    read -r m n u < <(awk -v t0="$t0" -v t1="$t1" -v ff="$dir/ttft.txt" -v tf="$dir/tpot.txt" \
                          -v gf="$dir/gaps.txt" '
      $1 == "send"  { send = $2; next }
      $1 == "usage" { u = $2; next }
      {
        n++
        if (first == "") first = $1
        if ($1 >= t0 && $1 <= t1) {
          if (m) print ($1 - prev) / 1000.0 >> gf; else win_first = $1
          prev = $1; lastwin = $1; m++
        }
      }
      END {
        if (send != "" && first != "") print (first - send) / 1000.0 >> ff
        if (m >= 2) print (lastwin - win_first) / 1000.0 / (m - 1) >> tf
        printf "%d %d %d\n", m + 0, n + 0, u + 0
      }' "$f")
    n_req=$(( n_req + (m > 0 ? 1 : 0) ))
    n_chunks=$(( n_chunks + n ))
    usage=$(( usage + u ))
  done
  printf '%s %s %s\n' "$n_req" "$n_chunks" "$usage"
}

selftest() {
  local d; d=$(mktemp -d) || return 1
  # A: send at 0, arrivals at 10/20/30 ms, 3 tokens. B: send at 0, one arrival before the window at
  # 5 ms, then 15/25/40 ms. Window 8..40 ms, so B keeps three of its four arrivals.
  printf 'send\t0\n10000\n20000\n30000\nusage\t3\n'       > "$d/0.tsv"
  printf 'send\t0\n5000\n15000\n25000\n40000\nusage\t4\n' > "$d/1.tsv"
  local n_req n_chunks usage bad=0
  read -r n_req n_chunks usage < <(aggregate 8000 40000 "$d" "$d/0.tsv" "$d/1.tsv")
  chk() { if [ "$2" = "$3" ]; then printf '  ok   %s\n' "$1"
          else printf '  FAIL %s: %s, want %s\n' "$1" "$2" "$3"; bad=1; fi; }
  chk "both requests counted"       "$n_req" 2
  chk "content chunks counted"      "$n_chunks" 7
  chk "usage summed"                "$usage" 7
  chk "TTFT p50"                    "$(pct "$d/ttft.txt" 0.50)" 5
  chk "TPOT p50 (A 10, B 12.5)"     "$(pct "$d/tpot.txt" 0.50)" 10
  chk "gap p50"                     "$(pct "$d/gaps.txt" 0.50)" 10
  chk "gap p95, nearest rank"       "$(pct "$d/gaps.txt" 0.95)" 15
  rm -rf "$d"
  return "$bad"
}

if [ "$SELFTEST" = 1 ]; then
  selftest || die "aggregation self-test failed"
  printf 'aggregation self-test: 7 check(s) ok; no server started\n'
  exit 0
fi

# ------------------------------------------------------------------ read and validate the manifest
# Deliberately strict: an unknown key is refused, so a typo in an immutable manifest cannot
# silently measure something other than what it names.
python3 - "$M" "$PLAN" <<'PY' || exit $?
import json, os, shlex, sys

path, out = sys.argv[1], sys.argv[2]

def bad(msg):
    print("mtp_matrix: manifest: " + msg, file=sys.stderr)
    sys.exit(2)

try:
    with open(path) as f:
        m = json.load(f)
except (OSError, ValueError) as e:
    bad("%s: %s" % (path, e))

if not isinstance(m, dict):
    bad("the top level must be an object")
if m.get("format") != "rad-bench-manifest/1":
    bad("format is %r, expected 'rad-bench-manifest/1'" % (m.get("format"),))

known = {"format", "id", "lock", "model", "server_args", "arms", "workloads",
         "concurrency", "repetitions", "order", "warmup_requests", "outputs", "notes"}
unknown = sorted(set(m) - known)
if unknown:
    bad("unknown key(s): " + ", ".join(unknown))
for key in ("id", "lock", "model", "arms", "workloads", "concurrency", "outputs"):
    if key not in m:
        bad("missing '%s'" % key)

def text(v, where, absolute=False):
    if not isinstance(v, str) or not v or "\n" in v:
        bad("%s must be a non-empty single-line string" % where)
    if absolute and not v.startswith("/"):
        bad("%s must be an absolute path" % where)
    return v

mid = text(m["id"], "'id'")
lock = text(m["lock"], "'lock'", absolute=True)
outputs = text(m["outputs"], "'outputs'")
model = m["model"]
if not isinstance(model, dict):
    bad("'model' must be an object")
extra = sorted(set(model) - {"path", "tokenizer", "revision"})
if extra:
    bad("'model': unknown key(s) " + ", ".join(extra))
mpath = text(model.get("path"), "'model.path'")

def arglist(v, where):
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        bad("%s must be a list of strings" % where)
    if any("\0" in x for x in v):
        bad("%s contains a NUL" % where)
    return v

def write_args(p, args):
    # NUL-separated, so an argument holding a space survives; an empty file means no arguments.
    with open(p, "wb") as f:
        if args:
            f.write(b"\0".join(x.encode() for x in args) + b"\0")

server_args = arglist(m.get("server_args", []), "'server_args'")
write_args(os.path.join(out, "server_args"), server_args)

arms = m["arms"]
if not isinstance(arms, list) or not arms:
    bad("'arms' must be a non-empty list")
os.makedirs(os.path.join(out, "arms"))
seen, enabled = set(), []
for i, a in enumerate(arms):
    if not isinstance(a, dict):
        bad("arms[%d] must be an object" % i)
    extra = sorted(set(a) - {"name", "args", "enabled"})
    if extra:
        bad("arms[%d]: unknown key(s) %s" % (i, ", ".join(extra)))
    name = a.get("name")
    if not isinstance(name, str) or not name or "/" in name or "\n" in name:
        bad("arms[%d]: 'name' must be a non-empty string without '/'" % i)
    if name in seen:
        bad("arm %r is named twice" % name)
    seen.add(name)
    args = arglist(a.get("args", []), "arms[%d].args" % i)
    on = a.get("enabled", True)
    if not isinstance(on, bool):
        bad("arms[%d].enabled must be a boolean" % i)
    write_args(os.path.join(out, "arms", name + ".args"), args)
    if on:
        enabled.append(name)
if not enabled:
    bad("no arm is enabled")

conc = m["concurrency"]
if not isinstance(conc, list) or not conc:
    bad("'concurrency' must be a non-empty list")
for c in conc:
    if not isinstance(c, int) or isinstance(c, bool) or c <= 0:
        bad("'concurrency' entries must be positive integers")
cmax = max(conc)
with open(os.path.join(out, "concurrency"), "w") as f:
    f.write("\n".join(str(c) for c in conc) + "\n")

base = os.path.dirname(os.path.abspath(path))
workloads = m["workloads"]
if not isinstance(workloads, list) or not workloads:
    bad("'workloads' must be a non-empty list")
os.makedirs(os.path.join(out, "workloads"))
names = []
for i, w in enumerate(workloads):
    if not isinstance(w, dict):
        bad("workloads[%d] must be an object" % i)
    extra = sorted(set(w) - {"name", "prompts"})
    if extra:
        bad("workloads[%d]: unknown key(s) %s" % (i, ", ".join(extra)))
    name = text(w.get("name"), "workloads[%d].name" % i)
    if "/" in name:
        bad("workloads[%d].name must not contain '/'" % i)
    p = text(w.get("prompts"), "workloads[%d].prompts" % i)
    full = p if os.path.isabs(p) else os.path.join(base, p)
    if not os.path.isfile(full):
        bad("workload %r: prompt file not found: %s" % (name, full))
    with open(full) as f:
        n = sum(1 for line in f if line.strip())
    if n < cmax:
        bad("workload %r holds %d prompt(s); concurrency %d needs at least that many distinct ones"
            % (name, n, cmax))
    with open(os.path.join(out, "workloads", name), "w") as f:
        f.write(os.path.abspath(full) + "\n")
    names.append(name)

reps = m.get("repetitions", 5)
if not isinstance(reps, int) or isinstance(reps, bool) or reps < 1:
    bad("'repetitions' must be a positive integer")
order = m.get("order", "alternate")
if order not in ("alternate", "random"):
    bad("'order' must be 'alternate' or 'random'")
warm = m.get("warmup_requests", 0)
if not isinstance(warm, int) or isinstance(warm, bool) or warm < 0:
    bad("'warmup_requests' must be a non-negative integer")

with open(os.path.join(out, "meta"), "w") as f:
    for k, v in (("id", mid), ("lock", lock), ("model", os.path.abspath(mpath)),
                 ("outputs", outputs if os.path.isabs(outputs) else os.path.join(base, outputs)),
                 ("base", base), ("order", order), ("reps", str(reps)), ("warmup", str(warm))):
        f.write("%s=%s\n" % (k, shlex.quote(v)))
with open(os.path.join(out, "arms", ".enabled"), "w") as f:
    f.write("\n".join(enabled) + "\n")
with open(os.path.join(out, "workloads", ".names"), "w") as f:
    f.write("\n".join(names) + "\n")
PY

. "$PLAN/meta"
mapfile -d '' -t SERVER_ARGS < "$PLAN/server_args"
mapfile -t ENABLED < "$PLAN/arms/.enabled"
mapfile -t CONC < "$PLAN/concurrency"
mapfile -t WORKLOADS < "$PLAN/workloads/.names"

arm_args() { mapfile -d '' -t ARM_ARGS < "$PLAN/arms/$1.args"; }

if [ "$CHECK" = 1 ]; then
  [ -x "$RADIANCE_BIN" ] || printf 'note: server binary is not executable: %s\n' "$RADIANCE_BIN"
  printf 'manifest %s  lock %s  model %s\n' "$id" "$lock" "$model"
  if [ "${#SERVER_ARGS[@]}" -gt 0 ]; then printf 'server args: %s\n' "${SERVER_ARGS[*]}"; fi
  printf 'arms (%d enabled):\n' "${#ENABLED[@]}"
  for a in "${ENABLED[@]}"; do
    arm_args "$a"; printf '  %-12s %s\n' "$a" "${ARM_ARGS[*]-}"
  done
  printf 'workloads:\n'
  for w in "${WORKLOADS[@]}"; do
    wf=$(cat "$PLAN/workloads/$w")
    printf '  %-12s %s (%s prompts)\n' "$w" "$wf" "$(awk 'NF{n++} END{print n+0}' "$wf")"
  done
  printf 'concurrency: %s\n' "${CONC[*]}"
  printf 'repetitions %s, order %s, warmup %s, outputs %s\n' "$reps" "$order" "$warmup" "$outputs"
  printf 'plan valid; nothing started\n'
  exit 0
fi

[ -x "$RADIANCE_BIN" ] || die "server binary not found or not executable: $RADIANCE_BIN"

# Hold the GPU pair for the whole matrix. The child does the work; the parent waits and can report
# a refused lock, which an `exec` could not.
if [ -z "${MTP_MATRIX_LOCKED:-}" ]; then
  command -v flock >/dev/null || die "flock is needed for the GPU-pair lock"
  MTP_MATRIX_LOCKED=1 flock -x -w 60 "$lock" "$SELF" --inside-lock "$M"
  rc=$?
  [ "$rc" -eq 0 ] || die "lock $lock is held by another run (flock exit $rc)"
  exit 0
fi

# ------------------------------------------------------------------ measurement

U=
SRV_PID=
RUNDIR=

stop_server() {
  [ -n "${SRV_PID:-}" ] || return 0
  if kill -0 "$SRV_PID" 2>/dev/null; then
    kill -TERM "$SRV_PID" 2>/dev/null
    local i=0
    while kill -0 "$SRV_PID" 2>/dev/null && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done
    kill -KILL "$SRV_PID" 2>/dev/null
  fi
  wait "$SRV_PID" 2>/dev/null
  SRV_PID=
}
trap 'stop_server; rm -rf "$PLAN"' EXIT
trap 'exit 130' INT TERM

start_server() {   # $1 arm, $2 port
  local log="$RUNDIR/server.log"
  "$RADIANCE_BIN" --model "$model" --port "$2" "${SERVER_ARGS[@]}" "${ARM_ARGS[@]}" \
      > "$log" 2>&1 &
  SRV_PID=$!
  local i=0
  while [ $i -lt 1200 ]; do          # 120 s for a cold container
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      printf 'mtp_matrix: arm %s: server died during startup; tail of %s:\n' "$1" "$log" >&2
      tail -20 "$log" >&2
      return 1
    fi
    if curl -sf -m 2 -o /dev/null "$U/health"; then return 0; fi
    sleep 0.1; i=$((i+1))
  done
  printf 'mtp_matrix: arm %s: no /health within 120 s; see %s\n' "$1" "$log" >&2
  return 1
}

metric() { curl -s -m 10 "$U/metrics" | grep -F "$1" | grep -v '^#' | awk '{print $NF}' | head -1; }

now_us() { local t=${EPOCHREALTIME/[.,]/}; printf '%s' "$t"; }

json_escape() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

# One streaming request. Records "send <us>", then one line per content chunk arrival, then
# "usage <n>" from the final accounting chunk.
stream_one() {   # $1 payload, $2 out file
  printf 'send\t%s\n' "$(now_us)" > "$2"
  local body
  while IFS= read -r line; do
    case $line in
      'data: '*)
        body=${line#data: }
        [ "$body" = "[DONE]" ] && break
        case $body in
          *'"completion_tokens":'*)
            printf 'usage\t%s\n' \
              "$(printf '%s' "$body" | sed 's/.*"completion_tokens":\([0-9]*\).*/\1/')" >> "$2" ;;
          *'"text":""'*) ;;                                  # an empty delta is not an arrival
          *'"text":"'*) printf '%s\n' "$(now_us)" >> "$2" ;;
        esac
        ;;
    esac
  done < <(curl -sN -m 300 -H 'Content-Type: application/json' --data-binary "@$1" \
                "$U/v1/completions")
}

write_payload() {   # $1 prompt, $2 file
  printf '{"model":"m","prompt":"%s","max_tokens":%d,"temperature":0,"ignore_eos":true,"stream":true,"stream_options":{"include_usage":true}}\n' \
    "$(json_escape "$1")" "$MT" > "$2"
}

# Nearest-rank percentile of one number per line; 0 for an empty file.

run_window() {   # $1 arm $2 workload $3 conc $4 rep $5 dir
  local arm=$1 wl=$2 c=$3 rep=$4 dir=$5
  local prompts=() pids=() files=() i
  mapfile -t prompts < "$PLAN/workloads/$wl"
  mkdir -p "$dir"
  for ((i = 0; i < c; i++)); do
    write_payload "${prompts[$i]}" "$dir/$i.json"
    stream_one "$dir/$i.json" "$dir/$i.tsv" &
    pids+=($!); files+=("$dir/$i.tsv")
  done

  # Ramp, then exactly C at both edges of the window.
  local j=0 at_c=1
  while [ $j -lt 240 ]; do
    [ "$(metric 'vllm:num_requests_running')" = "$c" ] && break
    j=$((j+1)); sleep 0.5
  done
  sleep 3                                     # let the ramp's ragged edge pass
  [ "$(metric 'vllm:num_requests_running')" = "$c" ] || at_c=0

  local s0 d0 r0 a0 p0 t0 s1 d1 r1 a1 p1 t1 run
  s0=$(metric 'radiance:engine_steps_decode_total'); d0=$(metric 'radiance:decode_tokens_total')
  r0=$(metric 'radiance:draft_tokens_total');        a0=$(metric 'radiance:draft_accepted_total')
  p0=$(metric 'radiance:prefill_tokens_total');      t0=$(now_us)
  sleep "$WIN"
  s1=$(metric 'radiance:engine_steps_decode_total'); d1=$(metric 'radiance:decode_tokens_total')
  r1=$(metric 'radiance:draft_tokens_total');        a1=$(metric 'radiance:draft_accepted_total')
  p1=$(metric 'radiance:prefill_tokens_total');      t1=$(now_us)
  run=$(metric 'vllm:num_requests_running')

  # Per request latency from the clients' own arrivals: TTFT from send to first chunk (so
  # pre-window queueing is included), TPOT over in-window arrivals, pooled chunk-to-chunk gaps.
  local n_req n_chunks usage
  read -r n_req n_chunks usage < <(aggregate "$t0" "$t1" "$dir" "${files[@]}")

  local dt steps tok draft acc pre
  dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.6f", (b - a) / 1000000}')
  steps=$(( ${s1:-0} - ${s0:-0} ));  tok=$(( ${d1:-0} - ${d0:-0} ))
  draft=$(( ${r1:-0} - ${r0:-0} ));  acc=$(( ${a1:-0} - ${a0:-0} ))
  pre=$(( ${p1:-0} - ${p0:-0} ))

  awk -v arm="$arm" -v wl="$wl" -v c="$c" -v rep="$rep" -v dt="$dt" -v steps="$steps" \
      -v tok="$tok" -v draft="$draft" -v acc="$acc" -v pre="$pre" -v at_c="$at_c" \
      -v run="${run:-0}" -v nreq="$n_req" -v nchunk="$n_chunks" -v usage="$usage" \
      -v ttft="$(pct "$dir/ttft.txt" 0.50)" -v tpot50="$(pct "$dir/tpot.txt" 0.50)" \
      -v tpot95="$(pct "$dir/tpot.txt" 0.95)" -v gap50="$(pct "$dir/gaps.txt" 0.50)" \
      -v gap95="$(pct "$dir/gaps.txt" 0.95)" \
      'BEGIN {
        printf "%s %s %s %s %.3f %d %.2f %.3f %.1f %.1f %d %d %d %.2f %.2f %.2f %.2f %.2f %s\n",
          arm, wl, c, rep, dt, steps, dt > 0 ? tok / dt : 0, steps > 0 ? tok / steps : 0,
          draft > 0 ? 100 * acc / draft : 0, dt > 0 ? pre / dt : 0,
          nreq, nchunk, usage, ttft, tpot50, tpot95, gap50, gap95, at_c
      }' >> "$OUT"

  wait "${pids[@]}" 2>/dev/null
}

run_arm() {   # $1 arm, $2 port
  arm_args "$1"
  RUNDIR="$OUTDIR/$1"
  mkdir -p "$RUNDIR"
  U="http://127.0.0.1:$2"
  start_server "$1" "$2" || return 1
  if [ "$warmup" -gt 0 ]; then
    write_payload "warmup" "$RUNDIR/warmup.json"
    local i=0
    while [ $i -lt "$warmup" ]; do
      curl -s -m 300 -o /dev/null -H 'Content-Type: application/json' \
           --data-binary "@$RUNDIR/warmup.json" "$U/v1/completions"
      i=$((i+1))
    done
  fi
  local wl c rep
  for wl in "${WORKLOADS[@]}"; do
    for c in "${CONC[@]}"; do
      for ((rep = 1; rep <= reps; rep++)); do
        run_window "$1" "$wl" "$c" "$rep" "$RUNDIR/$wl-c$c-r$rep"
      done
    done
  done
  stop_server
}

OUTDIR="$outputs/$id"
mkdir -p "$OUTDIR" || die "cannot create $OUTDIR"
cp "$M" "$OUTDIR/manifest.json"       # the immutable record this directory belongs to
OUT="$OUTDIR/rows.txt"
: > "$OUT"

PORTS=(); port=$PORT
for a in "${ENABLED[@]}"; do PORTS+=("$port"); port=$((port+1)); done

for ((rep = 1; rep <= reps; rep++)); do
  idx=("${!ENABLED[@]}")
  if [ "$order" = random ]; then
    RANDOM=$((rep * 7919))               # same manifest, same order on a rerun
    for ((i = ${#idx[@]} - 1; i > 0; i--)); do
      j=$((RANDOM % (i + 1)))
      tmp=${idx[i]}; idx[i]=${idx[j]}; idx[j]=$tmp
    done
  fi
  for i in "${idx[@]}"; do
    a=${ENABLED[$i]}
    run_arm "$a" "${PORTS[$i]}" || die "arm $a failed"
  done
done

# Median over repetitions per (arm, workload, C). The manifest asks for paired repetitions, so one
# row is a run, not a result.
awk '
  { k = $1 "\t" $2 "\t" $3
    n[k]++; t[k] = t[k] " " $7; s[k] = s[k] " " $8; a[k] = a[k] " " $9; g[k] = g[k] " " $18 }
  END {
    for (k in n) {
      split(k, f, "\t")
      printf "%-12s %-10s %4s %9.2f %9.3f %8.1f %8.2f %6d\n", f[1], f[2], f[3],
             med(t[k]), med(s[k]), med(a[k]), med(g[k]), n[k]
    }
  }
  function med(list,   v, m, i, j, x) {
    m = split(list, v, " ")
    for (i = 2; i <= m; i++) { x = v[i] + 0; j = i - 1
      while (j >= 1 && v[j] + 0 > x) { v[j+1] = v[j]; j-- } v[j+1] = x }
    return m % 2 ? v[(m+1)/2] : (v[m/2] + v[m/2+1]) / 2
  }' "$OUT" | sort -k1,1 -k2,2 -k3,3n > "$OUTDIR/summary.body"
printf '%-12s %-10s %4s %9s %9s %8s %8s %6s\n' arm workload C tok/s tok/step accept% gap95ms runs \
  > "$OUTDIR/summary.txt"
cat "$OUTDIR/summary.body" >> "$OUTDIR/summary.txt"

cat "$OUTDIR/summary.txt"
awk '$19 != 1 || $6 <= 0 { bad++ } END { if (bad) printf "warning: %d row(s) held no C or saw no decode steps; see %s\n", bad, FILENAME }' \
    "$OUT" || true
printf 'raw rows: %s\n' "$OUT"
