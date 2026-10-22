#!/usr/bin/env bash
# The adaptive-MTP benchmark matrix: one immutable manifest in, raw rows and a comparison out.
#
#   scripts/mtp_matrix.sh --check <manifest>    validate and print the plan, start nothing
#   scripts/mtp_matrix.sh <manifest>            run it, under the shared GPU-pair lock
#   scripts/mtp_matrix.sh --self-test           aggregation against synthetic logs, starts nothing
#   scripts/mtp_matrix.sh --mock-test           the runner's own paths against a mock server
#
# Method (docs/ADAPTIVE-MTP.md, "Benchmark manifest, version 1"). Per arm the script starts its
# own server on its own port, sends the workload as C concurrent streaming requests, and waits
# until exactly C are running before opening the measurement window -- cstep2.sh's rule, since a
# window with fewer than C at either edge measures no single concurrency. Both edges are checked
# and both are recorded in the row: the opening one before t0, the closing one after t1.
#
# The window is [t0, t1] and nothing else. Counters are read at both ends and differenced for
# throughput and acceptance. Every client records its own chunk arrival times; the streams are
# reaped completely before anything is aggregated, so a log is never read while it is still being
# written.
#
# A chunk is one arrival, never one arrival per token it carries, so the arrival intervals are
# reported as CHUNK GAPS (chunkgap50/95), not as a per-token latency. A token-arrival latency
# cannot be built from them: the server emits one text per generation step and may hold text back,
# so tokens per chunk is about 1 + the acceptance rate. `tok/chunk` is printed beside the gaps to
# make that ratio visible, and `usage_total` is the client-side usage count of the WHOLE stream --
# it is not the token count inside the window and is not used as one. The in-window token count is
# the decode counter's difference.
#
# A row is usable only when every client exited 0 with HTTP 200, both edges saw exactly C running,
# every client contributed an in-window arrival, and decode steps were counted. Rows that fail any
# of that are written with usable=0, kept in the raw file, excluded from the summary, and the run
# exits nonzero: a broken window is not a measurement.
#
# The lock is the agreed path, /tmp/radiance-tp2-bench.lock, not a manifest choice; a manifest
# naming any other path is refused, so a typo cannot measure beside a TP2 bench. A held lock is
# refused at once (`flock -n`), never waited out -- a wait puts another run's start inside
# someone's window.
#
# Only the server this script started is stopped: its PID is recorded and reaped, nothing else on
# the box is touched. A signal on the outer wrapper stops the child it started, and that child's
# own EXIT trap stops its server.
#
# The manifest is read once at startup by python3's standard library; jq is not assumed, and the
# repo's scripts already depend on python3. Nothing on the measurement path runs in Python:
# aggregation is awk, sort and shell.
#
# Workload files hold one prompt per line; their text is what is sent, and they must hold at least
# as many prompts as the largest concurrency, so no two concurrent requests share a prompt.
# Relative prompt and output paths resolve against the manifest's directory.
#
# Environment:
#   RADIANCE=   the server binary (default: $BUILD/bin/radiance)
#   BUILD=      the build directory (default: build/ beside the checkout)
#   PORT=       the first port to use (default: 8120); each arm takes the next one
#   WIN=        measurement window in seconds (default: 6)
#   MT=         max_tokens per request (default: 900)
#   RAMP=       ramp polls at 0.5 s before giving up on C running (default: 240)
#   SETTLE=     seconds between reaching C and the opening edge (default: 3)
set -u

SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
HERE=$(dirname "$SELF")
BUILD=${BUILD:-$HERE/../build}
RADIANCE_BIN=${RADIANCE:-$BUILD/bin/radiance}
PORT=${PORT:-8120}
WIN=${WIN:-6}
MT=${MT:-900}
RAMP=${RAMP:-240}
SETTLE=${SETTLE:-3}
LOCK=/tmp/radiance-tp2-bench.lock

die() { printf 'mtp_matrix: %s\n' "$*" >&2; exit 2; }

case ${1:-} in
  -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$SELF"; exit 0 ;;
esac

CHECK=0
SELFTEST=0
MOCK=0
[ "${1:-}" = "--check" ] && { CHECK=1; shift; }
[ "${1:-}" = "--self-test" ] && { SELFTEST=1; shift; }
[ "${1:-}" = "--mock-test" ] && { MOCK=1; shift; }
[ "${1:-}" = "--inside-lock" ] && shift   # the child started under the held lock
M=${1:-}
if [ "$SELFTEST" = 0 ] && [ "$MOCK" = 0 ]; then
  [ -n "$M" ] && [ -f "$M" ] || die "usage: $0 [--check|--self-test|--mock-test] <manifest.json>"
  command -v python3 >/dev/null || die "python3 is needed to read the manifest"
fi

PLAN=$(mktemp -d "${TMPDIR:-/tmp}/mtp-plan.XXXXXX") || die "no temp dir"
trap 'rm -rf "$PLAN"' EXIT

# ------------------------------------------------------------------ aggregation (no server needed)
# Split out from the run loop so it can be checked against synthetic arrivals, which is the only
# part of this script that can be verified without a card.

# Nearest-rank percentile of one number per line; 0 for an empty file.
pct() { sort -n "$1" 2>/dev/null | awk -v p="$2" '{a[NR]=$1} END{if(!NR){print 0;exit} i=int(NR*p+0.999999999); if(i<1)i=1; print a[i]}'; }

# One request log -> its TTFT and its chunk gaps. A log holds "send <us>", then one line per content
# chunk arrival (<us>), then "usage <n>". Only arrivals inside [t0, t1] are gap samples; the first
# arrival may predate t0, because TTFT runs from the send. Prints "n_req win_chunks chunks usage",
# where win_chunks counts the in-window arrivals and chunks every arrival in the stream.
aggregate() {   # $1 t0, $2 t1, $3 out dir, then the logs
  local t0=$1 t1=$2 dir=$3; shift 3
  : > "$dir/ttft.txt"; : > "$dir/gaps.txt"
  local f m n u n_req=0 win=0 n_chunks=0 usage=0
  for f in "$@"; do
    m=0; n=0; u=0          # an unreadable log must not inherit the previous one's counts
    read -r m n u < <(awk -v t0="$t0" -v t1="$t1" -v ff="$dir/ttft.txt" -v gf="$dir/gaps.txt" '
      $1 == "send"  { send = $2; next }
      $1 == "usage" { u = $2; next }
      {
        n++
        if (first == "") first = $1
        if ($1 >= t0 && $1 <= t1) {
          if (m) print ($1 - prev) / 1000.0 >> gf
          prev = $1; m++
        }
      }
      END {
        if (send != "" && first != "") print (first - send) / 1000.0 >> ff
        printf "%d %d %d\n", m + 0, n + 0, u + 0
      }' "$f")
    n_req=$(( n_req + (m > 0 ? 1 : 0) ))
    win=$(( win + m ))
    n_chunks=$(( n_chunks + n ))
    usage=$(( usage + u ))
  done
  printf '%s %s %s %s\n' "$n_req" "$win" "$n_chunks" "$usage"
}

selftest() {
  local d; d=$(mktemp -d) || return 1
  # A: send at 0, arrivals at 10/20/30 ms, 3 tokens. B: send at 0, one arrival before the window at
  # 5 ms, then 15/25/40 ms. Window 8..40 ms, so B keeps three of its four arrivals.
  printf 'send\t0\n10000\n20000\n30000\nusage\t3\n'       > "$d/0.tsv"
  printf 'send\t0\n5000\n15000\n25000\n40000\nusage\t4\n' > "$d/1.tsv"
  local n_req win n_chunks usage bad=0
  read -r n_req win n_chunks usage < <(aggregate 8000 40000 "$d" "$d/0.tsv" "$d/1.tsv")
  chk() { if [ "$2" = "$3" ]; then printf '  ok   %s\n' "$1"
          else printf '  FAIL %s: %s, want %s\n' "$1" "$2" "$3"; bad=1; fi; }
  chk "both requests counted"          "$n_req" 2
  chk "in-window chunks counted"       "$win" 6
  chk "whole-stream chunks counted"    "$n_chunks" 7
  chk "usage summed over the stream"   "$usage" 7
  chk "TTFT p50"                       "$(pct "$d/ttft.txt" 0.50)" 5
  chk "chunkgap p50"                   "$(pct "$d/gaps.txt" 0.50)" 10
  chk "chunkgap p95, nearest rank"     "$(pct "$d/gaps.txt" 0.95)" 15
  rm -rf "$d"
  return "$bad"
}

if [ "$SELFTEST" = 1 ]; then
  selftest || die "aggregation self-test failed"
  printf 'aggregation self-test: 7 check(s) ok; no server started\n'
  exit 0
fi

# ------------------------------------------------------------------ mock regression cases
# These exercise the runner's own plumbing against a local stand-in server -- the row write, the
# prompt text in the payload, the complete stream capture, client errors, both C edges, the held
# lock and the wrapper's signal cleanup. They need the shared lock free and a card is not needed;
# scripts/mtp_mock_server.py is not a model and measures nothing.
mocktest() {
  local d; d=$(mktemp -d) || return 1
  local bad=0
  chk() { if [ "$2" = "$3" ]; then printf '  ok   %s\n' "$1"
          else printf '  FAIL %s: %s, want %s\n' "$1" "$2" "$3"; bad=1; fi; }
  chk_has() { if printf '%s' "$2" | grep -q -- "$3"; then printf '  ok   %s\n' "$1"
              else printf '  FAIL %s: %s not found in output\n' "$1" "$3"; bad=1; fi; }

  printf 'the first prompt\nthe second prompt\n' > "$d/prompts.txt"
  cat > "$d/manifest.json" <<EOF
{"format": "rad-bench-manifest/1", "id": "mock", "lock": "$LOCK",
 "model": {"path": "mock"}, "server_args": [], "arms": [{"name": "a", "args": []}],
 "workloads": [{"name": "w", "prompts": "prompts.txt"}],
 "concurrency": [2], "repetitions": 1, "outputs": "out"}
EOF

  # A: both clients stream well past the window. Expect one usable row, the full stream captured
  # (600 chunks, usage 600) although only the in-window ones are gap samples, and the prompt text
  # -- not the prompt file's path -- in the payload.
  : > "$d/body.log"
  MOCK_CHUNKS=300 MOCK_GAP_MS=20 MOCK_PIDFILE="$d/pid" MOCK_BODY_LOG="$d/body.log" \
  RADIANCE="$HERE/mtp_mock_server.py" PORT=8120 WIN=1 RAMP=20 SETTLE=1 \
    "$SELF" "$d/manifest.json" > "$d/a.log" 2>&1
  local rc=$?
  local rows="$d/out/mock/rows.txt"
  chk "a: exit status"                "$rc" 0
  chk "a: one data row"               "$(awk 'NR > 1 && NF' "$rows" | wc -l | tr -d ' ')" 1
  chk "a: whole stream captured"      "$(awk 'NR == 2 {print $13}' "$rows")" 600
  chk "a: usage of the whole stream"  "$(awk 'NR == 2 {print $15}' "$rows")" 600
  chk "a: opening edge at C"          "$(awk 'NR == 2 {print $19}' "$rows")" 1
  chk "a: closing edge at C"          "$(awk 'NR == 2 {print $20}' "$rows")" 1
  chk "a: row usable"                 "$(awk 'NR == 2 {print $22}' "$rows")" 1
  chk_has "a: prompt text, not its path" "$(cat "$d/body.log")" 'the first prompt'

  # B: the stream ends while the window is open. The opening edge still sees C, the closing edge
  # sees 0, so the row must be marked unusable and the run must report failure instead of writing
  # a clean-looking measurement of an idle server.
  MOCK_CHUNKS=130 MOCK_GAP_MS=20 RADIANCE="$HERE/mtp_mock_server.py" PORT=8122 WIN=2 RAMP=20 SETTLE=1 \
    "$SELF" "$d/manifest.json" > "$d/b.log" 2>&1
  chk "b: exit status is a failure"   "$?" 1
  chk "b: opening edge at C"          "$(awk 'NR == 2 {print $19}' "$rows")" 1
  chk "b: closing edge recorded 0"    "$(awk 'NR == 2 {print $20}' "$rows")" 0
  chk "b: row unusable"               "$(awk 'NR == 2 {print $22}' "$rows")" 0

  # C: one client gets a 500. The row must be unusable even though the other stream parsed fine.
  MOCK_CHUNKS=300 MOCK_GAP_MS=20 MOCK_FAIL=1 RADIANCE="$HERE/mtp_mock_server.py" PORT=8124 \
    WIN=1 RAMP=6 SETTLE=1 "$SELF" "$d/manifest.json" > "$d/c.log" 2>&1
  chk "c: exit status is a failure"   "$?" 1
  chk "c: row unusable"               "$(awk 'NR == 2 {print $22}' "$rows")" 0

  # D: a held lock is refused at once, not waited out. The holder execs its sleep, so killing the
  # PID this case recorded also releases the lock.
  ( exec 9>>"$LOCK"; flock -x 9; exec sleep 3 ) &
  local holder=$!
  sleep 0.3
  local t0 t1 rc2
  t0=$(date +%s)
  RADIANCE="$HERE/mtp_mock_server.py" PORT=8126 WIN=1 "$SELF" "$d/manifest.json" > "$d/d.log" 2>&1
  rc2=$?
  t1=$(date +%s)
  chk "d: held lock refused"          "$rc2" 2
  chk "d: refused without waiting"    "$(( t1 - t0 < 2 ? 1 : 0 ))" 1
  kill "$holder" 2>/dev/null; wait "$holder" 2>/dev/null

  # E: a signal on the outer wrapper must not leave the server behind. Only the two PIDs this case
  # started are ever signalled.
  MOCK_CHUNKS=300 MOCK_GAP_MS=20 MOCK_PIDFILE="$d/pid2" RADIANCE="$HERE/mtp_mock_server.py" \
    PORT=8128 WIN=1 RAMP=20 SETTLE=1 "$SELF" "$d/manifest.json" > "$d/e.log" 2>&1 &
  local wrapper=$!
  local i=0
  while [ $i -lt 100 ] && [ ! -s "$d/pid2" ]; do sleep 0.1; i=$((i+1)); done
  local srv; srv=$(cat "$d/pid2" 2>/dev/null)
  chk "e: mock server started"        "$([ -n "$srv" ] && echo yes || echo no)" yes
  kill -TERM "$wrapper" 2>/dev/null
  wait "$wrapper" 2>/dev/null
  i=0
  while [ $i -lt 50 ] && kill -0 "$srv" 2>/dev/null; do sleep 0.1; i=$((i+1)); done
  chk "e: mock server stopped"        "$(kill -0 "$srv" 2>/dev/null && echo alive || echo gone)" gone

  rm -rf "$d"
  [ "$bad" = 0 ] && printf 'mock regression: all checks ok\n'
  return "$bad"
}

if [ "$MOCK" = 1 ]; then
  mocktest || die "mock regression failed"
  exit 0
fi

# ------------------------------------------------------------------ read and validate the manifest
# Deliberately strict: an unknown key is refused, so a typo in an immutable manifest cannot
# silently measure something other than what it names.
python3 - "$M" "$PLAN" "$LOCK" <<'PY' || exit $?
import json, os, shlex, sys

path, out, lock_path = sys.argv[1], sys.argv[2], sys.argv[3]

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
if os.path.realpath(lock) != os.path.realpath(lock_path):
    bad("lock is %r; the GPU-pair lock is fixed at %s and is not a manifest choice"
        % (lock, lock_path))
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
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if len(lines) < cmax:
        bad("workload %r holds %d prompt(s); concurrency %d needs at least that many distinct ones"
            % (name, len(lines), cmax))
    # The prompts themselves are what a request sends; the source path is kept only for --check.
    with open(os.path.join(out, "workloads", name), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "workloads", name + ".src"), "w") as f:
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
    printf '  %-12s %s (%s prompts)\n' "$w" "$(cat "$PLAN/workloads/$w.src")" \
      "$(awk 'NF{n++} END{print n+0}' "$PLAN/workloads/$w")"
  done
  printf 'concurrency: %s\n' "${CONC[*]}"
  printf 'repetitions %s, order %s, warmup %s, outputs %s\n' "$reps" "$order" "$warmup" "$outputs"
  printf 'plan valid; nothing started\n'
  exit 0
fi

[ -x "$RADIANCE_BIN" ] || die "server binary not found or not executable: $RADIANCE_BIN"

# The lock is a fixed path, not a manifest choice: a typo must not let this run measure beside a
# TP2 bench. The child does the work; the wrapper waits so a held lock can be reported at once.
if [ -z "${MTP_MATRIX_LOCKED:-}" ]; then
  command -v flock >/dev/null || die "flock is needed for the GPU-pair lock"
  exec 9>>"$LOCK" || die "cannot open the lock file: $LOCK"
  flock -n 9 || die "lock $LOCK is held by another run; not waiting it out"
  MTP_MATRIX_LOCKED=1 "$SELF" --inside-lock "$M" &
  CHILD=$!
  # A signal on this wrapper must not leave the measuring child behind. Only $CHILD is stopped,
  # and that child's own EXIT trap stops the server it started.
  trap 'kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null; exit 130' INT TERM
  wait "$CHILD"; rc=$?
  trap - INT TERM
  exit "$rc"
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
# "usage <n>" from the final accounting chunk, and "rc http" in a status file. The headers go to
# a file of their own so an HTTP error is visible and the body stream stays pure SSE. The loop runs
# to EOF rather than stopping at [DONE], so no chunk after the last one is lost and the status file
# is always complete by the time it is read.
stream_one() {   # $1 payload, $2 out file
  printf 'send\t%s\n' "$(now_us)" > "$2"
  local body
  while IFS= read -r line; do
    case $line in
      'data: '*)
        body=${line#data: }
        case $body in
          *'"completion_tokens":'*)
            printf 'usage\t%s\n' \
              "$(printf '%s' "$body" | sed 's/.*"completion_tokens":\([0-9]*\).*/\1/')" >> "$2" ;;
          *'"text":""'*) ;;                                  # an empty delta is not an arrival
          *'"text":"'*) printf '%s\n' "$(now_us)" >> "$2" ;;
        esac
        ;;
    esac
  done < <(curl -sN -m 300 -D "$2.hdr" -H 'Content-Type: application/json' \
                --data-binary "@$1" "$U/v1/completions"
           printf '%s\n' "$?" > "$2.rc")
  local rc http
  rc=$(cat "$2.rc" 2>/dev/null || printf '99')
  http=$(sed -n '1s/^[^ ]*[ ]*\([0-9][0-9]*\).*/\1/p' "$2.hdr" 2>/dev/null)
  printf '%s %s\n' "${rc:-99}" "${http:-0}" > "$2.status"
  [ "${rc:-99}" = 0 ] && [ "${http:-0}" = 200 ]
}

write_payload() {   # $1 prompt, $2 file
  printf '{"model":"m","prompt":"%s","max_tokens":%d,"temperature":0,"ignore_eos":true,"stream":true,"stream_options":{"include_usage":true}}\n' \
    "$(json_escape "$1")" "$MT" > "$2"
}

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
  local j=0 c_open=1
  while [ $j -lt "$RAMP" ]; do
    [ "$(metric 'vllm:num_requests_running')" = "$c" ] && break
    j=$((j+1)); sleep 0.5
  done
  sleep "$SETTLE"                             # let the ramp's ragged edge pass
  [ "$(metric 'vllm:num_requests_running')" = "$c" ] || c_open=0

  local s0 d0 r0 a0 p0 t0 s1 d1 r1 a1 p1 t1 run
  s0=$(metric 'radiance:engine_steps_decode_total'); d0=$(metric 'radiance:decode_tokens_total')
  r0=$(metric 'radiance:draft_tokens_total');        a0=$(metric 'radiance:draft_accepted_total')
  p0=$(metric 'radiance:prefill_tokens_total');      t0=$(now_us)
  sleep "$WIN"
  s1=$(metric 'radiance:engine_steps_decode_total'); d1=$(metric 'radiance:decode_tokens_total')
  r1=$(metric 'radiance:draft_tokens_total');        a1=$(metric 'radiance:draft_accepted_total')
  p1=$(metric 'radiance:prefill_tokens_total');      t1=$(now_us)
  run=$(metric 'vllm:num_requests_running')
  local c_close=0
  [ "${run:-}" = "$c" ] && c_close=1

  # Every client is reaped before a log is read: mid-stream a log holds the chunks so far and no
  # usage line, which would silently report a short stream as a measurement.
  local wr=0
  for i in "${pids[@]}"; do wait "$i" || wr=1; done

  # Per request latency from the clients' own arrivals: TTFT from send to first chunk (so
  # pre-window queueing is included) and the pooled chunk-to-chunk intervals.
  local n_req win n_chunks usage
  read -r n_req win n_chunks usage < <(aggregate "$t0" "$t1" "$dir" "${files[@]}")

  local clients=1 crc chttp f
  for f in "${files[@]}"; do
    read -r crc chttp < "$f.status" || clients=0
    [ "${crc:-x}" = 0 ] && [ "${chttp:-x}" = 200 ] || clients=0
  done
  [ "$wr" = 0 ] || clients=0

  local dt steps tok draft acc pre usable=1
  dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.6f", (b - a) / 1000000}')
  steps=$(( ${s1:-0} - ${s0:-0} ));  tok=$(( ${d1:-0} - ${d0:-0} ))
  draft=$(( ${r1:-0} - ${r0:-0} ));  acc=$(( ${a1:-0} - ${a0:-0} ))
  pre=$(( ${p1:-0} - ${p0:-0} ))
  if [ "$c_open" != 1 ] || [ "$c_close" != 1 ] || [ "$clients" != 1 ] ||
     [ "$n_req" != "$c" ] || [ "$steps" -le 0 ] || [ "$win" -le 0 ]; then usable=0; fi

  awk -v arm="$arm" -v wl="$wl" -v c="$c" -v rep="$rep" -v dt="$dt" -v steps="$steps" \
      -v tok="$tok" -v draft="$draft" -v acc="$acc" -v pre="$pre" \
      -v c_open="$c_open" -v c_close="$c_close" -v run="${run:-0}" -v nreq="$n_req" \
      -v winchunk="$win" -v nchunk="$n_chunks" -v usage="$usage" -v usable="$usable" \
      -v ttft="$(pct "$dir/ttft.txt" 0.50)" \
      -v gap50="$(pct "$dir/gaps.txt" 0.50)" -v gap95="$(pct "$dir/gaps.txt" 0.95)" \
      'BEGIN {
        printf "%s %s %s %s %.3f %d %.2f %.3f %.1f %.1f %d %d %d %.3f %d %.2f %.2f %.2f %d %d %s %d\n",
          arm, wl, c, rep, dt, steps, (dt > 0 ? tok / dt : 0), (steps > 0 ? tok / steps : 0),
          (draft > 0 ? 100 * acc / draft : 0), (dt > 0 ? pre / dt : 0),
          nreq, winchunk, nchunk, (winchunk > 0 ? tok / winchunk : 0),
          usage, ttft, gap50, gap95, c_open, c_close, run, usable
      }' >> "$OUT"
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
printf '%s\n' \
  'arm workload c rep dt_s steps tok/s tok/step accept% prefill/s nreq winchunk nchunk tok/chunk usage_total ttft_ms chunkgap50_ms chunkgap95_ms c_open c_close run_end usable' \
  > "$OUT"

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

rows=$(awk 'NR > 1 && NF { n++ } END { print n + 0 }' "$OUT")
[ "$rows" -gt 0 ] || die "no measurement row was written; see $OUT"

# Median over repetitions per (arm, workload, C), usable rows only. The manifest asks for paired
# repetitions, so one row is a run, not a result.
awk '
  NR > 1 && $22 == 1 {
    k = $1 "\t" $2 "\t" $3
    n[k]++; t[k] = t[k] " " $7; s[k] = s[k] " " $8; a[k] = a[k] " " $9; g[k] = g[k] " " $18
  }
  END {
    for (k in n) {
      split(k, f, "\t")
      printf "%-12s %-10s %4s %9.2f %9.3f %8.1f %8.2f %6d\n", f[1], f[2], f[3],
             med(t[k]), med(s[k]), med(a[k]), med(g[k]), n[k]
    }
  }
  function med(list,   v, m, i, j, x, k) {
    m = split(list, v, " ")
    for (i = 1, k = 0; i <= m; i++) if (v[i] != "") v[++k] = v[i]   # each list starts with " "
    m = k
    for (i = 2; i <= m; i++) { x = v[i] + 0; j = i - 1
      while (j >= 1 && v[j] + 0 > x) { v[j+1] = v[j]; j-- } v[j+1] = x }
    return m % 2 ? v[(m+1)/2] : (v[m/2] + v[m/2+1]) / 2
  }' "$OUT" | sort -k1,1 -k2,2 -k3,3n > "$OUTDIR/summary.body"
printf '%-12s %-10s %4s %9s %9s %8s %8s %6s\n' arm workload C tok/s tok/step accept% chunkgap95ms runs \
  > "$OUTDIR/summary.txt"
cat "$OUTDIR/summary.body" >> "$OUTDIR/summary.txt"

cat "$OUTDIR/summary.txt"
printf 'raw rows: %s\n' "$OUT"

# A failed client, an empty window or an edge that never held C is not a measurement: the row
# stays in the raw file marked usable=0, and the run reports failure.
bad=$(awk 'NR > 1 && NF && $22 != 1 { n++ } END { print n + 0 }' "$OUT")
if [ "$bad" -gt 0 ]; then
  printf 'mtp_matrix: %d of %d row(s) unusable (client error, empty window, or not C at an edge); see %s\n' \
    "$bad" "$rows" "$OUT" >&2
  exit 1
fi
