#!/bin/sh
# limit_check.sh -- prove that the harness's limits fire and that the harness leaves nothing behind.
#
#   sh test/limit_check.sh
#
# CPU only: every hanger here is a shell busy loop. No DSO is built, no HIP function is called, no
# device is opened -- the point is the limit, not the row. (run.sh hides the card as well; see its
# header.) Five cases, a few seconds, exit 0 only if all of them hold.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/bounded.sh"
trap 'bounded_kill' INT TERM HUP

LIMIT=${LIMIT:-2}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
say() { printf '%s\n' "$*"; }
ok()  { say "  ok    $*"; }
bad() { say "  FAIL  $*"; fails=$((fails + 1)); }
alive() { [ "${1:-0}" -gt 0 ] && kill -0 "$1" 2>/dev/null; }
pidof_() { cat "$1" 2>/dev/null || echo 0; }

# The hanger: a busy loop in a child of a child, so the pid we record is a grandchild of the
# harness and only a group signal can reach it.
cat > "$T/hang.sh" <<'EOF'
sh -c 'while :; do :; done' &
echo $! > "$1"
wait
EOF

say "limit ${LIMIT}s, kill grace ${GRACE}s, $(command -v setsid) + $(command -v timeout)"

# 1. A child that finishes is not touched, and its status survives.
say "1. green child"
bounded 10 sh -c 'exit 7'
rc=$?
[ "$rc" -eq 7 ] && ok "status 7 came back" || bad "status came back as $rc"

# 2. The limit fires and the exit status says so.
say "2. red: a child that never returns"
gc=$T/gc2.pid
t0=$(date +%s)
bounded "$LIMIT" sh "$T/hang.sh" "$gc"
rc=$?
t1=$(date +%s)
[ "$rc" -eq 124 ] && ok "exit 124 after $((t1 - t0))s" || bad "exit $rc instead of 124"
[ $((t1 - t0)) -le $((LIMIT + GRACE + 5)) ] && ok "stopped within ${LIMIT}s + grace" \
    || bad "took $((t1 - t0))s"
sleep 1
alive "$(pidof_ "$gc")" && bad "grandchild $(pidof_ "$gc") still running" \
    || ok "grandchild gone with the group"

# 3. A child that refuses TERM is still stopped, by the grace KILL. coreutils reports the
#    escalation honestly: 124 when the child took the TERM, 137 (128+9) when the KILL was needed.
#    Either way the limit won -- and 137 instead of 124 is what proves the escalation happened.
say "3. red: a child that ignores SIGTERM"
cat > "$T/stubborn.sh" <<'EOF'
trap '' TERM
while :; do :; done
EOF
t0=$(date +%s)
bounded "$LIMIT" sh "$T/stubborn.sh"
rc=$?
t1=$(date +%s)
case "$rc" in
  137) ok "exit 137 -- TERM ignored, KILLed ${GRACE}s later" ;;
  124) ok "exit 124 -- TERM was enough" ;;
  *)   bad "exit $rc, expected 124 or 137" ;;
esac
[ $((t1 - t0)) -ge "$LIMIT" ] && [ $((t1 - t0)) -le $((LIMIT + GRACE + 5)) ] \
    && ok "stopped within ${LIMIT}s + grace" || bad "took $((t1 - t0))s"
# 4. A signal that ends the harness takes the in-flight child's group with it.
#    This is the case that leaked three KFD contexts: the child lives in its own session, so it
#    never receives the harness's signal and survives unless bounded_kill is called.
say "4. red: the harness itself is signalled mid-run"
gc4=$T/gc4.pid
( sleep 1; kill -TERM $$ ) &
bounded 60 sh "$T/hang.sh" "$gc4"
rc=$?
[ "$rc" -eq 143 ] || [ "$rc" -eq 124 ] && ok "harness stopped with $rc" || bad "harness status $rc"
sleep 1
alive "$(pidof_ "$gc4")" && bad "grandchild $(pidof_ "$gc4") survived the signal" \
    || ok "grandchild gone with the group"

# 6. The combined case: the child itself ends on TERM, but the hanger it started ignores TERM.
#    That is the shape run.sh's own children have -- cc1 and ld under clang++, a grandchild under a
#    hanger -- and it is the one case neither 3 (the whole group ignores TERM, timeout escalates)
#    nor 4 (one signal, no escalation) covers: here the TERM does reach the child, the child dies
#    of it, and the grandchild survives unless bounded_kill escalates to KILL.
say "6. red: child ends on TERM, grandchild ignores TERM"
cat > "$T/polite.sh" <<'EOF'
sh -c 'trap "" TERM; while :; do :; done' &
echo $! > "$1"
wait
EOF
gc6=$T/gc6.pid
( . "$HERE/bounded.sh"; trap 'bounded_kill' INT TERM HUP
  bounded 60 sh "$T/polite.sh" "$gc6" ) &
hp=$!
sleep 2
if alive "$(pidof_ "$gc6")"; then ok "grandchild $(pidof_ "$gc6") is up and ignores TERM"
else bad "grandchild never started"; fi
kill -TERM "$hp" 2>/dev/null
wait "$hp"
rc=$?
if [ "$rc" -eq 143 ] || [ "$rc" -eq 124 ]; then ok "child ended with $rc"
else bad "child status $rc"; fi
sleep 1
if alive "$(pidof_ "$gc6")"; then bad "grandchild $(pidof_ "$gc6") survived the cleanup"
else ok "grandchild removed by the KILL escalation"; fi

# 7. Nothing of this run is left.
say "7. own children afterwards"
left=$(ps -eo pid=,args= | grep -F "$T/hang.sh" | grep -v grep || true)
[ -z "$left" ] && ok "no hanger left" || bad "still running: $left"

say
if [ "$fails" -eq 0 ]; then
  say "RESULT: OK -- the limit fires, the group dies with it, nothing is left behind."
  exit 0
fi
say "RESULT: $fails check(s) failed."
exit 1
