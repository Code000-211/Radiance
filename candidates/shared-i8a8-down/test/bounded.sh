#!/bin/sh
# bounded.sh -- every child this candidate's harness starts, started in its own session and under
# a limit it cannot outrun. Sourced by run.sh and by limit_check.sh.
#
#   . "$(dirname "$0")/bounded.sh"
#   bounded <seconds> <command...>
#
# Why this exists at all: `si8_test` loads a DSO that links libamdhip64. A child that gets past the
# row's refusals reaches hipLaunchKernel and holds a HIP/KFD context for as long as it lives, and
# the scheduler on this host prices VRAM by measurement -- three such children, each holding 147-177
# MB and burning hours of CPU, moved the measured budget by 322 MiB and made it refuse to start a
# job. A clang++ step has the same shape in a cheaper currency: it keeps ld/cc1 alive.
#
# `setsid -w` runs the child as its own session and process-group leader, so $! is the pgid and one
# group signal takes the whole subtree down -- the compilers under clang++, a hanger's own child.
# `timeout -k` gives that leader TERM, then KILL GRACE seconds later, and `wait` still reports
# timeout's status: the child's, 124 when it took the TERM, 137 (128+9) when the grace KILL was needed.
#
# A signal that ends the harness never reaches the child on its own -- it is in another session --
# so the caller must trap INT TERM HUP and call bounded_kill. limit_check.sh proves both paths.

command -v setsid >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1 \
    && command -v pgrep >/dev/null 2>&1 || {
    echo "bounded.sh needs util-linux' setsid, coreutils' timeout and procps' pgrep" >&2; exit 2; }

GRACE=${GRACE:-5}
_cur=0

bounded() {
    _seconds=$1; shift
    _what=$1
    setsid -w timeout -k "$GRACE" "$_seconds" "$@" & _cur=$!
    _status=0
    wait "$_cur" || _status=$?   # the caller's set -e must not swallow the status line below
    _cur=0
    # Say so, rather than leaving a silent 124/137: the step and its whole group are gone by here.
    if [ "$_status" -eq 124 ] || [ "$_status" -eq 137 ]; then
        echo "bounded: '$_what' and its group hit the ${_seconds}s limit (status $_status)" >&2
    fi
    return $_status
}

# Take the in-flight child's process group down with it. Two groups, not one: `timeout` puts the
# command it started in a group of its own, so killing only $! reaches timeout and leaves what the
# command spawned (clang++'s cc1 and ld, a hanger's own child) to whatever timeout forwards. That
# forward is one signal, once. Ask for timeout's child first, while it is still there to be asked.
# `-TERM -<pgid>`, not `-- -<pgid>`: the dash builtin reads the `--` as the pid and does nothing.
bounded_kill() {
    [ "${_cur:-0}" -gt 0 ] || return 0
    _kids=$(pgrep -P "$_cur" 2>/dev/null || true)
    kill -TERM "-$_cur" 2>/dev/null || true
    for _k in $_kids; do kill -TERM "-$_k" 2>/dev/null || true; done
    sleep 1
    kill -KILL "-$_cur" 2>/dev/null || true
    for _k in $_kids; do kill -KILL "-$_k" 2>/dev/null || true; done
    _cur=0
}
