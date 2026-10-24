#!/usr/bin/env python3
"""Linux/Multica GPU window: gpu_job.py --issue MAIN-N [--after in_progress] -- COMMAND..."""

import argparse
import ctypes
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

LOCK = Path("/tmp/radiance-tp2-bench.lock")
PROJECT = "a64348a9-c661-493a-b111-6777ed95a9a3"


def multica(*args: str) -> Any:
    result = subprocess.run(
        ["multica", *args, "--output", "json"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def issues_with_status(status: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = multica("issue", "list", "--project", PROJECT, "--status", status,
                       "--limit", "100", "--offset", str(offset))
        rows = page["issues"]
        found.extend(rows)
        if not page.get("has_more", len(rows) == 100):
            return found
        if not rows:
            raise RuntimeError(f"incomplete {status} issue listing")
        offset += len(rows)


def testing_issues() -> list[str]:
    return [i["identifier"] for i in issues_with_status("testing")]


HANDOVER = ("A GPU test window is free. Re-read this issue's blockers, then run its already-authorized "
            "GPU phase through: python3 scripts/gpu_job.py --issue {issue} -- <foreground command>. "
            "If the window is taken, stay ready_for_testing; the next finished runner hands off again.")
# Only a rule carrying this marker is a GPU continuation; anything else waits for its own reason.
GPU_RULE_MARKER = "scripts/gpu_job.py"


def open_children(issue_id: str) -> list[dict[str, Any]]:
    tree = multica("issue", "children", issue_id) or {}
    rows = list(tree.get("unstaged") or [])
    for stage in tree.get("stages") or []:
        rows.extend(stage.get("issues") or [])
    return [c for c in rows if c.get("status") not in ("done", "cancelled")]


def wake_target(candidate: dict[str, Any]) -> str | None:
    """Agent to wake: the assignee itself, or the leader for a squad-assigned issue."""
    kind, ident = candidate.get("assignee_type"), candidate.get("assignee_id")
    if kind == "agent" and ident:
        return ident
    if kind == "squad" and ident:
        return (multica("squad", "get", ident) or {}).get("leader_id")
    return None


class DispatchRejected(Exception):
    """The platform provably stored no wakeup; no run was queued."""


class DispatchAmbiguous(Exception):
    """A write failed without proof about whether the wakeup exists; never try a second one."""


def gpu_continuation(rule: dict[str, Any], agent_id: str) -> bool:
    """An enabled rule that wakes this candidate's own agent with a GPU-runner instruction."""
    return (bool(rule.get("enabled")) and rule.get("agent_id") == agent_id
            and GPU_RULE_MARKER in (rule.get("instruction") or ""))


def own_rules(issue_id: str, target: str) -> list[dict[str, Any]]:
    return [r for r in multica("issue", "wakeup", "list", issue_id) or []
            if gpu_continuation(r, target)]


def dispatch(candidate: dict[str, Any], target: str) -> None:
    """Wake one candidate through its own GPU continuation rule, else a dedicated one-shot rule.

    A nonzero exit of a write proves nothing, so a failed create is reconciled by re-reading the
    rules (a stored rule means it landed); without that proof, or for a failed trigger, the
    handover stops instead of risking a second start.
    """
    name = candidate["identifier"]
    try:
        rules = own_rules(candidate["id"], target)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise DispatchAmbiguous(f"cannot read rules: {exc}") from exc
    if rules:
        try:
            multica("issue", "wakeup", "trigger", candidate["id"], rules[0]["id"])
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            # Fired or rejected is not observable here; a second dispatch could double-start.
            raise DispatchAmbiguous(f"trigger failed: {exc}") from exc
        return
    try:
        multica("issue", "wakeup", "create", candidate["id"], "--kind", "at",
                "--after", "1s", "--agent-id", target,
                "--instruction", HANDOVER.format(issue=name))
    except subprocess.CalledProcessError as exc:
        # The CLI exited, so the write either committed or not; only a stored rule proves it did.
        try:
            landed = own_rules(candidate["id"], target)
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, RuntimeError) as read_exc:
            raise DispatchAmbiguous(f"create exit {exc.returncode}, rules unreadable: {read_exc}") from exc
        if landed:
            return
        raise DispatchRejected(f"create exit {exc.returncode}, no rule stored") from exc
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise DispatchAmbiguous(f"create failed: {exc}") from exc


def hand_over(finished: str) -> None:
    """One-shot dispatch of the next runnable waiter through existing Multica wakeups."""
    try:
        candidates = issues_with_status("ready_for_testing")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"handover: cannot read waiting candidates: {exc}", file=sys.stderr)
        return
    for candidate in sorted(candidates, key=lambda i: i["number"]):
        name = candidate["identifier"]
        if name == finished:
            continue
        try:
            target = wake_target(candidate)
            # A parent whose child work is still open must not duplicate the child's measurement.
            open_work = open_children(candidate["id"]) if target else []
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            # Nothing was written for this candidate; skipping it cannot double-start anything.
            print(f"handover: cannot inspect {name}; skipped: {exc}", file=sys.stderr)
            continue
        if not target:
            print(f"handover: {name} has no wakable agent or squad leader; skipped", file=sys.stderr)
            continue
        if open_work:
            print(f"handover: {name} still has open child work; skipped", file=sys.stderr)
            continue
        try:
            dispatch(candidate, target)
        except DispatchRejected as exc:
            print(f"handover: {name} rejected ({exc}); trying next candidate", file=sys.stderr)
            continue
        except DispatchAmbiguous as exc:
            print(f"handover: ambiguous dispatch for {name}; not trying another candidate: {exc}",
                  file=sys.stderr)
            return
        print(f"handover: dispatched {name}", file=sys.stderr)
        return
    print("handover: no runnable waiting candidate", file=sys.stderr)


def reap_children() -> None:
    """As a subreaper, also own descendants that double-fork or create new sessions."""
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    while True:
        for pid in children.read_text().split():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            return


def interrupted(signum: int, _frame: Any) -> None:
    raise InterruptedError(f"signal {signum}")


def run_job(issue: str, command: list[str], after: str) -> int:
    if not command:
        raise ValueError("missing command")
    # Do not promote blocked work, or turn a second invocation of the same issue into a waiter.
    record = multica("issue", "get", issue)
    if record["project_id"] != PROJECT or record["status"] != "ready_for_testing":
        raise RuntimeError("issue must belong to Radiance and be ready_for_testing")
    with open(os.open(LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"WAIT: lock held; recorded owner: {lock.read(4096) or 'legacy/unknown'}", file=sys.stderr)
            return 75
        active = testing_issues()
        if active:
            print(f"WAIT: another agent test window is recorded: {active}", file=sys.stderr)
            return 75
        # PR_SET_CHILD_SUBREAPER: orphaned grandchildren must not outlive the lock window.
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot become child subreaper")
        if multica("issue", "get", issue)["status"] != "ready_for_testing":
            raise RuntimeError("issue changed while acquiring GPU window")
        lock.seek(0)
        lock.truncate()
        json.dump({"issue": record["identifier"], "pid": os.getpid(),
                   "host": os.uname().nodename, "phase": "acquired"}, lock)
        lock.flush()
        previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        result = 1
        try:
            # Even an ambiguous/failed status write must go through the final status cleanup.
            multica("issue", "status", issue, "testing")
            process = subprocess.Popen(command, start_new_session=True, pass_fds=(lock.fileno(),))
            result = process.wait()
        except (OSError, subprocess.SubprocessError, ValueError, KeyboardInterrupt) as exc:
            print(f"GPU job failed: {exc}", file=sys.stderr)
        finally:
            for s in previous:
                signal.signal(s, signal.SIG_IGN)
            try:
                reap_children()
                # Keep the lock until both process cleanup and final status write have finished.
                multica("issue", "status", issue, after if result == 0 else "blocked")
                lock.seek(0)
                lock.truncate()
                json.dump({"issue": record["identifier"], "phase": "released", "exit": result}, lock)
                lock.flush()
            finally:
                for s, handler in previous.items():
                    signal.signal(s, handler)
        return_code = result if result >= 0 else 128 - result
    # Only after the lock is released: the next agent may acquire the window it is woken for.
    hand_over(record["identifier"])
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--after", choices=("in_review", "in_progress"), default="in_review")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        return run_job(args.issue, command, args.after)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        print(f"GPU window refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
