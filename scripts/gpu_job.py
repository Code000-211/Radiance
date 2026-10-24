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


def gpu_holders() -> list[int]:
    """Inspect open device files without initializing HIP; unreadable processes fail closed."""
    holders = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    device = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if device == "/dev/kfd" or device.startswith("/dev/dri/render"):
                    holders.append(int(proc.name))
                    break
        except FileNotFoundError:
            continue
    return holders


def testing_issues() -> list[str]:
    found: list[str] = []
    offset = 0
    while True:
        page = multica("issue", "list", "--project", PROJECT, "--status", "testing",
                       "--limit", "100", "--offset", str(offset))
        rows = page["issues"]
        found.extend(i["identifier"] for i in rows)
        if not page.get("has_more", len(rows) == 100):
            return found
        if not rows:
            raise RuntimeError("incomplete testing issue listing")
        offset += len(rows)


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
        holders = gpu_holders()
        active = testing_issues()
        if holders or active:
            print(f"WAIT: GPU PIDs={holders}; testing={active}", file=sys.stderr)
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
        return result if result >= 0 else 128 - result


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
