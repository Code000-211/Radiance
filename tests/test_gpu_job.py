"""CPU-only process/CLI integration tests; never open GPU devices or the production lock."""

import fcntl
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "gpu_job.py"
FAKE_CLI = '''#!/usr/bin/env python3
import fcntl, json, os, sys
from pathlib import Path
root = Path(os.environ["FAKE_STATE"])
args = sys.argv[1:]

def row(path):
    name = path.name
    return {"id": "id-" + name, "identifier": name, "number": int(name.split("-")[1]),
            "project_id": "a64348a9-c661-493a-b111-6777ed95a9a3", "status": path.read_text(),
            "assignee_id": "agent-" + name, "assignee_type": "agent"}

def dispatch(name, kind):
    try:
        with open(os.environ["FAKE_LOCK"], "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                state = "free"
            except BlockingIOError:
                state = "locked"
    except FileNotFoundError:
        state = "free"
    with (root / "events").open("a") as f:
        f.write("dispatch " + name + " " + kind + " " + state + "\\n")

if args[:2] == ["issue", "get"]:
    print(json.dumps(row(root / args[2])))
elif args[:2] == ["issue", "list"]:
    status = args[args.index("--status") + 1]
    rows = [row(p) for p in root.glob("MAIN-*") if p.read_text() == status]
    print(json.dumps({"issues": rows, "has_more": False}))
elif args[:2] == ["issue", "status"]:
    with (root / "events").open("a") as f:
        f.write(args[2] + " " + args[3] + "\\n")
    if args[3] != "testing" and (root / "fail-final").exists():
        sys.exit(1)
    (root / args[2]).write_text(args[3])
    if args[3] == "testing" and (root / "fail-testing").exists():
        sys.exit(1)  # ambiguous error: server already changed status
elif args[:2] == ["issue", "children"]:
    tree = root / ("children-" + args[2])
    print(tree.read_text() if tree.exists() else json.dumps({"stages": [], "unstaged": [], "total": 0}))
elif args[:2] == ["issue", "wakeup"] and args[2] == "list":
    rules = root / ("wakeups-" + args[3])
    print(rules.read_text() if rules.exists() else "[]")
elif args[:2] == ["issue", "wakeup"] and args[2] == "trigger":
    dispatch(args[3].removeprefix("id-"), "trigger")
elif args[:2] == ["issue", "wakeup"] and args[2] == "create":
    agent = args[args.index("--agent-id") + 1]
    name = args[3].removeprefix("id-")
    with (root / ("wakeups-" + args[3])).open("a") as f:
        f.write(json.dumps([{"id": "created", "enabled": True}]))
    with (root / "events").open("a") as f:
        f.write("agent " + name + " " + agent + "\\n")
    dispatch(name, "create")
else:
    sys.exit(2)
'''


class GpuJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = self.root / "pair.lock"
        cli = self.root / "multica"
        cli.write_text(FAKE_CLI)
        cli.chmod(0o700)
        for issue in ("MAIN-1", "MAIN-2"):
            (self.root / issue).write_text("ready_for_testing")
        self.env = dict(os.environ, FAKE_STATE=str(self.root), FAKE_LOCK=str(self.lock),
                        PATH=str(self.root) + os.pathsep + os.environ["PATH"])
        self.launcher = (
            "import importlib.util,sys; from pathlib import Path; "
            f"s=importlib.util.spec_from_file_location('gpu_job',{str(RUNNER)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            f"m.LOCK=Path({str(self.lock)!r}); "

            "sys.exit(m.main())"
        )

    def start(self, code, issue="MAIN-1", after="in_review"):
        process = subprocess.Popen(
            [sys.executable, "-c", self.launcher, "--issue", issue, "--after", after,
             "--", sys.executable, "-c", code],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=self.env,
        )
        self.addCleanup(self.close_process, process)
        return process

    @staticmethod
    def close_process(process):
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()

    def finish(self, process, expected):
        out, err = process.communicate(timeout=10)
        self.assertEqual(process.returncode, expected, out + err)
        return out, err

    def ready(self, process):
        self.assertTrue(select.select([process.stdout], [], [], 5)[0], "child did not start")
        return process.stdout.readline().strip()

    def unlocked(self):
        with self.lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def events(self, prefix):
        path = self.root / "events"
        if not path.exists():
            return []
        return [line for line in path.read_text().splitlines() if line.startswith(prefix)]

    def test_initial_dispatch_and_automatic_handover(self):
        (self.root / "MAIN-3").write_text("blocked")
        self.finish(self.start("pass"), 0)
        self.assertEqual((self.root / "MAIN-1").read_text(), "in_review")
        # Exactly one successor, woken for its own assignee, only after this window was released.
        self.assertEqual(self.events("dispatch "), ["dispatch MAIN-2 create free"])
        self.assertEqual(self.events("agent "), ["agent MAIN-2 agent-MAIN-2"])

    def test_handover_after_failure(self):
        self.finish(self.start("raise SystemExit(7)"), 7)
        self.assertEqual((self.root / "MAIN-1").read_text(), "blocked")
        self.assertEqual(self.events("dispatch "), ["dispatch MAIN-2 create free"])

    def test_handover_after_in_progress(self):
        self.finish(self.start("pass", after="in_progress"), 0)
        self.assertEqual((self.root / "MAIN-1").read_text(), "in_progress")
        self.assertEqual(self.events("dispatch "), ["dispatch MAIN-2 create free"])

    def test_handover_skips_parent_with_open_child(self):
        (self.root / "MAIN-3").write_text("ready_for_testing")
        (self.root / "children-id-MAIN-2").write_text(json.dumps(
            {"stages": [], "total": 1, "unstaged": [{"identifier": "MAIN-9", "status": "todo"}]}))
        self.finish(self.start("pass"), 0)
        self.assertEqual(self.events("dispatch "), ["dispatch MAIN-3 create free"])

    def test_handover_reuses_existing_wakeup(self):
        (self.root / "wakeups-id-MAIN-2").write_text(json.dumps([{"id": "rule", "enabled": True}]))
        self.finish(self.start("pass"), 0)
        self.assertEqual(self.events("dispatch "), ["dispatch MAIN-2 trigger free"])

    def test_loser_does_not_dispatch(self):
        (self.root / "MAIN-2").write_text("testing")
        self.finish(self.start("raise SystemExit('must not run')"), 75)
        self.assertEqual(self.events("dispatch "), [])

    def test_competing_jobs_only_one_testing(self):
        first = self.start("import sys; print('ready',flush=True); sys.stdin.read()")
        self.assertEqual(self.ready(first), "ready")
        self.assertEqual((self.root / "MAIN-1").read_text(), "testing")
        marker = self.root / "second-load"
        second = self.start(f"from pathlib import Path; Path({str(marker)!r}).touch()", "MAIN-2")
        _, err = self.finish(second, 75)
        self.assertIn("MAIN-1", err)
        self.assertFalse(marker.exists())
        self.assertEqual((self.root / "MAIN-2").read_text(), "ready_for_testing")
        self.finish(first, 0)
        self.assertEqual((self.root / "MAIN-1").read_text(), "in_review")
        self.unlocked()
        # The handover (not this test) dispatched MAIN-2; it can now reuse the released window.
        self.assertEqual(self.events("dispatch "), ["dispatch MAIN-2 create free"])
        self.finish(self.start("pass", "MAIN-2", "in_progress"), 0)
        self.assertEqual((self.root / "MAIN-2").read_text(), "in_progress")

    def test_failed_status_update_never_starts_load(self):
        (self.root / "fail-testing").touch()
        marker = self.root / "load"
        self.finish(self.start(f"from pathlib import Path; Path({str(marker)!r}).touch()"), 1)
        self.assertFalse(marker.exists())
        self.assertEqual((self.root / "MAIN-1").read_text(), "blocked")
        self.unlocked()

    def test_error_cleans_status(self):
        self.finish(self.start("raise SystemExit(7)"), 7)
        self.assertEqual((self.root / "MAIN-1").read_text(), "blocked")
        self.unlocked()

    def test_signal_reaps_detached_descendant(self):
        code = ("import subprocess,sys; "
                "p=subprocess.Popen([sys.executable,'-c','import signal; signal.pause()'],"
                "start_new_session=True,close_fds=True); "
                "print(p.pid,flush=True); sys.stdin.read()")
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            with self.subTest(signal=sig):
                (self.root / "MAIN-1").write_text("ready_for_testing")
                process = self.start(code)
                pid = int(self.ready(process))
                process.send_signal(sig)
                self.finish(process, 1)
                self.assertFalse(Path(f"/proc/{pid}").exists())
                self.assertEqual((self.root / "MAIN-1").read_text(), "blocked")
                self.unlocked()

    def test_final_status_failure_is_not_reported_as_success(self):
        (self.root / "fail-final").touch()
        code = ("import subprocess,sys; "
                "p=subprocess.Popen([sys.executable,'-c','import signal; signal.pause()'],"
                "start_new_session=True,close_fds=True); print(p.pid,flush=True)")
        out, _ = self.finish(self.start(code), 1)
        self.assertFalse(Path(f"/proc/{int(out.strip())}").exists())
        self.assertEqual((self.root / "MAIN-1").read_text(), "testing")
        self.assertEqual(self.events("dispatch "), [])
        self.unlocked()
        self.finish(self.start("raise SystemExit('must not run')", "MAIN-2"), 75)

    def test_success_reaps_orphan_before_unlock(self):
        code = ("import subprocess,sys; "
                "p=subprocess.Popen([sys.executable,'-c','import signal; signal.pause()'],"
                "start_new_session=True,close_fds=True); print(p.pid,flush=True)")
        process = self.start(code)
        out, _ = self.finish(process, 0)
        self.assertFalse(Path(f"/proc/{int(out.strip())}").exists())
        self.assertEqual((self.root / "MAIN-1").read_text(), "in_review")
        self.unlocked()

    def test_existing_testing_issue_refuses_second_agent(self):
        (self.root / "MAIN-2").write_text("testing")
        self.finish(self.start("raise SystemExit('must not run')"), 75)
        self.assertEqual((self.root / "MAIN-1").read_text(), "ready_for_testing")
        self.unlocked()

    def test_no_foreign_process_visibility_required(self):
        # Device/foreign FD scans are not part of the agent-serialization contract.
        self.launcher = self.launcher.replace(
            "sys.exit(m.main())",
            "from unittest.mock import patch; "
            "patch('pathlib.Path.iterdir', side_effect=PermissionError('foreign proc')).start(); "
            "patch('os.readlink', side_effect=PermissionError('foreign fd')).start(); "
            "sys.exit(m.main())",
        )
        self.finish(self.start("pass"), 0)
        self.assertEqual((self.root / "MAIN-1").read_text(), "in_review")
        self.unlocked()

    def test_blocked_issue_is_not_promoted(self):
        (self.root / "MAIN-1").write_text("blocked")
        self.finish(self.start("pass"), 1)
        self.assertEqual((self.root / "MAIN-1").read_text(), "blocked")
        self.assertFalse(self.lock.exists())


if __name__ == "__main__":
    unittest.main()
