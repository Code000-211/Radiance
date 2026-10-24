# One testing agent at a time (MAIN-84)

code211 clarified the scope on 2026-10-24: serialize the Radiance agents' GPU test
phases. **No GPU process inventory, foreign-load check, or complete `/proc`
visibility is required.** This is cooperative agent coordination, not device
isolation from unrelated software. Ordinary CPU preparation/unit tests are not a
reservation of the shared GPU test window.

## Run a test phase

Prepare on CPU first. The executing issue must belong to Radiance and already be
`ready_for_testing`; `blocked` issues are never promoted by the runner.

```bash
python3 scripts/gpu_job.py --issue MAIN-N -- bash foreground-measurement.sh
# If CPU work follows a successful test phase:
python3 scripts/gpu_job.py --issue MAIN-N --after in_progress -- ./hip-probe
```

Every participating agent uses `/tmp/radiance-tp2-bench.lock` on the **same
executing machine and in the same lock namespace**. The foreground command owns
the entire test: HIP initialization/probes, selftests, diagnostics, server,
benchmark clients, A/B arms, and cleanup. Do not nest a legacy script's independent
acquisition of the same lock; migrate its lock boundary with its owner first.

The runner takes the lock **nonblocking**, checks that no other project issue is
`testing`, and successfully writes the executing issue to `testing` before
launching its command. A failed/ambiguous status write launches nothing and
attempts to restore `blocked`. Lock metadata records the issue, host, and owner
PID; it is diagnostic information, not a substitute for the kernel lock.

Exit **75** means no test was launched: another agent holds the lock, or another
`testing` issue is recorded. The waiter stays `ready_for_testing`. Name the holder
from the lock metadata or recorded issue, without scanning GPU processes. Legacy
locks can lack metadata; report that rather than guessing. Never unlink the lock
file, sleep-retry, choose a different lock, or start a second uncoordinated test.

Use existing Multica wakeups/dispatch to resume. The runner itself hands off: after a
window closes (success, `--after in_progress`, or failure to `blocked`) it wakes exactly
one runnable waiter through the platform's persistent wakeups. A waiting agent does not
start a second runner or poll.

Handover candidate rules, applied after the lock is released and the final status is
written:

- only issues in `ready_for_testing` (so `blocked` work is never promoted); the finished
  issue is excluded, and a parent whose child issue is still open is skipped because the
  child carries that measurement;
- candidates ordered by issue number, one dispatched per finished window;
- the wake target is the candidate's own agent assignee, or the squad leader for a
  squad-assigned issue (the platform starts only that leader on a squad assignment). An
  enabled rule is reused only
  when it wakes that same assignee **and** its instruction names
  `scripts/gpu_job.py` (a GPU continuation). Foreign or differently-motivated rules are
  left untouched and a dedicated one-shot `at` continuation is created instead. No
  scheduler is added;
- a nonzero exit (or any error) of a `wakeup create`/`wakeup trigger` proves nothing on
  its own, so the handover stops instead of waking a second candidate: the write may
  already have queued a run. A visible own GPU rule — even one already fired and
  disabled — is treated as landed; an empty follow-up read is **not** evidence of
  rejection, because a local CLI exit neither contradicts a stored rule nor proves a
  delayed one absent. No sleep-retry is used; the Sweeper stays the backstop;
- the queue still advances past a candidate that is provably unexecutable **before any
  write**: a wake target (assignee agent or squad leader) that is no longer in the agent
  list, a squad without a leader, or an unreadable children list is skipped, so one
  unusable candidate cannot stall the handover.

If the handover call fails it is reported on stderr and does not rewrite the finished
test's exit code; use the existing Sweeper as backstop. A stale `testing` issue (owner
never reached its final write) blocks handover and needs coordination with its owner.
The runner cannot see arbitrary queued/running tasks, so the resumed agent must not start
a duplicate run; it rechecks its state before its GPU command.

```bash
# Created by the finishing runner, not by hand:
multica issue wakeup create MAIN-WAITER --kind at --after 1s --agent-id AGENT-ID \
  --instruction "A GPU test window is free. ... python3 scripts/gpu_job.py --issue MAIN-WAITER -- ..."
```

## Completion and cleanup

The lock remains held through the command, cleanup, and final status write.
Success goes to `in_review`, or explicitly `--after in_progress`; failure/handled
signals go to `blocked`. A Linux child subreaper kills and reaps its remaining
owned descendants, including double-forked/new-session children, before release.
Reading its **own child list** for cleanup does not require inspecting GPU users
or foreign process FDs. Unrelated processes are never killed.

SIGINT/SIGTERM/SIGHUP and command failures are tested. SIGKILL, host/runtime death,
uninterruptible kernel tasks, or an unavailable Multica final write cannot have
transactional cleanup/status guarantees. A final status error exits nonzero after
child cleanup; stale `testing` prevents another compliant agent from proceeding
until the owning job's state is reconciled. The runner is local-only: SSH, an
external supervisor, another host, or a private-container lock namespace would
escape its lifecycle/lock boundary. Persistent serving is not a supported handoff;
leftover owned servers are terminated when the foreground command ends.

## Agent and Sweeper contract

The approved instructions are applied to Radiance Testing, Code 1–4,
Implementation 1–4, Review 1–4, and Rescue (14 existing agents). They require the
runner for every already-authorized GPU test phase; no extra GPU permission is
created. A checkout without the runner must obtain the PR #47 version first.
Already-claimed agent runs do not receive new instructions retroactively.

The existing Sweeper autopilot preserves `ready_for_testing` and `testing`, uses
existing wakeups instead of another scheduler, and checks real issue blockers
before releasing blocked GPU work. Neither the runner nor the Sweeper requires
foreign GPU process visibility. Existing roles, models, runtimes, permissions,
concurrency limits, and the Sweeper schedule/assignee are unchanged.

The runner is the common boundary for agent-owned tests; it does not automatically
rewrite every engine binary, CTest case, repository serving script, or runtime-local
measurement script. An unwrapped/manual command outside this agent contract is
not excluded. Owners migrate legacy inner locks at safe test boundaries without
editing somebody else's running experiment. Do not interpret the previous
process-visibility audit/blocker as a remaining acceptance criterion: code211's
clarification explicitly superseded it.

## CPU-only verification

```bash
python3 -m unittest discover -s tests -p test_gpu_job.py -v
ruff check scripts/gpu_job.py tests/test_gpu_job.py
uvx mypy --check-untyped-defs scripts/gpu_job.py tests/test_gpu_job.py
```

Tests use a temporary lock and fake CLI, with no GPU load or production lock.
They prove two competing agents produce one test window/one `testing`; the loser
launches nothing and dispatches nothing, and a later run can reuse the lock. Also
covered: initial dispatch on an empty window (the command observes `testing` before
it starts), an executed continuation chain for success, failure to `blocked`, and
`--after in_progress` (the woken candidate's own runner starts without any second
manual start in the test body, after the finished issue's final status, and releases
the lock again), exactly one successor dispatched after release for its own agent,
skipping a blocked candidate and a parent with open child work, waking a squad
leader, reuse of an own GPU rule and ignoring foreign/unrelated rules, skipping a candidate whose wake target no
longer exists with the next candidate dispatched, and four no-double-start cases: a
create whose response was lost after committing, the same with the stored rule already
fired/disabled, the same with the rule not yet visible, and a lost trigger response,
blocked/stale issue refusal, failed initial/final status updates, command failure
(no handover), SIGINT/SIGTERM/SIGHUP, detached/orphan cleanup, both successful exit
statuses, and successful execution without foreign process/FD inspection
permissions.
