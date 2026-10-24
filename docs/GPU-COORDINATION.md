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

Use existing Multica wakeups/dispatch to resume; check the waiter's existing rules
before registering another. For example:

```bash
multica issue wakeup create MAIN-WAITER --until-issue MAIN-HOLDER \
  --until-issue-state in_review --expires-in 24h --on-timeout wake \
  --instruction-file ./resume-test.md
```

The instruction rechecks the issue's blockers and reacquires the lock. A failed
holder or stale `testing` needs coordination with its owner, not a polling loop.

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
launches nothing, and a later run can reuse the lock. Also covered: blocked/stale
issue refusal, failed initial/final status updates, command failure,
SIGINT/SIGTERM/SIGHUP, detached/orphan cleanup, both successful exit statuses,
and successful execution without foreign process/FD inspection permissions.
