# Radiance GPU windows (MAIN-84)

`gpu_job.py` is an **opt-in Linux runner**, not a scheduler or a device-access sandbox.
It does not yet make all existing entry points safe. Do not claim project-wide
exclusivity until the dispatch/configuration and legacy harness migration is approved
and completed. Never launch a second job simply because an agent run is only queued.

## One window

CPU preparation happens before the window. The executing issue must belong to
Radiance and already be `ready_for_testing`; `blocked` is never promoted by the runner.
Run on the GPU host, in the same PID/device visibility and mount namespace as all
other GPU jobs:

```bash
python3 scripts/gpu_job.py --issue MAIN-N -- bash foreground-measurement.sh
# If more CPU work follows a successful GPU phase:
python3 scripts/gpu_job.py --issue MAIN-N --after in_progress -- ./hip-probe
```

The **whole** foreground measurement belongs inside the window: server startup,
HIP initialization, probes/selftests, diagnostic runs, clients driving the server,
A/B arms, and server shutdown. A client attaching to somebody else's server is not
a new independent window. Never use `fnserve.sh` as a persistent service handoff
through this runner: it intentionally kills leftover descendants when the command
returns. Do not put secrets in command arguments.

The runner:

1. Takes `/tmp/radiance-tp2-bench.lock` nonblocking, on the executing machine.
2. Reads `/proc/*/fd` for KFD/render-device users without initializing HIP; unreadable
   process data fails closed. Refuses any existing project issue in `testing`.
3. Sets the actual executing issue to `testing`. A failed/ambiguous CLI write starts
   no command and attempts to restore `blocked`.
4. Starts the command, inheriting the lock FD. A Linux child subreaper owns even
   double-forked/new-session descendants; it kills and reaps remaining children
   before writing the final status and releasing the lock. It never kills foreign PIDs.
5. Success goes to `in_review` (or explicit `in_progress`); failure/handled signals
   go to `blocked`. Status CLI failure is an error, not success.

Exit **75** means no GPU command started: lock occupied, foreign device user, or
another `testing` issue. The waiting issue stays `ready_for_testing`. Output names
recorded lock metadata or device PIDs/testing issues. Metadata can be stale and
legacy locks can have no metadata: verify live open lock/device FDs and process
ancestry before attributing ownership. Never unlink the lock file (that creates
a second inode/mutex), reset a GPU, stop a foreign process, choose another card,
or retry in a sleep/poll loop.

Use Multica's existing persisted wakeups/dispatch to resume after the holder's
result, then acquire the lock again; issue status is not proof that the device is
free. Example, only after checking the waiting issue's existing wakeups:

```bash
multica issue wakeup create MAIN-WAITER --until-issue MAIN-HOLDER \
  --until-issue-state in_review --expires-in 24h --on-timeout wake \
  --instruction-file ./resume-gpu.md
```

The instruction must recheck blockers, device holders, and lock, not blindly start.
A failed holder may need manual resolution; do not repeatedly register wakeups.

## Limits and recovery

- This is cooperative exclusion. Unwrapped binaries, HIP tests, other containers,
  hosts, or privileged services can bypass it. The runner cannot prevent a foreign
  uncooperative process starting after the preflight device scan. It requires full
  `/proc` visibility; restricted visibility must not be treated as a free GPU.
- SIGINT/SIGTERM/SIGHUP and ordinary command failures are cleaned up. SIGKILL,
  host/runtime death, uninterruptible kernel tasks, or Multica being unavailable
  during the final status write cannot have transactional cleanup guarantees.
  A surviving child retaining the inherited FD still holds the lock, but a detached
  child closing it requires live recovery after catastrophic runner death. Do not
  infer freedom from a stale status, nor label SIGKILL as a tested safe exit.
- CLI status writes and the device mutex are not one platform transaction. If final
  status writing fails, the runner exits nonzero, after killing/reaping its children;
  `testing` may remain stale. A subsequent compliant run refuses to start. Reconcile
  with device/process evidence, then repair the status through Multica.
- Commands must execute locally. Do not use this wrapper around SSH, a container
  with a private `/tmp`, another supervisor, or an external job queue: that child
  lifecycle is not owned by the subreaper.
- Legacy scripts taking the same lock internally cannot simply be nested in this
  runner: the second lock acquisition will fail. Migrate their lock/status/cleanup
  boundary together, under explicit ownership, before running them here.

## Audit and pending rollout

Snapshot on 2026-10-24, host `vllm-radiance`: MAIN-80's Arm-1 server PID 101893
opened `/dev/kfd` and both render devices. Its parent `measure.sh` PID 101672 and
the server inherited FD 9 with the pair's FLOCK record. The wrapper and server
shared mount namespace `mnt:[4026533224]`. Worktree/log paths and the running
Multica MAIN-80 task `01a16f38-db04-7e02-9bc2-de3daec4049f` agree on ownership.
MAIN-80 incorrectly remained `todo`; it was reconciled to `testing` using that
live evidence. No process was stopped and no GPU code was executed by this audit.

MAIN-80's runtime-local `arm1/measure.sh` is **not versioned in this repo**. It has
an exclusive lock but no status lifecycle and only an EXIT server cleanup trap;
`arm1/drive.sh` retries exit 7 five times with 60-second sleeps. They remain legacy
entry points, not silently edited while another agent's measurements are running.

The existing `/home/orca/multica-tools/dispatch-sweep.py` nudges only exact `todo`
and `in_progress` states; it already ignores `ready_for_testing` and `testing`.
Do not add a second scheduler. Its autopilot also contains prose-driven triage
and blocked-work restarts: that configuration needs explicit GPU-state exceptions.
No agent/autopilot configuration was changed without approval.

A direct read-only call to the production `gpu_holders()` preflight in this runtime
failed with `PermissionError: /proc/1/fd`. The runner therefore correctly refuses
GPU work here. A complete system-wide holder inventory cannot be asserted from
this account; visible MAIN-80 evidence is positive ownership evidence, not proof
that no hidden foreign process exists. Rollout additionally requires an approved
execution context with complete read-only process/device visibility on the same
GPU host. Do not bypass this guard or silently widen runtime permissions.

Still **not wired**: runtime-local benchmark/probe scripts for MAIN-6/79/80 and
other issues; direct `radiance`, `rad-kbench`, HIP probe/selftest/CTest invocations;
repository serving/conversion/diagnostic scripts (including `fnserve`, `mcserve`,
`oracle`, `dprofd`, trace scripts, and clients driving an already-live server).
Some existing scripts stop servers globally; do not run these against foreign
processes just because an outer lock was acquired.

Rollout requires confirmation of this concrete configuration change: GPU-capable
agents must use this runner for every GPU phase, maintain `ready_for_testing`
between phases, and use persisted wakeups instead of retry loops; the existing
Sweeper must never nudge waiters, rewrite active `testing`, or release blocked
GPU work without checking its blockers. Migrate the owning agents' local harnesses
at a safe boundary after the active measurement, not by terminating it.

## CPU-only verification

```bash
python3 -m unittest discover -s tests -p test_gpu_job.py -v
ruff check scripts/gpu_job.py tests/test_gpu_job.py
uvx mypy --check-untyped-defs scripts/gpu_job.py tests/test_gpu_job.py
```

Tests use a temporary lock, fake Multica CLI, and no-op child commands. They cover
two competing issues (only one gets `testing`/starts), blocked/foreign/stale-state
refusal (including unreadable process tables), ambiguous failed status write,
final-status-write failure, command failure, signals, orphan/new-session
cleanup, lock reuse, and both successful exit statuses. They do not measure GPU
performance or exercise real Multica writes in failure tests.
