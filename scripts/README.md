# scripts/

Harnesses that drive a running engine: bring a model up, measure it, and assert on what comes back
over the wire. They are versioned with the engine so measurements stay reproducible, and each
header records its method (why a window is sampled the way it is, which counter to trust), which is
hard to reconstruct from the number alone.

Scripts that start a server or a conversion run the build at `$BUILD`, by default `build/` in the
checkout the script is in, and default to containers under `$HOME/models/rad/` and checkpoints under
`$HOME/models/<org>/<name>`. A script that starts a server first stops any `radiance` already
running (`oracle.sh` only the one on its own port) and logs to `$HOME/<script>.log`. Most defaults
are environment overrides (`M=` for the container, `P=` for the port, and so on); each script's
header lists its own. Scripts that only attach to a live server need `curl` and `jq`; `cardclock.py`
and `agentreplay.py` need `python3`. Scripts find each other through `$(dirname "$0")`.

## Multica GPU coordination

For issue-owned GPU work, see [GPU-COORDINATION.md](../docs/GPU-COORDINATION.md).
`gpu_job.py --issue MAIN-N -- COMMAND...` owns one nonblocking pair-lock window,
sets `testing` before launching, and cleans up children before the final status.
Agent-owned GPU tests must use this runner; it serializes participating agents
without inspecting GPU processes. Existing scripts below do not acquire this outer
agent-test window automatically and must not be assumed to preserve foreign servers.

## Bringing a model up

| | |
|---|---|
| `mcserve.sh` | MiniCPM5-2B with its DSpark drafter (`minicpm5-2b-dspark.v2.rad`), left running on :8100. Records why `--max-model-len` is the checkpoint's `n_ctx_train` (131072) and what a token of KV costs. |
| `fnserve.sh` | Qwen3.8-Flash-Next, the production container `w4nl64-i8lm.rad`, left running on :8100. Records the production flags and their reasons: the 200K context and how the expert/cache split sizes the pool for it, the 2048-token chunk, the host prefix-cache tier and the lossy `wht6` cross-rank wire. |
| `fnstop.sh` | Stops whichever server is running and waits for the process to exit. |
| `convdspark.sh` | Rebuilds the container `mcserve.sh` serves, with the drafter merged, as `minicpm5-2b-dspark.new.rad` beside it. The checkpoint and drafter paths are fixed; `RECIPE=` overrides the recipe. |
| `convfn.sh` | Rebuilds the container `fnserve.sh` serves, as `w4nl64-i8lm.new.rad` beside it, from `data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe`. Needs `CALIB=<dir>`: the recipe's expert rule runs GPTQ against the calibration Grams there (`docs/MOE-W4.md`). |
| `hfget.sh` | Fetches a Hugging Face repo with `curl` and `jq`. Resumable, and skips the duplicate weight formats a repo often carries beside its safetensors. `radiance --model <repo id>` runs it. |
| `krea2_native.py` | Rewrites a Krea 2 DiT single file in Krea's own key names (`turbo.safetensors`, civitai and ComfyUI finetunes) as diffusers' transformer file, bytes unchanged, so it converts as the `transformer/` of a Krea 2 pipeline directory (`docs/GUIDE.md`). |

## Measuring

| | |
|---|---|
| `cstep2.sh` | Steady-state decode ms/step at each concurrency, against a live server. Waits until exactly C sequences are running before sampling and re-checks at the end, since a window with fewer than C at either edge measures no single concurrency. Model-agnostic (`P=` picks the port). |
| `dprofd.sh` | Per-op device timing (`--profile-ops`) at depth and concurrency on the 27B dense model with its DFlash2 drafter (`qwen3.8-27b-fp8.rad`); starts its own server on :8129 and needs a ~16K-token prompt (`PROMPT=<file>`). The counters are cumulative and a run starts with prefill, so one dump mixes prefill and decode; difference two dumps taken after prefill has ended. |
| `fnpf.sh` | Prefill throughput of the server on `$P`, over the repository's own markdown, with a fresh nonce per rep so the prefix cache cannot serve it. Prints card temperature and clock beside every rep. |
| `agentreplay.py` | Several agent conversations at once against a live server, each growing turn by turn to `--context` tokens and handing tasks to subagents that share its system prompt while it waits (`--rewrite`: a client that replaces its last message every turn). Prints per kind of turn how much of the prompt the cache served and how many continuing turns prefilled their history again, and the prefix-cache counters across the run. The conversations follow `--seed` and carry a fresh nonce, so two servers (one flag apart, restarted between) are sent the same work and share nothing with an earlier run. Takes the key from `--api-key-file`, never the command line. |
| `fnconc.sh` | What a long prefill costs the decoders running beside it, and the prefill's own time, measured together: alone, then contended, with a warm pass discarded. |
| `fntrace.sh` | A Flash-Next prefill under `rocprofv3`, on its own server, with the `/stats` link counters across the same request, giving the MoE GEMM's streamed-read rate. |
| `mctrace.sh` | A MiniCPM5-2B decode trace under `rocprofv3`, on its own server, formatted for `libr4d/isa/tsum.sh`, mainly for its `blocks` column. |
| `ident.sh` | Run-to-run reproducibility at a fixed configuration: one hash per (temperature, question), to compare against another run of the same configuration. A speculative run need not match a non-speculative one byte for byte (the split-K count is chosen from M, so reduction orders differ and a near-tie can flip late in a long answer), but a divergence within the first sentence or two is a defect. |
| `cardclock.py` | What holds each card's clock under sustained load, and which card a tensor-parallel step waits for (the slowest rank ends the step). Reads every AMD card's `gpu_metrics` table ten times a second and counts a throttler bit only when the clock is lower while it is set (some firmware keeps a bit on at idle). Shows the engine's step counters per window, so a slow window is attributed to slower steps, fewer tokens per step or host time between steps. Drives its own greedy load, or with `--watch` only samples a server under real traffic; `--compare` shows two saved runs side by side. Read-only on the cards. |
| `statsdump.py` | Record `/stats` to JSONL so the dashboard's figures can be measured over time instead of only watched: one object per poll, appended, `python3` only. Records the engine's snapshot verbatim plus the poll's own time and the gap since the last good poll -- it derives nothing, because the page's best/recent/now lines are computed from this same stream and a second copy of that arithmetic is a second thing that can drift. A failed poll leaves no row (a gap in the file is the honest record of a gap in the data) and an empty run exits non-zero, so a dead server cannot be mistaken for a quiet hour. The decode floor resets with the process: `step_ns_decode_best` improving across a restart is a new process finding its own best moment, not a faster machine. |
| `oracle.sh` | Every op compared against libref on the operands the model issued, which rad-kbench cannot do: its operands are its own (dense, rank 2), while the engine's are arena slices and weights loaded from a file. Starts its own server on :8302. Read it for `MISMATCH` lines, and for `NOT CHECKED` lines naming rows whose weight layout has no published inverse and so cannot be compared. |

## Regression checks against a live server

These cover what `tests/chat_test.cpp` cannot: the server, the wire format, and a real model's
output. `mcagent.sh` and `mcdecl.sh` start their own MiniCPM5-2B server through `mcserve.sh`.

| | |
|---|---|
| `fndrop.sh` | Clients that disconnect mid-request, at points derived from a timed full request: the server must stay up and keep answering a reference question identically. |
| `tierident.sh` | Whether a session that left VRAM comes back byte for byte, attention blocks and recurrent state both. The repeat from VRAM is the control: without it a differing hash proves nothing, since a cache hit could be tie-divergent. A run where nothing moved is a failure. On a hybrid model the linear-state checks are counters, not hashes: a lost snapshot does not show in the text, because the engine replays from further back and produces the same answer. Needs a server started with `--prefix-cache-mib 512`. |
| `tiercross.sh` | The same check with two sessions in flight, the only way to catch one being served the other's context or state. Compares each restore against that session's device-hit run rather than its cold one, because a hybrid session's reuse is clamped to the last reachable checkpoint and a cold run reuses nothing. |
| `mctools.sh` | The tool-call shapes hardest to parse: an array argument, two calls in one reply, prose before a call, plain chat with tools available. Asserts `finish_reason` and that no `<function`/`<param`/CDATA marker reaches content. |
| `mcstream.sh` | Tool-call replies streamed, asserting on raw SSE fragments rather than the assembled message. Streamed content cannot be retracted, so a repair applied at the end is not a fix. |
| `mcagent.sh` | What speculation is worth on an agentic workload: `/v1/chat/completions` with tools, a constrained decode. Speculation on and off alternate across restarts because the speculation depth is a server-level setting. |
| `mcdecl.sh` | How often the drafter proposes, by context length. `draft_acceptance_ratio` is conditional on having drafted, so it can look healthy while tokens per step collapse; blocks per step separates the two. |
