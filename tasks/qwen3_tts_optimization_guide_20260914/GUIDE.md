# Qwen3-TTS optimization guide, top to bottom

Written 2026-09-14 from the Fun-CosyVoice3 and Qwen3-TTS work. Every fact here was read from the
code, a PR body, a task ledger or a run archive on that date. When a line names a file and a line
number, it is at upstream main `1f6b6843e` unless the line says otherwise. Verify a line before
acting on it; code moves.

Read in order the first time. Afterwards jump by section number.

```text
0  rules that never change
1  machines, checkouts, remotes, worktrees
2  where the work stands (CosyVoice stack, Qwen3-TTS PRs, held branches)
3  the strategy: origin first, from first principles
4  the process, one step at a time
5  profiling: Nsight, the SM window, the torch profiler step window, sync detection
6  benchmarking: the census, c1 identity, quality, delta tables
7  code design: what to write and what never to write, with examples from this work
8  git, stacks, PR bodies, titles, review handling
9  Qwen3-TTS: mechanics, remaining bottlenecks, the next targets in order
10 subagents
11 checklists
```

## 0. Rules that never change

These came from corrections during the work. Each one cost a rerun or a rewrite once.

1. **The origin, never the symptom.** Name the line that pays, not the metric that moved. "SM Issue
   is 10 percent" is a symptom. "solve_flow_euler_packed issues 18,100 launches per call at 280 ms
   host floor" is an origin. A ledger row without an origin line is not done.
2. **No constant that a measurement does not pin, no constant that pins to one GPU.** A padded
   frame budget, a host to GPU ratio, a "3,500 frames" floor: all withdrawn. If a number encodes
   how fast this H100 launches kernels, it is wrong on an H200. Fix the layout, the sync, the
   launch count. Never tune admission.
3. **The default single GPU launch is the deliverable.** MPS, process splits, extra GPUs, TensorRT
   and torch.compile are never the fix. The fix ships in `sgl-omni serve --model-path ...` with
   no flags.
4. **No SGLang patches.** SGLang is a pinned dependency (`sglang==0.5.19`, `flashinfer_python
   0.6.18`, pyproject.toml:34,37). Close a contract gap at an omni owned seam. Before building or
   removing an omni mechanism, grep the pinned SGLang checkout for its counterpart
   (`/Users/ratish/sglang`, confirm `git describe` prints the pinned tag first).
5. **Derive, never pattern match.** "SGLang does this" and "Qwen3-TTS does this" are not
   justifications, in code comments or in a design. Requirements first, then the mechanism, then an
   experiment with a name that gates the empirical part.
6. **Bit for bit before speed.** Every numerics question gets one experiment: every path against
   the same truth, per call, on real activations, inputs saved. Qualify a kernel on the model's own
   activation ranges before trusting it (flashinfer 0.6.18 returned zero rows on this DiT because
   its "minus infinity" is a finite minus 5e4 in the raw logit domain; a synthetic test never saw it).
7. **No local pytest, no local model runs.** Unit tests, benchmarks and profiles run on the H100
   venv of the branch under test. The Mac runs pre-commit hooks, greps, git, and standard library
   analysis over exported SQLite files. Nsight exports and heavy SQLite scans run in the container.
8. **One boot per arm and point.** The census is the measurement. Repeat only when a delta is
   inside about 2 percent. Never require an idle host: record the other GPUs and quote paired deltas.
9. **Never claim a profile you cannot defend.** A 32 request window is ramp and drain, not a
   measurement. If a reviewer can void it, it is void. Say what the window is and how long it ran.
10. **Discuss before runtime code changes.** Tests, runbooks and `tasks/` docs are pushed directly.
    Runtime code is discussed, then committed one mechanical change per commit, each reviewed.
11. **Tests test shipped contracts.** No mock and count tests, no incident replay fixtures
    (`fixtures/streaming_c16_gpu0_inbox.json` was removed from a PR for this), no comments inside
    tests, fakes model real resolved shapes. A test never justifies keeping wrong code.
12. **Verify every subagent's report yourself.** Subagents are Opus 5 medium, at most 5 at once.
    They do unpacking, greps, SQLite queries, tables and mechanics explanations. Design, decisions
    and final code stay with you. Only CONFIRMED findings enter a plan.
13. **Do not delete any CosyVoice or Qwen3-TTS branch without the user's word.** Worktrees stay.
14. **Continuity is 100 percent, for every streaming model, always.** C50 and C100 at 100.0 on the
    full corpus with zero failures is a gate, not a metric to trade. A change that buys first audio
    or throughput with a continuity point is withdrawn. Every fix is listed in the model's
    MECHANICS ledger before it is designed, so nothing needed is forgotten and nothing is tried twice.
15. **A and B are different trees, proven.** Archive `head.txt` per boot and diff the two heads
    before believing a delta. A PR squash merged into main means an "upstream main" arm fetched after
    the merge is the PR tree; the pair then measures run to run noise (this happened with #2169,
    merged 23:33 IST 2026-09-14).

## 1. Machines, checkouts, remotes, worktrees

### Mac

- Main checkout: `/Users/ratish/sglang-omni`, branch `main`. Remotes: `origin` is the fork
  `Ratish1/sglang-omni`, `upstream` is `sgl-project/sglang-omni`. You have push permission on
  upstream; PR branches are pushed to both.
- Pinned SGLang source for reading: `/Users/ratish/sglang`, must be at tag `v0.5.19`.
- Worktrees live under `/Users/ratish/sglang-omni/.worktrees/<name>`. `tasks/` is gitignored
  everywhere; a task directory is committed with `git add -f tasks/<dir>` on an analysis branch.
- Rules files the hooks and reviews follow: `/Users/ratish/sglang/.claude/rules/comment-style.md`,
  `general-code-style.md`, the omni `unit-test-admission.md`.
- Scratch for a session: the session's scratchpad directory, never `/tmp`.

### H100 box

- Container checkout: `/sgl-workspace/sglang-omni`. Results come back as zip or tar archives the
  user downloads to `/results/<name>` on the Mac; the readouts are written from those.
- One server per arm from that arm's worktree, started with `python -m sglang_omni.cli serve`.
  Never the `sgl-omni` console script: it imports the venv's editable install, which is the main
  checkout, from any directory. Archive `python -c "import sglang_omni; print(sglang_omni.__file__)"`
  per boot; the path must be under the arm's worktree or the boot is void.
- Gate every B boot on a log line only B prints. Write the exact line and its file:line into the
  runbook before the run.
- Weights from the default HF path; record the resolved snapshot.

### Worktree inventory on 2026-09-14 (branch, commits ahead of upstream main, last subject)

CosyVoice:

| worktree | branch | ahead | holds |
|---|---|---|---|
| cosyvoice-stack | perf/cosyvoice3-vocoder-load-path | 26 | the three PR stack, 13 + 5 + 8 commits |
| cosyvoice-utilization-20260912 | analysis/cosyvoice-utilization-20260912 | 41 | MECHANICS ledger, readouts, experiments E1 to E4, perfkit, diagnostics runbook |
| profie-cosy-workload | profie_cosy_workload (upstream branch) | 6 | `.claude/skills/model-profiling/cosyvoice3_default_nsys/` README, compute_sm_window.py, probe_one_request.py |
| cosyvoice-profile | perf/cosyvoice3-stream-scheduler-liveness-profile | 18 | NVTX instrumented tree for Nsight, never a measured boot |
| cosyvoice-stream-liveness | perf/cosyvoice3-stream-scheduler-liveness | 30 | the pre-stack history; superseded by the stack |

Qwen3-TTS, open PR branches:

| worktree | branch | ahead | PR |
|---|---|---|---|
| qwen3-tts-stage-ids-early | perf/qwen3-tts-stage-ids-early | 5 | #2123 |
| qwen3-tts-nonblocking-copies | perf/qwen3-tts-nonblocking-copies | 3 | #2126, applies after #2123 |
| qwen3-tts-reference-encoder-graphs | perf/qwen3-tts-reference-encoder-graphs | 5 | #2172 |

Qwen3-TTS, held or historical (read before reusing; none is evidence that its mechanism is in main):

| worktree | branch | ahead | what it is |
|---|---|---|---|
| qwen3-tts-predictor-chain | perf/qwen3-tts-predictor-chain | 37 | predictor chain history, most merged as #1947, #1971, #2057, #2108 |
| qwen3-tts-profiling | perf/qwen3-tts-profiling | 33 | profiler routes and torch trace triage; the step window method of #2123 |
| qwen3-tts-hidden-h2d-sync-v2 | perf/qwen3-tts-hidden-h2d-sync-v2 | 16 | the sync ledger `tasks/qwen3_tts_performance_pr_ledger_20260823.md`, `tasks/pytorch_optimization/design_method.md`, `models/qwen3_tts.md` |
| qwen3-tts-memory-provisioning, qwen3-tts-kv-pool | perf/... | 11, 4 | pool sizing history, merged as #2042 |
| qwen3-tts-predictor-startup-capture, predictor-capture, predictor-rope-store, bootstrap-graphs, prefix-prime | perf/... | 11, 4, 8, 4, 4 | capture history, merged as #1947, #2057, #2151 |
| qwen3-tts-single-repetition-penalty, -clean, qwen3-tts-suppress-mask, qwen3-tts-sampling-precision | fix/..., diagnostics/... | 3, 11, 4, 6 | repetition ownership work; #1750 merged the single application, the retraction contract remains (section 9) |
| qwen3-tts-e5-existing-traces | analysis/qwen3-tts-e5-existing-traces | 1 | offline E5 trace procedure `tasks/qwen3_tts_e5_investigation_20260912/H100_README.md` |
| qwen3-tts-stream-ready, stream-repro, vocoder-direct-decode, codec-precompile, decode-step, allocator-snapshot, async-waveform-publication, text-tokenizer-h2d, sampling-metadata-h2d, hidden-h2d-sync, predictor-warmup (branch perf/qwen3-tts-cudnn-attention) | | | one mechanism each; read `git log upstream/main..HEAD` before touching |

Related: talker-admission-cap, talker-step-syncs, talker-lookahead, step-ledger are Qwen3-Omni
talker work, not Qwen3-TTS.

### Creating a new branch for a slice

```bash
cd /Users/ratish/sglang-omni
git fetch upstream
git worktree add .worktrees/qwen3-tts-<slice> -b perf/qwen3-tts-<slice> upstream/main
cd .worktrees/qwen3-tts-<slice>
```

A stacked slice branches from the previous slice's head instead of upstream/main and its PR base is
that branch. Analysis material goes on `analysis/qwen3-tts-<topic>-<yyyymmdd>` with the task
directory force added.

On the box, fetch the branch from the fork and add a worktree; never apply a patch on top:

```bash
git fetch https://github.com/Ratish1/sglang-omni.git perf/qwen3-tts-<slice>
git worktree add /sgl-workspace/wt/qwen3-tts-<slice> FETCH_HEAD
```

## 2. Where the work stands

### 2.1 Fun-CosyVoice3 stack, all three approved and green on 2026-09-14

Stack #2173 links them; each PR's base is the previous branch.

| PR | branch, head | commits | what |
|---|---|---|---|
| #2169 | perf/cosyvoice3-vocoder-serving-loop, 4505a34fd | 13 | vocoder runs as steps from the serving loop: every queued message lands in state before a step, started streams ranked by playback slack, finals as steps; the WER fix |
| #2170 | perf/cosyvoice3-packed-flow-call, e893e6249 | 5 | every hop and every final is one packed Flow call; rows packed along the sequence, attention within each row (packed_dit.py) |
| #2171 | perf/cosyvoice3-vocoder-load-path, fd088faca | 8 | vocoder built before the KV pool is sized, per call host syncs dropped, one hop and one final warmed before readiness |

Census at the stack head (streaming c16, English seed-tts corpus, 1,088 requests, one boot per arm):

| read | main | stack |
|---|---|---|
| failures | 23 | 0 |
| WER | 6.82 | 1.13 |
| C50 continuity | 79.8 | 80.8 |
| req/s | 1.62 | 4.86 |
| audio s/s | 7.40 | 22.6 |
| first audio mean / p95 s | 0.91 / 2.17 | 1.83 / 2.43 |
| latency p99 s | 43.6 | 4.6 |

Nsight, 200 request window, both arms under the same tracer (PR #2170 body):

| arm | SM Issue | SMs Active | GR Active | Tensor |
|---|---|---|---|---|
| main streaming | 4.77 | 17.32 | 48.93 | 1.73 |
| stack streaming | 26.05 | 55.70 | 77.49 | 9.96 |
| main buffered | 10.77 | 31.59 | 65.65 | 4.00 |
| stack buffered | 10.99 | 32.30 | 63.48 | 4.11 |

Also measured: MOSS-TTS Local streaming c16 A/B (main 1f6b6843e against PR 1 head): req/s 6.929 to
12.869, TTFP p95 0.633 to 0.614 s, 200 of 200 on both. No change in what is emitted.

Open items on the stack, both recorded, neither blocking:

- **F4, PR #2170.** `select_step_participants` (models/fun_cosyvoice3/streaming_vocoder.py, the
  commit "batch every runnable hop regardless of its token window") ranks `started + unstarted`.
  With the config batch of 16 and c16 traffic, every runnable stream fits and the order never bites.
  Above 16 concurrent streams an unstarted stream waits for started streams to drain, with no aging.
  Measure before designing: c32 streaming A/B, main against the stack head, first audio p95 and C50.
  If it bites, the principled ranking is one slack order where an unstarted stream's slack is minus
  its waiting time, not a second queue.
- **F11, PR #2169, closed by two whole file reads on 2026-09-14.** On a streaming run of
  MOSS-TTS Local or dots.tts every hunk of #2169 is behaviour neutral: the loop's new branch is
  gated by `_has_ready_work`, which only CosyVoice overrides; the parked queue is empty on every
  chunk collector entry; the stream done path keeps the same lock holder and emission order. A
  MOSS-TTS Local c16 pair that read req/s 12.945 to 12.425 and C50 100 to 99.17 cannot be this PR;
  see `plans/10_streaming_scheduler_followups.md` on the CosyVoice analysis branch. Original note:
  `_collect_stream_chunk_batch` (scheduling/streaming_simple_scheduler.py:346)
  pulls from `_get_batch_message` (parked messages first, then the inbox) where main read the inbox
  only and skipped parked messages. Affects the coalescing schedulers: dots.tts (vocoder.py:152,
  batch 4), MOSS-TTS Local (streaming_vocoder.py:312, batch 8), Ming (streaming_vocoder.py:115, but
  its batch is pinned to 1 by config, unaffected). A batch is cut only when a non chunk message sits
  between two parked chunks, which is exactly the arrival order main violated; the cost is one extra
  pump. MOSS-TTS Local measured above: no regression. dots.tts not measured.
- First audio at c16 rose from 0.91 to 1.83 s mean: the packed step is launch bound, about 0.6 s per
  step; CUDA graphs for the packed step by total token bucket is the follow up (MECHANICS ledger,
  "Open").

Everything else that was tried and withdrawn is in
`.worktrees/cosyvoice-utilization-20260912/tasks/cosyvoice_utilization_20260912/MECHANICS.md`,
tables "Withdrawn" and "Open". Read it before proposing anything for CosyVoice.

### 2.2 Qwen3-TTS, open PRs on 2026-09-14

All three against main, review required, mergeable.

| PR | mechanism | measured | depends on |
|---|---|---|---|
| #2123 stage the token ids before the code predictor | the pinned copy of the layer 0 id and its event are enqueued before `code_predictor_forward`, so `_finalize`'s one blocking wait no longer covers the 4 ms predictor replay; host tail and next launch overlap the replay | step wall 8.04 to 6.25 ms at 1 row, 8.57 to 6.78 ms at 16 rows; c1 QPS +14.1 percent; c16 QPS +1.2 percent (churn steps still block); 1088 of 1088 WAVs identical seeded | none |
| #2126 copy the sampling restage and the finish payload without blocking | pinned ping pong source for the six sampling buffers in `prepare_decode_buffers`, non blocking; finished request's codes copied to pinned with an event on `result_ready_event`, waited in the stage runtime before routing | `prepare_decode_buffers` 3.15 to 0.08 ms, `apply_sglang_qwen3_tts_result` 2.99 to 0.16 ms at c16 churn steps; c16 QPS +1.9 percent, p99 -5.0 percent; bit exact | #2123 |
| #2172 replay reference encodes through captured graphs at bucketed lengths | Mimi conv padding integers moved to host at load (no per conv D2H sync), 16 quantizers not 32, one graph per reference length bucket 32 to 256 frames on the batcher's stream | TTFC p95 198 to 142 ms, preprocessing p95 107 to 47 ms, req/s flat (closed loop c16 is talker bound); SM Issue 13.9 to 14.7, GR 74.5 to 78.2; codes at a padded length not bit identical, quality census is the gate and holds | logically after #2123 (its regression was measured under early ids) |

Merged Qwen3-TTS performance PRs, newest first: #2151 reference prefixed bootstraps through
captured window graphs (09-13), #2108 predictor residual inside the following norm (09-11), #2057
rope kernel writes the predictor cache (09-09), #2042 resident stages before the KV pool (09-09),
#1971 dead work out of the predictor replay, batched feedback write (09-07), #1947 predictor CUDA
graphs at startup (09-04), #1786 vocoder graph capture readiness race (08-28), #1750 repetition
penalty once through SGLang (08-26). #1462 (not yours) introduced the persistent device masks and
pinned sampled ids the current per step owner is built around.

### 2.3 Held Qwen3-TTS work and why

- **Repetition ownership and retraction.** SGLang's `SamplingBatchInfo.from_schedule_batch`
  constructs a fresh penalizer orchestrator after re-prefill; Qwen3-TTS keeps `req.output_ids`
  across retraction, so the omni duplicate accidentally covered a generic SGLang history
  restoration gap. Order: a public SGLang batch construction contract that initializes the output
  history penalizers from retained `req.output_ids`; prove first prefill unchanged and re-prefill
  equal to uninterrupted decode; then split codec suppression from the repetition mask in omni.
  Never mutate SGLang private penalizer tensors from the runner. Full text:
  `.worktrees/qwen3-tts-hidden-h2d-sync-v2/tasks/pytorch_optimization/models/qwen3_tts.md`.
- **Hidden H2D sync branch.** Seven ranges proved free of blocking copies, no end to end speedup
  proved; the mechanisms that mattered were re-derived per owner and shipped as #2123 and #2126.
  The ledger is the reference for the sync detector method and for what not to resurrect.

## 3. The strategy: origin first, from first principles

### 3.1 What utilization means

Read four GPU metrics from Nsight's `GPU_METRICS` sampling, never `nvidia-smi utilization.gpu`:

- **SM Issue** (issue slots used): the utilization rate. This is the number to raise.
- **SMs Active**: a warp was resident. High SMs Active with low SM Issue means stalled, not busy.
- **GR Active**: the graphics engine had work. Low GR Active means the device was idle, which is a
  host problem: launches, syncs, Python.
- **Tensor Active**: tensor pipes busy; low with high SM Issue means the kernels are not matmuls.

A window mean is the arithmetic mean of every 100 Hz sample in the window, zeros included. It equals
the conditional means weighted by coverage, so split it by what was running (perfkit's "GPU metrics
by kernels present") to separate temporal loss (device idle) from spatial loss (kernels too small).

### 3.2 The four origins, in the order they are attacked

Every loss of SM Issue in a TTS pipeline has been one of these, and the order below is the order
of cost to fix and size of gain. Do not skip ahead.

1. **Serialization: the device waits for the host, or a thread waits for a lock.** Blocking
   `.item()`, `.cpu()`, `torch.tensor(list, device=cuda)`, pageable copies, `F.pad` reading device
   integers, `masks.sum()` on a bool mask, a step that cannot start until the inbox is empty. Read
   with `sync_calls_per_call` and the OSRT and GIL rows. Fix by deleting the transfer, keeping state
   on device, copying at the consumer boundary, or deferring the host read to its commit point
   (`design_method.md`, "Rewrite selection order"). #2123, #2126, #2172's conv padding, the
   CosyVoice conv cache and HiFT constants were all this.
2. **Launch bound: host microseconds per kernel exceed kernel microseconds.** `gpu_over_host`
   well below 1 with `mean_kernel_us` near the launch cost. Fix with fewer launches: CUDA graphs
   keyed by the shape the layout actually varies in (batch rows, or total tokens after packing),
   fused per token modules, dead work removed from replays. #1947, #1971, #2151, #2172 and the
   CosyVoice packed step follow up are this. Graph capture keys are decided by the layout, never by
   a table of observed shapes (config.py's 55 capture shapes are a symptom of the padded layout).
3. **Layout: padded rows pay for the widest row.** A batch padded to its longest member spends
   rows times width; the packed layout spends total tokens. Attention is the one module that must
   see a row boundary. Fix the layout, never the admission: a budget that keeps wide rows out of a
   step encodes a host to GPU ratio and pins to one GPU. packed_dit.py is the reference: per token
   modules on `(1, total, channels)`, attention on rows scattered to the padded layout under a key
   mask and the chunk causal mask, rotary gathered per position.
4. **Ordering: the right work is not the next work.** Started streams starve behind new ones, or
   finals starve behind hops, or a batch coalesces across a message that arrived earlier. Fix in the
   scheduler with one ranking derived from the playback contract (slack = audio emitted minus wall
   time since first emit), never with a second queue or a timer.

Kernels themselves come last. Qualify a faster kernel on the model's own activation ranges before
anything else (rule 6). A kernel that is exact on random inputs and wrong on the model is worse than
the slow one.

### 3.3 What counts as a fix

A change is a fix when it removes the line that pays and the code after it has fewer moving parts.
It is a heuristic when it adds a number, a branch, or a mode that decides how much of the defect to
tolerate. Ask, for every proposed line: which measurement pins this value, and what would it be on an
H200? If the answer is "the same, because it is a count of tokens" it is a fix. If the answer is "we
would retune it" it is withdrawn before it is written.

## 4. The process, one step at a time

```text
[pick the model and the point]          section 6: c1 and c16, streaming and buffered
        |
[read the tracker and the open PRs]     issues 1022, 1018, 1307; gh pr list --search <model>
        |
[read the whole files of the path]      never excerpts; write the ASCII flow with file:line
        |
[boot both arms, capture]               section 5: Nsight on the instrumented tree, census on the bare tree
        |
[analyse the exports]                   compute_sm_window.py, perfkit slice_trace.py, sqlite in the container
        |
[write the origin ledger]               MECHANICS.md: mechanism | origin line | evidence | commit | status
        |
[candidate sheet per origin]            section 4.5; withdrawn if any answer is a constant
        |
[bit for bit experiment]                E<n>: every path against one truth on saved real inputs
        |
[code, one mechanical commit each]      section 7; discuss runtime changes first
        |
[unit tests on the box]                 the branch venv; never local
        |
[census A/B, one boot per arm]          section 6; delta table; quality in band
        |
[PR, stacked if it shares the run]      section 8
        |
[review round]                          verify every finding against the code; only CONFIRMED gets a change
```

### 4.1 Pick the target

One model, one output mode, one concurrency per slice. c16 carries the SM target; c1 carries first
audio and the seeded identity gate. A slice is measured at both. Streaming and buffered are separate
runs with separate tables.

### 4.2 Read before designing

- Tracker issues 1022, 1018, 1307 and every open PR for the model. An open PR is not evidence its
  mechanism is in the baseline; read its head.
- Whole files, not excerpts, for every file on the path: the scheduler, the model runner, the stage
  factory, the config, the vocoder, the request builders. Write the flow as ASCII with file:line at
  every hop and put it in the plan doc.
- The pinned SGLang counterpart for every omni mechanism you plan to add or remove (rule 4).
- The withdrawn table of the model's MECHANICS ledger, so nothing is tried twice.

### 4.3 The plan doc

One directory `tasks/<model>_<topic>_<yyyymmdd>/` on an analysis branch, containing:

- `MECHANICS.md`, under 50 lines: one line per mechanism, columns mechanism, origin, evidence,
  commit, status; a withdrawn table with the reason; an open table with the origin to read first and
  the evidence expected. Updated with every commit.
- `readouts/<nn>_<arm>_<point>_<head>_<date>.md`: verdict first, provenance second, tables only.
- `experiments/e<n>_<name>.py`: one numerics question each, saves its inputs.
- `research/<topic>.md`: a source read with file:line, for anything claimed about a dependency.

Plan docs assert validated facts only. An unknown becomes a validation task with a name, never a
sentence with "should".

### 4.4 The ASCII flow

Every code path analysis comes with one, in the doc and in the reply. The stack's serving loop, as
an example of the form:

```text
StreamingSimpleScheduler.start            streaming_simple_scheduler.py:143
  _has_ready_work                          :133  any runnable stream under _state_lock
  _get_batch_message                       :209  parked first, then inbox, raises Empty
  _handle_message                          :186  new_request | stream_chunk | stream_done
    _collect_stream_chunk_batch            :346  coalesce parked+inbox chunks, cut at a non chunk
  _run_ready_step                          streaming_vocoder.py:391  _pump_one_step under the lock
    select_step_participants               models/fun_cosyvoice3/streaming_vocoder.py:304
    build_step_plan / run_step             :331 / :336  one packed Flow call, then HiFT per row
    on_step_failure                        scheduling/streaming_vocoder.py:506  abort every participant
```

### 4.5 The candidate sheet

No production worktree until every line is filled with a fact:

```text
Current source and exact runtime call path:
Tensor shape, dtype, device, layout, maximum bound:
Creator and authoritative owner:
Producer stream/thread/process and consumer stream/thread/process:
Current synchronization or launch mechanism and the trace row that shows it:
Next unavoidable host dependency after the change:
Expected recovered overlap, in ms per step, and added work:
Selected rewrite and the rejected alternatives (delete, device resident, view, consumer copy, deferred read):
Numerical oracle (bit exact, or which census band):
Batch growth/shrink, retract, abort, graph, streaming and non streaming consequences:
Which line of the sheet would change on an H200: (must be "none")
```

### 4.6 The experiment

Name it `E<n>`. One question. Every candidate path against the same truth, per call, on real
activations captured from a server (saved to disk with the script that made them). Report SNR in dB
per call and the count of calls under 40 dB, worst first. For a layout change the truth is the
unpacked padded call in float64; for a sync change the truth is byte identity of the outputs on a
seeded c1 pass. A step cost model (E3, E4 style) is allowed to size a decision but never to set a
constant in code.

### 4.7 Code, then tests, then the census

Section 7 for the code, section 6 for the census. Commit series: every commit reviewed and verified
before the next. A slice whose parts were validated by the same run ships as one PR; parts validated
by different runs are different PRs on the same stack.

## 5. Profiling

### 5.1 Two trees, two purposes

- The **bare tree** (the branch head) is the only thing that is measured for the census.
- The **instrumented tree** (branch head plus the NVTX commit, e.g. cosyvoice-profile 622bcd198,
  qwen3-tts-profiling for Qwen3-TTS) is the only thing Nsight runs on. Never quote req/s from a
  traced run against a bare run; compare traced with traced.

Do not add `python-gil` or `osrt` tracing to an arm whose cost is host launches: the tracer's own
overhead lands on that arm and moves the comparison. Use one capture with them on only to attribute
lock time, and say so.

### 5.2 The Nsight capture, whole serve

From `.claude/skills/model-profiling/cosyvoice3_default_nsys/README.md` on branch
`profie_cosy_workload` (upstream). The CosyVoice recipe generalizes; swap the model path and the
benchmark flags.

```bash
nsys profile \
  --trace=cuda,nvtx \
  --gpu-metrics-devices=cuda-visible \
  --gpu-metrics-set=gh100 \
  --gpu-metrics-frequency=100 \
  -o <arm>/nsys/serve --force-overwrite true \
  -- python -m sglang_omni.cli serve --model-path <model> --port 8000
```

Wrap the whole serve; never attach by PID. Nsight 2026.4 or later. After the workload, `TERM` the
serve only, wait for the `.nsys-rep` to stop growing and the nsys parent to exit; never `SIGKILL`
nsys. Keep only the `.nsys-rep`; export to SQLite in the container:

```bash
nsys export --type sqlite --force-overwrite=true -o <arm>/nsys/serve.sqlite <arm>/nsys/serve.nsys-rep
```

### 5.3 The workload and the window

One serve lifetime: probe (1) + preload (warmup 16 + 32 samples) + headline (warmup 16 + 200
samples). Only the headline 200 is the window; the probe and preload put 49 requests in front and
dilute the means. The window is `Benchmarking 200 requests` to `Results saved` in the headline
`bench.log`, converted to session relative nanoseconds through the `localTime` of
`TARGET_INFO_SESSION_START_TIME`. Metric ids are looked up by name in `TARGET_INFO_GPU_METRICS`.

```bash
python .claude/skills/model-profiling/cosyvoice3_default_nsys/compute_sm_window.py \
  --sqlite <arm>/nsys/serve.sqlite --bench-log <arm>/benches/headline200/bench.log
```

It prints each metric's mean, the sample count and the metric id, with the window length and the
headline req/s next to it. A 32 request window is invalid: it is ramp and drain. A tighter window,
first headline prefill to last kernel, is 1 to 2 s shorter and 0.1 to 0.3 pp lower; use one cut on
both arms. On the profie branch the benchmark flag is `--max-concurrency`; on main it is
`--concurrency`. Check `benchmarks/eval/benchmark_tts_seedtts.py` of the branch under test.

### 5.4 Reading a trace: the perfkit

`tasks/cosyvoice_utilization_20260912/perfkit/slice_trace.py` (analysis branch) reads one SQLite
export made with the pipeline annotations (`SGLANG_OMNI_PIPELINE_NVTX=1`,
`--trace=cuda,nvtx,osrt --cuda-graph-trace=node`) and writes the ledgers: AR thread host wall against
GPU union per step; vocoder thread budget; per call host, GPU union, host µs per launch, kernel
count, sync API counts; request timeline (first audio decomposition); streaming hops (ready, start,
run, queue delay); GPU idle gaps intersected with thread states; GPU metrics conditional on which
stage's kernels were present. Standard library only; runs on the Mac against the returned SQLite.

Attribution rules it enforces, and that every hand query must also follow: a kernel belongs to the
thread whose runtime API row shares its process and correlation id (graph node kernels share the id
of their `cudaGraphLaunch`); a kernel's stage is the innermost annotated range containing that launch
on that thread; ranges are never inferred from kernel names; a bare `correlationId` is not globally
unique; event handles are reused, join on `eventSyncId` plus process and context and verify order;
queue durations overlap and are never summed into wall time.

How to read the numbers:

- `gpu_over_host` well below 1 with `mean_kernel_us` near the launch cost: launch bound, fewer
  launches is the fix, not faster kernels.
- `sync_calls_per_call` above a handful: host tensors cross to the device inside the call; each is
  a pipeline drain.
- `host_us_per_kernel` on one thread rising while another thread is in `execute`: interpreter
  contention between two Python loops.
- Large `queue_delay` on finals: the pump loop starves finals; on batched follow ups: the vocoder is
  behind the AR.
- py-spy `--gil` gives ownership shares only; lock held time comes from the OSRT
  `pthread_cond_timedwait` rows per thread.

Thread identity comes from sampled stacks with real function names (`_collect_codes`,
`code_predictor_forward`, `_run_initial_worker`, `_Qwen3TTSRefCodeBatcher._run`), never from a
thread number. The E5 README lists the stack evidence per role.

### 5.5 The torch profiler step window

For a per step cost (the tables in #2123 and #2126): boot the bare tree, run the corpus at the
concurrency of interest, then `POST /start_profile` and `POST /stop_profile` on the server
(`sglang_omni/serve/launcher.py:265,350`; `/start_request_profile` and `/stop_request_profile` at
:318,361 give the event recorder JSONL without the torch profiler). Report p50 step wall, device idle
inside the step, replay wall and kernels per replay, at 1 row and at the target row count, and at a
churn step (a request joined or finished) separately from a steady step. Never quote a profiler
window as throughput.

### 5.6 Sync detection

`torch.cuda.set_sync_debug_mode("error")` on a c1 detector run finds synchronizing calls; it is not
exhaustive. CUDA graph capture of a region is the stricter detector for unpinned host copies inside
it. Count `cudaStreamSynchronize`, `cudaMemcpy` (pageable), `cudaEventSynchronize`,
`cudaStreamWaitEvent`, `cudaEventQuery` per call in the trace; `cudaStreamWaitEvent` is issued even
when the event has already completed, so zero wait calls does not mean inputs were ready.

For each sync record the window: last useful host launch before it, the sync, host dispatch after
return, recovered GPU occupancy, next unavoidable sync or external commit. A candidate is plausible
only when it moves every queue draining boundary in that window far enough for overlap; removing one
wait while the next drains the queue 200 µs later recovers nothing (the hidden H2D branch result).

### 5.7 What never to do in a profile

- Never attach nsys by PID; never SIGKILL it; never run it on the Mac.
- Never run heavy SQLite scans as background commands on the Mac; run them in the container through
  a runbook script and return compact CSVs.
- Never cut the window at "the first kernel after init".
- Never mix a traced req/s with a bare req/s, or a profiler window with throughput.
- Never read `nvidia-smi utilization.gpu` as utilization.
- Never claim a thread's role from its number, a kernel's stage from its name, or a request's work
  from a nearby timestamp.
- Never let one arm carry tracer flags the other does not.

## 6. Benchmarking

### 6.1 The protocol, standing since 2026-09-09 and 2026-09-11

- Unseeded, `--warmup 1`, full English seed-tts corpus (1,088 samples for TTS,
  `zhaochenyang20/seed-tts-eval-arrow`), one boot per arm and point.
- Points: c1 and c16; streaming (`--stream`) and buffered are separate runs.
- The census is the measurement; repeat only inside about 2 percent.
- A seeded c1 pass (`--seed 1234` on both arms) is the byte identity gate for a change that claims
  bit exactness, and gives c1 speed.
- Stacked slices reuse the previous B as A. Four boots on B for a bit exact slice.
- Never through the CI harness; the CI thresholds are a strictness decision, never proposed for
  change.
- Both arms the same day under the same host load; record `nvidia-smi dmon` on the GPU and the
  compute apps on the others; quote the paired delta next to the census.

### 6.2 The commands

Server, per arm, from the arm's worktree (the `CUDA_VISIBLE_DEVICES` prefix is the only env var on a
measured boot; anything else is a protocol change and goes in the readout):

```bash
BOOT=artifacts/<model>/<arm>-<mode>-en-c<n>; mkdir -p "$BOOT"
git rev-parse HEAD > "$BOOT/head.txt"
python -c "import sglang_omni; print(sglang_omni.__file__)" > "$BOOT/import_path.txt"
nvidia-smi > "$BOOT/gpus_before.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv >> "$BOOT/gpus_before.txt"
nvidia-smi dmon -i 0 -s pucv -d 1 > "$BOOT/dmon.log" &
CUDA_VISIBLE_DEVICES=0 python -m sglang_omni.cli serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 8000 2>&1 | tee "$BOOT/serve.log"
```

Client, from the repository root of the main checkout (the benchmark package), server still from the
arm's worktree:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --lang en \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --use-existing-server --host 127.0.0.1 --port 8000 \
  --concurrency 16 --warmup 1 --stream \
  --generate-only --output-dir "$BOOT/bench"
```

Drop `--stream` for buffered. Add `--seed 1234` only for the identity gate, on both arms. The client
passes `--max-new-tokens 2048` by default, which overrides the model's own length contract (CosyVoice
caps at 20 times the text tokens; six requests in one run ran to 81.92 s of audio for four words).
Same rows on every arm, so it is noise in the delta, but record it.

Quality, on the existing WAVs, never by synthesizing again:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts ... --transcribe-only --asr-model-path Qwen/Qwen3-ASR-1.7B --output-dir "$BOOT/bench"
python -m benchmarks.eval.benchmark_tts_seedtts ... --similarity-only --similarity-checkpoint <wavlm head> --output-dir "$BOOT/bench"
```

For CosyVoice the model is `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` and the diagnostics runner
`tasks/cosyvoice_utilization_20260912/diagnostics/run_seedtts.py` wraps the same client with the
identity files (`inputs.json`, `experiment.json`).

### 6.3 What a readout holds

Verdict first, provenance second. One table per point. Streaming rows: req/s, audio s/s, RTF mean,
TTFC (first chunk) mean, p50, p95, p99, inter chunk mean, failures, WER, speaker similarity,
continuity (C50) where the model has it, peak memory. Buffered rows: req/s, latency median, p95,
p99, RTF, WER, similarity, peak memory. Deltas as absolute and percent. Quality noise is one line.
No reruns or extra boots narrated in a PR body; the readout on the analysis branch keeps them.

A B boot is valid only when `head.txt` is the branch head, `import_path.txt` is under the arm's
worktree, the branch only log line appears, `foreign_pids` recorded, and no memory drop inside the
window. Otherwise void; say so and rerun.

## 7. Code design

### 7.1 What a change looks like

- One mechanical change per commit, subject short and lowercase, no PR or issue refs, no trailers
  (co-author trailers only when asked, on one empty commit). Examples from the stack: "pack the flow
  rows along the sequence and attend within each row", "batch finals through the packed
  non-streaming flow adapter", "install the conv cache patch once".
- Names are mechanical with cheap words: split, not plan; rows, not batch_plan; `hop_batch`,
  `leftover_batch`, `pack_rows`, `gather_rows`, `scatter_rows`.
- A function does one thing in the layout it is named for. `generate_flow_packed` takes packed rows
  and returns the padded layout the mel split reads; it says so in three lines.
- Comments: 1 to 3 lines, `# note(ratish):` on ours, only where the next reader would otherwise
  ask why. Never a hardware name, never a vendor, never "SGLang does this", never a doc or discussion
  reference, never backticks. Delete an upstream comment that has become false ("Keep in sync with
  vocoder token_hop_len" was removed when the coupling ended).
- f-strings for log lines built from more than two values.
- Decorators sit on the callable they gate. `@torch.inference_mode()` on a dataclass is a warning
  and a wrapper, not inference mode (fixed in #2170).
- Module level flags over instance sentinels: `CAUSAL_CONV_CACHE_PATCHED` replaced a
  `getattr(obj, "_patched", False)` check. No `getattr`/`hasattr` defensive reads on objects whose
  type is known.
- No defensive branches beyond the plan: a `numel() > 0` guard on a path the hop invariant already
  makes non empty is a branch nobody can reach; leave it out. If a subclass could break an invariant,
  state the invariant in the hook's docstring, do not add a third return state.
- Lazy imports only where the module is optional on a platform; otherwise top of file.
- Host syncs: `~mask.any(dim=-1, keepdim=True)` not `mask.sum(dim=-1) == 0` (the sum materializes
  an int64 copy, 8 bytes per element, the 2.29 GiB in the OOM traceback). Build masks on device
  from lengths, never from `.item()` reads.

### 7.2 Good and bad, from this work

| bad, withdrawn | why | good, shipped |
|---|---|---|
| `DEFAULT_FLOW_STEP_PAD_FRAMES = 3500`, a step padding budget | encodes an H100 host to GPU ratio; retuned on another GPU; treats the layout defect as an admission problem | rows packed along the sequence; each row pays its own frames |
| KV pool cap derived from an admission bound | stops preallocation, wastes HBM; the real question was the vocoder's working set | build the vocoder before the pool is sized so its 3.4 GB is out of the pool's view |
| warmup at cap derived shapes | shapes defined by a cap the step no longer has; negative token count at small caps | warm one real hop and one real final before readiness |
| flashinfer row attention with a workspace constant | 0.6.18's finite sentinel zeroes rows whose logits sit below minus 5e4; the DiT has no QK norm and does exactly that | SDPA on rows scattered to the padded layout under an IEEE mask; bit identical to `DiT.forward` |
| replay test with a recorded c16 inbox fixture | incident specific, not a shipped contract | tests of the ranking contract on fakes with real resolved shapes |
| tri-state return from `_pump_one_step` to cover an empty failure list | the hook's contract already makes it unreachable | keep the contract in the docstring, two states |
| `torch.tensor(list, device=cuda)` six times in `prepare_decode_buffers` | pageable H2D, each a queue drain on churn steps | two pinned sources alternating with the event of their last copy |
| pinned copy of the layer 0 id after the predictor | the one blocking wait covers a 4 ms replay the host never reads | enqueue the copy and its event before the predictor |
| capture shape table of 55 (batch, frames) pairs | a symptom of padding; keys should follow the layout | (open) graphs keyed by total tokens after packing |

### 7.3 Tests

Test shipped contracts and edge cases: off grid and boundary inputs, not only shipped values
(read `PrefillAdder` before claiming a per forward ceiling). Fakes model real resolved shapes and,
after #2170, real position dependence (the fake packed estimator adds a term from `rows.positions`
so a wrong gather order fails). No comments inside tests. Grep the family classes for
`@abstractmethod` before deleting an override. An unrun test is never "good to go": it runs on the
box in the branch venv before the PR opens. Tests, runbooks and `tasks/` docs may be pushed
directly; runtime code is discussed first.

## 8. Git, stacks, PR bodies, review

### 8.1 Mechanics

```bash
# fix into an owning commit
git commit --fixup=<sha>
GIT_SEQUENCE_EDITOR=true git rebase -i --autosquash --update-refs upstream/main
# verify the squashed diff is exactly the edit
git diff <old head> <new head> --stat
# push both remotes with lease, all stack branches
git push --force-with-lease upstream <b1> <b2> <b3>
git push --force-with-lease origin   <b1> <b2> <b3>
# confirm PR commit counts
gh pr view <n> --json commits --jq '.commits|length'
```

Stack PRs: PR 1 against main, PR 2 against branch 1, PR 3 against branch 2, then
`gh stack link <n1> <n2> <n3>`. Bases must be upstream repo branches. Reviewers test the stack
head. A fixup for PR 1 rebases 2 and 3 automatically with `--update-refs`.

zsh: an unquoted `$VAR` is not word split (use `bash -c` for scripts that need it); `$c:path` is a
history modifier (write `"${c}:path"`); a bare `===` is an equals expansion (quote it).

Never bare `git stash`; the stash is shared across worktrees. Use a WIP commit.

### 8.2 PR body

Sections: summary (the mechanism, in the order the code runs), changes (file by file, one line
each, tests last), testing (the census tables, B against A only, deltas, the Nsight rows when they
exist with the window stated, one line on quality noise, one line on a known follow up). No testing
history, no reruns, no comparisons with a third PR. Titles in the house style: one scope tag then a
verb phrase naming the mechanism, e.g. "[Fun-CosyVoice3] Run each vocoder step as one Flow call
over rows packed along the sequence".

### 8.3 Review handling

Every finding is verified against the code before an answer: read the line, find every caller,
find every override in every model (`grep -rn "def <hook>" sglang_omni`), find the test that pins
it. Answer with the file and line. A finding that names an unreachable state gets the invariant that
makes it unreachable, not a defensive branch. A finding that names a real gap gets a fixup into the
owning commit. Review comments you write on others' PRs: 1 to 2 plain lines, in the user's voice, on
a pending review, only about what cannot break any hardware path.

## 9. Qwen3-TTS: mechanics, remaining bottlenecks, next targets

### 9.1 The step, after #2123 and #2126

```text
talker backbone (SGLang decode, CUDA graph)           model_runner.py  _collect_codes
  sample layer 0 id (omni suppression mask, SGLang sampler)
  stage the id: pinned copy + event                    before code_predictor_forward   (#2123)
  code predictor chain: 1 graph replay, 4.0 ms, 1,062 kernels   (#1947, #1971, #2057, #2108)
  feedback write into next step's talker input (device views)
  stream chunk leaves as a device view + event; vocoder waits the event   (#2046)
scheduler tail: prepare_decode_buffers restage (pinned ping pong, #2126)
                apply_sglang_qwen3_tts_result finish copy (pinned + result_ready_event, #2126)
```

Measured: step wall 6.25 ms at 1 row, 6.78 ms at 16 rows, device idle inside the step about 1.0 ms
(the in graph gaps of the replay). The step is device bound; c16 closed loop is talker bound, so
first chunk moves and req/s does not (#2172 census: req/s 15.88 to 15.89, TTFC p95 198 to 142 ms).

### 9.2 Remaining origins, with the line to read first and the evidence expected

| origin | read first | evidence that decides |
|---|---|---|
| predictor replay: 1,062 kernels for 4.0 ms, in graph gaps about 1 ms of a 6.3 ms step | the captured predictor graph (#1947 capture path, #1971 dead work list) and the per layer kernel list from `--cuda-graph-trace=node` | kernel count per predictor layer and mean kernel µs; the gaps between nodes; whether the chain is launch shaped (many tiny kernels) or dependency shaped |
| talker backbone step at 16 rows: 6.78 minus predictor 4.27 minus sample | SGLang's decode graph for the talker (pinned tree, `python/sglang/srt/model_executor/cuda_graph_runner.py`), attention backend chosen at boot (startup log) | SM Issue conditional on backbone kernels present; tokens per second per row against the 1 row step |
| SM Issue 14.7 at c16 with GR 78: kernels are small, not the device idle | perfkit "GPU metrics by kernels present" for the Qwen3-TTS tree | which stage's kernels carry the low issue rate; if it is the predictor, the fix is inside the graph (fusion, fewer nodes), not outside it |
| vocoder path: scalar D2H, code H2D, waveform D2H per decode (`models/qwen3_tts/streaming_vocoder.py`, `_run_initial_batch`, `_run_followup_batch`) | the sync counts per call from the trace | `sync_calls_per_call` on the vocoder threads; whether the initial worker waits a later chunk's event than its first plan needs (E5 open question: the planner concatenates retained chunks before slicing, so the dependency is real today and must be changed at input selection, not at the wait) |
| reference encode above 256 frames takes the eager path (#2172) | `reference_encoder_cuda_graph.py`, bucket list | share of references over 20.5 s in the corpus; TTFC p99 (284 ms) row |
| repetition penalty ownership across retraction | section 2.3 | seeded token parity uninterrupted against retract and re-prefill |
| preprocessing: text tokenizer H2D, speaker embedding publication, cache key D2H | `request_builders.py` per range; the sync ledger's seven ranges | per request preprocessing segment p50 and p95 (23.5 and 46.8 ms after #2172) |

### 9.3 Order of attack

1. Land #2123, #2126, #2172 (review round, then merge in that order).
2. Nsight, 200 request window, streaming c16, on the instrumented tree at the #2172 head, both
   arms (main and head) same day. Read the conditional SM Issue by stage. This decides between the
   predictor graph and the backbone as the next origin.
3. If the predictor: per node kernel list, fuse or drop nodes, bit exact gate on the codes (seeded
   c1, 1088 of 1088). If the backbone: read the attention backend and the graph bucket in the startup
   log first; anything here is an SGLang question and lands at an omni seam or not at all.
4. Vocoder syncs, one PR per owner, byte identity on the WAVs.
5. First audio at c1 after the above: the `admission to first audio` decomposition from the
   request timeline ledger.

## 10. Subagents

- Opus 5 medium only. Trivial work: unpacking archives, log greps, SQLite and event queries,
  building tables, explaining the mechanics of a file. Never design, never decide, never final code.
- Brief them with the exact files and lines, the question, the output format, and the rules that
  bind (no local pytest, no runtime edits). Verify every claim yourself against the code before it
  enters a doc; the "bf16 rounding" claim about flashinfer was wrong and the source read found the
  finite sentinel.
- At most 5 concurrent. A review subagent (high effort) is for a brutal read of a finished PR stack
  within scope, with a blast radius table across every model that shares the changed base class;
  its findings are each re-derived before an answer.

## 11. Checklists

Before a boot:
- [ ] arm worktree at the head in the readout, `head.txt` and `import_path.txt` archived
- [ ] the branch only log line named with file:line, grep command ready
- [ ] other GPUs and compute apps recorded, dmon running
- [ ] client flags identical across arms: corpus, `--warmup 1`, seed or no seed, `--stream` or not

Before a commit:
- [ ] one mechanism, one subject, lowercase, no refs
- [ ] no constant a measurement does not pin, no hardware name anywhere
- [ ] comments 1 to 3 lines, `# note(ratish):`, no vendor, no "SGLang does"
- [ ] no `getattr`/`hasattr` sentinels, no defensive branch past the plan, decorators on callables
- [ ] tests pin the contract on real shapes, no fixtures from an incident, no test comments
- [ ] pre-commit hooks passed on the Mac, unit tests passed on the box in the branch venv

Before a PR:
- [ ] census at c1 and c16 for the mode the change touches, one boot per arm, delta table
- [ ] seeded c1 identity if the change claims bit exactness
- [ ] quality in band, one line
- [ ] Nsight window stated with length and req/s, both arms traced, or no Nsight row at all
- [ ] MECHANICS ledger row updated with commit and status
- [ ] stack linked, bases right, commit counts confirmed on GitHub

Before answering a review:
- [ ] every finding re-read at the line, every caller and override listed, the pinning test named
- [ ] CONFIRMED, unreachable with the invariant, or not a scenario for this model, with file:line
