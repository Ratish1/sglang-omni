# 14. Runtime seams across the fleet: coalescing, async decode, the methodology

Scope: every AR stage that runs on OmniScheduler, TTS and ASR included, not
only Qwen3-Omni. Facts carry a file:line anchor into main 216e946dd or the
pinned sglang 0.5.18. Measurements cited are either merged PR bodies (the
record of what shipped) or this task's own runs (doc 06, doc 12). No
earlier campaign is relied on. What is not measured is a task in section 9.

Files read whole for this doc: sglang_omni/scheduling/omni_scheduler.py
(2683 lines), sglang_omni/model_runner/base.py (1008),
sglang_omni/model_runner/thinker_model_runner.py (491),
sglang_omni/models/qwen3_omni/talker_model_runner.py (504),
sglang_omni/models/qwen3_omni/talker_scheduler.py (159),
sglang_omni/model_runner/sglang_execution.py (129), sglang
managers/prefill_delayer.py (486) and min_free_slots_delayer.py (38), the
sglang overlap loop and run_batch (scheduler.py:1754-1826, :3626-3843) and
tp_worker.py:574-682.

## 1. The step model

An AR stage is one scheduler thread, one GPU stream, one KV pool, and a GIL
shared with whatever frontend threads the stage hosts. Its cycle is
max(host per step, GPU per step) plus the events that serialize the two
(a synchronous prefill, a drain, a host round trip inside a step). Batch
size raises GPU work slowly (memory bound decode) and host work in
proportion to the rows the host touches. Throughput is rows per cycle,
latency is cycles per request.

What each shipped mechanism can do under that model:

- Coalescing lowers the number of prefill steps, so a cost that is fixed
  per prefill step is paid fewer times. It lowers nothing per step.
- The one step lookahead hides GPU time under host time. It cannot hide
  host time under host time, so on a host bound stage it is worth the
  event wait it removes and no more.
- The observed KV reservation raises rows per cycle where the pool binds
  admission. Where the pool does not bind it changes nothing.

So the levers, in order of what they can move: host time per step (every
model), rows per cycle where memory binds (pool sizing, reservation), then
the serializing events (prefill, drain).

## 2. Prefill coalesce

### 2.1 Mechanics

Gate: `OmniScheduler.get_new_batch_prefill` (omni_scheduler.py:1336-1370).
Knobs: `__init__` :191-195, validated :290-310 (disabled for tp_size > 1
because the deadline reads each rank's clock). Enqueue stamp
`req._coalesce_enqueue_t` :1202. The gate runs once per loop iteration
through upstream `get_next_batch_to_run` (:1322-1334), which under load is
once per decode step.

```
get_new_batch_prefill(running_batch)                 omni_scheduler.py:1336
  |-- K <= 1 or chunked_req active ------------------> upstream   :1342
  |-- decode idle and not when_idle -----------------> upstream   :1344
  |-- requires_pending_builds and no build in flight
  |     (unless after_builds_during_decode and busy) -> upstream   :1347
  |-- queue empty or len(queue) >= K ----------------> upstream   :1358
  |-- oldest enqueue stamp older than T ms ----------> upstream   :1361
  '-- else hold: NextBatchPlan(batch_to_run=None)                 :1370
```

Upstream then admits the whole waiting queue through the PrefillAdder
(sglang scheduler.py:3157-3182, :3257-3274).

### 2.2 Who enables it

| stage | K | T ms | when_idle | requires_pending_builds | after_builds_during_decode | source |
|---|---|---|---|---|---|---|
| Qwen3-ASR | 16 | 40 | yes | yes | yes | models/qwen3_asr/config.py:80-84 |
| ArkASR | 16 | 32 | yes | yes | no | models/arkasr/config.py:67-70 |
| MOSS-Transcribe-Diarize | 4 | 12 | yes | yes | yes | models/moss_transcribe_diarize/config.py:63-67 |
| Whisper | 2 | 6 | yes | yes | no | models/whisper_asr/config.py:76-80 |
| Qwen3-Omni thinker | 0 | 60 | no | no | no | models/qwen3_omni/stages.py:1012-1014 |
| Qwen3-Omni talker | not plumbed | | | | | models/qwen3_omni/bootstrap.py:244-258 |
| Higgs, MOSS-TTS-Local, Fun-ASR | plumbed, default 0 | | | | | PR #1071, #1073 |

### 2.3 The shipped evidence

| model, PR | workload | result |
|---|---|---|
| Higgs, #1071 | seed-tts-eval voice clone, cap = conc 96, K 32, T 300 | qps 49.5 to 64.9 |
| MOSS-TTS-Local, #1073 | same, K 32, T 300 | qps 9.00 to 9.98 |
| Qwen3-ASR, #1404 | SeedTTS EN 1088, K 16, T 24, H100 | c8 +30.3%, c32 +32.5%, c1 +0.9%, prefill calls 953 to 362 at c32 |
| Qwen3-ASR, #1432 | H200, after_builds_during_decode | c8 +3.6%, c64 +20.5% |

The fixed cost it amortizes, per the same PRs and this task's trace: about
25 ms host per Higgs prefill step (#1071), 28 to 46 ms encoder plus prefill
per Qwen3-ASR admission on the scheduler thread (#1404's companion profile,
docs/developer_reference/qwen3_asr_concurrency_profile.md:166-170), 36.5 ms
host for 8.75 ms of kernels per talker prefill with every running decode
stopped for 45 ms (doc 06 section 3.4).

### 2.4 The sglang counterparts

sglang 0.5.18 ships two prefill holds, both opt in, both count based:

- `PrefillDelayer` (managers/prefill_delayer.py:72-344): hold when free
  slots are below the recent prefill size, `max_running_requests -
  running_bs < max_prefill_bs` (:266-269), `max_prefill_bs` the high
  watermark of the last 16 attempts (:22-46, scheduler.py:1221,
  :3173-3180). The wait is counted in forward passes, cap 30
  (server_args.py:3336-3338). A wall clock appears only as a 5000 ms cap on
  the optional queue trigger (:3354-3372). Requires the overlap loop
  (:141-143).
- `MinFreeSlotsDelayer` (managers/min_free_slots_delayer.py:25-38): hold
  while `running_bs > 0 and num_allocatable_reqs < min_free_slots`, no
  timer, checked before the adder (scheduler.py:3209-3217), opt in
  (server_args.py:3379-3392).

### 2.5 Verdict

The mechanism is correct for a cost that is fixed per step, and it is
measured. Two parts are fitted rather than derived: the wall clock unit,
since T ms is ceil(T / decode step period) steps and so means different
things on H100, H200 and across batch sizes, and the five per model pairs,
which read as each model's step period. A derived form keeps the gate and
changes its inputs: K from free slots and a 16 window high watermark of
prefill sizes, the hold counted in decode steps, T as the TTFT guard only.
That is one policy and one cap in place of five pairs. It is a hypothesis
until measured by the protocol in section 7, on every model that has the
gate on, at its CI concurrency and at c1.

The lever above the gate is the fixed cost itself: a prefill graph where
the model allows it, fewer launches per layer, and the request build off
the scheduler thread with its device work stream ordered. Each ms removed
from the prefill step lowers what any K and T can buy.

## 3. Async decode

### 3.1 Mechanics

Loop: `_event_loop_async_decode` (omni_scheduler.py:2493-2580). Runner
split: `execute_launch` (base.py:283-344), `execute_resolve` (:346-383),
hooks `post_decode_launch` (:700-722) and `post_decode_resolve` (:724-740).
Bridge: sglang_execution.py, single stream by design (:16-20), completion
event :124-128, FutureMap relay :100-122.

```
iteration                                             omni_scheduler.py
  recv_requests / process_input_requests                :2505-2507
  paused: resolve pending, sleep                        :2508-2512
  mixed chunk and a prefill is possible: resolve first  :2514-2522
  batch = get_next_batch_to_run()                       :2524
  lookahead if decode and bs >= async_decode_min_batch_size (2)
             and runner.lookahead_eligible(batch)       :2530-2535
  |-- launch N:  build ForwardBatch, before_decode, forward,
  |              post_decode_launch (sample, pinned D2H), publish,
  |              record event, batch.copy()             base.py:283-344
  |   resolve N-1: event.query or synchronize, post_decode_resolve,
  |              _finalize, drop rows finished before N-1,
  |              process_batch_result                   :2349-2386
  '-- sync (prefill, empty, small, ineligible): drain N-1, drop stale
      rows, run_batch, or idle sleep                    :2560-2576
```

On one stream, step N's kernels queue behind N-1, so the resolve of N-1
returns without waiting whenever N-1 finished during the host work of
launching N. Everything the host does between two launches overlaps the
GPU: resolve, result processing, recv, admission, batch selection,
ForwardBatch build, before_decode, the graph launch call.

### 3.2 What sglang's own overlap loop does differently

scheduler.py:1754-1826, tp_worker.py:574-682:

- every batch overlaps, prefill included, only consecutive prefills can be
  serialized by an env flag (:1844-1848),
- two streams plus a copy stream, with a WAR barrier between the
  scheduler's KV writes and the forward's reads (:1705-1716, :3665-3666,
  :3735-3740),
- a delayed sample: batch N is sampled after batch N-1's result is
  processed (:1817-1820, :3886-3909, tp_worker.py:628-647), so sampling
  state that depends on committed tokens never reads a stale view. Omni
  routes those batches to sync instead (`lookahead_eligible`,
  base.py:660-682).

Omni refuses sglang's loop (:2314-2330) because model runners read
`Req.inflight_middle_chunks` at forward time under a same iteration
result contract.

### 3.3 Who runs which loop

| stage | loop | min bs | hooks | source |
|---|---|---|---|---|
| Higgs | lookahead | 2 | own | models/higgs_tts/config.py:59, model_runner.py:116-186 |
| MOSS-TTS-Local | lookahead | 2 | own | #758 |
| Qwen3-ASR, ArkASR, MOSS-TD, Whisper | lookahead | 2 | base plain LM hooks | cookbooks, configs |
| Qwen3-Omni thinker | lookahead, sync for audio output and logprob | 2 | own | thinker_model_runner.py:411-490 |
| Qwen3-Omni talker | sync | | none | bootstrap.py:244-258 |
| Qwen3-TTS | sync | | `_stage_token_ids` only | doc 05 section 1.1 |
| Zonos2, Dots | hooks present | | own | models/zonos2/model_runner.py, models/dots_tts/model_runner.py |

### 3.4 The refactor list, from the code

State that should change, independent of any measurement because each item
is work the step does for nothing or a round trip the step does not need:

1. `isolate_sampling=True` on both execute paths (base.py:242, :302) runs
   `copy_for_forward` every step, which allocates a `[bs, vocab]` fp32
   tensor and runs the penalizer accumulate (sglang sampling_batch_info.py
   `update_penalties`) for runners that never read it. The talker samples
   in graph from its own buffers (doc 06 section 3.3). The copy exists for
   the lookahead's penalizer isolation and belongs only there.
2. The reporting token crosses the bus twice per step in opposite
   directions: device ids to a pinned buffer (`_stage_token_ids`
   base.py:121-134), `.tolist()` into `req.output_ids` (sglang result
   processor), then `_apply_repetition_penalty` rebuilds a device tensor
   from those lists with a pageable H2D (base.py:946-953). Token history
   belongs on the device, the host list is needed at finish and per chunk.
3. Four private pinned ping pong pairs implement one mechanism
   (`_token_id_host_bufs` base.py:117, `_host_staging_buffers` :109, the
   thinker's `_th_host_bufs` :31, Higgs' `_logprob_host_staging` :68).
   One slot indexed pinned ring owned by the bridge.
4. `async_decode_min_batch_size` defaults to 2 (:281, :2532) from one
   model's break even (#590). MOSS-TD measured +21 percent qps and -18
   percent p95 at c1 at 1 with c8 and c16 unchanged (#1454, open). Per
   model measurement, not a fleet default.
5. Delayed sample under lookahead instead of the sync fallback (3.2),
   where penalties are the default.
6. `_drop_stale_overrun` (:2401-2491) exists because the batch is built
   before the drain, and its extend branch (:2422-2483) re-slices per token
   fields by hand. Draining whenever a prefill is possible (the :2514-2522
   condition without the mixed chunk qualifier) deletes the branch at the
   cost of overlap on held iterations.
7. Runners without hooks run fully synchronous: the talker
   (talker_model_runner.py has no launch or resolve hook, scheduler built
   without async at bootstrap.py:244-258) and Qwen3-TTS. The talker's
   `post_decode` is device only apart from the id staging (two clones of
   the fixed address output buffers :133-134, an outbox put per row and a
   feedback append :145-154), so its launch half is mechanical once the two
   per step stream syncs are gone (E2, doc 05 section 7.2). #1320 built it
   and held on a measured loss for that reason.
8. Thinker audio output is refused (thinker_model_runner.py:428), so the
   speech thinker runs the synchronous loop at every batch size. PR D in
   doc 05 section 6 snapshots the hidden capture at launch.

## 4. The observed KV reservation, with a guard

Branch perf/observed-kv-reservation (81a87e474). It sets sglang's
`new_token_ratio_tracker.current` from the fraction of `max_new_tokens`
that recently finished requests used. The one requirement now is that it
cannot touch the normal path. Guard: apply `min(observed, sglang's own
current)`, never a larger value. Under the min, admission is never fewer
rows than today, and on a pool that does not bind it is identical by
construction, at c1 included, because the ratio only enters the
PrefillAdder's reservation (schedule_policy.py:654-661). Experiment: bf16
colocated Qwen3-Omni SeedTTS at c1, c16, c32 with the pool fix in both
arms, plus the MiniMax Music 3 cookbook pass, judged by qps, p95, WER and
identical c1.

## 5. What outweighs both mechanisms

### 5.1 Per request host work becomes per step work

Every step does O(batch) Python on the scheduler thread: one outbox
message per row per step from the talker to code2wav
(talker_model_runner.py:135-154), per request finish checks, KV release and
stream emission in the upstream result processor, per row loops in
`_finalize` (base.py:530-539), `_apply_codec_suppress_tokens` (:970-1001),
`_write_feedback_buffers` (talker :361-371), `_decode_collect_host` (Higgs
model_runner.py:378-408). The talker step grows 17 percent from bs 1 to
bs 6 for 0.6 ms more GPU, the rest is this work (doc 06 section 3.1).

The shape that removes it: per request device state in a slot keyed by a
stable id, the host touching a request at admission and finish only, one
message per step per edge carrying the batch. Higgs keeps its sampler
state that way (`_sampler_pool[rows]`, model_runner.py:339-346). The
talker rebuilds its decode buffers on every composition change and on
every extend (`invalidate_decode_buffers`, doc 06 section 3.3).

### 5.2 Host round trips inside a step, as a contract

Every sync found so far was found by reading a trace: the talker's Python
True write and mrope `.item()` (doc 06 section 3.3), the thinker's two
pageable `.tolist()` per step and the deepstack `nonzero` (doc 05 section
5), the `.any()` and mask writes in the talker request build (doc 06
section 3.5). PyTorch's `torch.cuda.set_sync_debug_mode("error")` raises on
synchronizing operations (`.item()`, `.tolist()`, `nonzero`, blocking
copies), documented as experimental and not catching every sync. One GPU
test per runner that runs one decode step and one prefill step under that
mode makes the invariant something every runner inherits, and each
failure is one refactor item with its stack.

### 5.3 The prefill as work, not as a step

The prefill's fixed cost is host dispatch, the inline build where it is
inline, and the drain. A prefill dispatched from a second thread on a
second stream would overlap its host time with decode host time instead of
replacing it, which is the one design that outweighs coalescing outright.
Two things decide it and both are measurable in one capture: whether the
GIL is saturated on the stage (the scheduler thread's share of GIL held
time), and whether launch calls from two threads serialize in the CUDA
runtime enough to stretch the decode launches. If they serialize, the
fallback is fewer launches per prefill and the gate stays at a derived
default.

### 5.4 The arch class, checked per stage

The talker pool bug was config plumbed wrong, found by reading one boot
line against the model's shape. Cheap checks of the same class, every
model:

- Graph hit rate. Each decode batch already logs `cuda graph: <bool>`
  (sglang metrics_reporter.py:780) and the runner output carries
  `can_run_cuda_graph` per step. A stage whose realized batch sizes miss
  its capture buckets runs eager at several times the host cost.
- Dead per step work: the `copy_for_forward` allocation above, two batch
  copies per launch, per step `_emit_prefill_end_for_batch`.
- Poll sleeps on a queue driven stage: `_sleep_during_idle` sleeps 1 ms
  when nothing is ready (omni_scheduler.py:938-943). The talker defers its
  decode until thinker text arrives (talker_scheduler.py:110-115), so at
  c1 every arrival can be up to 1 ms late. At c1 the talker waits on the
  thinker's 15.8 ms step for 1.7 ms of GPU (doc 05 section 5.5), so the
  thinker's synchronous step sets speech latency at c1.
- Sampling backend pinned to pytorch for the thinker and talker
  (stages.py:1161-1169). The pytorch path sorts the full vocabulary per
  step for top k and top p. The talker samples in graph, the thinker does
  not, and its per step sort is unmeasured (the E0 thinker trace had no
  GPU events).
- Pool and row limits per stage after the fix: the talker's fp8 pool of
  about 290k tokens for 32 rows can give memory back to the thinker (doc
  12). The same boot line read for Qwen3-TTS, MOSS-TTS, Dots is open.
- The talker at `chunked_prefill_size: 0` with no prefill graph
  (talker_scheduler.py:29-35) and one request build worker
  (omni_scheduler.py:196, bootstrap.py:244-258), while the ASR stages
  build on 2 to 8 workers.

## 6. The methodology

### 6.1 The step ledger, in tree

One small omni owned instrument that answers, per stage and per step, the
only question the step model asks: host or GPU, and by how much. Off by
default, enabled by a server flag or env, read through the existing
`_admin_model_info` (omni_scheduler.py:1912-1939) and logged at shutdown.

Per step it records: mode (extend or decode), rows, extend tokens, host
wall of the launch or sync call (perf_counter around `_run_batch` :1394 and
`_run_batch_launch` :1483), GPU time from a start event recorded before
the forward and the completion event already recorded at launch
(sglang_execution.py:124-128, elapsed read at resolve after the wait so it
never syncs), `can_run_cuda_graph` from the runner output, the iteration
wall from the previous iteration start, whether the iteration slept, the
CUDA allocation counter delta (`torch.cuda.memory_stats()` allocation
count, which exposes tensor churn per step), and in a diagnostic mode the
count of synchronizing operations reported by
`torch.cuda.set_sync_debug_mode("warn")` routed through a warnings filter.

Aggregated per (stage, mode, rows bucket): steps, host p50 and p90, GPU
p50, exposed host = host minus GPU where positive, graph hit share, sleeps
per step, allocations per step, syncs per step. The extend rows give the
fixed prefill cost as host minus GPU at small token counts, which is the
number that sets K and T. Two CUDA events and a few clock reads per step
is the overhead.

The pipeline view comes from running the ledger on every stage process at
once: the stage with the largest exposed host per row at the CI
concurrency is the bottleneck, and the request recorder (`/start_request_profile`,
docs/developer_reference/profiler.md) already stitches per request across
stages, so the two together give the overlap between stages without any
external tool.

### 6.2 Contracts as tests

- Sync free step: one decode step and one prefill step per runner under
  `torch.cuda.set_sync_debug_mode("error")`, GPU test, for the talker,
  thinker, Higgs, Qwen3-TTS, Qwen3-ASR, MOSS-TTS-Local, Zonos2.
- Graph hit: at CI concurrency every decode step of every stage replays a
  graph, asserted from the ledger.
- No churn: steady decode steps allocate zero new tensors, from the
  ledger's allocation delta.
- Pool sizing: the boot line's per token KV bytes equal the sub model's
  layers times heads times head dim times dtype, for every stage, the doc
  12 test generalized.

### 6.3 The experiment protocol

- Two fresh servers per arm, interleaved on the same GPU, the model's CI
  dataset, at c1 and at the CI concurrency, plus c32 or c64 where the model
  serves it.
- Fixed seeds where the model draws them from the request, and identical
  admission order where the pool does not bind.
- Judged by qps, p50, p95, and the model's quality metric (WER, CER, or
  the codes), with the ledger rows as the mechanism proof: a change must
  move its own row (exposed host, graph hit, syncs, allocations) or it did
  not engage, whatever qps says.
- nsys with in tree NVTX ranges around the same phases, under
  `/start_profile`, only for the deep dive once the ledger names a stage
  and a phase.

### 6.4 The fleet table to fill

One row per stage and concurrency, from the ledger: host per step, GPU per
step, exposed host, rows, graph hit, syncs, allocations, prefill fixed
cost, sleeps. Models: Qwen3-Omni thinker and talker, Qwen3-TTS, Higgs,
MOSS-TTS-Local, MOSS-TD, Qwen3-ASR, Whisper, Fun-ASR, ArkASR, Zonos2, Dots,
MiniMax Music 3, CosyVoice3. The table decides the per model order in
section 7, replacing every argument from reading.

## 7. Per model candidates, from the code read this session

Each item names the ledger row that proves it.

| model | candidate | proof row |
|---|---|---|
| Qwen3-Omni talker | E2 syncs, per row outbox messages, buffer rebuilds, inline build, eager prefill, then lookahead (doc 05 section 7, doc 13 section 4) | syncs per step, exposed host, prefill fixed cost |
| Qwen3-Omni thinker | PR C token staging, PR D audio output lookahead, deepstack zeros and nonzero (doc 05 sections 5, 6) | syncs, exposed host at c1 and c16 speech |
| Higgs | per row collect loop, threshold at 1 | exposed host by rows, c1 qps |
| Qwen3-TTS | sync loop, hooks, sub talker fallback path | exposed host, graph hit |
| MOSS-TTS-Local | bytes per token (KV dtype), threshold | GPU per step dominates, exposed host small |
| ASR family | derived gate, threshold at 1, encoder and build placement | prefill fixed cost, sleeps at c1 and c8 |
| Zonos2 | sampler fusion on sgl_kernel renorm (doc 11) | GPU sampler share from the trace |
| Fish, MiniMax | one topk each (doc 11) | small, measure first |
| vocoders, code2wav | D2H on the current stream, codec graphs | exposed host of the vocoder stage |

## 8. Order

1. The ledger (6.1) and the two cheapest contracts (sync free step, graph
   hit). Each is small, omni owned, and each can expose a talker class bug
   on any model.
2. The fleet table at CI concurrency and c1 for every model.
3. The refactor list of 3.4 items 1 to 3 on the base runner, measured on
   the ledger rows they touch.
4. The observed KV experiment with the guard (section 4).
5. Per model work in the order the table gives, the talker's E2 first if
   the table agrees with doc 06.
6. The derived gate (2.5) and the second stream prefill (5.3), each as an
   A/B under 6.3.

## 9. Tasks

- T1 Ledger: built on branch perf/step-ledger (worktree
  .worktrees/step-ledger), local profiling branch, no PR. Files:
  sglang_omni/profiler/step_ledger.py, hooks in
  scheduling/omni_scheduler.py, model_runner/base.py,
  pipeline/stage/runtime.py, fields documented in
  docs/developer_reference/profiler.md. Switch: the request profile run.
  Output: step_ledger_<stage>_<pid>.json beside the events, one log line
  per batch shape, and the live aggregate under step_ledger in /model_info.
- T2 Sync free step test per runner.
- T3 Graph hit and no churn assertions from the ledger.
- T4 Pool sizing test generalized to every stage's boot line.
- T5 Fleet table, one run per model.
- T6 `isolate_sampling` only under lookahead, measured on the talker.
- T7 Token history on the device, host list at finish and per chunk.
- T8 One pinned ring in the bridge replacing the four pairs.
- T9 Threshold 1 against 2 per model, the #1454 protocol.
- T10 Delayed sample under lookahead, parity test on penalties.
- T11 Observed KV with the min guard, bf16 Omni and MiniMax passes.
- T12 Derived gate A/B on every model with the gate on.
- T13 Second stream prefill: GIL share and two thread launch serialization
  in one capture, then the A/B.
- T14 In tree NVTX ranges under `/start_profile` for the deep dive.
