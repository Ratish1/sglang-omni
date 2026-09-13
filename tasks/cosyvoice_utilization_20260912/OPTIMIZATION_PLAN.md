# CosyVoice3 H100 utilization plan

Baseline for this plan: Omni `645b472cd` plus upstream `main` at `f58228dfb` (2026-09-13), which
adds the buffered whole-solver Flow CUDA graphs (#1861) and the Apple port (#1964). SGLang pinned
at `v0.5.19`. Evidence: the profiler-off English full-corpus matrix and the sixteen NSYS captures
returned on 2026-09-13, sliced with the perfkit ([results](perfkit/RESULTS_20260913.md)). Source
line references are to this branch unless a file is named with `upstream/main:`.

Goal as stated by the team: raise SM Active at concurrency 16 from the reported 2 to 3 percent to
30 percent while raising goodput (successful audio seconds per wall second). The measured c16
baseline, one definition each (results section 0):

| c16 cell, eager | GR Active % | SMs Active % | Tensor Active % | nvidia-smi utilization % |
|---|---:|---:|---:|---:|
| streaming | 40 to 43 | 19.5 | 2.5 to 2.6 | 67 to 70 mean, 98 peak |
| buffered | 53 to 54 | 28 to 31 | 3.9 to 4.4 | 66 to 75 mean, 98 peak |

The Nsight counters are 100 µs samples averaged over the HTTP cohort window; nvidia-smi is the
5 s sample mean over the profiler-off full-corpus run. No counter reads 2 to 3 percent except
Tensor Active, so the team's figure is either that counter or a window this branch has not seen;
task M2 reproduces it with the team's method before it is used. The plan's targets are the
SMs Active values above. Everything else is anchored on the perfkit ledgers, which do not depend
on that reproduction.

## 1. What the traces establish

1. At c16 streaming the AR thread finishes all 16 requests' tokens in under 2 s and the vocoder
   thread then works alone for 8.5 s, busy for 95 to 99 percent of the window, launching GPU work
   for 26 to 29 percent of it. The vocoder thread's Python launch path is the bottleneck, not the
   GPU and not the AR.
2. One eager Flow solve is 18.1k kernel launches (10 Euler steps of a 22-block DiT). At batch 1 the
   GPU is busy for a quarter of the call and SM Active is 15 to 25 percent while its kernels run.
   At batch 10 to 14 the same call is GPU bound and SM Active is 58 to 86 percent. Spatial
   efficiency comes from batch, temporal efficiency from removing launches.
3. Every Flow call blocks on `cudaStreamSynchronize` 4 times per batch row plus 15 per call, and
   every HiFT call about 85 times. The sources are host tensors moved to the device inside the
   call. A HiFT call at batch 1 is 27 to 72 ms of host time for 4.5 to 9 ms of GPU time.
4. The vocoder thread's host cost per launch doubles (15 to 40 µs) while the AR thread is running:
   both threads are Python loops sharing one interpreter lock.
5. Streaming runs HiFT once per request per hop on the whole accumulated mel, inside the batched
   step, so a batched first hop of 14 requests spends 491 ms in one packed Flow call and then
   380 ms in 14 serial HiFT calls before any participant's PCM is yielded.
6. The AR decode step is 1.0 to 1.3 ms of GPU time (flat in batch, 291 to 316 graph nodes) inside
   2.4 to 4.0 ms of wall time: 35 eager sampling kernels (0.6 to 1.1 ms host), the `tolist` wait,
   and 0.5 ms of scheduler time. Custom eager prefill is 344 launches at 60 to 110 µs each, 21 to
   38 ms of host time for 1.5 ms of GPU time. SM Active while AR decode kernels run is 12 to 17
   percent: the 0.5B decode is latency bound at every batch size up to 32.
7. Full-corpus c16 fails acceptance before any optimization: 22 to 23 of 1088 streaming requests
   time out at 300 s after first audio; 128 to 135 buffered eager requests fail in the ONNX cuBLAS
   handle creation and cuDNN attention execution (memory exhaustion); matched-sample streaming WER
   is 4.8 to 6.3 percent at c16 against 1.1 to 1.4 percent at c1.

## 2. Order of work and why

The order is root cause first. A change lower in the ladder cannot be measured at c16 until the
rungs above it hold.

```
C  correctness at c16: token order under coalescing, final-decode liveness, memory budget
V  vocoder host path: remove per-call syncs, one Flow entry, graphs for streaming and batch > 2,
   batched and graphed HiFT
A  AR host path: one host snapshot, fewer eager sampling launches, prefill batching
P  preprocessing: cold reference cost bounds first audio at c16
R  placement: only if the perfkit still shows interpreter contention after V
```

Every PR below names its owner, the measured trigger, the mechanism, the identity or quality gate,
and the rollback. Each ships on its own worktree branched from upstream main and stacked in this
order; the analysis branch keeps the annotations and the perfkit and is never the base of a PR.

## 3. Correctness gates

### C1. Streaming vocoder scheduler: in-order ingestion and final-decode liveness

Owner: `sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py` and
`sglang_omni/scheduling/streaming_simple_scheduler.py`.

Mechanism today (`streaming_vocoder.py:272-304, 337-360, 455-509`; `streaming_simple_scheduler.py:124-181`):

```
serving loop (streaming_simple_scheduler.py:124-145)
  _next_message: _pending_messages first, else inbox.get(0.1 s)
  stream_chunk -> _collect_stream_chunk_batch (streaming_vocoder.py:272-304)
        first chunk per request joins the batch; a second chunk of the same
        request is appendleft'd to _pending_messages ("deferred")
     -> on_stream_chunk_batch: ingest batch, then _pump_streams
  _pump_streams (486-509): loop
        select_step_participants (589-628): ready first hops win, else the
           largest (hop, offset) follow-up group
        run_step -> packed Flow (2B CFG rows, 10 Euler steps) or native Flow
           -> per participant HiFT on the whole mel history -> D2H -> outbox
        _ingest_ready_inbox (455-468): reads the inbox directly, ingests
           stream chunks of any request, stops at the first control message
     until no participant is ready
  stream_done -> on_stream_done (scheduling/streaming_vocoder.py:225-249):
        final decode_delta, only reached after the pump returns
```

Two defects follow from the mechanism and match the measured failures:

- A chunk parked in `_pending_messages` is ingested after a later chunk of the same request that
  `_ingest_ready_inbox` or a peer wait reads directly from the inbox. The token list of that
  request is then out of order and Flow receives a corrupted prefix. This path is exercised only
  when the AR is ahead of the vocoder, which is the c16 regime and not the c1 regime; it is the
  candidate cause of the c16-only WER regression. PR #2086 describes the same mechanism (missing
  words when the producer is ahead) and fixes the shared collectors and the peer waits to drain
  `_pending_messages` before the inbox.
- Finals run only when the pump drains. Under sustained load the ready set stays non-empty, so a
  request whose AR has finished waits for every other request's hops. The 64-request capture shows
  a final queued for 24.6 s; the full run shows 300 s timeouts after first audio. PR #2110 also
  rewrites `_ingest_ready_inbox` to park control messages and keep draining instead of stopping,
  and reports a 299.7 s long-tail stall on main disappearing on its branch.

Design: make ordering and liveness properties of the scheduler, not side effects of the collectors.

1. Ingestion is a single FIFO per request: every stream message for a request is appended to that
   request's queue in inbox order, from one place. Peer waits and between-step ingestion consume
   from the same drain function. No message is ever left in the inbox behind a parked sibling.
2. `stream_done` becomes a participant state, not a serving-loop event: a request whose AR is
   complete and whose remaining tokens do not fill the next hop is a `final` candidate in
   `select_step_participants`, ordered by ready time with the other candidates.
3. Selection is by oldest ready time with batch formation on top: gather the oldest ready
   candidate, then add every candidate with the same key up to the cap. First-hop priority is kept
   only as a tie break at equal age. This is the fairness rule the current largest-group rule
   lacks; the perfkit hop ledger (queue delay per hop kind) is its acceptance metric.

Reconcile with #2086 and #2110 by taking their tests, not their collector patches: the design above
supersedes both mechanisms. #2110's timestep layout change is separate and is handled in V3.

Gate: a scheduler unit test that replays a recorded inbox sequence (from the perfkit hop ledger of
the c16 capture) through the scheduler with a fake vocoder and asserts per-request token order and
that every final runs within one pump of its `stream_done`. Then the full English corpus at c16
streaming: zero timeouts, and matched-sample WER equal to the c1 WER within run-to-run noise
(c1 runs differ by 0.1 to 0.2 points between repeats). Rollback: revert the scheduler PR; no state
format changes.

### C2. GPU memory budget

Owner: `sglang_omni/models/fun_cosyvoice3/engine_builder.py:60-77`
(`mem_fraction_static: 0.85`) and the vocoder factory.

Trigger: 120 cuBLAS handle allocation failures inside the ONNX speech tokenizer and 15
`mha_graph.execute` failures (cuDNN attention workspace) in one eager buffered c16 run; telemetry
sampled the process at 81 GB. The AR reserves 85 percent of the device before the Flow, HiFT, ONNX
sessions and the merged Flow graphs allocate. For a Qwen2 backbone with 24 layers, 2 KV heads and
head size 64 in bf16, 32 requests at the 4096 context limit need about 1.6 GB of KV cache; the
actual backbone dimensions come from the installed `CosyVoice-BlankEN` config and are recorded by
the census before the budget is set.

Design: declare the AR KV budget in bytes through the existing stage KV contract
(`stage_kv_budget`, report 05) from the measured need (context length times max running requests
times per-token bytes plus the decode graph pool) and leave the remainder to the vocoder. This is a
config PR with a memory census: `torch.cuda.memory_stats` peak per stage at c16 buffered and
streaming, recorded by the perfkit protocol. Gate: zero allocation failures on the full corpus at
c16 buffered and streaming, eager and compile; peak reserved bytes below device capacity with
headroom for the graph pool of V3. Rollback: restore the fraction.

### C3. Quality items outside this plan's scope, kept as gates

Runaway generations to the 2048-token cap (81.92 s of audio for short texts) occur at c1 and c16 in
both variants and dominate the matched WER tails. They are a sampling contract issue, not a
utilization item; the A/B protocol compares matched samples so they cancel between arms.

## 4. Vocoder host path

### V1. No synchronization inside a Flow or HiFT call

Owners: `stages.py:135-200` (`_pack_flow_inputs`), `stages.py:350-419` (`_generate_flow`),
`stages.py:1111-1171` (`token2wav_chunk`), `stages.py:1188-1210` (`_hift_delta`), and the pinned
CosyVoice `generator.py` call sites listed in the results (section 5).

Trigger: 23 to 71 syncs per packed Flow call, 20 per native call, about 85 per HiFT call; 114 to
170 ms of sync host time inside a batch 10 to 14 Flow call.

```
_pack_flow_inputs (stages.py:135-200)        per row: prompt_token.to(device)   H2D + sync
                                                       item.token.to(device)     H2D + sync
                                                       prompt_feat.to(device)    H2D + sync
                                             cat of embedding.to(device)         H2D + sync
_generate_flow (350-419)                     rand_noise[:, :, :M].to(device)     H2D + sync
                                             t_span linspace                    device
token2wav_chunk (1111-1171)                  4 x torch.tensor(len).to(device)    H2D + sync
_hift_delta / HiFT.inference                 hann window .to(device) x2, sine and
                                             noise prefix .to(device), f0 float64
```

Design: every constant becomes a device-resident buffer created once at load (noise prefix at the
maximum supported length, cosine time span, Hann window, sine and noise source prefixes); every
per-request tensor crosses to the device once, at the stage boundary, with a pinned staging buffer
and an event, and the packer indexes device tensors. Lengths are built on the device from Python
integers without host tensors. The f0 predictor is converted to float64 once at load. Arithmetic
and kernels do not change, so the gate is byte identity of mel and waveform for frozen inputs at
c1 and for a frozen packed batch. Rollback: revert; no persistent state.

### V2. One Flow entry point

Owner: `stages.py:1111-1171` (native singleton path) and `streaming_vocoder.py:657-671, 805-843`.

Trigger: 19 native singleton calls against 4 packed calls in the 16-request c16 capture; the
native path cannot be graphed and carries its own sync points; two code paths mean two identity
proofs for every later change.

Design: singleton hops call the packed causal adapter with one row. The adapter's packed
singleton-versus-native identity is already unit tested (report 02); the gate is the same test on
real hops plus the seeded c1 byte-identity pass. Rollback: keep the native branch behind a flag
for one release.

### V3. Flow graphs for streaming hops and for batches above 2

Owner: `upstream/main:sglang_omni/models/fun_cosyvoice3/stages.py:318-520, 630-662, 1857-1911`
(`FlowCudaGraphRunner`) and `config.py:19-46` (capture shape table).

What merged: the runner captures the whole Euler solve per `(batch, frames)` entry of a fixed
table, right-pads inputs to a 16-frame bucket, copies into static inputs, replays, clones the
cropped output, and falls back to eager when the shape has no graph. The table covers batch 1 at
304 to 640 frames and batch 2 at 384 to 544 frames. `generate_flow` bypasses the runner whenever
`streaming` is true or `finalize` is false. Consequently no streaming hop and no buffered batch
above 2 is graphed on main today. #2141 proposes another fixed table up to batch 16 at 416 to 608
frames. Both tables are constants that no measurement pins.

Trigger: at batch 1 the eager solve is launch bound (GPU busy a quarter of the call); packed
batches of 4 to 5 are half busy; even batches 10 to 14 lose 80 to 170 ms per call to launches and
syncs. Streaming is 100 percent eager on main.

Design:

1. Bucket policy from the workload, not a table: batch buckets `{1, 2, 4, 8, 16}` and frame
   buckets at a coarse step chosen from the measured prompt plus generated length distribution of
   the English corpus (the perfkit call ledger gives the histogram), bounded by a graph pool budget
   measured at capture time; oversized shapes fall back to eager and the fallback counter is a
   perfkit metric. Padding waste per bucket is the other counter.
2. Causal mode: capture `streaming=True, finalize=False` per bucket. The chunk mask is a function
   of the padded length only (`subsequent_chunk_mask(L, 50)` ANDed with per-row validity, report
   12), so it is captured once per bucket with validity as a static input; the empty-row repair in
   the mask helper is a host read and gets the same graph-safe replacement the merged runner
   already applies to the non-streaming branch (`upstream/main:stages.py:1871-1896`).
3. The lookahead body split of `_apply_pre_lookahead` for unequal rows stays outside the graph; the
   graph starts at the Euler loop, exactly as merged.
4. Time span and noise are static inputs after V1; the timestep layout question raised by #2110
   (scalar row against `2B` rows) is settled inside this PR by a frozen-input mel comparison of
   the two layouts before capture; the layout that matches native CosyVoice is captured.

Gate: mel byte identity between graph replay and eager at identical padded shapes for every
bucket in both modes (replay executes the same kernels on the same memory), then full-corpus A/B
at c1 and c16 in both modes. Memory: graph pool bytes recorded per bucket set, checked against C2.
Rollback: the existing `enable_flow_cuda_graph` flag.

### V4. HiFT: batched per step, then graphed

Owner: `streaming_vocoder.py:673-734` (per-participant `_hift_delta` inside the batched step),
`stages.py:1188-1210, 1271-1306`.

Trigger: HiFT is 16 to 30 percent of the vocoder thread at c16 streaming; 14 serial calls of 27 ms
follow each batched first hop; SM Active 13 to 15 percent at batch 1 against 51 percent at batch
10 to 12.

Design, in two PRs after V1:

1. Batch the participants of one streaming step with equal accumulated mel length first
   (identical histories after the same hop sequence), concatenate histories, one HiFT call, crop
   each row at its own `speech_offset`. Unequal lengths use the buffered right-zero-pad path, whose
   documented deviation is confined to the final mel frame of padded rows (`stages.py:640-646`);
   the streaming step never emits that frame before finalization because of the lookahead, which
   the gate verifies on frozen mels.
2. Capture HiFT per `(batch, frames)` bucket with the same runner design as V3 once its inputs are
   device resident and sync free.

Gate: waveform byte identity for equal-length batching against per-request calls on frozen mels;
for padded batching, identity of every emitted sample plus the documented final-frame bound; then
full-corpus A/B with continuity metrics. Rollback: per-request path behind a flag.

### V5. Placement, only on evidence

PR #1933 moves the vocoder to its own process with fixed 0.80 and 0.12 memory fractions and reports
24 to 26 percent QPS gains on H200. The perfkit measures the interpreter contention it targets
directly: host µs per launch on the vocoder thread while the AR is active against inactive. After
V1 to V4 the vocoder launches two orders of magnitude fewer kernels, so that metric is re-measured
first. If contention remains material, #1933's topology is adopted with C2's budget instead of
fixed fractions. Not before.

## 5. AR host path

### A1. One host snapshot, no per-token host allocations

Owner: `model_runner.py:125-146` (`_collect_tokens`), `model_runner/base.py:575-620` (`_finalize`).

Trigger: two `tolist` materializations per step (codec collection and output processing), one CPU
tensor allocation per emitted token per step, 0.44 to 1.25 ms of `tolist` wait per step.

Design: as plan 01 A1, with the staging helper and one host snapshot reused by `_finalize`;
codec tokens are appended as Python integers and materialized as one tensor per emitted chunk.
Gate: identical codec sequences under the seeded c1 pass. Small PR, no scheduler change.

### A2. Sampling launches

Owner: upstream `Sampler.forward` PyTorch backend (report 11), `engine_builder.py:60-77`
(`sampling_backend: "pytorch"`).

Trigger: 31 to 35 eager kernels per step for 0.09 to 0.14 ms of GPU time and 0.6 to 1.1 ms of
host time, 25 to 30 percent of the decode step wall time.

Design: measure the FlashInfer sampling backend on the unseeded default path (the seeded filtered
path is not supported there, report 11) with identical token sequences under greedy and identical
distributions under the corpus WER; keep PyTorch sampling for seeded requests. This is the SGLang
mechanism for the same problem; no Omni kernel. Gate: seeded c1 identity for the retained path,
full-corpus WER parity for the FlashInfer path. Also record the owner of the 20 to 109 ms one-off
sampling stall on the first multi-row prefill of each cohort with a Python-sampled capture (M1).

### A3. Prefill batching

Owner: `omni_scheduler.py:1380-1414` (prefill coalescing hold, off by default:
`prefill_coalesce_requests=0`, `omni_scheduler.py:202-206`), `model_runner.py:237-263`.

Trigger: 21 to 38 ms of host time per eager prefill at 1.5 ms of GPU time, cost per launch not per
request; at c16 requests leave preprocessing one at a time and 55 of 58 prefills in the
64-request run were batch 1.

Design: measure the existing coalescing hold (requests target and wait) against the first-audio
budget at c16 before any prefill graph work; the perfkit AR ledger reports prefill batch sizes and
the request ledger reports first-audio. A prefill CUDA graph for the custom embedding prefill is
the next step only if coalescing cannot fill batches without moving first audio.

### A4. Overlap scheduling

Not implementable as a flag (#1669 is that flag). Omni's overlap loop raises and the generic async
decode requires history-free penalties, which the 1.21 repetition penalty and the minimum-length
suppression are not (reports 04, 05, 10). This remains the SGLang-native answer to AR host time
and is scheduled after V and A1 to A3, with one research task first: how upstream v0.5.19's overlap
loop feeds `BatchedRepetitionPenalizer` with the FutureMap lag and whether that contract can be
adopted by Omni's loop. That is an Opus mechanics-only task if the reports' coverage of
`overlap_utils.py` and `batch_result_processor.py` proves insufficient.

## 6. Preprocessing

Owner: `request_builders.py:277-330, 704-739`, ONNX sessions in `utils.py:42-137`.

Trigger: at c16 the HTTP-to-queue-enter time is 0.5 to 1.1 s p50, larger than the AR time; a cold
reference costs 437 to 770 ms and followers wait on the leader; the full corpus is all cold. The
finalize lock waits 6 ms mean and is not the limiter.

Design: batch the S3 tokenizer across concurrent misses. PR #1693 does this by replacing the ONNX
session with the `s3tokenizer` PyTorch model and reports exact token equality on eight clips; it
conflicts with main and changes a dependency, so it is reconciled, not merged as is. Gate: exact
reference token, feature and speaker embedding equality on the full English reference set between
backends; then c16 first-audio and goodput A/B. The ONNX CUDA session also allocates on the shared
GPU and is one of the two failure sites in C2.

## 7. Open PRs reconciled (audit of 2026-09-13, 415 open PRs screened)

| PR | state | relation to this plan |
|---|---|---|
| #1861 whole-solver Flow graphs | merged | base of V3; buffered finalize only, fixed shape table |
| #2141 new capture shape table | open, approved | superseded by V3's bucket policy |
| #2086 deferred chunk order | open | same defect as C1; take its tests, supersede its collector patch |
| #2110 timestep rows plus ingest rewrite | open, approved, conflicting | ingest part superseded by C1; timestep layout settled by the V3 frozen-input comparison |
| #1933 isolated vocoder process | open, approved | V5, conditional on the post-V4 contention measurement |
| #1693 batched S3 tokenizer | open, conflicting | P1 base; backend parity gate required |
| #1669 overlap flag | open | insufficient; A4 |
| #2128 seed tensor cache | open, draft | shared base; not on the unseeded benchmark path |
| #2137 skip singleton peer wait | open, draft | c1 buffered latency only (32 ms per request); independent |
| #2136 thread-local capture mode for the metadata glue graph | open, draft | unrelated to the observed cuDNN failures, which are memory exhaustion (C2) |
| #1673 configurable Flow steps | open | quality change, out of scope |
| #1817 repetition penalty | open | behavior already in main |
| #1995, #1216, #1304 profiler rank fix | open | observability only |

## 8. Existing optimizations assessed

| path | measured behavior | assessment |
|---|---|---|
| 30 ms peer waits, first hop and follow-up (`streaming_vocoder.py:79, 362-453`) | 0.8 to 3.7 ms total in 16-request cohorts, 152 ms in the 64-request run | never the limiter at c16; a constant no measurement pins; C1 replaces the policy |
| hop growth 25 to 50 to 100 | recompute of the whole prefix per hop grows Flow cost per hop with prefix length | kept; incremental Flow state is a numerics project (plan 04 H2) after V3 |
| adaptive buffered Flow grouping (#1899) | groups of 5 to 13 at c16 buffered, GPU bound at those sizes | correct lever; V3 removes its remaining host cost |
| estimator torch.compile | halves launches to 7.8k, call still launch bound at batch 1; 44 s recompile stall in one capture; c16 streaming WER worse than eager | not production safe with dynamic shapes; V3 graphs replace it |
| buffered Flow graphs (#1861) | streaming and batch above 2 fall back to eager | V3 extends it |
| SGLang decode graphs | 1.0 to 1.3 ms per step, 12 to 17 percent SM Active | correct; the AR is latency bound by model size, not by the runner |
| PyTorch sampling | 35 eager launches per step | A2 |
| custom eager prefill | 344 launches, 21 to 38 ms host | A3 |

## 9. Measurement protocol

M1. Perfkit and captures (this branch). `perfkit/slice_trace.py` produces the ledgers used above;
`perfkit/README.md` documents them. Capture protocol v2 adds: `--trace=cuda,nvtx,osrt,python-gil`
on one c16 capture to quantify interpreter waits per thread; `--python-sampling=true` on one c16
capture to attribute the HiFT sync sites and the one-off sampling stall; an `ingest` mark with
chunk id and token count in the vocoder; a long c16 cohort (at least 128 requests) with an
interior window for steady state; and the event-recorder JSONL mode of the perfkit for
profiler-off full runs, which is how the 300 s timeouts are localized to a hop.

M2. Reproduce the team's SM metric: GPU metrics only, no CUDA trace, over an interior window of a
full-corpus c16 streaming run, with the same tool and aggregation the team used. Until this exists
the 2 to 3 percent figure is unverified and the perfkit conditional means are the targets.

M3. A/B protocol for every PR: full English corpus at c1 and c16, streaming and buffered, one boot
per arm on the dedicated GPUs, census recorded, paired deltas of goodput, p95 completion, streaming
first audio and continuity, matched-sample WER, SIM and UTMOS; a seeded c1 warmup-1 pass for byte
identity where the PR claims identity; stacked slices reuse the previous B as A. Profiling is never
on during an acceptance run.

## 10. What the ladder can deliver at c16

Measured component efficiencies bound the outcome. AR decode kernels run at 12 to 17 percent SM
Active and cannot be made spatially efficient at this model size; packed Flow at batch 8 to 16 runs
at 58 to 86 percent; HiFT at batch 10 to 12 at 51 percent. The window mean is the coverage-weighted
sum of these. With the vocoder path graphed and batched (V1 to V4) the vocoder's host time per hop
falls to about its GPU time, coverage rises from 0.30 toward the GPU-bound limit, and the streaming
window shortens by the vocoder's removed host time, which is 7 s of the 10.5 s streaming cohort.
Whether that reaches 30 percent SM Active depends on the realized batch per hop after C1's fairness
rule, which the perfkit hop ledger reports after each rung. No rung is promised a number; each is
measured before the next is designed.
