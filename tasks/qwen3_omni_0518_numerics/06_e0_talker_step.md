# 06. E0, the talker step measured

Run: e0-prep4-20260831 (bundle in artifacts/), checkout 2f5d2ab5 in the
container, sglang 0.5.18, torch 2.13.0+cu130, H100 80GB, bf16 colocated
config (examples/configs/qwen3_omni_colocated_h100_bf16.yaml, one GPU),
voice clone workload (seed-tts-eval-50-arrow en). Three server boots:
events at c16 (50 requests), events at c16 with the decode log at interval 1
(50 requests), torch profiler traces with `SGLANG_TORCH_PROFILER_WITH_STACK=1`
for 8 requests at c1 then 16 requests at c16 in one profiling window.

The B run (preprocessing replicas) did not start: the colocated speech config
class rejects any replicated process
(sglang_omni/models/qwen3_omni/placement.py:70-82, added by #1175). Section 6.

## 1. Headline numbers

| run | requests | success | qps | latency mean s | RTF mean | RTF p50 |
|---|---|---|---|---|---|---|
| events c16 | 50 | 50 | 5.73 | 2.447 | 0.761 | 0.696 |
| events c16, decode log 1 | 50 | 50 | 6.24 | 2.412 | 0.734 | 0.716 |
| traces c1 | 8 | 8 | 1.39 | 0.719 | 0.203 | 0.193 |
| traces c16 | 16 | 16 | 4.63 | 2.448 | 0.732 | 0.682 |

Audio per request: median 3.3 s, mean 3.5 s, max 7.3 s at 24 kHz. Thinker
text per request 10 to 22 tokens. The talker emits one chunk per frame
(median 42 chunks per request, 2279 chunks for 50 requests), about 12
frames per audio second (386 decode steps for 9 requests of 3.67 s mean
audio in the c1 trace).

The two c16 event runs differ by 9 percent in qps with the same code and the
same 50 requests, so single c16 runs of 50 requests carry at least that much
noise. Nothing below rests on a difference smaller than that.

## 2. Where a c16 request spends its 2.4 s

From the events run (51 requests including warmup), p50 unless stated:

| interval | p50 ms | mean ms |
|---|---|---|
| admission to thinker first emit | about 100 | |
| thinker first emit to thinker complete | 217 | 229 |
| thinker complete to talker request build start | 12 | 71 |
| talker request build | 14 (mean) | 14 |
| talker queue enter to prefill start (admission wait) | 1020 | 911 |
| talker prefill | 45 (mean) | 45 |
| talker prefill end to talker complete (generation) | 860 | 900 |
| talker first chunk to code2wav first audio | 239 | 238 |
| admission to code2wav first audio | 1705 | 1639 |
| admission to code2wav complete | 2346 | 2366 |

The talker starts its request only after the thinker has finished the whole
text (thinker complete to talker build start 12 ms p50, `enable_partial_start`
is false in this config), so the thinker's 217 ms is serial with the talker.

The largest term is the talker admission wait. The talker never ran more than
9 requests at once in the 50 request run and spent 64 percent of its busy
time at 6 to 8 running (from the prefill_start and stage_complete events):

| running in talker | 0 | 1 | 3 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| share of time | 0.04 | 0.19 | 0.06 | 0.06 | 0.24 | 0.24 | 0.16 | 0.01 |

The talker KV pool holds 21373 tokens (server log, talker `KV Cache is
allocated ... #tokens: 21373`) and every talker request carries
`max_new_tokens` 4096: `_resolve_talker_sampling_config` reads
`params.get("talker_max_new_tokens", 4096)` (request_builders.py:1091), the
only ways to change it are the per request HTTP field
(serve/protocol.py:95, openai_api.py:1024-1033), there is no server arg or
stage config knob. The talker stops on the codec EOS code
(`stop_token_ids` and `eos_token_ids` from `codec_eos_token_id`,
request_builders.py:785,799) or on `max_new_tokens`, text exhaustion is not
a stop: after the thinker is done the talker feeds `tts_pad_embed` rows
(talker_model_runner.py:470-476) until EOS. Nothing sizes a buffer from
the value, the static talker buffers are sized by `max_running_requests`
and the vocab (talker.py:879-998), and the only other reader is the
admission check `len(origin_input_ids) + max_new_tokens <= max_req_len`
(omni_scheduler.py:1269-1291, max_req_len 32767 here). SGLang admits a
new request only while the pool can hold every running request's
remaining reservation, `min(max_new_tokens, 4096)` times `new_token_ratio`
(0.7 at the start of a decode run, decaying over 600 steps), so 21373
tokens admit 6 to 9 requests of about 150 prompt tokens with the rest of
the pool reserved for outputs that never come: the median talker output
is 42 frames. Section 4 has the arithmetic.

## 3. The talker step, from the with_stack trace

Trace: talker_ar process, 20.1 s span, 592 batches on the scheduler thread
(22 prefills, 570 decode steps), 1.13 M kernels, 3.25 s of kernel time.
Phase 1 is the c1 run (404 batches, 386 decode steps, batch size 1), phase 2
is the c16 run (188 batches, 173 decode steps, batch size 1 to 6). The
per step tables below use the `run_batch` call on the scheduler thread as the
step and the interval from one `run_batch` start to the next as the cycle.

### 3.1 Steady steps

A steady step is a decode step with no buffer rebuild and no stream sync
stall (342 of 386 at c1, 98 of 173 at c16). Medians in ms:

| item | c1, bs 1 | c16, bs 6 |
|---|---|---|
| cycle | 8.40 | 9.38 |
| run_batch | 7.98 | 8.83 |
| outside run_batch (get_next_batch_to_run, process_input_requests, process_batch_result) | 0.42 | 0.53 |
| GPU kernel time in the cycle | 5.13 | 5.76 |
| event wait in `_resolve_host_token_ids` | 4.70 | 5.01 |
| cudaGraphLaunch host time (`full_cuda_graph_backend.replay`) | 1.66 | 1.70 |
| `decode_cuda_graph_runner.load_batch` | 0.37 | 0.42 |
| `_build_forward_batch` (`ForwardBatch.init_new`) | 0.25 | 0.27 |
| `before_decode` (`prepare_decode_buffers` 0.05, `_write_feedback_buffers` 0.10 to 0.22) | 0.16 | 0.30 |
| `post_decode` | 0.17 | 0.22 |
| `_publish_next_tokens` | 0.08 | 0.08 |
| rest of run_batch | 0.20 | 0.25 |

Reading: the GPU runs the graph for 5.1 to 5.8 ms and the host waits for it
in the event wait, so the GPU time is fully exposed. The host time that is
not overlapped with the GPU is the cycle minus the kernel time, 3.3 ms at
bs 1 and 3.6 ms at bs 6. cudaGraphLaunch is the largest host item at 1.7 ms,
but the GPU starts executing the graph about 40 us after the launch call
begins, so most of the launch call already overlaps GPU work. The graph has
about 1890 kernel nodes per step (1.13 M kernels over 592 launches), at
3 us average per kernel, which is what makes both the launch call and the
GPU time long for a model this size.

The step is 17 percent longer at bs 6 than at bs 1. The GPU time grows
0.6 ms (memory bound decode), the rest is host work that scales with the
batch (`_write_feedback_buffers`, `post_decode`, `process_batch_result`).

Runtime calls per steady step: 1 cudaGraphLaunch, 1 cudaEventSynchronize,
2 cudaStreamSynchronize (both return immediately when the GPU is idle,
section 3.3), 15 cudaMemcpyAsync (one DtoH of the sampled ids to pinned
memory, one pinned HtoD, two pageable HtoD of 8 bytes and 1 byte, the rest
DtoD), 14 to 23 direct kernel launches outside the graph.

### 3.2 Steps that are not steady

Two things make a step longer than the steady numbers.

Buffer rebuild after a prefill or a batch size change. `prepare_decode_buffers`
takes 3 to 5 ms instead of 0.05 ms and `ForwardBatch.init_new` 1 to 2.3 ms
instead of 0.27 ms. At c1 this is the first decode step after each prefill
(9 of 386 steps). At c16 it is the first decode step after each prefill and
every step where the batch size changed (25 of 173 steps), cycle 15 to
18 ms. Prefill batches themselves are separate (section 3.4).

Stream sync stalls from code2wav on the same GPU. Every 10 frames of a
request, code2wav runs its codec decode on the shared GPU. During those
kernels the talker's two per step cudaStreamSynchronize calls
(`_compute_mrope_positions_decode` in sglang's `ForwardBatch.init_new`, and
`_reuse_decode_buffers` in components/talker.py:1029) block for 1 to 2.4 ms
each, because the copies they wait for queue behind the other context's
time slice. All 34 stalled steps at c1 and all 61 at c16 overlap code2wav
kernels in the step window, none of the 352 and 109 quiet steps do (the
thinker had no kernels in either window). A stalled step costs 12.2 ms at
c1 (8.6 steady) and 14.8 ms at c16 (10.7 steady), 34 of 386 steps at c1
and 61 of 173 at c16. The GPU contention itself is inherent to colocation.
What is not inherent is that the host stops at a synchronization point
instead of continuing to prepare the step: without the two syncs the delay
would land in the event wait and overlap the remaining host work.

Both effects together explain the difference between the steady cycle and
the average: c1 average cycle 8.93 ms, c16 average 12.26 ms.

### 3.3 The synchronization points, with their sources

Per steady step the two stream syncs cost 0.01 to 0.02 ms because nothing
is queued ahead of them. Their cost appears under contention (3.2), and
they are the reason an overlap loop cannot pay as built (section 5).

- `_reuse_decode_buffers`, components/talker.py:1046:
  `self._repetition_mask[rep_rows, self._sampled_token_ids[rep_rows]] = True`.
  The value is the Python literal True. Advanced indexing with a Python
  scalar value materializes a 1 byte CPU bool tensor and copies it to the
  device with a blocking copy, which is the 1 byte pageable HtoD and the
  cudaStreamSynchronize inside `_reuse_decode_buffers` on every step.
  `rep_rows` is non None whenever any row has a repetition penalty other
  than 1.0 (talker.py:1177-1184), and the talker default is 1.05
  (request_builders.py:1095), so it fires on every decode step.
- `_compute_mrope_positions_decode`, sglang forward_batch_info.py:1146-1210,
  reached from `_build_forward_batch` (base.py:438-443) because the talker is
  an mrope model. It reads `mm.mrope_position_delta.item()` per request
  (:1177, host only, the delta is a CPU tensor built at request build time
  by `linear_mrope_positions` with device None, request_builders.py:820-827,
  mrope_positions.py:12-27) and then builds
  `torch.tensor(deltas_list, dtype=torch.int64, device=cuda)` (:1178-1180),
  a pageable blocking HtoD of 8 times the batch size bytes. That is the
  second cudaStreamSynchronize and the 8 byte pageable HtoD at bs 1. The
  precomputed branch at :1160-1165 is skipped because the talker's
  `mrope_positions` tensor covers only the prompt (request_builders.py:812-833),
  so from the first decode step `seq_len` exceeds it.
- The event wait, base.py:203-208, records at base.py:152-153 right after
  the pinned DtoH of the sampled ids in `_stage_token_ids` (base.py:142-155),
  and waits in `_finalize` (base.py:541) one line before the only consumer,
  `output_processor.process` calling `ids.tolist()`
  (scheduling/sglang_backend/output_processor.py:35-38). The wait is at the
  earliest point the host needs the ids, but only `_emit_code_chunks_and_feedback`
  and `_publish_next_tokens` (0.4 ms) run between the record and the wait,
  so it exposes the whole GPU step.
- `sglang_execution.forward_context` (:91-93) calls
  `sampling_info.copy_for_forward()` every step, which runs sglang's
  `update_penalties` (sampling_batch_info.py:266-281): a `[bs, vocab]` fp32
  zeros allocation and the penalizer accumulate. The talker never reads
  that `sampling_info`, it samples inside the graph from its own static
  buffers (talker.py:1327-1391). Dead kernels, part of the 14 to 23 direct
  launches per step.

The rebuild path (`prepare_decode_buffers` slow path, talker.py:1064-1184)
is what the 3 to 5 ms steps in 3.2 run:

- The reuse check (talker.py:1032-1042) needs the same rids in the same
  row order and every `len(req.output_ids)` advanced by exactly one.
- Every extend forward calls `invalidate_decode_buffers`
  (talker.py:1251-1253), the only invalidation site, because a prefill's
  sampled token bypasses `_sampled_token_ids`. Prefill wins over decode in
  `get_next_batch_to_run` (sglang scheduler.py:3121-3131), so every
  admitted request forces the next decode step onto the slow path whether
  or not the batch size changed. A finished request compacts the rows
  (`filter_batch` in `update_running_batch`, scheduler.py:3485) and shifts
  rid order, which also fails the check. That covers the 73 rebuild steps
  at c16 and the 9 at c1.
- The slow path clears two `[bs, vocab]` masks (:1068-1069), rebuilds
  `unique = {int(tok) for tok in req.output_ids}` over the whole output
  history of every request (:1082-1122, linear in codes generated so far,
  while base.py:967-983 has the incremental version the talker does not
  use), waits on the previous staging event (:1124-1127), fills six pinned
  host rows and copies them once (:1128-1143), six DtoD slices
  (:1144-1154), then builds the repetition and suppress index pairs with
  `torch.tensor(..., device=cuda)` (:1156-1170). The suppress list is 1023
  tokens per request (request_builders.py:1085-1089), so the pairs tensor
  is bs x 1023 x 2 int64, about 98 KB at bs 6, copied pageable and
  blocking, followed by two index writes with a Python True (two more
  1 byte pageable copies) and a third pageable `torch.tensor` for
  `_decode_prep_rep_rows` (:1181). The 3 to 5 ms is this path.

### 3.4 Prefill batches

22 prefill batches, run_batch p50 45.6 ms: eager talker forward 36.5 ms of
host time for 8.75 ms of kernels (no prefill graph for the talker, the
prefill backend is disabled in this config), sampling 1.8 ms, `_finalize`
3.1 ms, `_build_forward_batch` 0.3 ms. Two batches took 385 and 494 ms:
the first prefill of the window (bs 1) and the first prefill batch of two
requests, where `top_k_top_p_min_p_sampling_from_probs_torch` took 272 ms
and `apply_logits_bias` 160 ms, which is the torch.compile of the sampler
path for a new batch shape (the talker runs with `sampling_backend=pytorch`).
At c16 the 13 prefills took 1.1 s of the 3.76 s window, and each prefill
stops every running request's decode for 45 ms.

### 3.5 Request build on the scheduler thread

`_run_request_builder` runs on the scheduler thread, 9.1 ms per request
(26 requests, 237 ms). The talker scheduler is built with the default
`request_build_max_workers=1` (omni_scheduler.py:196, no override in
bootstrap.py:244-258 or stages.py:1182-1193), and with one worker the
executor is None (omni_scheduler.py:243-246) so `process_input_requests`
builds inline (:897-904) before `get_next_batch_to_run` in the same
iteration. No decode step runs during a build. Inside
`build_prompt_prefill` (components/talker_prefill.py:192-253):
`_load_prompt_token_embeddings` does a pageable blocking `.to(device)`
(:380-383) and `inverse.to(device)` (:394) for a prompt length int64
tensor, `merge_prompt_modality` (:97-119) writes through a CPU boolean
mask into CUDA tensors (:115-119), and `build_user_part`
(components/talker_input.py:45-65) calls `.any()` on device masks at :60
and :63 (each a device to host sync) and boolean mask index writes at :61
and :64 (each a `nonzero` sync). At c16 that is 96 ms of the 3.76 s
window during which no decode step runs. Thinker text chunks are also
ingested on the scheduler thread (`recv_requests` -> `_on_stream_chunk`,
omni_scheduler.py:794-810, 1670-1685), each running a text projection
GEMM and a `torch.cat` of the remaining queue
(pending_text_queue.py:82-97).

### 3.6 The loop between steps

Between steps: `get_next_batch_to_run` 0.04 ms per call (11465 calls,
mostly idle polling), `process_input_requests` 0.03 ms, `process_batch_result`
0.26 ms per step (`process_batch_result_decode` 0.22 ms with a stream sync on
some steps), `_sleep_during_idle` 1.08 ms per call when nothing is ready.
At c16, 3.03 s of the 3.76 s window is inside run_batch, 0.44 s in idle
sleeps (the head and tail of the run), 0.38 s in the loop functions, 0.10 s
in request builds.

## 4. Admission accounting, why the talker runs 6 to 9 requests

SGLang 0.5.18, srt/managers/schedule_policy.py (the container's line
numbers match the release/v0.5.18 tree for every frame in the trace).

- `PrefillAdder.__init__` charges every running request
  `min(max_new_tokens - len(output_ids), CLIP_MAX_NEW_TOKENS) * new_token_ratio`
  (:555-562, :654-661) and defines
  `rem_total_tokens = available + evictable - that sum` (:680-685).
  `CLIP_MAX_NEW_TOKENS` is `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION`, default
  4096 (:69-75, the comment says it "only clips the estimation in the
  scheduler but does not change the stop condition").
- `add_one_req` admits a candidate only if
  `extend_input_len + min(max_new_tokens, 4096) + page_size < rem_total_tokens`
  (:1219-1237). The candidate is charged its full clipped `max_new_tokens`,
  unscaled.
- `new_token_ratio` starts at 0.7 (`SGLANG_INIT_NEW_TOKEN_RATIO`, environ.py:518)
  and decays by 0.001 per decode step that did not retract
  (scheduler.py:3553-3554, new_token_ratio_tracker.py:20-35) to a floor of
  0.098 after 600 steps. It resets to 0.7 whenever the scheduler goes idle
  (scheduler.py:4074-4075). The talker goes idle between bursts, so every
  burst starts at 0.7.
- `max_running_requests` is 32 for the talker (models/qwen3_omni/stages.py:1151),
  and the decode graphs are captured for bs 1 to 32, so 32 is the ceiling
  the accounting never reaches.

With the talker's pool of 21373 tokens, a talker prompt P of about 150
tokens, page size 1, and G frames generated so far, N running requests
leave room for one more only while

    N * (P + G + r * (4096 - G)) < 21373 - P - 4097

At r = 0.7 and G = 0 that is N < 5.8: five running, a sixth admitted, the
seventh waits. At r = 0.5 (200 steps into the burst) N < 7.8, at r = 0.3
N < 12. The events run shows exactly this: 6 running at the start of the
burst, 7 and 8 later, 9 at most, with 10 requests waiting a median of
1.0 s. The physical need is P + 42 frames, about 200 tokens per request,
so the pool could hold 100 such requests and the graph ceiling of 32
would bind first.

If the estimate were 256 instead of 4096 (the clip, not the stop), the
running charge would be P + G + r * (256 - G) and the candidate
P + 257, so N < 60 at r = 0.7, and the ceiling of 32 would bind. The
stop condition would stay at 4096 (or the codec EOS), and the pool would
be protected the way SGLang protects it for every LLM: when
`check_decode_mem` fails, `retract_decode` evicts the youngest requests
and they re prefill (scheduler.py:3490-3526, schedule_batch.py:2816-2865).

## 5. What this settles for E

### 5.1 Findings

- The step is not 21 ms. That number in 04 section 2.1 was the GPU span
  divided by graph launches with the idle time between c1 requests inside
  it. The measured cycle is 8.4 ms at bs 1 and 9.4 ms at bs 6 in steady
  state, 8.9 and 12.3 ms on average with rebuilds and contention stalls.
  #1018's 13 ms at c8 with 3.45 ms of exposed host time is the same shape
  as our 12.3 ms at bs 6 with 3.6 to 4.4 ms of exposed host time.
- At c16 the talker is admission bound before it is step bound. Six to
  nine requests run while ten wait a second each, because the KV
  reservation uses `max_new_tokens` 4096 for outputs of 42 frames
  (section 4). This is the largest single term in the 2.4 s c16 latency.
- The exposed host time per step is 3.3 to 4.4 ms against 5.1 to 5.8 ms
  of GPU time, so a one step lookahead could shorten the steady cycle by up
  to 40 percent, but only if the host path has no synchronization with the
  stream. It has two per step (section 3.3), each of which under lookahead
  waits for the in flight step, so the host work serializes behind the GPU
  and the loop gets slower. That is the mechanism behind #1320's measured
  regression. The order in 05 section 7.3 stands, now with evidence: syncs
  out first, then overlap.
- The rebuild after every prefill (3 to 5 ms, 25 of 173 steps at c16) is
  dominated by re sending a 1023 entry suppress list per request as
  pageable index pairs and rebuilding the repetition set over the whole
  output history, neither of which changes between steps.
- The prefill itself (45 ms eager for 8.75 ms of kernels, plus 385 to
  494 ms on the first prefill of a new batch shape from the torch.compile
  of `apply_scaling_penalties`) and the 9 ms request build on the
  scheduler thread stop every running request at c16, 1.2 s of the 3.76 s
  window.

### 5.2 The design, one PR per seam, in order of measured value

E1, talker admission estimate. Set `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION`
for the talker process from the talker stage's `env_defaults` (stage
processes are spawned with the spawn context and the stage env is applied
before the child imports sglang, pipeline/stage_workers.py:161-185,
mp_runner.py:95). The value is the reservation estimate, not the stop:
`max_new_tokens` stays 4096 and the codec EOS stays the real stop. This is
the knob SGLang built for outputs that are usually far shorter than their
cap. Tradeoff: if enough requests run long at once the pool fills and
SGLang retracts. The PR must therefore prove one of two things before it
lands: that the talker survives `retract_decode` (a retracted talker
request re prefills and would re emit frames to code2wav, which must not
happen), or that with the chosen estimate and `max_running_requests` 32
the pool cannot fill for any prompt the request builder accepts
(32 x (P_max + 4096) is far above 21373, so this needs the retract path
to be correct or a talker side bound on running requests times worst
case). Expected effect at c16: the running count goes from 6 to 9 to 16,
the 1.0 s admission wait disappears, latency p50 from 2.4 s toward 1.4 s.
Measured, not assumed: the run in section 8.

E2, the two stream syncs and the per step pageable copies, talker side.
(a) request_builders.py:812-833 attaches a `MultimodalInputs` carrying
linear positions and delta 0 whenever `talker_can_use_linear_mrope` holds,
which is every request without image or video grids. SGLang computes
exactly those positions itself for requests with no multimodal input:
`arange(prefix_len, prefix_len + extend_len)` on extend
(forward_batch_info.py:1212-1246) and `seq_lens - 1` on decode
(:1160-1179), both on the device with no host read. Leaving
`req.multimodal_inputs` None in the linear case removes the
`.item()`, the pageable `torch.tensor`, and the stream sync from every
decode step with bit identical positions. The non linear case (video in
the talker prompt) keeps the current path. (b) components/talker.py:1046
writes a Python True through advanced indexing. Writing a preallocated
device bool scalar (or `index_put_` with a device value tensor) keeps the
same mask update with no host copy and no sync. (c) The slow path's
`torch.tensor(rows + toks, device=cuda)` pairs (:1156-1170, :1181) become
pinned staged copies or, for the suppress list, one static mask row
computed once per model and broadcast, since the list is the same 1023
tokens for every request (request_builders.py:1085-1089). (d)
`_execution_context(isolate_sampling=True)` (base.py:263) runs
`update_penalties` every step, a `[bs, vocab]` fp32 zeros plus the
penalizer accumulate, for a `sampling_info` the talker's decode never reads
(it samples inside the graph from its own buffers, talker.py:1327-1391).
The prefill's sglang sampler does read it (base.py:787), so the skip is
decode only and is a validation item, not a design item, until the prefill
path is shown to see the same penalties without the copy. Proof for a to c: the same trace, zero
cudaStreamSynchronize per steady step, zero pageable HtoD per steady step,
identical codes on the numerics harness.

E3, the rebuild after a prefill. The prefill invalidation exists because a
prefill's sampled token bypasses `_sampled_token_ids` (talker.py:1051-1054).
Writing the prefill's sampled token into `_sampled_token_ids` at its row
(`post_prefill` already holds it as `result.next_token_ids`,
talker_model_runner.py:78-103) lets the reuse check hold across a
prefill, and the row order shift after `filter_batch` is handled by
keeping the per row state keyed by rid instead of by row index. With E2c
the remaining rebuild is the six pinned sampling rows and the repetition
set, which base.py:967-983 already maintains incrementally. Proof: rebuild
steps only on genuinely new rows, c16 average cycle within 10 percent of
the steady cycle.

E4, the request build and the prefill. Request builds move off the
scheduler thread with `request_build_max_workers` above 1, which the
scheduler already supports (omni_scheduler.py:196, 240-246), after the
build's device work (the text projection GEMM and the mask writes) is
checked for stream safety on a worker thread. The prefill's eager 36 ms
for 8.75 ms of kernels is a launch bound forward and needs either the
prefill graph that omni's policy currently refuses for the talker (04
section 5.1) or fewer launches, which is model code work sized after E1
to E3 land.

E5, overlap, only after E2 and E3, on top of #1204 and #1320 rather than
beside them, re measured against the steady cycle.

Not in scope: the 1.7 ms cudaGraphLaunch and the 5 ms GPU step both
scale with the 1890 kernel nodes per step, which is the model's launch
count (the code predictor's per codebook sub steps inside the graph). That
is model code work with its own measurement, not a loop change.

## 6. B, what the run showed

`processes.preprocessing.num_replicas: 4` on the fp8 colocated config fails
at placement validation with the ValueError from
`_validate_colocated_qwen_replicas`, naming preprocessing as the replicated
stage
(sglang_omni/models/qwen3_omni/placement.py:70-82). The rule is blanket, it
rejects any replicated process under `Qwen3OmniSpeechColocatedPipelineConfig`,
including a GPU less one. The disaggregated speech config does not have the
rule (placement.py:56-68 only checks thinker and talker GPU sharing).
The fp8 baseline arm did not boot either: `image_encoder` was killed with
SIGKILL during startup twice, with no cgroup OOM event. That is outside the
code and needs a host side look before the next run.

So B needs a change before it can be measured on the colocated config:
either the placement rule admits replicas of processes with no GPU stage, or
the run moves to the disaggregated speech server on two GPUs. 05 section 4
is revised accordingly.

## 7. Corrections to earlier documents

- 04 section 2.1: talker step period 21 ms, replaced by section 3 here.
- 05 section 7.1: the disagreement with #1018 is resolved, not open.
- 05 section 4.3: the colocated config cannot run replicas.
- 04 section 5.4: one message per token, not two.

## 8. Validation tasks

- E1 run: the talker at c16 and c32 with the estimate set on the talker
  stage env, reading the running count from the events, the admission wait,
  RTF and latency, plus one run that forces the pool to fill (long texts at
  c32) to observe what a talker retract does to the audio.
- E2 run: the same with_stack trace after E2, counting cudaStreamSynchronize
  and pageable HtoD per steady step (expected zero), and the c16 average
  cycle.
- Confirm the admission arithmetic against a decode log at interval 1 on a
  boot with its own port (the run script names logs by port and the traces
  boot overwrote the decode log boot's file).
- Check the fp8 SIGKILL on the host (dmesg, container runtime logs) before
  the next fp8 run.
- B: decide between admitting GPU less replicas in the colocated placement
  rule and running B on the disaggregated speech server, then run.
