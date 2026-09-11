# 18. Qwen3-TTS decode step overlap plan

Written 2026-09-11 from whole file reads by four Opus agents, every anchor below re-read by
hand on main `80b5aaed7` plus S3 (branch `perf/qwen3-tts-decode-step`), sglang pinned at
0.5.19. Readout 17 and the E2 timeline (runbook 15 section 7) are the measurements. Nothing
here is asserted from a report alone.

## 1. Requirement

Remove the device idle of the decode step without changing the output bits. E2 measured the
step on main at c1 as 8.56 ms with 3.28 ms of device idle, and at c16 as 9.55 ms with 3.79 ms
idle. The idle sits in three spans, from the E2 timeline of the median c1 step:

```
host   |launch bb 0.53|  eager sampling + prep 1.06  |launch predictor 1.47   |stage|      wait 3.9       | tail 1.37 |
       0            0.53  0.61                      1.69                   3.16  3.31                   7.22        8.56
device |  idle 0.53   |    backbone 2.0 (367 nodes)   | idle 0.52 | predictor 4.03 (1062 nodes, 0.67 gaps) | idle 1.37 |
```

- 0.53 ms before the backbone: the host is inside `cudaGraphLaunch` for the backbone graph,
  367 nodes at about 1.4 us per node, and the device has nothing queued.
- 0.52 ms before the predictor: the predictor launch is issued at 1.69 after the eager layer 0
  sampling and its wrappers, and its 1062 node submission ends at 3.16, after the backbone
  ended at 2.53.
- 1.37 ms after the predictor, 2.05 at c16: the host tail. The scheduler thread waits for the
  predictor to finish, then runs result processing and the next batch's preparation with the
  device empty.

Every metric matters: throughput, latency, time to first chunk, inter chunk latency, quality,
memory. The talker's output bits must not move.

## 2. Mechanics, verified

### 2.1 The synchronous loop and its one wait

Qwen3-TTS runs `_event_loop_normal` (omni_scheduler.py:2313-2344): `get_next_batch_to_run`
(:2330), `run_batch` (:2334), `process_batch_result` (:2336). The builder passes no async
flag (engine_builder.py:218-225), so the scheduler default `enable_async_decode=False`
(omni_scheduler.py:199) applies.

One `execute` (base.py:297-349) launches the backbone forward, samples, then calls
`post_decode` which for Qwen3-TTS is `_collect_codes` (model_runner.py:170-197):

```
model_runner.py:188   self.model.code_predictor_forward(layer0_codes, hidden, ...)   device, 4 ms
model_runner.py:196   self._stage_token_ids(result, result.next_token_ids)           D2H of [bs] ids, then Event.record
```

The ids D2H is enqueued after the whole predictor chain. `_finalize` (base.py:575-620) then
waits on that event, `_resolve_host_token_ids` at base.py:183-188, the single blocking wait
of the step. It therefore waits for the predictor, although the ids it needs existed 4 ms
earlier, right after the sample.

### 2.2 What the host does after the wait, and what it reads

After the wait: `output_processor.process` turns the pinned ids into `RequestOutput.data`
(host only, output_processor.py:35-62); `post_process_outputs` (model_runner.py:199-227)
clones `_output_codes` and `_output_embeds` (device to device, stream ordered after the
predictor), records `codes_ready`, and per row, gated on `int(req_output.data) == eos_id`,
appends the code view, the stream chunk view and the feedback view to host lists of device
tensors; then `process_batch_result`, the finish path, and the next `get_next_batch_to_run`.

The only host resolved value on the whole step N to step N+1 chain is the layer 0 token id,
for the EOS gate. Everything the predictor produces is consumed on the device:

- the next step's talker input is `_output_embeds[row]` plus the next text row, stacked by
  `_write_feedback_buffers` into the graph's embedding buffer (model_runner.py:292-313), a
  device op over device views; the relayed token id is never used to build it;
- the stream chunk leaves as a device view plus `codes_ready_event`, and the vocoder waits on
  the event (request_builders.py:1611-1645, streaming_vocoder.py, #2046);
- `prepare_decode_buffers` (sglang_model.py:989-1093) reads static per request fields only,
  restages on a batch composition change with six pageable H2D copies (:1075-1092).

Two other synchronizations exist on the scheduler thread, neither per step: the finish path
`torch.cat(...).cpu()` per finished request (request_builders.py:1546-1549, a pageable D2H
that drains the stream), and the pageable restage copies above when the batch changes.

### 2.3 The async decode loop, and why it is not the tool here

`_event_loop_async_decode` (omni_scheduler.py:2525-2612) launches step N (`_run_batch_launch`,
:2571) before resolving step N-1 (`_resolve_and_process`, :2580). The runner split is
`execute_launch` (base.py:351-412, decode only by assertion at :375) and `execute_resolve`
(:414-451). The overrun of a request that finished in step N-1 but was launched into N is
handled once, in the scheduler and base: a pre resolve snapshot of finished or retracted rows
(:2398-2400), emit suppression by `skip_rids` (:1462, :1536), and a lockstep trim of
`batch.reqs` and `next_token_ids` (:2406-2416). The five ASR models run it with no runner
code; Higgs, MOSS-TTS-Local, Zonos2 and the Qwen3-Omni thinker override the hooks in four
different staging styles. `enable_overlap` (sglang's own loop) is refused on the omni scheduler
(:2357-2362) and is exclusive with the async flag (:294-300).

Three facts make it the wrong tool for this step:

1. It changes timing the user cares about. A decode step's chunks are emitted at resolve, one
   iteration later than in the sync loop (:1536 against :1437), so the first chunk of every
   request arrives one step late; the other models accept that behind a minimum batch size.
2. It changes numerics unless the repetition penalty moves. `lookahead_eligible` (base.py:728-758)
   routes any batch with `repetition_penalty != 1.0` to the sync path, and Qwen3-TTS defaults
   to 1.05 (request_builders.py:1446), so today every batch would be ineligible. sglang's own
   overlap loop accumulates the penalty one token behind (scheduler.py:1865 before :1899,
   schedule_batch.py:3311 reading `req.output_ids`), which is a numerics change relative to
   our sync loop, not a bit exact one.
3. It needs the codec collect ported into the launch and resolve hooks. Today the runner
   overrides neither (model_runner.py:77-106), so enabling the flag would run the base plain
   LM hooks, never call the predictor, and `_write_feedback_buffers` would fall back to the
   token embedding (model_runner.py:277-281) without raising: a silent quality failure.

The lookahead's remaining value, once the predictor no longer serializes the host, is hiding
the backbone launch submission and the sampling prep behind the previous step, which
section 3's slices already do inside the sync loop.

## 3. Design

The predictor is a 4 ms device tail whose only host consumer is a token that exists before it
starts. The step serializes the host behind it because of the order of one enqueue. Reordering
that enqueue lets the sync loop pipeline by itself: the host tail, the next batch's
preparation and the next backbone launch all run while the predictor executes, and the device
sees the next backbone queued behind the predictor.

```
before (c1)
host   |bb launch|  prep  |pred launch|stage|        wait for predictor        |   tail   |bb launch| ...
device |  idle   | backbone |  idle   |            predictor              |   idle   |  idle   | backbone
after slice A
host   |bb launch|  prep  |pred launch|stage|wait bb+sample|  tail  |bb launch|  prep  |pred launch| ...
device |  idle   | backbone |  idle   |            predictor              | backbone  |  idle |  predictor
```

Host work per step after the change: tail 1.4 + backbone launch 0.53 + prep 1.06 + predictor
launch 1.47, about 4.5 ms at c1, against device work of about 6 ms, so the host stays ahead
and the step becomes device bound: backbone 2.0 plus the gap before the predictor 0.5 plus
predictor 4.0, about 6.5 ms against 8.56, if the two remaining synchronizations are also
removed. At c16 the tail is 2.05 and the host is closer to the device; the measurement says.

### Slice A. Stage the token ids before the predictor

`_collect_codes`: call `_stage_token_ids` right after the sample, before
`code_predictor_forward`. The event then covers the backbone, the sample and the ids copy,
and `_finalize` returns while the predictor runs. Everything after the wait is unchanged: the
clones in `post_process_outputs` are enqueued behind the predictor by stream order, the
feedback stack reads device views, the stream chunk carries its event. Bit exact by
construction: no kernel changes, only the position of one independent copy in the queue.

Two consequences the slice must carry:

- `codes_ready` is recorded after the clones, as today, so the vocoder's wait is unchanged.
- The talker's `_has_pending_code_step` flag spans `post_decode` to `post_process_outputs`
  inside one serial `execute` (model_runner.py:52, 177, 197, 206-208); unchanged, the loop
  stays serial.

The slice also declares what the research found latent: the runner is sync only. A
`lookahead_eligible` override returning False with a one line why, so the async flag can never
run the plain LM hooks against this model silently.

### Slice B. The finish path copy off the stream drain

With slice A the per finished request `.cpu()` becomes the step's remaining drain: on a finish
step the host waits for the predictor again. Replace the pageable copy with a pinned
non blocking copy and an event recorded after it, the same shape as `_stage_token_ids`, and
carry the event with the payload. The vocoder side waits the event before reading: in the
same process the payload passes by reference and the vocoder's serving thread waits; across
processes the runtime waits the event on the stage's asyncio thread before packing the tensor
through shared memory (stage_io.py:377-441), so the scheduler thread never waits. Bit exact.
Its cost is one pinned buffer per in flight finish, bounded by the outbox.

### Slice C. The restage copies non blocking

`prepare_decode_buffers` builds six device tensors from Python lists on every batch change
(sglang_model.py:1075-1092). Build them into pinned host buffers and copy non blocking, so a
batch change does not drain the stream either. Bit exact. At c16 this is the p90 of
`prepare_decode_buffers`, 293 us against 19 at p50 in E2.

### Slice D, its own plan. Close the gap before the predictor

After A to C the remaining idle is the 0.5 ms between the backbone and the predictor: the eager
layer 0 sampling and suppress, 1.06 ms of host work, plus the predictor launch submission.
Capturing that sampling into the replay, the way the sub step sampling already is, lets the
predictor launch follow the backbone launch directly. The layer 0 sampler is sglang's
(`model_runner.py:1840: sample` in E2), the seam is `_sample_next_token_ids` in our runner.
Separate plan; it changes which kernels sample and needs its own identity experiment.

Fewer predictor nodes, doc 04's remaining fusions, shorten the predictor itself and its
launch; they continue independently of this plan.

## 4. Ownership

| Piece | Owner | Contract this plan relies on |
| --- | --- | --- |
| token rail `output_tokens_buf` | sglang FutureMap (overlap_utils.py:87-120, 598-612) | device gather at forward entry, no host |
| `_output_codes`, `_output_embeds` | omni talker (sglang_model.py:929-937) | rewritten in place by the replay, snapshot by clone |
| `pending_feedback_queue`, `output_codes` | omni request data | host deques of device views, consumed on device |
| the EOS gate | omni runner (model_runner.py:221) | reads the host token id only |
| `codes_ready_event` | omni (#2046) | recorded after the clone, waited by every reader |
| finish payload `audio_codes` | omni (request_builders.py:1534-1565) | slice B adds its readiness event |

## 5. Gates

- Bit identity: seeded c1, 1088 of 1088 WAVs equal to main, for A, B and C, since none of them
  touches a kernel. The census unchanged in every count.
- The E2 timeline after slice A: the event wait ending before the predictor's device start,
  the next backbone launch issued during the predictor, device idle in the step down by the
  tail. This is the mechanism gate, read from `perfkit.py timeline` and `hosttail`.
- A/B on the box, one boot per arm and point: c1 and c16 throughput, median, p95, p99, RTF,
  WER and similarity inside the band, peak memory equal as a level. Streaming pair at c16,
  since the step timing is what streaming feels: requests per second, audio seconds per
  second, time to first chunk, inter chunk latency, playback continuity.
- Nothing worse on any metric beyond the boot spread.

## 6. Order

1. Slice A, one commit plus the sync only declaration, on a branch from main after #2108
   merges: `perf/qwen3-tts-stage-ids-early`. Unit tests: the enqueue order (the ids copy and
   its event are recorded before the predictor is called, on a fake model that records the
   call order), the resolved ids equal the sampled ids, the feedback and code views still come
   from the post predictor snapshot, `lookahead_eligible` False.
2. Box run per section 5. If the step at c1 does not fall toward the device bound, the
   timeline says which synchronization is left, and B or C is next; otherwise B and C follow
   as their own slices on the same protocol.
3. Slice D's plan from a whole file read of the sampler path, after the A to C readouts.

## 7. Findings recorded, not acted on here

- The `history` clone in `_write_feedback_buffers` (model_runner.py:307) allocates a
  `[bs, hidden]` block per step that every request in the batch keeps alive for its life;
  only the retract replay reads it. A separate item.
- Two `torch.cuda.Event` objects are constructed per step (base.py:132, model_runner.py:217).
- The base async gate does not check `return_logprob`, so a logprob request on the ASR models
  reaches a blocking `.tolist()` inside `execute_launch` (base.py:946). Not our model; noted
  for the tracker.
- `scheduler_prefill_start` is stamped at launch and `scheduler_first_emit` at resolve on the
  async loop (omni_scheduler.py:1516, :1468-1480), so first token metrics on that loop
  include the lag on one side only. Relevant to how #1204 and #1320 read their TTFA numbers,
  whose four features were never measured separately.
