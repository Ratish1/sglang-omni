# 21. Qwen3-TTS, the two blocking copies of the decode loop (slices C and B of plan 18)

Written 2026-09-12 from whole file reads by three Opus agents (sglang_model.py, model_runner.py,
request_builders.py, streaming_vocoder.py, stage_io.py, stages.py, omni_scheduler.py,
model_runner/base.py, pipeline/stage/runtime.py), every anchor below re-read by hand on
`perf/qwen3-tts-stage-ids-early` at `b26971164` (main with S3 merged, slice A in). Readout 20
section 3 is the measurement that orders this work.

## 1. Requirement

After slice A the host runs ahead of the device and the predictor sits queued on the stream.
Two pageable copies on the scheduler thread now wait for that queue instead of an empty
device. Readout 20, c16 churn steps, p90 self time per step:

| frame | main (readout 17) | after slice A |
| --- | ---: | ---: |
| `sglang_model.py: prepare_decode_buffers`, the restage | 0.29 ms | 3.15 ms |
| `request_builders.py: apply_sglang_qwen3_tts_result`, the finish copy | 0.10 ms | 2.99 ms |

They run on the steps where a request joined, finished or was retracted, which at c16 is most
steps (28 of 1077 steps ran a full batch). Both copies must become non blocking with the
output bits unchanged, and no reader of either payload may observe it before it is complete.

## 2. Mechanics, verified

### 2.1 The restage

`prepare_decode_buffers(requests)` (sglang_model.py:989-1095), called from `before_prefill`
(model_runner.py:66) and `before_decode` (:87), both ahead of the forward in
`_prepare_and_forward` (base.py:506-516). It keys a skip on the ordered list of
`(request_id, epoch)` (:998-1013), which changes on every finish, join, retraction (a
permutation, schedule_batch.py:3107) and abort, and on every prefill that runs between two
decode steps, since the prefill restages rows `[0, P)` for its own requests (:1095 caches the
prefill's list, the next decode misses).

On a change it reads six static per request fields on the host (:1023-1055) and writes six
persistent device buffers of shape `(max_running_requests,)` allocated once (:942-959), each
as

```
sglang_model.py:1080   self._sub_temperature_tensor[:batch_size] = torch.tensor(
sglang_model.py:1081       sub_temperatures, device=device, dtype=...)
```

that is a pageable CPU tensor, a pageable host to device copy on the current stream, then a
device to device copy into the buffer. Six of each per restage. The buffers are read only on
the device: five inside the predictor chain (:1618-1622, :1632-1635), whose slices are baked
into the captured predictor graph (:1319-1387, capture state :1221-1250), and the semantic
seed as a view installed on `sampling_info.sampling_seed` every step (model_runner.py:166-174)
for sglang's layer 0 sampler. No host read of any of them anywhere (grep clean for `.item`,
`.tolist`, `.cpu` in sglang_model.py). The writes are in place, so the graph sees new values;
nothing rebinds the attributes.

Between the restage and the next predictor replay the stream carries the feedback stack, the
backbone replay, the sampler and the ids staging, with no host synchronization
(model_runner.py:88, base.py:517-521, model_runner.py:191-201).

### 2.2 The finish copy

`apply_sglang_qwen3_tts_result` (request_builders.py:1534-1565) runs on the scheduler thread
inside `stream_output` (omni_scheduler.py:1692), after step N's `_finalize` and before step
N+1's `before_decode` (:2352 returns before :2346). It stacks the per step code views, which
are views of the clones `post_process_outputs` enqueued in earlier steps
(model_runner.py:218-229), prefixes `ref_code`, and does

```
request_builders.py:1546   codes = torch.cat([...], dim=0).cpu()
```

a pageable device to host copy that waits for everything queued, including the current step's
predictor, whose codes this request will never use (the EOS step's codes are not appended,
model_runner.py:226). The result goes into `StagePayload.data["audio_codes"]` (:1556-1557)
as a CPU int64 tensor of shape `[ref_code_len + steps, num_code_groups]`, and the scheduler
puts `OutgoingMessage(type="result", data=result)` on the outbox (:1731-1737) with
`metadata=None`; `OutgoingMessage` has a `metadata` field (scheduling/messages.py:18-23).
Request data is detached in the same `finally` before the put (:1720, :98-100), so the
payload is the only thing that outlives the request on the AR side.

Who reads `audio_codes`:

- The stage runtime's asyncio thread drains the outbox (runtime.py:1098-1149,
  `run_in_executor` for the blocking get) and routes the result (:1170-1229): stream done
  signals first (:1187-1192), then `_send_to_stage`. In the shipped layout every stage is in
  one process (config.py:54-85), so the payload goes by Python reference
  (`_local_dispatcher.send_payload`, :1289-1295), no copy.
- The vocoder's scheduler thread. Streaming requests: the payload is stored
  (streaming_simple_scheduler.py:470-479) and only its usage counts are read
  (streaming_vocoder.py:2762-2777); `audio_codes` is read only by the nothing emitted
  fallback (:2753-2760, :2840-2852). Non streaming requests: `_vocode_payloads`
  (:2782-2812) does `torch.as_tensor(state.audio_codes, dtype=torch.long)` and hands it to
  the codec tokenizer's decode.
- Across processes, if a layout ever splits the stages: the sender's asyncio thread packs
  every tensor (`stage_io.py:377-411`, device move at :644-646) or exports CUDA IPC handles
  (:156-181) with no readiness wait; the receiver may `.cpu()` (:713). Payload metadata is
  never forwarded; `strip_process_local_metadata` (:124-138) applies to stream chunks.

So there is exactly one seam that every reader passes through before any of them can touch
the tensor: the outbox drain on the asyncio thread, which today receives the message with an
already complete CPU tensor because the scheduler thread blocked for it.

### 2.3 The pinned helpers that exist

`base.py:148-181 _pinned_pingpong` alternates two pinned buffers and carries no event;
`_stage_token_ids` (:122-135) records its own event and `_finalize` waits it (:186, :588).
`utils/cuda_staging.py` has `GrowablePinnedBuffer` and `PinnedTransferSlot` (one buffer, one
reusable event, the owner holds the policy, :4-10), used by the streaming vocoder and
code2wav, not by the AR path. There is no host to device staging helper anywhere.

Lifetimes that matter: a non blocking copy from a pinned tensor allocated with
`pin_memory=True` goes through torch's caching host allocator, which records the copy's
stream event on the block and does not hand the block out again before that event, so a
pinned tensor may be dropped by Python right after the copy is enqueued. A device temporary
freed after an enqueued copy from it is safe on one stream by the caching allocator's stream
ordered reuse. Neither needs `record_stream` on a single stream.

## 3. Design

### Slice C. The restage from pinned twins, non blocking

```
before
host   | six torch.tensor(list, device=cuda): each blocks until its pageable copy is staged,
       |   behind everything queued (the predictor of the previous step)         | forward launch ...
device |        previous predictor                                | H2D x6 | D2D x6 | backbone ...
after
host   | fill six pinned twins (host memcpy) | six copy_(non_blocking) | event | forward launch ...
device |        previous predictor                                      | H2D x6 | backbone ...
```

`Qwen3TTSTalker.__init__` allocates, next to the six device buffers, two slots of six host
twins of the same shape and dtype, pinned when the device is CUDA, plain CPU otherwise, and
one `torch.cuda.Event` per slot. `prepare_decode_buffers` on a change:

1. takes the slot whose turn it is and waits its event if one was recorded (already complete
   in practice: `_finalize` of the previous step waited an event recorded after that step's
   sample, which is behind the previous restage's copies on the same stream; the wait is the
   contract, not the wait position of another function);
2. fills the six twins from the same six lists (`twin[:bs] = torch.tensor(list, dtype)`, a
   host only op);
3. issues `device_buf[:bs].copy_(twin[:bs], non_blocking=True)` six times on the current
   stream, so the copies are ordered before the forward and the predictor exactly where the
   pageable copies were;
4. records the slot's event and flips the slot.

The six device to device copies disappear with the temporaries. The values, the device buffer
rows and the stream position are unchanged: bit exact. The predictor graph binding is
untouched. On a CPU device the copy is a memcpy and no event is recorded.

Host cost per restage after the change: the Python loop and six small non blocking copies,
about what readout 17 measured for the scan, against 3 ms at p90 today.

### Slice B. The finish copy non blocking, the wait on the routing thread

```
before
scheduler thread | ... _finalize | stream_output: cat, .cpu() waits for the queued predictor (3 ms) | put | next step
asyncio thread   |                                                                              drain, route
after
scheduler thread | ... _finalize | stream_output: cat, pinned copy_(non_blocking), event | put | next step
asyncio thread   |                                              drain: wait the event (executor thread), route
```

`apply_sglang_qwen3_tts_result`: the cat stays on the device, the result is copied with
`non_blocking=True` into a fresh pinned CPU tensor of the same shape and dtype
(`torch.empty(..., pin_memory=True)` when the codes are on CUDA, the caching host allocator
reuses blocks by size class), an event is recorded after the copy, and the event is left on
the request data as `result_ready_event`. The payload carries the pinned tensor under
`audio_codes` as today, same dtype and shape.

`OmniScheduler.stream_output`, at the put (:1731-1737): if the request data carries a
`result_ready_event`, it goes into the message's `metadata` under that key. That is the whole
scheduler change, model agnostic: a result may carry a device readiness event.

`_drain_outbox_external` (runtime.py:1098-1149): for a result whose metadata carries
`result_ready_event`, `await loop.run_in_executor(None, event.synchronize)` before
`_route_result`. The event completes when the predictor of the finishing step completes, the
same moment the scheduler thread used to unblock, so the stream done signal and the payload
leave the stage at the same time as today, every reader downstream (the vocoder thread, the
cross process packer, the coordinator) sees a complete tensor exactly as today, and the
scheduler thread moves on to the next step 3 ms earlier. The follower drain (:1152-1168)
does not route and needs no wait.

A copy stream that waits only the request's own `codes_ready_event` instead of the whole
stream would let the finish payload leave up to one predictor earlier than today. It needs
`record_stream` on the clones and the reference codes, or a held reference until the event,
and is not required to remove the blocking: recorded here for the streaming readout to
decide, since the final window's arrival is what it would move.

### Ownership

| piece | owner | contract |
| --- | --- | --- |
| six device sampling buffers | talker (sglang_model.py:942-959) | written in place, read by the predictor graph and the layer 0 sampler on the device |
| pinned twins and slot events | talker, new | a slot is rewritten only after its event completed |
| finish payload `audio_codes` | request builders (:1534-1565) | CPU int64 `[T, Q]`, complete when `result_ready_event` has completed |
| `result_ready_event` | request data, new field on `Qwen3TTSSGLangRequestData` | recorded after the copy, forwarded by the scheduler as message metadata |
| message `metadata["result_ready_event"]` | omni scheduler and stage runtime | the runtime waits it off the scheduler thread before routing; never forwarded |

## 4. Tests

Slice C, `tests/unit_test/qwen3_tts` on the fixture talker:

- the device buffers hold the staged values after a change, and hold the second batch's
  values after two changes in a row (order sensitive key, permuted rows);
- CUDA: with a long sleep enqueued on the stream, a restage returns while the stream is still
  busy, and after synchronize the device buffers hold the values (the copy is non blocking
  and stream ordered);
- CUDA: two restages with different values while the stream is busy leave the device with
  the second set after synchronize, and neither restage blocked (two slots);
- the unchanged batch still skips, and an intervening prefill row set restages the next
  decode (the existing key behaviour, kept).

Slice B:

- `apply_sglang_qwen3_tts_result` on CUDA data: `audio_codes` is a pinned CPU tensor of the
  right shape and dtype, `result_ready_event` is set, and after its synchronize the values
  equal the stacked codes with the reference prefix; on CPU data the payload is unchanged and
  no event is set;
- `OmniScheduler.stream_output` puts the event in the result message's metadata when the
  data carries one, and leaves metadata None otherwise (bare scheduler, fake adapter);
- `_drain_outbox_external` routes a result only after its event completed: a real event
  recorded behind a stream sleep and a `_route_result` stub that asserts `event.query()`
  (CUDA), and a result without metadata routed unchanged.

## 5. Gates

- Bit identity: seeded c1, 1088 of 1088 hashes equal to the previous B (slice A at
  `b26971164`), the census unchanged.
- The mechanism: E2 hosttail at c16 rows 8, the two frames of section 1 back at their main
  p90 (0.3 and 0.1 ms); churn step idle down; the c1 step unchanged at 6.25 ms.
- c16 boot: throughput and latency against slice A's B boots; the c16 corpus gain is the
  point of these slices.
- Streaming pair at c16: TTFC, inter chunk, continuity unchanged or better; the final
  window's arrival not later (slice B keeps the stream done timing by construction).
- By the 2026-09-12 protocol: A is slice A's B boots (same base), so four boots on B.

## 6. Order

1. Branch `perf/qwen3-tts-nonblocking-copies` from `perf/qwen3-tts-stage-ids-early`
   at `b26971164`. Commit 1: slice C with its tests. Commit 2: slice B in three files
   (request_builders.py, omni_scheduler.py, runtime.py) with its tests. One measured PR
   for both, since they are one issue (the copies that slice A left blocking) and the E2
   hosttail attributes each frame separately; two PRs only if the box run shows one of them
   moving a streaming metric the other does not.
2. Box run per section 5 with runbook 22.
3. Then slice D's plan (the layer 0 sampling into the replay) against the c1 residue of 0.13
   ms and the churn step sampler cost readout 20 recorded.

## 7. Findings recorded, not acted on

- Padding rows `[live, bucket)` of the six device buffers are never refreshed; the graph
  samples them and the rows are sliced away. Wasted work only.
- The restage key `(request_id, epoch)` never bumps the epoch for a live request, so a
  mid stream change of a request's sampling fields would not restage. No caller mutates
  them today.
- The reference audio encoder's convolutions from the request build threads run on the
  same stream as the decode (readout 20 section 3, 1.7 ms of a churn step). A separate
  item.
