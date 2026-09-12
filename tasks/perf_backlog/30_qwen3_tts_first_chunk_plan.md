# 30. Plan: remove the first chunk cost of the decode step overlap

Basis: doc 29 (the interpreter lock is held 79 percent of the time at c16 on early ids
against 69 on control, and the three sibling stages pay the first chunk in lock waits),
and the mechanics inventory of 2026-09-13 (Opus, every anchor below verified by hand).
Target: first chunk mean at c16 within 10 ms of control while keeping early ids'
throughput. Every item is one slice on top of early ids, measured as one pair on the plain
server command in one session, event recorder in pass 2, dmon on, GPUs 1 to 3 recorded.

Not in this plan, all measured dead: the switch interval (E4a), sibling stream priorities
(E4b), split processes on one GPU (E4c), the bounded run ahead (docs 27 and 29), #2126
(doc 29). A second GPU for the siblings is a deployment option, not a fix of this cost.

Lock holders on early ids, from doc 29 section 5, 20 s window: talker scheduler 29
percent (5.1 ms per step), initial vocoder worker 15 (11 ms per bootstrap), preprocessing
workers 10, reference encoder 9, follow up workers 5, the rest 11. The plan takes the two
largest first.

## P1. Prime the codec state with the reference prefix before the first frame

What. Today the first stream chunk is the reference codes concatenated with the first
generated frame (request_builders.py:1623-1640), so the initial worker's first decode has
fresh frames equal to reference frames plus initial chunk frames (streaming_vocoder.py:1440
and 1450), a width that is never a captured graph key (cold keys are the initial chunk
frames only, 774-782; incremental_codec_cuda_graph.py:464 rejects the width), so every
reference prefixed bootstrap runs the eager incremental decoder, about 1000 Python
dispatched kernels and 2(K+1) gather and scatter ops per request (incremental_codec.py:
610-673, codec_state_arena.py:183-238), in a singleton cohort keyed on its own width
(streaming_vocoder.py:2471), holding the lock 11 ms and waiting 48 ms for it, on the first
chunk's critical path.

Change. The talker sends the reference codes as a prefix message of their own as soon as
the request is built (the data carries `ref_code` from preprocessing), before prefill. The
vocoder accepts stream before payload already (config.py:74). The vocoder's contract gains
a prefix only chunk (today `ingest` rejects it, streaming_vocoder.py:1213-1218) that the
initial worker consumes into the request's arena slot with zero generated frames (today
`_build_incremental_plan` returns None at 1397-1399 for that case). When the first frame
arrives its fresh width is the initial chunk frames, a captured cold key, and bootstraps of
different requests share a cohort (batch buckets 1, 2, 4, 8).

Why it is sound. The stateful decoder is specified to give the same waveform for any
partition of the frame sequence, and its test asserts that at rtol 2e-5
(tests/unit_test/qwen3_tts/test_incremental_codec.py:376-392); the graph against eager
tolerance is 2e-4 (636). Priming is one more partition boundary, inside the envelope the
vocoder already ships with. The reference audio was never emitted (reference_trim_frames,
1467), so the prime produces no output.

Expected. The bootstrap on the critical path goes from an eager 1000 op decode to one
replay, and cohorts batch, so the lone bootstrap segment (45 ms on early ids, 27 on
control) and the queue behind it both drop. The prime's own Python moves 50 to 70 ms
earlier (preprocessing plus build plus prefill) and off the first chunk path; its lock
hold stays until a follow on buckets it. About 13 points of the lock's 79.

Gate. Lone bootstrap segment, code chunks received before first audio, first chunk mean
and p99, req/s; the partition test extended with a prime boundary; WER and similarity in
band on the c16 pass.

Files. request_builders.py stream_output_builder (prefix message, first chunk without the
reference), streaming_vocoder.py latch_stream_contract, ingest, _build_incremental_plan,
_run_initial_batch (prime path), one test per changed contract.

## P2. Cut the scheduler's per step Python

The scheduler holds 5.1 ms per 15 ms step. The inventory found where:

- a. `_write_feedback_buffers` (model_runner.py:253-313): two 16 iteration loops issuing
  six tensor ops per request through four static method indirections
  (talker_model_runner.py:432-474, pending_text_queue.py:88-99), 96 dispatches per step,
  plus a full [16, hidden] clone. Change: keep the pending feedback and text rows in one
  preallocated device buffer per slot and build the batch with one index_select each,
  history from the same clone. Same values, same stack order, bit exact. Largest item
  here.
- b. Object churn: `_build_sched_output` (omni_scheduler.py:1456-1465) builds 16
  SchedulerRequest objects and a runtime import per step, `output_processor.process`
  (output_processor.py:53-61) 16 RequestOutput objects, and `_finalize` (base.py:596-611)
  walks the same 16 requests three more times. Change: one pass, objects reused across
  steps. Bit exact by construction.
- c. `_emit_prefill_start_for_batch` (omni_scheduler.py:1568-1585) loops all 16 requests
  every decode step; its sibling has the early out (1600-1601). Change: the same early
  out.

Not changed: sglang's per step penalizer allocation (sampling_batch_info.py:267-279,
driven by the 1.05 repetition penalty at request_builders.py:1446) and
cumulate_penalty_output_tokens; those are the dependency's, and the penalty is the
model's default.

Gate. Scheduler GIL hold per step and lock held fraction in a Nsight pair, req/s, first
chunk mean; identity at c1 seeded, 1088 of 1088.

## P3. Cut the preprocessing per request Python

- a. `_build_qwen3_tts_pad_embed` (request_builders.py:654-671, called at 1286)
  recomputes a request independent constant with seven ops per request. Change: build
  once per model. Bit exact.
- b. `build_embedding_cache_key_ids` (request_builders.py:644-651) copies the whole
  [P, hidden] prompt embedding to the host, a pageable synchronization of the side
  stream, then hashes P rows in Python. Change: key from the host side inputs that
  determine the rows (text ids, reference code ids, speaker embedding digest) with the
  same key semantics. This is also where the warm pass radix hit gap of doc 25 section 4
  lives. Needs its own correctness argument (a key must change when and only when the
  rows would); measured separately.
- Not changed: the ICL codec embedding loop (sglang_model.py:805-815) uses a different
  table per codebook and cannot collapse.

Gate. Preprocessing segment, first chunk mean, radix hit counts in the serve log.

## P4. The chunk path

- a. Three `_emit_event` calls per chunk on the loop thread build metadata dicts before the
  recorder returns on inactive (runtime.py:1477, 1484, 544; event_recorder.py:188). Change:
  check the recorder before building. 48 dict builds per step at c16.
- b. Sixteen chunks per step take three queue hops each with three wrapper objects
  (omni_scheduler.py:1498, runtime.py:1104 and 947, streaming_vocoder.py:2097). Change:
  one message per step for the batch's chunks; the vocoder's `_can_batch_stream_chunks`
  path exists and other vocoders use it (streaming_simple_scheduler.py:42).
- c. `_handle_stream_chunk` holds `_state_lock` across ingest and decode scheduling
  (streaming_simple_scheduler.py:483-486) and the initial worker re-acquires it to plan
  and to commit (streaming_vocoder.py:2229, 2482). Change: schedule outside the lock.

Gate. Follow up worker and initial worker lock waits, inter chunk mean, first chunk mean.

## Order and accounting

P1, P2a, P3a, P4a in that order, each its own pair; P2b, P2c, P3b, P4b, P4c after, each
its own pair. The lock fraction to recover is about 10 points to reach control's 69 and
about 20 to go below it; P1 and P2a together are the 18 the accounting reaches, the rest
are single digits each. #2123 merges when the first chunk gate holds on the stacked
result; #2126 is re-measured on top of that stack and merges only if it moves req/s.
