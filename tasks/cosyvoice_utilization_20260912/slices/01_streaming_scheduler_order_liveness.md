# Slice 01: streaming vocoder scheduler, message order and step liveness

Base: PR #2086 head `4de7afccc` (stacked on upstream `main` `f58228dfb`; byte-identical to the
closed #2144, which measured the ordering defect on an H200 and reported zh c16 streaming corpus
WER 6.7 to 7.2 percent falling to 0.4 to 0.8 and 300 s timeouts falling from 12 to 1 per 500).
#2086 implements invariant 1 below in the collectors and in CosyVoice's three peer readers; this
slice does not reimplement it. The residual timeout is the liveness defect this slice removes.
Owners: `sglang_omni/scheduling/streaming_simple_scheduler.py`,
`sglang_omni/scheduling/streaming_vocoder.py`, `sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py`.
Line references are to the analysis branch, which differs from main in these files only by the
diagnostic marks.

Trigger (perfkit, results sections 6 and 0): at c16 streaming the AR is ahead of the vocoder for
the whole window; a final decode waited 24.6 s in the 64-request capture; the full corpus shows
22 to 23 timeouts at 300 s after first audio and matched-sample WER of 4.8 to 6.3 percent against
1.1 to 1.4 at c1. Buffered mode is not affected by this slice's defects; it shares the base
classes and receives the ordering fix without behavior change.

## 1. Mechanics today

One inbox FIFO per stage, filled by the stage runtime in arrival order (`runtime.py:928-971`):
`new_request` (payload), `stream_chunk`, `stream_done`. For a streaming request the AR side sends
chunks, then `stream_done`, then the payload.

```
StreamingSimpleScheduler.start (streaming_simple_scheduler.py:124-145)
  _next_message (175-181): _pending_messages.popleft() else inbox.get(0.1 s)
  _handle_message (158-173)
    new_request  -> _collect_new_request_batch (249-308): reads self.inbox directly,
                    parks non-batchable messages with append (283, 285, 294)
                    or appendleft (304)
    stream_chunk -> _collect_stream_chunk_batch (310-340): reads self.inbox directly,
                    parks with appendleft (330, 335); with distinct-requests a second
                    chunk of the same request is parked
                 -> _handle_stream_chunk_batch -> on_stream_chunk_batch
    stream_done  -> _handle_stream_done (508-519): on_stream_done runs the final
                    decode now (scheduling/streaming_vocoder.py:225-249), then
                    _clear_request_state

FunCosyVoice3StreamingVocoderScheduler
  _collect_stream_chunk_batch (272-304): own copy, parks siblings with appendleft
  on_stream_chunk_batch (base 196-213): ingest batch, then _pump_streams
  _pump_streams (486-509): loop until select_step_participants is empty
    _wait_for_first_hop_peers / _wait_for_follow_up_peers (362-453): read
      self.inbox directly for up to 30 ms, ingest chunks and payloads inline,
      park anything else with appendleft and return
    _pump_one_step (470-484): select (589-628: ready first hops, else the
      largest (hop, offset) group), run_step, emit
    _ingest_ready_inbox (455-468): reads self.inbox directly between steps,
      stops at the first non-chunk message and parks it
```

Two defects follow from this structure.

Order. `_pending_messages` and `self.inbox` are two sources, and every consumer except
`_next_message` reads the inbox directly. A chunk parked in `_pending_messages` by the collector
is ingested after any later chunk of the same request that a peer wait or
`_ingest_ready_inbox` reads from the inbox. The request's token list is then out of order and Flow
decodes a corrupted prefix. This needs the AR to be ahead of the vocoder, which is the c16 regime
and not the c1 regime. The base collectors have the same two-source read, and they park with both
`append` and `appendleft`, so the parked order itself is not arrival order when more than one
message is parked.

Liveness. `stream_done` is parked by the pump and handled only after `_pump_streams` returns,
which requires an empty ready set. Under sustained load the ready set stays non-empty, so a
request whose AR has finished waits for every other request's hops. Nothing bounds that wait.

The mid-pump inbox reads exist to let a new first hop preempt a follow-up backlog. That goal is
kept; the mechanism is replaced.

## 2. Invariants the redesign establishes

1. One ordered message source. Every consumer takes messages through one method that returns
   parked messages first, and parks only to the front in arrival order. No subclass reads
   `self.inbox`.
2. Ingestion never blocks on compute. The serving loop drains every queued message into
   per-request state, then runs at most one decode step, then drains again. `stream_done` is a
   state transition, not a compute call.
3. Step selection is a pure function of state at the time of the step. Urgency is measured, not
   configured: a stream's playback slack is the audio it has emitted minus the wall time since
   its first emission. Least slack runs first; first hops and finals have zero slack by
   definition (the client is waiting with nothing to play); an underrunning follow-up has
   negative slack and beats them. Ties break by ready time. A starved stream's slack falls
   monotonically, so no stream can be starved past its own playback buffer.
4. Batch formation on top of urgency: the most urgent candidate defines the key (first hop key,
   or `(hop, offset)` for follow-ups); every candidate with that key joins in urgency order up
   to the cap. The peer window (30 ms today, `streaming_vocoder.py:79`) is a delay before running
   a singleton whose expected peer is not ready yet; it is expressed as the loop's blocking
   timeout, never as an inbox read, and it is skipped when the candidate's slack is smaller than
   the window.

## 3. Changes by file

### `scheduling/streaming_simple_scheduler.py` (shared by ten schedulers)

- Ordered source: #2086's `_get_batch_message(timeout)` (pending first, then inbox) and its
  hold-aside parking (`extendleft(reversed(deferred))`) are the primitives; `_next_message`
  switches to `_get_batch_message(0.1)` so the loop getter and the collectors share one source.
  The qwen3_omni scheduler's own `_next_message` override keeps working because it is the loop
  getter, not the collector.
- Serving loop: drain with `_take_message(0)` and handle each message; when the queue is empty and
  `_has_ready_work()` is true, call `_run_ready_step()`; when neither, block with
  `_take_message(self._ready_step_delay())`, where the default delay is 0.1 s. Defaults:
  `_has_ready_work` returns False and `_run_ready_step` is a no-op, so every existing subclass
  behaves as before.
- `_handle_stream_done`: `on_stream_done` may return `None`, meaning completion is deferred; the
  existing tail (emit messages, `_clear_request_state`) becomes `_complete_stream_request(rid,
  messages)` so a subclass can complete a stream from inside a step.

### `scheduling/streaming_vocoder.py` (shared vocoder base)

- Extract the body of one `_pump_streams` iteration into `_pump_one_step()` and implement
  `_run_ready_step` with it. `_pump_streams` keeps its loop for the subclasses that pump inside
  `on_stream_chunk_batch` (dots, ming, moss local). No default changes for them.

### `models/fun_cosyvoice3/streaming_vocoder.py`

Removed: `_collect_new_request_batch`, `_collect_stream_chunk_batch`, `_first_hop_group_size`,
`_has_joinable_first_hop_peer`, `_ingest_peer_message`, `_wait_for_first_hop_peers`,
`_follow_up_group_size`, `_has_joinable_follow_up_peer`, `_wait_for_follow_up_peers`,
`_ingest_ready_inbox`, `_pump_one_step`, `_pump_streams`, and the `on_streaming_new_request` and
`_handle_new_request_batch` pump triggers. `_stream_chunk_batch_distinct_requests` becomes False:
all queued chunks of a request ingest in one ordered batch.

Added or changed:

- `_CosyVoice3StreamState` gains `ready_since: float | None` (monotonic time the current hop or
  final became eligible, set in `ingest` and on done), `first_emit_at: float | None`,
  `emitted_samples: int`, `done: bool`. `speech_offset` already counts emitted samples; the two
  timestamps are new.
- `on_stream_chunk_batch`: ingest only (base ingestion under `_state_lock`), no pump.
- `on_stream_done`: mark `state.done`, set `ready_since`, return `None`. The payload must already
  be latched (the base keeps `_pending_done` otherwise, unchanged).
- `_has_ready_work`: any candidate. `_ready_step_delay`: the peer window remaining for a singleton
  candidate with a joinable peer whose slack permits waiting, else zero.
- `select_step_participants`: candidates are `first` (offset zero, tokens ready), `follow_up`
  (tokens ready for the next hop), `final` (done and no full hop left). Sort by `(slack,
  ready_since)`; take the head's key; fill up to the cap with same-key candidates in that order.
  A `final` never shares a step because its Flow call is finalize mode.
- `run_step`: `first` and `follow_up` as today (packed causal for two or more rows, one row
  through the same packed adapter is slice V2, not this one); `final` runs the leftover through
  `decode_delta(is_final=True)`, then `_complete_stream_request` with the stream chunk and the
  terminal result, including the nothing-emitted fallback the base performs today.
- After every emission update `first_emit_at` and `emitted_samples`.

## 4. Tests

- New `tests/unit_test/fun_cosyvoice3/test_streaming_replay.py`: feed the recorded inbox
  sequences `perfkit/fixtures/streaming_c16_gpu0_inbox.json` and
  `streaming_c16_gpu1_compile_64req_inbox.json` through the scheduler with the existing fake Flow
  and HiFT at their recorded arrival order, one message per call, with a fake clock. Assert per
  request: tokens seen by the vocoder equal the concatenation of its chunks in chunk-id order;
  the final result is emitted; the number of steps between its `stream_done` and its final is
  bounded by the number of requests in flight; no message is left in `_pending_messages`.
- Adapt the five contract tests that encode the removed mechanism
  (`test_queued_peer_chunk_joins_first_hop_batch_during_wait`,
  `test_queued_peer_chunk_joins_follow_up_batch_during_wait`,
  `test_c1_first_hop_does_not_wait_for_peers`, `test_c1_follow_up_stays_native_and_does_not_wait`,
  `test_inbox_first_hop_preempts_follow_up_backlog`) to the loop-level behavior: the peer window
  is the loop timeout, a first hop runs before a follow-up backlog only when the backlog has
  slack, and an underrunning follow-up runs first.
- Base tests: add order tests for `_take_message` and `_park_front` with more than one parked
  message; keep every existing collector test.
- Every existing test in the three files and in `tests/unit_test/scheduling/test_streaming_vocoder.py`
  must pass unchanged except the five above.

## 5. Gates and measurement

1. Unit tests above, locally.
2. Full English corpus at c16 streaming, eager: zero timeouts, zero failures, matched-sample WER
   equal to the c1 WER within run-to-run noise (0.2 points). C1 and buffered c16 must not change
   (this slice does not touch their paths; the A/B confirms).
3. Perfkit on one 16-request c16 capture: hop ledger `final` queue delay bounded by one step per
   in-flight request; `first` queue delay p50 not worse than baseline (1274 ms) once the vocoder
   host path is unchanged; continuity C50 in the profiler-off run not worse than baseline.
4. Rollback: revert the PR; no state or wire format changes.

## 6. Out of this slice

The memory budget (C2) runs as a census on the box in parallel and is a config PR. The vocoder
host path (V1 to V4) is unchanged here so the A/B isolates the scheduler. The ordering fix is
#2086 and is the base of this slice, not part of it; #2110's ingest rewrite is superseded by the
loop change. The 30 ms peer window value is unchanged; whether it earns its place at c2 to c8 is a
separate measured slice.
