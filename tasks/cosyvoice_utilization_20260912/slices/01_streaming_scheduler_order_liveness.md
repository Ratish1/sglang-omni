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
   its first emission, and a stream that has emitted nothing has zero slack (the client has
   nothing to play). One formula covers every kind: a first hop is at zero, an underrunning
   follow-up is negative and beats it, a final of a stream with buffered audio waits behind
   more urgent hops and runs once its own slack is the smallest. Ties break by ready time. A
   waiting stream's slack falls monotonically and every hop that runs raises the runner's
   slack by at least one hop of audio, so once a stream is the most urgent, every other
   in-flight stream can run at most once before it.
4. Batch formation on top of urgency: the most urgent candidate defines the key `(hop, offset)`,
   the token window of its next hop; every candidate of the same kind with that key joins in
   urgency order up to the cap. Finals never share a step (finalize mode). The peer window
   (30 ms today, `_peer_wait_ms`) is a delay before running a singleton whose expected peer,
   a live stream with the same key that is still receiving tokens, is not ready yet; it is
   expressed as the loop's blocking timeout, never as an inbox read, and it is skipped for a
   stream that has emitted and whose slack is smaller than the window.

## 3. Changes by file

### `scheduling/streaming_simple_scheduler.py` (shared by ten schedulers)

- Ordered source: #2086's `_get_batch_message(timeout)` (pending first, then inbox) and its
  hold-aside parking (`extendleft(reversed(deferred))`) are the primitives; `_next_message`
  switches to `_get_batch_message(0.1)` so the loop getter and the collectors share one source.
  The qwen3_omni scheduler's own `_next_message` override keeps working because it is the loop
  getter, not the collector.
- Serving loop (`start`): when `_has_ready_work()` is true, take messages through
  `_get_batch_message(timeout=self._ready_step_delay())` and handle each; on `Empty` call
  `_run_ready_step()` and loop. When it is false, the loop is unchanged: `_next_message()`, which
  now delegates to `_get_batch_message(0.1)`. Defaults: `_has_ready_work` False,
  `_ready_step_delay` 0.0, `_run_ready_step` no-op, so every existing subclass, including the
  ming_tts and qwen3_omni `_next_message` overrides, takes the unchanged branch.
- `_handle_stream_done`: `on_stream_done` may return `None`, meaning completion is deferred; the
  existing tail (emit messages, `_clear_request_state`) is `_complete_stream_request(rid,
  messages)` so a subclass can complete a stream from inside a step. qwen3_tts's own
  `_handle_stream_done` override still calls this base path for its synchronous finals.

### `scheduling/streaming_vocoder.py` (shared vocoder base)

- `_pump_one_step()` is one iteration of the old `_pump_streams` body: returns `None` when no
  stream is ready, else the ids `on_step_failure` aborted (empty after a successful step).
  `_pump_streams` loops over it for the subclasses that pump inside `on_stream_chunk_batch`
  (dots, ming, moss local) with the same return contract as before. `_run_ready_step` runs one
  step under `_state_lock` and the abort callbacks off it.
- `_finish_stream(rid)` is the body of the old `on_stream_done` (flush remainder, fallback,
  stream chunk, result, record completed); `on_stream_done` returns it, so the base default is
  unchanged and a subclass can run the same sequence from a step.

### `models/fun_cosyvoice3/streaming_vocoder.py`

Removed: `_collect_new_request_batch`, `_collect_stream_chunk_batch`, `_first_hop_group_size`,
`_has_joinable_first_hop_peer`, `_ingest_peer_message`, `_wait_for_first_hop_peers`,
`_follow_up_group_size`, `_has_joinable_follow_up_peer`, `_wait_for_follow_up_peers`,
`_ingest_ready_inbox`, `_pump_one_step`, `_pump_streams`, `_first_hop_key`, `_follow_up_key`,
and the `on_streaming_new_request` and `_handle_new_request_batch` pump triggers.
`_stream_chunk_batch_distinct_requests` returns to the base default False: all queued chunks of
a request ingest in one ordered batch. `_first_hop_peer_wait_ms` is renamed `_peer_wait_ms`
because it applies to follow-ups too. No inbox read is left in the file.

Added or changed:

- `_CosyVoice3StreamState` gains `done: bool`, `ready_since: float | None` (clock time the
  current hop or final became runnable; cleared and restamped after every step) and
  `first_emit_at: float | None`. `speech_offset` already counts emitted samples.
- `self._clock` (`time.monotonic`) is the one time source, replaceable by tests.
- `on_stream_chunk_batch`: ingest only under `_state_lock`, no pump. `on_streaming_new_request`,
  `ingest` and `on_stream_done` end with `_mark_ready(state)`. `on_stream_done` marks
  `state.done` and returns `None`.
- `_step_kind(state)`: `first` (offset zero, tokens ready), `follow_up` (ready, offset > 0),
  `final` (done and no full hop left), else not runnable. `_step_key(state)`: `(hop, offset)`.
  `_slack_s(state, now)`: the formula of invariant 3.
- `_ranked_candidates()` sorts runnable streams by `(slack, ready_since, request_id)`;
  `_select(ranked)` takes the head and fills same-kind same-key candidates up to the cap
  (`_can_batch_stream_chunks` and `_can_batch_follow_up_hops` gate the fill as before).
  `_has_ready_work`, `_ready_step_delay` and `select_step_participants` are built on these.
- `build_step_plan` carries the kind; `run_step` for `final` calls `_finish_stream` and
  `_complete_stream_request` and returns nothing to emit; for hops it runs the packed batch or
  the native single hop as before, then stamps `first_emit_at` on the first emission and
  restamps `ready_since`.

## 4. Tests

- New `tests/unit_test/fun_cosyvoice3/test_streaming_replay.py` with the two recorded inbox
  sequences copied to `tests/unit_test/fun_cosyvoice3/fixtures/`. The replay feeds the events at
  their recorded times against a fake clock, drains and steps like the serving loop (honouring
  `_ready_step_delay` by advancing the clock), and charges every step 20 ms. Asserts: no errors,
  one result per request and last for that request, the finalize Flow call of every request
  carries its chunks in arrival order, emitted samples equal tokens times frames times samples,
  no state or parked message is left, packed batches occurred, and the wall time between a
  request's final becoming runnable and its result is at most its slack at that moment plus one
  round of (in-flight + 1) steps and peer windows.
- Contract tests in `test_streaming.py` moved to the loop level (`_serve` helper: drain, then
  ready steps until idle; `_Clock` fake): finals are deferred to a step, a first hop runs before
  follow-ups that have slack, underrunning follow-ups run before a first hop, three-way least
  slack ordering, queued peer chunks are ingested before the step, the peer delay and its
  skip rules, pending before inbox order with the final after the last chunk, fallback errors
  surface as error messages. Removed with their mechanism: the two peer-wait tests, the two
  peer-reader tests, the streaming payload collection test, and the distinct-request chunk
  collection expectation.
- Base tests: the loop drains pending then inbox before the ready step; the ready-step delay
  waits for late messages; `on_stream_done` returning `None` defers completion until
  `_complete_stream_request`. Vocoder base: `_pump_one_step` returns `None` when idle;
  `_run_ready_step` aborts a failed step's participants and runs the abort callback.
- Every other existing test in the four files is unchanged.

## 5. Gates and measurement

1. Unit tests above on the H100 venv:
   `pytest tests/unit_test/fun_cosyvoice3/test_streaming.py tests/unit_test/fun_cosyvoice3/test_streaming_replay.py tests/unit_test/pipeline/test_streaming_simple_scheduler.py tests/unit_test/scheduling/test_streaming_vocoder.py tests/unit_test/moss_tts_local tests/unit_test/ming_tts tests/unit_test/dots_tts tests/unit_test/qwen3_omni tests/unit_test/qwen3_tts -q`.
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
