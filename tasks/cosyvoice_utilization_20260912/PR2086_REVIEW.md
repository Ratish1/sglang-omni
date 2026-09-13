# PR 2086 review and bisection

Reviewed at head `4de7afccc` (merge of upstream main `f58228dfb` into
`fix/cosyvoice-stream-chunk-order`). Diff under review: `git diff f58228dfb 4de7afccc`,
4 files.

PR commits, in order:

1. `62b70e326` [Bugfix][Fun-CosyVoice3] Preserve deferred stream chunk order
2. `28c272900` Remove author tags from streaming-order comments
3. `cd486d29d` fix: preserve batching while consuming deferred stream messages

All file:line anchors below are at `4de7afccc`.

---

## 1. What the PR does and why the defect is real

The CosyVoice3 streaming vocoder scheduler has two message sources, not one: the
thread-safe `inbox` queue, and a scheduler-loop-local `_pending_messages` deque
(`sglang_omni/scheduling/streaming_simple_scheduler.py:72`). The deque exists because
collectors read ahead of the dispatch loop and have to push messages back.

On main the deque was written by several sites but read by exactly one:
`_next_message` (`streaming_simple_scheduler.py:175-181`). Every other reader went
straight to the inbox:

- `_collect_new_request_batch` (base) read `self.inbox.get_nowait()`
- `_collect_stream_chunk_batch` (base and the CosyVoice override) read
  `self.inbox.get_nowait()`
- `_wait_for_first_hop_peers`, `_wait_for_follow_up_peers`, `_ingest_ready_inbox`
  (CosyVoice) read `self.inbox.get_nowait()` / `self.inbox.get(timeout=...)`

That is a genuine ordering defect, and the corruption path is concrete. The CosyVoice
chunk collector parks a request's duplicate chunk aside rather than stopping on it
(`sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py:277-282`), pushing it back to
the head of `_pending_messages` at
`sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py:285-288`. Immediately after,
`_handle_stream_chunk_batch` runs the pump, whose first act is
`_wait_for_first_hop_peers` (`streaming_vocoder.py:488`). On main that reader went to
the inbox and could ingest the same request's *next* chunk while the earlier one was
still parked. Ingestion appends to a flat list,
`state.tokens.extend(int(token) for token in codes.tolist())`
(`streaming_vocoder.py:561`), so chunk N+1 landing before chunk N permanently corrupts
the token sequence for that request; every later hop decodes garbage and the leftover
finalize decodes a wrong-length tail. The same reader would also ingest `stream_done`
ahead of a parked chunk, finalizing a request whose tokens are incomplete. Corrupted
audio and 300 s timeouts at c16 are consistent with that trace.

The fix direction is right: `_pending_messages` holds strictly older messages than the
inbox, so every reader must drain it first. The PR adds a shared helper
`_get_batch_message` (`streaming_simple_scheduler.py:249-253`), routes all four
collectors through it, and inlines the same pending-first branch into the three
CosyVoice peer readers (`streaming_vocoder.py:361-362`, `429-430`, `453-456`).

What the PR gets wrong is the *scope* of the barrier. Only per-request ordering is
required. Nothing in `_ingest_peer_message` (`streaming_vocoder.py:322-345`),
`_ingest_stream_item` or `_handle_streaming_new_request` touches state belonging to a
different request id. The PR instead installs a global FIFO barrier across all request
ids, and that is what costs 5.2x on time to first audio.

---

## 2. Message path at 4de7afccc

```
AR stage (asyncio thread, sglang_omni/pipeline/stage/runtime.py)
  _route_stream_item          runtime.py:946-949  --> inbox.put(stream_chunk)
  _receive_stream_signal      runtime.py:928-932  --> inbox.put(stream_done)
  _execute                    runtime.py:966-971  --> inbox.put(new_request)
  (for a streaming request the payload arrives AFTER its chunks and done)

                     +---------------------------+
                     |  self.inbox  (Queue)      |  newer messages
                     +---------------------------+
                                  ^
                                  |  read ONLY when _pending_messages is empty
                                  |
   +-----------------------------------------------------------+
   |  _pending_messages (deque)  older messages, loop-local      |
   +-----------------------------------------------------------+
        ^ push back                          | pop left
        |                                     v

scheduler thread: StreamingSimpleScheduler.start()   streaming_simple_scheduler.py:124-145
  |
  +-> _next_message              streaming_simple_scheduler.py:175-181
  |      pending.popleft() else inbox.get(0.1)
  |
  +-> _handle_message            streaming_simple_scheduler.py:158-173
        |
        +-- new_request  --> FunCosyVoice3._collect_new_request_batch
        |                      streaming_vocoder.py:193-228
        |                      reads via _get_batch_message()  (line 209)   [PR]
        |                      pushes back via appendleft+break (215, 224)
        |                    then _handle_new_request_batch  streaming_vocoder.py:230-253
        |                      --> _pump_streams
        |
        +-- stream_chunk --> FunCosyVoice3._collect_stream_chunk_batch
        |                      streaming_vocoder.py:255-289
        |                      reads via _get_batch_message()  (line 269)   [PR]
        |                      parks duplicates in `deferred`, first non-chunk
        |                      message (usually stream_done) in `leftover`
        |                      restores to pending head at 285-288
        |                    then _handle_stream_chunk_batch
        |                      streaming_simple_scheduler.py:498-516
        |                      --> StreamingVocoderBase.on_stream_chunk_batch
        |                          scheduling/streaming_vocoder.py:196-213
        |                          (takes _state_lock)  --> _pump_streams
        |
        +-- stream_done  --> _handle_stream_done   streaming_simple_scheduler.py:518-529

  _pump_streams (CosyVoice)      streaming_vocoder.py:478-501
    first iteration:  _wait_for_first_hop_peers   streaming_vocoder.py:347-372
                      [PR] line 361-362: pending.popleft() before any inbox read
                      (optional) _wait_for_follow_up_peers  streaming_vocoder.py:413-440
                      [PR] line 429-430: same
    later iterations: _ingest_ready_inbox          streaming_vocoder.py:442-460
                      [PR] line 453-456: pending.popleft() before any inbox read
                      then _wait_for_first_hop_peers if group size == 1 (496-497)
    every iteration:  _pump_one_step               streaming_vocoder.py:462-476
                        select_step_participants   streaming_vocoder.py:581-613
                        (first hops beat follow-ups: line 593)
    returns only when select_step_participants() == []

  _ingest_peer_message           streaming_vocoder.py:322-345
    stream_chunk                       -> ingest, return True
    new_request with stream=True       -> latch, return True
    new_request with stream=False      -> appendleft, return False   <-- BARRIER
    stream_done (any)                  -> appendleft, return False   <-- BARRIER
    aborted request id (any type)      -> drop, return True
```

The two `return False` arms at `streaming_vocoder.py:342` and `:344` are the hinge of
this review. They existed on main unchanged. What the PR changed is that the parked
message is now *visible* to the next reader.

---

## 3. Findings, most severe first

### F1 [P1-PERF] A parked `stream_done` becomes a permanent barrier for the whole pump, so a new request's first hop cannot preempt the backlog

`streaming_vocoder.py:453-456` (`_ingest_ready_inbox`), with
`streaming_vocoder.py:342-345` (`_ingest_peer_message`) and
`streaming_vocoder.py:285-288` (`_collect_stream_chunk_batch` restore).

Trace, request A backlogged and finishing, request B arriving:

1. `_collect_stream_chunk_batch` for A scans, parks A's duplicate chunks in `deferred`,
   hits A's `stream_done`, sets `leftover = done_A` and breaks
   (`streaming_vocoder.py:272-274`).
2. Lines 285-288 restore pending as `[deferred..., done_A, ...older]`.
3. `_handle_stream_chunk_batch` runs the pump under `_state_lock`
   (`scheduling/streaming_vocoder.py:201-211`).
4. `_pump_streams` first iteration calls `_wait_for_first_hop_peers`
   (`streaming_vocoder.py:488`). Line 361 pops pending, ingests A's parked chunks, then
   pops `done_A`. `_ingest_peer_message` line 344 pushes it back and returns False, so
   line 371-372 returns. The reader never touches the inbox.
5. `_pump_one_step` runs one hop. The loop continues, `first` is now False, so
   `_ingest_ready_inbox` runs (`streaming_vocoder.py:495`). Line 453 sees pending
   non-empty, pops `done_A`, gets False back at line 459, returns. Again the inbox is
   never touched.
6. Steps 4-5 repeat for every remaining iteration. `_pump_streams` only returns when
   `select_step_participants()` is empty (`streaming_vocoder.py:499-501`), that is when
   every already-latched request has no ready hop left.
7. Only then does control return to `start()`, `_next_message` pops `done_A`, finalizes
   A, and the *next* loop iteration finally reads B's `new_request` / chunks from the
   inbox.

On main step 4 and step 5 went straight to the inbox on every call, so B's chunks were
ingested between hops, B became first-hop ready, and `select_step_participants`
preferred first hops over A's follow-ups (`streaming_vocoder.py:593-600`). That is
exactly the contract the existing test `test_inbox_first_hop_preempts_follow_up_backlog`
(`tests/unit_test/fun_cosyvoice3/test_streaming.py:714-742`) protects, and that test
still passes only because it runs with an empty `_pending_messages`.

Once a live `stream_done` sits at the pending head, the between-hop ingestion path is
dead for the rest of the pump. At c16 in steady state there is nearly always a
`stream_done` in flight, and the collector parks the first non-chunk message it meets,
so the barrier is close to permanent. That matches the measured shape: time to first
audio 0.73 s -> 3.80 s (each new request waits out a full backlog drain), inter-chunk
interval 1.02 s -> 0.21 s (a request's backlog now drains back to back with no
preemption), no latency tail because nothing is dropped any more.

Fix: make the barrier per request id rather than global. `_ingest_ready_inbox` and the
two peer waits should keep a non-ingestible message parked but continue scanning past
it for messages belonging to *other* request ids, and only stop when they reach a
message of the same request id as a parked one. Concretely, scan `_pending_messages`
left to right holding a `blocked: set[str]` of request ids that have a parked message
ahead, take messages whose request id is not in `blocked`, and only then read the inbox
with the same rule. Nothing in `_ingest_peer_message` needs cross-request ordering.

### F2 [P1-PERF] The 30 ms peer wait collapses to zero whenever anything is parked, so first-hop and follow-up coalescing fall back to B=1

`streaming_vocoder.py:359-372` and `streaming_vocoder.py:427-440`.

Same mechanism, separate effect. `_wait_for_first_hop_peers` is supposed to hold a
singleton first hop for up to `_first_hop_peer_wait_ms = 30`
(`streaming_vocoder.py:78`) so equal-shape peers can join one causal Flow call. With
the PR, line 361 takes the pending branch unconditionally, ignoring `remaining`
entirely. The first non-ingestible pending message returns False at line 371 and the
wait exits after zero elapsed time. `_wait_for_follow_up_peers` has the identical shape
at line 429.

So the batching knife that `select_step_participants` depends on
(`streaming_vocoder.py:596-612`) is disarmed in exactly the regime it was built for.
This is consistent with the measured RTF 0.50 -> 0.69 and QPS 3.99 -> 3.75, though part
of main's apparent speed is its dropped chunks and shortened audio, so I am not
claiming the whole delta.

Second, smaller problem at the same lines: the pending branch ignores the deadline, so a
deep parked backlog is ingested in full inside what is documented as a 30 ms window,
under `_state_lock`, with a `codes.tolist()` per chunk (`streaming_vocoder.py:561`).

Fix: fold the pending drain into the deadline check, and apply the per-request rule from
F1 so an unrelated parked message does not end the wait.

### F3 [P4-PROCESS] The PR ships a test that locks in the liveness loss, and no test covers preemption across a parked message

`tests/unit_test/fun_cosyvoice3/test_streaming.py:919-950`,
`test_peer_readers_do_not_pass_pending_done`, asserts for all three readers that with a
`stream_done` for `req-a` parked, `req-b`'s chunk sitting in the inbox is *not*
ingested (`assert scheduler._stream_states["req-b"].tokens == list(range(start))` and
`assert scheduler.inbox.qsize() == 1`). That is the regression written down as the
intended contract.

The existing liveness contract at `test_streaming.py:714-742` only ever exercises an
empty `_pending_messages`, so nothing in the suite fails when the between-hop ingestion
path dies. This is why a 5.2x first-audio regression shipped green.

Fix alongside F1: change this test to assert that the readers skip past the parked
`stream_done` for other request ids while still refusing to ingest a later message of
the *same* request id, and add a variant of `test_inbox_first_hop_preempts_follow_up_backlog`
that seeds `_pending_messages` with a done marker.

### F4 [P2-MAINTAIN] The pending-first branch is written four times, three of them by hand

`streaming_simple_scheduler.py:249-253` introduces `_get_batch_message` for exactly this
purpose, then `streaming_vocoder.py:361-366`, `429-434` and `453-456` reimplement it
inline. `_ingest_ready_inbox` at 453-456 is a literal duplicate of the helper and should
call `self._get_batch_message()`. The two peer waits need a blocking variant; give
`_get_batch_message` the `block` semantics they need rather than forking the logic.
Four copies of a rule this subtle will drift.

### F5 [P2-MAINTAIN] `_get_batch_message` makes "appendleft then break" a load-bearing, undocumented invariant

Now that collectors read `_pending_messages`, any site that pushes back into it inside a
collector loop and then `continue`s would re-read its own pushback forever. The PR
avoids this correctly in `_collect_new_request_batch`
(`streaming_simple_scheduler.py:266`, `290`, `292`, `301`, `311`, `317`) by using a local
`deferred` list, but four other sites still `appendleft` directly and rely on an
immediately following `break`: `streaming_simple_scheduler.py:340`, `:345`,
`streaming_vocoder.py:215`, `:224`. None is buggy today. The rule needs to be stated on
`_get_batch_message`, or those four sites should move to the same `deferred` pattern.

### F6 [P2-MAINTAIN] Parked backlog is rescanned on every chunk collection

`streaming_vocoder.py:267-288`. The CosyVoice chunk collector now pops the entire
`_pending_messages` deque through `_get_batch_message`, re-defers every duplicate, and
restores it. With k messages parked, each subsequent chunk message re-walks what is left
of them, so draining a k-deep parked backlog costs O(k^2) deque operations in the
scheduler loop. On main the collector only walked the inbox. Bounded and pure Python, so
not a blocker, but it is new per-message work in the serving loop at exactly the
concurrency where the backlog is deep.

### F7 [P3-STYLE] `self.inbox.get(timeout=0.0)` instead of `get_nowait()`

`streaming_simple_scheduler.py:253`. With the default `timeout=0.0` the helper takes the
blocking `Queue.get` path, which computes `endtime = time() + timeout` and then a
`remaining` before raising `Empty`. Two extra `time()` calls per message in a collection
loop that runs per chunk, and the "blocking get with a zero timeout" idiom reads worse
than `get_nowait()`. Branch on the timeout: `get_nowait()` when `timeout <= 0`,
`get(timeout=timeout)` otherwise.

### F8 [P3-STYLE] Naming and comments

`streaming_simple_scheduler.py:249-250`. `_get_batch_message` says nothing about the
rule it enforces, and the docstring "Consume deferred work before new arrivals without
running dispatch hooks" explains the second half (no hooks) but buries the first half,
which is the whole point and the thing callers must not violate. `[C2-FUNCTION-SHAPE]`.
Something like `_take_oldest_message` with a docstring that states "the pending deque
holds strictly older messages than the inbox; never read the inbox while it is
non-empty" carries the invariant.

`streaming_vocoder.py:451-452`, "The collector may have pushed back an older chunk or
done marker", is now wrapped oddly after the author-tag strip in `28c272900` and reads
as a two-line fragment. Minor.

---

## 4. Bisection

### `62b70e326` - [Bugfix][Fun-CosyVoice3] Preserve deferred stream chunk order

Touches `sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py` only, plus 150 lines of
tests.

Behaviour changes:

- Adds `if self._pending_messages: return [first_msg]` at the top of both CosyVoice
  collectors (`_collect_new_request_batch`, `_collect_stream_chunk_batch`). This stops
  the collectors reading the inbox past a parked message, which closes the corruption
  hole, but it also disables cross-request batching entirely whenever anything is
  parked: no first-hop payload coalescing and no chunk coalescing.
- Adds the pending-first branch to `_wait_for_first_hop_peers`,
  `_wait_for_follow_up_peers` and `_ingest_ready_inbox`.

Introduces: **F1** and **F2** (the readers), plus a batching kill switch of its own, plus
the first version of the test in F3.
Fixes: the ordering defect in the collectors and in the three readers.

**This is the commit that introduces the first-audio regression mechanism.** The three
reader hunks in this commit are byte-identical to what is at `4de7afccc`; neither later
commit touches `_wait_for_first_hop_peers`, `_wait_for_follow_up_peers` or
`_ingest_ready_inbox`. The trace is the one in F1: after this commit, `_ingest_ready_inbox`
pops the parked `stream_done`, `_ingest_peer_message` returns False at
`streaming_vocoder.py:344`, and the reader returns without ever calling
`self.inbox.get_nowait()`, for every iteration of the pump.

### `28c272900` - Remove author tags from streaming-order comments

Comment-only, four lines across the source file and the test file. No behaviour change,
introduces and fixes nothing. Leaves the two awkward comment wraps noted in F8.

### `cd486d29d` - fix: preserve batching while consuming deferred stream messages

Touches the shared base for the first time.

Behaviour changes:

- Deletes both `if self._pending_messages: return [first_msg]` early-outs from the
  CosyVoice collectors and replaces the inbox reads with `_get_batch_message()`, so
  batching works again while still honouring the pending-first rule. This fixes the
  batching kill switch that `62b70e326` introduced.
- Adds `_get_batch_message` to `StreamingSimpleScheduler`
  (`streaming_simple_scheduler.py:249-253`) and routes the base
  `_collect_new_request_batch` and `_collect_stream_chunk_batch` through it.
- Rewrites the base `_collect_new_request_batch` pushback to a local `deferred` list
  restored with `extendleft(reversed(deferred))` at the end
  (`streaming_simple_scheduler.py:266`, `:315-317`). This is *required* by the
  `_get_batch_message` change (otherwise the collector re-reads its own pushback in an
  infinite loop) and it is also a small correctness improvement over main, which used
  `append` for deferred done markers and would have restored them behind older pending
  messages, and `appendleft` for the cost-boundary message, an inconsistency this commit
  removes.

Introduces: **F4** (the helper exists but the three readers keep their hand-rolled
copies), **F5**, **F6**, **F7**, **F8**.
Fixes: the batching loss from `62b70e326`. Does **not** touch F1 or F2.

---

## 5. Plain answers

### (a) Is the PR safe to merge as a correctness fix?

The correctness argument holds: the ordering defect is real, the traced corruption path
(`state.tokens.extend` at `streaming_vocoder.py:561` applying chunk N+1 before chunk N)
is exactly what the reviewer saw on 8xH200, and I found no new ordering hole, lost
message, duplicated message or abort interaction introduced by the PR. Aborted messages
are still dropped correctly at `streaming_vocoder.py:323-324` and
`streaming_simple_scheduler.py:281`, so an aborted `stream_done` does not become a
barrier, and the pump still terminates.

But it is not safe to merge as it stands, because the fix is scoped to all request ids
when only per-request ordering is required, and that over-scoping costs 5.2x on time to
first audio on the default streaming path (F1) and disarms the coalescing the vocoder
depends on (F2). Both are in one commit, `62b70e326`, and both are confined to three
readers. Fixing them is a bounded change: make the pending scan skip messages whose
request id has nothing parked ahead of it. Land the corrected version rather than this
one. If something has to ship today because c16 is corrupting audio, this is a
defensible stopgap, but it should go in with the follow-up already written, not as the
end state.

### (b) Which problems remain after merge?

- F1: new requests cannot preempt a backlog while a live `stream_done` is parked; time
  to first audio 0.73 s -> 3.80 s at c16.
- F2: the 30 ms first-hop and follow-up peer windows collapse to zero whenever anything
  is parked, so Flow runs B=1 where it used to coalesce.
- F3: a shipped test asserts F1's behaviour as correct, and nothing in the suite guards
  preemption across a parked message.
- F4, F5: four hand-written copies of the pending-first rule and an undocumented
  "appendleft then break" invariant that the new helper makes load-bearing.
- F6: O(k^2) deque rescanning of a k-deep parked backlog in the chunk collector.
- F7, F8: zero-timeout blocking `get`, helper naming and docstring.

Not introduced by this PR but worth recording while this code is open: the peer waits
block on `self.inbox.get(timeout=remaining)` for up to 30 ms while holding
`_state_lock` (`scheduling/streaming_vocoder.py:201`), which serializes every other
scheduler path for the duration.

### (c) Do the shared-base changes affect the other schedulers?

The base changes are `_get_batch_message` (new), `_collect_new_request_batch`
(pending-first plus the `deferred` restore) and `_collect_stream_chunk_batch`
(pending-first), all in `streaming_simple_scheduler.py`. Subclasses of
`StreamingSimpleScheduler` / `StreamingVocoderBase`: Zonos2, dots_tts, ming_tts,
nemotron_voicechat, fun_cosyvoice3 (both the CUDA scheduler and the MLX one in
`stages.py:1617`), minimax_music3, higgs_tts, qwen3_tts, moss_tts, qwen3_omni Code2Wav,
moss_tts_local, fishaudio_s2_pro.

In practice nothing but CosyVoice changes behaviour, for two separate reasons.

For every subclass that does not override `_next_message`, `_pending_messages` is
written only by the base collectors, and it is always empty when a collector starts.
Induction: `_collect_new_request_batch` leaves pending as `[stream_done markers...,
one terminal message]` (`streaming_simple_scheduler.py:315-317`) and
`_collect_stream_chunk_batch` leaves at most one pushed-back message
(`streaming_simple_scheduler.py:340`, `:345`). The serving loop pops one message per
iteration at `streaming_simple_scheduler.py:176-177`; the `stream_done` markers dispatch
to `_handle_stream_done` and never enter a collector, so by the time the single terminal
message is popped and reaches a collector, the deque is empty and `_get_batch_message`
degenerates to the old inbox read. The `append` -> `deferred` + `extendleft` rewrite is
likewise a no-op when the deque starts empty, and is strictly more correct when it does
not.

Qwen3-Omni's `Code2WavScheduler` is the one subclass that bulk-fills `_pending_messages`
from the inbox in its own `_next_message`
(`sglang_omni/models/qwen3_omni/components/code2wav_scheduler.py:608-647`), so it is the
one that could have been hit. It is not, on either path. Its
`super().__init__(None, sample_rate=..., stream_source_hint=...)`
(`code2wav_scheduler.py:161-165`) leaves `batch_compute_fn=None` and `max_batch_size=1`,
so `_collect_new_request_batch` returns at `streaming_simple_scheduler.py:259-264`
before the loop. And `_collect_stream_chunk_batch` is effectively unreachable: with
batching on, `_next_message` groups chunk runs itself and returns `None`
(`code2wav_scheduler.py:612-636`); with batching off, `_handle_message` routes chunks to
`_on_chunk` at `streaming_simple_scheduler.py:165-168`. The only residual is a narrow
race, a chunk arriving between `_drain_inbox` and the `inbox.get` at
`code2wav_scheduler.py:643`, which would enter `_collect_stream_chunk_batch` with a
non-chunk parked at the pending head and now stop immediately instead of coalescing from
the inbox. That costs at most one coalescing opportunity per occurrence and Code2Wav
already does its own run-grouping, so it is not worth blocking on.

Net: the shared-base half of this PR is safe for the other schedulers. All of the risk
is in the CosyVoice reader changes from `62b70e326`.
