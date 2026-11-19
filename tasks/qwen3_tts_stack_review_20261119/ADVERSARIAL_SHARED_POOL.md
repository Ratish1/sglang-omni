# Adversarial review: one codec slot set for streams and whole requests

Commit under review: `8b6f6985a` ("share one codec slot set between streams and whole requests") on `4c41ba4c8`, worktree `/Users/ratish/sglang-omni/.worktrees/pr4077-replace` (HEAD is `8b6f6985a`, tree clean, so every line number below is the commit's own). PR range read: `2a66cb083...8b6f6985a`.

Paths are relative to that worktree. Short names used below:

- `SV` = `sglang_omni/models/qwen3_tts/streaming_vocoder.py`
- `ARENA` = `sglang_omni/models/qwen3_tts/codec_state_arena.py`
- `GRAPH` = `sglang_omni/models/qwen3_tts/incremental_codec_cuda_graph.py`
- `BASE` = `sglang_omni/scheduling/streaming_vocoder.py`
- `SIMPLE` = `sglang_omni/scheduling/streaming_simple_scheduler.py`
- `COORD` = `sglang_omni/pipeline/coordinator.py`

How this was done: code reading only. No pytest, no box, no edit in the repository. Files read in full: `ARENA`, `SV` (all 4,180 lines), `cutedsl/state_copy.py`, `GRAPH`, `BASE`, `SIMPLE`. Read in the relevant parts: `COORD` (120 to 300, 485 to 1145), `request_builders.py` (680 to 720, 2040 to 2220), `config.py`, `config/topology.py` diff, `stages.py` diff, `pipeline/stage/runtime.py` abort path, `pipeline/sessions.py` 800 to 870, the test diffs and the whole-request test block (3523 to 4475), SGLang v0.5.23 `kv_cache_configurator.py` 2990 to 3038 for the docs claim on the KV pool. One measurement was taken, on the Mac, with a stand-in loop that contains no repository code (finding 1 says exactly what it is).

## Verdict

| Severity | Count | Finding |
| --- | --- | --- |
| P0 | 0 | Nothing found. On default sizing I could not build wrong output, a hang, a crash or lost capacity. |
| P1 | 1 | A whole request with no row turns the initial worker into a busy loop that never releases the interpreter. New in this commit. Reachable whenever `codec_state_slots` is below the number of row holders. |
| P2 | 1 | Below the cap, the two kinds now take each other's rows. A non-streaming request waits until a stream ends, with no log and no counter, and a stream that meets whole requests on the last rows decodes on the left-context path for its whole life. |
| P3 | 1 | `arena.release` is now called at four places on the initial worker where a device fault that has not yet set `cuda_decode_failed` makes it raise. The worker thread dies silently and the request is never answered. |
| P4 | 6 | Docs, warning text, stats and test gaps. |

The premise itself holds. I found no path where one request holds two rows, and no row holder the coordinator does not count, outside the abort and failure window the docs already name.

## P0

None found.

What would have been P0 and was ruled out: a row reused before its last holder's GPU work finished (section "Checked and found correct", item 2), a double release (item 3), a whole request that can wait forever on default sizing (item 4).

## P1

### 1. A whole request that finds no row makes the initial worker spin without releasing the interpreter

New in this commit. Before it, a whole request could only wait on rows held by other whole requests, and those are all in `self.whole_requests` and each pass replays one of them, so every pass did GPU work and blocked in `resolve_partial`. Now the holders can be streams, which the initial worker does not advance, so a pass can do nothing at all and the loop has no blocking call in it.

Flow of one idle pass, with W a whole request that has no row and `arena.free` empty:

```
SV:2995  run_initial_worker: while self.run_initial_pass(): pass
   |
SV:3003  if not self.whole_requests        -> False, W is waiting
   |        (the only blocking call, wait_for_first_chunks SV:3005, is skipped)
SV:3018  take_queued_first_chunks()
   |        SV:3033 initial_queue.qsize() -> 0, returns [] without blocking
SV:3027  run_whole_request_replay()
   |        SV:3143 plan_whole_request_replay()
   |           SV:3064 is_aborted(W)        -> False   (abort_lock, uncontended)
   |           SV:3083 joins_replay         -> True    (replay is None)
   |           SV:3096 arena.acquire(should_count_exhaustion=False)
   |                     ARENA:153-159 free is empty -> None
   |           SV:3100 waiting.append(W)
   |           SV:3118 self.whole_requests = waiting
   |           SV:3119 return None
   |        SV:3144 replay is None -> return
SV:3028  return True
   |
   +--> back to SV:2995 at once
```

Nothing in that path sleeps, waits on a queue, or makes a device call. It repeats until some other thread calls `arena.release`.

Concrete sequence (operator sizing below the cap, everything else default):

1. Server started with `--vocoder.factory.codec_state_slots 32`. The cap is 128 (`config.py:40-41`, `config/topology.py` `resolve_coordinator_max_in_flight`). All three stages share one process (`docs/cookbook/qwen3_tts/tuning.md:120`).
2. 32 streams have had their first decode. Each holds a row from `SV:1890-1903`. `arena.free` is empty.
3. A non-streaming request finishes generation. The vocoder's main thread builds it and queues it (`SV:2900-2906`).
4. The initial worker takes it (`SV:2799-2801`) and enters the loop drawn above.
5. The loop runs until one of the 32 streams ends (`SV:3973-3981` then `SV:1966-1981`) or is aborted. A stream holds its row from its first decode to its last, so that is the rest of an utterance.
6. For that whole time every other Python thread in the process (the talker's scheduler, both follow-up workers, the vocoder's main thread, the stage event loop that delivers aborts at `pipeline/stage/runtime.py:2679-2687`) has to take the interpreter back from a thread that never gives it up voluntarily.

When it can happen:

- Default sizing (slots equal to the cap, 128). The waiting request is itself counted and holds no row, so at most 127 counted requests can hold rows and one row is free. The loop is reachable only inside the abort and failure window: with U rows still held by aborted or failed requests, at least 128 minus U of the other counted requests must hold rows at that moment. Requests queued in the engine or before their first chunk hold none, and at most 64 run in the engine, so at least 64 of the 128 rows must be held by requests that are past their engine slot or aborted and not yet released. I consider it unreachable in normal serving. If it happens, it lasts until the vocoder handles the abort.
- `codec_state_slots` set below the cap. Reachable in steady state whenever the number of requests past their first decode reaches the slot count and one more non-streaming request arrives. The docs offer this setting as a memory trade (`tuning.md:148`, last sentence).
- No cap (`max_queued_requests` unbounded). The slot count is then always operator set, and the same applies under overload.

Evidence for the cost. Derived from the code: the pass has no blocking call. Measured, as a stand-in only: a pure Python loop of the same shape (a deque check, `queue.Queue.qsize()`, two uncontended lock acquisitions, one bisect) against a second thread that makes a 0.2 ms blocking call, on the Mac with the repository's `.venv` Python 3.11.7, script outside the repository at `/private/tmp/claude-501/-Users-ratish-sglang-omni/755476ed-3ef4-4375-b034-a20d476b4162/scratchpad/spin/gil_spin_standin.py`:

| | mean | median | p95 |
| --- | --- | --- | --- |
| 0.2 ms blocking call, no spinner | 0.306 ms | 0.306 ms | 0.310 ms |
| 0.2 ms blocking call, with spinner | 7.603 ms | 7.821 ms | 7.837 ms |

The spinner ran about 1.0 million passes per second. The interpreter's switch interval was 5.0 ms. This is not a measurement of the server. It shows the mechanism: each time another thread comes back from a call that released the interpreter, it waits for the switch interval before it runs again. Not measured: the real loop on the box, and the effect on first audio or RTF.

The new test `test_qwen3_tts_whole_request_waits_for_the_slot_a_stream_holds` (`tests/unit_test/qwen3_tts/test_pipeline.py:4074-4097`) calls `run_initial_pass()` twice with every row held and asserts nothing happened. It passes because the pass returns at once, so it records the busy loop as the expected behavior.

Smallest fix: when a pass planned no replay and whole requests are still waiting, block on `initial_queue` instead of looping. Two shapes:

- Without a timer. `acquire(should_count_exhaustion=False)` marks a waiter under `arena.lock` when it misses, `release` reads and clears that mark under the same lock and, when it was set, puts a wake item on `initial_queue`. The two share one lock, so a release either comes before the miss (the acquire then finds the row) or after it (the release then wakes the worker). `run_initial_pass` then takes the blocking branch (`SV:3003-3011`) when the last plan found no row. The two places that set `cuda_decode_failed` (`SV:470`, `SV:2427`) must put the same wake item, or the waiting requests that `SV:3048-3056` is meant to fail stay blocked until the next arrival.
- With a timer. Give `wait_for_first_chunks` a timeout and take it when the last plan found no row. Smaller change, but it is a poll.

Either way `test_qwen3_tts_whole_request_waits_for_the_slot_a_stream_holds` has to change, because its second `run_initial_pass()` would block.

## P2

### 2. Below the cap the two kinds take each other's rows, and the wait is neither bounded by a decode nor visible

This is the design of the commit, so it is a regression only for servers whose slot count is below the number of row holders. On default sizing neither half can happen outside the abort window. It is listed because the place where the operator chooses the value does not say it, and because nothing in the logs or stats shows it.

Non-streaming side. Before the commit a whole request waited only when `codec_state_slots` other whole requests were mid decode, and each of those holds its row for a few replays. Now it waits whenever that many requests of either kind hold rows, until one frees. If the holders are streams, that is until a stream ends. Sequence: the one in finding 1, steps 1 to 5. The request has already finished generating, and the mode without window graphs would decode it at once (`SV:2878-2882`). There is no fallback, no log line and no counter: `acquire(should_count_exhaustion=False)` is the only trace (`SV:3096-3101`, `ARENA:155-158`), and it deliberately counts nothing.

Streaming side. Before the commit a whole request could never take a stream's row. Now, with 32 slots on a cap of 128: 30 streams hold rows, a burst of non-streaming requests takes the last two rows for a few replays, a stream's first chunk is planned in that moment, `SV:1891-1899` finds no row, sets `incremental_codec_fallback` and the stream decodes on the left-context path to its end, although the rows free a few replays later. `use_incremental_path` (`SV:1839-1845`) never lets it back.

Who gets a freed row: `run_initial_pass` plans first chunks before whole requests (`SV:3018-3027`), so a first chunk already queued wins. In the spin of finding 1 a pass with no queued first chunk follows within microseconds of any release, so in practice the waiting whole request wins. Neither kind starves outright. I found no interleaving where a waiting whole request is passed over forever while rows keep freeing.

Smallest fix: say it where the operator decides. In `tuning.md:148` the last sentence names only left-context decodes as the price of a lower slot count. Add that a non-streaming request then waits for a stream to end. The two startup warnings (`SV:1393-1405`) say only "streams past the slots decode eagerly on the left-context path" and should name the non-streaming wait as well. Count a whole request's first miss once per request (not per pass) so `model_info` shows it. If the wait itself is not acceptable below the cap, the remedy is a design choice outside this commit (a fallback decode, or a small reserve), and I am not proposing one here.

## P3

### 3. `arena.release` can now raise on the initial worker with nothing to catch it, and the request is left unanswered

Before the commit, returning a whole request's row was `idle_whole_request_slots.append(...)`, which cannot raise. Now it is `arena.release`, which creates and records a device event before it looks at anything else (`ARENA:178-180`). On a device in a sticky error state that call raises.

The commit knows this: `release_whole_request_slot` skips the release once `cuda_decode_failed` is set (`SV:3121-3127`), and the new test simulates exactly that raise. But the flag is set in only two places, both after a decode launch or resolve failed and the stream then failed to synchronize (`SV:470`, `SV:2427`). A device fault that surfaces anywhere else first leaves the flag False while the device is already dead. The four call sites are then unguarded:

| Site | Where | Inside a handler | What is lost if it raises |
| --- | --- | --- | --- |
| `SV:3065` | plan, aborted request holding a row | no | the worker thread |
| `SV:3136` via `SV:3108` | plan, `start_slot` or `wait_codes_ready` raised after a good acquire | yes, raises out of the `except` | the worker thread, and `emit_error` at `SV:3138` never runs |
| `SV:3188` | after a replay, aborted request | no | the worker thread, and the rows after it in the same replay |
| `SV:3213` | after a request's last window | no | the worker thread, and a fully decoded request: `SV:3214-3220` never run |

`run_initial_worker` has no guard (`SV:2992-2996`), the thread is a daemon started at `SV:1488-1492`, and nothing watches it. `schedule_initial` only checks that the attribute is not None (`SV:2746`). So after an escape every later first chunk and every later non-streaming request sits in `initial_queue` and is never answered, and the only sign is a traceback from the thread.

Concrete sequences:

- Fault and abort together. Whole request A is mid decode and holds row 5. A device fault occurs in a follow-up worker's stream. No launch or resolve has tripped the flag yet. A's client disconnects. Next pass: `SV:3064` sees A aborted, `SV:3065` calls `release_whole_request_slot`, the flag is False, `ARENA:179-180` raises, the exception leaves `plan_whole_request_replay`, `run_whole_request_replay`, `run_initial_pass` and the thread.
- Fault after the last resolve. A's last replay resolves (`SV:3170-3179`). The device faults before `SV:3213`. The release raises, A's audio is never stored or emitted.
- Flag set by another thread between the check and the call. `SV:3122` reads False, a follow-up worker then hits the unrecoverable path and the device is dead, `ARENA:179-180` raises. Same result.

All three need a device fault inside a narrow window, so this is a failure path of a failure path. The point is the outcome: before the commit these requests got an error, now they hang and take the worker with them.

Two related points in `acquire` (`ARENA:148-174`):

- `wait_event(released)` at `ARENA:164-165` is outside the `try`. If it raises, the row is neither free nor held. That is the lost row the new `try` was added to prevent, one line earlier.
- The handler at `ARENA:170-173` calls `self.release(slot)`. `zero_slot` allocates nothing (`ARENA:226-236`), so the only way I found for it to raise on CUDA is a device already in a sticky error state, and then the event record in `release` raises too. The row is not returned and the original exception is replaced by the one from `release`. So on CUDA the handler returns a row only in cases I could not construct. This is an inference from the calls, not something I ran. The unit test for it runs on a CPU arena with `zero_slot` replaced, where `release` records no event. Both callers catch the exception (`SV:2856-2857`, `SV:3107-3109`), so nothing escapes here.

Smallest fix: make `release` unable to raise. If creating or recording the event fails, retire the slot (it cannot be ordered, so it must never be reused) and return. That closes the four sites, the `acquire` handler, and the same exposure on the stream side, which is older than this commit (`SV:1981`, `SV:1997-1998`, where the event is also recorded before the retired check at `ARENA:184`). Move `ARENA:164-165` inside the `try`. The flag check in `release_whole_request_slot` can then stay or go.

## P4

### 4. Docs, sentence by sentence

Changed sentences in `docs/cookbook/qwen3_tts/tuning.md`, each against the code.

Line 148:

| Sentence | Verdict |
| --- | --- |
| "A slot takes 2.74 MiB on the 1.7B checkpoints, so the arena grows by 2.74 MiB for every request `max_queued_requests` adds." | The halving is backed (`ARENA:86`, `ARENA:134-136`, no second set). The per request figure holds for one engine replica only. The slot count is (running + queued) times the engine replicas (`config/topology.py` `engine_stage_capacity`), and each vocoder replica takes the whole count (`config.py:148-149`), so with R engine replicas each vocoder arena grows by R times 2.74 MiB per queued request. The 2.74 MiB figure is unchanged from before the commit. I did not verify it. |
| "The vocoder allocates the arena before the engine measures free memory for its KV pool, so with the default memory fraction most of that memory comes out of the KV pool." | Backed. The arena is built in the scheduler's constructor (`SV:1017-1023`), the vocoder stage is listed before the engine (`config.py:94-104`). SGLang v0.5.23 reads free memory before the engine loads (`distributed/bootstrap.py:193-209`) and sizes the pool as free memory after load minus that earlier reading times (1 minus the fraction) (`mem_cache/kv_cache_configurator.py:2990-3038`). An allocation A resident before both readings lowers the budget by the fraction times A, which is most of A. With a stage memory budget set, `sglang_omni/model_runner/sglang_model_runner.py:276-304` and `sglang_omni/utils/gpu_memory.py:303-304` use total times the stage fraction minus this process's memory, so all of A in a shared process and none of it in a split vocoder process. I read "with the default memory fraction" as excluding that case. |
| "Setting `codec_state_slots` below the cap trades that memory against left-context decodes under overload." | Unchanged text, now incomplete. After this commit the price also includes non-streaming requests waiting for a stream to end (finding 2). |

Line 150:

| Sentence | Verdict |
| --- | --- |
| "The cap counts a request until its result, or until the server publishes its abort or failure." | Backed. Result: `COORD:996`, `COORD:1023`. Abort: broadcast at `COORD:806`, then removed at `COORD:831`. Failure: broadcast at `COORD:945-947`, then removed at `COORD:957`. If the failure broadcast itself fails the request stays counted (`COORD:948-955`), which errs on the safe side. |
| "The vocoder returns that stream's slot only once it has handled the abort and any decode of the stream already running has finished, so a stream admitted in that moment can find the slots full and decode on the left-context path to its end, and a non-streaming request waits for a slot." | Backed (`SV:1966-1981`, `SV:1987-1998`, `SV:1891-1899`, `SV:3096-3101`). Gap: it describes only an aborted stream's row. An aborted non-streaming request also holds its row past the abort, until the initial worker's next plan or the end of its replay (`SV:3064-3066`, `SV:3187-3189`). |

Line 154:

| Sentence | Verdict |
| --- | --- |
| "On CUDA with asynchronous stateful decode and the window graphs, non-streaming requests decode on the same stateful decoder." | Backed (`SV:934-940`, `SV:1469-1471`). |
| "Each finished request holds one of the same `codec_state_slots` while it steps through the window graphs, with the first decode of a Base request on the tail graphs." | Backed (`SV:3096`, `SV:3213`, `SV:2940-2960`). |
| "A request counts once in the in-flight cap whether it streams or not, so the slots cover both kinds." | Backed. See "Checked and found correct", item 1. |
| "A non-streaming request that finds every slot held waits for one to free instead of decoding eagerly." | Backed (`SV:3099-3101`). It does not say that the slot may be a stream's, held to the stream's end. |
| "Any other mode, and a server whose window graphs fail to capture, decodes non-streaming requests whole, on the whole-utterance graphs where they are captured." | Backed (`SV:2878-2882`, `SV:1469-1475`). The removed clause about reserved slots staying allocated is correctly gone: no second set exists. |

No em dash and no semicolon in the changed sentences.

### 5. Startup warnings name only streams

`SV:1394-1398` and `SV:1400-1405` say that streams past the slots decode on the left-context path. With shared rows a server below the cap, or with no cap, also makes non-streaming requests wait. The text is not wrong, it leaves out the half this commit added. Fix: one clause in each.

### 6. Stats

- `exhausted` (`ARENA:155-158`) counts stream misses only. That is the stated intent and it is right for the counter's meaning. The side effect is that a whole request waiting for a row appears nowhere: not in `exhausted`, not in `left_context_fallbacks`, not in any log line. Fix as in finding 2.
- `active_slots` (`ARENA:142-146`) now includes rows held by whole requests. Before the commit those rows were invisible. This is correct and more useful.
- After `cuda_decode_failed`, a whole request's row is neither released nor retired (`SV:3122-3127`), so `active_slots` counts it for the rest of the process. Only rows in the failing replay are retired (`SV:2428-2430`, `SV:473-475`). Cosmetic, the device is dead by then.
- `total_bytes` and the startup line (`ARENA:134-136`, `SV:1147-1149`) report `num_slots` rows. The arena allocates one more, the scratch row (`ARENA:90`, `ARENA:103-104`). Same as before the commit.
- `model_info` (`SV:1409-1419`, `SV:1303-1332`, `ARENA:390-397`): every field is still computed correctly.

### 7. Tests

Does each new or changed test fail on the code it guards against:

| Test | Guards | Fails without the guarded code |
| --- | --- | --- |
| `test_codec_state_arena_returns_a_slot_whose_reset_fails` | the `try` in `acquire` | yes, `active_slots()` would be 1 |
| `test_codec_state_arena_counts_only_misses_that_do_not_wait` | `should_count_exhaustion` | yes |
| `test_qwen3_tts_whole_requests_past_their_slots_wait_for_one_to_finish` | waiting, and the flag at the call site (`exhausted_count == 0`) | yes |
| `test_qwen3_tts_whole_request_waits_for_the_slot_a_stream_holds` | waiting on a row held outside the whole requests | yes. It also fixes the busy loop as expected behavior (finding 1) |
| `test_qwen3_tts_whole_request_never_reuses_a_retired_slot` | `release` ignoring a retired slot | yes |
| `test_qwen3_tts_whole_requests_waiting_for_a_slot_fail_once_cuda_decode_is_disabled` | the flag check at `SV:3048-3056` | yes |
| `test_qwen3_tts_whole_request_fails_without_a_device_release_once_cuda_decode_is_disabled` | the flag check at `SV:3122` | yes, the raise escapes `run_initial_pass` |

Gaps:

- The release and reraise boundary in the plan (`SV:3094-3109`) lost its failure test. `test_qwen3_tts_whole_request_device_fault_on_its_slot_fails_only_itself` used to drive it through `zero_slot`. That fault is now absorbed inside `acquire`, so the plan's `except` sees a request with no slot. No test raises from `start_slot` or `wait_codes_ready` after a good acquire, which is now the only way into `SV:3108` with a row held.
- No test takes a row through a real stream. Both waiting tests hold rows with bare `arena.acquire()`. Nothing drives `release_codec_slot` or the deferred release in `finish_codec_slots` while a whole request waits, and nothing shows a stream's first chunk falling back because whole requests hold the last rows.
- No test covers a `release` that raises while the flag is still False (finding 3).
- Stale docstring at `tests/unit_test/qwen3_tts/test_pipeline.py:4261`: "retired every whole-request slot". There are no whole-request slots now. In that test the two late requests also wait on the batch bound (`SV:3088`), not only on a row.
- The removed `test_codec_state_arena_never_hands_out_reserved_slots` guarded a property that no longer exists. Nothing else was lost with it: slot counts, release and retire are still covered at `tests/unit_test/qwen3_tts/test_incremental_codec.py:1683-1745`.

### 8. Rest of the PR diff

Nothing provably wrong found in `2a66cb083...4c41ba4c8`. Checked:

- `build_incremental_graph_runners` dropped `graph_enabled` (`enabled and async_decode and not deterministic`) for an early return on `codec_arena is None` (`SV:1184-1187`). Equivalent: the arena is None exactly when there is no incremental decoder or decode is synchronous (`SV:1123-1125`), and deterministic stateful decode forces synchronous (`SV:923-931`).
- The vocoder's slot count and the coordinator's cap come from the same function with the same plan (`config.py:130-134`, `pipeline/mp_runner.py:714-717`), so they cannot disagree.
- `decodes_whole_requests` True implies the arena and the window runner both exist (`SV:934-940`, `SV:1123`, `SV:1211-1236`).

## Checked and found correct

### 1. The premise: rows in use never exceed the in-flight count

- One row per stream. `codec_slot` is a single field (`SV:175`), taken once (`SV:1890-1903`). After any miss or failure `incremental_codec_fallback` is set (`SV:1893`, `SV:2827`, `SV:3583`) and `use_incremental_path` (`SV:1839-1845`) never takes a row again. No fallback then re-acquire.
- One row per whole request. `slot` is a single field (`SV:199`), taken only when None (`SV:3093-3099`). A request that fails in the plan is dropped from both lists (`SV:3108-3109`).
- A request is one kind or the other. The talker sends chunks only when `params["stream"]` is true (`request_builders.py:2125-2130`). The vocoder classifies the terminal payload on the same field (`BASE:296-307`). So a request with a stream state never reaches `run_non_streaming_batch`, and a non-streaming request never gets a stream state: `stream_done` for it creates none (`SV:3946-3954`).
- A streaming request with `stream_codec_output` off sends no chunks and holds no row. Its final decode uses per-stream state on the main thread (`SV:1715-1739`).
- Warmup takes no row. Capture decodes against the scratch row only (`GRAPH:338`, `GRAPH:567-573`).
- Session operations that bypass the cap are read and close (`pipeline/sessions.py:865`). The vocoder answers them without creating stream state (`BASE:240-283`).
- Completion order. A whole request returns its row before its result is emitted (`SV:3213-3220`). A stream emits its result and returns its row in the same critical section, with no blocking call between (`SV:3979-3981`, `SIMPLE:674-686`). The coordinator removes the request only when the result arrives (`COORD:996`).
- A finished stream's release is never deferred. By the time `finish_async_stream` runs, `finish_codec_slots` has already cleared the mark (`SV:3558-3560` runs before the commit at `SV:3337-3339` and `SV:3868-3870`).
- The only uncounted holders are aborted or failed requests whose row the vocoder has not yet returned. Abort is broadcast before the request leaves the count (`COORD:806-831`), delivered on the stage event loop straight into `scheduler.abort` (`pipeline/stage/runtime.py:2679-2687`), and deferred by at most one decode (`SV:1975-1978`).
- Replicas: each vocoder replica is sized to the whole cap (`config.py:148-149`).

### 2. Thread and stream ordering of every reuse

Every acquire, for both kinds, is on the initial worker thread with `decode_stream` current (`SV:2994`, `SV:3704-3714`): streams at `SV:1891` from `run_initial_batch`, whole requests at `SV:3096`. Follow-up workers never acquire, because a stream reaches them only after its first decode. Each acquire waits on the row's release event and then zeroes on that stream (`ARENA:164-169`).

| Release site | Thread, stream | Why the previous holder's GPU work is finished or ordered |
| --- | --- | --- |
| Stream finished (`SV:3981` to `SV:1981`) | initial or follow-up worker, its own stream | the cohort was resolved on the host before the commit (`SV:3516`), and the mark was cleared after it |
| Stream aborted, not in flight (`SV:1981`) | stage event loop or main thread, its current stream | not in flight means the last decode was resolved on the host. The mark spans plan to resolve (`SV:1946`, `SV:3558-3559`) |
| Stream aborted, in flight (`SV:1997-1998`) | the worker that ran the decode | called from the `finally` after `resolve_partial` (`SV:3514-3560`) |
| Stream launch failed (`SV:3475-3478`) | the worker | the launch path synchronized its stream (`SV:2420-2421`), or retired the rows and set the flag |
| Stream planning failed (`SV:2829`) | the planning worker | only zeroing was queued, on that same thread's stream, and the event is recorded after it |
| Whole request done, aborted or failed after a replay (`SV:3136`, `SV:3188`, `SV:3213`) | initial worker, `decode_stream` | after `resolve_partial` (`SV:3170-3179`), or after the launch path synchronized, or skipped under the flag |
| Whole request failed in the plan (`SV:3108` to `SV:3136`) | initial worker, `decode_stream` | only zeroing and `start_slot` were queued, on that stream, and the event is recorded after them |
| Whole request aborted between replays (`SV:3065`) | initial worker, `decode_stream` | its last replay was resolved at `SV:3170-3179` |

So every release happens with the row's GPU work complete on the host or queued on the releasing thread's own stream ahead of the event. The event and the free list entry are published together under the arena lock (`ARENA:183-198`), and the acquirer pops both under the same lock (`ARENA:153-163`).

Also checked:

- A stale stream state cannot touch a row after it is reused. `release_codec_slot` clears `codec_slot` first (`SV:1973`), and the late commit and fallback paths test the registry (`SV:3614`, `SV:3881`, `SV:3578-3580`).
- A slot cannot be on the free list while still marked in flight or deferred (`SV:1975-1981`, `SV:1993-1998`). Whole requests never use those sets and do not need them: only the initial worker releases their rows.
- The scratch row moved from index 2N to N. Every user reads `arena.scratch_slot` (`GRAPH:570`, `GRAPH:653`, `ARENA:307`, `ARENA:321`, `cutedsl/state_copy.py:164`, `cutedsl/state_copy.py:175`). Storage, rings and the index staging buffers are sized `scratch_slot + 1` (`ARENA:90`, `ARENA:104`, `ARENA:257-260`), and a cohort never has more rows than `num_slots`.
- Cost on the whole path: one event record per release and one event wait per acquire, on the initial worker. No added kernel launch. `zero_slot` was already there.
- No lock order problem. `arena.lock` is a leaf. `codec_lock` is dropped before `arena.release` (`SV:1975-1981`).

### 3. Failure paths

- `zero_slot` fault in `acquire`: handled inside the arena, both callers catch (`SV:2856`, `SV:3107`). Before the commit the stream path lost the row here. Caveat in finding 3.
- `start_slot` or `wait_codes_ready` fault: the request fails alone and its row goes back (`SV:3107-3109`, `SV:3136`). Caveat in finding 3.
- Launch or replay failure with the stream recovered: every request in the replay fails and returns its row (`SV:3180-3183`). Requests not in the replay continue.
- Launch or resolve failure that cannot be recovered: rows in the replay are retired, the flag is set, the releases are skipped, and the next plan fails every waiting request (`SV:3048-3056`). Requests still in `initial_queue` join `whole_requests` on a later pass and fail there.
- `cuda_decode_failed` set by another thread in the middle of a plan: an aborted request's release is skipped, an acquire either raises and is caught or returns a row whose launch then raises at `SV:2124-2127` and is caught at `SV:3180`. The next plan fails the rest. No request is left unanswered, apart from the window in finding 3.
- Retire: reachable only together with the flag (`SV:470-475`, `SV:2427-2430`). `release` returns silently for a retired slot (`ARENA:184-185`).
- Abort while queued, between replays, in flight: `SV:3064-3066`, `SV:3187-3189`. A request with no row releases nothing (`SV:3122`).
- Double release: none found. `release_whole_request_slot` clears `request.slot` (`SV:3127`), each request is in exactly one list, and each path through `run_whole_request_replay` handles a request once. On the stream side each slot is released by exactly one of `release_codec_slot` and `finish_codec_slots`, decided under `codec_lock`.
- Graphs torn down at run time (`GRAPH:669-678`, `GRAPH:708-709`): queued whole requests continue eagerly against their rows (`SV:2246-2249`), new ones take the batch path (`SV:2878-2882`). Older than this commit.

### 4. Liveness

- No head of line blocking. A request that cannot get a row goes to `waiting` and the next request seeds the replay (`SV:3083`, `SV:3099-3101`).
- Waiters keep their order and try first on the next pass (`SV:3118`, `SV:3209`).
- Every row holder advances. A request that cannot join one replay keeps its row and leads a later one.
- No deadlock among whole requests: each holds at most one row and takes it in one step.
- A request that cannot join a replay does not take a row (`SV:3090-3092`), so rows are not hoarded by requests that are not decoding.
- Default sizing: a whole request cannot wait outside the abort window (item 1), so it cannot wait forever.
- Below the cap: the wait ends when any holder releases. It is unbounded only if no holder ever ends.
- First chunks are not starved: every pass takes all queued first chunks before the plan (`SV:3018-3026`).
- Stop: the sentinel ends the loop even while requests wait (`SV:1517`, `SV:3035-3036`).

### 5. Style, on the commit's own lines

Read against `.claude/skills/code-review/coding-style.md`. The boolean parameter carries the `should_` prefix, the broad `except` reraises, comments are single lines signed `note (ratish)`, explicit `else: pass` matches the file. Nothing to report.

## Not verified

- The 2.74 MiB per slot figure.
- The busy loop on the box: the pass rate and its effect on first audio and RTF. Only the stand-in above was run.
- That stage factories are built in list order. Taken from the note at `config.py:94-95`.
- Any behavior on a real device. Every device claim here is from reading the calls, not from a run.

# Review of the fix commit `f89fc2f6d`

Commit: `f89fc2f6d` ("wait a poll for a stream's codec slot and retire a slot whose release cannot be recorded") on `8b6f6985a`. HEAD of the worktree is `f89fc2f6d` and the tree is clean, so line numbers in this section are that commit's. They are shifted from the section above (for example the plan moved from `SV:3043` to `SV:3061`).

Same rules: reading only, no pytest, no box, no edit in the repository. One local check was run: the class hierarchy of `torch.AcceleratorError` in the repository's `.venv`.

## Verdict on the fix

| First review finding | Status after `f89fc2f6d` |
| --- | --- |
| P1 busy loop | Closed. A plan that found no row is now followed by a blocking wait of at most 5 ms. |
| P2 below the cap the kinds take each other's rows | Closed as documentation: both warnings and the docs sentence name the wait. The wait is still invisible at run time (no log, no counter). I leave that as P4. |
| P3 `arena.release` raising on the initial worker | The raise is closed at its source for all four sites, for `acquire`, and for the stream paths. One liveness gap on the same failure path remains, below. |
| P4 docs, tests, stale docstring | Closed. The stats items were not touched and stay P4. |

New or remaining, ranked:

| Severity | Finding |
| --- | --- |
| P0 | None. |
| P1 | None. |
| P2 | None. |
| P3 | Slots can now be retired without `cuda_decode_failed` being set. Once every slot is retired, a whole request waits forever. Present at `8b6f6985a` too, where I missed it. |
| P4 | Five small items: the freed-row race now leans toward streams, retired slots appear in no stat, two comments disagree, three small test notes, a doubled warning. |

## P3

### R1. Every slot retired with `cuda_decode_failed` still False: whole requests wait forever

The plan fails waiting requests only on the flag (`SV:3066-3075`). Its note says "a retired slot never frees, so a request waiting for one fails here". That was true while a slot could only be retired together with the flag (`SV:472-477`, `SV:2435-2438`). `ARENA:185-192` now retires a slot on its own, and nothing sets the flag there.

Sequence. It needs a vocoder whose device or CUDA context has failed while requests still reach it, so the vocoder in its own process or on its own card (`--vocoder.process vocoder`), and non-streaming traffic only:

1. The vocoder's context enters a sticky error while no vocoder decode is in flight. No launch or resolve has run since, so `cuda_decode_failed` is False.
2. Whole request W1 arrives. Plan: `acquire` pops a row (`ARENA:165`), `wait_event` or `zero_slot` raises (`ARENA:168-172`), the handler calls `release`, the event record raises, the row is retired (`ARENA:185-192`), the original exception is reraised (`ARENA:176`).
3. `SV:3126-3128` catches it and fails W1 with an error. That is correct. No replay is built, so `launch_decode_plans` never runs and the flag stays False.
4. Each later whole request repeats steps 2 and 3 and retires one more row. After as many requests as there were free rows (128 on default sizing) the free list is empty and every row is retired.
5. The next whole request: `acquire` returns None (`ARENA:157-162`), it goes to `waiting`, `is_whole_request_waiting_for_slot` becomes True (`SV:3139`), and the worker polls every 5 ms forever. No row can ever free. The request gets no answer, stays counted at the coordinator until its client gives up, and so does every one after it.

What bounds it:

- Default single process layout: the talker shares the context, so a dead context stops requests reaching the vocoder. Not reachable there.
- Any streaming traffic: the first stream decode after the fault goes through `launch_async`, fails, fails to synchronize and sets the flag (`SV:2435`). The next plan then fails every waiting request.
- A whole request holding a row at the time of the fault: it joins a replay, the launch fails and sets the flag.

Compared with before the PR's shared rows (`4c41ba4c8`): there a failed `zero_slot` put the row back on the idle list, so every request got an error and none ever waited. So this is a regression of the shared design on this failure path, not of `f89fc2f6d`. At `8b6f6985a` the same sequence ended the same way, with the rows lost instead of retired and a busy loop instead of a poll. My first report noted the lost row under P3 and did not follow it to the waiting request. That was a miss.

Smallest fix: make the plan's first check cover the state the note describes. Either of:

- fail the waiting requests when no slot can ever free: `self.cuda_decode_failed or len(self.codec_arena.retired) == self.codec_arena.num_slots` at `SV:3066`. No new state. The first requests still fail one by one at `acquire`, each retiring a row, and the rest fail at the plan.
- have the arena remember that a release could not be recorded, and test that beside the flag. Every whole request then fails from the first failed record on, and no further rows are consumed.

A test for it: hold no rows, make every `acquire` fail with the failed device from `fail_device` (`tests/unit_test/qwen3_tts/test_incremental_codec.py:1160`), send `num_slots + 1` whole requests and assert that every one of them is answered.

## P4

### R2. A freed row now goes to a stream more often than to the waiting whole request

First chunks are planned before whole requests in every pass (`SV:3028-3045`), and the whole request now plans again only when the wait ends. A row freed while the worker sits in the timed `get` goes to a stream if a first chunk arrives before the poll expires, because that first chunk is batched and planned (`SV:1899`) before the plan at `SV:3045`. Under the busy loop the whole request re-planned within microseconds and almost always won.

Derived, not measured: the residual poll time after a release averages 2.5 ms, so at 20 first chunks per second (the load the docs measure at) a first chunk lands inside it about one release in twenty. Each freed row is a separate chance, so the whole request is delayed by another release, not starved. I found no interleaving in which it loses forever: a row freed while a first-chunk batch is decoding is taken by the plan at the end of that same pass.

This only exists when the arena is full, so below the cap or with no cap. No change needed. It is listed because the fix's message says no wakeup can be lost, which is true, and this is the cost that remains.

### R3. Retired slots appear in no stat

`describe()` (`ARENA:402-409`) reports `slots`, `active_slots`, `bytes_per_slot`, `total_bytes` and `exhausted`. `active_slots` excludes retired rows (`ARENA:145-149`). Before the fix a retirement always came with `cuda_decode_failed` and an error log. Now `release` can retire a row with only a warning line, and `model_info` shows a healthy arena with fewer usable rows than `slots`. A `retired` field in `describe()` would show it.

Still open from the first review, unchanged: a whole request waiting for a row appears in no log line and no counter.

### R4. Two comments in `run_initial_pass` disagree

`SV:3017`: "a waiting whole request means a replay is due, so only an idle worker waits for a batch." `SV:3020`: "no replay is due while every whole request waits for a slot that streams hold". The first is no longer true as written. In it, "waiting" meant queued.

Also `SV:3067`: "a retired slot never frees, so a request waiting for one fails here" holds only for slots retired with the flag (R1).

### R5. Test notes

- `test_qwen3_tts_whole_request_waiting_for_a_stream_slot_blocks_between_plans` (`tests/unit_test/qwen3_tts/test_pipeline.py:4301`) imports `QWEN3_TTS_WHOLE_REQUEST_SLOT_POLL_S` at module level (`test_pipeline.py:93`). On `8b6f6985a` the whole file fails at import, so the test cannot fail on its own assertion there without substituting the literal. With the literal it fails where it claims: the second pass never calls `get`, so `timeouts == []`.
- `test_codec_state_arena_never_loses_a_slot_whose_wait_fails` (`test_incremental_codec.py:1188`): the failing wait and the failing record raise the same message, so the test cannot tell that the original exception is the one that leaves `acquire`. It still fails on `8b6f6985a` for the right reason (`retired` is empty there).
- The fix says a first chunk, a whole request or the stop sentinel end the timed wait at once. That is correct by reading (below). No test drives any of the three through the timed branch, and none sets `cuda_decode_failed` while `is_whole_request_waiting_for_slot` is True.

### R6. A slot retired with the flag logs a second warning at release

With the guard gone, `fail_whole_request` calls `arena.release` for rows the failing launch already retired (`SV:2438`, then `SV:3201-3202`, `SV:3156`). On the dead device the record fails again and `ARENA:187-190` logs "could not record its release and is retired" with a traceback for each, after the error that already reported the failure. Noise only.

## The four questions

### 1. Does each fix close its finding

P1. Yes. After a plan with no replay and requests left, `SV:3139` sets the flag and the next pass takes `SV:3019-3023`, which blocks in `queue.Queue.get(timeout=0.005)` (`SV:2803`). That releases the interpreter. The note at `SV:3138` is right: a request reaches `waiting` either because it could not join a replay that exists (`SV:3109-3111`) or because `acquire` returned None (`SV:3118-3120`), and a request holding a row always seeds an empty replay, so `replay is None` with requests left means every one of them missed a row. No other pass can loop without work: with no requests left the worker takes the untimed wait at `SV:3018`.

P3. Yes for the raise. `ARENA:182-192` catches the failed record, so `SV:3084`, `SV:3156`, `SV:3208` and `SV:3233` cannot raise from it. At `SV:3233` a request whose last window resolved before the device failed now gets its result. `acquire` keeps the row accounted when the wait fails (`ARENA:167-176`) and reraises the original exception, because `release` no longer raises over it. The stream paths (`SV:1989`, `SV:2006`) get the same protection. What remains is R1.

P2. The two warnings (`SV:1396-1414`) and `tuning.md:148` now name the wait. `decodes_whole_requests` is final when the warning is built, because `log_cuda_graph_coverage` runs after `warmup_now` (`stages.py:572-576`).

P4. `tuning.md:148` "for every request the in-flight cap adds" is right for each vocoder replica at any replica count. `tuning.md:150` "the request's slot" and "any decode already running on the slot" cover both kinds (`SV:1974-1989`, `SV:3083-3085`, `SV:3207-3209`). No em dash and no semicolon in the changed sentences.

### 2. Does any fix introduce a new problem

`run_initial_pass` (`SV:3011-3046`):

| Concern | Result |
| --- | --- |
| Stop sentinel during the timed wait | Handled. `get` returns None at once, `SV:2806-2807` returns None, `SV:3026-3027` returns False. A sentinel that arrives after the timeout is caught by `take_queued_first_chunks` (`SV:3053-3054`). |
| First chunk during the timed wait | `get` returns at once, then the usual batching window of `initial_batch_wait_s` (`SV:2810-2825`), then `run_initial_batch`, then the remaining queued first chunks, then the plan. That is the idle worker's path, so a first chunk waits no longer than on an idle worker. It never waits for the poll to expire. |
| Whole request during the timed wait | Appended and the wait ends (`SV:2812-2814`). The plan runs in the same pass. |
| Stale flag True | The flag is written only by the plan (`SV:3074`, `SV:3139`) and read only at `SV:3019`, both on the initial worker. It is True only right after a plan that left requests with no row. A row freed since then is picked up at the next plan, at most one poll later. If all waiting requests are aborted the next plan empties the list and sets it False. It cannot be True with an empty list, and the branch is not reached with one. |
| Stale flag False | After a replay the flag is False and the next pass plans without waiting, which is the old behavior. |
| Flag ownership | One thread. Not reset in `on_serving_start`, like `whole_requests` itself. |

Release and retire for streams:

- `release_codec_slot` (`SV:1974-1989`) and `finish_codec_slots` (`SV:1995-2006`): on a failed device the row is retired and the call returns, so an abort completes and `finish_incremental_group`'s `finally` no longer raises over its result. This removes the older exposure I noted on the stream side.
- Deferred releases: unchanged. The retire happens inside `arena.release`, after the in-flight and deferred sets are settled under `codec_lock`.
- Double release detection: skipped only on the failed-record path, where `retire` is idempotent and removes the slot from the free list if it is there (`ARENA:218-224`). On a working device the check at `ARENA:200-203` is unchanged.
- What callers do next on a dead device: every caller continues as if the row was returned. For whole requests that means the error or result is emitted. For streams the next launch fails and sets the flag.
- With the guard gone, a release after `cuda_decode_failed` on a device that can still record puts the row back on the free list. Harmless: launches are refused at `SV:2132-2135` and the plan fails every whole request at `SV:3066`.
- Risk I could not rule out from the code: on any backend where an event record can raise on a working device, a release now costs a row for the life of the process and says so only in a warning. I know of no such case on CUDA.

Exception type. `except RuntimeError` is right. `torch.cuda.Event.record` raises through `C10_CUDA_CHECK`, which surfaces as `torch.AcceleratorError`, and that class derives from `RuntimeError`. Checked locally: `torch.AcceleratorError.__mro__` is `(AcceleratorError, RuntimeError, Exception, BaseException, object)` on torch 2.13.0 in the repository's `.venv`. The pin is `torch==2.14.1` (`pyproject.toml:23`), which is not installed here, so that exact build is unchecked. The file's other device catches use the same type (`commit_decoded_incremental`, `reserve_slot`, `GRAPH` `synchronize_device`).

### 3. Remaining liveness gaps

- `cuda_decode_failed` set while whole requests wait: noticed within one poll. The timed `get` expires, the pass reaches the plan, `SV:3066-3075` fails them.
- A stream release while whole requests wait: noticed within one poll, or sooner if anything arrives on `initial_queue`.
- Stop while whole requests wait: the sentinel ends the wait at once.
- Every slot retired without the flag: not noticed. R1.
- A holder that never releases (for example a stream whose worker thread died): the wait is unbounded, as before. The fix makes worker death less likely, since `arena.release` was one of the calls that could kill a worker.

### 4. Do the new tests fail on `8b6f6985a`

By reading, not by running.

| Test | On `8b6f6985a` |
| --- | --- |
| `test_codec_state_arena_retires_a_slot_whose_release_cannot_be_recorded` | Fails. `release` raises the `RuntimeError` from `record`. |
| `test_codec_state_arena_never_loses_a_slot_whose_wait_fails` | Fails. `acquire` raises as expected, then `arena.retired == {0}` is false because the wait was outside the `try`. |
| `test_qwen3_tts_whole_request_waiting_for_a_stream_slot_blocks_between_plans` | Fails, but at import of the new constant. With the literal in its place it fails on `timeouts == []`. See R5. |
| `test_qwen3_tts_vocoder_warns_when_streams_can_outnumber_codec_slots`, new case `(65, True, ...)` | Fails. The text is absent. |
| `test_qwen3_tts_whole_request_whose_codes_never_land_fails_only_itself` | Passes on both. It is the coverage test for the plan's release boundary, and it fails if `fail_whole_request` stops returning the row (`run_whole_requests` asserts no active slot). |
| `test_qwen3_tts_whole_request_waits_for_the_slot_a_stream_holds`, changed | Passes on both. It now frees through `release_codec_slot`, the path that is not deferred. The deferred path through `finish_codec_slots` is still not driven with a whole request waiting. |

## Not verified in this section

- `torch.AcceleratorError` on the pinned torch 2.14.1. Checked on 2.13.0 only.
- That `zero_slot` and `wait_event` raise on a failed CUDA context, which R1 depends on. Read from the calls, not run.
- The 5 % figure in R2. Derived from the poll length and the docs' request rate.
- No test was run on either commit.
