## Problem

Qwen3-TTS admits at most `running + queued` requests (64 + 64). A request past that cap gets HTTP 503 at once, even when a slot frees a moment later, so a burst is refused while the server would have served it.

Two more refusals sit inside the cap:

- The engine applies SGLang's own `max_queued_requests`, which counts every request not yet running. A burst that reaches an idle engine therefore exceeds 64 queued while running slots are still free, and the engine refuses requests the coordinator admitted.
- On the slot-fix head, 200 streams sent at once to an idle server lose 72 per round: 65 at the coordinator, 7 at the engine.

## Change

Past the in-flight cap a request waits in the coordinator, before preprocessing, and the oldest waiting request starts when an in-flight request finishes.

- `Coordinator.acquire_admission` keeps waiters in a deque of futures, oldest first. `forget_request` is the one place a request leaves the in-flight set, and it grants the freed slot to the oldest live waiter. A waiter cancelled before or after its grant never holds a slot.
- `--admission.max_waiting_requests` bounds the waiting requests; past it a request gets HTTP 503. Qwen3-TTS defaults to 256, every other model to 0 (refuse at the cap, as before).
- The Qwen3-TTS engine queue has no bound of its own by default, since the coordinator's cap already holds it. An operator's `max_queued_requests` still bounds both.
- The HTTP disconnect watch awaits `request.receive()` until `http.disconnect` instead of polling `request.is_disconnected()` every 50 ms. One definition serves every route that watches.

### Why the disconnect watch changed

With requests waiting, a 100 req/s overload pair profiled the API process: its main thread ran a full core, `/health` took 493 ms at p90, and 66 % of the interpreter lock samples were the poll. Each `is_disconnected()` opens an anyio cancel scope and cancels it; anyio's `CancelScope.cancel` formats the task's repr into its reason, which walks the coroutine and its callbacks, and a process-wide audit hook runs on each access. A blocking `receive()` costs nothing until the client leaves. uvicorn (h11 and httptools), starlette's TestClient and httpx's ASGITransport all block in `receive()` once the body is read and return `http.disconnect` when the connection closes, and every caller reads its body first.

### Why the waiting queue is bounded

A waiting request keeps about 210 Python objects in the API process (measured with a burst of 2,000 streams), most of them framework: closure cells, about 30 coroutines of the uvicorn, starlette, FastAPI and pipeline await chain, three tasks. CPython's full collection walks every one of them while every running stream waits for its next chunk. Without a bound, at 100 req/s the API process's full collections grew from 66 to about 460 ms as the queue filled, and running streams underran in clusters (128 to 145 requests against 4 to 21). At about 0.12 ms per waiting request on an H100 host, 256 keeps the added pause under half of a stream's 80 ms first chunk.

Waiting absorbs bursts but adds no throughput. Under sustained overload completion is the same as refusing at the cap, and each served request also waits for the queue ahead of it.

## Results

One H100 pair, the codec slot fix (#4077, `dade67035`) against this branch, both servers at once, one per card, each pinned to its card's half of the socket, the cards swapped between seeds. Nari Labs' open client unmodified (`tts_bench` `a2d65e1`): 1.7B CustomVoice, Ryan, English, the SeedTTS EN sentences, Poisson arrivals, 30 s warm-up and 180 s measured.

| req/s | seed | completed | requests with underruns | largest stall, ms | first byte p50 |
|---|---|---|---|---|---|
| 100 | 0 | 70.3 % → 69.7 % | 2 → 4 | 72 → 4.5 | 0.89 → 4.55 s |
| 80 | 0 | 86.6 % → 85.8 % | 4 → 3 | 30 → 26 | 0.87 → 4.61 s |

Bursts on an idle server, default settings: 200 and 384 streams at once all complete with none refused (384 is the in-flight cap plus the bound). A mixed round of 216 requests (128 streams, 16 whole requests, 48 clients that give up before their first audio while they wait, 24 that drop after the first chunk), three rounds per model on 1.7B Base and CustomVoice: every request that should finish does, none refused (the slot fix refuses 10 to 38 per round).

## Tests

- `test_live_server_aborts_a_speech_stream_whose_client_leaves_before_first_audio`: a real uvicorn server and socket; the client closes before first audio and the stream is aborted.
- The disconnect fakes implement `receive()`, so every existing disconnect test runs the new watch.
- `test_coordinator_holds_submits_past_the_in_flight_cap_and_admits_them_oldest_first`, `test_coordinator_rejects_submits_past_the_waiting_bound`, `test_waiting_submit_fails_when_the_coordinator_fails`.
- `test_cancelled_waiting_submit_passes_its_turn_to_the_next`: cancelled before its grant.
- `test_waiting_submit_cancelled_after_its_grant_passes_the_slot_on`: granted, then cancelled before it resumes; without the release the next waiter is never admitted.
- `test_only_the_coordinator_bounds_admission_by_default`: the cap is 128, the bound 256, and the engine has no queued default.
