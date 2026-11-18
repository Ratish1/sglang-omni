## Problem

The streaming vocoder keeps each stream's incremental codec state in an arena of 64 slots, the engine's running bound.

A stream takes its slot at its first decode and returns it at its final decode, after its engine slot is already free, so under load more than 64 streams are live in the vocoder.

Each stream past the 64th took the left-context decoder, which costs more per chunk than the incremental one. The vocoder fell further behind, more streams stayed live, and more took the fallback.

Past the knee the server collapses: at 60 req/s of Poisson arrivals, 1.7B CustomVoice, under 2 % of requests complete, almost every one with playback underruns, and the rest get HTTP 503 at the 128 in-flight cap.

## Change

With async decoding, a stream that finds the arena full waits for a slot and keeps the incremental decoder.

- `build_incremental_plan` parks the stream in `codec_slot_waiting_queue`, oldest first, instead of setting `incremental_codec_fallback`. A new stream parks behind streams already waiting.
- The initial worker is the only place a parked stream gets a slot. `admit_waiting_streams` runs at the start and the end of every initial batch, claims a row for the oldest waiting streams while the arena has one, and queues their first decodes, so picking a stream and taking its row are one step on one thread.
- A returned row only wakes that worker: `wake_codec_slot_admission` queues the oldest waiter's first decode unless one is pending. A stream that leaves before its turn holds no row and takes no turn.
- A stream waits only while a live owner can return a row. Once CUDA decode is disabled, each waiting stream's error and abort commit under the lock that takes it off the queue, so a stream whose generation ends meanwhile is never finished synchronously, and a new stream takes the fallback, whose launch fails it.
- A fault while the arena zeroes a row retires the row instead of dropping it from the count. Once every stream row is retired (`usable_slots()` reaches 0, which never grows back), the waiting streams and every later stream take the left-context decoder, which still runs.
- A waiting stream is not rescheduled by its new chunks, and one whose generation ends keeps the async finish.
- The synchronous decoder keeps the left-context fallback. The codec stats gain `slot_waits`.

## Results

One H100 pair, main `ea8216220` against this branch, both servers at once with one per card and each pinned to its card's half of the socket, the cards swapped between seeds.

The load is Nari Labs' open client unmodified (`tts_bench` `a2d65e1`): 1.7B CustomVoice, Ryan, English, the SeedTTS EN sentences, Poisson arrivals, 30 s warm-up and 120 s measured per rate.

| req/s | seed | completed, main → branch | first byte p50, ms | first byte p95, ms | requests with underruns |
|---|---|---|---|---|---|
| 60 | 0 | 1.8 % → 100 % | 54 → 19 | 104 → 28 | 2,334 → 0 |
| 60 | 1 | 98.0 % → 100 % | 20 → 20 | 31 → 29 | 157 → 0 |
| 70 | 0 | 90.3 % → 100 % | 918 → 24 | 1,051 → 230 | 1 → 1 |
| 70 | 1 | 89.7 % → 100 % | 943 → 24 | 1,065 → 325 | 6 → 6 |
| 80 | 0 | 79.3 % → 92.8 % | 965 → 780 | 1,069 → 881 | 3 → 0 |
| 80 | 1 | 78.9 % → 92.6 % | 981 → 798 | 1,095 → 894 | 10 → 0 |

Main's latencies at 60 req/s seed 0 cover only the requests it completed. The branch's failures at 80 req/s are HTTP 503 at the 128 in-flight cap; no stream takes the left-context decoder (`left_context_fallbacks` 0, about 9,000 `slot_waits` per run).

Few slots, every way a client uses a stream: one slot, 1.7B Base with a reference clip and 1.7B CustomVoice, six rounds each of 16 streams, 2 whole requests, 6 clients that give up before their first audio (0.05 to 0.55 s deadlines, while their streams wait) and 3 that drop after the first chunk, then one stream on the idle server. Every stream, whole request and idle follow-up returned full audio in every round, 114 and 112 streams waited, none fell back, and no slot stayed active.

Below the knee no stream waits (`slot_waits` 0) and every stream takes main's path. SeedTTS EN, 1.7B Base, the full 1,088 prompts, unseeded, six warm passes each (three on the same servers in both card orders):

| cell | RTF, main → branch | first audio mean, ms | first audio median, ms | req/s |
|---|---|---|---|---|
| streaming c16 | 0.0955 → 0.0955 | 29.2 → 29.2 | 28.5 → 28.5 | 40.3 → 40.3 |
| streaming c64 | 0.2216 → 0.2223 | 54.2 → 54.1 | 41.4 → 40.9 | 68.4 → 68.4 |

Main against main in the same way on the same node reads streaming c64 RTF +0.3 % (single passes -1.4 % to +1.8 %).

## Tests

- `test_qwen3_tts_full_codec_arena_holds_a_stream_until_a_slot_frees`: a stream that finds the arena full and ends while parked streams every frame and one result once the slot frees, and the row is reused.
- `test_qwen3_tts_new_stream_waits_behind_streams_already_waiting_for_a_slot`
- `test_qwen3_tts_slot_freed_while_a_stream_parks_goes_to_that_stream`: the slot returns between the failed acquire and the park.
- `test_qwen3_tts_a_cancelled_waiter_hands_its_turn_to_the_next`
- `test_qwen3_tts_waiting_streams_fail_once_the_decoder_is_disabled`: two waiters, the second one's generation ends as soon as the lock is released; each gets one error and no synchronous decode runs.
- `test_qwen3_tts_streams_take_the_fallback_once_the_last_slot_faults`: the only row faults on reset with a stream behind the faulted one; the faulted stream, the one behind it and a later stream each stream every sample and one result.
- `test_arena_withdraws_a_slot_whose_zeroing_faults`: a held row keeps `usable_slots()` at 1 while `acquire()` returns `None`; the last row faulting brings it to 0.
