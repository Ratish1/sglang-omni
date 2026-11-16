## Problem

The streaming vocoder keeps each stream's incremental codec state in an arena of 64 slots, the engine's running bound.

A stream takes its slot at its first decode and returns it at its final decode, after its engine slot is already free, so under load more than 64 streams are live in the vocoder.

Each stream past the 64th took the left-context decoder, which costs more per chunk than the incremental one. The vocoder fell further behind, more streams stayed live, and more took the fallback.

Past the knee the server collapses: at 60 req/s of Poisson arrivals, 1.7B CustomVoice, under 2 % of requests complete, almost every one with playback underruns, and the rest get HTTP 503 at the 128 in-flight cap.

## Change

With async decoding, a stream that finds the arena full now waits for a slot and keeps the incremental decoder.

- `build_incremental_plan` queues the stream (`is_waiting_for_codec_slot`) instead of setting `incremental_codec_fallback`; a new stream queues behind streams already waiting, and a stream leaves the queue where it acquires its slot.
- A slot returned to the arena schedules the first decode of the oldest waiting streams, one per slot the arena can hand out (`Qwen3TTSCodecStateArena.free_slots`).
- A waiting stream is not rescheduled by its new chunks. One whose generation ends keeps the async finish instead of a synchronous whole decode on the message thread.
- A stream whose incremental planning fails leaves the queue with its fallback, and a stream that ends leaves it with its other resources.
- While no stream waits, none of this takes a lock. The synchronous decoder keeps the left-context fallback. The codec stats gain `slot_waits`.

## Results

One H100 pair, main `ea8216220` against this branch, both servers at once with one per card and each pinned to its card's half of the socket, the cards swapped between seeds.

The load is Nari Labs' open client unmodified (`tts_bench` `a2d65e1`): 1.7B CustomVoice, Ryan, English, the SeedTTS EN sentences, Poisson arrivals, 30 s warm-up and 120 s measured per rate.

| req/s | seed | completed, main → branch | first byte p50, ms | audible first audio p50, ms | requests with underruns |
|---|---|---|---|---|---|
| 60 | 0 | 1.1 % → 97.3 % | 57 → 24 | 351 → 103 | 2,030 → 31 |
| 60 | 1 | 1.4 % → 100 % | 55 → 21 | 309 → 98 | 2,298 → 0 |
| 60 | 2 | 1.7 % → 100 % | 52 → 20 | 305 → 97 | 2,317 → 0 |
| 70 | 0 | 84.7 % → 99.5 % | 1,005 → 279 | 1,083 → 358 | 21 → 2 |
| 70 | 1 | 84.3 % → 97.9 % | 975 → 371 | 1,053 → 450 | 39 → 2 |
| 70 | 2 | 86.5 % → 99.2 % | 990 → 239 | 1,068 → 319 | 2 → 0 |
| 80 | 0 | 22.7 % → 83.2 % | 1,022 → 833 | 1,101 → 911 | 1,534 → 1 |
| 80 | 1 | 79.2 % → 93.0 % | 975 → 785 | 1,053 → 863 | 4 → 1 |
| 80 | 2 | 77.1 % → 90.6 % | 992 → 816 | 1,071 → 893 | 6 → 0 |

Main's first byte p50 at 60 req/s covers only the requests it completed. The branch's remaining failures are HTTP 503 at the 128 in-flight admission cap; no stream takes the left-context decoder (`left_context_fallbacks` 0, about 12,000 `slot_waits` per run).

Below the knee the arena never fills, so no stream waits (`slot_waits` 0, no failed acquire) and every stream takes main's path. SeedTTS EN, 1.7B Base, the full 1,088 prompts, unseeded, on the same pair of cards with each server pinned to its card's half of the socket. c16 and c64 are eight warm readings each (four passes on the same servers in both card orders); c1 is one pass in both card orders.

| cell | RTF, main → branch | first audio mean, ms | first audio p95, ms | req/s |
|---|---|---|---|---|
| streaming c1 | 0.0605 → 0.0605 | 18.8 → 18.9 | 22.9 → 22.8 | 3.98 → 3.99 |
| streaming c16 | 0.0972 → 0.0968 | 30.4 → 30.2 | 43.2 → 43.2 | 39.7 → 39.8 |
| streaming c64 | 0.2267 → 0.2281 | 55.3 → 54.9 | 179.7 → 165.5 | 67.0 → 66.6 |
| whole c1 | 0.0595 → 0.0595 | | | 4.06 → 4.07 |
| whole c16 | 0.0900 → 0.0898 | | | 42.8 → 42.8 |

Main against main in the same way on the same node reads streaming c16 RTF -0.1 % and streaming c64 RTF +0.3 % (single passes -1.4 % to +1.8 %).

| streaming c16, full corpus | WER | speaker similarity | requests above 50 % WER |
|---|---|---|---|
| main | 1.06 % | 71.47 | 1 |
| branch | 1.06 % | 71.54 | 0 |

## Tests

- `test_qwen3_tts_full_codec_arena_holds_a_stream_until_a_slot_frees`
- `test_qwen3_tts_new_stream_waits_behind_streams_already_waiting_for_a_slot`
- `test_qwen3_tts_waiting_stream_leaves_the_queue_when_incremental_planning_fails`
- `test_qwen3_tts_waiting_stream_keeps_async_finish_when_generation_ends`
- `test_arena_free_slots_counts_only_slots_acquire_can_hand_out`
