## Problem

The streaming vocoder keeps each stream's incremental codec state in an arena of 64 slots, the engine's running bound.

A stream takes its slot at its first decode and returns it at its final decode, after its engine slot is already free, so under load more than 64 streams are live in the vocoder.

Each stream past the 64th took the left-context decoder, which costs more per chunk than the incremental one. The vocoder fell further behind, more streams stayed live, and more took the fallback.

Past the knee the server collapses: at 60 req/s of Poisson arrivals, 1.7B CustomVoice, under 1 % of requests complete, each with playback underruns, and the rest get HTTP 503 at the 128 in-flight cap.

## Change

With async decoding, a stream that finds the arena full now waits for a slot and keeps the incremental decoder.

- `build_incremental_plan` marks the stream `is_waiting_for_codec_slot` instead of setting `incremental_codec_fallback`; a new stream queues behind streams already waiting.
- `plan_stream_decode` keeps the waiting streams oldest first. A slot returned to the arena schedules the first decode of the oldest waiting streams, one per free slot.
- A waiting stream is not rescheduled by its new chunks. One whose generation ends keeps the async finish instead of a synchronous whole decode on the message thread.
- While no stream waits, none of this takes a lock.
- The synchronous decoder keeps the left-context fallback. The codec stats gain `slot_waits`.

## Results

One H100 pair, main `ea8216220` against this branch, both servers at once with one per card and each pinned to its card's half of the socket, the cards swapped between seeds.

The load is Nari Labs' open client unmodified (`tts_bench` `a2d65e1`): 1.7B CustomVoice, Ryan, English, the SeedTTS EN sentences, Poisson arrivals, 30 s warm-up and 120 s measured per rate.

| req/s | seed | completed, main → branch | first byte p50, ms | requests with underruns |
|---|---|---|---|---|
| 60 | 0 | 0.8 % → 100 % | 58 → 21 | 2,153 → 0 |
| 60 | 1 | 0.8 % → 100 % | 56 → 24 | 2,138 → 1 |
| 70 | 0 | 81.1 % → 98.7 % | 1,062 → 525 | 18 → 0 |
| 70 | 1 | 86.7 % → 96.6 % | 986 → 755 | 10 → 0 |
| 80 | 0 | 70.4 % → 86.5 % | 1,104 → 874 | 15 → 0 |
| 80 | 1 | 75.7 % → 85.0 % | 1,025 → 908 | 15 → 2 |

On main the server logs `incremental Codec state arena is full` about 2,000 to 3,000 times a minute at 60 req/s; on this branch the fallback no longer occurs and `slot_waits` counts the streams that waited.

Below the knee the arena never fills (0 waits at SeedTTS EN c1, c16 and c64), so served streams take main's path.

Pending in this draft: the same-time SeedTTS EN pair at c1, c16 and c64 streaming and c1 and c16 whole requests with WER and SIM, a main against main band, and seed 2 of the table above.

## Tests

- `test_qwen3_tts_full_codec_arena_holds_a_stream_until_a_slot_frees`
- `test_qwen3_tts_new_stream_waits_behind_streams_already_waiting_for_a_slot`
- `test_qwen3_tts_waiting_stream_keeps_async_finish_when_generation_ends`
