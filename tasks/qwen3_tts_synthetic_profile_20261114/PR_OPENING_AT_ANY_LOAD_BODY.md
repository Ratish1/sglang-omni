## Problem

Streamed CustomVoice requests for Ryan in English open with a silent codec frame. On 1.7B it is 80 ms of digital silence in every SeedTTS EN request. Since #1901 the vocoder withholds that frame's audio and decodes one more frame into the first chunk, so the client's first audio is speech and its playback lead stays the same.

Two paths still sent the silent frame first.

- The vocoder skipped the trim once more than 24 streams were live (`suppress_bootstrap_max_streams`). #1901 set the limit from a 20 req/s run in September near that server's capacity, where the extra first-chunk frame cost about 10 points of completion. On Pipeshift's benchmark today (Nari Labs' client, Ryan, `language: "English"`) the audible first audio p95 at 40 req/s is 100 ms, 80 ms of it leading silence.
- A request's result payload reaches the vocoder after its codec chunks, and the payload reset the first-chunk width from the request parameters. When it landed before the stream's first decode ran, the compensating frame was lost. At the default width the first chunk held the silent frame alone, so it was emitted. With a larger requested width the trimmed first chunk came out one frame short. This happens at any load to requests that finish generating before their first decode starts, most often short replies queued behind other first decodes.

## Change

- The vocoder withholds the silent frame of every eligible stream at any number of live streams. `suppress_bootstrap_max_streams` is removed from the vocoder factory and the scheduler.
- The request payload sets a stream's first-chunk width only while no codec frame has arrived. Once one has, the first chunk's metadata owns the width. The engine writes the request's `initial_codec_chunk_frames` and the eligibility flag there, from the same request parameters, and the scheduler's contract already holds latched values fixed.
- Eligibility, the runtime silence check, the per-request opt-out (`"suppress_bootstrap_silence": false`) and the deployment switch (`--vocoder.factory.suppress_bootstrap_silence false`) are unchanged. A stream still opens with its silent frame when it generates one frame in total, when its first chunk is 0 frames or the full stride, or when it opts out.

A deployment that sets `--vocoder.factory.suppress_bootstrap_max_streams`, or the same YAML key, now stops at startup with the factory's unknown parameter error naming the key. Delete it. `--vocoder.factory.suppress_bootstrap_silence false` keeps the leading silence at every load. No setting brings back a stream-count limit.

## Results

Every request eligible: SeedTTS EN, all 1,088 texts, Ryan, `language: "English"`, streamed, closed loop on one H200, main `dfb6b43b3` against this change, both orders, two passes per server, mean of four passes. Above 24 live streams main sends the silent frame and this change withholds it.

| | c32 | c64 |
|---|---|---|
| Requests per second | 48.9 → 49.5 | 60.0 → 61.2 |
| RTF | 0.1366 → 0.1389 | 0.2202 → 0.2202 |
| First byte median, ms | 25.1 → 33.8 | 29.6 → 42.3 |
| Audible first audio median, ms | 102.9 → 34.7 | 107.4 → 43.2 |
| Underrun requests, four passes | 7 → 5 | 8 → 8 |

- Audible first audio is timed as a listener hears it: from the request's send, played from the first byte, to the first point where DC-removed RMS over 20 ms windows reaches -45 dBFS for two windows in a row (Pipeshift's rule).
- The first byte waits one more talker step. On main that byte was the silent frame.
- The c32 RTF is lower in one order and higher in the other. Underruns come only in the first pass after each server starts, on both arms.

Short replies, where the payload often lands before the first decode: 48 concurrent requests with `max_new_tokens: 3`, three rounds, one H200.

| | Stream limit removed, payload reset kept | This change |
|---|---|---|
| Silent openings | 16 / 16 / 16 | 0 / 0 / 0 |
| Requests finished before their first decode | 16 / 16 / 16 | 16 / 16 / 16 |
| First decode of those requests, frames | 1 | 2 |

- With `initial_codec_chunk_frames: 4` and `max_new_tokens: 8`, the requests finished before their first decode decoded 4 frames first without this change and 5 with it.
- With `"suppress_bootstrap_silence": false` all 48 open with the silent frame and every first decode is 1 frame.

Pipeshift's benchmark on one H100, main `bac76771a` against the stream-limit removal, seeds 0 to 2, median of seeds, ms. Every request succeeded, with no underrun and no stall. Its sentences finish well after their first decode (leading silence p99 20 to 40 ms with the change), so the payload change does not move them.

| req/s | 20 | 30 | 40 |
|---|---|---|---|
| Audible first audio p95 | 32.4 → 31.9 | 37.0 → 34.1 | 100.1 → 37.2 |
| Audible first audio p50 | 21.3 → 21.2 | 22.6 → 22.6 | 25.4 → 24.1 |
| First byte p50 | 20.9 → 20.9 | 22.1 → 22.3 | 22.4 → 23.7 |
| First byte p95 | 26.3 → 26.5 | 29.2 → 29.1 | 30.8 → 32.1 |
| Leading silence p95 | 10 → 10 | 10 → 10 | 80 → 10 |

- At 40 req/s the three seeds' audible p95 are 36.8 to 37.6 ms with the change.

Past the H200's knee, near 50 req/s, two runs of main differ by up to 16 points of completion and 3x in underrun requests at 50, 60 and 70 req/s, seeds 0 and 1. Main against this change stays inside that spread in both directions, and leading silence p95 goes from 90 to 10 ms at every rate.

Under vocoder load the second chunk reaches the client closer to the end of the first: on H100 at 70 and 80 req/s, margin p50 5 to 7 ms and p1 6 to 9 ms. The talker's first frame after a prefill comes about 10 ms after the prefill frame, and later frames about 17 ms apart. With the silent frame played, the second chunk (frames 2 and 3) waits for one fast gap and one slow one. With it withheld, the second chunk (frames 3 and 4) waits for two slow gaps. The vocoder's own queue and decode times are equal on both arms.

Both CustomVoice sizes offer Ryan. On 0.6B the first frame is not digital silence: it passes the vocoder's silence guard in 1,038 of 1,088 SeedTTS EN requests, and 0.6B opens with about 430 ms below -45 dBFS. One request at a time over 100 texts, the trim plays speech 76 ms sooner (audible first audio p50 448 → 372 ms) for 4 ms more first-byte wait. The other 4.6 % keep their first frame and pay only the 4 ms.

SeedTTS EN sends no language, so no request is eligible, and the payload change runs on every streamed request. CustomVoice, all 1,088 prompts, main `dfb6b43b3` against this change on one H200, both orders, streaming at 1, 16 and 64 concurrent requests and whole at 1 and 16:

- Streaming rows read level or better: first audio mean at 1 request 16.3 → 15.4 ms, at 64 requests 43.8 → 40.9 ms, RTF at 64 requests -1.8 %.
- Whole requests never reach the changed code. Whole throughput at 16 requests reads 0.5 and 0.6 % lower in the two orders, with latency mixed in sign.
- WER 1.66 / 2.48 % → 1.93 / 2.20 %, below 50 % WER 1.45 / 1.42 % → 1.31 / 1.38 %.
- Seeded requests, with and without a language, streamed and whole, are byte-identical to main.

## Tests

- `test_qwen3_tts_bootstrap_suppression_trims_the_first_chunk_at_any_stream_count` sends an eligible stream through the chunk path with 64 other streams live, one codec frame per chunk, runs its first decode from the queue and checks that the first message holds only the second frame's audio. A second case delivers stream done and the payload before the first decode runs. Main fails both cases and the stream-limit removal alone fails the second, each with the silent frame first.
- The test that asserted the trim stops above the limit is removed with the limit.
