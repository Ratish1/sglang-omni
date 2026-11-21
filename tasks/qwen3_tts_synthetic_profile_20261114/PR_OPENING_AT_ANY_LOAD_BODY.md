## Problem

Streamed 1.7B CustomVoice requests for Ryan in English open with one silent codec frame, 80 ms of digital silence. Since #1901 the vocoder withholds that frame's audio and decodes one more frame into the first chunk, so the client's first audio is speech and its playback lead stays the same.

The vocoder skipped the trim once more than 24 streams were live (`suppress_bootstrap_max_streams`). Past 24 streams every eligible request played its 80 ms of silence first, so what a client heard first depended on how loaded the server was.

#1901 set the limit from a 20 req/s run in September, when the server served about 20 req/s at most: there the extra first-chunk frame cost about 10 points of completion and raised underruns from 0.34 % to 10.8 %. Today the server serves about 75 req/s of the same load, and the limit's cost shows instead: on Pipeshift's benchmark (Nari Labs' client, Ryan, `language: "English"`) the audible first audio p95 at 40 req/s is 100 ms, 80 ms of it leading silence.

## Change

- The vocoder withholds the silent frame of every eligible stream, at any number of live streams.
- `suppress_bootstrap_max_streams` is removed from the vocoder factory and the scheduler. The per-request opt-out (`"suppress_bootstrap_silence": false`) and the deployment switch (`--vocoder.factory.suppress_bootstrap_silence false`) stay.
- Eligibility, the runtime silence check and the first-chunk compensation are unchanged.

## Results

Pipeshift's benchmark on one H100, main `bac76771a` against this change, seeds 0 to 2, median of seeds, ms. Every request succeeded, with no underrun and no stall.

| req/s | 20 | 30 | 40 |
|---|---|---|---|
| Audible first audio p95 | 32.4 → 31.9 | 37.0 → 34.1 | 100.1 → 37.2 |
| Audible first audio p50 | 21.3 → 21.2 | 22.6 → 22.6 | 25.4 → 24.1 |
| First byte p50 | 20.9 → 20.9 | 22.1 → 22.3 | 22.4 → 23.7 |
| First byte p95 | 26.3 → 26.5 | 29.2 → 29.1 | 30.8 → 32.1 |
| Leading silence p95 | 10 → 10 | 10 → 10 | 80 → 10 |

- At 40 req/s the three seeds' audible p95 are 36.8 to 37.6 ms with the change.
- The first byte comes one frame later above 24 streams (+1.3 ms p50 at 40 req/s). That byte was the silent frame.

Past capacity, on #4077's arena sizing (each arm alone on one H100, seeds 0 and 1, 120 s per rate):

| req/s | 60 | 70 | 80 |
|---|---|---|---|
| Audible first audio p50, ms | 97 / 97 → 29.5 / 28.9 | 101 / 100 → 70 / 36 | 829 / 796 → 825 / 789 |
| Completed, % | 100 / 100 → 100 / 100 | 100 / 100 → 100 / 100 | 94.0 / 95.3 → 90.8 / 92.9 |

- Leading silence p95 goes from 90 to 10 ms, and underruns do not increase.
- The 80 req/s completion is past the knee, where four readings of the same pair moved by -305, -227, +535 and +24 requests.
- Under that load the second chunk reaches the client 5 to 7 ms closer to the end of the first (margin p50 at 80 req/s 56.4 → 50.2 ms, p1 40.7 → 32.3 ms). The talker's first frame after a prefill comes about 10 ms after the prefill frame, and every later frame about 17 ms apart. With the silent frame played, the second chunk (frames 2 and 3) waits for one fast gap and one slow one. With it withheld, the second chunk (frames 3 and 4) waits for two slow gaps. The vocoder's own queue and decode times are equal on both arms.

SeedTTS EN sends no language, so no request is eligible and the changed lines never run. CustomVoice over all 1,088 prompts on main `eabad7188`, each arm alone on one H100 in both orders (streaming at 1, 16 and 64 concurrent requests, whole at 1 and 16): every row reads within main's own run-to-run spread, and WER is 1.98 / 2.39 % on main against 1.74 / 2.44 % with the change.

## Tests

- `test_qwen3_tts_bootstrap_suppression_trims_the_first_chunk_at_any_stream_count` latches a stream with 64 others live and decodes its first chunk: it takes one more frame than an ineligible stream and holds only the second and third frames' audio. It fails on main, where the stream keeps the silent frame.
- The test that asserted the trim stops above the limit is removed with the limit.
