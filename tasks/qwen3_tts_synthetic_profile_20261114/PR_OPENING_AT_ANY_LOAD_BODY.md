## Problem

Streamed CustomVoice requests for Ryan in English open with a silent codec frame. Since #1901 the vocoder withholds that frame's audio and decodes one more frame into the first chunk, so the client's first audio is speech and its playback lead stays the same. The first codec chunk carries the eligibility flag, and `latch_stream_contract` adds the extra frame to the stream's first-chunk width when that chunk arrives.

A request's result payload reaches the vocoder after its codec chunks, and stream done usually lands before it. The payload branch of `latch_stream_contract` then set the width again from the request parameters, which dropped the extra frame. When that happened before the stream's first decode ran, the first decode was planned on the reset width.

- At the default width the first chunk held only the silent frame. The trim cannot remove a whole chunk, so the client got the silent frame first.
- With a larger requested `initial_codec_chunk_frames`, the trim still ran and the first chunk came out one frame short.

Any request that finishes generating before its first decode starts is affected: short replies, and requests queued behind other first decodes. A two-frame reply sent alone opened with the silent frame every time.

## Change

- The payload sets a stream's first-chunk width only while no codec frame has arrived. Once one has, the first chunk's metadata owns the width.
- The engine writes the request's `initial_codec_chunk_frames` and the eligibility flag into that metadata from the same request parameters, so a stream that is not eligible keeps the width it had before.
- The live-stream limit on the trim (`suppress_bootstrap_max_streams`, 24) is unchanged.

## Results

Eligible requests released together on one H200, 1.7B CustomVoice, Ryan, `language: "English"`, streamed. Request events show which requests the engine finished before their first decode was dispatched. Silent openings per round, main `dfb6b43b3` against this change:

| Requests | Main | This change |
|---|---|---|
| 1 | 1 / 1 / 1 / 1 | 0 / 0 / 0 / 0 |
| 16 | 5 / 4 / 4 / 4 | 0 / 0 / 0 / 0 |
| 24 | 3 / 2 / 4 / 2 | 0 / 0 / 0 / 0 |
| 48 | 48 / 48 / 48 / 48 | 24 / 24 / 24 / 24 |

- Each request generates two frames (`max_new_tokens: 2`). On main every silent opening is a request the engine finished before its first decode, whose first decode then took one frame. With this change every such request takes two frames and opens with speech.
- At 48 requests the 24 streams past the live-stream limit keep the silent frame on both, by design. On main the other 24 lost the extra frame as well.
- With `initial_codec_chunk_frames: 4` and five frames per request, at 1, 16 and 24 requests, 13 of the 14 requests finished before their first decode took 4 frames first on main. With this change every such request took 5.
- Every run ended with no codec slot held, no arena exhaustion and no left-context fallback.

Longer requests finish well after their first decode, so this change does not move them. With 48 concurrent ordinary sentences the 24 streams past the limit open with the silent frame and the rest with speech, on main and with this change alike. With 16, every stream opens with speech on both.

SeedTTS EN sends no language, so no request is eligible, and the payload branch runs on every streamed request. CustomVoice, all 1,088 prompts, main `dfb6b43b3` against this change on one H200, both orders, streaming at 1, 16 and 64 concurrent requests and whole at 1 and 16:

- No streaming row is worse in both orders. First audio mean 13.7 → 13.3 ms at 1 request, 21.6 → 21.6 ms at 16, 41.6 → 36.7 ms at 64.
- Whole requests never reach the changed code. At 1 request whole latency reads 251 and 254 ms against 248 and 250 ms, where main itself read 248, 250, 252 and 303 ms over two pairs on this card. At 16 it reads 2 and 3 ms lower.
- WER 1.52 / 1.36 % → 1.39 / 2.04 %, below 50 % WER 1.49 / 1.28 % → 1.39 / 1.52 %.
- Seeded requests sent one at a time, with and without a language, streamed and whole, are byte-identical to main.

## Tests

- `test_qwen3_tts_bootstrap_suppression_keeps_the_extra_frame_when_the_payload_lands_before_the_first_decode` sends an eligible stream's first two codec frames through the chunk path, delivers stream done and then the payload, and runs the first decode from the queue. The first message must hold only the second frame's audio. On main it holds the silent frame.
