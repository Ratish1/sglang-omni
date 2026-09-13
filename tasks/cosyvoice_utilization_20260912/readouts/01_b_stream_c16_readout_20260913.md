# B stream c16 readout, 2026-09-13

## Verdict

The admission rule in `a8700d833` over `691c18371` is a small gain: C50 9.3 to 13.2, req/s 1.95
to 2.05, WER 1.59 to 1.38 percent. It does not close the continuity gap. Continuity at c16 is
capacity bound, not policy bound: the vocoder cannot process 16 concurrent streams' hops fast
enough to keep any admission order from underrunning, so the rule chooses who underruns and by
how much, not whether underruns happen.

## Provenance

Full A/B, H100, English corpus, 1088 requests, streaming c16, warmup 1, no seed, one boot per arm,
one session on GPU 0. Archived at
`artifacts/full-20260913T171454Z/cosyvoice-ab-gpu0/<arm>/stream-c16/speed_results.json` and
`wer_results.json` in the local checkout's `artifacts/` directory, which is outside this repo.
A is upstream main `51e2f7ec2`. B `691c18371` is the earlier version of our branch, B `a8700d833`
is the PR head. PR 2086 is `4de7afccc` plus the graph fix, reviewed separately in
[PR2086_REVIEW.md](../PR2086_REVIEW.md); the `cosyvoice-slice-gate-20260913-result` run mislabelled
this arm as "Arm A", corrected in [findings item 4](../FINDINGS_20260913.md). The Nsight capture
is on `622bcd198`, the same tree as `a8700d833` plus NVTX instrumentation, 16 requests at c16,
archived at `profile-622bcd198/perfkit-report.md` in the same directory.

## 1. Full A/B table

| arm | failures | WER corpus | first audio mean / p95 s | C50 | req/s | audio s/s | inter chunk mean / p99 s | latency p99 s |
|---|---|---|---|---|---|---|---|---|
| A upstream main 51e2f7ec2 | 23 timeouts | 6.82 percent on 1065 scored | 0.91 / 2.17 | 79.8 | 1.62 | 7.40 | 1.28 / 21.8 | 43.6 |
| B 691c18371 | 0 | 1.59 percent | 3.35 / 6.60 | 9.3 | 1.95 | 9.47 | 2.40 / 6.92 | 14.5 |
| B a8700d833 | 0 | 1.38 percent | 3.32 / 6.35 | 13.2 | 2.05 | 9.72 | 2.26 / 6.55 | 13.4 |
| PR 2086 | 0 | 1.27 percent | 11.58 / 15.39 | 98.6 | 1.32 | 6.39 | 0.26 / 1.03 | 21.8 |

A second A boot (`A/stream-c16-retest`) gave 1.49 req/s, first audio 1.07 s, C50 77.1.

## 2. Underrun anatomy

Underrun is the benchmark's own definition, the max gap past the buffer once playback starts at
the first chunk (`benchmarks/metrics/playback_continuity.py:26-35`), read from the per request
chunk timings. B a8700d833 has 944 of 1088 requests (86.8 percent) with an underrun over 50 ms,
214 at chunk 1 and 716 at chunk 2, median underrun 2.0 s. A has 215 (20.2 percent), 157 at chunk 1,
median 1.6 s, p90 17.6 s. PR 2086 has 15 (1.4 percent). Requests average 3.0 chunks, first chunk
0.84 s of audio, total 4.7 s.

PR 2086's first audio (11.58 s) is within 0.5 s of its mean latency (12.10 s): the vocoder runs
about ten seconds behind the AR, every request's chunks are complete before it reaches them, and
they arrive 0.26 s apart against 0.84 to 1 s chunk durations, so playback never drains. That is
continuity by delivering the request nearly whole, not by keeping up.

## 3. Profile facts

Nsight capture on `622bcd198`, 16 requests at c16: one singleton Flow call costs 270 to 290 ms
host with about 18,100 kernel launches and 67 to 88 ms GPU; one packed causal call with 14 rows
costs 547 ms host, 454 ms GPU. So a hop costs 325 ms alone and 53 ms inside a batch of 16 (855 ms
for 16). Finals are always singletons, 16 of them at 324 ms median.

Vocoder thread budget over the 9.86 s window: flow native 4688 ms (47.6 percent), flow estimator
2266 ms, HiFT 1571 ms, idle 1245 ms. SM active mean 20.7 percent, kernel coverage 31 percent.

Over the full B a8700d833 run the server log shows 192 batched causal calls carrying 1907 hops
(mean batch 9.9) out of about 3264 hops; the 1088 finals alone are about 354 s of the 530 s run.

## 4. Code facts (revision 622bcd198)

`_step_key` is `(token_offset, hop_len)` at
`sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py:81-82`. The batched call takes
`token_offset` and `hop_len` from the head only (`:418-421`, `:440`, `:450`) while the Flow call
itself already pads per row lengths (`stages.py:185-225`) and masks per row (`stages.py:603-623`),
so the key restriction is stricter than the call it gates. The final runs `streaming=False,
finalize=True` through the raw module (`streaming_vocoder.py:496-503`, `stages.py:1376,1397`) and
never through `FlowCudaGraphRunner`, which is reachable only from the non-streaming batch path
(`stages.py:644`). Every hop recomputes the whole token history from frame 0
(`streaming_vocoder.py:424-426`), and HiFT reruns over the entire accumulated mel
(`stages.py:1441-1454`).

## 5. Qwen3-TTS and MOSS comparison

Qwen3-TTS groups decode plans only by decoder input shape
(`sglang_omni/models/qwen3_tts/streaming_vocoder.py:2462-2472`) and replays CUDA graphs keyed by
frames and batch bucket. MOSS local batches any due stream with a uniform frame count
(`sglang_omni/models/moss_tts_local/streaming_vocoder.py:549-573`) with graphs keyed by (B, T).
Neither keys its batching decision on an offset that varies per request the way CosyVoice3's
`token_offset` does.

## What the numbers pin

- The per call host cost is the origin: a singleton Flow call is 270 to 290 ms host time behind
  about 18,100 kernel launches, and this cost recurs on every hop and every final regardless of
  scheduling.
- The same key batching restriction (`_step_key` on head `token_offset` and `hop_len`) and
  singleton finals leave most hops unbatched: 1907 of about 3264 hops go through 192 batched
  calls, and the 1088 finals alone cost about 354 s of the 530 s run.
- A scheduling policy alone can only choose who underruns: at 9.72 audio s/s against 16 real-time
  streams, capacity is below demand, so admission order redistributes the underrun count and
  timing (B a8700d833's 86.8 percent versus A's 20.2 percent) without closing it.
