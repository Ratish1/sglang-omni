# Streaming scheduler first audio regression, 2026-09-15

MOSS-TTS Local and Qwen3-TTS streaming c16 first audio were reported to rise after the streaming
scheduler change merged into upstream main as `442e559b4`. For MOSS-TTS Local a paired A/B on one
GPU shows no regression: the gap is inside the run to run spread of identical code
(`readouts/02_moss_tts_local_paired_ab_c16_20260915.md`). Qwen3-TTS is not paired yet. The user's A
runs below stay as the reference `scripts/compare_to_a.py` compares against.

## Arms

- **A**: upstream main before `442e559b4`, as run by the user. Numbers in the tables and in
  `baselines/stream_en_c16.json`.
- **PR 1 head**: the user's runs of the streaming scheduler branch before it merged. Its runtime
  tree equals `442e559b4`. Kept next to A for reference.
- **B**: a boot from this branch's worktree. Since the merge commit `4bd0235cc` every file outside
  `tasks/` equals upstream main `442e559b4`; each runbook checks this on the box before booting.

## Fixed reference, streaming, English, c16, 1,088 requests

MOSS-TTS Local (`OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`):

| metric | A | PR 1 head |
|---|---|---|
| completed / failed | 1088 / 0 | 1088 / 0 |
| req/s | 12.945 | 12.425 |
| RTF mean | 0.2737 | 0.2741 |
| RTF p95 | 0.3301 | 0.3591 |
| RTF p99 | 0.3924 | 0.4694 |
| first audio mean s | 0.260 | 0.309 |
| first audio p50 s | 0.256 | 0.282 |
| first audio p95 s | 0.380 | 0.555 |
| first audio p99 s | 0.546 | 0.792 |
| C50 | 100.0 | 99.17 |
| C100 | 100.0 | 99.63 |
| latency mean s | 1.218 | 1.281 |

Qwen3-TTS:

| metric | A | PR 1 head |
|---|---|---|
| completed / failed | 1088 / 0 | 1088 / 0 |
| req/s | 15.863 | 15.690 |
| RTF mean | 0.2443 | 0.2462 |
| RTF p95 | 0.2909 | 0.2889 |
| RTF p99 | 0.3281 | 0.3194 |
| first audio mean s | 0.1427 | 0.1709 |
| first audio p95 s | 0.249 | 0.3214 |
| first audio p99 s | 0.4105 | 0.5326 |
| C50 | 99.91 | 100.0 |

## Code facts, read at `442e559b4`

- On both models' paths the only files that differ from A are
  `sglang_omni/scheduling/streaming_simple_scheduler.py` and
  `sglang_omni/scheduling/streaming_vocoder.py`.
- Changed lines both models run before first audio: the loop head `_has_ready_work()`
  (streaming_simple_scheduler.py:148) and `_next_message` through `_get_batch_message` (:203-213).
  MOSS-TTS Local also runs the chunk collector read (:362) and the split pump
  (streaming_vocoder.py:362-389). For both models these execute the same logic as A.
- The paired measurement agrees with the reading: the vocoder's own first audio time is the same at
  A and B (readout 02).

## Index

- `baselines/stream_en_c16.json`: the two tables above, keyed by the benchmark's summary names.
- `scripts/compare_to_a.py`: prints A, PR 1 head and B with B minus A for one `speed_results.json`;
  `--a-speed-results` takes A from a paired run instead.
- `scripts/first_chunk_anatomy.py`: per hop first audio breakdown from the request event recorder,
  copied unchanged from `fb590a2c3:tasks/perf_backlog/scripts/first_chunk_anatomy.py`.
- `scripts/vocoder_event_timings.py`: vocoder-side timings from one recorder directory: first audio
  after chunk 0 and chunk 1, later chunks, backlog stalls, first audio by requests in flight,
  transport, and ten time windows.
- `runbooks/01_moss_tts_local_stream_c16.md`: B census plus the recorder pass and the breakdown,
  with the benchmark at its default settings as in the A runs.
- `runbooks/02_moss_tts_local_paired_ab_c16.md`: the same two passes for A (`1f6b6843e`) and B
  (`442e559b4`) back to back on one GPU, with paired deltas.
- `readouts/02_moss_tts_local_paired_ab_c16_20260915.md`: the paired result, the six runs, the
  first audio breakdown and the run to run variance.
