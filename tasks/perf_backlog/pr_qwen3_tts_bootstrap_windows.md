# PR draft: [Qwen3-TTS] Replay reference prefixed bootstraps through captured window graphs

Branch `perf/qwen3-tts-bootstrap-graphs` at f73586369 on upstream main 3060470a8.
Evidence: doc 36 (session 2 archive `bw-session-results-no-wavs.tar.gz`), doc 34 (bench).

## Body

### Mechanism

A streaming request's first decode on the vocoder's initial worker is the reference
frames plus the first chunk, a width no CUDA graph captured, so it ran eager: about 860
kernel launches per request on one thread of a process where 18 threads share the
interpreter lock. Every eager launch is a lock handoff, and the scheduler, the
preprocessing workers and the prefill all paid for it in their own segments. With this
PR that decode is about 3 graph replays per request (9,632 replays for 3,264 requests
on the census boot) and the cold runner's uncaptured width misses go from 2,688 to 0.

This PR gives the initial worker a second graph runner with widths 1, 2, 4, 8, 16, 32, 64
at the same batch buckets as the cold and warm runners. A bootstrap whose width no runner
captured is consumed as a sequence of those widths, largest first, against the stream's
own arena slots, so the state and the waveform end where one wide decode would (the
decoder's incremental step is partition invariant). Each replay's output is copied into
one waveform the cohort owns before the next replay overwrites it. Width 8 replays the
compiled step the warm runners already trace; every other width is captured eager.
64 is the measured knee: above it a replay's per frame cost outweighs the floors it saves.

### Changes

- `incremental_codec_cuda_graph.py`: a `window` runner mode, `plan_decode_windows`
  (greedy largest first cover), `plan_windows` and `largest_batch_bucket` on the runner.
- `streaming_vocoder.py`: the window runner built next to the cold runner, the cohort
  decode path (cold replay, else windows, else eager), the windowed decode, cohort
  splitting at the window runner's bucket, the `incremental_codec_cuda_graph_window_frames`
  knob (empty disables), a `window` entry in the codec state log line.
- `incremental_codec.py`: `precompile` traces on the runner's own capture tensors under
  the runner's inference context. Before, it traced on plain tensors and Dynamo guarded
  on the tensor kind, so every compiled shape traced twice (24 graphs for 12 shapes in the
  bench). Bit exact; boot time falls by more than the new captures cost.
- `stages.py`: the factory forwards the knob.
- Tests: window planning, cohort batching and splitting, remainder handling, the miss
  path, runner construction and knob validation, precompile tracing on the given tensors.

Non streaming is untouched: the whole sequence path does not use the incremental decoder.

### Census, default launch, streaming c16, full seed-tts corpus, H100, one boot per arm

| read | main 3060470a8 | this PR | delta |
| --- | ---: | ---: | ---: |
| req/s | 15.43 | 15.78 | +2.3 percent |
| audio s/s | 63.8 | 65.6 | +2.8 percent |
| RTF mean | 0.2515 | 0.2445 | -2.8 percent |
| TTFC mean ms | 129.7 | 117.0 | -12.7 |
| TTFC p50 ms | 117.3 | 109.8 | -7.5 |
| TTFC p99 ms | 326.9 | 301.7 | -25.2 |
| inter chunk mean ms | 112.2 | 110.3 | -1.9 |
| first frame to first audio, mean ms | 35.3 | 29.9 | -5.4 |
| first frame to first audio, p95 ms | 65.2 | 48.8 | -16.4 |
| cold graph misses (uncaptured width) | 2,688 | 0 | |
| SM utilization during traffic, nvidia-smi dmon, percent | 78.3 | 80.9 | +2.6 |
| launch to ready, s | 96 | 90 | -6 |

Quality on this PR: WER 1.05 percent, speaker similarity 71.53, inside the bands.

Window runner footprint 2.1 GB at buckets 1, 2, 4, 8. Knob to disable:
`incremental_codec_cuda_graph_window_frames=[]`.
