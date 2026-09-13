# 34. Readout: the codec window bench (doc 33 step 0)

Archives `codec-window-measurements-20260913.tar.gz`,
`codec-window-repeat-measurements-20260913.zip`,
`codec-window-static-dynamic-b14-20260913.zip`, plus the first eager run's JSON. GPU 0,
`tasks/perf_backlog/scripts/codec_window_bench.py`, code branch fd54363d6. Every number
below was read from the JSON files, not from the box's summaries.

## 1. The decoder

8 transformer layers with a sliding window of 72 (71 retained keys per layer), convolution
histories up to 54 frames. The retained state of the last layer depends on inputs up to
8 x 71 = 568 frames back. Front padding a reference of 40 to 160 frames to a captured
width would therefore change the state the generated frames decode from: one-replay
padding is not exact for any real reference. Windows, which replay the reference's own
frames in sequence against the same slot, are exact at every length.

## 2. Eager against replays, device ms, bucket 1

| | eager | one replay, uncompiled | one replay, compiled |
| --- | ---: | ---: | ---: |
| width 1 | | 3.03 | 1.59 |
| width 8 | | 3.53 | 1.81 |
| width 16 | | 4.02 | 2.06 |
| width 32 | | 6.43 | 4.17 |
| width 64 | | 10.19 | 7.13 |
| 40 frames | 10.6 | | |
| 80 frames | 13.8 | | |
| 120 frames | 16.2 | | |
| 160 frames | 20.0 | | |

Eager host time is 8.6 to 9.0 ms at every total; a replay's host time is 0.05 ms. Eager
device time tracks the host: the GPU idles between the roughly 860 launches. A replay has
a floor (the decoder's kernels at their minimum duration) plus about 0.11 ms per frame;
compile halves the floor by fusing kernels.

## 3. Window sequences, device ms, bucket 1, cap 64

| frames | eager | windows uncompiled | windows compiled | windows, dynamic compile |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 10.6 | 9.9 | 5.6 to 6.0 | 6.5 |
| 80 | 13.8 | 14.2 | 8.7 to 9.0 | 9.9 |
| 120 | 16.2 | 24.1 | 14.4 to 15.0 | 16.3 |
| 160 | 20.0 | 26.9 | 17.7 to 18.2 | 20.0 |

Every extra replay costs one floor. Uncompiled windows lose above 80 frames; compiled
windows at cap 64 cost less than eager at every total, stable across three runs, with the
host at 0.1 to 0.2 ms. Cap 64 is the knee: above it the per frame cost dominates. Cap 32
and cap 16 are 1 to 2 ms worse at 120 and 160 frames.

Decision: the bootstrap runs as compiled window replays, ladder 1, 2, 4, 8, 16, 32, 64.

## 4. Compile cost

Static per shape compile, `dynamic=False`, `fullgraph=True`: 3.8 s per shape (5.7 s for
64 x 4). Total capture time was 100.7 s for 12 shapes at buckets 1 and 4, and Dynamo
reported 24 unique graphs for those 12 shapes: every shape compiles twice. Cause,
reproduced on a CPU with the same pattern: `Qwen3TTSIncrementalDecoder.precompile`
builds its sample state outside inference mode while the runner's warmup and capture
gather the state under inference mode; inference tensors carry a different dispatch key
set, the guards differ, and Dynamo traces the shape again. Building the precompile inputs
under inference mode traces it once. The warm runner pays the same double compile today
for its steady stride at four buckets.

Dynamic shape compile (`dynamic=True`, one trace meant to serve every shape): 10 unique
graphs instead of one, 134 s of compile for 12 shapes, 199 s capture, and replays 1 to 21
percent slower than static (the bucket 4 rows the most). Rejected.

The inductor cache does not remove the per process cost: a second identical run captured
in 51.0 s against 50.9 s.

Capture footprint: 1.68 GB for widths 1 to 64 at buckets 1 and 4; 432 MB at bucket 1.

## 5. Boot budget once the double compile is fixed

About 4.2 s per compiled shape (3.8 compile plus capture). Width 8 is already compiled
by the warm runner at every bucket and shared through the decoder's compiled shape set.
Widths 1, 2 and 4 uncompiled cost a 3.0 to 3.3 ms floor instead of 1.6 ms, at most once
each per bootstrap.

| window widths compiled | buckets | shapes | boot cost |
| --- | --- | ---: | ---: |
| 16, 32, 64 | 1 | 3 | about 13 s |
| 16, 32, 64 | 1, 2, 4, 8 | 12 | about 50 s |
| every rung | 1, 2, 4, 8 | 28 | about 120 s |

The fix also halves the warm runner's existing compile (4 shapes, about 15 s saved).
Same width window cohorts of more than one row need buckets above 1; they occur when
concurrent requests share one reference, common with a fixed production voice, rare in
the seed-tts corpus.

## 6. What the branch needs, in order, each its own commit

1. `precompile` builds its sample state under inference mode (one trace per shape), with
   the bench's `--inference-precompile` run as the measurement: 12 unique graphs for 12
   shapes, capture near 55 s at buckets 1 and 4.
2. The compile configuration the runner lacks: Dynamo's recompile limit raised for the
   shapes the vocoder compiles, mirroring SGLang's `set_torch_compile_config`
   (cpu_graph_runner.py:121-131 in the pinned tree, called by the decode graph runner at
   decode_cuda_graph_runner.py:391). The inductor cache is already on by default in this
   torch.
3. The window runner compiles widths 16, 32 and 64 (8 shared, 1, 2, 4 uncompiled).
4. The default ladder 1 to 64 with these numbers in the message. The bucket list follows
   the other runners unless the boot budget says otherwise.

Then the three test files on the box, then the pair of runbook 33.
