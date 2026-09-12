# 25. E4 readout and the trace evidence behind the first chunk regression, 2026-09-12

Sources: the E4 archive (`qwen3-tts-e4-dc55923e2-20260912.tar.gz`, every arm at slice A
`04b62c255`, one server per experiment, two streaming passes at c16, event recorder without
the torch profiler in pass 2), the streaming pairs archive of doc 24, the process wide kineto
traces the 2026-09-11 c16 census wrote for main `b2cc93b0a` and slice A `51c5bb064`
(`artifacts/sliceA/results/{A/census_valid,B/census}/census_c16/trace_preprocessing_*.gz`),
and the user's audit in `tasks/qwen3_tts_streaming_pairs_20260912/`.

Doc 24 section 3 leaned on the interpreter lock. Section 1 below rejects that lean, section
3 replaces it with what the traces show, section 5 lists the corrections to doc 24.

## 1. What E4 measured

Streaming, c16, pass 2 of each experiment against the clean slice A control (streaming pair
2, arm A, pass 2, no source change):

| arm | req/s | TTFC mean / p99 | inter chunk mean | latency mean | C200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| slice A control | 17.64 | 0.178 / 0.401 s | 0.090 s | 0.902 s | 100 |
| E4a, switch interval 0.5 ms | 17.37 | 0.190 / 0.421 s | 0.090 s | 0.915 s | 100 |
| E4b, preprocessing and vocoder streams at -1 | 9.64 | 0.158 / 0.385 s | 0.185 s | 1.651 s | 100 |
| E4c, three processes on one GPU | 8.63 | 0.227 / 0.638 s | 0.201 s | 1.846 s | 98.6 |

- E4a changes nothing. A shorter switch interval only shortens how long a waiting thread
  waits for a holder that runs pure Python; the scheduler thread releases the lock at every
  torch call anyway, so the interval was never the limiter. The lock, if it matters, matters
  as capacity, and no interval buys capacity.
- E4b halves throughput and doubles inter chunk latency, and both sibling segments recover
  (section 2). The readout on the box called -1 the highest priority. It is not: PyTorch
  clamps the device range to (0, -3) and a more negative number is a higher priority
  (c10/cuda/CUDAStream.h:162-182, CUDAStream.cpp:204 and 357), so the vocoder's resolver
  (`_decode_stream_priority`, streaming_vocoder.py:1591) already runs the decode streams at
  -2 and E4b lowered them to -1 while raising the preprocessing stream from 0 to -1. The
  collapse is the preprocessing stream sitting above the talker: eight worker threads keep
  small eager kernels pending most of the time, and at a higher priority every one of them
  goes ahead of the next node of the talker's predictor replay.
- E4c is not the layout doc 24 meant. Three processes on one GPU are three CUDA contexts
  and the device time slices between them, which is why request build went from 0.8 to
  15 ms (cross process payload import) and throughput halved. It says nothing about the
  lock; the vocoder in its own process still needed 31 ms for the first chunk because the
  talker's context held the device between slices.

None of the three is a mitigation. E4b is still informative: when the talker's stream is
starved, both siblings run at the speed they had on main.

## 2. Where the first chunk goes under slice A

p50 / p95 ms from the pass 2 event files, admission to the coordinator's receipt of the
first audio chunk. E4a is slice A with a change that had no effect, so it is the slice A
anatomy; main's anatomy under the same recorder has not been recorded (E5a below).

| segment | E4a (slice A) | E4b | E4c |
| --- | ---: | ---: | ---: |
| preprocessing dispatch to complete | 78.4 / 176.7 | 32.8 / 131.0 | 73.8 / 171.2 |
| engine input to request build start | 5.7 / 13.9 | 13.3 / 35.7 | 17.0 / 38.5 |
| request build | 0.8 / 1.6 | 0.5 / 1.2 | 15.2 / 76.8 |
| build end to queue enter | 8.9 / 21.4 | 20.7 / 35.6 | 15.0 / 31.6 |
| queue enter to prefill start | 2.8 / 6.2 | 10.0 / 13.7 | 11.0 / 21.0 |
| prefill | 24.6 / 43.1 | 25.8 / 44.8 | 23.9 / 32.3 |
| first emit to first chunk sent | 0.3 / 0.7 | 0.4 / 0.6 | 0.5 / 8.2 |
| vocoder first chunk received to sent | 70.2 / 145.3 | 24.7 / 91.0 | 31.3 / 91.1 |
| total, admission to first chunk at the coordinator | 198.8 / 363.6 | 138.3 / 349.2 | 208.8 / 368.8 |

Two segments carry the first chunk: preprocessing (78 ms) and the vocoder's first decode
(70 ms). Everything the talker does between them is 43 ms. Under E4b both sibling segments
drop by 45 ms while every talker segment grows, and the vocoder segment drops although its
priority went down: what the siblings wait on is the talker's device work, and the vocoder's
existing priority does not buy it a prompt first decode.

The vocoder's first decode is the eager, reference prefixed bootstrap (`_run_initial_batch`,
streaming_vocoder.py:2221): the whole reference prefix plus the first frames go through the
incremental decoder without a graph (every archived cold runner has zero replays), planned
after `_wait_codes_ready` orders the worker's stream behind the talker's newest chunk event
(streaming_vocoder.py:1584). Nothing in that path changed between the arms.

## 3. What the census traces show, main against slice A

The 2026-09-11 c16 census windows were recorded with the torch profiler on, and the profiler
captures the whole process: every kernel on every stream and every launch call on every host
thread. Both arms were recorded the same way, so the comparison is like for like, but the
windows ran at 7.8 and 6.1 req/s against 16 to 17 in production and slice A's window opened
with a 13 s stall (its first 24 requests saw preprocessing at 414 ms p50), so the absolute
segment numbers of doc 24 section 2 are profiler loaded, not production. The relative facts:

Who runs where (thread attribution through the launch correlation ids):

| thread | stream | priority | kernels in the window, main / slice A |
| --- | --- | ---: | ---: |
| talker scheduler | 7, the default stream, channel 0 | 0 | 1.65M / 1.78M, 2084 / 2249 graph launches |
| reference encoder batcher | 17, channel 7 / 2 | 0 | 81.8k / 76.2k |
| eight preprocessing workers | 21, channel 1 / 7 | 0 | 45.6k / 45.6k |
| vocoder, non streaming whole utterance decode | 7, the default stream | 0 | 81.8k / 71.1k |

The non streaming vocoder decodes on the calling thread's current stream (`_vocode_payloads`
to `tokenizer.decode`, streaming_vocoder.py:2782), which is the default stream, so in the
census the vocoder stage sits FIFO behind the talker by construction. That is why its segment
grew 222 to 281 ms once the talker's queue stopped draining each step; it says nothing about
streaming, where the first decode runs on a -2 stream.

GPU execution of the sibling kernels did not change: the same kernels take 2 to 6 percent
longer on slice A (elementwise 5.9 to 6.2 us, fp32 gemm 18.6 to 18.4, cutlass bf16 gemm
40.4 to 39.3), and the preprocessing streams' total busy time is 2.6 ms per request on both
arms. Preprocessing is not GPU bound; its stage time is host time plus waits.

Dispatch of a sibling kernel launched onto an idle stream (nothing of its own queued ahead):

| stream | main p50 / p90 / p99 / mean us | slice A p50 / p90 / p99 / mean us |
| --- | ---: | ---: |
| 17, encoder batcher | 6.5 / 12.1 / 77 / 17.2 | 7.1 / 14.3 / 552 / 27.6 |
| 21, preprocessing workers | 7.1 / 13.7 / 100 / 18.8 | 7.3 / 15.4 / 527 / 28.1 |
| 7, talker, eager kernels | 6.9 / 11.2 / 26 / 9.2 | 7.0 / 11.9 / 28 / 10.6 |

The median is the same, the tail grows five to seven times, the mean grows by 10 us per
kernel. The streams sit on their own hardware channels, so this is the device scheduler
serving a continuously fed default stream ahead of equal priority arrivals, not queue sharing.
At 450 preprocessing kernels and 1100 encoder kernels per batch it is 5 to 15 ms per request.

Host side of the sibling threads, launch calls and the Python between them:

| metric, eight preprocessing workers | main | slice A |
| --- | ---: | ---: |
| launch call p50 / p99 us | 5.0 / 105 | 5.3 / 649 |
| launch call total in the window | 819 ms | 1333 ms |
| gap between consecutive launches p50 / p90 us | 30.8 / 214 | 39.8 / 284 |
| stream and event waits, total | 48 ms | 88 ms |

The launch call tail is inside the driver; the gap growth is Python and lock handoff. Both
are a few ms per request. The waits on the device did not grow. So the traces put slice A's
effect on the siblings in three places, device dispatch tail, driver launch tail and Python
glue, each small per kernel and paid on every one of a few thousand kernels per request, and
none of them is a wait on GPU execution. The remaining difference between these sums and
the 45 ms E4b recovered per segment is not resolved by a profiler loaded window; E5a
measures it in production.

## 4. Other findings from the archives

- Runaway generations. Eight of the 35 archived full c16 passes carry one request that hit
  the 2048 frame cap (163.84 s of audio, 13 to 33 s latency): main arms (the c16 followup
  main arm, the full repeat main arm), slice A arms, and the stacked branch, on different
  texts each time, including a complete sentence. It is the model's unseeded sampling at
  about one request in four thousand, not a branch defect. It adds a 10 s plus tail to the
  pass, so the c16 pair quoted in the #2123 body (main 16.03 with a runaway against 17.03)
  overstated the gain; the clean same session pair is the seeded c16 pair, main 16.66 against
  slice A 16.86 (+1.2 percent, p99 latency -4.5 percent). The body is corrected. The check
  is one line: the maximum audio duration of the pass.
- Radix hits on the warm pass. Non streaming c16 runs hit 12.5 percent on both arms. The
  second streaming pass, on a warm tree, hit 80 percent on main and 67 to 71 percent on the
  slice A arms. Prompt keys hash the prompt embedding rows, which come out of the reference
  encoder, and the reference cache (256 entries) misses and re-encodes in batches of up to
  eight whose composition follows arrival timing, so a re-encoded reference can key
  differently from its first encode. It is an efficiency item of its own (a warm pass should
  hit near 100 percent), not the first chunk mechanism: prefill is 25 ms in every arm.
- The nonblocking branch (#2126) moves streaming by nothing: +10.9 ms TTFC on the matched
  pass, 614 of 1088 slower, QPS +0.5 percent, inside two boots of the same code.

## 5. Corrections to doc 24

- Section 3's mechanism: the interpreter lock is not established and the switch interval is
  rejected; the traces show device dispatch tail, driver launch tail and Python glue, all
  host visible, none a wait on execution.
- Section 2's segment table is from profiler loaded windows at a third of production
  throughput, one of them with a startup stall; keep the direction, not the magnitudes.
- Section 1: pair 1 stays unusable, so the streaming regression main against slice A in the
  default layout has no clean same session pair yet. Its direction is supported four ways
  (pair 1, the cross session control at +22 percent, the router runs at +13 percent, the
  census segments); its size is E5a's job.
- The vocoder's priority is -2, not -1, and the non streaming decode shares the talker's
  stream.

## 6. E5

All streaming at c16, one independent server per arm, two passes, warmup 1, no seed, the
event recorder on for a 200 request window in pass 2 with `enable_torch` false (the E4
protocol, `protocol.md` in the E4 archive), GPUs 1 to 3 recorded before each boot, one
session.

- E5a, the anatomy pair. Main (current upstream main) and slice A `04b62c255` with no source
  change. Read the section 2 table for both arms and the client TTFC. This is the number the
  PR needs and the baseline every mitigation is read against.
- E5b, bounded run ahead. Slice A plus an uncommitted change in model_runner.py: record an
  event after `code_predictor_forward` in `_collect_codes` and wait on it at the top of
  `before_decode` for the same batch, so the host tail still overlaps the predictor but the
  next backbone is launched onto a drained stream. Expected cost at c1: the backbone graph
  launch no longer overlaps, a fraction of the 1.8 ms step gain. Expected effect: the
  talker's queue empties once per step, the scheduler thread blocks for the predictor's
  remainder (lock released), and the siblings dispatch into that window. Read: section 2
  segments, TTFC, req/s, plus a c1 seeded pass for the step cost.
- E5c, non streaming vocoder off the talker's stream. Slice A plus `_vocode_payloads` run
  under `_decode_stream_context()` (streaming_vocoder.py:2782); `tokenizer.decode` returns
  host arrays so it synchronizes its stream before returning and no consumer reads the
  device. Non streaming c16 pair against slice A: the vocoder stage segment and QPS. This
  is independent of the first chunk question and may be a c16 latency win on its own.
- Only if E5a leaves the vocoder segment unexplained: a streaming window with the torch
  profiler on both arms (`enable_torch` true, 192 requests) and the two scripts under
  `tasks/perf_backlog/scripts/` (`trace_streams.py`, `trace_dispatch.py`) applied to the
  initial worker's stream. Archive the traces for that run; the reports only rule does not
  cover a dispatch question.

## 7. Consequence for the PRs

#2123 stays open and unmerged until E5a puts a number on the streaming cost in the default
layout and E5b or another mitigation is measured against it. Its c16 table now carries the
clean seeded pair. #2126 follows #2123 and needs nothing further from this investigation.
