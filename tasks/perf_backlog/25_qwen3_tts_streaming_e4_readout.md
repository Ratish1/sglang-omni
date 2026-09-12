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
(70 ms). Everything the talker does between them is 43 ms.

Under E4b both sibling segments drop by 45 ms, but E4b and E4c also admit half as many
requests per second, so their siblings run under half the load (the user's audit,
`tasks/qwen3_tts_e4_investigation_20260912/README.md`). The vocoder segment binned by how
many other requests were between their first code receipt and their first audio send at the
moment a request's first code arrived:

| other requests mid bootstrap | E4a n / p50 ms | E4b n / p50 ms | E4c n / p50 ms |
| ---: | ---: | ---: | ---: |
| 0 | 82 / 51.8 | 162 / 23.5 | 147 / 25.9 |
| 1 | 82 / 64.7 | 34 / 38.8 | 46 / 49.1 |
| 2 | 63 / 72.2 | 5 / 62.6 | 7 / 67.6 |
| 3 | 35 / 91.4 | 1 / 65.6 | 4 / 82.5 |
| 4 | 11 / 108.6 | 1 / 89.2 | 4 / 108.6 |

The slope is the same in every arm, 15 to 20 ms per request ahead in the queue, and E4b and
E4c mostly bootstrapped alone. The vocoder's first decode is a serial queue: the initial
worker takes a batch off its queue, plans every request in it under the state lock
(`_run_initial_batch`, streaming_vocoder.py:2221), each plan waiting on the newest chunk's
event and concatenating every retained chunk (`_build_incremental_plan`, 1384 and 1441), and
reference prefixed bootstraps fall into singleton cohorts decoded one after another through
the eager incremental decoder (no cold graph replays in any arm). At 17 req/s a request
usually finds one to three others ahead of it. That queue predates both PRs.

What is left once queueing is removed is the lone bootstrap: 52 ms in E4a against 24 to
26 ms in the two arms whose talker was starved. That residual, and the same question for
preprocessing, is what a main against slice A trace has to split between the talker's
device pressure and host time; section 3 says the device execution of sibling kernels is not
it. The median number of code chunks already received when first audio leaves is 4 in E4a
and 1 to 2 in E4b and E4c, so a plan in E4a waits on a chunk event up to a few steps newer
than the frame it decodes; that costs at most a step or two per plan and is worth reading
directly.

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

The procedure is the user's `tasks/qwen3_tts_e4_investigation_20260912/next_diagnostic.md`:
control `645b472cd` and control plus `early_ids.patch` (the same base, so #2115 is on both
arms), the default single process layout, streaming c16, control then early ids then control,
two passes per boot, then a 15 to 25 s Nsight Systems window on the server process tree
with CUDA, OS runtime, Python sampling and GIL tracing, plus temporary ranges at the seven
sites its table lists (scheduler publish, vocoder ingest and initial enqueue, initial batch
planning with the lock, cohort launch, handle wait and commit, outbox send, prefix key
digest). Read for a delayed first chunk: when its codes were device ready, when the initial
worker took it, when the decode was submitted, started and finished, when the audio left the
outbox. The decision table in that file maps each outcome to the fix.

Two additions from this doc:

- The segments of section 2 and the queue table come for free from the event recorder in
  pass 2, so run it on both arms alongside; the queue depth at first code receipt and the
  lone bootstrap time are the two numbers to compare between control and early ids.
- If Nsight is not installed on the box, the fallback is a torch profiler window on both
  arms (`enable_torch` true) read with `tasks/perf_backlog/scripts/trace_streams.py` and
  `trace_dispatch.py` on the initial worker's stream and thread; it gives dispatch, launch
  and execution but not lock ownership, and the lock question then stays open. Archive the
  raw traces for this run.

Deferred until that trace has been read: the bounded run ahead (an event after
`code_predictor_forward` waited at the top of the next `before_decode`, so the stream
drains once per step at a fraction of the 1.8 ms gain), a raised priority for the
preprocessing stream alone, and the non streaming vocoder decode moved onto its priority
stream (`_vocode_payloads`, streaming_vocoder.py:2782, a c16 latency candidate of its own).

## 7. Consequence for the PRs

#2123 stays open and unmerged until E5a puts a number on the streaming cost in the default
layout and E5b or another mitigation is measured against it. Its c16 table now carries the
clean seeded pair. #2126 follows #2123 and needs nothing further from this investigation.
