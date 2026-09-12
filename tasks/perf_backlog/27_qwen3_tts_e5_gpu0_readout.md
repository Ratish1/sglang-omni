# 27. E5 on GPU 0, control against early ids, bounded run ahead, clean Nsight, 2026-09-12

Archive `e5-gpu0-rerun-22951f0ee.tar.gz`. Every number was recomputed from the raw files
(speed_results.json, client_timestamps.jsonl, the event recorder files, serve.log,
telemetry.txt, the SQLite exports of the two clean Nsight windows). The box's REPORT.md
and NSIGHT_THREAD_READOUT.md were read afterwards; where they agree with the raw data it
is said, where the archive lacks something it is flagged in section 7.

## 1. Provenance and regime

- Base 645b472cd on every arm; early ids is the base plus the validated patch (worktree
  diff SHA 7afb15aa matches patch_validation.json); the bounded run ahead is early ids plus
  the event patch in `run_ahead/bounded_run_ahead.patch` (an event recorded after
  `code_predictor_forward`, synchronized at the top of the next `before_decode`). The
  first `import.txt` of step 1 was taken from the repository root by mistake; the post run
  checks and the 43 and 45 Python source paths Nsight recorded inside each worktree
  establish that the servers ran the intended trees.
- Physical GPU 0, empty before every boot; GPUs 2 and 3 held a tenant at 100 percent and
  348 W throughout; GPUs 0 to 3 share NUMA node 0. Telemetry during step 1: P0, SM clock
  1980 MHz, memory clock 2619 MHz, no throttle reason, 205 to 214 W mean, node load 3 to 4
  on 128 logical CPUs.
- The regime is the same as the GPU 2 session and not the E4 session: 28.6 to 33.5 ms
  between decode log lines per 40 steps against 15.0 to 15.6 in the E4 session and pair 2,
  generation 445 to 516 tokens per second at 15 or more rows against 850 to 870,
  9.1 to 10.0 req/s against 17.6. So the half speed regime is not GPU 2, not clocks, not
  power, not thermal. The clean Nsight windows show where it sits: the two talker graphs
  replay in 13.6 ms (graph 217, 613 launches) and 6.0 ms (graph 250) on control, 12.7 and
  6.6 ms on early ids, against about 4.3 ms for the predictor and under 2 ms for the
  backbone in every earlier profile at 16 rows, and the reference GEMM kernel runs
  11.9 us against 7.4 us in the 11 Sept census on the same GPU 0. The device executes the
  same graphs three times slower at full clocks. Pair 1 ran the same base at 13:19 the same
  day at 14.4 req/s; E4 ran at 14:35 at 17.4; every session from 15:54 on has been at the
  slow regime, on GPU 2 and on GPU 0. The cause is not in the archive (section 7). Both
  arms of every pair below ran in the same regime, so each pair is valid as a pair.
- Runaways: none in the GPU 0 session. Early pass 1 has a single stall: its first 16
  requests, admitted at +1.17 s, all saw first audio after 5.5 to 6.9 s and one completion
  gap of 6.8 s at +1.1 s; nothing is logged during it. It is a warm up event of the first
  c16 batch (the single warmup request does not exercise the batched shapes), it does not
  recur in pass 2, and it is why pass 2 is the measurement.

## 2. Client level, pass 2

| pass 2 | control | early ids | delta | bounded run ahead | against early ids |
| --- | ---: | ---: | ---: | ---: | ---: |
| req/s | 9.146 | 9.991 | +9.2% | 9.926 | -0.7% |
| TTFC mean | 126.5 ms | 136.8 ms | +10.3 ms | 140.7 ms | +3.9 ms |
| TTFC p50 | 117.9 ms | 129.1 ms | +11.2 ms | 130.3 ms | +1.2 ms |
| TTFC p99 | 372 ms | 362 ms | -10 ms | 346 ms | -16 ms |
| inter chunk mean | 200.1 ms | 179.9 ms | -10.1% | 181.4 ms | +0.8% |
| latency mean / p99 | 1.741 / 2.830 s | 1.591 / 2.642 s | -8.6% / -6.6% | 1.604 / 2.660 s | +0.8% |

The GPU 2 session read +15.9 ms TTFC mean and +7.0 percent req/s on the same base; this
one reads +10.3 ms and +9.2 percent. The direction and the size are reproduced.

## 3. Stage level, pass 2 event windows (p50 ms)

| segment | control | early ids | bounded |
| --- | ---: | ---: | ---: |
| preprocessing dispatch to complete | 27.7 | 34.9 | 32.7 |
| engine input to request build start | 8.4 | 13.9 | 13.9 |
| build end to queue enter | 23.2 | 20.0 | 19.7 |
| queue enter to prefill start | 1.0 | 10.4 | 10.3 |
| prefill start to first emit | 35.5 | 23.4 | 23.8 |
| vocoder first code to first audio | 19.7 | 25.9 | 30.9 |
| admission to first audio | 126.2 | 139.1 | 142.8 |
| first audio at zero requests ahead | 18.9 (n 174) | 22.9 (n 160) | 23.1 (n 154) |
| first audio at one request ahead | 35.5 (n 35) | 38.8 (n 37) | 38.6 (n 45) |
| code chunks received before first audio | 1 | 2 | 2 |

The two talker segments trade as in doc 26: queue wait +9.4 ms, prefill -12.1 ms, the
first frame is emitted at the same absolute time (36.5 against 33.8 ms after queue enter,
early ids slightly earlier). The absolute regression is again preprocessing (+7.2), the
engine's build executor pickup (+5.5) and the vocoder's first decode (+6.2, of which +4.0
at depth zero and the rest a deeper queue: 2 chunks received before first audio).

The bounded run ahead changed nothing that matters: same queue wait, same prefill, same
depth zero bootstrap, vocoder +5 ms, req/s -0.7 percent. In this regime the host tail of
a step (about 17 ms, section 4) is longer than the predictor, so the event it waits on is
already complete when `before_decode` runs and the patch is a no op. It has to be judged
in the regime where the predictor outlives the host tail, which is the E4 regime.

## 4. Thread level, the clean Nsight windows (200 requests each, control 26.1 s, early 23.2 s)

Threads identified by their kernel mix and their CUDA calls, from the SQLite exports
(script `tasks/perf_backlog/scripts/nsys_threads.py`). A is the launch call's host
duration, Q the launch to device start, GIL wait and hold are the NVTX GIL ranges.

| thread | metric | control | early ids |
| --- | --- | ---: | ---: |
| talker scheduler | graph launches (2 per step) | 1581 | 1621 |
| | ids wait, cudaEventSynchronize mean | 16.0 ms x 864 | 8.1 ms x 879 |
| | cudaStreamSynchronize mean | 6.9 us x 2826 | 702 us x 2732 |
| | GIL hold / wait | 3.75 s / 2.88 s | 3.69 s / 3.43 s |
| | GIL acquisitions (wait ranges) | 91.4k | 110.0k |
| initial worker | launches, A mean | 198k, 4.5 us | 197k, 6.8 us |
| | kernel execution sum | 1686 ms | 1718 ms |
| | Q mean / p50 / p90 | 608 / 6.8 / 2311 us | 1184 / 14.4 / 4062 us |
| | cudaStreamWaitEvent | 402 | 402 |
| | cudaStreamSynchronize mean | 8.5 us x 367 | 640 us x 366 |
| | decode done cudaEventSynchronize mean | 2.52 ms x 166 | 2.24 ms x 165 |
| | GIL wait / acquisitions | 822 ms / 31.5k | 1474 ms / 48.8k |
| follow up workers (2) | A mean | 14.8 / 11.8 us | 88.2 / 86.4 us |
| | decode done cudaEventSynchronize mean | 281 / 295 us | 2703 / 2265 us |
| | GIL wait | 294 / 288 ms | 557 / 554 ms |
| preprocessing workers (8) | A mean | 6.0 to 7.1 us | 8.4 to 19.3 us |
| | Q mean | 11 to 17 us | 17 to 37 us |
| | kernel execution sum | 154 ms | 159 ms |
| | GIL wait, all eight | 1973 ms | 2594 ms |
| reference encoder batcher | A mean | 4.8 us | 7.6 us |
| | GIL wait | 1201 ms | 1573 ms |
| whole process | GIL wait / hold | 9.48 s / 8.11 s | 12.52 s / 8.32 s |
| | per request | 47.4 / 40.6 ms | 62.6 / 41.6 ms |

Reading, from the scheduler outward:

- The scheduler's one long wait per step (16 ms in the ids wait, which on control covers
  the predictor) becomes an 8 ms wait plus 3.5 stream synchronizations of 0.7 ms per step.
  Those synchronizations are the pageable copies of the churn steps (the restage in
  `prepare_decode_buffers` and the finish copy) now waiting behind the queued predictor;
  they are what #2126 removes. The scheduler holds the lock for the same total time but
  acquires it 20 percent more often, in shorter slices.
- Every sibling thread acquires the lock more often and waits longer for it: the initial
  worker 55 percent more acquisitions and +3.3 ms of waiting per bootstrap, the eight
  preprocessing workers +3.1 ms per request together, the follow up workers and the
  encoder batcher likewise. Process wide the wait grows by 15 ms per request while the
  hold grows by 1 ms. This is contention through handoff frequency, not through more
  holding, which is why a shorter switch interval (E4a) could not help.
- Sibling launch calls get slower on the host (initial worker +50 percent, preprocessing
  +40 to +170 percent, follow up workers 6 times), about +2 ms per bootstrap and +1 ms per
  preprocessing request. Both windows are under Nsight, whose per call callbacks add to
  this, so the magnitude is not a production number; the direction was already in the
  11 Sept census under a different profiler (doc 25 section 3).
- Sibling GPU execution is unchanged (initial worker 1686 against 1718 ms of kernels for
  the same request count, preprocessing 154 against 159 ms). Sibling dispatch tails grow
  (initial worker Q p90 2.3 to 4.1 ms) and the initial worker's own stream
  synchronizations now wait 0.64 ms each, both consequences of a device queue that no
  longer drains each step. The decode done waits are not longer (2.52 against 2.24 ms),
  so the first decode is not slower on the device.
- The follow up workers' decode done waits grow ten times (0.28 to 2.7 ms): a warm decode
  plan waits on the newest chunk's readiness event, and in this regime the predictor that
  event trails takes 13 ms. Inter chunk latency still improves because the talker step is
  shorter.

So the first chunk on early ids pays three host side costs on each sibling stage, lock
handoff, launch calls and readiness waits, and none of them is a wait on GPU execution.
On this regime they sum to about 10 ms at the client; on the E4 regime, where the initial
worker's queue was deep, the same costs were amplified to 30 to 100 ms.

## 5. Is the code misplaced

Same answer as doc 26: no. The change is correct and every cost above is a systemic
consequence of the scheduler thread no longer parking once per step. The pre existing
places that amplify it are the serial initial worker (doc 25 section 2), the pageable
churn copies (#2126) and the readiness dependency on the newest chunk (doc 25). The bounded
run ahead is not a fix in this regime and is unmeasured in the E4 regime.

## 6. What this settles and what it does not

- Settled, twice on two GPUs: in the default layout at c16 the overlap trades a first
  chunk cost of 10 to 16 ms mean at 9 to 10 req/s for +7 to +9 percent throughput, -8 to
  -10 percent inter chunk and -7 to -9 percent request latency, with the first frame
  emitted at the same time and the cost paid by the preprocessing, request build and
  vocoder stages through lock handoff, launch calls and readiness waits.
- Not settled: the same numbers at the production regime (17 req/s), where the E4 archive
  and pair 1 put the first chunk cost at 30 to 100 ms; the bounded run ahead in that
  regime; and any SM level reading (no GPU metrics were sampled).

## 7. Missing from the archive, flagged

- The cause of the half speed regime. Nothing in the archive explains a device that
  replays the same graphs three times slower at full clocks. The launch environment of the
  shell is not recorded (`nsys_start.txt` is empty, no `env` dump), other CUDA contexts on
  GPU 0 during the runs are not recorded (`nvidia-smi --query-compute-apps` was not
  sampled), and nothing was run on the E4 head from the root checkout to separate the
  worktree layout from the box's state. Until this is explained no session compares with
  E4, and the E4 regime mitigation experiments cannot run.
- GPU metrics (GR active, SM active, SM occupancy): not sampled, as agreed.
- The exact Nsight command line (only `--cuda-flush-interval=1000` is mentioned).
- Early pass 1's stall has no log line; the first c16 batch's warm up is invisible.

## 8. Next

1. Regime first, on the box, in this order: (a) from a fresh login shell, `env | grep -i
   "cuda\|nsys\|cupti\|preload\|torch\|omp"`; (b) during a c16 run, `nvidia-smi
   --query-compute-apps=pid,process_name,used_memory --format=csv` for all GPUs and
   `ps -eo pid,comm,args | grep -i "mps\|dcgm\|nsys"`; (c) a graph microbenchmark: capture
   a CUDA graph of 1000 tiny kernels on GPU 0, replay it 500 times, report ms per replay
   (about 2 to 3 ms on an idle H100; 8 ms or more means the device front end is slowed
   by something outside our process); (d) boot the E4 head 04b62c255 from the root checkout
   with the E4 command, streaming c16, one pass: 17 req/s means the tmp worktree layout is
   the cause, 9 means the box is.
2. Once a session reads 15 ms per 40 step log line again: repeat step 1 on both arms with
   `--gpu-metrics-devices` on the Nsight arm and `nvidia-smi dmon -s pucv` on the plain
   arms, then the bounded run ahead, then #2126 stacked on early ids (its stream
   synchronizations are the 0.7 ms x 3.5 per step above).
