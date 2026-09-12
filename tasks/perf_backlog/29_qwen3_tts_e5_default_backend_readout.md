# 29. E5 on the default backend: the lock is the mechanism, 2026-09-13

Archive `e5-default-backend-10e0aa1dc-compact.tar.gz`. Recomputed from the raw files;
the box's REPORT.md agrees with them. The Nsight SQLite files stayed on the box (compact
archive); the GPU metric means and the per thread readout it produced with
`nsys_threads.py` are reported as the box's numbers where I could not recompute them.

## 1. Provenance and regime

- Base 645b472cd on every arm; control, early ids (patch SHA 7afb15aa), bounded run
  ahead (early ids plus the next step predictor wait), combined (early ids plus the
  #2126 patches, six files, +99 -26). Plain server command on every boot, no environment
  variable. GPU 0, GPUs 1 to 3 idle before every boot, 1980 MHz, no power or thermal
  violation in the 1 s dmon logs.
- Regime restored: 0.60 to 0.71 s per 40 step decode log line, 830 to 940 tokens per
  second at 15 or more rows. Control reads 15.9 req/s, which matches the clean main
  numbers we had (16.0 to 16.7 on the seeded pair and the followup), and early ids reads
  17.7, which matches pair 2. The Nsight boots ran slower (0.71 to 0.78 s per 40 steps),
  as profiled boots do.
- Runaways: early ids pass 1, the control Nsight window client and the early Nsight
  preconditioning pass carry a 163.84 s output; every quoted pass below is clean.

## 2. Client level, pass 2, unprofiled

| pass 2 | control | early ids | delta | bounded | combined (+#2126) |
| --- | ---: | ---: | ---: | ---: | ---: |
| req/s | 15.90 | 17.67 | +11.1% | 17.21 | 17.71 |
| TTFC mean / p50 | 124.7 / 112.5 ms | 170.8 / 158.2 ms | +46 / +46 ms | 182.1 / 166.1 | 185.9 / 174.9 |
| TTFC p99 | 360 ms | 430 ms | +70 ms | 413 | 498 |
| inter chunk mean | 108.7 ms | 90.5 ms | -16.7% | 92.2 | 88.4 |
| latency mean / p99 | 1.001 / 1.695 s | 0.901 / 1.511 s | -10.0% / -10.9% | 0.924 / 1.510 | 0.899 / 1.440 |

This is the production number for #2123's streaming cost in the default layout: +37
percent on the first chunk mean, with +11 percent throughput, -17 percent inter chunk and
-10 percent request latency. The bounded run ahead costs 2.6 percent throughput and
gains nothing on the first chunk. #2126 on top of early ids adds nothing to throughput
(+0.25 percent) and adds 15 ms to the first chunk mean and 68 ms to its p99.

## 3. Stage level, pass 2 event windows (p50 ms)

| segment | control | early ids | bounded | combined |
| --- | ---: | ---: | ---: | ---: |
| preprocessing dispatch to complete | 37.3 | 70.2 | 71.4 | 76.0 |
| engine input to build start | 7.0 | 5.2 | 6.0 | 5.0 |
| build end to queue enter | 11.6 | 9.2 | 9.0 | 7.8 |
| queue enter to prefill start | 1.1 | 2.6 | 2.6 | 3.0 |
| prefill | 21.3 | 23.6 | 22.9 | 24.4 |
| vocoder first code to first audio | 29.9 | 57.9 | 60.2 | 62.3 |
| admission to first audio | 114.2 | 172.8 | 181.0 | 187.7 |
| first audio at zero requests ahead | 27.4 (n 191) | 45.0 (n 113) | 45.5 (n 127) | 52.5 (n 92) |
| first audio at one request ahead | 34.1 (n 75) | 51.7 (n 100) | 53.9 (n 94) | 55.4 (n 95) |
| code chunks received before first audio | 2 | 4 | 4 | 5 |

The talker path is a wash (its four segments sum to 41.0 against 40.6 ms). The first
chunk cost is preprocessing (+33 ms) and the vocoder's first decode (+28 ms, of which
+18 ms on a lone bootstrap and the rest a queue that now holds one to three requests
where control held zero or one). Combined is the worst on every sibling segment.

## 4. GPU level

- dmon at 1 s during traffic, unprofiled (GR active proxy): control mean 35 percent,
  p50 16, p90 78; early ids mean 40, p50 15, p90 90; memory controller 10 to 12 percent;
  power 200 to 217 W mean. The device is idle most of each second and saturated in
  bursts; early ids raises the bursts, not the mean.
- Nsight GPU metrics at 20 kHz over the 20 s windows (the box's extraction): GR active
  50.3 percent control, 51.2 early ids; SM active 26.1 and 25.5 percent. Under the
  profiler the engine has a kernel resident half the time and a quarter of the SMs are
  active on average, the same on both arms.
- The sibling GPU side improved on early ids: initial worker kernel execution 11.0
  against 11.3 ms per bootstrap, its launch to start mean 386 to 207 us, its decode done
  wait 1.58 to 0.64 ms, the follow up workers' decode done wait 0.28 to 0.20 ms.

So at 17 req/s the GPU is not where the first chunk cost is.

## 5. Thread level, Nsight windows (the box's `nsys_threads.py` output, 20 s each)

| thread | metric | control | early ids |
| --- | --- | ---: | ---: |
| talker scheduler | steps in window | 1156 | 1130 |
| | ids wait, cudaEventSynchronize | 5.86 ms x 1262 | 0.93 ms x 1224 |
| | cudaStreamSynchronize | 8 us x 4176 | 102 us x 3776 |
| | GIL hold / wait | 5.59 / 6.10 s | 5.80 / 9.51 s |
| | GIL acquisitions with a wait | 195k | 239k |
| initial worker | bootstraps (decode done syncs) | 231 | 214 |
| | GIL wait per bootstrap | 20.3 ms | 48.2 ms |
| | GIL hold per bootstrap | 7.8 ms | 11.0 ms |
| | waits per bootstrap, mean wait | 585, 35 us | 997, 48 us |
| | GPU execution per bootstrap | 11.0 ms | 11.3 ms |
| preprocessing workers (8) | GIL wait per request | 19.7 ms | 32.4 ms |
| reference encoder batcher | GIL wait | 3.80 s | 6.55 s |
| follow up workers (2) | GIL wait | 1.02 / 1.05 s | 1.80 / 1.80 s |
| whole process | GIL hold, of 20 s | 13.8 s | 15.7 s |
| | GIL wait summed over threads | 25.7 s | 44.8 s |

Reading:

- On control the scheduler parks 5.9 ms of every 17.5 ms step in the ids wait, which
  covers the predictor and holds no lock. That is a third of the step in which every
  sibling runs its Python uncontended. On early ids the park is 0.9 ms of a 15 ms step
  plus 3.3 stream synchronizations of 0.1 ms, six percent.
- The lock is near saturation: it is held 69 percent of the window on control and 79
  percent on early ids, by eighteen threads. The scheduler's hold barely changes (5.6
  to 5.8 s); the extra 1.9 s of holding is the siblings themselves, doing the same work
  slower because each of their Python steps now waits. Waits per bootstrap go from 585
  to 997 and the mean wait from 35 to 48 us: more handoffs and slower handoffs.
- Per unit of work: +28 ms of lock waiting per bootstrap against a lone bootstrap that
  got 18 ms slower, and +13 ms per request across the preprocessing workers against a
  stage that got 33 ms slower (the rest is queueing in the eight worker pool and the
  batcher, which waits 72 percent more).
- Why E4a's shorter switch interval did nothing: the scheduler releases the lock at
  every torch call, so no holder runs pure Python for 5 ms; the cost is the number of
  handoffs and the wake up latency of each, which the interval does not touch.
- Why the bounded run ahead does nothing: its wait sits after the host tail, and at
  17 req/s the host tail (about 10 ms) outlives the 4 ms predictor, so the wait returns
  at once and the scheduler never parks. A park that helps has to sit where control's
  was, before the tail, and that is the overlap itself.
- Why #2126 makes it worse: the 3.3 stream synchronizations of 0.1 ms per step were the
  last lock free moments the scheduler left; removing them takes them from the
  siblings. Its throughput gain in this layout is nil.

## 6. Conclusion

In the default single process layout at c16, the decode step overlap moves the
scheduler thread from parking a third of each step to parking six percent of it, and the
three sibling stages that share the interpreter lock with it pay for their first chunk
work in lock waits: +37 percent first chunk mean, +70 ms at p99, for +11 percent
throughput, -17 percent inter chunk and -10 percent request latency. GPU execution,
dispatch and readiness waits of the siblings all improve. The fix is not on the device
and not in the scheduler's ordering, it is in how much Python the sibling paths run per
request under one lock, or in which process they run it.

Consequences for the PRs: #2123 cannot merge into the default layout as a pure win; it
is a throughput and steady state latency win with a first chunk cost that the deployment
has to accept or remove. #2126 adds no throughput at 17 req/s in this layout and costs
first chunk latency; it should not merge on these numbers.

## 7. Next

Candidates, each one measured pair on this regime, event recorder and dmon on:

1. Siblings in their own process on a second GPU (preprocessing and vocoder on GPU 1,
   the talker on GPU 0): the interpreter lock and the device both stop being shared.
   This is the deployment shaped answer and the first to measure.
2. Fewer Python steps per bootstrap: the reference prefixed bootstrap through a captured
   graph (pad the reference prefix to a few buckets), and the speaker and reference
   encoders through graphs; every launch removed is a handoff removed.
3. Only if 1 and 2 are out of reach: make the overlap conditional on the layout, off
   when the vocoder shares the process, and measure that #2123 then reads as main.

GR and SM rates for non streaming c16 are still unmeasured; add `--gpu-metrics-devices`
to the next non streaming census.
