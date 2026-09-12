# 32. P1 readout: the reference prefix prime did not work, and the origin plan

Archive `p1-prefix-prime-7ceaf8f2c-compact.tar.gz`, session of 2026-09-12 evening on the
H100 box. Every number below was recomputed from the archived event files, speed results,
dmon logs and serve logs, not taken from the box's REPORT.md. The box's anatomy counted the
prefix message as the first code chunk, which put the prime inside the vocoder segment and
made B look worse than it is; the script is fixed (section 3) and the numbers here use the
first generated frame.

## 1. Verdict

P1 does what it was designed to do at the point it was aimed at, and loses overall.

- When the initial worker is idle, the first generated frame reaches audio in 16 ms on
  B against 30 ms on A (main pair), 41 against 53 (early ids pair). The frame decode
  replays the cold graph: 897 replays on B against 0 on A.
- Everything around it got worse. The prefill of the same request slows by 4 to 7 ms at
  p50 because the prime's eager decode now overlaps it. The initial worker carries two
  jobs per request instead of one, so more requests find other bootstraps in flight and
  the mean and tail of the segment rise. Throughput drops 4 percent on both pairs.
- Client level: TTFC neutral on main (180.7 against 179.8 ms in pass 1), worse on early
  ids (186.9 against 205.6 ms, p99 480 against 585), req/s 17.09 against 16.34 on early
  ids and 15.38 against 14.73 on main pass 1. The doc 31 gate (B within 10 ms of main's
  control at early ids' throughput) fails by 25 ms with less throughput.

P1 is parked on `perf/qwen3-tts-prefix-prime` at ac29b71b7 (runtime 2287c6e26 and
651de81e3, test fixes 7ceaf8f2c and ac29b71b7). No PR. Section 8 item 1 makes the
bootstrap a graph replay, which is what the prime was trying to reach, so P1 is superseded
by it rather than re-measured.

## 2. Provenance

- A: upstream main d90d71c37. B: 7ceaf8f2c (the runtime commits plus two test fixes).
  Early ids arms carry `early_ids.patch` on model_runner.py only.
- Plain server command, port only, sglang 0.5.19, torch 2.13.0+cu130, physical GPU 0.
  GPUs 1 and 3 idle on every boot. GPU 2 held an unrelated CosyVoice server the whole
  session (80 GB, 62 to 71 percent utilization). A first main A boot overlapped a CosyVoice
  restart on GPU 0 and was discarded before traffic.
- First boot decode log gap 0.71 s per 40 steps (n 97), the main regime of the memory
  note (0.70 s at 15.9 req/s), so no forced backend.
- Full unit files on B: test_pipeline 243 passed 1 failed (the engine builder fake still
  returned three adapters, fixed in ac29b71b7), test_scheduler 82 passed, codec 36 passed.
- Main A pass 2 has a 163.84 s runaway (2048 frames), so its req/s 13.65 is not
  quotable; its event window is still valid for the anatomy (the runaway holds one slot).

## 3. Client level

Streaming c16, 1088 requests, warmup 1, no seed, zero failures everywhere.

| pair | arm, pass | req/s | TTFC mean / p99 ms | inter chunk mean ms | latency mean s | dmon GR active mean, traffic |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| main | A pass 1 | 15.38 | 180.7 / 873 | 106.1 | 1.034 | 67.2 |
| main | A pass 2 | runaway | 129.2 / 497 | 109.6 | 1.038 | 68.4 |
| main | B pass 1 | 14.73 | 179.8 / 975 | 111.3 | 1.081 | 66.8 |
| main | B pass 2 | 14.85 | 132.3 / 452 | 116.8 | 1.072 | 65.1 |
| early ids | A pass 1 | 17.06 | 273.9 / 907 | 81.7 | 0.934 | 78.7 |
| early ids | A pass 2 | 17.09 | 186.9 / 480 | 92.3 | 0.932 | 74.4 |
| early ids | B pass 1 | 16.51 | 300.7 / 1103 | 82.4 | 0.965 | 71.8 |
| early ids | B pass 2 | 16.34 | 205.6 / 585 | 95.4 | 0.974 | 72.7 |

Pass 1 TTFC is 50 to 90 ms above pass 2 on every arm (cold caches after one warmup
request), so pairs read pass against pass. Quality on main B pass 2: WER 0.91 percent,
similarity 71.22, in band. Seeded c1 non streaming: A 2.434 req/s, B 2.427, 1088 of 1088
WAVs identical, which says nothing about the prime (non streaming never sends a prefix).

## 4. The anatomy from the events, pass 2, first generated frame as the anchor

| read | main A | main B | early A | early B |
| --- | ---: | ---: | ---: | ---: |
| requests with first audio in the window | 328 | 225 | 217 | 222 |
| preprocessing dispatch to complete, p50 ms | 42.8 | 51.0 | 85.4 | 89.0 |
| build end to queue enter, p50 ms | 11.6 | 12.9 | 9.4 | 9.4 |
| prefill at batch size 1, p50 ms | 21.7 | 25.5 | 24.3 | 26.4 |
| prefill at batch size 2, p50 ms | 23.4 | 30.3 | 24.2 | 29.9 |
| prefix received to first frame received, p50 ms | | 29.4 | | 31.8 |
| first frame received to first audio sent, p50 / mean / p95 ms | 31.6 / 38.2 / 71.8 | 24.1 / 42.7 / 119.6 | 69.4 / 87.0 / 209.6 | 65.3 / 92.3 / 241.6 |
| same, no other bootstrap in flight (ahead 0), p50 ms | 29.7 | 16.4 | 52.5 | 41.2 |
| share of requests with ahead 0 | 62 % | 48 % | 28 % | 20 % |
| share with ahead 3 or more | 5.5 % | 16 % | 25 % | 36 % |
| generated chunks received before first audio, mean | 2.48 | 2.76 | 5.68 | 6.46 |
| bootstrap interval per request (job start to first audio), mean ms | 38.2 | 75.9 | 87.0 | 125.2 |
| bootstrap intervals summed, in units of the window | 0.59 | 1.10 | 1.43 | 2.07 |
| admission to first audio, p50 / mean ms | 121.8 / 136.4 | 134.2 / 164.7 | 213.5 / 235.8 | 206.1 / 253.1 |
| talker code chunk cadence, p50 / mean / p95 ms | 13.5 / 18.2 / 41.9 | 12.9 / 19.8 / 60.4 | 11.1 / 16.3 / 46.3 | 10.7 / 16.2 / 49.8 |

Job start is the first code chunk the vocoder receives for the request: the prefix on B,
the first frame on A. Ahead counts other requests whose job start to first audio interval
contains this request's first frame arrival.

Codec state at the last stats line of each serve log: cold graph replays 0 / 897 / 0 / 807
(A, B, A, B), eager bootstraps (`uncaptured_fresh_frames`) 1288 / 1060 / 1330 / 1295, no
arena exhaustion, no left context fallbacks, warm replays unchanged in rate.

## 5. The prefill under the prime

Prefill duration at batch size 1 binned by how many bootstrap intervals overlapped it:

| overlapping bootstraps | main A p50 ms | main B p50 ms | early A p50 ms | early B p50 ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 20.1 (n 100) | 21.0 (n 2) | 23.7 (n 37) | |
| 1 | 21.9 (n 42) | 27.6 (n 101) | 26.8 (n 27) | 31.2 (n 39) |
| 2 | 22.6 (n 22) | 21.1 (n 17) | 25.8 (n 19) | 28.2 (n 12) |

On B a request's own prime overlaps its prefill by construction (133 of 135 on main): the
prefix leaves at queue enter, the prefill starts 1.6 ms later, and an eager bootstrap
under this load takes longer than the 29 ms of lead the prefix has (doc 29: 11 ms of GPU
work, 8 to 11 ms holding the lock and 20 to 48 ms waiting for it). So the prime is still
running when the first frame arrives, the frame's graph replay queues behind it in the
serial initial worker (which is why ahead 0 reads 16 ms rather than a replay's few ms),
and the prefill on the scheduler thread runs against the prime's eager launches the whole
time. A's bootstraps overlap other requests' prefills by chance and cost them 2 ms; B's
prime overlaps its own and costs 6 ms. The same contention lifts the cadence tail (p95 42
to 60 ms on main).

## 6. Why the plan's order was wrong

Doc 30 read doc 29's lock holder table, took the two largest holders and shortened the
critical path of one of them. The cost it went after is not the codec decode on the
device (11 ms) but the eager launch path under a lock shared by eighteen threads (20 to 48
ms of waiting per bootstrap). P1 moved that eager work 30 ms earlier and added a second
job and a second commit; it removed no Python step from the process. In a shared
interpreter every Python step added anywhere is paid by every other thread, so a local win
inside the process is refunded through the lock. The same reasoning covers doc 30's P2 to
P4: they trim host work on one thread and are worth at most the handoffs they remove.

The origin is above all of these segments: three stages in one process, one interpreter
lock. Doc 29 measured the talker's own scheduler thread waiting for that lock 6.10 s of a
20 s window on control and 9.51 s on early ids, 5.3 and 8.4 ms of every 17.5 and 15 ms
step, and the initial vocoder worker waiting 20 and 48 ms per bootstrap, the preprocessing
workers 20 and 32 ms per request. Every segment we have measured since doc 24 is a slice
of that waiting. Two prior data points say the layout, not the device, carries it:

- E4c (doc 25): the vocoder in its own process, even time sliced against two other CUDA
  contexts, bootstrapped alone in 25.9 ms against 51.8 ms in the shared process.
- Doc 24 section 3: the two worker router layout with a separate vocoder process paid
  +13 percent TTFC for slice A against +22 to +70 percent in the shared process.

Those two layouts are not the fix. Separate processes on one GPU time slice their CUDA
contexts (E4c, throughput halved) unless an MPS daemon runs, and a second GPU is a
deployment choice. Neither can be the default launch, and the default launch is what has
to be fastest. They are measurements of what the lock costs, nothing more, and they are not
pursued. What can change in the default launch is the number of handoffs: one per eager
torch call, on every thread. P1 moved eager work and removed none of it. The plan below
removes it, path by path, largest first.

## 7. The origin in numbers: eager launches per request, per thread

What is in hand. Streaming c16, the E5 GPU0 Nsight thread summaries (doc 27's session,
forced torch backend; the sibling threads do not use the fused ops so their counts stand,
the talker's do not), 20 s windows:

| thread | launches in the window | per unit of work |
| --- | ---: | ---: |
| initial vocoder worker | 198k, 231 bootstraps | about 860 per bootstrap |
| reference encoder batcher | 137k | about 430 per request at 15.9 req/s |
| preprocessing workers, 8 | 47k | about 150 per request |
| follow up workers, 2 | 11k, 867 chunks | about 13 per chunk, 35 per request |

Non streaming c16 census on the default backend (main, `trace_A_main.json`, 23.7 s):
eager kernel launches per second on the talker's stream 7757 (the scheduler thread:
sampling, prefill, and the non streaming vocoder decode which runs on the caller's
stream), the encoder batcher's stream 3449, the preprocessing workers' stream 1777. At
15.9 req/s that is about 490, 220 and 110 eager launches per request.

Doc 29's per thread lock waits on the default backend order the same way: the initial
worker 20 to 48 ms per bootstrap, the preprocessing workers 20 to 32 ms per request, and
the scheduler thread 5.3 to 8.4 ms per step waiting behind all of them.

Validation task, step 0 of the next session: `nsys_threads.py` on the E5 default backend
SQLite (session 10e0aa1dc, on the box) so the streaming launch counts per thread are on
the default backend, with the scheduler's per step count split into decode steps and
prefills. The order below is expected to hold; the numbers replace the table.

## 8. The plan: fewer launches in the default launch, ordered by handoffs removed

Every item is one slice on upstream main, one pair in the default layout on the plain
command (doc 31 protocol), read by TTFC mean and p99, req/s, inter chunk, the anatomy
script, and one Nsight window per arm through `nsys_threads.py`: launches per request and
lock wait per thread. A slice passes only if the launches it targets are gone from the
census and the sibling lock waits fell with them; that is the check that it reached the
origin rather than a segment.

1. **The vocoder bootstrap through captured graphs.** Today a reference prefixed
   bootstrap is one eager decode of `ref_frames + initial_chunk_frames` frames, a width
   the cold runner never captured (`incremental_codec_cuda_graph_cold_frames` resolves to
   the initial chunk widths, streaming_vocoder.py:777-783), so `decode_slots` misses
   (incremental_codec_cuda_graph.py:464) and `_launch_async` runs the eager decoder on a
   gathered copy of the slot (streaming_vocoder.py:1764-1775): about 860 launches per
   request, 11 ms of GPU time, 20 to 48 ms of lock waiting. The change: the initial
   worker consumes that width in windows of captured widths against the same slot, each
   window one graph replay (`decode_slots` writes the arena directly, so no gather and
   scatter), the last window carrying the generated frames whose samples are emitted.
   The stateful decoder is partition invariant (the P1 test at rtol 2e-5), so the audio
   is the same up to graph replay numerics. The window widths are not a guess: a replay
   time per width microbenchmark on the box (the `graph_replay_bench.py` pattern against
   the incremental runner) picks the set whose total replay time for the corpus's
   reference lengths stays at or under the eager 11 ms, and the cold runner captures
   those widths at boot (capture memory is logged per key, 165 MB for the eight keys
   today). Expected: the initial worker's launches per bootstrap from about 860 to the
   number of windows times a few, the ahead 0 segment from 30 and 52 ms toward its GPU
   time, and every other thread's lock wait down by the handoffs removed. This supersedes
   P1 and doc 30's P1.
2. **The scheduler thread's eager launches per step.** First the census of what they are
   on the default backend (a kineto window of a few steps, ops by name on the scheduler
   thread: layer 0 sampling, predictor input assembly, per row stream output building,
   recorder work, prefill), then the reductions with the most launches first. Two are
   already known: doc 30's P2 and P4 (one pass over the batch rows instead of per row
   tensors, one stream message per step), and the Base prefill through the breakable
   prefill graph. CustomVoice already uses that graph; Base keeps the eager prefill only
   because its shapes were never measured (engine_builder.py:107-113). They are now:
   from main A's serve log, single request prefills are 1 / 34 / 68 / 114 tokens at
   p10 / p50 / p90 / max, batches up to 384, all inside the default ladder with the 1
   token bucket. The prefill runs on the scheduler thread, so its launches are what every
   sibling waits behind during the 22 to 28 ms it takes.
3. **The reference encoder batcher and the preprocessing workers through graphs.** About
   220 and 110 eager launches per request on the default backend census, on nine threads,
   and the preprocessing segment is the largest in the first chunk chain (43 ms on main,
   85 on early ids). Shapes vary with the reference length, so the same bucket and window
   reasoning as item 1 applies, after items 1 and 2 show what remains.

After each item the census is re-read and the order re-checked. #2123 (the decode step
overlap) is re-measured only once the handoff count is down, since its first chunk cost
is the siblings' lock waits (doc 29); #2126 after it.

## 9. Archive per boot

Head, import path, the full server command, gpus before, dmon log, both passes'
speed_results and client logs, the event files of pass 2, serve.log, the Nsight SQLite
export of the window and the `nsys_threads.py` JSON.
