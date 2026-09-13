# 36. Bootstrap windows readout, session 2, 2026-09-13

Archive `bw-session-results-no-wavs.tar.gz` (second upload, 400 MB). A 3060470a8 from
`tmp/main`, B f73586369 from `tmp/bw`, both booted with `python -m sglang_omni.cli serve`,
import path archived under the right worktree for all six boots, both B boots pass the
window gate (`window_frames=(1, 2, 4, 8, 16, 32, 64)`, 28 captured keys). GPU 1 for every
boot, clear before each. No `recompile_limit` or `disabled the` line anywhere. Decode log
gap 0.38 to 0.63 s per 40 steps. Zero failed requests in any pass. Step 1: 36, 22, 242
tests pass. s2b pass 1 holds a 163.84 s runaway and is not quoted; every pass 2 is clean.

## 1. Verdict

The slice removes the origin it was written for and moves every client read the right
way in both launches. Item 1 of doc 32 is done. The #2123 gate is not closed by it alone:
on early ids the first chunk mean lands 10.8 ms above main, and the remaining gap is the
preprocessing segment, doc 32 item 3.

## 2. The default launch, step 3, pass 2, streaming c16

| read | A main | B windows | delta |
| --- | ---: | ---: | ---: |
| req/s | 15.43 | 15.78 | +2.3 percent |
| audio s/s | 63.8 | 65.6 | +2.8 percent |
| TTFC mean ms | 129.7 | 117.0 | -12.7 |
| TTFC p50 ms | 117.3 | 109.8 | -7.5 |
| TTFC p99 ms | 326.9 | 301.7 | -25.2 |
| inter chunk mean ms | 112.2 | 110.3 | -1.9 |
| bootstrap segment, ahead 0 p50 ms | 28.1 | 26.1 | -2.0 |
| bootstrap segment, mean ms | 35.3 | 29.9 | -5.4 |
| bootstrap segment, p95 ms | 65.2 | 48.8 | -16.4 |
| preprocessing p50 ms | 39.4 | 36.3 | -3.1 |
| prefill p50 ms | 21.8 | 18.7 | -3.1 |
| dmon SM active during traffic, percent | 78.3 | 80.9 | +2.6 |
| launch to ready s | 96 | 90 | -6 |

Pass 1 req/s 15.77 against 16.10. The req/s delta sits inside the 4 percent band of doc 35;
the TTFC, p99 and bootstrap segment deltas do not. Quality on B pass 2: WER 1.05 percent,
similarity 71.53, both inside the bands. Seeded streaming c1: 10 of 1088 identical, not
gated (doc 35, the streaming path is not reproducible across boots on identical code).

## 3. Early ids, step 2, pass 2, streaming c16, the pair that decides #2123

| read | A main + early ids | B windows + early ids | delta |
| --- | ---: | ---: | ---: |
| req/s | 15.55 | 17.94 | +15.4 percent |
| audio s/s | 64.5 | 74.4 | +15.2 percent |
| TTFC mean ms | 234.0 | 140.5 | -93.5 |
| TTFC p50 ms | 215.8 | 132.1 | -83.7 |
| TTFC p99 ms | 508.4 | 367.9 | -140.5 |
| inter chunk mean ms | 98.0 | 92.8 | -5.2 |
| bootstrap segment, ahead 0 p50 ms | 63.2 | 28.4 | -34.8 |
| bootstrap segment, mean ms | 89.6 | 34.2 | -55.4 |
| chunks received before first audio, p50 | 5 | 3 | -2 |
| bootstrap time summed over the window | 1.40 x | 0.62 x | |
| preprocessing p50 ms | 85.6 | 57.8 | -27.8 |
| prefill p50 ms | 26.8 | 16.8 | -10.0 |
| prefill p50 with 1 bootstrap overlapping, ms | 28.8 | 13.2 | -15.6 |
| talker cadence p50 ms | 11.4 | 10.8 | -0.6 |
| dmon SM active during traffic, percent | 73.6 | 83.8 | +10.2 |

Against the main control of step 3 (A, 129.7 ms TTFC mean, 15.43 req/s): B on early ids
is 10.8 ms above on the mean, 14.8 ms on the p50, at 16 percent more throughput. The gate
of doc 31 was within 10 ms at early ids throughput; the throughput half is exceeded and
the first chunk half misses by about a millisecond. What remains is the preprocessing
segment, 57.8 against 39.4 ms p50, the reference encoder and preprocessing launches of
doc 32 item 3.

## 4. The origin check, Nsight 20 s windows on the early ids arms, c16

Per thread from `nsys_threads.py`. The initial worker is the thread whose launches fall
and whose graph launches appear.

| thread | A launches | A graph launches | B launches | B graph launches |
| --- | ---: | ---: | ---: | ---: |
| initial worker | 274,594 | 0 | 16,423 | 874 |
| preprocessing worker | 186,477 | 0 | 209,641 | 0 |
| scheduler (talker) | 153,128 | 2,585 | 163,007 | 2,747 |
| follow-up workers (2) | 9,633 and 9,548 | 766 and 761 | 9,933 and 9,868 | 776 and 772 |
| codec pool threads (7) | 7,430 to 10,145 each | 0 | 7,745 to 10,421 each | 0 |
| process total | 701,720 | 4,112 | 482,310 | 5,169 |
| per request | 2,157 | | 1,387 | |

Requests in the window: about 325 on A, 348 on B. The initial worker went from about 845
launches per request to about 47 plus 2.5 graph replays, which matches the runner's own
counters over the whole boot (window replays 5,525 for 2,176 requests on s2b, 9,632 for
3,264 on s3b, cold misses 0 on both B boots against 1,325 and 2,688 on A). The process
launch count per request fell 36 percent. The lock wait columns are zero on both arms:
this build has no NVTX ranges on the interpreter lock, so the wait side of the origin was
not measured; the effect on the other threads shows instead in their segments (prefill
under one overlapping bootstrap 28.8 to 13.2 ms, preprocessing 85.6 to 57.8 ms).

GPU metrics from the same windows: GR active 72.7 to 79.0 percent, SMs active 39.2 to
44.7 percent, warps in flight 16.7 to 19.0 per cycle, clock 1985 MHz on both. Streaming
c16 had no GR or SM read before this session; these are its first.

## 5. Non streaming

Not measured and not expected to move. The window runner lives on the streaming
vocoder's initial worker and serves incremental bootstraps only; the non streaming path
decodes whole sequences through `chunked_decode` and does not touch the incremental
decoder or its runners. The precompile fix changes when a shape is traced, not what the
warm runner replays, so it is bit exact by construction on both paths.

## 6. Cost

Window runner footprint 2,108 MB at buckets 1, 2, 4, 8 (doc 34 measured 1.68 GB at
buckets 1 and 4). Boot time fell 6 to 11 s with the branch, the precompile fix returning
more than the 28 eager captures cost.

## 7. Next

1. Open the precompile PR (b2c9abe13) and the window PR (f73586369) with the step 3 pair
   as the census and the step 2 pair as the early ids evidence.
2. Doc 32 item 3, the preprocessing segment: the reference encoder and preprocessing
   launches, 186 to 210 thousand per 20 s on the preprocessing thread, are now the largest
   eager launcher in the process and the remaining gap to the #2123 gate.
