# 24. Why the decode step overlap raises the first chunk latency, 2026-09-12

Written from the single server streaming pairs (archive
`qwen3-tts-runbook22-streaming-pairs-898dc3234-20260912.tar.gz`) and the per request event
files the c16 census windows recorded on 2026-09-11 (slice A archive, `events/` under
`A/census_valid/census_c16` and `B/census/census_c16`, 192 requests each, main `b2cc93b0a`
against slice A `51c5bb064`, same box, same day, no Python stacks in either).

## 1. What the streaming pairs measured

One independent server per arm, c16, two passes, second pass:

| pair | TTFC mean / p50 | inter chunk mean | latency mean | req/s | condition |
| --- | --- | ---: | ---: | ---: | --- |
| main | 0.146 / 0.136 s | 0.119 s | 1.106 s | 14.39 | GPU 1 tenant 36% |
| slice A (pair 1) | 0.249 / 0.229 s | 0.101 s | 1.057 s | 15.06 | GPU 1 tenant 43%, pass 1 sick (9.5 req/s, 20 gaps) |
| slice A (pair 2) | 0.178 / 0.166 s | 0.090 s | 0.902 s | 17.64 | GPUs 1 to 3 idle |
| slice A + C + B | 0.189 / 0.175 s | 0.088 s | 0.897 s | 17.74 | GPUs 1 to 3 idle |

Pair 1 is not a usable main against slice A comparison: both boots ran under a tenant on
GPU 1 and the slice A boot was sick in pass 1. Pair 2 is clean and says slices C and B move
streaming by nothing: TTFC +6 percent, inside two boots of the same code (0.178 against
0.249). The signal that matters is slice A against main, and the clean version of it is the
c16 census events below, not pair 1.

## 2. Where the time goes, per request, per stage

The single process layout (config.py:54-85) runs preprocessing, the vocoder and the talker
engine in one process with one interpreter lock. The c16 census windows recorded every
request's path through the three stages. p50 over 192 requests:

| segment | main | slice A | delta |
| --- | ---: | ---: | ---: |
| preprocessing stage, dispatch to complete | 160.5 ms | 217.2 ms | +57 ms |
| engine request build | 0.8 ms | 0.9 ms | |
| queue enter to prefill start | 1.5 ms | 3.7 ms | +2.2 ms |
| prefill | 32.7 ms | 31.0 ms | |
| talker decode, prefill end to model path end | 761 ms | 690 ms | −71 ms |
| vocoder stage, dispatch to complete | 222 ms | 281 ms | +59 ms |
| admission to terminal | 1326 ms | 1322 ms | 0 |

The talker got 9 percent faster and the two stages sharing its process got 35 and 27
percent slower, and the request as a whole did not move. Streaming feels this on the first
chunk, which is preprocessing plus prefill plus a few frames plus the vocoder's first
window; steady state chunks come from the faster talker, which is why inter chunk and
request latency improve while TTFC worsens.

Batching dynamics are not it. Binned by how many requests overlapped in the stage:

| overlapping requests | preprocessing main | preprocessing slice A | vocoder main | vocoder slice A |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 153 ms | 210 ms | 167 ms | 211 ms |
| 6 | 154 ms | 217 ms | 222 ms | 230 ms |
| 8 | 210 ms | 283 ms | 274 ms | 308 ms |

Slower at equal concurrency: the stages compete for something the talker now uses more.

## 3. The mechanism

Main's scheduler thread blocked in `event.synchronize` for 3.9 ms of every 8.6 ms step at c1
and 4.1 of 9.5 ms at c16 rows 16 (the S3 era E2 host tail reports, 43 to 45 percent of the
step). A thread blocked in a CUDA wait holds no interpreter lock, so that was the CPU budget
of every other Python thread in the process: the preprocessing executor, the request build
pool, the vocoder's scheduler and decode workers, the asyncio thread that routes chunks.

Slice A removes the wait by design. The scheduler thread now blocks 0.8 ms of a 6.7 ms full
step (12 percent) and 0.04 ms of a churn step (readout 20 section 3, readout 23 section 3),
and holds the lock for the rest. CPython lets a waiting thread take the lock only after the
holder has run for the switch interval, 5 ms by default, so every small piece of Python the
sibling stages need now waits up to 5 ms, many times per request. The two worker router
layout, where the vocoder lives in its own process, showed a TTFC cost of +13 percent against
+22 to +70 percent in the single process layout with the same code; the layout difference is
the lock, the GPU is the same in both.

GPU time is the second candidate: the talker's stream is busy 85 percent of the step after
slice A against 62 percent before. The vocoder's decode streams already run at a resolved
priority (`_decode_stream_priority`), the preprocessing stream does not. The experiments
below tell the two apart; neither is assumed.

## 4. Experiments, E4

Each is slice A (`04b62c255`) against itself with one change, one independent server at
c16, two streaming passes, and the event recorder on for a window of 200 requests in pass 2
(`/start_profile` with `event_dir` and `enable_torch` false), so the segments of section 2
are read directly for streaming requests, including `scheduler_first_emit` and the vocoder's
`stage_first_stream_chunk_sent`.

- E4a, the lock. `sys.setswitchinterval(0.0005)` in the pipeline process, an uncommitted
  one line at the top of `stage_workers._run_process` for the experiment. If preprocessing
  and vocoder segments and TTFC return to main's, the lock is the mechanism.
- E4b, the GPU. The preprocessing stream and the vocoder's decode streams created with the
  highest priority (`torch.cuda.Stream(priority=-1)`), an uncommitted change at
  request_builders.py:200 and streaming_vocoder.py:799-808. If the segments return, the
  GPU is the mechanism.
- E4c, the layout. Preprocessing and the vocoder in their own process, the topology the
  config already supports (`preprocessing_in_own_process`, the CI streaming layout's
  separate vocoder process). This is the structural answer if E4a says the lock: the
  talker's scheduler thread is CPU bound by design after the overlap and cannot share an
  interpreter with latency sensitive Python.

Read: the per segment table of section 2 for each experiment, then TTFC, inter chunk,
latency, throughput against main and slice A from the same session.

## 5. Consequence for the two PRs

The decode step overlap (#2123) is correct and its talker gain is real, but in the default
single process layout it shifts about as much time into the sibling stages as it removes,
and streaming pays it on the first chunk. It must not merge without the mitigation E4
selects, measured in the same layout. #2126 is bit exact, removes the two churn step stalls,
and moves streaming by nothing on its own; it follows #2123.
