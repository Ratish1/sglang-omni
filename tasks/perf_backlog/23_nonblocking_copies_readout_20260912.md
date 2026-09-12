# 23. Slices C and B readout, the two blocking copies, 2026-09-12

Archive `qwen3-tts-nonblocking-copies-83aa2c84c-20260912-compact.tar.gz`. B is `83aa2c84c`, the
two slices on slice A at `04b62c255`. A is slice A's boots of the same day (readout 20
section 8): the follow up c16 boot and its three streaming passes, both with GPU 1 idle and
GPUs 2 and 3 at 98 percent, which is the load every B boot ran under, so the pairs hold.

## 1. The mechanics, and the sglang reference for them

Slice B is the shape of sglang's own overlap result path, `sglang/srt/managers/utils.py:31-41`
`_async_d2h`: `torch.empty(shape, dtype, pin_memory=True)`, `copy_(src, non_blocking=True)`,
and a completion event the consumer waits (`batch_result_processor.py:249, 402, 827, 871,
883` wait `result.copy_done`, recorded at `scheduler.py:4108, 4168, 4313`). Ours records the
event on the request data and the stage runtime waits it off the scheduler thread. sglang adds
`record_stream` because its copy runs on a second stream; ours runs on the producer's stream,
where the caching allocator's stream ordered reuse already covers the source.

Slice C is the shape of sglang's FutureMap (`overlap_utils.py:194-213`): pinned host buffers
allocated once, `copy_(non_blocking=True)` into ring slots and one `copy_done` event per slot,
`query()` before reuse (`:240`). Ours has two slots and waits the slot's event before rewriting
it.

Both are bit exact by construction: the same values reach the same device rows at the same
stream position, and the same CPU tensor reaches every reader after the same completion.

## 2. Suites

New files 8 of 9, the Qwen3-TTS directory 471 of 472, the pipeline files 109 of 111. The
three failures were the tests, not the code, fixed in `perf/qwen3-tts-nonblocking-copies`
after the run:

- two scheduler tests build request data as a bare class or SimpleNamespace without the
  new `result_ready_event` field, the scheduler reads it as a real attribute; the fakes got
  the field;
- `test_cuda_codes_leave_as_a_pinned_copy_that_the_event_completes` enqueued the stream sleep
  before the first pinned allocation of its size class. torch's caching host allocator fills
  a class with `cudaHostAlloc`, which synchronizes the device, so the builder returned after
  the sleep. The test now fills the class once first, as the first request of that size does
  in serving. This is the same first fill cost sglang's `_async_d2h` carries; in the census
  the builder's p90 is 0.16 ms, so the classes are warm after the first requests.

## 3. Mechanism gate, passed

Census boot with stacks on B, `perfkit.py` with the row label fix.

| p50 per step | slice A (readout 20) | B |
| --- | ---: | ---: |
| c1, 1 row, wall / idle | 6.25 / 0.98 ms | 6.01 / 0.83 ms |
| c16, 16 rows, wall / idle | 6.78 / 1.04 ms | 6.48 / 0.81 ms |
| c16, 8 rows, wall / idle | 8.61 / 2.97 ms (census), 14.6 / 8.9 (E2 boot) | 8.89 / 3.30 ms |
| kernels per replay | 1062 | 982 (S3 is in) |

The two frames at rows 8, p90 self time: `prepare_decode_buffers` 3.15 to 0.08 ms,
`apply_sglang_qwen3_tts_result` 2.99 to 0.16 ms. The c1 and 16 row steps gained S3's 80
kernels (about 0.25 ms) and nothing else, as expected.

One correction to readout 20 section 3: the 14.6 ms churn step there came from the E2 boot,
which runs the profiler with Python stacks, and that profiler slows the host enough to put it
behind the device on churn steps, which is when the two pageable copies waited on the
predictor. Slice A's own unprofiled census boot had the rows 8 step at 8.61 ms. So the stalls
were real, up to 3 ms at p90 under profiling, but smaller in serving, and the throughput
they cost is accordingly smaller than the E2 picture suggested.

## 4. c16, one boot per arm, same load

| | slice A | B | delta |
| --- | ---: | ---: | ---: |
| req/s | 17.034 | 17.360 | +1.9% |
| audio s/s | 70.433 | 71.961 | +2.2% |
| median, p95, p99 s | 0.914, 1.311, 1.513 | 0.906, 1.290, 1.438 | −0.9, −1.6, −5.0% |
| RTF mean | 0.2333 | 0.2277 | −2.4% |
| WER, similarity | | 0.988%, 71.316 | recorded, in band |

Inside the 2 percent boot spread on req/s and median, p99 −5 percent. Against upstream main
of the same session (16.03 req/s) the stack reads +8.3 percent.

## 5. Streaming, three passes per arm

Per request records; main E3, slice A follow up, B this archive, all two workers on GPUs 0
and 1.

| pass | TTFC mean / p99 s | inter chunk mean / p99 s | latency mean / p99 s | req/s | gaps over 200 ms |
| --- | --- | --- | --- | ---: | ---: |
| main 1, 2, 3 | 0.182/0.78, 0.133/0.32, 0.126/0.30 | 0.078/0.17, 0.081/0.19, 0.081/0.19 | 0.81/1.42, 0.79/1.25, 0.78/1.28 | 19.5, 20.3, 20.4 | 0, 0, 0 |
| slice A 1, 2, 3 | 0.208/0.93, 0.150/0.38, 0.133/0.31 | 0.072/0.19, 0.076/0.19, 0.074/0.19 | 0.81/1.40, 0.78/1.23, 0.73/1.17 | 19.6, 20.4, 21.6 | 1, 5, 0 |
| B 1, 2, 3 | 0.210/0.92, 0.156/0.47, 0.126/0.33 | 0.070/0.16, 0.069/0.20, 0.071/0.16 | 0.79/1.40, 0.72/1.22, 0.70/1.13 | 20.1, 22.2, 22.8 | 7, 6, 0 |

Pass 3, the warm pass, is the serving picture: B's TTFC mean equals main's (0.126 s), the
slice A cost of readout 20 section 8 is gone, inter chunk is −13 percent against main and
−5 against slice A, request latency −11 percent against main, throughput +12 percent
against main and +6 against slice A, continuity 100 percent, 3264 of 3264 in every arm.

Passes 1 and 2 carry playback gaps over 200 ms that pass 3 never has, on all three arms in
proportion to the talker's speed: main 0, slice A 1 and 5, B 7 and 6, in clusters of adjacent
request indexes, that is one worker pausing about 300 ms with several requests in flight. No
lazy predictor graph capture happened in any worker (0 capture lines in six worker logs).
The stall is on the vocoder side and warm up shaped: the codec decoder builds a cuDNN plan
per new audio length (backlog T40), and new lengths run out during the second pass. A faster
talker fills the vocoder's queue deeper, so the same pause crosses the 200 ms line for more
requests. Owned by T40, not by these slices; the test that pins it is a fourth pass after a
vocoder restart, which should reproduce the pass 1 gaps on any arm.

## 6. Identity gate, owed

The box compared B's seeded c1 (warmup 1) against slice A's archived seeded c1, which ran
with warmup 0, and got 0 of 1088: the benchmark's per request seeds follow request order, so
the warmup request shifts every seed. The only valid reference is an A boot at `04b62c255`
with `--seed 1234 --warmup 1`, one boot, against `seeded_c1/B/bench/wav_sha256.json` of this
archive. Slice A's own identity (readout 20) was two warmup 0 boots, which is why it held.

## 7. Memory

Streaming: engines 69.4 and 71.0 GB, vocoders 4.1 and 3.8 GB (slice A: 71.0 and 69.0, 3.6 and
3.8). Single process c16 peak 76977 MiB against slice A's 79555 on the same readiness level.

## 9. The finish test, settled

Follow up archive `qwen3-tts-nonblocking-copies-7bd3b86db-followup-20260912.tar.gz`: the
identity gate passed with the same warmup on both arms, 1088 of 1088, A at `04b62c255`
seeded c1 warmup 1 against B's hashes. The two scheduler tests pass with their fakes fixed.
The finish timing test still failed with the pinned class primed, so the probe of runbook
22 section 7 ran on the box: with the stream busy, `stack` 0.04 ms, `cat` 0.07, the pinned
allocation 0.04, the D2H copy 0.07, the event 0.01 and its record 0.04, the stream busy
after every one and the event not done. The builder's calls do not synchronize. The
difference between the probe and the test was that the probe had launched `stack` once
before the sleep: the test's first `stack` and `cat` of the process came after the sleep,
and a kernel's first launch under lazy module loading synchronizes the device. Serving
pays that once on the warmup request. The test now runs the builder once on a warmup
request before the timed call (`9900cb82e`). The mirror test of the restage sources'
shapes and dtypes went under the admission rule (`2a10d594f`).

## 8. What is left on this branch

1. The A seeded c1 warmup 1 boot for the identity gate.
2. The fixed tests rerun on the box (three files).
3. Then the PR: mechanism, the sglang reference, the rows 8 frames as the census table,
   the c16 table and the pass 3 streaming table with main and slice A as the two A columns.
