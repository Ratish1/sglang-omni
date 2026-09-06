# Readout of the 2026-09-06 revalidation archives

Archives `qwen3-tts-revalidation-20260906-results.tar.gz` and
`qwen3-omni-chain-revalidation-20260906-results.tar.gz`. A is `7989a5ed2`, B is `0a88253c6`
in both. Every number below is read from the archive files named, and every code statement
from the pinned checkouts.

## 1. Why both arms are slower than the previous A/B

| Point | Previous A (91e9c3095) | New A (7989a5ed2) | Previous B (2c00eb688) | New B (0a88253c6) |
| --- | ---: | ---: | ---: | ---: |
| c1 qps | 2.246 | 2.224 | 2.323 | 2.285 |
| c16 qps | 15.038 | 14.287 | 15.611 | 15.259 |
| c16 latency mean s | 1.058 | 1.113 | 1.020 | 1.043 |
| c16 latency p99 s | 1.866 | 2.081 | 1.825 | 1.922 |

The code did not change between the two A arms in any file the run executes: main moved by
one commit, `7989a5ed2` (router media routing), and `git diff --stat 91e9c3095..7989a5ed2`
touches nothing under sglang_omni/models, scheduling, model_runner, pipeline, serve or config.
The bits did not change either: all 1088 c1 WAVs of the new A equal the previous A's hashes,
the new B equals the previous B, and A equals B (`c1_wav_sha256.json` of both archives).

The device work per step did not change. The census tables (`steps.md`) at 16 rows:

| 16 rows | step wall p50 ms | backbone busy | predictor busy | predictor wall | idle in step |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous A | 9.151 | 1.999 | 4.428 | 5.258 | 2.574 |
| new A | 9.243 | 2.004 | 4.438 | 5.267 | 2.652 |
| previous B | 8.489 | 2.004 | 3.943 | 4.706 | 2.416 |
| new B | 8.605 | 2.002 | 3.950 | 4.710 | 2.522 |

Busy and wall times of the kernels are equal to the previous run within 0.01 ms. What grew is
the time the GPU waits for the host: at 8 rows, the batch filling and draining phases, idle in
step went from 4.4 to 7.3 ms on A and from 7.0 to 10.5 ms on B. The full corpus shows the same
shape: the mean latency of every 136 request window of the new A c16 is 0.04 to 0.08 s above
the previous run's window, uniformly across the run, and the new B's is 0.02 to 0.04 s above.
A uniform host side slowdown across a two minute run is a host condition, not a code path.

The host was not idle. From the lane 2 logs and the environment captures:

- `slim_ab.log`: from 07:04 to 07:17 a process the harness did not own held 69.6 GB on GPU 2
  with activity (memory growing 69581 to 70181 MiB, util 7 percent), and the harness refused
  to launch.
- `env/nvidia_smi.txt` of lane 2: GPUs 0, 4 and 5 at 1980 MHz SM clock at capture time, GPUs
  2, 3, 6 and 7 at 345 MHz. Other tenants' jobs were running on this host.
- Lane 2 ran during every full corpus point of lane 1: the fp8 Omni servers 07:17:47 to
  07:21:46 and the ASR and UTMOS scoring to 07:23:39 during A c1 (07:19:12 to 07:28:11), the
  two bf16 30B boots and retraction runs 07:24:05 to 07:26:49 also during A c1, and from
  07:27 to 07:45 an occupant of GPU 2 the harness could not identify, during B c1, B c16
  (07:37:02 to 07:39:06) and A c16 (07:39:06 to 07:41:14).
- The first ASR scoring server of lane 2 died at startup with exit code -9 (SIGKILL,
  `serve_asr_31011_attempt1_killed.log`). The harness did not send it and the cause is not
  established. The kernel's out of memory killer sends exactly this signal, and host memory
  pressure slows every process on the host, so `dmesg -T` on the box for that minute is a
  validation task, not a conclusion.

The Qwen3-TTS decode loop at c16 spends 2.5 to 2.7 ms of its 8.6 to 9.2 ms step waiting on the
host, and the batch filling phases spend far more, so host contention lands directly in qps.
The two lane design for the A/B points was my error: suites, probes, coverage and census can
share the host, the full corpus points cannot. The previous A/B ran alone on the host.

Consequence for the B against A delta: the new c16 delta, plus 6.8 percent, is not a cleaner
measurement than the previous plus 3.8 percent. A c16 ran last, under whatever occupied GPU 2
and the host from 07:27 to 07:45, and B c16 ran two minutes earlier under the same unknown.
The c1 delta, plus 2.7 percent against plus 3.4 percent before, is the robust one: the
per step census at 1 row reproduces the previous run to 0.03 ms (8.063 against 8.052 on A,
7.793 against 7.767 on B).

## 2. Qwen3-TTS gates, what passed and what did not

| Gate | Result | Evidence |
| --- | --- | --- |
| suites | pass on A, B and the rope store head, one pre existing failure | `test_compile.py::test_runner_specs_defer_factory_signature_import_to_child` needs two visible GPUs (`GPU id 1 out of range; only 1 visible device(s)`), fails on A too. cpu_no_cuda reruns pass |
| reviewer probes | not run | `remote_probes.py` was never pushed, it lives in the git excluded tasks folder. Now at `tasks/perf_backlog/scripts/remote_probes.py` on the analysis branch with the startup counter line |
| 128 running coverage | pass | `Captured 39 ... in 11.0 s`, mixed run 192 of 192, no budget warning, argmax keys 1 and 2 captured lazily |
| lazy signature | pass with a harness correction | `--top-k 64` sets the backbone top k, not the subtalker's. The corrected run injected `stage_params.tts_engine.subtalker_top_k=64` and captured the sampled 64 keys at rows 1, 2, 4, 8, 12 and 16 with no budget exhaustion |
| corrected run completion | fail, 2 of 64 | `CUDNN_STATUS_INTERNAL_ERROR` in `F.conv1d` of the speaker encoder, reference encode path (`extract_speaker_embedding`), allocator warning at 6 MiB free (`serve_coverage_corrected.log:223, 377`) |
| greedy census | pass | A and B argmax replays both 1221 kernels, names identical (`kernel_names_summary.json`) |
| retraction | pass for mechanics | 12 test retractions per arm, 192 of 192 complete on both, no exception |
| retraction memory | not obtained | both traces carry zero allocator events although `run.sh:126` exports the profiler memory flag for that phase. Traces stayed on the box |
| A/B c1 | pass | 1088 of 1088 byte identical, WER and similarity equal, plus 2.7 percent qps |
| A/B c16 | pass on quality, perf not clean | WER 0.996 against 1.105 percent, similarity 71.20 against 71.18, both inside the identical kernel band. qps read under host contention, section 1 |
| memory | at the edge on both arms | allocator retry warnings on A too (`A/logs/serve_census.log:292, 655`, `A/logs/serve_retract.log:614`), peaks 80595 (A) and 80727 MiB (B) of 81079 |

The corrected coverage failure is the same condition as the rope store c16 failure, with one
addition: the failing cuDNN call is a convolution, not attention. Any cuDNN operation that
needs a plan or workspace at request time fails once the card is full, so the fix is the pool
provisioning, not a switch on one operator.

## 3. Qwen3-Omni lane

| Point | A qps | B qps | B latency mean | A/B WER errors of 564 words | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| fp8 c1 | 1.603 | 1.601 | 0.625 against 0.624 s | 11 against 7 | equal |
| fp8 c16 | 8.942 | 8.842 | 1.667 against 1.667 s | 11 against 16 | |
| fp8 c32 | 12.182 | 11.339 | 2.265 against 2.128 s | 10 against 13 | |
| bf16 retract c16 | 8.145 | 7.592 | 1.984 against 1.866 s | 9 against 4 | 4 test retractions per arm, re prefills of N plus 1 tokens, no exception |

UTMOS equal at every point. Every paired WER interval covers zero. No natural KV retraction
and no AttributeError in any log. The fp8 thinker pools differ between arms, 287799 against
290871 tokens, and the stage order in the boot line differs too, so the colocated boot sizes
its pools in a boot dependent order, an observation, not an effect of the branch.

The three concurrency points at or above c16 all read B below A by 1 to 7 percent from one
boot of 50 requests each. The recorded noise floor for this workload is 9 percent of qps
between identical boots (`06_e0_talker_step.md:31-34`), and the fp8 points ran during lane
1's A c1 on the same host. There is no mechanism in the diff for a talker slowdown of that
size: the helper changes are two static calls per row per step, and the compaction ran four
times per run on histories of at most 155 rows. But three same direction readings are not
dismissed by that argument, they are resolved by measurement: the full corpus, alone on the
host, two boots per arm and point.

## 4. What runs next

1. Qwen3-TTS c1 and c16 pair again, alone on the host, two boots per arm per point in
   A B B A order, with a host gate recorded before every boot: every GPU on the host at 0 MiB
   and 0 percent, the load average, and `dmesg -T | tail` for kills.
2. Qwen3-Omni on the full corpus, alone on the host, fp8 colocated, c1, c16 and c32, two
   boots per arm per point, then WER and UTMOS:

```bash
python -m benchmarks.eval.benchmark_omni_seedtts --generate-only --model qwen3-omni \
  --meta zhaochenyang20/seed-tts-eval-arrow --lang en --voice-clone \
  --port 31000 --max-concurrency $C --output-dir $DIR
python -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only --model qwen3-omni \
  --meta zhaochenyang20/seed-tts-eval-arrow --lang en --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --port 31011 --output-dir $DIR
python -m benchmarks.eval.benchmark_omni_seedtts --utmos-only --output-dir $DIR --device cuda:0
```

   The script has no seed flag, so bits are not comparable between arms; WER, UTMOS and
   speed are.
3. The reviewer probes from `tasks/perf_backlog/scripts/remote_probes.py`.
4. The retraction memory reading: check a retract trace for `[memory]` events before
   trusting the census script's empty memory table.
5. The memory provisioning slice before any further run at 128 running or on the rope store
   c16.
