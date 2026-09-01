# 08. A/B of the observed KV reservation (2026-09-01)

Bundle artifacts/ab-reservation.tar.gz. A = upstream/main 68c88dae6, B =
perf/scheduler-observed-reservation adc09ff5d (the same scheduler change
now at 9bdaab055 with the test stubs fixed). bf16 colocated config, voice
clone (SeedTTS 50, warmup), one boot per arm and workload, GPU idle before
each boot, talker decode log at every step, request profiler events at c16
and c32.

## 1. Bench

| run | qps | latency p50 s | latency p95 s | RTF mean | RTF p95 | WER |
|---|---|---|---|---|---|---|
| A c1 | 2.237 | 0.422 | 0.640 | 0.141 | 0.200 | 1.60 |
| B c1 | 2.214 | 0.428 | 0.684 | 0.138 | 0.195 | 0.89 |
| A c16 | 6.737 | 2.297 | 3.400 | 0.695 | 1.033 | 1.42 |
| B c16 | 8.387 | 1.812 | 2.952 | 0.519 | 0.698 | 1.06 |
| A c32 | 7.595 | 3.712 | 5.192 | 1.160 | 2.024 | 1.24 |
| B c32 | 10.852 | 2.391 | 4.474 | 0.757 | 1.104 | 1.42 |

50 of 50 requests completed in every run. UTMOS 4.43 to 4.46 in every run.

c1 is inside the boot to boot range of the earlier three base boots (qps
2.16 to 2.25, latency p50 0.409 to 0.442, doc 07), and the per sample
latency ratio B over A has p50 1.027 with p10 0.88 and p90 1.22, which is
the sampling variance of unseeded requests, not a shift. Nothing in the
change runs at c1 beyond one deque append per finished request and a
cached maximum per admission.

## 2. Mechanism, from the talker events

| run | talker running max | time weighted mean | admission wait p50 ms | p95 ms | first audio p50 ms |
|---|---|---|---|---|---|
| A c16 | 9 | 6.3 | 935 | 1509 | 1407 |
| B c16 | 16 | 10.7 | 4 | 481 | 702 |
| A c32 | 9 | 6.1 | 1533 | 3264 | 2153 |
| B c32 | 32 | 16.0 | 314 | 842 | 1080 |

The 27 read from the decode log for A is a thinker line: the log merges
both schedulers' Decode batch lines and neither carries a stage name. The
talker running count is the events number above.

## 3. What the run did not establish

- Speaker similarity: every run scored 2.3 to 3.9 with per sample values
  from -20 to +34 (std about 9), against a CI floor of 60.0 for this model
  (test_qwen3_omni_tts_ci.py:78). The scorer did not run with the fine
  tuned WavLM head (popsoda2002/seedtts-wavlm-sim, auto downloaded by
  benchmarks.metrics.speaker_similarity_assets when
  SEEDTTS_SIM_CHECKPOINT is unset). Both arms are equally invalid, the
  scoring has to be rerun with the asset paths logged.
- Retract path: SGLANG_TEST_RETRACT was set for the whole server, so the
  hook ran in both schedulers. The log shows 17 test retractions, 13
  retracting nothing (a batch of one) and 4 retracting one request, with no
  stage name on the line. The talker events cannot show a re prefill either,
  because `_emit_prefill_start_for_batch` emits once per rid
  (omni_scheduler.py:1519-1529). So whether the talker's replay path ran
  is not known from this run. The next run sets the hook on the talker
  stage only (stage `env` in a copy of the yaml, applied at spawn to that
  process), so every retraction line is the talker's.
