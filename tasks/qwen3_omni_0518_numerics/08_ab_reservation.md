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

Both items below were closed by the followup runs in section 4.

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

## 4. Followup runs (2026-09-01, same host, B at 9bdaab055)

Bundle ab-reservation-followup.tar.gz. B's scheduler unit tests: 231 passed
in the container (arm_B_talker_retract.txt).

### 4.1 Talker only retract

The hook was scoped with a stage env in a copy of the bf16 yaml
(bf16_talker_retract.yaml): `stages.talker_ar.env` carries
`SGLANG_TEST_RETRACT: "1"` and `SGLANG_TEST_RETRACT_INTERVAL: "50"`. A stage
env becomes the `env_defaults` of that stage's launch config
(pipeline/mp_runner.py:150) and `_patched_spawn_env` sets those keys only
while that stage's process group is spawned, restoring them afterwards
(pipeline/stage_workers.py:161-203). The events carry one pid per stage
(thinker 1701082, talker_ar 1701084), so the hook lived in the talker
process only.

The log confirms the attribution without a stage name on the line: every
retraction sits between Decode batch lines at token usage 0.07 for about
1.5k tokens, which is the talker's 21373 token pool (the thinker's pool is
over 100k), and each is followed by a Prefill batch of
`new_tokens_gained + 1` tokens with 0 cached tokens, which is the replay of
the retracted request from `_decode_input_history`
(talker_model_runner.py:328-340).

55 firings, 9 retractions of one request each, 46 of nothing: sglang
0.5.18 enters the retract branch every `TEST_RETRACT_INTERVAL` forward
steps (scheduler.py:3492) and `retract_decode` keeps the last request when
one is running (schedule_batch.py:2826), so the empty firings are batches
of one (all of c1 and the tails of c16 and c32).

| run | qps | latency p50 s | latency p95 s | rtf p95 | completed | talker running max |
|---|---|---|---|---|---|---|
| B c1 | 2.214 | 0.428 | 0.684 | 0.1951 | 50/50 | 1 |
| B talker retract c1 | 2.150 | 0.441 | 0.704 | 0.1878 | 50/50 | 1 |
| B c16 | 8.387 | 1.812 | 2.952 | 0.6975 | 50/50 | 16 |
| B talker retract c16 | 8.712 | 1.567 | 2.964 | 0.6818 | 50/50 | 16 |
| B c32 | 10.852 | 2.391 | 4.474 | 1.1035 | 50/50 | 32 |
| B talker retract c32 | 10.960 | 2.428 | 4.310 | 1.0267 | 50/50 | 32 |

No errors of any kind in the server log during the runs (the only ERROR
line is the nixl import at boot). The talker events hold one
`scheduler_prefill_start` per rid in all runs (66 rids at c16, 82 at c32,
the 16 or 32 warmup requests included).

Not scored on these three runs: WER and UTMOS. Only similarity was run,
which section 4.2 shows is uninformative. Task: score WER and UTMOS on
B_talker_retract_c16 and c32 (the audio is on the host).

### 4.2 Speaker similarity

The rerun used the fine tuned head explicitly and logged it
(wavlm_large_finetune.pth sha256 51f07e3b..., wavlm_large.pt sha256
6fb4b3c3...). Scoring a generated wav against itself gives 100.0, and the
original six scores reproduced to the digit.

| run | c1 | c16 | c32 |
|---|---|---|---|
| A | 3.88 | 2.33 | 2.88 |
| B | 3.60 | 3.53 | 2.70 |
| B talker retract | 3.22 | 2.46 | 4.06 |

These values are the known state of this benchmark for Qwen3-Omni, not a
scorer fault and not an arm difference. The omni seedtts benchmark sends
the reference as an input audio of a chat request with the instruction
"read the following text out loud in the same voice and style"
(benchmarks/tasks/tts.py:800-838), and the talker answers with the
configured speaker (Ethan), so the score compares Ethan with the dataset
speaker. test_qwen3_omni_tts_ci.py:68-78 records five earlier runs at 2.90
to 3.48, keeps 60.0 as a placeholder and has the assertion disabled
pending issue #483 (open: whether the talker consumes the prompt's
multimodal features at all). Section 3's first bullet, which blamed the
missing head, is withdrawn.

The measurement that does test "voice unchanged" for a preset speaker
model is the similarity of A's wav against B's wav for the same sample id
(same speaker, same text), read against B versus B talker retract as the
run to run baseline. Task.

### 4.3 fp8 colocated, video (bench_amme, c16)

A 68c88dae6 versus B 9bdaab055, one boot each (A's first boot ran no
workload because the copied helper rejected the older VideoEvalConfig, the
retry boot is the A arm).

| arm | completed | accuracy | qps | latency mean s | peak GPU MiB |
|---|---|---|---|---|---|
| A | 50/50 | 33/50 | 1.001 | 13.844 | 65054 |
| B | 50/50 | 32/50 | 1.022 | 13.471 | 64850 |

CI floor 0.62 (test_qwen3_omni_videoamme_ci.py:34). No retraction line, no
error and no capture in either log during the run window (every graph
capture finished before the first request). Peak memory is 80 percent of
the 81559 MiB card in both arms.

Where the time goes: the preprocessing stage spans 13.7 to 14.0 s per
request (p50) while `preprocess_start` to `preprocess_end` is about 1.0 s,
so the stage serializes the 16 concurrent videos and the pipeline's 1.0
qps is its rate. The image encoder takes 172 to 181 ms per request and the
thinker 0.8 to 0.9 s. This is plan item B (preprocessing replicas) in
05_pr_plan.md.

Thinker admission, from the thinker events (`scheduler_queue_enter` to
`scheduler_prefill_start`): p50 1.2 ms in A and 1.0 ms in B. A's only waits
above 10 ms were a burst of five arrivals at t=24.9 s with waits of 10,
148, 370, 368 and 739 ms, which is the sum of the preceding prompts'
prefills in that burst (14.4k tokens each, chunked prefill), not the
reservation: on this workload the thinker's per request reservation is at
most 256 tokens (max_tokens 256) against a pool of 120k tokens. B saw no
burst because its arrivals were evenly spaced. Thinker running max 8 in A
and 2 in B, pool usage max 0.43 in A and 0.14 in B.

A also had two thinker stalls that B did not: the first request's decode
took 5632 ms (warmup, B 504 ms) and a request entering at t=18.5 s took
6451 ms to prefill (B never above 166 ms). Nothing in the log explains
them. They are in the arm without the change, and the 2.1 percent qps
difference sits inside them.

The answer flip: item 003-2 at temperature 0.0, A "C" (correct) and B
"D". The two responses diverge from the first sentence. The two arms ran
with different batch compositions (unseeded serving, staggered arrivals),
and no second A run exists to give this benchmark's run to run flip rate.
Task, optional: one more A video run at c16.

## 5. Remaining before the PR

- WER and UTMOS on B_talker_retract_c16 and c32.
- Same sample similarity A versus B, with B versus B talker retract as the
  baseline.

## 6. Which stages the change reaches

OmniScheduler is the scheduler of every sglang engine stage except two:
`SGLangGenerationEngineBuilder._make_scheduler` (scheduling/engine_factory.py:338,
the five ASR builders), `TtsEngineBuilder.make_scheduler`
(engine_factory.py:411, eleven TTS builders), the Qwen3-Omni thinker
(models/qwen3_omni/bootstrap.py:112), the Ming-Omni thinker
(models/ming_omni/bootstrap.py:96), and the two subclasses
QwenTalkerScheduler (talker_scheduler.py:39, overrides
get_next_batch_to_run only) and MiniMaxMusic3Scheduler
(minimax_music3/scheduler.py:50, its get_new_batch_prefill calls super).
Out of reach: llada2_uni (DllmScheduler, diffusion) and audar_tts
(llama_cpp behind SimpleScheduler).

Per request reservation today, 0.7 times min(cap, 4096), by the cap each
request builder sends when the request carries none:

| stage | cap | reserved today | max running |
|---|---|---|---|
| minimax_music3 ar | 9001 (two engine rows per request) | 2867 per row | 16 requests, 32 rows |
| moss_tts, moss_tts_local, voxtral_tts | 4096 | 2867 | 16 |
| qwen3_omni talker_ar | 4096 | 2867 | 32 |
| moss_transcribe_diarize | max(5120, 10 per audio second) | 2867 | 16 |
| fishaudio_s2_pro, higgs_tts | 2048 | 1434 | 64 |
| qwen3_omni thinker, ming_omni thinker | 2048 | 1434 | 64, 16 |
| qwen3_tts | 2048 at a 12 Hz codec | 1434 | 16 |
| fun_cosyvoice3 | min(2048, 20 per text token) | up to 1434 | 32 |
| zonos2 | 1024 | 717 | 16 |
| whisper_asr, arkasr | 256 | 179 | 64, 32 |
| qwen3_asr | max(128, 10 per audio second) | 90 to 210 | 64 |
| fun_asr | min(200, 200 per 30 s) | up to 140 | 64 |
| ming_tts | 200 steps | 140 | 8 |

Whether the reservation binds a stage is a property of its pool and its
prompts, not of the cap alone, and only the talker's pool has been read
(21373 tokens). The stages where the cap is far above the output are the
candidates: qwen3_tts (a few seconds of speech is 50 to 120 frames against
2048), the three 4096 TTS stages, and the two 64 request stages. The ASR
stages with duration derived caps and ming_tts have little to gain by
construction. Task, per model: read the pool from the boot log ("KV Cache
is allocated ... #tokens") and the stage's admission wait from events at
the CI concurrency, A versus B, before claiming a number.

One consequence had to be fixed before the PR. Stages whose outputs run
past the 4096 clip (minimax_music3 typically fills its 9000 frame cap,
moss_transcribe_diarize on an hour of audio) would have recorded a fraction
above 1.0, because the fraction divided the output by the clipped cap.
sglang keeps the ratio within (0, 1]: the tracker clamps init and min at
1.0 and the post retract estimate at 1.0
(new_token_ratio_tracker.py:20-32, 54), and the PrefillAdder's running
budget multiplies the unclipped cap by the ratio
(schedule_policy.py:1085-1088), so a ratio of 2.2 would have turned a 9001
cap into a 19800 token expectation per running request and returned
NO_TOKEN until the running requests finished. The fix records
min(output, clipped cap) / clipped cap, so such outputs read as 1.0
(working tree of perf/scheduler-observed-reservation, not yet committed). With 1.0 those stages
reserve min(cap, 4096) per request instead of 0.7 of it, which is the
value sglang itself uses for ignore_eos requests (schedule_policy.py:1086).
That is more conservative than today for cap filling stages. Task: A
versus B on minimax_music3 at its CI concurrency, reading admission wait
and retractions in both arms, before merging.
