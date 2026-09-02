# 09. The reservation change, traced, and the full Qwen3-Omni A/B (2026-09-02)

Code read: perf/scheduler-observed-reservation at ce2735c02 (the 44 added
lines in sglang_omni/scheduling/omni_scheduler.py) against sglang 0.5.18
(python/sglang/srt/managers). Line numbers below are from those two trees.

## 1. The path of one scheduler iteration, with the two hooks

Both Omni event loops (omni_scheduler.py:2327 sync, 2563 overlap) run the
same skeleton per iteration. The change adds one call at the top of
admission and one call at request finish. Nothing else in the loop moved.

```
recv_requests, process_input_requests
        |
        v
get_next_batch_to_run            omni 1359 -> upstream scheduler.py:3015
        |   merge the last prefill batch into running_batch (3067-3092)
        v
get_new_batch_prefill            omni 1373                          <- hook 1
   |  _apply_observed_new_token_ratio()                    omni 729-734
   |     deque empty  -> leave tracker.current as sglang left it
   |     deque filled -> tracker.current = max(deque)
   |                     (the max is cached until the next finish)
   |  coalesce hold off may return an empty plan here (1380-1408)
   v
_Upstream.get_new_batch_prefill  -> _get_new_batch_prefill_raw (3184)
   |  batch_is_full or empty waiting queue -> no prefill (3202-3205)
   |  PrefillAdder(new_token_ratio=tracker.current, ...)   3257-3274
   |     offset = sum over running rows of
   |              min(cap - out, 4096) * ratio      schedule_policy.py:654-661
   |     rem_total_tokens = available + evictable - offset        664-685
   |  for each waiting req, add_one_req (1201):
   |     need = extend_len + min(cap - out, 4096) + page_size    1219-1226
   |            (the candidate's own reservation is NOT scaled)
   |     need >= rem_total_tokens -> NO_TOKEN, admission stops    1236
   |     admitted -> offset += extend + min(cap, 4096) + page     1168, 876
   v
prefill batch found?  yes -> run_batch(prefill) -> process_batch_result
   |
   no
   v
update_running_batch             upstream 3481 (decode rows only)
   |  check_decode_mem false, or the test hook (3491-3493):
   |     retract_decode                     schedule_batch.py:2816
   |        pops the rows with the fewest output tokens until the
   |        next step fits, keeps at least one row (2824-2836)
   |     tracker.current = (sum out + 20 n) / (sum cap + 1), max 1.0
   |                                        3526, tracker.py:41-51
   |  otherwise tracker.decay_step()        3554, tracker.py:34-35
   |     current = max(current - (0.7 - 0.098) / 600, 0.098)
   v
run_batch(decode) -> process_batch_result   omni 1417 -> upstream
   |  batch_result_processor -> output_streamer.stream_output
   |     (omni binds it to its own stream_output, 686-691)
   v
stream_output                    omni 1622                          <- hook 2
   |  finished, not aborted, not a stale alias (1632-1667)
   |  _record_finished_output(req)                          omni 715-727
   |     clipped = min(cap, 4096)
   |     deque.append(min(len(output_ids), clipped) / clipped)
   |     cached max = None
   |  terminal payload, callbacks, KV release
   v
no batch this iteration -> self_check_during_idle -> tracker.reset()
                                                     omni 736-737
```

Three sglang writes to the ratio survive in the code and are overwritten
at the next iteration's hook 1 once the deque holds an entry: the decay
step (3554), the idle reset to 0.7 (omni 737) and the raise after a KV
full retract (3526). Until the first finish of the process, all three act
exactly as before.

The deque has maxlen max_running_requests (omni 427-429), so an entry
leaves after that many later finishes. Retracted rows are not finished and
are not recorded. Aborted rows, including the last row aborted by
retract_decode when nothing fits (schedule_batch.py:2838-2852), take the
is_aborted branch before hook 2 and are not recorded either.

## 2. What the change is, measured against sglang's own mechanism

sglang's reservation is one scalar per scheduler that scales the remaining
clipped cap of every running row. It starts at 0.7 times
schedule_conservativeness, decays linearly to 0.098 over 600 decode steps,
returns to 0.7 at every idle iteration, and after a KV full retract is
set to the pooled fraction of the rows that stayed, plus 20 steps each,
clamped at 1.0. The candidate's own reservation is never scaled. Nothing
in sglang measures how much of the cap finished requests used.

The change keeps every consumer, every clamp and the retract path, and
replaces the source of the scalar after the first finish: the largest
fraction of the clipped cap used by a request in the last cohort of
max_running_requests finishes. It is stage agnostic because the cap and
the output length come from the request, and the window from the engine
argument. What it gives up and what bounds that:

- Retract time feedback. After a KV full retract sglang raises the ratio
  from the rows still running, the change puts the observed maximum back
  at the next iteration. The retracted row cannot come straight back in:
  retract_decode frees only what the next decode step needed, and
  readmission needs available tokens above its extend length plus its
  whole clipped cap (schedule_policy.py:1226, 1236). The ratio rises when
  a request longer than the cohort's maximum finishes, which is the first
  moment the scheduler can know that the distribution moved.
- One scalar for rows with different caps. A request that fills a short
  cap (a 32 token answer at max_tokens 32) pins the ratio at 1.0 for one
  cohort and every running row reserves its whole clipped cap. sglang has
  the same scalar and the same pooled estimate after a retract. The bound
  is rows times min(cap, 4096) per stage, the reservation sglang itself
  uses for ignore_eos rows (schedule_policy.py:1086).
- The floor. sglang decays to 0.098 whatever the outputs are and repairs
  under reservation by retracting. The change reserves what the last
  cohort needed. For a stage whose outputs use more than a tenth of the
  clipped cap this is more than sglang's floor and fewer retractions, for
  a stage whose outputs use less (every TTS stage surveyed) it is less
  than sglang's floor and the 600 step wait to reach it is gone.

Where a reservation of any size stops mattering: when rows times the
reserved tokens plus the prompts fit the pool at max_running_requests, the
row limit binds and neither arm can differ. That is the test for every
stage in section 3.

## 3. Every stage the hook runs in, and the bound on an adverse effect

OmniScheduler is the scheduler of every sglang engine stage except
llada2_uni and audar_tts (08_ab_reservation.md section 6 lists the
builders). The largest reservation the change can ever set is rows times
min(cap, 4096), at ratio 1.0. sglang's start is 0.7 of that, its floor is
0.098 of that.

| stage | rows | clipped cap | B at 1.0, tokens | A at 0.7 | A floor | fraction seen | pool read |
|---|---|---|---|---|---|---|---|
| qwen3_omni talker_ar (0.10 fraction) | 32 | 4096 | 131072 | 91750 | 12845 | tens of frames of 4096 | 21373 |
| qwen3_omni thinker | 64 | 2048 default, 256 or 32 in the video and mmsu benches | 131072 | 91750 | 12845 | not recorded | about 120000 (fp8 colocated) |
| minimax_music3 ar | 32 | 4096 | 131072 | 91750 | 12845 | 0.36 to 0.39 | 118366 and 166892 |
| moss_tts, moss_tts_local, voxtral_tts | 16 | 4096 | 65536 | 45875 | 6423 | not measured | not read |
| moss_transcribe_diarize | 16 | 4096 | 65536 | 45875 | 6423 | not measured | not read |
| fishaudio_s2_pro, higgs_tts | 64 | 2048 | 131072 | 91750 | 12845 | not measured | not read |
| qwen3_tts | 16 | 2048 | 32768 | 22938 | 3211 | not measured | not read |
| fun_cosyvoice3 | 32 | up to 2048 | 65536 | 45875 | 6423 | not measured | not read |
| zonos2 | 16 | 1024 | 16384 | 11469 | 1606 | not measured | not read |
| whisper_asr, arkasr | 64, 32 | 256 | 16384, 8192 | 11469, 8192 | 1606, 803 | not measured | not read |
| qwen3_asr | 64 | 128 to 210 | 8192 to 13440 | 5734 to 9408 | 803 to 1317 | not measured | not read |
| fun_asr | 64 | up to 200 | 12800 | 8960 | 1254 | not measured | not read |
| ming_tts | 8 | 200 | 1600 | 1120 | 157 | not measured | not read |

Read from the three pools known: the talker is the case where A's start
binds (91750 against 21373) and B's observed value does not. The minimax
run at 16 requests admitted all 32 rows in B and A bound only through the
smaller pool of its boot, at a fraction below 0.7. The thinker's benches
carry caps of 256 and 32, so its reservation is under 16384 tokens at any
ratio against a pool near 120000.

qwen3_asr in tokens: Qwen3-ASR-0.6B has 28 layers, 8 KV heads and head
dim 128 (config.json in the local HF cache), so one KV token is 114688
bytes in bf16 and the largest reservation the change can set, 13440
tokens, is 1.47 GiB. Any pool above that plus the prompts cannot bind on
the reservation. qwen3_tts at its largest is 32768 tokens and its per
token size was not read here. Task, unchanged from section 6 of doc 08:
read "KV Cache is allocated ... #tokens" from one boot of each model
before claiming anything about it, and read the stage's admission wait
from events at the CI concurrency.

## 4. The c1 numbers

At one running row the offset is one term and the candidate check is the
only gate, so the ratio cannot change which requests run. Doc 08 section
1 shows A c1 at qps 2.237 and B c1 at 2.214, inside the boot to boot band
of the three earlier base boots (2.16 to 2.25) and inside B's own band
(2.150 on the talker retract boot, 2.214 on the plain boot). The extra
work at c1 is one deque append per finish and one max over at most 32
floats per iteration, both on the scheduler thread between forwards. A
c1 difference on Qwen3-Omni is boot and sampling variance until a run
shows a shift larger than that band, and the full A/B below runs c1 on
every stage so the band is measured again on both arms.

## 5. The full Qwen3-Omni A/B

One pass over every stage of the Qwen3-Omni CI
(.github/workflows/test-qwen3-omni-ci.yaml), each stage being its CI test
file run with pytest at the CI settings, so the thresholds the tests assert
are the verdict of each arm and nothing is re implemented. Arms as in every
run so far: A 68c88dae6 (the base of perf/scheduler-observed-reservation),
B ce2735c02. main has moved by three commits since the base (a README news
order fix, the 0.1.4 release preparation, a README for .claude/skills),
which the branch has to take before the PR opens. The test files are
identical in the two arms (git diff of tests/test_model is empty).

| stage | test file | server | input to output | samples at c16 | what the test asserts |
|---|---|---|---|---|---|
| thinker_length | test_qwen3_omni_thinker_length.py | bf16 thinker TP2, max_seq_len 128 | text | contract posts | finish_reason and context length HTTP contract |
| tts | test_qwen3_omni_tts_ci.py | bf16 colocated, router with two workers | ref audio and text to speech | 50 | speed P95 gates, WER, UTMOS, similarity (disabled, issue #483) |
| mmmu | test_qwen3_omni_mmmu_ci.py | fp8 colocated, two workers | image and text to text | 50 | accuracy 0.6, speed P95 gates |
| mmmu_talker | test_qwen3_omni_mmmu_talker_ci.py | bf16 disagg, two GPUs | image and text to text and speech | 20 | accuracy 0.7, WER, speed and rtf gates |
| mmsu | test_qwen3_omni_mmsu_ci.py | bf16 thinker only, two workers | audio and text to text | 2000 | accuracy 0.7035, speed P95 gates |
| mmsu_talker | test_qwen3_omni_mmsu_talker_ci.py | fp8 thinker TP2 | audio and text to text and speech | 40 | accuracy 0.625, WER, speed and rtf gates |
| videomme | test_qwen3_omni_videomme_ci.py | bf16 disagg | video to text | 50 | accuracy 0.58, speed P95 gates |
| videomme_talker | test_qwen3_omni_videomme_talker_ci.py | bf16 disagg | video to text and speech | 20 | accuracy 0.6, WER, speed and rtf gates |
| videoamme | test_qwen3_omni_videoamme_ci.py | fp8 colocated, two workers | video with audio to text | 50 | accuracy 0.62, speed P95 gates |
| videoamme_talker_tp2 | test_qwen3_omni_videoamme_talker_tp2_ci.py | fp8 thinker TP2 | video with audio to text and speech | 10 | accuracy 0.5, WER, speed and rtf gates |

Every stage boots its own server, so the finished output window never
carries from one benchmark into another. Inside a stage it carries from
the warmup requests into the measured run, per worker, which is what
production sees.

### 5.1 Running it

scripts/full_ab.sh runs the ten stages in CI order, both arms per stage
with the order alternating from stage to stage (A then B, then B then A)
so neither arm always pays a cold cache or follows a hot GPU. Between runs
it cleans the GPUs with the CI script (.github/scripts/delete_gpu_process.sh
with the thresholds of run_all_wer_ci_aligned.sh), switches the checkout
with git checkout --detach and restores the original ref at the end. Each
run gets its own pytest basetemp under $OUT/<stage>/<arm>/tmp, which holds
the result JSONs and, with GITHUB_ACTIONS=true set for pytest, the
server.log of every server the fixtures start
(benchmarks/benchmarker/utils.py:67-72). The script must run from a copy
outside the tree because the checkout changes under it.

```
cp -r "$OMNI_ROOT/tasks/qwen3_omni_0518_numerics/scripts" "$OUT/scripts"
OMNI_ROOT=/path/to/checkout OUT=/data/ab-full GPU=0,1 bash "$OUT/scripts/full_ab.sh"
OMNI_ROOT=/path/to/checkout OUT=/data/ab-full GPU=0 bash "$OUT/scripts/full_ab.sh" tts_c1
python "$OUT/scripts/full_ab_compare.py" "$OUT" --md "$OUT/readout.md"
```

The tts_c1 pass is the c1 addendum of section 4: the voice clone bench at
concurrency 1 on the bf16 colocated profile through run_bench.py, one
manual boot per arm, the same measurement as doc 08 section 1.

Budget: the CI stage timeouts sum to 240 min, so one arm is at most 4 h
and the pass at most 8 h plus two boots for c1. STAGES="tts videoamme"
runs a subset.

### 5.2 Reading it

full_ab_compare.py prints, per stage, the pytest exit code of both arms,
every *_results.json under A next to the same file under B with the
numeric leaves of summary, speed, speed_metrics and wer.summary and the
ratio B over A, then the KV pool sizes at boot and the count of
retraction lines from the server logs of both arms. Checked against the
clip fix bundle's c16 result files, the readout reproduces the doc 08
section 1 numbers.

What counts as a regression: an assertion that fails in B and passed in A
on the same stage, a retraction line in B's logs with none in A's, or a B
metric worse than A by more than the boot to boot spread measured for
that stage (doc 07 for the voice clone stage). For the other stages no
spread has been measured, so a single A/B there catches an assertion flip
and a shift larger than the CI's own P95 slack, not a few percent. If a
stage shows a few percent shift in either direction, the next step is
three repeats of that stage on both arms, not a conclusion.

Not in this pass: the admission wait and running count per stage, which
need the request profiler on a manual boot (doc 08 section 2 shows how
the talker's were read). The pytest path does not start the profiler.
