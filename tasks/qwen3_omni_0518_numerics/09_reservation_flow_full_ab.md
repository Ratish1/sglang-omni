# 09. The reservation change, traced, and the full Qwen3-Omni A/B (2026-09-02)

Code read: perf/scheduler-observed-reservation at 9769867a0, the tracker extraction
on top of ce2735c02 (sglang_omni/scheduling/finished_output_tracker.py and
the two hooks in omni_scheduler.py) against sglang 0.5.18
(python/sglang/srt/managers). Line numbers below are from those trees.

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
get_new_batch_prefill            omni 1341                          <- hook 1
   |  observed = finished_output_tracker.max_fraction        omni 1347
   |     None (nothing finished yet) -> tracker.current stays as sglang left it
   |     else tracker.current = observed
   |        (max over the window, finished_output_tracker.py:41-42)
   |  coalesce hold off may return an empty plan here (1350-1376)
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
stream_output                    omni 1592                          <- hook 2
   |  finished, not aborted, not a stale alias (1602-1637)
   |  finished_output_tracker.observe_finished(req)         omni 1639
   |     clipped = min(cap, CLIP_MAX_NEW_TOKENS)   finished_output_tracker.py:44-54
   |     window.append(min(len(output_ids), clipped) / clipped)
   |  terminal payload, callbacks, KV release
   v
no batch this iteration -> self_check_during_idle -> tracker.reset()
                                                     omni 704-705
```

Three sglang writes to the ratio survive in the code and are overwritten
at the next iteration's hook 1 once the window holds an entry: the decay
step (3554), the idle reset to 0.7 (omni 705) and the raise after a KV
full retract (3526). Until the first finish of the process, all three act
exactly as before.

The window holds max_running_requests entries (omni 416-418,
finished_output_tracker.py:38), so an entry leaves after that many later
finishes. Retracted rows are not finished and
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
| qwen3_omni talker_ar | 32 | 4096 | 131072 | 91750 | 12845 | tens of frames of 4096 | 21373 (bf16 colocated, 0.10), 120769 (fp8 colocated, 0.12, fp8 talker weights) |
| qwen3_omni thinker | 64 | 2048 default, 256 or 32 in the video and mmsu benches | 131072 | 91750 | 12845 | not recorded | 109029 (fp8 colocated, 0.50) |
| minimax_music3 ar | 32 | 4096 | 131072 | 91750 | 12845 | 0.36 to 0.39 | 118366 and 166892 |
| moss_tts, moss_tts_local, voxtral_tts | 16 | 4096 | 65536 | 45875 | 6423 | not measured | not read |
| moss_transcribe_diarize | 16 | 4096 | 65536 | 45875 | 6423 | 1.0 on long audio by construction, see 5.3 | not read (0.80 fraction) |
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
B 9769867a0 (ce2735c02 plus the tracker extraction, same behavior). main has moved by three commits since the base (a README news
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
OMNI_ROOT=/path/to/checkout OUT=/data/ab-full GPU=0 bash "$OUT/scripts/full_ab.sh" tts_conc
OMNI_ROOT=/path/to/checkout OUT=/data/ab-full GPU=0 bash "$OUT/scripts/full_ab.sh" tts_score
python "$OUT/scripts/full_ab_compare.py" "$OUT" --md "$OUT/readout.md"
```

The CI stages run at c16 only. The tts_conc pass is the concurrency
sweep on the one stage where the reservation binds: the voice clone bench
at c1, c16 and c32 on the bf16 colocated profile through run_bench.py,
one manual boot per arm with the three runs on that boot, the measurement
of doc 08 section 1 repeated on the final commit. tts_score then runs the
CI's own WER scorer (against a Qwen3-ASR-1.7B server, the CI's WER model)
and the UTMOS scorer on every run, so each concurrency has speed, WER and
UTMOS on both arms.

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

### 5.3 The other models the hook runs in

The clip is always on in sglang: CLIP_MAX_NEW_TOKENS is a module constant
of schedule_policy.py read from SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION
(default 4096) with no switch, applied to every running row's reservation
and to every candidate's own reservation. The tracker imports that
constant, so the two always agree.

full_ab.sh takes three more stages when named in STAGES, each the model's
own CI test file at CI settings:

| stage | test file | why it is worth a run |
|---|---|---|
| moss_td | test_asr_ci_multi_speaker.py (movies800, aishell4, googletime, c16) | the one stage where a 1.0 fraction is expected: the default cap scales with audio duration (request_builders.py:452-457) and a transcript past the 4096 clip reads as 1.0, so B reserves 16 times 4096 (65536 tokens) against A's 45875 at the start. Its pool at mem_fraction_static 0.80 (stages.py:79) has not been read |
| qwen3_asr | test_asr_ci_seedtts.py with --asr-ci-model qwen3 | the ASR bound of section 3, at most 13440 tokens, for the record |
| qwen3_tts | test_tts_ci.py with --tts-ci-model qwen3-tts (all three tts stages) | the 12 Hz codec case, at most 32768 tokens, pool not read |

MiniMax Music 3 has no CI test. Its before and after follows the
cookbook (docs/cookbook/minimax_music3.md) with
scripts/minimax_cookbook_ab.sh: the cookbook's serve line, then the five
reference requests one at a time, then the same five at once, per arm.
The sequential pass is a quality check by construction. Each request is a
batch of one row pair with the same seed and prompt in both arms, and the
scheduler change cannot reach a single request's numerics, so the five
wavs must be byte identical across the arms (the cookbook documents seed
determinism). The parallel pass is where admission differs and gives the
wall time. Byte identity is not expected there because batch composition
changes the arithmetic, which the c16 run of doc 08 section 7 showed with
5 of 16 identical.

Why quality cannot move: the change decides only when a request is
admitted. It does not touch a forward pass, a sampling draw, a KV entry or
the retract replay, which is sglang's path and was measured on the talker
(doc 08 sections 4.1 and 5). What admission timing does change is which
requests share a batch, and batched kernels are not bitwise invariant to
their batch mates. That is the same variance any two boots have and it is
what the full A/B reads as run to run spread.

### 5.4 The slim pass, the one that runs

The CI pass of 5.1 is more than the question needs. The pass that runs is
scripts/slim_ab.sh: one manually started server per arm, no router, no
pytest. The profile is the fp8 colocated yaml of the H100 CI stages
(examples/configs/qwen3_omni_colocated_h100_fp8.yaml, thinker 0.55, talker
0.12, one GPU), which is what serves on H100. The bf16 colocated profile
can run out of memory on the video benchmarks on an 80 GB card, and the
CI keeps bf16 for video on a two GPU disagg topology for that reason. On
the fp8 profile the talker pool is still a 0.12 fraction, so the
reservation binds there exactly as it did on bf16 (doc 08 section 2).

| stage | concurrency | what it reads |
|---|---|---|
| seedtts_c1, c16, c32 | 1, 16, 32 | speed, then WER and UTMOS from the score pass |
| mmsu_c16 | 16, the 2000 clips | accuracy, speed |
| videomme_talker_c16 | 16, 20 clips, speech output | thinker text accuracy, rtf, latency, qps |

The five runs share the boot in that order, A then B, so the finished
output window carries from the voice clone requests into the MMSU and
video requests as it would on a production server taking mixed traffic.
run_bench.py sends CI identical requests (top_logprobs off, the field does
not exist on either arm). The score pass starts a Qwen3-ASR-1.7B server
and runs the seedtts scorers in transcribe only and utmos only mode on
every voice clone output directory. The readout is full_ab_compare.py
over $OUT. PROFILE=bf16 runs the CI's bf16 topologies instead, three boots
per arm and two GPUs. Wall time on fp8 is about 35 minutes per arm plus
15 minutes of scoring.

fp8 changes the numerics of both arms in the same way, so A against B is
unaffected. What it changes is the reference band: the doc 08 voice clone
bands are bf16 boots and do not carry over, and the fp8 video run of doc
08 section 4.3 (accuracy 32 to 33 of 50, qps about 1.0 at c16) is the
only fp8 band on file. The pass criteria are therefore A against B on
this profile, with the doc 08 bands as a sanity check on the bf16 run
only.

## 6. The slim run on fp8 (2026-09-02, bundle ab-slim.tar.gz)

A 68c88dae6 against B 9769867a0 on the fp8 colocated profile, one boot per
arm on one H100, the five runs of section 5.4 in order, WER and UTMOS
scored afterwards. No retraction line in either arm, no failed request,
50 of 50 and 2000 of 2000 and 20 of 20 completed everywhere.

| run | metric | A | B |
|---|---|---|---|
| seedtts c1 | qps, latency p95 s, WER, UTMOS | 1.530, 1.001, 0.0142, 4.445 | 1.519, 1.064, 0.0160, 4.471 |
| seedtts c16 | qps, latency p95 s, WER, UTMOS | 8.976, 2.439, 0.0089, 4.462 | 8.272, 2.869, 0.0106, 4.471 |
| seedtts c32 | qps, latency p95 s, WER, UTMOS | 11.535, 3.059, 0.0177, 4.441 | 11.455, 3.154, 0.0089, 4.463 |
| mmsu c16 | accuracy, qps, latency p99 s | 0.7125, 60.8, 2.109 | 0.7105, 76.8, 0.464 |
| videomme talker c16 | accuracy, qps, latency p95 s | 11/20, 0.773, 21.7 | 11/20, 0.776, 23.1 |

### 6.1 Why the talker gain of doc 08 does not appear here

The pools differ from the bf16 profile by the talker's weight format. On
this boot the talker loads as fp8 (Load weight end, type Qwen3OmniTalker,
quant fp8, mem usage 3.32 GB) and its 0.12 fraction leaves a pool of
120769 tokens (K 2.76 GB, V 2.76 GB). On the bf16 profile the talker's
0.10 fraction held bf16 weights and left 21373 tokens (doc 08 section 2).
The thinker at 0.50 has 109029 tokens.

sglang's start reservation at 32 rows is 32 times 2867 plus the prompts,
about 96k tokens, and the candidate check needs another 4.2k, so all 32
rows fit the 120769 token pool at once. The log agrees: in the c32
window A reaches 29 running rows and B 32 with at most 4 and 3 queued on
two lines each, in the c16 window both arms admit every request in the
first prefill pass (A 15, B 16 running, nothing queued). With the pool
never binding there is no mechanism by which the two arms can differ, and
section 2's rule applies: the row limit binds, not the reservation. On
this profile the start reservation would first bind above about 38 rows,
past the talker's max_running_requests of 32.

The c16 difference (8.98 against 8.27 qps) is 0.47 s of wall time on a 50
request run that lasts 6 s, carried by five requests above 2.5 s in B
against three in A. Both arms ran the same admission in that window, so
it is boot and sampling variance, and c32 on the same boots agrees to
0.7 percent. A three repeat run at c16 would put a band on it. Task, only
if a number at c16 on this profile is ever needed.

### 6.2 The other two runs

MMSU: accuracy 1425 against 1421 of 2000, four answers out of 2000 under
unseeded sampling. A's lower qps and its 2.1 s p99 come from two windows
of 4.0 s and 2.0 s at the start of its run in which neither scheduler
logged a prefill or a decode and nothing was queued, so the stall sits
before the schedulers (client, preprocessing or the audio encoder), in
code both arms share. B's run has no gap above one second. The thinker's
reservation on this workload is at most 22 tokens per row (cap 32), so
the change cannot reach it.

Video-MME with the talker: identical answers (11 of 20, the same
mc_fallback), qps equal. The thinker pool is what binds here in both
arms: 7 running rows at token usage 0.83 to 0.84, because each prompt is
14210 tokens against a 109029 token pool. The reservation of 256 tokens
per row is noise next to the prompts.

### 6.3 What this says about the change

The code is doing what it does on bf16: it sets the ratio from finished
outputs and both arms admit everything the pool allows. What differs is
the pool. The gain exists where rows times 0.7 times min(cap, 4096) plus
the prompts exceed the stage's pool: the bf16 colocated profile of the
CI's TTS stage (21373 tokens, plus 25 percent qps at c16 and plus 43
percent at c32 in doc 08), and any profile with a talker pool under about
115k tokens at 32 rows. On the fp8 colocated H100 profile the talker pool
is 120769 tokens and the change is neutral at the default row limit,
which is what this run shows: no gain and no regression. Profiles not
read yet, with the pool line as the one thing to read before expecting
anything: the MPS DP yamls at fractions 0.20 to 0.37, the H20 and H200
colocated yamls (talker 0.12 and 0.123 with bf16 weights), and fp8 with
max_running_requests raised above 38.

### 6.4 Where the talker reservation can bind, by profile

Inputs read from boot logs: the talker's KV is 45.7 KB per token in both
runs (bf16 run: 0.92 GB for 20136 tokens, fp8 run: 5.52 GB for 120769),
its bf16 weights take 6.35 GB and its fp8 weights 3.32 GB (Load weight
end lines), and the fraction minus weights minus pool leaves 0.70 GB on
both boots (the runner's reserve). sglang's start reservation at 32 rows
is 32 times (2867 plus a 170 token prompt and output) plus the 4.2k the
candidate check needs, about 101k tokens. The doc 08 bf16 run also shows
the thinker on that profile at 5248 tokens (0.73 fraction over 56.9 GB
of bf16 weights), so that profile is starved on both stages.

| profile | card | talker fraction and weights | pool, tokens | rows at which the start reservation binds | at 32 rows |
|---|---|---|---|---|---|
| H100 bf16 colocated (CI TTS stage) | 79.65 GB | 0.10, bf16 | 20136 read | 6 | binds hard, the doc 08 gain |
| H100 fp8 colocated | 79.65 GB | 0.12, fp8 | 120769 read | 38 | nothing, this run |
| H200 bf16 colocated (qwen3_omni_colocated_h200.yaml) | 141 GB card | 0.123, bf16 | about 223k derived | about 72 | nothing |
| H20 bf16 colocated | 96 GB card | 0.12, bf16 | about 95k derived | about 30 | two rows wait, marginal |

The derived rows assume sglang reports the whole card as total and the
same 0.70 GB reserve. Task before quoting either: read the pool line from
one boot on that card.

So the change is a correction to the admission estimate that pays only
where a stage's pool is small next to rows times 0.7 times the clipped
cap: the bf16 H100 colocated profile, where 57 GB of bf16 weights leave
sub gigabyte pools and the card is 0.96 allocated so no fraction can
grow, the MPS DP profiles at fractions 0.20 to 0.37 (pools not read), H20
at the row limit, and any profile with max_running_requests raised past
the row count in the table. On fp8 H100 and on H200 at the default 32
rows it changes nothing, and that is what section 6 measured.

One more bound on the gain even where it binds: sglang's own decay
reaches its floor after 600 decode steps without an idle iteration, and
the idle reset returns it to 0.7. Under continuous saturated traffic the
two arms therefore converge after about 600 steps (roughly 15 s of talker
decode), and the gain lives in the first 600 steps after every idle gap,
which is what bursty TTS traffic and a 50 request benchmark both are.

## 7. The clean branch (2026-09-02)

perf/observed-kv-reservation at 81a87e474, one commit on main 216e946dd,
carries the same change as 9769867a0 with the tracker class inside
omni_scheduler.py (no new module): the class, the construction in
__init__, the push in get_new_batch_prefill and the observe call in
stream_output, plus the two test files. The scheduling unit tests pass
locally against the Apple venv (227 passed, 3 skipped for CUDA). The
runner defaults now point at 216e946dd against 81a87e474. The PR is not
opened yet.

The two measurements left before the PR opens, on that pair:

1. Voice clone on the bf16 colocated profile at c1, c16 and c32 with WER
   and UTMOS, the profile where the change binds:
   `PROFILE=bf16 SLIM_STAGES=seedtts bash slim_ab.sh` then `slim_ab.sh score`.
2. MiniMax Music 3 through the cookbook requests on both arms
   (minimax_cookbook_ab.sh), five sequential wavs byte identical across
   the arms and the parallel wall time no worse.

The fp8 slim run of section 6 stands as the neutral case and is not
repeated.

### 6.5 Memory split or estimate

Read from the doc 08 bf16 serve logs, lines attributed to a stage by the
pool their token usage implies (5248 thinker, 20136 talker):

| arm | talker running max | talker queued max | talker pool usage max | thinker running max | thinker queued |
|---|---|---|---|---|---|
| A 68c88dae6 | 9 | 27 | 0.07 (1.4k of 20136 tokens) | 3 | 0 |
| B | 32 | 27 | 0.23 (4.6k of 20136 tokens) | 4 | 0 |

The talker's pool holds 32 voice clone rows at under a quarter of its
size. What stopped A at 9 rows was the estimate, 2867 reserved tokens per
row for outputs of about 45 frames, not the memory. To satisfy that
estimate at 32 rows the talker would need about 101k tokens, 4.6 GB of
KV and five times what the rows use, and on this card that memory does
not exist: the thinker's 0.73 leaves it 0.48 GB of KV over 56.9 GB of
bf16 weights, so nothing can move from it to the talker. The bf16 H100
colocated split is at the card's limit, and it is sized correctly for
the workload once the estimate is right.

A memory side calculation that sized each stage from rows times expected
tokens would need the expected output length per stage, the quantity the
tracker measures at runtime, as a static constant. It would also change
nothing on fp8 H100 or H200, whose pools already exceed the estimate.
sglang's 0.7 is a fair guess for chat, where outputs approach the cap. It
is wrong for TTS stages whose cap is a safety stop 60 to 90 times the
typical output, which is a property of Omni's request builders, and the
measured fraction is the generic correction at the scheduler seam. The
thinker on this profile carries the benchmark's max_tokens of 256
(benchmarks/tasks/tts.py:779) and was never queued.
