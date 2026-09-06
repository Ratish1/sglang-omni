# Cross model validation of the predictor chain branch

Branch `perf/qwen3-tts-predictor-chain`. The review fixes changed two files that other models
execute: `sglang_omni/scheduling/omni_scheduler.py` (history compaction at requeue) and
`sglang_omni/models/qwen3_omni/talker_model_runner.py` (the decode input helpers). This plan
records what each model executes of that diff, the hazard the audit found and its fix, and the
box work that validates it, arranged as two lanes that run at the same time on two GPUs. The
audit reports are `research/scheduler_retraction_scope.md` and
`research/talker_helper_scope_and_benchmarks.md`, every claim used here re-read at the cited
lines.

## 1. What each model executes of the diff

| Model | Scheduler | Request data | Executes the helper changes | Executes the compaction | Effect |
| --- | --- | --- | --- | --- | --- |
| Qwen3-TTS | OmniScheduler | SGLangARRequestData subclass | yes, peek and pop per decode step, replay on re-prefill | yes, load bearing: rows are views of one clone per step (model_runner.py:342) | validated by runbook 06 |
| Qwen3-Omni talker | QwenTalkerScheduler(OmniScheduler) | SGLangARRequestData | yes, every decode step through `_take_next_decode_input_embed` (talker_model_runner.py:497-513) and the readiness gate (171-177, 388-396) | yes on retraction, but its rows are fresh sums (476-495), so the stack copies rows that were already independent | host Python only, one copy per retracted request |
| Every other OmniScheduler stage (thinker stages, ASR, other TTS, MiniMax) | OmniScheduler or subclass | SGLangARRequestData subclass with an empty history | no | one attribute read and one truth test per retracted request (omni_scheduler.py:91-93) | none |
| moss_tts, moss_tts_local | OmniScheduler | subclass ARRequestData directly | no | moss_tts_local had no history field, so a retraction raised AttributeError at omni_scheduler.py:91 | fixed, section 2 |
| llada2_uni, audar_tts | DllmScheduler, SimpleScheduler | not attached to this scheduler | no | never reached | none |

Retraction reaches the wrapper on three routes, all model independent: KV pressure in
sglang's `update_running_batch` (scheduler.py:3491, requeue at 3552), the test switch
`SGLANG_TEST_RETRACT` read at import (scheduler.py:352-354, live because the omni scheduler
advances `forward_ct` itself at omni_scheduler.py:1396-1402), and the admin pause endpoint in
retract mode (omni_scheduler.py:1968, 2202-2225), which the weight update lifecycle requires
(omni_scheduler.py:2189-2195). Both sglang paths and the omni pause path mark the request through
`reset_for_retract` (schedule_batch.py:1684, reached from 1939 by both `release_req` callers),
which is the flag the wrapper reads.

```
retraction routes                              wrapper                        per model
KV pressure  scheduler.py:3491 -> 3552 ---+
test switch  scheduler.py:3492 ----------+--> OmniScheduler._add_request_to_queue  2197
pause retract omni_scheduler.py:1968 ----+      if req.is_retracted:              2198
  (retract_all -> reset_for_retract 1684)        _compact_decode_input_history    87-94
                                                    history empty  -> return       every stage but two
                                                    Qwen3-TTS      -> drops K per step (bs, hidden) snapshots
                                                    Qwen3-Omni     -> one (K, hidden) copy of independent rows
                                                  _Upstream._add_request_to_queue 2200
```

## 2. The hazard and its fix

`MossTTSLocalSGLangRequestData` subclasses `ARRequestData` directly and declares neither
history field (moss_tts_local/request_builders.py:35-66), and its stage builds a plain
OmniScheduler through `TtsEngineBuilder.make_scheduler` (engine_factory.py:492). With commit
`c93ba9858` alone, any retraction on that stage raised instead of requeueing.

The scheduler already treated both fields as part of every request's data: it clears
`prefill_input_embeds` and `decode_input_embeds` on every finished request
(omni_scheduler.py:1676-1677), and moss_tts had redeclared both on its own class for that
reason. Commit `d26ac7a1e` declares both fields once on `ARRequestData`
(scheduling/types.py) and removes the duplicate declarations from `SGLangARRequestData` and
`MossTTSSGLangRequestData`. Every class the scheduler attaches derives from `ARRequestData`,
so the contract is now enforced by inheritance. Construction is keyword only everywhere, so the
field order change is inert. The test
`test_retracted_request_with_model_owned_data_is_requeued` requeues a retracted request whose
data is the real moss_tts and moss_tts_local class. The commit is local until pushed.

## 3. Two lanes on the box

Both lanes need the branch head pushed. A is `7989a5ed2` in both lanes.

### Lane 1, Qwen3-TTS, one H100

Runbook 06 in this order, shortest first so the long A/B runs last:

1. reviewer probes (06 §2)
2. coverage at 128 running (06 §3)
3. greedy census (06 §4)
4. retraction run on both arms (06 §5)
5. the full A/B and scoring (06 §6)

### Lane 2, suites and Qwen3-Omni, one H100

1. Suites once, on the branch head, SHA recorded:

```bash
pytest tests/ -v -m "not benchmark and not accelerator" -x
pytest tests/ -v -m "accelerator and not benchmark" -x
```

2. Qwen3-Omni slim A/B, voice clone stage only, the protocol of
`tasks/qwen3_omni_0518_numerics/scripts/slim_ab.sh` (fp8 colocated on one GPU, c1, c16, c32,
then the score pass):

```bash
cp -r "$OMNI_ROOT/tasks/qwen3_omni_0518_numerics/scripts" "$OUT/scripts"
A_SHA=7989a5ed2 B_SHA=<branch head> PROFILE=fp8 SLIM_STAGES=seedtts OMNI_ROOT=... OUT=... GPU=<lane 2 gpu> \
  bash "$OUT/scripts/slim_ab.sh"
OMNI_ROOT=... OUT=... GPU=<lane 2 gpu> bash "$OUT/scripts/slim_ab.sh" score
python "$OUT/scripts/full_ab_compare.py" "$OUT" --md "$OUT/readout.md"
```

Read against the recorded noise floor: two identical c16 runs of the same 50 requests differ
by 9 percent in qps (`06_e0_talker_step.md:31-34`), so c1 latency is the metric that can
resolve a per step host change, and c16 and c32 are regression guards, not a measurement of
the helpers. Serve logs of B: no exception, no `AttributeError`.

3. Qwen3-Omni talker retraction, both arms, bf16 colocated with the test switch scoped to the
talker stage through the stage env, the form the earlier run used (`08_ab_reservation.md`
§4.1): a copy of `examples/configs/qwen3_omni_colocated_h100_bf16.yaml` whose
`stages.talker_ar.env` carries `SGLANG_TEST_RETRACT: "1"` and
`SGLANG_TEST_RETRACT_INTERVAL: "50"`. Voice clone at c16, 50 samples, then the score pass.
Expected in B: `Testing retraction. #retracted_reqs: 1` lines each followed by a re-prefill of
`new_tokens_gained + 1` tokens, every request completes, WER inside the earlier retract run's
band, and no exception. This is the run that executes the compaction on the talker and the
replay through `_generated_prefill_slice` after it.

4. moss_tts_local. The unit test is the gate. A serving smoke with the same stage env at c4
runs only if the box holds the MOSS-TTS Local checkpoint, which is a validation task below.

## 4. Order of the slices

1. Lanes 1 and 2 above, then the chain PR is complete.
2. The memory provisioning slice (backlog item from the S2 readout). The rope store branch's
   c16 failure was cuDNN's plan build in the reference encoder with the card two MiB from
   full, on both arms, so S2 cannot pass c16 before the pool provisioning changes. S2's code is
   done (`5e4424f4a` fixed its unit tests). Recommended seam: a builder default
   `max_total_tokens` derived from `max_running_requests` times the context length, the two
   settings the admission bound already uses, so the pool covers every admitted request and
   KV retraction cannot occur, and the rest of the card stays free for cuDNN and allocator
   bursts. No new constant. The plan doc 05 follows the seam decision.
3. The rope store branch takes the chain commits, A becomes the chain head, and its A/B runs
   with the memory slice applied on both arms.

## 5. Validation tasks

- Whether the Omni talker hits KV retraction naturally at c32 on the fp8 profile: grep the
  lane 2 serve logs for `KV cache pool is full`.
- Whether the box holds a MOSS-TTS Local checkpoint for the c4 smoke.
- The dotted CLI form of a stage env key, if the yaml copy is inconvenient:
  `stages.<stage>.env` is a dict at config/schema.py:368, the CLI form was not exercised.
