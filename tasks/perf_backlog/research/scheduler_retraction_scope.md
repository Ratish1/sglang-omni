# Scope of `_compact_decode_input_history` across sglang-omni engine stages

Branch under study: `perf/qwen3-tts-predictor-chain` at `0a88253c6`, worktree
`/Users/ratish/sglang-omni/.worktrees/qwen3-tts-predictor-chain`. Base main commit `7989a5ed2`.
Pinned sglang: `/Users/ratish/sglang`, `git describe --tags` prints `v0.5.18` (confirmed).

The change under study is two additions to
`/Users/ratish/sglang-omni/.worktrees/qwen3-tts-predictor-chain/sglang_omni/scheduling/omni_scheduler.py`:

- `_compact_decode_input_history(data)` at lines 87-94.
- `OmniScheduler._add_request_to_queue(self, req, is_retracted=False)` at lines 2197-2200.

```python
# omni_scheduler.py:87-94
def _compact_decode_input_history(data: Any) -> None:
    history = data.decode_input_embeds
    if not history:
        return
    data.decode_input_embeds = list(torch.stack(history).unbind(0))

# omni_scheduler.py:2197-2200
def _add_request_to_queue(self, req: Any, is_retracted: bool = False) -> None:
    if req.is_retracted:
        _compact_decode_input_history(req._omni_data)
    _Upstream._add_request_to_queue(self, req, is_retracted=is_retracted)
```

The guard is `req.is_retracted`, the Req flag, not the `is_retracted` keyword argument. Those two are
not the same predicate. See section 4.

---

## Per-model table

"Engine stage runs OmniScheduler" means the stage constructs `OmniScheduler` or a subclass of it.
"Populates history" means some code in the repository appends rows to `decode_input_embeds` for that
model's request data.

| Model | Engine stage | Scheduler class | Request data class on `req._omni_data` | Has `decode_input_embeds` | Populates history | Retraction reachable | History consumer on re-prefill |
|---|---|---|---|---|---|---|---|
| qwen3_omni thinker | thinker AR | `OmniScheduler` (bootstrap.py:112) | `SGLangARRequestData` (request_builders.py:697) | yes, inherited, always `[]` | no | yes (KV pressure, TEST switch, omni retract pause) | none, list stays empty |
| qwen3_omni talker | talker AR | `QwenTalkerScheduler` (bootstrap.py:244) | `SGLangARRequestData` (request_builders.py:850) | yes | yes, `QwenTalkerModelRunner._write_feedback_buffers` (talker_model_runner.py:353-386) | yes, same three routes | `QwenTalkerModelRunner._generated_prefill_slice` (talker_model_runner.py:318-351) |
| qwen3_tts | TTS AR | `OmniScheduler` (engine_factory.py:492) | `Qwen3TTSSGLangRequestData` (request_builders.py:1428) | yes, inherited | yes, `Qwen3TTSModelRunner._write_feedback_buffers` (model_runner.py:288-348) | yes, same three routes | `Qwen3TTSModelRunner._build_prefill_input_embeds` (model_runner.py:366-397) into `_generated_prefill_slice` |
| ming_omni thinker | thinker AR | `OmniScheduler` (bootstrap.py:96) | `SGLangARRequestData` (bootstrap.py:125) | yes, inherited, always `[]` | no | yes | none |
| minimax_music3 | AR | `MiniMaxMusic3Scheduler` (engine_builder.py:126) | `MiniMaxMusic3SGLangRequestData` (sglang_request_builder.py:106,119) | yes, inherited, always `[]` | no | yes | none |
| arkasr | ASR AR | `OmniScheduler` (engine_factory.py:419) | `ArkASRRequestData` (request_builders.py:215) | yes, inherited | no | yes | none |
| dots_tts | TTS AR | `OmniScheduler` (engine_factory.py:492) | `DotsTTSSGLangRequestData` (request_builders.py:115) | yes, inherited | no | yes | none |
| fishaudio_s2_pro | TTS AR | `OmniScheduler` | `S2ProSGLangRequestData` (request_builders.py:135) | yes, inherited | no | yes | none |
| fun_asr | ASR AR | `OmniScheduler` | `FunASRRequestData` (request_builders.py:313) | yes, inherited | no | yes | none |
| fun_cosyvoice3 | TTS AR | `OmniScheduler` | `CosyVoice3SGLangRequestData` (request_builders.py:793) | yes, inherited | no | yes | none |
| higgs_tts | TTS AR | `OmniScheduler` | `HiggsSGLangRequestData` (request_builders.py:118) | yes, inherited | no | yes | none |
| ming_tts | TTS AR | `OmniScheduler` | `MingTTSSGLangRequestData` (engine_io.py:86) | yes, inherited | no | yes | none |
| moss_transcribe_diarize | ASR AR | `OmniScheduler` | `MossTranscribeDiarizeRequestData` (request_builders.py:504) | yes, inherited | no | yes | none |
| moss_tts | TTS AR | `OmniScheduler` | `MossTTSSGLangRequestData` (request_builders.py:748) | yes, own field at request_builders.py:84, never written | no | yes | none |
| moss_tts_local | TTS AR | `OmniScheduler` | `MossTTSLocalSGLangRequestData` (request_builders.py:376) | **no** | no | yes | none, and the wrapper raises `AttributeError`, see section 5 |
| qwen3_asr | ASR AR | `OmniScheduler` | `Qwen3ASRRequestData` (request_builders.py:385) | yes, inherited | no | yes | none |
| voxtral_tts | TTS AR | `OmniScheduler` | `VoxtralSGLangRequestData` (request_builders.py:71) | yes, inherited | no | yes | none |
| whisper_asr | ASR AR | `OmniScheduler` | `WhisperASRRequestData` (request_builders.py:310) | yes, inherited | no | yes | none |
| zonos2 | TTS AR | `OmniScheduler` | `Zonos2SGLangRequestData` (request_builders.py:221) | yes, inherited | no | yes | none |
| audar_tts | none | `SimpleScheduler` only (stages.py:167,209,303,340) | not applicable | not applicable | no | not applicable | not applicable |
| llada2_uni | thinker | `DllmScheduler` (bootstrap.py:64) | `SGLangDLLMRequestData` (request_builders.py:158) | no | no | not applicable, see section 6 | not applicable |

"Retraction reachable" is a statement about code paths, not about whether any shipped deployment
actually hits KV pressure. Section 6 separates the two.

---

## 1. Construction and delegation

### Composition, not inheritance

`OmniScheduler` is a plain class (omni_scheduler.py:166). It never subclasses
`sglang.srt.managers.scheduler.Scheduler`, which it imports as `_Upstream` (omni_scheduler.py:37).
Its `__init__` (omni_scheduler.py:181-511) mirrors by hand the instance state that upstream methods
read off `self`, including `waiting_queue` (364), `running_batch` (365), `num_retracted_reqs` (374),
`disaggregation_mode = DisaggregationMode.NULL` (465), `dllm_config = None` (444),
`try_preemption = server_args.enable_priority_scheduling` (407), and the scheduler component objects
built in `_init_upstream_scheduler_components` (589-702).

Delegation is `__getattr__` at omni_scheduler.py:717-741:

```python
attr = getattr(_Upstream, name)
if callable(attr):
    return types.MethodType(attr, self)
```

`getattr(_Upstream, name)` walks the upstream MRO, which is
`Scheduler(SchedulerDisaggregationDecodeMixin, SchedulerDisaggregationPrefillMixin,
SchedulerMultiplexMixin, SchedulerPPMixin, SchedulerDllmMixin, SchedulerMlxOverlapMixin)`
(sglang scheduler.py:383-390). Any upstream method not defined on `OmniScheduler` is bound to the
omni instance and runs against omni state. Methods defined on `OmniScheduler` shadow the upstream
ones, because `__getattr__` only fires on a normal lookup miss.

Two upstream methods are also called explicitly rather than through the descriptor:
`_Upstream.get_next_batch_to_run(self, self.running_batch, self.last_batch)` at omni_scheduler.py:1340,
and `_Upstream._add_request_to_queue(self, req, is_retracted=is_retracted)` at omni_scheduler.py:2200.
The second is the delegation inside the new wrapper.

### The two subclasses

`QwenTalkerScheduler(OmniScheduler)` (qwen3_omni/talker_scheduler.py:39) and
`MiniMaxMusic3Scheduler(OmniScheduler)` (minimax_music3/scheduler.py:13) are the only subclasses.
Neither overrides `_add_request_to_queue`. `QwenTalkerScheduler` overrides `get_next_batch_to_run`
(talker_scheduler.py:110-115), which calls `super().get_next_batch_to_run()` and so still reaches
`_Upstream.get_next_batch_to_run`, hence still reaches `update_running_batch`.

### Construction sites

- `SGLangGenerationEngineBuilder._make_scheduler` at engine_factory.py:388-419 constructs
  `omni_scheduler.OmniScheduler(**scheduler_kwargs)`. This covers every `AsrEngineBuilder` subclass.
- `TtsEngineBuilder.make_scheduler` at engine_factory.py:477-505 constructs `OmniScheduler` directly.
  This covers every `TtsEngineBuilder` subclass except MiniMax.
- `MiniMaxMusic3EngineBuilder.make_scheduler` at minimax_music3/engine_builder.py:123-132 constructs
  `MiniMaxMusic3Scheduler`.
- `qwen3_omni/bootstrap.py:112` constructs `OmniScheduler` for the thinker.
- `qwen3_omni/bootstrap.py:244` constructs `QwenTalkerScheduler` for the talker, then binds the
  runner at 266 through `bind_model_runner` (omni_scheduler.py:513-542).
- `ming_omni/bootstrap.py:96` constructs `OmniScheduler` for the Ming thinker.

### Upstream callers of `_add_request_to_queue` in v0.5.18

Every call site in pinned sglang, and whether omni can reach it:

| Site | Enclosing code | Reachable from OmniScheduler |
|---|---|---|
| scheduler.py:3552, `is_retracted=True` | `update_running_batch` KV-pressure retraction | **yes**, this is the primary path |
| scheduler.py:3377 | `_get_new_batch_prefill_raw`, `adder.preempt_list` drain | yes but configuration gated, see below |
| scheduler.py:3193 | `_get_new_batch_prefill_raw`, ready grammar requests | no, `_NoOpGrammarManager.has_waiting_grammars` returns False (omni_scheduler.py:150-151) |
| scheduler.py:2483, 2505, 2515, 2529, 2542, 2554, 2566, 2596, 2610, 2636, 2654, 2675, 2680 | `handle_generate_request` | no, omni never calls it. Omni admits through `process_input_requests` (omni_scheduler.py:844) and `_enqueue_built_request`, which appends to `self.waiting_queue` directly at omni_scheduler.py:1215 |
| scheduler.py:2895, 2905, 2910 | `handle_embedding_request` | no, same reason |
| scheduler.py:4649 | upstream `pause_generation` retract mode (scheduler.py:4584) | no, omni implements its own `_admin_pause_generation` (omni_scheduler.py:1951) and never calls `pause_generation`. The only omni references to that name are the HTTP route (serve/openai_api.py:456), the client (client/client.py:321) and the coordinator (pipeline/coordinator.py:251), all of which reach `ADMIN_PAUSE_GENERATION` (proto/admin.py:10), routed at omni_scheduler.py:1900 |
| disaggregation/decode.py:2280 | `get_new_prebuilt_batch` | no, `disaggregation_mode` is `NULL` (omni_scheduler.py:465) |
| dllm/mixin/scheduler.py:265 | `_update_state_for_batch` | no, `dllm_config` is `None` (omni_scheduler.py:444), so `get_next_batch_to_run` takes the non-dLLM branch at scheduler.py:3104-3109 |

The preemption path at scheduler.py:3375-3377 also produces requests with `is_retracted` set true,
because `PrefillAdder.preempt_to_schedule` calls `self.running_batch.release_req(...)`
(schedule_policy.py:1493) which reaches `release_req` -> `req.reset_for_retract()`
(schedule_batch.py:1939). It is gated on `self.enable_priority_preemption` at scheduler.py:3312,
which omni computes at omni_scheduler.py:547-550 from
`server_args.enable_priority_scheduling and not server_args.disable_priority_preemption`.
`enable_priority_scheduling` defaults to `False` in sglang (server_args.py:844-848) and no
sglang_omni file assigns it. The only omni references are reads
(omni_scheduler.py:403, 406, 407, 548). It is settable through a stage's
`server_args_overrides`, so this is a configuration-gated path rather than a dead one.

### Omni's own caller

`OmniScheduler._retract_running_requests` (omni_scheduler.py:2202-2225) calls
`self._add_request_to_queue(req)` at line 2222 with the default `is_retracted=False`, on requests
whose `is_retracted` was just set true by `retract_all` (line 2212). So it goes through the new
compaction even though it passes `is_retracted=False`.

---

## 2. Where `req._omni_data` is attached and what it is

### Attachment

`_enqueue_built_request` sets `req._omni_data = req_data` at omni_scheduler.py:1214, just before the
request is appended to `self.waiting_queue` at 1215. `req_data` is whatever the stage's
`request_builder` callable returned (omni_scheduler.py:940). `MiniMaxMusic3Scheduler._enqueue_cfg_uncond`
performs the same attachment for the CFG unconditional row at minimax_music3/scheduler.py:47.

Detachment is `_detach_request_data` at omni_scheduler.py:97-99, which sets `req._omni_data = None`.
It is called from `abort` for waiting-queue entries (line 1817), from `stream_output` on the aborted
branch (1638), from `_close_completed_request` (2639), and from `_remove_from_batch` (2692).

### Class hierarchy

- `ARRequestData` (scheduling/types.py:72-87). Backend-neutral. **No `decode_input_embeds`.**
- `SGLangARRequestData(ARRequestData)` (sglang_backend/request_data.py:14-33). Declares
  `prefill_input_embeds` (line 25) and `decode_input_embeds: list["torch.Tensor"] = field(default_factory=list)`
  (line 26), plus `pending_feedback_queue` (28), `pending_text_queue` (29), `tts_pad_embed` (30),
  `thinker_chunks_done` (32).
- `SGLangDLLMRequestData` (sglang_backend/request_data.py:37-43). Standalone, no
  `decode_input_embeds`. Used only by llada2_uni.

Model-specific subclasses of `SGLangARRequestData`, all of which inherit `decode_input_embeds`:
`Zonos2SGLangRequestData` (zonos2/request_builders.py:125),
`MossTranscribeDiarizeRequestData` (moss_transcribe_diarize/request_builders.py:59),
`DotsTTSSGLangRequestData` (dots_tts/request_builders.py:32),
`FunASRRequestData` (fun_asr/request_builders.py:44),
`CosyVoice3SGLangRequestData` (fun_cosyvoice3/request_builders.py:160),
`WhisperASRRequestData` (whisper_asr/request_builders.py:49),
`ArkASRRequestData` (arkasr/request_builders.py:50),
`MingTTSSGLangRequestData` (ming_tts/engine_io.py:30),
`HiggsSGLangRequestData` (higgs_tts/request_builders.py:33),
`MiniMaxMusic3SGLangRequestData` (minimax_music3/sglang_request_builder.py:37),
`Qwen3TTSSGLangRequestData` (qwen3_tts/request_builders.py:124),
`Qwen3ASRRequestData` (qwen3_asr/request_builders.py:72),
`S2ProSGLangRequestData` (fishaudio_s2_pro/request_builders.py:22),
`VoxtralSGLangRequestData` (voxtral_tts/request_builders.py:20).

Two model classes subclass `ARRequestData` directly instead:

- `MossTTSSGLangRequestData(ARRequestData)` (moss_tts/request_builders.py:74) redeclares
  `decode_input_embeds: list[torch.Tensor] = field(default_factory=list)` at line 84. Nothing in the
  repository writes to it for moss_tts, so it stays empty.
- `MossTTSLocalSGLangRequestData(ARRequestData)` (moss_tts_local/request_builders.py:35-66) declares
  no such field. Reading `data.decode_input_embeds` on it raises `AttributeError`.

### `req.is_retracted` in pinned sglang

`Req.__init__` initializes `self.is_retracted = False` at schedule_batch.py:1024.
`Req.reset_for_retract` sets `self.is_retracted = True` at schedule_batch.py:1684 and
`self.retracted_stain = True` at 1685. It is called from module-level `release_req`
at schedule_batch.py:1939, which is the shared tail of both `ScheduleBatch.retract_decode`
(via `ScheduleBatch.release_req`, schedule_batch.py:2907-2923) and `retract_all`
(schedule_batch.py:1942-1962).

The flag is cleared at `ScheduleBatch.prepare_for_extend`, schedule_batch.py:2497
(`req.is_retracted = False`), that is, when the retracted request is actually re-prefilled.
Between retraction and re-prefill the flag stays true, so any `_add_request_to_queue` call on the
request in that window triggers compaction regardless of the keyword argument.

`reset_for_retract` also discards generated tokens when `req.input_embeds is not None`
(schedule_batch.py:1711-1712). Both talker models build their `Req` with `input_embeds=None`
(qwen3_omni/request_builders.py:797) or without passing it at all
(qwen3_tts/request_builders.py:1410-1418, where the sglang default is `None` per
schedule_batch.py:821, 877), so `output_ids` survives retraction for them.

Omni's own retract path sets the flag through the same upstream function:
`_retract_running_requests` calls `retract_all(...)` at omni_scheduler.py:2212. Nothing else in
sglang_omni assigns `is_retracted`. All other omni references are reads
(omni_scheduler.py:2260, 2381, 2431, model_runner/base.py:896, models/zonos2/callbacks.py:79,
models/dots_tts/model_runner.py:318, models/moss_tts_local/model_runner.py:672).

---

## 3. Who appends rows, and whether they are views

Repository-wide, the only writers of `decode_input_embeds` are:

- `QwenTalkerModelRunner._decode_input_history` (talker_model_runner.py:424-429), which lazily
  installs a list when the field is `None`.
- `QwenTalkerModelRunner._append_decode_input_history` (talker_model_runner.py:431-435), which
  appends `row.detach()`.
- `_compact_decode_input_history` (omni_scheduler.py:94), the new function.
- `stream_output` terminal cleanup, which sets the field to `None` (omni_scheduler.py:1677).

`_append_decode_input_history` has exactly three call sites:
talker_model_runner.py:343, talker_model_runner.py:372, and qwen3_tts/model_runner.py:344.

### Qwen3-Omni talker: independent per-row allocations

`QwenTalkerModelRunner._write_feedback_buffers` (talker_model_runner.py:353-386) loops rows and calls
`_take_next_decode_input_embed` (498-513), which calls `_combine_feedback_with_next_text` (477-495).
That returns `_decode_row(feedback, ...) + _decode_row(next_text, ...)`. `_decode_row` (437-451) only
does `row.reshape(-1)` and a device/dtype check, so it yields a view, but the `+` between the two
views allocates a fresh one-dimensional tensor. The row appended at line 372 is therefore its own
allocation of `hidden` elements, not a slice of a batch tensor.

The upstream feedback rows do come from a batched snapshot: `_emit_code_chunks_and_feedback`
(talker_model_runner.py:126-157) makes `embeds_snap = self.model._output_embeds[:bs].detach().clone()`
at line 137 and pushes `embeds_snap[idx]` onto `pending_feedback_queue` at line 157. Those rows are
views of the `(bs, hidden)` snapshot, but they are consumed and dropped by the `+` above, so they do
not enter `decode_input_embeds`.

Consequence: for the talker, `_compact_decode_input_history` replaces N independent `(hidden,)`
allocations with N views of one `(N, hidden)` allocation. It changes allocation count and layout, not
the set of large tensors kept alive.

### Qwen3-TTS: rows are views of the per-step snapshot

`Qwen3TTSModelRunner._write_feedback_buffers` (qwen3_tts/model_runner.py:288-348) writes the whole
decode batch into the staged embedding weight and then clones once:

```
target  = weight[:batch_size]                       # model_runner.py:328
...     torch.stack(feedback_rows, dim=0, out=target)   # 330
        target.add_(torch.stack(text_rows, dim=0))      # 331
history = target.detach().clone()                   # 342, one (bs, hidden) allocation per step
for row_idx, sched_req in enumerate(requests):      # 343
    QwenTalkerModelRunner._append_decode_input_history(sched_req.data, history[row_idx])  # 344-346
```

`history[row_idx]` is a view into that per-step `(bs, hidden)` clone. Every request in the batch
retains a reference to the same snapshot object. A request that has run K decode steps holds K rows
that between them pin K distinct `(bs, hidden)` snapshots, so the retained bytes scale as
`K * bs * hidden` rather than `K * hidden`. That is the situation the new docstring at
omni_scheduler.py:88-90 describes.

This is new on the branch. `git diff 7989a5ed2..HEAD -- sglang_omni/models/qwen3_tts/model_runner.py`
shows the pre-branch loop appended `combined`, a per-row result of
`QwenTalkerModelRunner._take_next_decode_input_embed(...)` or a fresh embedding lookup, both
independent allocations. The batching commit turned per-row appends into snapshot views, and the
inline note at model_runner.py:340-341 records the intent.

### How the history is replayed on re-prefill

Both models replay through `QwenTalkerModelRunner._generated_prefill_slice`
(talker_model_runner.py:318-351):

```
history = QwenTalkerModelRunner._decode_input_history(data)      # 331
while len(history) < gen_end:                                    # 332
    combined = _take_next_decode_input_embed(...)                # 333
    if combined is None: raise RuntimeError(...)                 # 338-342
    _append_decode_input_history(data, combined)                 # 343
rows = [_decode_row(row, ...) for row in history[gen_start:gen_end]]  # 345-348
return torch.stack(rows, dim=0)                                  # 351
```

Note the top-up loop at 332-343. If the request generated more tokens than the recorded history, for
example a token whose feedback row was still queued when the retraction landed, the loop drains
`pending_feedback_queue` and `pending_text_queue` to extend the history in place before slicing.

Callers:

- Qwen3-Omni talker: `before_prefill` (talker_model_runner.py:40-57) -> `_compose_prefill_embeds`
  (179-230) -> `_projected_prefill_slice` (232-283) -> `_generated_prefill_slice` at line 271.
- Qwen3-TTS: `before_prefill` (qwen3_tts/model_runner.py:41-57) -> `_build_prefill_input_embeds`
  (366-397) -> `QwenTalkerModelRunner._projected_prefill_slice` at line 381 -> same
  `_generated_prefill_slice`.

Qwen3-TTS additionally reseeds the sglang repetition penalizer on the re-prefill through
`_execution_context` (qwen3_tts/model_runner.py:28-39) calling `_restore_repetition_penalty_history`
(130-177), because `prepare_for_extend` builds a fresh sampling state while the retained
`req.output_ids` survive retraction.

No other model in `sglang_omni/models/*` reads `decode_input_embeds`. Grep over the whole worktree for
the field name returns only `omni_scheduler.py`, `sglang_backend/request_data.py`,
`moss_tts/request_builders.py` (declaration only), `qwen3_omni/talker_model_runner.py`,
`qwen3_tts/model_runner.py`, and tests.

### ASCII flow, Qwen3-TTS

```
decode step
  OmniScheduler._event_loop_normal            omni_scheduler.py:2296
    get_next_batch_to_run                     omni_scheduler.py:1332  -> _Upstream 3015
      update_running_batch                    sglang scheduler.py:3481
        batch.check_decode_mem()              schedule_batch.py:2809      <-- False under KV pressure
        batch.retract_decode(server_args)     schedule_batch.py:2816
          ScheduleBatch.release_req           schedule_batch.py:2907
            release_req -> reset_for_retract  schedule_batch.py:1939, 1668
              req.is_retracted = True         schedule_batch.py:1684
        for req in retracted_reqs:
          self._add_request_to_queue(req, is_retracted=True)   scheduler.py:3552
            |
            +--> OmniScheduler._add_request_to_queue          omni_scheduler.py:2197
                   if req.is_retracted:                       omni_scheduler.py:2198
                     _compact_decode_input_history(req._omni_data)   omni_scheduler.py:2199
                       torch.stack(history).unbind(0)         omni_scheduler.py:94
                   _Upstream._add_request_to_queue(...)       omni_scheduler.py:2200
                     self.waiting_queue.append(req)           sglang scheduler.py:2725

    run_batch (decode)                        omni_scheduler.py:1382
      Qwen3TTSModelRunner._write_feedback_buffers    qwen3_tts/model_runner.py:288
        history = target.detach().clone()            qwen3_tts/model_runner.py:342
        _append_decode_input_history(data, history[row_idx])   qwen3_tts/model_runner.py:344

re-prefill of the requeued request
  get_new_batch_prefill                       omni_scheduler.py:1346 -> _Upstream 3157
    ScheduleBatch.prepare_for_extend          schedule_batch.py:2419
      req.is_retracted = False                schedule_batch.py:2497
  run_batch (extend)                          omni_scheduler.py:1382
    Qwen3TTSModelRunner.before_prefill        qwen3_tts/model_runner.py:41
      _build_prefill_input_embeds             qwen3_tts/model_runner.py:366
        QwenTalkerModelRunner._projected_prefill_slice   talker_model_runner.py:232
          _generated_prefill_slice                       talker_model_runner.py:318
            reads data.decode_input_embeds               talker_model_runner.py:331
```

---

## 4. What the wrapper does to each model's requests

Given the table and the mechanics above, the wrapper partitions the models into four behaviours.

**Qwen3-TTS.** Compaction is load bearing. On every retraction and on every omni retract pause, the
request's K history rows are restacked into one `(K, hidden)` tensor, dropping references to K
per-step `(bs, hidden)` snapshots. The re-prefill then reads the compacted rows through
`_generated_prefill_slice`. The rows are numerically identical, `torch.stack` copies the same values,
so the replayed prefill embeds are unchanged.

**Qwen3-Omni talker.** Compaction runs and succeeds, but the rows it consolidates were already
independent per-row allocations (section 3). It coalesces K small allocations into one contiguous
buffer. It does not release any snapshot, because none is held.

**Every other `SGLangARRequestData` subclass**, that is thinker stages, ASR stages, and the other TTS
stages listed in the table. `decode_input_embeds` is the inherited `field(default_factory=list)` from
sglang_backend/request_data.py:26 and nothing ever appends to it, so the guard `if not history:` at
omni_scheduler.py:92-93 returns immediately. The wrapper costs one attribute read and one truth test
per retracted request. Also applies to moss_tts, which declares its own always-empty field at
moss_tts/request_builders.py:84.

**moss_tts_local.** `MossTTSLocalSGLangRequestData` (moss_tts_local/request_builders.py:35-66)
subclasses `ARRequestData` (scheduling/types.py:72), which has no `decode_input_embeds`. The
dataclass declares no such field either. Reading `data.decode_input_embeds` at omni_scheduler.py:91
raises `AttributeError` on a `MossTTSLocalSGLangRequestData` instance. This makes any retraction on
the moss_tts_local engine stage an exception rather than a requeue.

---

## 5. Requests that can reach the wrapper without usable `_omni_data`

Two distinct failure shapes exist, and only one of them is reachable through omni's own code paths.

### Missing field, reachable

moss_tts_local, as described above. The engine stage builds an `OmniScheduler` through
`MossTtsLocalEngineBuilder(TtsEngineBuilder)` (moss_tts_local/engine_builder.py:19) and
`TtsEngineBuilder.make_scheduler` (engine_factory.py:492). Nothing about the model prevents
retraction. The wrapper's read is unguarded, so a retraction under KV pressure on that stage raises.
The same argument applies to any future request data class that does not inherit
`SGLangARRequestData`.

### `_omni_data is None`, not established as reachable

The wrapper does not guard against `req._omni_data is None`, and `_compact_decode_input_history`
reads `data.decode_input_embeds` before any None check. Tracing the omni detachment sites:

- `abort` at omni_scheduler.py:1814-1820 detaches only requests it simultaneously removes from
  `waiting_queue`, so those never re-enter `_add_request_to_queue`.
- `abort` with `defer_running_cleanup=False` reaches `_remove_from_batch` (1837-1840), which both
  detaches and removes the request from `running_batch.reqs` (omni_scheduler.py:2686-2697). A request
  removed from the batch is not visible to `retract_decode`.
- `abort` with the default `defer_running_cleanup=True` calls `_mark_running_request_aborted`
  (2240-2264), which sets `to_finish = FINISH_ABORT()` but leaves `_omni_data` intact, and which
  explicitly skips requests already retracted at line 2260.
- `stream_output` detaches on the aborted branch (1638) and through `_close_completed_request` (2639)
  on the terminal branch. Both apply to requests that have finished, and `retract_decode` runs after
  `batch.filter_batch()` at sglang scheduler.py:3485.

So in the paths read here a request reaching the wrapper with `is_retracted` true still carries live
request data. I did not find a concrete interleaving that produces `_omni_data is None` at that point.
Whether one exists under the async decode loop's lagged batches
(`_resolve_and_process`, omni_scheduler.py:2364, and `_drop_stale_overrun`, 2416) is **UNVERIFIED**:
those paths drop retracted requests from the lagged batch rather than requeue them, so they do not
themselves call `_add_request_to_queue`, but I did not exhaustively enumerate abort-versus-retract
interleavings across threads.

---

## 6. Is retraction reachable in each shipped configuration

Three distinct entry routes exist. All three are model independent, so the answer is the same for
every model whose engine stage runs `OmniScheduler`.

### Route A, KV pressure

`update_running_batch` at sglang scheduler.py:3491 tests
`not batch.check_decode_mem()`. `ScheduleBatch.check_decode_mem` (schedule_batch.py:2809-2814):

```python
num_tokens = self.new_tokens_required_next_decode(selected_indices)
evict_from_tree_cache(self.tree_cache, num_tokens)
return self.token_to_kv_pool_allocator.available_size() >= num_tokens
```

`new_tokens_required_next_decode` (schedule_batch.py:2782-2796) with no speculative decoding, which is
omni's case (`spec_algorithm = SpeculativeAlgorithm.NONE`, omni_scheduler.py:443), counts one page per
request sitting exactly on a page boundary:
`new_pages = sum(1 for r in requests if r.kv_committed_len % page_size == 0)` then
`new_pages * page_size`. So the check fails when the free KV pool cannot cover the pages the next
decode step needs after tree eviction.

Retraction itself is `ScheduleBatch.retract_decode` (schedule_batch.py:2816-2864). It pops from the
back of `_get_decode_retraction_order` until `check_decode_mem` passes, always keeps at least one
request (2827-2829), and if even the last request does not fit it aborts that request with
`FINISH_ABORT` rather than retracting it (2839-2855).

Whether the pool can be exhausted is a runtime property of the KV pool size, which omni sizes per
stage either from `mem_fraction_static` in the builder's `generation_defaults`, for example
`0.85` for Qwen3-TTS (qwen3_tts/engine_builder.py:91), or from a stage's declared
`engine.kv_cache_bytes` threaded through `stage_kv_budget.stage_kv_cache_budget`
(scheduling/stage_kv_budget.py:34-60) and consumed once at engine construction
(`consume_stage_kv_cache_bytes`, 63-80). When `kv_cache_bytes` is declared, the builder drops its
default `mem_fraction_static` (engine_factory.py:136-145).

Omni bounds a single request against the pool at admission. `_request_kv_capacity_error`
(omni_scheduler.py:1292-1319) rejects any request whose `len(origin_input_ids) + max_new_tokens`
exceeds `self.max_req_len`, which is
`min(server_args.context_length - 1, effective_max_total_num_tokens - 1)`
(omni_scheduler.py:329-332). So one request alone cannot exhaust the pool. Retraction requires an
aggregate over-subscription across concurrently running requests, which is possible whenever
`max_running_requests` times the realized sequence lengths exceeds the pool. For Qwen3-TTS the shipped
defaults are `max_running_requests = 16`, `max_queued_requests = 16`, `context_length = 8192`
(qwen3_tts/engine_builder.py:41, 83-84). Whether 16 concurrent 8192-token sequences fit in the pool at
`mem_fraction_static = 0.85` on a given GPU is a runtime number the code computes at startup and is
**UNVERIFIED** here, since it depends on model size, dtype and free device memory.

### Route B, the sglang test switch

`sglang/srt/environ.py:387-389` defines `SGLANG_TEST_RETRACT = EnvBool(False)`,
`SGLANG_TEST_RETRACT_INTERVAL = EnvInt(3)` and `SGLANG_TEST_RETRACT_NO_PREFILL_BS = EnvInt(2**31)`.
They are read once at import into module constants at sglang scheduler.py:352-354.

`update_running_batch` at scheduler.py:3491-3493 fires retraction on
`(kv_full_retract_flag := not batch.check_decode_mem()) or (TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0)`.
`OmniScheduler` maintains `self.forward_ct` itself, incrementing it in `_stamp_batch_launch`
(omni_scheduler.py:1396-1402), so the modulo has a live counter. With `SGLANG_TEST_RETRACT=1` set in
the environment, every third forward retracts, on every omni model, with no KV pressure required.
`SGLANG_TEST_RETRACT_NO_PREFILL_BS` additionally suppresses prefill above a running batch size
(scheduler.py:3235-3239). No sglang_omni source file sets or reads these variables. The only omni
mention is a test comment at tests/unit_test/pipeline/test_scheduler.py:772.

### Route C, the omni retract pause

`_admin_pause_generation` (omni_scheduler.py:1951-1977) accepts `mode` in `{"abort", "retract", "in_place"}`
and calls `_retract_running_requests` for `mode == "retract"` at line 1968.
`_retract_running_requests` (2202-2225) filters the running batch, snapshots `batch.reqs`, calls
`retract_all(...)` (2212-2219), clears `batch.reqs` (2220), then requeues each request through
`self._add_request_to_queue(req)` at line 2222. `retract_all` reaches `reset_for_retract`, so every
one of those requests has `is_retracted` true when the wrapper sees it, even though the wrapper passes
`is_retracted=False` downstream.

The mode is operator supplied end to end: `POST /pause_generation` (serve/openai_api.py:455-465) with
`PauseGenerationRequest.mode`, default `"abort"` (serve/protocol.py:608-609), through
`Client.pause_generation` (client/client.py:321) and `Coordinator.pause_generation`
(pipeline/coordinator.py:251-259) to `ADMIN_PAUSE_GENERATION` (proto/admin.py:10) dispatched at
omni_scheduler.py:1900.

The weight-update lifecycle also depends on this mode. `_can_update_active_requests`
(omni_scheduler.py:2189-2195) only allows a weight update with active requests when the engine was
paused with `mode == "retract"`, and the error message at 2032-2036 tells operators to use exactly
that. So the retract pause is on the documented weight-update path, not just a debug affordance.

Note the scope limit: `_retract_running_requests` only walks `self.running_batch`
(omni_scheduler.py:2203). It does not walk `cur_batch`, `last_batch`, or the async pending batch, and
it sets `self.chunked_req = None` at line 2224 without retracting it.

### Route D, priority preemption

Covered in section 1. Reachable only when a stage sets `enable_priority_scheduling` through
`server_args_overrides`. No sglang_omni file sets it, and the sglang default is `False`
(server_args.py:848).

### Models where retraction does not apply

- `audar_tts` has no `OmniScheduler` stage at all. Its stages are `SimpleScheduler`
  (audar_tts/stages.py:167, 209, 303, 340).
- `llada2_uni` runs `DllmScheduler` (llada2_uni/bootstrap.py:64), a standalone class
  (scheduling/dllm_scheduler.py:29) that shares only the inbox/outbox contract with `OmniScheduler`
  and has no `_add_request_to_queue`, no `update_running_batch`, and no retraction.

---

## 7. Loose ends found while reading

These are observations, not recommendations.

- `sglang_omni/models/ming_omni/pipeline/engine_io.py:108` imports `SGLangARRequestData` from
  `sglang_omni.engines.omni.runtime.sglang_ar`. No `sglang_omni/engines` directory exists in this
  worktree. The import is inside the function body of `build_sglang_thinker_request`, so it only
  fails when that function is called. The Ming OmniScheduler stage does not use it: it uses the
  inline `request_builder` in `ming_omni/bootstrap.py:121-...`, which imports from
  `sglang_omni.scheduling.sglang_backend` at bootstrap.py:125. Whether
  `ming_omni/pipeline/stages.py:153` is reachable in a shipped pipeline is **UNVERIFIED**.

- The wrapper's guard and the upstream keyword disagree by design in two places. Upstream passes
  `is_retracted=True` only at scheduler.py:3552. Omni's own `_retract_running_requests`
  (omni_scheduler.py:2222) and the upstream preemption drain (scheduler.py:3377) both pass the
  default `False` on requests whose `req.is_retracted` is `True`. Since the wrapper reads the Req
  flag, it covers all three. The keyword is only forwarded to upstream, where under
  `DisaggregationMode.NULL` it is unused, see sglang scheduler.py:2721-2726.

- `_compact_decode_input_history` calls `torch.stack(history)` without a device or dtype check.
  `_decode_row` (talker_model_runner.py:437-451) enforces a single device and dtype for rows entering
  the history for both talker models, so within those two producers the stack is homogeneous. For any
  other producer this is **UNVERIFIED**, and today there is no other producer.

- Branch tests for the new behaviour are at
  tests/unit_test/pipeline/test_scheduler.py:444-485 (storage identity and the empty-history no-op)
  and tests/unit_test/qwen3_tts/test_retract_prefill.py:57-163 (history recording and re-prefill
  replay). Neither exercises a request data class lacking the field.
