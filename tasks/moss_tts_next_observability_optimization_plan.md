# MOSS-TTS Local Next Optimization Observability Plan

Date: 2026-06-13

Branch: `perf/issue-752-moss-tts-compile-investigation`

## Problem Summary

The sampler-only `torch.compile` path is now isolated as the safe compile
boundary for the clean PR. The next MOSS-TTS Local work should stop widening
`torch.compile` and instead profile the rest of the serving stack: frame
orchestration, pool-state movement, async decode launch/resolve, scheduler
batching, radix keying, vocoder/code2wav, and stage handoff.

Success criteria for this observability pass:

- restore MOSS Local frame/profiler instrumentation on the current post-#759
  pool-state/async-decode code, not the older pre-pool `_collect_frame()` shape;
- follow SGLang's profiler style: use `torch.profiler.record_function(...)`
  ranges for Chrome/Perfetto and keep request-event JSONL as the low-overhead
  timeline source;
- add no tensor materialization, no `.item()`, no `.cpu()`, and no syncs in the
  profiler path;
- produce profiles that explain where the remaining RTF is going before any
  next optimization PR is chosen.

Out of scope for this pass:

- new production optimization code;
- resurrecting full-frame or logits/projection `torch.compile`;
- deterministic inference as a production workaround;
- custom kernels before the profile proves which boundary deserves one.

## Current-State Evidence

- `sglang_omni/profiler/torch_profiler.py`
  - Omni's torch profiler runs continuously between `/start_profile` and
    `/stop_profile`; no `schedule(...)` or `step()` is required.
  - Expensive options are env-gated:
    `SGLANG_TORCH_PROFILER_RECORD_SHAPES`,
    `SGLANG_TORCH_PROFILER_PROFILE_MEMORY`,
    `SGLANG_TORCH_PROFILER_WITH_STACK`,
    `SGLANG_TORCH_PROFILER_WITH_FLOPS`.

- `/Users/ratish/sglang/python/sglang/srt/managers/scheduler_pp_mixin.py`
  - SGLang uses direct `with torch.profiler.record_function("...")` ranges in
    scheduler hot paths such as `recv_requests`, `get_next_batch_to_run`, and
    `process_batch_result`.

- `/Users/ratish/sglang/python/sglang/srt/model_executor/model_runner.py`
  - SGLang's model runner adds a forward-step range only when
    `torch.autograd._profiler_enabled()` is active. This avoids overhead for
    every forward in the normal server path.

- `sglang_omni/profiler/event_recorder.py`
  - JSONL events are no-ops when inactive and serialize tensor metadata as
    summaries instead of materializing tensor contents.
  - Stage ownership is carried by the active-stage binding, so model runner code
    can emit with `stage=None`.

- `docs/developer_reference/profiler.md`,
  `sglang_omni/profiler/views.py`, and
  `tests/unit_test/profiler/test_views.py`
  - The debug branch already documents and aggregates MOSS fine-frame intervals
    such as `moss_tts_local_frame_decode_cuda_graph`.
  - Current `sglang_omni/models/moss_tts_local/model_runner.py` no longer emits
    those events after the #759 pool-state rewrite. The docs/views are ahead of
    the runtime implementation.

- `sglang_omni/models/moss_tts_local/model_runner.py`
  - Current hot MOSS path is split across:
    - `before_decode()` / `_write_decode_input_embedding()`;
    - `_collect_frame()` publishing radix ids;
    - `_run_frame_decode()` doing local frame decode, graph/eager path
      selection, row build, feedback pool writes, repetition history, and
      journal creation;
    - `post_decode_launch()` / `post_decode_resolve()` for async one-step
      lookahead;
    - `post_process_outputs()` and `on_generation_steps_advanced()` for
      host-side output and pool step commits.

- `sglang_omni/model_runner/base.py`
  - The generic AR runner owns the sync and async lifecycle:
    - sync: build forward batch -> prepare/forward -> post decode -> finalize;
    - async: launch current decode -> record CUDA event -> resolve previous
      decode -> finalize previous step.
  - This is the right layer for cross-model async launch/resolve timing if we
    need whole-stack visibility.

- `sglang_omni/scheduling/omni_scheduler.py`
  - The async decode loop decides whether a batch uses lookahead, flushes
    pending steps, drops stale overrun rows, launches current decode, and
    resolves previous decode.
  - Previous P/P+D2 results improved local frame scopes but regressed
    end-to-end; this means scheduler-loop and launch/resolve boundaries must be
    included before judging the next optimization.

- `sglang_omni/models/moss_tts_local/stages.py`
  - MOSS Local preprocessing and vocoder each load the ~1B audio tokenizer.
  - `create_vocoder_executor()` batches `processor.decode_audio_codes(...)` and
    then moves decoded wav tensors to CPU with `detach().to("cpu")`. The AR
    frame-loop profile alone cannot explain full RTF without code2wav/vocoder
    timings.

## Boundary Map

```text
client/coordinator
  -> preprocessing stage
       CPU parsing + reference audio codec encode on codec GPU
       emits prepared MOSS request
  -> tts_engine stage / OmniScheduler
       scheduler loop:
         recv/process input
         get_next_batch_to_run
         run_batch
           ModelRunner._build_forward_batch
           MOSS before_decode: pool row -> decode embedding staging
           SGLang Qwen3 backbone forward under decode CUDA graph
           MOSS _run_frame_decode:
             local frame CUDA graph/eager fallback
             output clone / row build / radix hash
             feedback pool write / repetition history / journal
         process_batch_result / async resolve / output routing
  -> vocoder stage
       batch code tensors -> processor.decode_audio_codes
       wav detach + D2H
       payload assembly
  -> coordinator response / stream
```

Changed observability boundaries:

```text
torch trace labels:
  record_function ranges around scheduler/model-runner/MOSS/vocoder operations

request event JSONL:
  request-aligned milestones and optional fine per-frame intervals

profiler views:
  interval pairing for JSONL event names, aggregated by stage and interval
```

The profiling layer must observe these boundaries without changing their
contracts. It may add labels and scalar metadata only.

## Mechanical Contracts

### Torch Profiler Ranges

- Inputs: string range name; Python control scope.
- Outputs: Chrome/Perfetto trace ranges when torch profiler is active.
- Ownership: callsite owns range placement; `TorchProfiler` owns start/stop and
  export.
- Invariants:
  - no syncs;
  - no tensor reads for metadata;
  - no allocation-heavy formatting inside the per-frame hot path.
- Failure behavior: range should not affect serving if profiler is inactive.
- Cost model: per-batch or per-frame Python scope entry; acceptable only while
  profiling or if the callsite is already consistent with SGLang hot-path
  practice.

### Request-Event JSONL

- Inputs: `request_id`, `event_name`, optional scalar metadata.
- Outputs: JSONL events under `<event_dir>/events_<stage>_<pid>.jsonl`.
- Ownership: event recorder owns file lifecycle; callsites own event semantics.
- Invariants:
  - inactive recorder is no-op;
  - fine-frame events are opt-in via
    `SGLANG_MOSS_TTS_LOCAL_FINE_FRAME_EVENTS=1`;
  - MOSS frame events describe shared batch intervals, not unique per-request
    GPU time.
- Failure behavior: write errors are swallowed after one warning.
- Cost model: per-request event emission; broad events are cheap enough for
  n50/n1088 request-level profiling, fine-frame events are scoped-run only.

### MOSS Local Frame Runner

- Inputs:
  - hidden states from Qwen backbone, shape `[B, H]` or `[B, T, H]`;
  - request list and pool rows;
  - sampling params and generation/sampling steps from the state pool.
- Outputs:
  - `result.next_token_ids` for SGLang radix/scheduler;
  - `schedule_batch.output_ids`;
  - `MossTTSLocalDecodeJournal` for output collection;
  - updated feedback embeddings and repetition history in the state pool.
- Ownership:
  - state pool owns row-indexed GPU state;
  - model runner owns per-step row/journal publication;
  - scheduler owns batch lifecycle and final generation-step commits.
- Invariants:
  - no repetition-penalty row may use the graphed frame path;
  - chunked-prefill rows must not advance sampling/generation state;
  - graph outputs must be cloned before the next graph replay;
  - async launch/resolve must preserve the launched step's `next_token_ids`.
- Cost model:
  - per-frame, per-batch critical path;
  - mixes Python list work, GPU tensor gathers/copies, CUDA graph replay, kernel
    launches, and async event synchronization.

## Execution Plan

This should be a multi-phase debug-branch implementation. It crosses scheduler,
model runner, MOSS state, profiler views, and benchmark analysis; each phase
needs its own verification gate.

### Phase 0: Resync The Debug Branch

Reason: current branch diff against `upstream/main` includes unrelated removal
of recently merged RL admin-control files. Any profiler patch on top of this
state will be hard to review and risky to push.

Steps:

1. Merge current `upstream/main` into
   `perf/issue-752-moss-tts-compile-investigation`.
2. Resolve conflicts by preserving upstream non-MOSS changes and the debug
   branch's MOSS-specific harnesses/plans.
3. Verify the diff against upstream is MOSS/profiler/debug-only before adding
   new instrumentation.

Exit criteria:

- `git diff --name-only upstream/main..HEAD` has no unrelated admin/router/API
  file removals.
- Existing MOSS Local compile/sampler debug harness files still exist.

### Phase 1: Restore MOSS Frame Observability On The Current Runner

Reason: docs/views already describe MOSS fine-frame events, but the current
post-#759 `model_runner.py` does not emit them. We must adapt the older
`bb6d71d5` instrumentation to the current `_run_frame_decode()` split instead
of copying the old pre-pool `_collect_frame()` wholesale.

Files:

- `sglang_omni/models/moss_tts_local/model_runner.py`
- `sglang_omni/profiler/views.py`
- `docs/developer_reference/profiler.md`
- `tests/unit_test/profiler/test_views.py`

Implementation shape:

- Add `_profile_scope(name, recorder, requests, metadata)` using:
  - direct `with torch.profiler.record_function(name)`;
  - optional JSONL start/end events only when
    `SGLANG_MOSS_TTS_LOCAL_FINE_FRAME_EVENTS=1`.
- Restore broad request events:
  - `moss_tts_local_collect_frame_start/end`;
  - `moss_tts_local_frame_decode_start/end`.
- Place scopes on the current code, not the old code:
  - `moss_tts_local.before_decode.prepare_active_rows`
  - `moss_tts_local.before_decode.feedback_gather_copy`
  - `moss_tts_local.before_decode.input_ids_write`
  - `moss_tts_local.collect_frame.run_frame_decode`
  - `moss_tts_local.collect_frame.radix_hash_publish`
  - `moss_tts_local.run_frame_decode.setup`
  - `moss_tts_local.run_frame_decode.param_gather`
  - `moss_tts_local.run_frame_decode.sampling_state`
  - `moss_tts_local.run_frame_decode.path_select`
  - `moss_tts_local.frame_decode.cuda_graph`
  - `moss_tts_local.frame_decode.eager`
  - `moss_tts_local.run_frame_decode.graph_output_clone`
  - `moss_tts_local.run_frame_decode.row_build`
  - `moss_tts_local.run_frame_decode.eager_feedback_embed`
  - `moss_tts_local.run_frame_decode.emit_filter`
  - `moss_tts_local.run_frame_decode.feedback_write`
  - `moss_tts_local.run_frame_decode.audio_history_update`
  - `moss_tts_local.run_frame_decode.journal`
  - `moss_tts_local.post_decode_launch.radix_hash`
  - `moss_tts_local.post_decode_launch.snapshot`
  - `moss_tts_local.post_decode_resolve.restore_next_token_ids`
  - `moss_tts_local.post_process_outputs.output_rows_append`
  - `moss_tts_local.generation_step_commit.pool_write`
- Metadata must be scalar:
  - `batch_size`;
  - `frame_decode_path`;
  - `fallback_reason`;
  - `frame_graph_max_bs`;
  - `repetition_penalty_rows`;
  - `chunked_count`;
  - `emitted_count`;
  - `is_decode` / `is_extend`;
  - `async_enabled` / `is_lookahead` when available.

Exit criteria:

- request-event-only n8 run shows broad collect/frame intervals;
- fine-event n8 run shows all scoped MOSS intervals in
  `python -m sglang_omni.profiler <event_dir> --format table`;
- torch trace contains source-frame/function activity even if Chrome user labels
  are partially dropped.

### Phase 2: Add SGLang-Style Scheduler And Base ModelRunner Ranges

Reason: previous pool-state/native experiments reduced local `_collect_frame`
scopes but regressed end-to-end. We need to see whether the cost moved into
scheduler launch/resolve, finalize, queueing, or stage routing.

Files:

- `sglang_omni/scheduling/omni_scheduler.py`
- `sglang_omni/model_runner/base.py`
- docs/views/tests only if adding JSONL events; pure `record_function` ranges do
  not need profiler-view changes.

Implementation shape:

- Follow SGLang's `scheduler_pp_mixin.py` style: direct
  `torch.profiler.record_function(...)` around scheduler hot-path blocks.
- In normal/overlap/async loops add ranges:
  - `omni.recv_requests`
  - `omni.process_input_requests`
  - `omni.get_next_batch_to_run`
  - `omni.run_batch`
  - `omni.process_batch_result`
  - `omni.async.resolve_pending`
  - `omni.async.launch_batch`
  - `omni.async.resolve_previous`
  - `omni.async.drop_stale_overrun`
- In `ModelRunner` add ranges:
  - `omni_model_runner.build_forward_batch`
  - `omni_model_runner.before_decode`
  - `omni_model_runner.forward`
  - `omni_model_runner.post_decode`
  - `omni_model_runner.post_decode_launch`
  - `omni_model_runner.event_record`
  - `omni_model_runner.resolve_event_query`
  - `omni_model_runner.resolve_event_synchronize`
  - `omni_model_runner.post_decode_resolve`
  - `omni_model_runner.finalize.output_processor`
  - `omni_model_runner.finalize.post_process_outputs`
  - `omni_model_runner.finalize.generation_step_commit`

Do not add JSONL events for every scheduler loop by default. Use torch trace
ranges first; add request events only if we need per-request timeline alignment.

Exit criteria:

- n8 torch trace shows scheduler/model-runner ranges around the MOSS frame
  scopes.
- async query hit/miss counters can be correlated with resolve sync time.

### Phase 3: Add MOSS Codec/Vocoder Stage Observability

Reason: full RTF includes preprocessing/reference encode and vocoder/code2wav.
MOSS Local has two audio-tokenizer instances and the vocoder explicitly moves
decoded wavs to CPU. Without codec timing, AR improvements may not translate to
end-to-end RTF.

Files:

- `sglang_omni/models/moss_tts_local/stages.py`
- `sglang_omni/profiler/views.py`
- `docs/developer_reference/profiler.md`
- `tests/unit_test/profiler/test_views.py`

Implementation shape:

- Add request events and record_function ranges:
  - `moss_tts_local_ref_encode_start/end`
  - `moss_tts_local_ref_cache_lookup_start/end`
  - `moss_tts_local_vocoder_prepare_codes_start/end`
  - `moss_tts_local_vocoder_decode_batch_start/end`
  - `moss_tts_local_vocoder_d2h_start/end`
  - `moss_tts_local_vocoder_store_result_start/end`
- Metadata:
  - `batch_size`;
  - `codes_count`;
  - `codes_shape` as small list only if derived from tensor shape without
    materialization;
  - `sample_rate`;
  - `cache_hit` / `cache_miss` / `cache_merged` for reference encode.

Exit criteria:

- n50 request-event profile reports preprocessing/vocoder intervals alongside
  AR frame intervals.
- torch trace separates `processor.decode_audio_codes(...)` from D2H/store
  work.

### Phase 4: Profile Matrix And Decision Gate

Run the smallest matrix that answers where to optimize next:

1. **Request events only, c8 n50**
   - `enable_torch=false`;
   - no fine-frame events;
   - goal: end-to-end stage distribution.

2. **Torch trace, c8 n8**
   - `enable_torch=true`;
   - no expensive shape/stack/memory flags;
   - goal: scheduler/model-runner/frame/vocoder trace attribution.

3. **Fine-frame JSONL, c8 n8**
   - `SGLANG_MOSS_TTS_LOCAL_FINE_FRAME_EVENTS=1`;
   - `enable_torch=false` or `true` depending trace size;
   - goal: robust MOSS scope timings even if Chrome drops user labels.

4. **Memory/shape focused trace, c8 n4**
   - `SGLANG_TORCH_PROFILER_RECORD_SHAPES=1`;
   - `SGLANG_TORCH_PROFILER_PROFILE_MEMORY=1`;
   - goal: allocation/copy diagnosis only after the first three runs identify a
     suspect boundary.

Decision rules:

- If scheduler async resolve or event synchronize dominates, optimize
  launch/resolve overlap before touching model math.
- If `before_decode.feedback_gather_copy` or pool writes dominate, investigate
  pool-native graph input/output layouts.
- If `frame_decode.cuda_graph` is small and `row_build/radix/feedback_write`
  dominate, consider a fused row/radix/feedback kernel.
- If vocoder decode or D2H dominates full RTF, prioritize codec batching,
  stream chunk sizing, or code2wav placement before more AR work.
- If broad stage intervals disagree with torch trace windows, run ABAB repeats;
  do not optimize from one profiler-active run.

## Validation Plan

Local validation:

- `python -m py_compile` on edited files.
- `pytest tests/unit_test/profiler/test_views.py -q` if local dependencies are
  available.
- If unit dependencies are missing on Mac, record that and rely on py_compile
  plus remote profiler smoke.

Remote H100 validation:

- Start server with normal MOSS Local config first.
- Run c8 n8 torch trace and c8 n50 event-only profile.
- Confirm:
  - server does not crash on `/start_profile` / `/stop_profile`;
  - event files contain broad MOSS frame events;
  - fine-frame env emits scoped start/end pairs;
  - trace gzip is produced;
  - all generated requests complete;
  - no WER/speed claims are made from profiler-only small runs.

Correctness guard:

- Profiling code must not change generated code hashes under c1 direct code
  trace.
- Full SeedTTS WER is not required for observability-only commits, but any later
  optimization candidate must still run the sampler-compile-style gates:
  direct parity, c1 code trace, c8 speed/quality, then full SeedTTS.

## Risks And Mitigations

- **Risk: instrumentation changes runtime timing.**
  - Keep broad events low-volume.
  - Keep fine-frame events env-gated.
  - Use scalar metadata only.

- **Risk: Chrome trace drops user `record_function` labels again.**
  - Keep JSONL fine-frame fallback.
  - Parse function-level timings from trace as secondary evidence.

- **Risk: branch carries unrelated upstream reversions.**
  - Merge current `upstream/main` before implementation and verify diff scope.

- **Risk: profiles identify local wins that regress end-to-end.**
  - Always pair MOSS frame scopes with scheduler/model-runner/stage/vocoder
    scopes before choosing the next optimization.

## Next Implementation Order

1. Merge `upstream/main` into the debug branch.
2. Restore/adapt MOSS frame instrumentation onto current `model_runner.py`.
3. Add scheduler/base-runner `record_function` ranges.
4. Add MOSS vocoder/reference encode events.
5. Run c8 n8/c8 n50 profile matrix and write a triage report.
6. Pick the next optimization PR only after the triage identifies a dominant
   boundary.

## Implementation Pass 2026-06-13

Implemented on `perf/issue-752-moss-tts-compile-investigation` after merging
`upstream/main`.

What changed:

- MOSS Local AR runner:
  - broad request events:
    - `moss_tts_local_collect_frame_start/end`;
    - `moss_tts_local_frame_decode_start/end`;
  - fine `record_function` and optional JSONL scopes for:
    - `moss_tts_local.before_decode`;
    - `moss_tts_local.before_decode.prepare_active_rows`;
    - `moss_tts_local.before_decode.feedback_gather_copy`;
    - `moss_tts_local.before_decode.input_ids_write`;
    - `moss_tts_local.collect_frame.setup`;
    - `moss_tts_local.collect_frame.pool_rows`;
    - `moss_tts_local.collect_frame.param_gather`;
    - `moss_tts_local.collect_frame.sampling_state`;
    - `moss_tts_local.collect_frame.path_select`;
    - `moss_tts_local.frame_decode.cuda_graph`;
    - `moss_tts_local.frame_decode.eager`;
    - `moss_tts_local.collect_frame.graph_output_clone`;
    - `moss_tts_local.collect_frame.row_build`;
    - `moss_tts_local.collect_frame.radix_hash`;
    - `moss_tts_local.collect_frame.eager_feedback_embed`;
    - `moss_tts_local.collect_frame.emit_filter`;
    - `moss_tts_local.collect_frame.feedback_write`;
    - `moss_tts_local.collect_frame.audio_history_update`;
    - `moss_tts_local.collect_frame.journal`;
    - `moss_tts_local.async_launch.radix_hash_publish`;
    - `moss_tts_local.async_resolve.restore_next_token_ids`.
- Base model runner:
  - added `omni_model_runner.*` ranges around forward-batch build,
    before hooks, forward, post hooks, sampling, output processing, async
    launch, async event wait, async resolve, and generation-step advance.
- Omni scheduler:
  - added `omni_scheduler.*` ranges around request receive/process, batch
    selection, sync batch execution, async launch, previous-step resolve,
    pending drain, stream output, and batch-result processing.
- MOSS Local vocoder:
  - broad request events:
    - `moss_tts_local_vocoder_batch_start/end`;
    - `moss_tts_local_vocoder_decode_start/end`;
  - `record_function` ranges for code preparation, codec decode, sample-rate
    lookup, wav CPU materialization, and result store.
- Profiler views/docs/tests:
  - added the new MOSS AR/vocoder interval pairs to the stage breakdown view;
  - documented the new trace ladder in `docs/developer_reference/profiler.md`.

Hot-path safety notes:

- No tensor value materialization was added to the MOSS AR path.
- The new AR metadata is scalar/shape/path information only.
- Fine per-frame JSONL events remain opt-in through
  `SGLANG_MOSS_TTS_LOCAL_FINE_FRAME_EVENTS=1`.

Local verification:

- `python -m py_compile` on touched runtime/profiler files.
- `pytest -q tests/unit_test/profiler/test_views.py`.
