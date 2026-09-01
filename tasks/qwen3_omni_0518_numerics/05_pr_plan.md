# Per PR plan for the Qwen3-Omni stage path problems

Companion to 04_where_the_time_goes.md. That document holds the measurements
and the causes. This one holds the design and proof for the fixes, one
section per PR, in the order they should land. Items F (tuned triton MoE
config) and G (CI JIT caches) are out of scope by decision. G stays a
research item until its CI findings are read.

Status: Conditional. D, C and A are ours to implement. E is a measurement
(E0) before any code, because the overlap design already has a held PR
with a measured regression. B is a run before it is a PR.

Revision: worktree `.worktrees/qwen3-omni-0518-numerics` at 2f5d2ab52 on
`analysis/qwen3-omni-0518-numerics`, forked from `origin/main` at d5eac2627.
`origin/main` is at 216e946dd. Every `path:line` below was read at
2f5d2ab52. The five commits `origin/main` has on top (216e946dd, abc764044,
c08f96082, ef9479460, 302f79d09) touch Ming-Omni-TTS, MPS, the radix cache
and CI thresholds, none of the files cited here. Pinned upstream is
sglang v0.5.18 (pyproject.toml:32), read from a detached worktree of the
`v0.5.18` tag. torch 2.12.0 from the repo venv.

Requirements (from the user, this session):

- Make the omni TTS, ASR and speech paths fast at low and high concurrency
  without touching voice quality, reference audio caching or the answer
  tokens.
- Fix real, measured problems only. No sglang patches, close every gap at
  an omni owned seam.
- One mechanism per PR, each provable by one measurement and one numerics
  oracle, each revertible on its own.

Non goals: the FlashInfer untuned shape, the MoE config warnings, torchcodec,
the FP8 thinker kernels (all measured flat in 04), the CI cache mount (G),
the triton MoE tuning (F), the code2wav launch pattern (04 item 8, under
70 ms per request).

## 0. Prior art, read before every item

The Qwen3-Omni serving work has a tracker (issue #1022, PR map) and an RFC
with measured evidence and gates (issue #1018). Process replicas are tracked
in #1307. Checked on 2026-08-31 with `gh pr list --state all --search`.

| Item | PR | State | What it establishes | Consequence here |
|---|---|---|---|---|
| E (tracker 2.3) | #1204 | open, held | device resident feedback slots, prerequisite of #1320, c64 TTFA regression in all three boots | do not duplicate |
| E | #1320 | open draft, held, conflicts with main | talker launch/resolve split. #1018: overlap on is 6.2 percent slower on TTFA, p = 6.8e-12, the E2E and RTF wins were a launch order artifact | the overlap design as built is refuted, not the problem. E becomes E0 |
| E | #1409 | open | removes two host blocking syncs on the talker thread, output neutral, no promotable win | leave to the author |
| E | #1346 | open | coalesces codec frames into per chunk messages, 10x fewer talker to code2wav messages | the host work reduction the loop needs, review with our traces |
| E, item 5 | #1563 | open | opt in prefill/decode interleave on the talker | the admission wait, leave to the author |
| E | #1211 | open | lookahead decision, launch and resolve events | the observability E0 wants |
| D (tracker 1.4) | #1256 | closed | first attempt | gone |
| D | #1258 | open, stale, no reviews since 2026-08-04 | publishes capture layers through `LogitsProcessorOutput` and removes the hook based capture, which #1380 has since replaced with static buffers | not mergeable as is. #1018 item 1.4 says to re-derive double buffering on top of static capture, which is section 6. Say so on #1258 before opening |
| B | #1175 | merged | process replicas with entry stage resolution in the coordinator | mechanism exists |
| B | #1262 | closed | entry replicas and logical source names | its entry stage half is on main |
| B | #1564 | merged, half removed | in process concurrent preprocessing dispatch was tried and dropped: non reproducible benefit with negative expected value, and an unresolved AsyncClient event loop ownership issue | the `max_concurrency` alternative is dead, replicas of the preprocessing process are untried |
| B | #1307 | open tracker | fan in from replicated stages keeps logical names, validation across pipelines | reference in the run and the PR |
| C | none | | #1161 removed per request syncs in the prefill merge, the deepstack zeros and mask scatter and the thinker's two device reads on the synchronous path remain | ours |
| A | none | | #1591, #1628 and #1564 touched the encoder and left the store before send and the host tensor hit path as they are | ours |
| E3 | #1643 | closed | Qwen3-TTS async decode penalty handoff, results did not support default on | dropped |

What #1320 found that section 7 had missed: a retract while a step is
pending loses the frame that step emitted and desyncs the replay, so the
pending step must be drained before `retract_decode`, and
`mask[rows, toks] = True` materializes a Python bool and blocks the host once
the loop runs ahead. Both go into any future talker loop work.

What #1018 section 2.3 measured on the talker step at concurrency 8: 13.0 to
13.2 ms per step, GPU kernel floor 6.29 ms, pure host work 4.38 ms of which
3.45 ms is exposed. Our trace at concurrency 16 (04 section 2.1): 21 ms per
step for 5.5 ms of GPU. Both say the loop is host bound, so overlap can only
hide the wait, and the gate in #1018 is "stall time attributed and removed".

## 1. Evidence ledger

Facts are repository facts unless marked. Measurements come from 04 and
the compare report in `artifacts/2026-08-30-numerics3-slim`.

### 1.1 The synchronous AR step (talker, Qwen3-TTS, thinker at bs 1 and for audio output)

- `OmniScheduler.start` picks `_event_loop_async_decode` when
  `enable_async_decode`, else `_event_loop_overlap`, else `_event_loop_normal`
  (sglang_omni/scheduling/omni_scheduler.py:1701-1711). The overlap loop
  raises `NotImplementedError` (:2289-2305).
- `_event_loop_normal` (:2256-2287): `get_next_batch_to_run`, `run_batch`,
  `process_batch_result`, next iteration. `run_batch` calls
  `model_runner.execute` then `_emit_stream_output` then
  `_make_batch_result` (:1369-1385).
- `ModelRunner.execute` (sglang_omni/model_runner/base.py:250-302):
  `_build_forward_batch`, `_prepare_and_forward` (before hook, forward,
  optional sample), `post_decode` or `post_prefill`, `_ensure_next_token_ids`
  (samples when `next_token_ids` is still None, :575-596),
  `_publish_next_tokens` (FutureMap stash, :612-624), `_finalize` (:528-573).
- `_finalize` calls `_resolve_host_token_ids` (:203-208), which waits on the
  staging event when a runner staged ids with `_stage_token_ids` (:142-155,
  pinned ping pong buffers :168-201) and otherwise returns None. With None,
  `SGLangOutputProcessor.process` runs `.tolist()` on
  `model_output.next_token_ids`
  (sglang_omni/scheduling/sglang_backend/output_processor.py:35-38), and
  `_make_batch_result` hands the device tensor to upstream
  (omni_scheduler.py:1441-1456) whose `_normalize_decode_outputs` runs
  `.tolist()` again (v0.5.18 scheduler_components/batch_result_processor.py:934,
  reached from `process_batch_result_decode` :805 via scheduler.py:3922-3933).
- `_stage_token_ids` callers: Qwen3-TTS `_collect_codes`
  (sglang_omni/models/qwen3_tts/model_runner.py:238) and the Qwen talker
  `post_prefill` and `post_decode`
  (sglang_omni/models/qwen3_omni/talker_model_runner.py:99, :117). The thinker
  never calls it.
- The omni worker forward does not sample:
  `forward_batch_generation` returns a `GenerationBatchResult` with
  `logits_output` and `can_run_cuda_graph` only
  (sglang_omni/model_runner/model_worker.py:257-298). Sampling is either the
  runner's `_sample_next_token_ids` (base.py:787-823) or, for the talker, in
  the model forward itself (sglang_omni/models/qwen3_omni/components/talker.py:1284-1294
  writes `_sampled_token_ids` and runs `code_predictor_forward`).

### 1.2 The one step lookahead (shipped, used by Higgs and MOSS-TTS-Local)

- `_event_loop_async_decode` (omni_scheduler.py:2468-2555): a decode batch
  with `len(reqs) >= async_decode_min_batch_size` and
  `runner.lookahead_eligible(batch)` is launched first (`_run_batch_launch`
  :1458-1467, `execute_launch` base.py:304-365), then the previous pending
  step is resolved (`_resolve_and_process` :2324-2361, `execute_resolve`
  base.py:367-404). Prefill, empty and ineligible batches drain the pending
  step and run synchronously (:2526-2551).
- `execute_launch` runs `_prepare_and_forward(is_lookahead=True)`, then
  `post_decode_launch` (runner hook, returns the resolve payload), then
  `_ensure_next_token_ids`, `_publish_next_tokens`, records a device event
  (`SGLangExecutionBridge.record_completion`,
  sglang_omni/model_runner/sglang_execution.py:124-128) and snapshots the
  batch with `schedule_batch.copy()` (base.py:351-357).
- `execute_resolve` waits on the event (`query` then `synchronize`),
  computes `skip_rids` from `req.finished()` or retracted, calls
  `post_decode_resolve`, then `_finalize(skip_rids=...)` (base.py:377-404).
  `_finalize` skips those rows in the generation step loop and calls
  `on_generation_steps_advanced(advanced_steps, forward_batch)` for the rest
  (base.py:549-562, hook :520-526).
- The default `post_decode_launch` samples if needed and copies ids into a
  pinned staging buffer with `non_blocking=True` (base.py:721-743). The
  default `post_decode_resolve` points `result.next_token_ids` at the host
  snapshot (:745-761).
- The base `lookahead_eligible` refuses batches with repetition, frequency
  or presence penalty, `min_new_tokens` or a custom logit processor, because
  the sglang penalizer state lags one token under lookahead (base.py:681-703).
- The thinker overrides it and additionally refuses audio output requests
  and logprob requests (sglang_omni/model_runner/thinker_model_runner.py:411-447,
  audio at :428, test tests/unit_test/qwen3_omni/test_thinker_lookahead_eligible.py:65-67).
  The docstring gives the reason: "Audio can overwrite hidden-state capture
  before resolve" (:412-415).
- The thinker's `post_decode_launch` copies ids into a pinned ping pong
  buffer, `post_decode_resolve` turns the pinned slice into a CPU long tensor
  (:449-490). On the lookahead path the thinker's token readback is therefore
  already host only.
- The scheduler comment on the bs 1 fast path cites "benchmark_results.md /
  stall_analysis.md" (omni_scheduler.py:275-279). Neither file exists in the
  repository.
- Test harness: tests/unit_test/pipeline/test_async_decode.py (`_StubRunner`
  over `ModelRunner`, `FakeExecutionBridge` from tests/unit_test/fakes.py,
  patched device Event, :1-113) with 35 tests over launch, resolve, overrun,
  retract, drain and failure paths.

### 1.3 Hidden state capture for the talker

- `install_hidden_capture_hooks` registers one buffer per capture layer on
  the text model and a forward pre hook that copies the layer input into
  `buffer[:num_rows]` (sglang_omni/model_runner/_hidden_capture.py:77-161,
  copy at :72). `StaticAuxHiddenCapture.views(n)` returns views of those
  buffers (:35-39). Graph replay refreshes the same addresses without Python
  (tests/unit_test/model_runner/test_hidden_capture.py:114-129).
- The thinker's output processor reads `static_capture.views(rows)` at
  process time and clones each request's slice
  (output_processor.py:84-93, :145-172). Process time is `_finalize`, which
  under lookahead runs after the next step's launch. Capture layers for the
  speech thinker are `[0, 24]` (sglang_omni/models/qwen3_omni/bootstrap.py:47).
- The per token stream to the talker carries `extra["hidden_states"]`
  through `_build_stream_output`, which sends one message per token with the
  embed row as data and `{"token_id": int}` as metadata
  (sglang_omni/models/qwen3_omni/request_builders.py:938-993, message at
  :977-990). The talker appends it to `pending_text_queue`
  (sglang_omni/models/qwen3_omni/talker_scheduler.py:146-152).
- The speech thinker's default request has `repetition_penalty` 1.0
  (request_builders.py:635), so the base sampling gate does not exclude it.

### 1.4 The talker step

- `QwenTalkerModelRunner.before_decode` calls `model.prepare_decode_buffers`
  and `_write_feedback_buffers` (talker_model_runner.py:56-76).
  `prepare_decode_buffers` (talker.py:1056-1184) builds six host lists per
  request, waits on the previous staging event, fills a pinned staging
  tensor, one H2D copy, six device to device copies, and up to two pageable
  H2D index tensors. `_reuse_decode_buffers` (:1029-1049) skips all of that
  when the batch holds the same rids in the same order and every
  `len(req.output_ids)` grew by exactly one (:1040-1041), updating only the
  repetition mask from the device `_sampled_token_ids` (:1044-1046).
  `invalidate_decode_buffers` runs on every extend forward (:1258-1259).
- `post_decode` (talker_model_runner.py:105-121) clones `_sampled_token_ids`,
  stages the pinned copy, clones `_output_codes` and `_output_embeds`
  (:133-134), puts one outbox message per request for code2wav (:145-153)
  and appends the feedback row to `pending_feedback_queue` (:154). This runs
  before `_finalize`'s event wait, so the code chunk leaves the runner before
  the GPU has finished the step.
- `is_decode_batch_ready` (:168-174) requires a feedback row and either a
  pending text row or `thinker_chunks_done` with a pad embed
  (:385-398). `QwenTalkerScheduler.get_next_batch_to_run` returns None and
  rolls the prepared batch back when a decode batch is not ready
  (talker_scheduler.py:96-137).
- The talker request carries `talker_repetition_penalty` 1.05 by default
  (request_builders.py:1095), applied in the model forward from
  `_repetition_mask` and `_repetition_penalties`
  (talker.py:1327-1345). The base `lookahead_eligible` would therefore refuse
  every talker batch even though the talker owns that state itself.
- The talker scheduler sets `disable_overlap_schedule` (talker_scheduler.py:33-34),
  `create_talker_scheduler` passes no `enable_async_decode`
  (bootstrap.py:244-258), so the talker runs `_event_loop_normal`.
- The talker uses no `stream_output_builder` (bootstrap.py:244-258), so
  `_emit_stream_output` is a no op for it (omni_scheduler.py:1405-1406).
- code2wav ingests one frame per message in arrival order and drops frames
  whose leading code is the codec EOS
  (sglang_omni/models/qwen3_omni/components/code2wav_scheduler.py:250-303).
- Qwen3-TTS: `sample_before_post_decode` True (model_runner.py:98-102),
  `_collect_codes` runs the predictor and stages ids (:212-239),
  `post_process_outputs` at `_finalize` clones codes and embeds and appends
  the feedback row (:241-264). Its repetition penalty defaults to 1.05
  (sglang_omni/models/qwen3_tts/request_builders.py:1189) and is applied by
  the sglang penalizer, not by the model (model_runner.py:123-130). Its
  engine builder sets `enable_torch_compile` True and compiles the backbone
  once (sglang_omni/models/qwen3_tts/engine_builder.py:52, :111-117).

### 1.5 Transport ordering (why device tensors may leave before the GPU is done)

- Same GPU, different process: the stage runtime serializes a device
  tensor with `ForkingPickler` (sglang_omni/comm/stage_io.py:776-781,
  payload :139-165, stream chunk :235-249, runtime send
  sglang_omni/pipeline/stage/runtime.py:1298-1325 and :1507-1543).
  torch's `reduce_tensor` calls `storage._share_cuda_()` and ships
  `event_handle` and `event_sync_required` (torch/multiprocessing/reductions.py:343-353),
  and `rebuild_cuda_tensor` passes them to `_new_shared_cuda`
  (:154-200). External fact to verify at implementation: the C++ side
  (torch/csrc/StorageSharing.cpp) records an interprocess event on the
  producer's current stream at share time and makes the consumer's current
  stream wait on it when `event_sync_required` is set.
- Cross GPU: the cuda_ipc relay copies on `torch.cuda.current_stream(device)`
  of the sending thread and records `ready_event` after the copy
  (sglang_omni/relay/cuda_ipc.py:765-780). The receiver waits on that event
  (:908-935).
- Both orderings hold only if the sending thread's current stream is the
  stream the forward used. Omni's AR forward runs on the default stream:
  `model_worker.py:290` calls `model_runner.forward` directly, no
  `torch.cuda.Stream` or `torch.cuda.stream` appears under
  sglang_omni/model_runner, sglang_omni/scheduling or
  sglang_omni/models/qwen3_omni outside code2wav, and
  sglang_execution.py:16-20 documents single stream execution. The stage
  runtime's send loop is an asyncio thread on the same legacy default
  stream. Today's talker already relies on this: `post_decode` enqueues the
  chunk (:118-121) before `_finalize` waits (base.py:297, :541).
- Encoder to thinker is direct CUDA IPC when both sit on one GPU in
  different processes (sglang_omni/comm/router.py:52-59). Every
  `Qwen3OmniSpeech*` config puts the encoders and the thinker on the
  thinker GPU (sglang_omni/models/qwen3_omni/config.py:255-268).

### 1.6 The encoder output cache

- `StageOutputCache.put` runs `_detach_value` inside the cache lock
  (sglang_omni/scheduling/stage_cache.py:113-132). With `pin_memory` False,
  `_detach_value` does `value.to(device="cpu")`, a blocking pageable copy
  (:45). With `pin_memory` True it allocates pinned memory and copies with
  `non_blocking=False` (:22-27), still blocking, and falls back to pageable
  on allocation failure (:36-44). `pin_memory` is forced False without CUDA
  (:96).
- Both Qwen3-Omni encoder executors construct the cache with
  `max_size=QWEN3_ENCODER_CACHE_MAX_ENTRIES` (64),
  `max_bytes=QWEN3_ENCODER_CACHE_MAX_BYTES` (4 GiB), `cache_device="cpu"`
  and no `pin_memory` (sglang_omni/models/qwen3_omni/stages.py:849-853,
  :929-933, constants :56-57).
- The store happens inside the batch function before results are returned
  (`_batch_image_encoder_payloads` :577-583 then :585 and :597, audio :771,
  single path :186), and `SimpleScheduler._run_batch` emits results only
  after the batch function returns
  (sglang_omni/scheduling/simple_scheduler.py:186-207).
- The pinned option was added by PR #1591 (10733c987, 2026-08-19) for the
  Whisper pre LM encoder, with a `stage_host_copy` hook that issues the
  copy on the encoder stream before its batch barrier. No Qwen3-Omni caller
  passes `pin_memory=True` (grep over sglang_omni). Tests exist for the pinned
  path (tests/unit_test/scheduling/test_stage_cache.py:155-217).
- torch's caching host allocator records the stream on a pinned block and
  does not reuse a freed block until that stream's work has passed
  (torch/include/ATen/cuda/CachingHostAllocator.h:14-30, external fact from
  the header in the venv).
- Measured: 612 `cudaMemcpyAsync` calls at 3.2 ms host time in the image
  encoder trace, about 80 ms per request, and a 100 to 130 ms hop to the
  thinker (04 sections 2.2 and 3, item 4).

### 1.7 Preprocessing replicas

- The preprocessing stage is `SimpleScheduler(_preprocess)` with default
  `max_concurrency=1` (stages.py:801-826, simple_scheduler.py:43, :64),
  which runs `_start_serial`: one request to completion, then the next
  (simple_scheduler.py:227-255). `max_concurrency > 1` exists and spawns N
  coroutine workers over `asyncio.to_thread` (:257-310), tested in
  tests/unit_test/pipeline/test_simple_scheduler_concurrent.py.
- `ProcessConfig.num_replicas` and `replica_devices`
  (sglang_omni/config/schema.py:179-180). A process with no GPU stage may set
  `num_replicas > 1` and must not set `replica_devices`
  (sglang_omni/config/topology.py:167-189). `expand_replica_stages` copies
  every member stage per replica with instance names `name@rN` and process
  names `process@rN` (sglang_omni/pipeline/replicas.py:109-186). Round robin
  binding at admission (:238-250, :253-278). Logical names in `next`,
  `stream_to`, `wait_for` are resolved at send time (:9-13,
  runtime.py:188, :190, tests/unit_test/pipeline/test_replicas.py:645).
- A CPU only process replicated twice runs end to end in
  tests/unit_test/pipeline/test_process_replicas_runtime.py:21-75 (two pids,
  process names `process-pair@r0` and `process-pair@r1`).
- In both speech pipelines preprocessing is its own process named
  `preprocessing` (config.py:280-288, used at :352-358 and :389-395). In the
  text pipeline every stage shares the `pipeline` process (:232-240).
- The coordinator in flight cap counts engine stage replicas only
  (sglang_omni/pipeline/mp_runner.py:47-76).
- The example speech launcher builds the config in code from `--gpu-*`
  flags and exposes no `processes` setting
  (examples/launchers/qwen3_omni.py:239-318, :387-467). Yaml deployments
  carry `processes:` (examples/configs/qwen3_omni_speech_code2wav_replica2_ci.yaml:11-14).
- Per request video preprocessing is 0.75 to 1.0 s of CPU on both images
  (04 section 1, preprocess A/B). Each replica loads its own
  `Qwen3OmniPreprocessor` (HF processor and tokenizer) and video decode runs
  in an 8 thread pool per process
  (sglang_omni/preprocessing/resource_connector.py:27,
  sglang_omni/preprocessing/video.py:196-231).

### 1.8 Deepstack prefill buffer

- `_forward_with_omni_embeds` (thinker_model_runner.py:361-409) builds
  `ds_input = torch.cat(layer_tensors, dim=-1)`, allocates
  `full_ds = torch.zeros(num_tokens, layers * hidden)` and scatters with a
  boolean mask `full_ds[visual_pos_masks] = ds_input` (:377-389). The mask
  path calls `nonzero` on the device, which synchronizes the host. The row
  index tensor already exists in `_inject_multimodal_embeds` as
  `torch.cat(visual_rows)` (:340).
- Upstream consumes the buffer per layer as
  `input_deepstack_embeds[:, sep : sep + hidden]` and adds it through
  `post_residual_addition` (v0.5.18 python/sglang/srt/models/qwen3_vl_moe.py:66-75,
  :113-119). The thinker is `Qwen3MoeLLMModel` (qwen3_omni_moe.py:485-495).
- This forward returns `can_run_cuda_graph=False` (:407-409), so it never
  runs under the prefill graph.
- Measured: 42 `aten::zeros` calls at 34 ms each in the video speech thinker
  window (04 section 3, item 3).

### 1.9 Corrections to 04

- 04 section 5.4 says the thinker sends two relay objects per token (a
  data tensor and a metadata tensor). The code sends one message per token
  with an integer metadata (request_builders.py:977-990). The 11 to 13 ms
  per chunk measurement stands, the two object explanation does not. To be
  corrected in 04 when it is next edited.

### 1.10 Inferences and assumptions

- Inference: the talker's 21 ms step is the sum of launch host time, the
  wait for the GPU, and the post wait host work (finalize, upstream result
  processing, stream output, next batch build, `prepare_decode_buffers`,
  feedback write). 04 section 2.1 measured the wait (4.7 ms) and the launch
  (1.9 ms), the remainder is not attributed to functions yet. The design of
  E does not depend on that attribution, only the size of its win does.
- Assumption A1: pinned allocation for entries up to 4 GiB per encoder
  process is acceptable on the serving hosts. Verified
  by the A run (resident and locked memory per encoder process).
- Assumption E1: at bs 1 the lookahead's fixed cost (batch copy, event,
  pending bookkeeping) is smaller than the 9 ms of host work it overlaps for
  the talker. The scheduler's own comment records a bs 1 regression for an
  earlier adopter (omni_scheduler.py:275-279) without a surviving document.
  Verified by the E run at c1 with `async_decode_min_batch_size` 1 and 2.

## 2. Shared qualification recipe

Every PR is qualified on the H100 container with the numerics3 scripts, the
candidate against the same image with the PR reverted, same caches, warm:

- `tasks/qwen3_omni_0518_numerics/scripts/h100_runs.sh`: `serve_bf16_colocated`
  (examples/configs/qwen3_omni_colocated_h100_bf16.yaml), the video speech
  and video text benches (Video-MME with and without audio output) at c16,
  `bench_seedtts_vc` at c1 and c16,
  `run_tts_events` for per request intervals, `run_tts_traces` for kineto
  traces, `run_compare OLD NEW` for the report.
- Oracle for tokens: the compare report's per sample answer flips (thinker
  `output_ids` per sample) must be zero between arms. Oracle for codes: the
  talker `output_ids` per request, dumped by the events or the result JSON,
  bit identical between arms at the same seed. The talker's per row seed is
  derived from the request id when the request carries none
  (talker.py:1093-1101), so identical ids give identical seeds.
- Oracle for speed: the metric named in each PR section, from the report's
  events and trace sections, three repeats per arm, worst of three.
- Cold versus warm: the first run of each server is discarded (04 section
  3, item 6).

## 3. PR A, encoder output cache off the request path

Title: `[Perf] Store Qwen3-Omni encoder outputs after they are sent`

Not started. A first implementation on `perf/encoder-cache-async` made the
cache write asynchronous with per entry CUDA events and an event gated
`get`. That was the wrong design: it kept the write on the request path and
hid it, and it is not how pinned memory is used anywhere in SGLang or in
this repository. The branch is parked and will be reset to this design.

### 3.1 What the cache is for and what it costs

The key is a content hash of the media, computed in preprocessing from the
path before decoding (sglang_omni/models/qwen3_omni/components/preprocessor.py:496-499),
carried to the encoder as `cache_key` and to the thinker, where it becomes a
hashed placeholder id so the radix cache can prefix match identical media
(request_builders.py:604-624). A repeated media input can hit twice: the
encoder cache skips the encoder forward, the radix cache skips the prefill
of the shared prefix.

Miss path today: encoder forward, then `cache.put` copies the whole output
to pageable host memory under the cache lock (stage_cache.py:113-132, :45),
then the batch function returns, then `SimpleScheduler` emits the results
(simple_scheduler.py:186-207), then the stage sends the device tensors
through direct CUDA IPC with no copy (runtime.py:1298-1325). A video request
at 128 frames and 401408 px is 25088 visual tokens (patch 16, spatial merge
2, temporal patch 2, out hidden 2048, three deepstack layers, from the HF
vision config), so the entry is four `[25088, 2048]` bf16 tensors, 411 MB,
and the copy is the measured 80 ms. The request paying it gains nothing.

Hit path today: `_lookup_cached_encoder_output` returns the host tensors and
they go into the payload as they are (stages.py:176-193 single path,
:418-425 image batch, :681-690 audio batch). The router sees CPU tensors and
picks shared memory (sglang_omni/comm/router.py:262-282), three host copies.
The thinker then moves each embedding with `non_blocking=True` from pageable
memory, a synchronous copy (thinker_model_runner.py:326-332). A video hit
costs roughly 50 ms of shm plus about 70 ms of pageable H2D, close to the
forward it replaces.

The defect is ordering: a side effect of the batch sits between the forward
and the send.

### 3.2 Design

1. Emit first, store after. The encoder batch function returns its results
   together with the store it has deferred, `SimpleScheduler._run_batch`
   emits the results and then runs the deferred store. The store is the
   existing synchronous pinned copy (`pin_memory=True`, stage_cache.py:22-27),
   no events, no change to `StageOutputCache`. It costs the encoder thread
   about 20 ms per video after the thinker already holds the device tensors,
   and nothing on the request path. First touch page locking per size class
   lands there too.
2. A hit returns device tensors. On a hit the encoder restores the entry to
   its device with a non-blocking H2D from pinned memory before
   `apply_encoder_result`, so hits and misses leave through the same direct
   CUDA IPC path. The restore is stream ordered ahead of the IPC event the
   send records. This is the same pattern SGLang's processor applies to
   decoded video before its H2D (v0.5.18 multimodal/processors/qwen_vl.py:270).
3. Locked memory cannot swap, so a pinned budget is checked once at
   startup against `psutil.virtual_memory().available` minus a reserve: an
   explicit `cache_max_bytes` that does not fit is an error, the default
   (the existing 4 GiB per encoder process) shrinks with a warning, a
   pageable cache keeps whatever was asked for.

Tradeoffs: page locked RAM equal to the budget plus the host allocator's
power of two rounding. About 20 ms of encoder thread time per video miss,
which only matters once the encoder is saturated (it is 24 percent busy at
c16). One small `SimpleScheduler` contract (a batch result may carry work to
run after emission).

### 3.3 The larger waste on the same path

Preprocessing ships `pixel_values_videos` as float32 through pageable shared
memory (617 MB and three copies per video, the measured 65 to 78 ms hop),
and the encoder then moves it to the device with a synchronous H2D from
pageable memory (image_encoder.py:168, :186-188) before casting to bf16 on
the device. Casting at the source halves the bytes with identical rounding,
and a pinned staging buffer on the receiving side makes the H2D
asynchronous. Both costs are of the cache write's size. Measurements to
take before designing it: the `Memcpy HtoD` time per video in the image
encoder trace and the shm relay time per hop from the comm trace.

### 3.4 Proof

| Invariant | Violation | Measurement | Accept |
|---|---|---|---|
| Tokens unchanged | wrong or stale entry | answer flips on video, image and audio inputs at c16, plus a repeat run where every request is sent twice | zero flips, repeat run identical to the first |
| Write off the request path | store still precedes the send | events: encoder end to `stage_hop_sent` gap, image encoder trace `cudaMemcpyAsync` host time inside the batch interval | store after the hop event, no D2H inside the batch |
| Hit stays on device | shm transport chosen | events `stage_hop_sent` transport on the repeat run | `torch_cuda_ipc` for every encoder to thinker hop |
| Hit saves time | restore slower than the forward | repeat run: encoder interval on a hit against a miss | hit under 25 ms |
| Memory bounded | budget exceeded | locked memory of both encoder processes after the c16 video run | at most the budget plus rounding |

Rollback: `pin_memory=False` at the executors restores the pageable cache,
the deferred store stays.

Status: Ready to implement, after E.

## 4. PR B, preprocessing replicas

Title after the run: `[Config] Replicate the Qwen3-Omni preprocessing process`

Class: cross boundary (config, pipeline runtime, launcher). Owner: the
pipeline config (`processes`) and the launcher that builds it.

### 4.1 Current mechanics

Section 1.7. One preprocessing process, one request at a time, 0.75 to
1.0 s per video, so video throughput is one over that number and at c16 a
request spends 6 to 13 s in the preprocessing queue while the GPU stages
sit at 24 percent busy (04 section 3, item 1).

### 4.2 Design

Credible designs:

1. `processes.preprocessing.num_replicas: N`. N processes, round robin at
   admission, each with its own preprocessor and thread pool. Shipped and
   runtime tested for a CPU process. Cost: N times the preprocessor's RAM and
   N x 8 decode threads.
2. `max_concurrency=N` on the preprocessing `SimpleScheduler`. One process,
   N requests in flight through `asyncio.to_thread`, sharing the 8 thread
   pool. Shipped and unit tested at the scheduler layer, but the
   preprocessor's re entrancy and the GIL share of the HF processor are
   unverified.

Decision: 1 for the run. 2 was tried in #1564 and removed before merge:
"non reproducible benefit with negative expected value" and an unresolved
AsyncClient event loop ownership issue, so it is not the fallback. The run
decides N and whether a PR follows.

### 4.3 What the run showed and what it needs

The run on 2026-09-01 (06 section 6) did not start: the colocated speech
config class rejects any replicated process
(`_validate_colocated_qwen_replicas`, placement.py:70-82, added by #1175)
whether or not the process has a GPU stage. The disaggregated speech
config has no such rule. So the run needs one of two things first, and
the choice is part of the B PR: the placement rule admits replicas of
processes with no GPU stage (preprocessing is `gpu: null`), or the run
moves to the disaggregated two GPU server (`serve_bf16_disagg`). The
first is the serving default this plan wants, so the rule change comes
first and the run follows on the colocated config:

```yaml
processes:
  preprocessing:
    num_replicas: 4
```

Run video speech and video text at c16 and c1, with events. Read: qps, the
`wait.preprocess` interval, GPU busy fraction of the thinker and encoders,
RSS per replica, and the next bound (expected the thinker prefill). Repeat
with 2 and 8 to find the knee against the host's core count (`nproc` in the
container).

Validation items the run answers:

- Replica processes start and take requests in turn (log lines from
  `expand_replica_stages`, replicas.py:169-178, and admission bindings).
- The aggregate stage's `wait_for=["preprocessing", ...]` resolves the
  instance names (runtime.py:188, covered by test_replicas.py:645).
- `stage_hop_sent` events carry `preprocessing@rN` as the stage, so
  compare_runs.py's hop pairing needs the `@rN` suffix stripped
  (compare_runs.py `hops_for`, tooling change in this branch).
- c1 latency unchanged.
- The fp8 boots on that host died with an external SIGKILL of
  image_encoder before readiness (no cgroup OOM), which needs a host side
  look before any fp8 run.

### 4.4 The PR, if the run is positive

- sglang_omni/models/qwen3_omni/config.py: the speech pipeline configs
  (`Qwen3OmniSpeechPipelineConfig` :331-371 and the colocated subclass
  :373-395) declare `processes={"preprocessing": ProcessConfig(num_replicas=N)}`
  by default (`PipelineConfig.processes` is a dict field with an empty
  default, sglang_omni/config/schema.py:460), so every launcher, including the example speech launcher that
  builds the config in code (examples/launchers/qwen3_omni.py:387-467), and
  every yaml deployment gets the serving default. Yaml can still override it.
- examples/configs/qwen3_omni_colocated_h100_bf16.yaml and the fp8 twin: no
  change unless the host needs a different N.
- Unit test (new, tests/unit_test/qwen3_omni/test_preprocessing_replicas.py):
  the default speech config compiles through `prepare_pipeline_runtime` and
  `_build_stage_groups` into N CPU processes named `stage-preprocessing@r0`
  onward with `gpu=None` (pattern tests/unit_test/higgs_tts/test_process_replicas.py:22-60,
  name rule sglang_omni/pipeline/stage_workers.py:864-872). A text pipeline
  config with the same block must fail validation (the `pipeline` process
  has GPU stages, topology.py:178-183).

Non goals: the text pipeline (one process for every stage), reordering
requests across replicas (round robin changes admission order at the
thinker relative to arrival, correctness is unaffected).

Rollback: remove the flag or the yaml block.

Status: Blocked on the run (RAM per replica, core count, the next bound).

## 5. PR C, thinker synchronous step host work

Title: `[Perf] Stage thinker token ids and reuse the deepstack buffer`

Class: localized in two owners. C1 owner: `ModelRunner.execute`, which
materializes the reporting tokens (base.py:575-596) and hands them to
`_finalize`. C2 owner: `ThinkerModelRunner._forward_with_omni_embeds`.

### 5.1 Current mechanics

Section 1.1 and 1.8. On the synchronous path the thinker's ids are read
from the device twice per step by two blocking pageable copies. Every
visual prefill allocates a zero buffer for all tokens and scatters with a
boolean mask, which synchronizes on `nonzero`.

### 5.2 Design

C1: after `_ensure_next_token_ids` in `execute()`, if the result has no
staged host copy and `next_token_ids` is a tensor, call
`_stage_token_ids(batch_result, batch_result.next_token_ids)`. Then
`_finalize` resolves the pinned copy after an event wait, the output
processor reads a CPU tensor, and `_make_batch_result` passes the host copy
to upstream (omni_scheduler.py:1450-1451), so both `.tolist()` calls are
host only. Runners that stage in their hooks are untouched (the guard sees
`_host_token_ids`). Runners that return CPU ids get the alias path
(base.py:145-148). Prefill only batches stage the zero vector, harmless.

Placed in the base because the base owns the reporting token contract, and
the thinker cannot stage in `post_decode` (its sampling happens later in
`_ensure_next_token_ids`, `sample_before_post_decode` is False by default).
Behavioural change for every synchronous runner: identical values, one
event wait instead of one or two memcpy waits per step.

C2: keep a persistent buffer on the runner, `self._deepstack_buffer`, of
shape `[capacity, layers * hidden]` grown on demand to
`len(forward_batch.input_ids)`, and per prefill: `buf = buffer[:num_tokens]`,
`buf.zero_()`, `buf.index_copy_(0, visual_rows, ds_input)`.
`_inject_multimodal_embeds` returns the row index tensor it already builds
(:340) in place of the boolean mask, and `_forward_with_omni_embeds` takes
rows instead of a mask. Same values at the same rows, the model's
`post_residual_addition` is unchanged, so numerics are identical. Removes
the allocation and the `nonzero` synchronization from every visual prefill.
The memset remains and is one kernel.

### 5.3 Changed contracts

- `_inject_multimodal_embeds` returns `(input_embeds, ds_embeds, visual_rows)`
  with `visual_rows` a device long tensor, previously a bool mask. Callers:
  `custom_prefill_forward` (thinker_model_runner.py:39-55) and the Qwen3-Omni
  adopter's `before_prefill` (models/qwen3_omni/thinker_model_runner.py:280-295,
  which only checks the third element is None). Tests to adapt:
  tests/unit_test/qwen3_omni/test_thinker_mm_embed_merge.py and
  test_thinker_model_runner.py:60.
- `execute()` stages ids for every runner. No public surface.

### 5.4 Slices

One PR, two commits (C1 then C2), each with its unit test:

- C1 test (new, tests/unit_test/model_runner/test_execute_stages_token_ids.py):
  a `_StubRunner` in the style of test_async_decode.py whose output processor
  records the `host_token_ids` argument, run `execute()`, assert the
  processor received a host copy and that a runner which staged in
  `post_decode` is not staged twice (the ping pong slot advances once).
- C2 test: extend test_thinker_mm_embed_merge.py for the row tensor, and a
  new test that two consecutive prefills of different lengths reuse one
  buffer and leave non visual rows zero.

### 5.5 Proof

| Invariant | Violation | Measurement | Accept |
|---|---|---|---|
| Tokens unchanged | staging reads a stale slot | answer flips at c1 and c16 on video speech, video text and voice clone | zero |
| No pageable readback | a path still hits the device tensor | thinker trace: `cudaMemcpyAsync` count per decode step (two today) | zero pageable copies per step |
| Step shrinks | the copies were not the cost | thinker step period at c1 on voice clone (15.8 ms for 1.7 ms GPU) and video speech c1 | measured, reported |
| Deepstack values unchanged | wrong rows | compare answer flips on video stages, plus the unit test | zero |
| Zeros gone | allocation moved | thinker trace: `aten::zeros` count and host time in the video speech window (42 x 34 ms) | zero calls on the prefill path |

Validation task carried from 04: whether the 34 ms is `cudaMalloc` growth
or the memset. C2 removes the allocation either way, the trace after C2
answers it.

Rollback: revert, no persisted state.

Status: Ready.

## 6. PR D, thinker lookahead for audio output requests

Title: `[Scheduler] Snapshot hidden capture at launch and allow audio output on the thinker lookahead`

Class: cross boundary (runner hook, output processor). Owner: the thinker
runner's launch hook publishes the step's device state, the output
processor consumes it.

### 6.1 Current mechanics

Section 1.2 and 1.3. The lookahead is on for the thinker by default
(config.py:136-139, stages.py:1010) and refused for every audio output
batch, because the hidden capture buffers are static and the next launch
overwrites them before the previous step's output processor clones its
slices. So the speech thinker runs the synchronous loop at every batch
size, 42 ms per step for 10 ms of GPU at c16 on video speech (04 section 3,
item 3).

### 6.2 Design

This is the re-derivation on static capture that #1018 item 1.4 asks for.
#1258 chose a different mechanism (packing the capture layers into the
graph output) against the capture code that predates #1380, and would have
to be rewritten to land. Before opening D, state on #1258 and the tracker
that this design supersedes it, so the author can object or rebase.

Snapshot at launch. In `ThinkerModelRunner.post_decode_launch`
(thinker_model_runner.py:468-481), after the pinned id copy, when the
output processor captures hidden states, clone `capture.views(n)` for each
layer (two `[n, 2048]` bf16 device clones per step, stream ordered after
the forward) and return `(host_buf, hidden_snapshot)`. In
`post_decode_resolve` (:483-490) unpack, set `result.next_token_ids` as
today and attach `result.omni_hidden_snapshot`. In
`SGLangOutputProcessor._build_hidden_extras_by_request`
(output_processor.py:84-93) prefer `model_output.omni_hidden_snapshot`
over `static_capture.views(...)`, with the same per request slicing.
Remove the audio clause from `lookahead_eligible` (:428), keep the logprob
and sampling clauses and the fail closed branch for missing data.

Why not double buffer the capture: the hook writes a fixed address inside
the decode CUDA graph (section 1.3), so alternating buffers would need two
graph sets. The clone is two small kernels per step.

Ordering consequences: the per token message to the talker is emitted at
resolve, one loop iteration later than in sync (omni_scheduler.py:1469-1488,
:1398-1415). The talker consumes the text rows in arrival order per request
(section 1.3), the stream done signal follows the last chunk (terminal
`stream_output` runs after `process_batch_result` at resolve), and the
overrun row of a finished request is dropped by `skip_rids`
(:2341-2343, :1407-1410). The first talker input arrives one thinker step
later, the thinker step itself gets shorter.

### 6.3 Changed contracts

- `post_decode_launch` return value of the thinker runner becomes a tuple.
  Only its own `post_decode_resolve` reads it (base.py:391-397 passes it
  through).
- `GenerationBatchResult` gains an omni attribute `omni_hidden_snapshot`
  in process, never serialized.
- `lookahead_eligible` admits audio output. Test
  test_thinker_lookahead_eligible.py:65-67 inverts, a new test asserts the
  snapshot is used: a fake capture whose buffers are overwritten after
  launch, resolve must produce the pre overwrite values.

### 6.4 Proof

| Invariant | Violation | Measurement | Accept |
|---|---|---|---|
| Talker sees step N's hidden states | stale capture | unit test above, and codes bit identical between sync (`--thinker.factory.enable_async_decode false`) and lookahead at c16 with the same seeds | identical codes per request |
| Chunk order and count | overrun chunk or missing last chunk | code2wav receives the same number of frames per request in both arms (events `stage_stream_chunk_sent` per request) | equal counts |
| WER unchanged | any of the above | video speech WER and voice clone WER | unchanged within the run to run noise recorded in 04 |
| Step shrinks | lookahead not engaging | thinker trace at c16 video speech: step period (42 ms) and `_async_query_hit` versus `_async_query_miss` (base.py:136-137, not logged today, log them at shutdown) | period reported, misses under 10 percent |
| c1 unchanged | bs 1 stays on the fast path | voice clone c1 latency | unchanged |

Rollback: `--thinker.factory.enable_async_decode false` restores the
synchronous loop without a code change, or revert.

Status: Ready.

## 7. E, the talker step, measured (06)

Status: E0 done on 2026-09-01, doc 06 has the attribution and the
admission arithmetic. The overlap design (#1320 on #1204) stays held, 06
section 5 gives the mechanism behind its regression: two stream syncs per
step that serialize the host behind the in flight step. The work is three
groups, each one PR with one proof, accepted before the next starts.

### 7.1 Group E1, talker admission cap (request builder)

Title: `[Qwen3-Omni] Bound the talker max_new_tokens by the thinker text`

Seam: `_resolve_talker_sampling_config` and `_build_talker_request_data`
(request_builders.py). At build time with the thinker done the builder
knows the text length (one chunk per thinker token,
talker_prefill.py `extract_chunk_token_ids`).

Change: when the request did not pass `talker_max_new_tokens` and the
thinker is done, `max_new_tokens = min(4096, 64 + 32 * text_tokens)`.
Explicit `talker_max_new_tokens` is honored unchanged. Partial start
(thinker not done at build) keeps 4096. Constants from 06: 12.5 frames per
audio second, 1.8 to 5.2 frames per text token observed (p50 2.9), digit
heavy text can exceed the ratio, so 32 per token plus 64 leaves six
times the observed maximum.

Why the cap and not the SGLang estimate env
(`SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION`): the env changes only the
reservation and leaves the worst case unbounded, so under load the pool
can fill and SGLang retracts, and a retracted talker request re prefills
and re emits frames to code2wav. The per request cap bounds the worst
case too: with `max_running_requests` 32 and 14 token texts,
32 x (150 + 512) fits the 21373 token pool. Tradeoff: a runaway
generation stops at 5 s plus 2.6 s per text token of audio instead of at
328 s.

Tests: tests/unit_test/qwen3_omni/test_talker_max_new_tokens.py, the
bound function, the builder with a fake prefill builder (cap from the
chunk count, explicit override wins, thinker not done keeps the default).

Proof (container): voice clone events at c16 and c32 against the
baseline: running count in the talker (max and time share, from
prefill_start and stage_complete), admission wait p50, latency p50, RTF.
Codes bit identical at c1 (the cap binds on no request there).

### 7.2 Group E2, per step host syncs and copies (model code)

Title: `[Qwen3-Omni] Remove the talker decode step host syncs`

- E2a request_builders.py:812-833 leaves `req.multimodal_inputs` None
  when `talker_can_use_linear_mrope` holds. SGLang computes the identical
  positions on the device for requests without multimodal input
  (forward_batch_info.py:1212-1246 extend, :1160-1179 decode), the
  `MultimodalInputs` with linear positions only routes the decode step
  through the host read and the pageable copy (06 section 3.3). Contract
  change: `test_build_talker_request_uses_linear_mrope_without_mm_markers`
  asserts None, plus a position equivalence test against the arange
  SGLang uses.
- E2b components/talker.py:1046 writes a preallocated device bool scalar
  instead of the Python True (the 1 byte pageable copy and the sync on
  every step).
- E2c the suppress list is the same 1023 tokens for every request
  (request_builders.py:1085-1089), so `_suppress_mask` is built once per
  model as one row and broadcast into the batch rows. The repetition pairs
  and `_decode_prep_rep_rows` go through the pinned staging buffer instead
  of `torch.tensor(..., device=cuda)`.
- E3 the reuse check keyed by rid: rows already prepared are moved to
  their new row index with an index_select on the masks and parameter rows
  when SGLang compacts or appends, only genuinely new rows (a prefilled or
  retracted request) are built from their `output_ids`. With that,
  `invalidate_decode_buffers` on extend is no longer needed, since a
  prefilled request is a new row whose prefill token is in its
  `output_ids`.

Tests: test_talker.py decode buffer cases (reuse across a prefill, reuse
across a finished request, rebuild only on new rows, suppress mask equal
to the pair path), test_mrope_positions.py rewritten case.

Proof (container): the with_stack trace through trace_ingest.py and
trace_steps.py: cudaStreamSynchronize per steady step 0, pageable HtoD
per steady step 0, rebuild steps only where a row is new, c16 average
cycle within 10 percent of the steady cycle. Codes bit identical.

### 7.3 Group E4, request build and prefill (after E1 and E2 are measured)

- `request_build_max_workers` above 1 for the talker so the 9 ms build
  leaves the scheduler thread (omni_scheduler.py:196, 240-246), after the
  build's device work is checked for stream ordering on a worker thread.
- The prefill's eager 36 ms of host time for 8.75 ms of kernels, either
  the prefill graph that omni's policy refuses for the talker (04 section
  5.1) or fewer launches in the talker prefill forward.

### 7.4 E5, overlap

Only after E2 and E3, on top of #1204 and #1320, measured against the
steady cycle from 06.

### 7.5 The sweep, per group

One container session per group on the bf16 colocated server (E does
not depend on the thinker precision), the same functions from
scripts/h100_runs.sh, baseline and change interleaved on the same GPU:

- unit tests: `pytest tests/unit_test/qwen3_omni -q`.
- voice clone: `_events_run PORT LABEL bench_seedtts_vc` at c16, the same
  at c32 (`bench_seedtts_vc PORT LABEL 32` inside a request profile), and
  `bench_seedtts_vc PORT LABEL 1`.
- trace: `SGLANG_TORCH_PROFILER_WITH_STACK=1` then `_traces_run PORT LABEL seedtts-vc`,
  read with trace_ingest.py and trace_steps.py (with `--other` for the
  code2wav trace).
- numerics: `run_compare OLD NEW` on the codes at c1 with the fixed seeds.
- read: 06 section 3.1 table for the change, 06 section 2 table for the
  request, and the bench table.

Accept per group: the group's proof holds and nothing else in the
tables moved by more than the run to run noise (9 percent on qps at c16
for 50 requests, 06 section 1).

## 8. Order and dependencies

1. E0 on the H100 box (measurement, feeds tracker 2.3), and the B run, both
   container work.
2. D, code, after the note on #1258.
3. C, code.
4. A, code, then the encoder input path of 3.3.

Each item has one metric the others do not move, so the compare report
attributes them separately.

## 9. Open decisions

- Pinned budget for the encoder caches (A): keep 4 GiB per process or
  lower it for pinned mode. Owner: user. Evidence: locked memory from the A
  run. Affects A's resource line only.
- Replica count for preprocessing (B): the run. Owner: user with the host
  facts. Affects the B PR's defaults.
- Qwen3-TTS penalty ownership (E3): device side mask or synchronous
  penalty requests. Owner: user. Affects whether E3 is a runner change or a
  model change.
