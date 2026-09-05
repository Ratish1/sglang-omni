# Qwen3-TTS GPU memory provisioning and consumption mechanics

Repository read: `/Users/ratish/sglang-omni/.worktrees/qwen3-tts-predictor-chain` (read only).
Pinned dependency read: `/Users/ratish/sglang`, `git describe --tags` reports `v0.5.18`.

## Summary

For `Qwen/Qwen3-TTS-12Hz-1.7B-Base` launched from the shipped profile, the three stages
`preprocessing`, `tts_engine` and `vocoder` all declare `process="pipeline"`, so the launcher
builds one OS process group and spawns one OS process that owns all three stage objects. The
premise that there are three processes does not hold for the default configuration. Splitting
preprocessing out is opt in through configuration, and the code that supports it is present.

`tts_engine` and `vocoder` both declare `gpu=0` and neither declares `gpu_memory_fraction` nor
`total_reserve_bytes` nor `engine.kv_cache_bytes`, which is exactly what the observed placement
log reports. Because those are unset, the omni KV configurator falls through to upstream SGLang
profiling: the KV pool is sized as device-wide free memory after weight load minus
`pre_model_load_memory * (1 - mem_fraction_static)`, divided by the per-token KV cell size. The
0.15 residual that `mem_fraction_static=0.85` leaves is the entire budget from which the vocoder's
second copy of the speech tokenizer, the vocoder decode CUDA graphs, the predictor CUDA graphs,
the SGLang decode and prefill graphs, all activations and all per-request buffers must come. The
logged `avail mem=11.79 GB` is that residual measured immediately after the KV pool allocation and
before every stage-owned capture that follows.

The report covers the process and device layout, the KV sizing path in pinned SGLang and the omni
seam that feeds it, the reference-audio encode path in preprocessing, the vocoder's two decode
paths and its CUDA graph inventory, and the engine's non-KV device state.

---

## 1. Process and device layout

### 1.1 What the profile declares

`Qwen3TTSPipelineConfig.stages` declares three stages, all with `process="pipeline"`:

- `preprocessing`, `sglang_omni/models/qwen3_tts/config.py:48-57`. Factory
  `stages.create_preprocessing_executor`, `next="tts_engine"`. It declares no `gpu`. The note at
  `config.py:53-55` states that sharing the engine's process means the stage holds no GPU budget of
  its own, and that a split frontend passes `--preprocessing.gpu` with its own fraction.
- `tts_engine`, `config.py:58-66`. An `EngineStageConfig` with `factory=FactoryArgs(dtype="bfloat16")`,
  `gpu=0`, `next="vocoder"`, `stream_to=["vocoder"]`.
- `vocoder`, `config.py:67-75`. `factory=FactoryArgs(dtype="bfloat16")`, `gpu=0`, `terminal=True`,
  `can_accept_stream_before_payload=True`.

The shipped example config for this checkpoint sets nothing else:
`examples/configs/qwen3_tts_1_7b.yaml` contains only `config_cls: Qwen3TTSPipelineConfig` and
`model_path: Qwen/Qwen3-TTS-12Hz-1.7B-Base`.

`stage_factory_kwargs` at `config.py:82-103` adds `load_frontend=True` for `preprocessing` only when
`preprocessing_in_own_process()` (`config.py:78-80`) reports that the preprocessing stage's
`process` differs from the engine's. With the shipped profile both are `"pipeline"`, so
`load_frontend` stays False.

### 1.2 How that becomes OS processes

`compile_logical_processes` groups stages by `StageConfig.process` preserving config order
(`sglang_omni/config/topology.py:101-143`). `build_process_topology_plan` builds one
`ProcessGroupPlacement` per group name (`topology.py:283-300`), and `_resolve_group_gpu_id`
(`topology.py:340-357`) resolves the group's single GPU from the member stages, which is 0.

`_build_stage_groups` then creates one `StageGroup` per process-plan group, holding exactly one
`StageWorkerProcessSpec` whose `stage_specs` list is the group's stages in order
(`sglang_omni/pipeline/mp_runner.py:208-224`). `StageGroup.spawn` starts one
`multiprocessing.Process` per `process_spec` (`sglang_omni/pipeline/stage_workers.py:269-299`), so
the shipped profile spawns exactly one worker process, named `process-pipeline` by
`_process_name` (`stage_workers.py:955-963`).

Inside that process, `_run_process` constructs the stages in `spec.stage_specs` order, then runs
them concurrently on one asyncio event loop (`stage_workers.py:448-509`). Its docstring at
`stage_workers.py:462-465` states that scheduler construction is serialized by `gpu_startup_lock`
per GPU, so cold start for N same-GPU stages in one process degrades from max to sum.

`_validate_gpu_process_colocation` (`topology.py:404-475`) only raises the missing-footprint error
when a GPU carries more than one process group (`topology.py:456-458`). With a single group named
`pipeline` on GPU 0, the check is skipped, which is why `tts_engine` and `vocoder` can appear in
`missing_fraction_stages` without failing the launch. That list is produced by
`GpuPlacement.missing_fraction_stage_names` (`sglang_omni/config/placement.py:235-236, 250`) and
rendered by `_placement_log_summary` (`sglang_omni/serve/launcher.py:189-198`). The per-stage
`stage_runtime` block that reported `None` for every budget is
`_stage_runtime_log_summary` (`launcher.py:127-147`), which reads
`stage.gpu_memory_fraction`, `stage.engine.kv_cache_bytes`, `stage.total_reserve_bytes` and
`stage.engine.mem_fraction_static` straight off the stage config. `preprocessing` is absent from
that block because `launcher.py:137-138` skips a stage with no `gpu`, no fraction and no
`kv_cache_bytes`.

### 1.3 Device binding and startup allocations, in construction order

`_construct_stage` calls `current_platform.set_device(int(gpu_id))` only when the stage spec has a
GPU id (`stage_workers.py:619-622`). `_construct_scheduler` takes `gpu_startup_lock(gpu_id)` around
the factory call when a GPU id is present (`stage_workers.py:880-885`), and applies a per-process
torch allocator cap only when `total_reserve_bytes` is set and enforcement is on
(`_apply_total_reserve_cap`, `stage_workers.py:810-845`). Neither `tts_engine` nor `vocoder`
declares `total_reserve_bytes`, so no `torch.cuda.set_per_process_memory_fraction` cap is applied.

1. **preprocessing.** `gpu_id` resolves to None because the stage declares no `gpu`
   (`mp_runner.py:128`, `placement.py:179-186`), so `set_device` is not called and no startup lock
   is taken. `create_preprocessing_executor` with `load_frontend=False`
   (`sglang_omni/models/qwen3_tts/stages.py:106-136`) returns a `ThreadedSimpleScheduler` wrapping
   `preprocess_qwen3_tts_payload` at `max_concurrency=8`. No device allocation happens here. The
   stage reaches its model objects at request time through the module-level
   `_PREPROCESSING_CONTEXT` that the engine builder publishes
   (`sglang_omni/models/qwen3_tts/request_builders.py:169-190`).

2. **tts_engine.** `set_device(0)`, then `gpu_startup_lock(0)`, then
   `create_sglang_tts_engine_executor` (`stages.py:187-210`), which runs
   `Qwen3TtsEngineBuilder().build(...)` (`sglang_omni/scheduling/engine_factory.py:64-261`). In
   order, that allocates:
   - talker weights, through `ModelWorker` inside `create_sglang_infrastructure`
     (`sglang_omni/scheduling/bootstrap.py:159-164`),
   - the KV pool and the request pool, `model_runner.alloc_memory_pool()`
     (`bootstrap.py:180`),
   - attention backend workspaces, `model_runner.init_attention_backends()` (`bootstrap.py:181`),
   - the per-model persistent decode buffers, allocated in `Qwen3TTSTalkerTextModel.__init__`
     (`sglang_omni/models/qwen3_tts/sglang_model.py:255-273`) and `Qwen3TTSTalker.__init__`
     (`sglang_model.py:886-956`), both sized by `server_args.max_running_requests`, which is 16,
   - the speech tokenizer, loaded on the talker's device in `setup_model`
     (`sglang_omni/models/qwen3_tts/engine_builder.py:123-129` calling
     `stages._load_qwen3_tts_tokenizer`, `stages.py:44-68`),
   - the HF `AutoProcessor` (`engine_builder.py:130-133`), host side,
   - the reference-encode service and its private CUDA stream, created by
     `set_qwen3_tts_preprocessing_context` (`request_builders.py:184`) which builds
     `_Qwen3TTSAdhocReferenceHook` and therefore `_Qwen3TTSRefCodeBatcher`
     (`request_builders.py:883-894`, stream at `request_builders.py:749` and `716-725`),
   - SGLang decode and prefill CUDA graphs, `init_sglang_cuda_graphs`
     (`engine_factory.py:222-223`, `bootstrap.py:42-65`),
   - the predictor CUDA graph set, `setup_model_resources` calling
     `model.capture_predictor_graphs(...)` (`engine_builder.py:150-169`).

   Note that `create_sglang_tts_engine_executor` does not declare a `total_gpu_memory_fraction`
   parameter (`stages.py:187-197`), and `resolve_factory_signature_args` injects a placement default
   only when the factory declares the parameter (`sglang_omni/config/runtime.py:159-183`). A
   `gpu_memory_fraction` written on this stage would therefore never reach the KV configurator.
   `engine.kv_cache_bytes` uses a different route, a thread-local scope, and does reach it (see
   section 2.3).

3. **vocoder.** `set_device(0)`, `gpu_startup_lock(0)`, then `create_vocoder_executor`
   (`stages.py:216-280`). It calls `_load_qwen3_tts_tokenizer` again (`stages.py:245-250`), so a
   second full copy of the speech tokenizer weights becomes resident in the same process on the
   same device. It then constructs `Qwen3TTSStreamingVocoderScheduler` and calls
   `scheduler.warmup_now()` before returning (`stages.py:276-279`), with the note that factory
   construction completes before the stage process publishes readiness, so CUDA capture cannot
   overlap request-time GPU work from colocated stages. `warmup_now`
   (`sglang_omni/models/qwen3_tts/streaming_vocoder.py:699-704`) captures the initial decode graphs
   and every follow-up worker's graphs. Because this runs after the engine's KV pool allocation,
   these captures draw from whatever the KV sizing left behind.

---

## 2. How the KV pool size is derived

### 2.1 The upstream path when no omni budget is declared

`KVCacheConfigurator.configure` calls `_resolve_memory_pool_config`
(`/Users/ratish/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py:276-309`), which calls
`_profile_available_bytes`, then `config_from_budget`, then `resolve_max_num_reqs`
(`kv_cache_configurator.py:1928-1944`). It logs `Memory pool end. avail mem=...` at
`kv_cache_configurator.py:294-297`, after the pools are already allocated, using
`get_available_gpu_memory`.

`_profile_available_bytes` (`kv_cache_configurator.py:1764-1811`) is the sizing rule. Its own
comment at lines 1765-1767 states that the KV pool budget is currently free GPU memory minus the
non-static runtime slack `pre_model_load_memory * (1 - mem_fraction_static)`, so whatever is already
resident is charged against it. Concretely:

```
available_gpu_memory = get_available_gpu_memory(device, gpu_id, ...)   # line 1768
slack_gb   = pre_model_load_memory * (1 - mem_fraction_static)         # line 1775
mm_res_gb  = mm_runtime_reservation_gb(is_multimodal, mm_feature_transport)  # line 1785
rest_memory = available_gpu_memory - slack_gb - mm_res_gb              # line 1789
return int(rest_memory * (1 << 30))                                    # line 1811
```

`get_available_gpu_memory` for CUDA empties the caching allocator and returns
`torch.cuda.mem_get_info(gpu_id)` free bytes
(`/Users/ratish/sglang/python/sglang/srt/utils/common.py:414-443`). That is device-wide free memory,
not per-process, so a colocated stage that has already allocated on the card shrinks this number.

`pre_model_load_memory` is captured in `init_torch_distributed` after distributed setup and before
weight load, with the same device-wide free-memory call
(`/Users/ratish/sglang/python/sglang/srt/distributed/bootstrap.py:132-137`, returned at line 158,
stored on the ModelRunner at `model_executor/model_runner.py:1042`).

`mm_runtime_reservation_gb` (`kv_cache_configurator.py:130-147`) is the function that logs
`Reserving %.2f GB of the KV budget for post-sizing multimodal allocations`. It returns 0.0 unless
`model_config.is_multimodal`, and it only logs when the reserved amount is positive. The reserve is
`SGLANG_VLM_CACHE_SIZE_MB`, plus `SGLANG_MM_FEATURE_CACHE_MB` when `mm_feature_transport` is
`cuda_ipc` or `cuda_vmm`. The omni bootstrap does flip `model_config.is_multimodal` to True
temporarily, but only around `init_cuda_graphs` and only when prefill input embeds are enabled
(`sglang_omni/scheduling/bootstrap.py:56-65`), which is after KV sizing.

### 2.2 Bytes to tokens

`config_from_budget` (`kv_cache_configurator.py:1946-1968`) hands the byte budget to
`create_memory_pool_configurator(self).calculate_pool_sizes(budget_bytes, page_size)`, applies
`_apply_token_constraints`, and recomputes from a constrained token count if the constraint bit.

For a dense MHA model the configurator is `DefaultPoolConfigurator`
(`/Users/ratish/sglang/python/sglang/srt/model_executor/pool_configurator.py:141-430`).
`calculate_pool_sizes` is simply

```
max_total_num_tokens = available_bytes // self._cell_size          # line 418-422
max_total_num_tokens = max_total_num_tokens // page_size * page_size  # line 423
```

and the cell size for the non-MLA, non-sparse branch is
(`pool_configurator.py:327-333`)

```
n = model_config.get_num_kv_heads(tp_size, dcp_size)
cell_size = n * (head_dim + v_head_dim) * effective_num_layers * kv_size
```

with `kv_size = torch._utils._element_size(kv_cache_dtype)`, which is 2 for bfloat16.

`_apply_token_constraints` (`kv_cache_configurator.py:1844-1871`) is where `max_total_tokens`
interacts. When `server_args.max_total_tokens` is set it is applied as
`token_capacity = min(token_capacity, user_limit)`, and a user limit above the profiled value only
logs a warning and is ignored. It is a cap, never a floor. The Qwen3-TTS profile does not set it
(`EngineArgs.max_total_tokens` defaults to None, `sglang_omni/config/schema.py:120`, and
`Qwen3TtsEngineBuilder.generation_defaults` does not include it,
`engine_builder.py:82-95`), so no clamp applies.

`resolve_max_num_reqs` (`kv_cache_configurator.py:1873-1926`) then computes the per-worker running
cap: with an explicit `max_running_requests` it is
`min(max_running_requests // attn_dp_size, token_capacity // 2)`. At 16 requested and 589142 tokens
the token term is not binding.

The `KV Cache is allocated. dtype: ..., #tokens: N, K size: ... GB, V size: ... GB` line comes from
`_finalize_allocation_log` in
`/Users/ratish/sglang/python/sglang/srt/mem_cache/memory_pool.py:1670-1696`. The same line reads
`VA upper bound` instead of `is allocated` when `post_capture_active` is set (line 1685), so the
observed wording means post-capture KV sizing was not active for this run.
`ServerArgs.post_capture_kv_sizing_planned` (`/Users/ratish/sglang/python/sglang/srt/server_args.py:5006-5055`)
returns False unless a long list of conditions holds, among them a non-disabled decode CUDA graph
backend and, for a non-decode disaggregation mode, a prefill graph backend that is not disabled with
`max_prefill_buffer_tokens()` inside the captured prefill bucket list (lines 5037-5047). The Base
profile leaves the prefill graph backend at SGLang's default, since
`cuda_graph_backend_prefill` is only set for CustomVoice checkpoints
(`engine_builder.py:96-107`).

Consistency check on the observed numbers, using only the formulas above.
`31.46 + 31.46 = 62.92` GiB over 589142 tokens gives 114688 bytes per token, which is exactly
`8 * (128 + 128) * 28 * 2`. That fixes the product `n * (head_dim + v_head_dim) * layers`, though the
individual factors are not read from the checkpoint here (see Unverified). Working the other
direction, `rest_memory` of about 62.93 GiB plus the reported post-allocation free memory of 11.79 GB
implies `pre_model_load_memory * 0.15` is about 11.79, that is `pre_model_load_memory` about 78.6 GB.
The reported `avail mem=11.79 GB` is therefore, to within allocator rounding, precisely the slack
term the formula reserved.

### 2.3 How a stage byte budget or fraction would reach that code

Two distinct routes exist, and the Base profile uses neither.

**Byte budget.** `StageConfig.engine.kv_cache_bytes` (`schema.py:128-142`, parsed by
`parse_memory_bytes`, `schema.py:46-76`) is copied into the launch spec by
`_stage_byte_budget_kwargs` (`mp_runner.py:293-302`) as a first-class field, never a factory kwarg,
with the note at `stage_workers.py:71-72` that no factory signature can absorb it. In the worker,
`_construct_scheduler` wraps the single factory invocation in
`stage_kv_cache_budget(stage_name, kv_cache_bytes)` (`stage_workers.py:874-885`), a thread-local
scope defined in `sglang_omni/scheduling/stage_kv_budget.py:34-60`, which raises on exit if the
budget was never consumed. `create_sglang_infrastructure` consumes it at the single engine
construction choke point via `consume_stage_kv_cache_bytes` (`bootstrap.py:131`,
`stage_kv_budget.py:63-80`, which raises if a second engine tries to consume the same budget) and
passes it into `ModelWorkerConfig` (`bootstrap.py:132-139`). From there it reaches
`SGLModelRunner` (`sglang_omni/model_runner/model_worker.py:73-74, 250-251`) and finally
`_OmniKVCacheConfigurator.kv_cache_bytes` through `init_kv_cache_configurator`
(`sglang_omni/model_runner/sglang_model_runner.py:560-578`).

`_OmniKVCacheConfigurator._profile_available_bytes`
(`sglang_model_runner.py:68-147`) then short-circuits: when `kv_cache_bytes` is set it is
authoritative and returned directly after a free-memory check (lines 93-120), and mamba or hybrid
models are refused (lines 96-102). `post_capture_resize_kv_pool` (`sglang_model_runner.py:448-474`)
raises rather than silently shrinking a declared byte budget.

Setting the byte budget also strips the builder's fraction. `SGLangGenerationEngineBuilder.build`
peeks the scoped budget and, when present, pops `mem_fraction_static` out of the overrides so the
headroom derives cleanly (`engine_factory.py:134-145`). Schema validation refuses the combination
outright anyway (`schema.py:147-160`), and also refuses `kv_cache_bytes` together with
`max_total_tokens` (`schema.py:154-160`), on the ground that the lower token cap would silently
shrink the byte-derived pool.

**Fraction budget.** `StageConfig.gpu_memory_fraction` (`schema.py:331-341`) becomes
`total_gpu_memory_fraction` in `factory_arg_defaults`
(`sglang_omni/config/runtime.py:152-153`) and is injected only if the factory declares that
parameter (`runtime.py:179-182`). For Qwen3-TTS the engine factory does not
(`stages.py:187-197`). When it does reach a factory, it travels into
`create_sglang_infrastructure(total_gpu_memory_fraction=...)` (`bootstrap.py:107, 136`) and drives
`_profile_available_bytes_from_process_memory` or
`_profile_available_bytes_from_stage_load_delta` (`sglang_model_runner.py:122-209`), which size the
KV headroom as `total_memory * fraction - accounted_used`
(`calculate_stage_budget_available_bytes`, `sglang_omni/utils/gpu_memory.py:224-249`).

`StageConfig.total_reserve_bytes` (`schema.py:342-355`) is a third, separate control. Its
description says it drives placement capacity checks and MPS preflight arithmetic and, unless
`enforce_total_reserve` is false, is enforced at stage startup through a per-process torch allocator
cap so that a stage that outgrows its budget fails itself instead of a co-tenant. That cap is
`_apply_total_reserve_cap` (`stage_workers.py:810-845`), which sums the declared reserves per device
across the stages in one worker process and calls `torch.cuda.set_per_process_memory_fraction`.

### 2.4 What the default Qwen3-TTS profile sets and leaves unset

`Qwen3TtsEngineBuilder.generation_defaults` (`engine_builder.py:77-107`) sets, for `dtype="bfloat16"`:

- `max_running_requests: 16`, `max_queued_requests: 16`
- `cuda_graph_max_bs: 32`, `torch_compile_max_bs: 32`
- `dtype: "bfloat16"`, `disable_cuda_graph: False`, `disable_overlap_schedule: True`,
  `enable_torch_compile: False`
- `mem_fraction_static: 0.85`
- `max_prefill_tokens: 8192`
- `sampling_backend: "pytorch"`, `trust_remote_code: True`

and, only for a checkpoint whose `tts_model_type` is `custom_voice`, also
`cuda_graph_backend_prefill: BREAKABLE` and `cuda_graph_bs_prefill` from
`QWEN3_TTS_PREFILL_CUDA_GRAPH_BS`. The note at `engine_builder.py:100-104` says Base prefills also
carry reference audio, giving them a different shape distribution, so they keep the eager path until
measured. `Qwen3TtsEngineBuilder.context_length = 8192` (`engine_builder.py:41`).

Left unset by the profile: `max_total_tokens`, `chunked_prefill_size`, `kv_cache_bytes`,
`total_reserve_bytes`, `gpu_memory_fraction`, `cpu_offload_gb`, `quantization`,
`page_size`. `EngineStageConfig` gives `tts_engine` an empty `EngineArgs`
(`schema.py:456-461`), and `EngineArgs.overrides()` (`schema.py:162-174`) emits only keys that are
actually set, so SGLang's own defaults stay in charge for everything the builder does not name.

`build_generation_batch_overrides` (`sglang_omni/scheduling/generation_batch_policy.py:99-217`)
fills `cuda_graph_bs` from `build_default_cuda_graph_bs(cuda_graph_max_bs)` when it is not declared
(line 153-154). With `cuda_graph_max_bs=32` that ladder is `[1, 2, 4, 8, 12, 16, 24, 32]`
(`generation_batch_policy.py:38-50`). `validate_generation_batch_policy`
(`generation_batch_policy.py:220-295`) requires `max(cuda_graph_bs) == cuda_graph_max_bs` and
`cuda_graph_max_bs >= max_running_requests`, both satisfied at 32 and 16.

---

## 3. The reference audio encoder in the preprocessing stage

### 3.1 Request to encoder forward

`preprocess_qwen3_tts_payload` (`request_builders.py:1261-1291`) is the stage compute function. It
reads the module-level `_PREPROCESSING_CONTEXT`, calls `_prepare_qwen3_tts_request`, and, when the
context is not standalone, stores the prepared object in the module-level `_PREPARED_REQUESTS` dict
keyed by `payload.request_id` and returns a payload carrying only a marker
(`request_builders.py:1282-1291`). The engine stage pops it with
`pop_prepared_qwen3_tts_request` (`request_builders.py:214-229`), which raises if it is missing.

`_prepare_qwen3_tts_request` (`request_builders.py:1177-1258`) dispatches on task type. For a Base
checkpoint the task must be `Base` (`_validate_qwen3_tts_model_task`, `request_builders.py:1062-1080`),
so `_prepare_qwen3_tts_base_request` (`request_builders.py:1083-1139`) runs. It has three branches:

- an uploaded voice whose `SpeakerCacheKey` hits the process-wide speaker cache
  (`request_builders.py:1089-1098`), no encode,
- an ad-hoc reference, `cache_key is None`, which goes through
  `_get_qwen3_tts_adhoc_reference_service(model, wrapper).get_or_encode(state, ...)`
  (`request_builders.py:1099-1104`),
- an uploaded voice that missed, which calls `wrapper.create_voice_clone_prompt(...)` directly and
  populates the speaker cache (`request_builders.py:1105-1122`).

`ReferenceEncodeService.get_or_encode`
(`sglang_omni/scheduling/reference_encoder.py:300-369`) does keyed single-flight: a cache hit
returns `load_artifact(stored)`, a concurrent same-key request follows the leader's future, and a
miss encodes, stores, revalidates and caches under one guard.

`_Qwen3TTSAdhocReferenceHook.encode_one` (`request_builders.py:923-959`) is the encode itself. Under
`torch.no_grad()` it normalizes the audio through `wrapper._normalize_audio_inputs`, submits the
waveform to the reference-code batcher, resamples the waveform on the host with `librosa` when the
sample rate differs from `model.speaker_encoder_sample_rate`, runs
`model.extract_speaker_embedding(...)`, and then waits on the batcher future with a 130 second
timeout.

The codec encode is inside `_Qwen3TTSRefCodeBatcher._run`
(`request_builders.py:819-869`). Under `torch.inference_mode()` and inside the batcher's private
CUDA stream context, it calls

```
self._speech_tokenizer.encode(waveforms, sr=sample_rate).audio_codes    # line 840-843
```

falling back to per-item `encode(waveform, sr=...)` when the batched call raises (lines 849-857).

### 3.2 Dtype

The speech tokenizer is constructed by `_load_qwen3_tts_tokenizer` (`stages.py:44-68`), which does
`torch_dtype = getattr(torch, dtype)` and passes `device_map=device, dtype=torch_dtype` to
`Qwen3TTSTokenizer.from_pretrained`. In the engine process the `dtype` argument is the builder's
`self.dtype`, set from `build(dtype=...)` (`engine_factory.py:90`), which is `"bfloat16"` because
the stage declares `FactoryArgs(dtype="bfloat16")` (`config.py:62`) and the factory forwards it
(`stages.py:191, 207`). So the request is for bf16 weights on the talker's device. `attn_implementation`
is not passed by default (`stages.py:64-65`, factory default None at `stages.py:193`).

### 3.3 Batching

Two independent batchers exist and only one of them is active for Qwen3-TTS.

`ReferenceEncodeService` is constructed with `max_items=256, max_bytes=64 MiB, timeout_s=130.0`
and no `max_batch_size` (`request_builders.py:1028-1034`), so `max_batch_size` stays at its default
of 1 (`reference_encoder.py:141`). `self._batching` requires both `max_batch_size > 1` and
`hook.can_encode_batch()` (`reference_encoder.py:172`), and `KeyedReferenceEncodeHook` does not
override `can_encode_batch`, which returns False at `reference_encoder.py:57-58`. The service
therefore never starts its batch worker and `_encode_leader` calls `hook.encode_one` directly
(`reference_encoder.py:201-206`).

The batching that does happen is `_Qwen3TTSRefCodeBatcher` (`request_builders.py:737-869`),
constructed once per hook with `max_batch_size=8, max_batch_wait_ms=2.0`
(defaults at `request_builders.py:743-744`, constructed at `request_builders.py:889-892`). It owns a
daemon thread `qwen3-tts-ref-code`. `_drain` takes one queued item, then collects up to seven more
within a 2 ms deadline (`request_builders.py:774-795`). `_run` groups the drained batch by sample
rate and issues one `speech_tokenizer.encode(list_of_waveforms, sr=...)` per group
(`request_builders.py:827-843`). This is why `create_preprocessing_executor` defaults
`max_concurrency=8`, with the note at `stages.py:125-128` that a serial executor would only ever
show the batcher batches of one and that the default matches the batcher's `max_batch_size`.

The speaker-embedding forward is not batched. Each `encode_one` calls
`extract_speaker_embedding` for its own waveform on the calling thread's current stream
(`request_builders.py:946-949`).

### 3.4 What is cached and its byte bound

Two caches.

- The ad-hoc reference cache inside `ReferenceEncodeService` is a `StageOutputCache` with
  `max_size=256` and `max_bytes=64 * 1024 * 1024` (`reference_encoder.py:154`, arguments at
  `request_builders.py:1030-1031`). `StageOutputCache.put` rejects any single entry larger than
  `max_bytes` and otherwise evicts LRU until the byte total fits
  (`sglang_omni/scheduling/stage_cache.py:113-132, 173-184`), counting bytes by
  `numel * element_size` for tensors and recursing into dicts, lists and tuples
  (`stage_cache.py:59-68`). The stored value is host resident: `store_artifact`
  (`request_builders.py:961-966`) builds the artifact through `_cacheable_qwen3_tts_voice_prompt`
  (`request_builders.py:666-685`), whose tensor helper is
  `value.detach().to(device="cpu").clone()` (`request_builders.py:688-689`). The cache is
  constructed with `cache_device=None`, so `_detach_value` leaves the device alone
  (`stage_cache.py:30-46`), which means the 64 MiB bound is a host-memory bound, not a GPU one.
- The uploaded-voice cache is the process-wide `SpeakerArtifactCache` with
  `DEFAULT_SPEAKER_CACHE_BYTES = 512 * 1024 * 1024`
  (`sglang_omni/scheduling/speaker_cache.py:15, 29-39, 125-129`), sized by `estimate_cache_bytes`
  (`speaker_cache.py:83-106`). Entries go in through the same
  `_cacheable_qwen3_tts_voice_prompt` helper (`request_builders.py:1116-1122`), so they too are host
  resident.

Loading from either cache clones back out: `_qwen3_tts_voice_prompt_from_cache`
(`request_builders.py:692-706`) clones each embedding and each reference code tensor per request.

### 3.5 Every device allocation on the path

Walking `_prepare_qwen3_tts_base_request` and its callees, in order:

1. One private CUDA stream per hook instance, created once at context publication
   (`_new_cuda_encode_stream`, `request_builders.py:716-725`, called at `request_builders.py:749`).
2. `speech_tokenizer.encode(...)` allocates the codec activations and the returned `audio_codes`
   on that private stream (`request_builders.py:836-843`). The batcher waits on a
   `torch.cuda.Event` recorded on the private stream before resolving futures
   (`_synchronize_outcomes`, `request_builders.py:797-817`), and consumers call
   `_record_ref_code_consumer_stream` (`request_builders.py:728-734`) so the caching allocator
   cannot recycle the block while consumer reads are still queued.
3. `librosa.resample` on the host when the sample rates differ (`request_builders.py:938-945`), a
   numpy allocation, not a device one.
4. `model.extract_speaker_embedding(...)` on the calling thread's current stream
   (`request_builders.py:946-949`), returning a device tensor.
5. `store_artifact` copies the reference codes and the speaker embedding device-to-host
   (`request_builders.py:688-689`), and `load_artifact` clones them back on the host
   (`request_builders.py:699-704`).
6. `wrapper._tokenize_texts(...)` for the assistant text, the reference text and the optional
   instruction (`request_builders.py:1124-1130`, `_build_instruct_id` at
   `request_builders.py:644-651`).
7. `model.build_voice_clone_inputs(...)` under `torch.no_grad()` (`request_builders.py:1131-1139`),
   which returns `input_embeds`, `attention_mask`, `trailing_text_hidden` and `ref_code`. This is
   the largest per-request device allocation on the path, of order prompt length times hidden size
   in bf16.
8. `prompt_input_embeds` and `trailing_text_hidden` are moved to the talker feedback buffer's device
   and dtype (`request_builders.py:1226-1244`), and `ref_code` to that device
   (`request_builders.py:1245-1246`).
9. `build_embedding_cache_key_ids(prompt_input_embeds)` (`request_builders.py:614-621`) does
   `.to(dtype=torch.float32, device="cpu")` on the whole prompt embedding, so it allocates a host
   float32 copy of the full `[L, hidden]` block and then runs one blake2b per row.
10. `_build_qwen3_tts_pad_embed(model)` (`request_builders.py:624-641`) runs a text embedding lookup
    plus the text projection on the device, once per request.

When preprocessing runs in its own process the prepared tensors are attached to the state and
shipped in the payload instead of the registry (`_store_prepared_qwen3_tts_payload`,
`request_builders.py:1304-1323`), which also drops `state.ref_audio` with the note that the
reference clip is consumed there and read by nothing downstream. The engine side rehydrates with
`_load_prepared_qwen3_tts_request` (`request_builders.py:1326-1362`), moving each tensor onto the
feedback buffer's device and dtype and clearing the payload fields it consumed.

---

## 4. The vocoder stage

The vocoder stage's compute functions live on `Qwen3TTSStreamingVocoderScheduler`
(`sglang_omni/models/qwen3_tts/streaming_vocoder.py:399-1938`), which extends
`StreamingVocoderBase` (`sglang_omni/scheduling/streaming_vocoder.py:84-130`), which extends
`StreamingSimpleScheduler` (`sglang_omni/scheduling/streaming_simple_scheduler.py:34-78`).

### 4.1 Non-streaming whole utterance decode

`_vocode_payloads` (`streaming_vocoder.py:1865-1895`) is the batch compute function, registered
through `super().__init__(self._vocode_payload, batch_compute_fn=self._vocode_payloads, ...)`
(`streaming_vocoder.py:679-686`). It rebuilds one `Qwen3TTSState` per payload, requires
`state.audio_codes`, and calls

```
wavs, sample_rate = self._tokenizer.decode([{"audio_codes": item} for item in codes])
```

once for the whole batch (`streaming_vocoder.py:1884-1886`). Under
`enable_deterministic_inference` it instead loops one decode per item
(`streaming_vocoder.py:1877-1882`). `_store_vocoder_result` (`streaming_vocoder.py:1897-1921`) trims
the reference prefix proportionally by `ref_code_len / total_frames` and builds the audio payload.

The batch composition for this path is the base scheduler's:
`max_batch_size=8` and `max_batch_wait_ms=2` from the factory defaults
(`stages.py:223-224`) forwarded to the base class (`streaming_vocoder.py:684-685`). The collection
rule is `SimpleScheduler._collect_batch` (`sglang_omni/scheduling/simple_scheduler.py:106-120`),
which only batches when a `batch_compute_fn` exists and `max_batch_size > 1`, and waits at most
`max_batch_wait_ms` when idle.

The whole-utterance path also serves as the streaming fallback: `fallback_full_decode`
(`streaming_vocoder.py:1836-1843`) delegates to `_decode_state_audio`
(`streaming_vocoder.py:1923-1935`), a single-item `tokenizer.decode`.

### 4.2 Streaming path

Chunk arrival, in `Qwen3TTSStreamingVocoderScheduler`:

- `latch_stream_contract` (`streaming_vocoder.py:760-831`) records `num_quantizers`,
  `ref_code_len` (as `pending_ref_frames`), the first-chunk frame count, and whether bootstrap
  silence suppression applies. Suppression is only enabled when the number of live streams is at
  most `_suppress_bootstrap_max_streams`, which the factory sets to 24 (`stages.py:242`), and it
  bumps the first chunk by one frame with the note at `streaming_vocoder.py:818-823` that the
  withheld frame is also withheld from the client's playback buffer.
- `validate_chunk` (`streaming_vocoder.py:833-866`) normalizes to `[T, Q]`, checks the quantizer
  count, and range-checks against `_QWEN3_TTS_CODEBOOK_SIZE = 2048` only for host tensors.
- `ingest` (`streaming_vocoder.py:868-884`) appends the chunk and advances `total_frames`.
- `should_decode` (`streaming_vocoder.py:886-891`) fires when the generated frame count reaches
  `_next_decode_threshold`, which is `next_decode_generated_frames` once set and otherwise the
  stream's `initial_chunk_frames` (`streaming_vocoder.py:893-896`).

`_decode_and_emit` (`streaming_vocoder.py:1446-1486`) routes to the async workers when
`_async_decode` is on, which it is on CUDA by default (`streaming_vocoder.py:614-622`). The first
decode of a stream goes to `_schedule_initial`, every later one to `_schedule_followup`
(`streaming_vocoder.py:1453-1458`).

`_build_decode_plan` (`streaming_vocoder.py:1028-1070`) sets the decode window:

```
absolute_emitted = ref_frames + emitted_generated_frames
window_start     = max(0, absolute_emitted - stream_left_context_frames)
window_end       = ref_frames + generated_frames
```

so the decoder input is `[1, Q, window_end - window_start]`. The note at
`streaming_vocoder.py:1050-1052` says `window_start` only moves forward, so frames behind it are
dead and whole chunks are pruned to keep the concatenation O(window) rather than O(stream).
`_extract_delta` (`streaming_vocoder.py:1366-1376`) trims the left-context prefix and emits exactly
`(generated_frames - emitted_generated_frames) * samples_per_frame` samples.

Batch composition rules for the two async workers:

- **Initial worker.** One thread, `qwen3-tts-vocoder-initial` (`streaming_vocoder.py:713-717`).
  `_collect_async_batch` (`streaming_vocoder.py:1526-1549`) blocks for one item, then collects up
  to `initial_max_batch_size` more within `initial_batch_wait_s`. Factory defaults are 32 and 2 ms
  (`stages.py:232-233`). `_run_initial_batch` (`streaming_vocoder.py:1562-1592`) caps each plan at
  `max_generated_frames = state.initial_chunk_frames or stream_stride`.
- **Follow-up workers.** `followup_worker_count = 2` threads by default (`stages.py:236`), reduced
  to 1 under deterministic inference (`streaming_vocoder.py:519-521`). They drain a
  `PriorityQueue` keyed by `state.playback_deadline_s` (`streaming_vocoder.py:656-659`,
  `_enqueue_followup` at `streaming_vocoder.py:1512-1524`). `_collect_followup_batch`
  (`streaming_vocoder.py:1686-1705`) takes one item then collects up to
  `followup_max_batch_size` within `followup_batch_wait_s`, defaults 8 and 1 ms
  (`stages.py:234-235`). Collection is serialized behind `_followup_collect_lock`
  (`streaming_vocoder.py:1680`) with the note at `streaming_vocoder.py:665-669` that two workers
  draining in parallel would split one batch into two single-request decodes, and that only
  collection is serialized so decodes still overlap. `_run_followup_batch` caps each plan at
  `_next_decode_threshold(state)` (`streaming_vocoder.py:1719`).
- **Shape grouping.** Whatever a worker collected, `_group_decode_plans`
  (`streaming_vocoder.py:1623-1633`) partitions the plans by exact `decoder_input.shape`, and each
  group is launched as one concatenated batch (`_launch_decode_plans`,
  `streaming_vocoder.py:1130`). So a launched batch is uniform in both quantizer count and frame
  count.

`_next_followup_stride` (`streaming_vocoder.py:1427-1444`) walks the configured chunk ramp by
emitted frame count and then settles on the steady follow-up stride.

### 4.3 CUDA graphs captured at startup, and their static buffers

`_Qwen3TTSInitialDecodeGraphs` (`streaming_vocoder.py:307-396`) holds one graph per
`(input_frames, batch_size)` key. `capture` (`streaming_vocoder.py:333-377`) allocates a static
input `torch.zeros((batch_size, num_quantizers, input_frames), dtype=torch.long)` per key, runs two
warmup decodes on a dedicated capture stream, then captures the decoder call into a
`torch.cuda.CUDAGraph` sharing one `graph_pool_handle` across all keys of that holder. The static
output is whatever `self._decoder(static_input)` returned inside the capture. `decode`
(`streaming_vocoder.py:379-396`) zeroes the static input, copies the live rows in, replays, and
clones the first `batch_size` rows of the static output, so the returned waveform is a fresh
allocation each replay.

Three holders exist: one initial (`streaming_vocoder.py:568-579`) and one per follow-up worker
(`streaming_vocoder.py:580-594`, with the note at `streaming_vocoder.py:671-675` that each worker
owns its graphs and stream so two threads cannot race on the static buffers). All three are
captured by `warmup_now` before the process publishes readiness (`streaming_vocoder.py:699-704`,
called from `stages.py:279`).

The frame-count set is computed by `_decode_graph_frame_counts`
(`streaming_vocoder.py:54-90`), whose docstring explains the two families of window size. With the
shipped defaults, `stream_left_context_frames=16`, ramp `(1, 2, 4)` giving
`initial_chunk_frames=1` and a follow-up ramp of `(2, 4)`
(`streaming_vocoder.py:473-479`), and `stream_followup_stride=8`, the base set is
`(1, 3, 7, 15, 17, 18, 19, 20, 21, 22, 23, 24)`. Because `suppress_bootstrap_silence` is True and
`initial_chunk_frames < stream_stride`, the bumped schedule is unioned in
(`streaming_vocoder.py:549-567`), adding `(2, 4, 8, 16)`, for a union of 16 distinct frame counts.
The batch-size ladder is `(1, 2, 4, 8)` unless deterministic inference forces `(1,)`
(`streaming_vocoder.py:573, 586`). That is 16 times 4, so 64 graph keys per holder and 192 capture
attempts across the three holders. Each key holds at least a static input of
`batch_size * num_quantizers * input_frames * 8` bytes plus the decoder output tensor and the
intermediates retained in the holder's shared graph pool. A capture that raises is logged and
skipped (`streaming_vocoder.py:362-369`).

The comment at `streaming_vocoder.py:535-542` records why the span is captured whole: capturing only
`left_context + {ramp}` left roughly 72 percent of decodes on the eager path, 4 to 5 times slower.

### 4.4 Pinned staging and how activation memory scales

Every async launch reserves the calling thread's `_DecodeSlot`
(`_thread_decode_slot`, `streaming_vocoder.py:1256-1272`), a `GrowablePinnedBuffer` for input codes
plus a `PinnedTransferSlot` for output samples
(`sglang_omni/utils/cuda_staging.py:36-74, 77-171`). `_reserve_slot`
(`streaming_vocoder.py:1274-1300`) grows both to the exact requirement of this launch:
`input_numel` is zero when the input is already on the device, and `output_numel` is
`sum(generated_frames - emitted_generated_frames) * samples_per_frame` over the batch. The buffers
only ever grow (`cuda_staging.py:55-62`), so a thread's pinned footprint settles at the largest
batch it has ever seen. Slots are per thread and threads are the initial worker plus the follow-up
workers, plus any thread that takes the synchronous `decode_delta` path.

`_launch_async` (`streaming_vocoder.py:1150-1254`) stages the input, replays a graph when one
matches and otherwise calls `self._decoder.chunked_decode(gpu_input)`, extracts the deltas, copies
them into the pinned slot and records the completion event. `resolve`
(`streaming_vocoder.py:211-241`) waits on that event, clones the pinned views into owned CPU
tensors, and releases the slot. When either the event or the stream cannot be synchronized, the
buffers are moved to the module-level `_CONTEXT_FATAL_RETAINED` list and kept for the life of the
process (`streaming_vocoder.py:178-183, 249-274, 1229-1250`), and CUDA decoding is disabled.

Activation scaling:

- **Streaming.** The decoder input is `[B, Q, window]` with `window` bounded by
  `stream_left_context_frames + fresh_frames`, that is at most 24 frames with the shipped defaults.
  The window does not grow with utterance length, because `window_start` advances with
  `emitted_generated_frames` and older chunks are pruned (`streaming_vocoder.py:1047-1057`). So the
  per-decode device working set scales with batch size and with the fixed window, and the emitted
  waveform per row is `fresh_frames * samples_per_frame` samples. The retained per-stream state is
  the un-pruned code chunks, at most about `left_context` frames of `[T, Q]` int64 plus the current
  arrivals (`_Qwen3TTSStreamState`, `streaming_vocoder.py:93-111`).
- **Non-streaming.** `_vocode_payloads` decodes the entire code sequence in one call, so the
  decoder working set scales with batch size times utterance frame count, and the output waveform
  with batch size times frames times `samples_per_frame`. There is no windowing on this path.

The stateful incremental decoder (`sglang_omni/models/qwen3_tts/incremental_codec.py`) is a third
path that keeps per-stream convolution histories, transposed-convolution overlaps and transformer
KV in `Qwen3TTSIncrementalCodecState` (`incremental_codec.py:13-38`), cloned on every decode step
(`streaming_vocoder.py:957`). It is off by default: `enable_stateful_codec_decoder=False`
(`stages.py:240`), and enabling it also disables the decode graphs
(`streaming_vocoder.py:574-578, 587-591`) and async decode (`streaming_vocoder.py:614-617`).

---

## 5. The engine process beyond the KV pool

### 5.1 Predictor CUDA graphs

`_PredictorDecodeGraph` (`sglang_model.py:111-173`) is one graph per
`(bucket_size, signature)`, where the signature pins the host branches of the sampling path. Its
constructor allocates exactly three persistent device buffers per graph
(`sglang_model.py:134-140`):

- `layer0_codes`, `[bucket, 1]` int64,
- `talker_hidden`, `[bucket, 1, hidden_size]` in the hidden dtype,
- `semantic_positions`, `[bucket]` int64.

`result_codes` and `summed_embeddings` are set during capture from
`_code_predictor_forward_incremental`, which returns views of the model's persistent
`_output_codes` and `_output_embeds` buffers (`sglang_model.py:1470-1471`), so they are not new
allocations. Everything else the captured region touches lives in one shared graph pool:
`_predictor_graph_memory_pool` (`sglang_model.py:1241-1246`) with the note that private per-graph
pools would retain intermediates per key and scale with key diversity.

`replay` (`sglang_model.py:145-173`) copies the live rows into the static buffers, zeroes the tail
rows when the live batch is below the bucket, replays, and returns row-limited views of the shared
output buffers.

`capture_predictor_graphs` (`sglang_model.py:1258-1307`) is called from
`Qwen3TtsEngineBuilder.setup_model_resources` (`engine_builder.py:150-169`) after SGLang's own
graphs, with the note at `engine_builder.py:160-161` that the bucket warmups also build cuDNN's
attention plans. It resolves the subtalker sampling values from the merged generate kwargs
(`resolve_subtalker_sampling`, `request_builders.py:112-120`, code fallbacks
`do_sample=True, temperature=0.9, top_p=1.0, top_k=50`). When sampling is on it captures two
signatures, with and without argmax rows mixed in, and iterates buckets in descending order so the
smaller ones reuse the pool of the larger ones. The mixed signature skips bucket 1
(`sglang_model.py:1291-1292`).

The bucket ladder is `_normalize_predictor_graph_batch_sizes`
(`sglang_model.py:1151-1175`), which takes SGLang's resolved decode `cuda_graph_bs` and keeps
entries in `[1, max_batch_size]`, appending `max_batch_size` if the top falls short. With
`cuda_graph_bs = [1, 2, 4, 8, 12, 16, 24, 32]` from `cuda_graph_max_bs=32` and
`max_batch_size = max_running_requests = 16`, the predictor ladder is `(1, 2, 4, 8, 12, 16)`. So
the startup capture is 6 graphs for the non-mixed signature plus 5 for the mixed one, 11 total,
sharing one graph pool.

Beyond the startup set, `_predictor_forward_graphed` (`sglang_model.py:1379-1448`) captures lazily
on demand, capped at `_PREDICTOR_GRAPH_MAX_LAZY_KEYS = 32` keys beyond the startup count, after
which uncached keys fall back to eager. Capture failures disable that key, and after
`_PREDICTOR_GRAPH_MAX_FAILURES = 8` failures predictor graphs are disabled entirely.
`_resolve_predictor_graph_enabled` (`sglang_model.py:1248-1256`) also disables them when
`disable_cuda_graph` is set or `tp_size != 1`, with the note that capture under TP would record
collectives and the graphed chain is only validated single rank.

### 5.2 Persistent model buffers sized by max_running_requests

All of these are allocated once at model construction with `max_batch_size = server_args.max_running_requests`,
which is 16 for this profile.

In `Qwen3TTSTalkerTextModel.__init__` (`sglang_model.py:255-273`):
`_feedback_buffer` `[16, hidden]`, `_feedback_mask` `[16]` bool, and
`_decode_feedback_embedding`, an `nn.Embedding(16, hidden)` whose weight is the staging target for
decode inputs.

In `Qwen3TTSTalker.__init__` (`sglang_model.py:886-956`): `_predictor_positions`
`[num_code_groups + 1]`, `_predictor_position_rows` `[predictor_len, 16]`,
`_predictor_k_cache` `[n_predictor_layers, 16, kv_heads, predictor_len, head_dim]` in the codec
embedding dtype, `_predictor_v_cache` the same shape, `_sampled_token_ids` `[16]`,
`_output_codes` `[16, num_code_groups]` int64, `_output_embeds` `[16, hidden]`,
`_predictor_embedding_buffer` `[16, hidden]`, and the per-row sampling staging tensors
`_sub_temperature_tensor`, `_sub_top_p_tensor`, `_sub_top_k_tensor`,
`_semantic_sampling_seed_tensor`, `_sub_sampling_seed_tensor`, `_sub_do_sample_tensor`, each `[16]`.

`prepare_decode_buffers` (`sglang_model.py:978-1084`) restages those per-row values only when the
batch composition changed, keyed on `(request_id, per-data epoch)`.

### 5.3 Decode input history clones

`Qwen3TTSModelRunner._write_feedback_buffers`
(`sglang_omni/models/qwen3_tts/model_runner.py:288-348`) runs before every decode forward. It builds
each row as the sum of the popped feedback embedding and the next text embedding, stacks them
directly into `weight[:batch_size]` of `_decode_feedback_embedding`, and then does

```
history = target.detach().clone()          # model_runner.py:342
```

with the note at `model_runner.py:340-341` that the history outlives the buffer and a retracted
request replays it in its re-prefill. Each row of that one `[batch_size, hidden]` clone is appended
to the owning request's history list by `_append_decode_input_history`
(`model_runner.py:343-346`, implementation at
`sglang_omni/models/qwen3_omni/talker_model_runner.py:431-435`), which stores
`row.detach()`. Because those rows are views of a single clone, one surviving row keeps the whole
`[batch_size, hidden]` block alive. So over a request's decode run the process retains one
`batch_size * hidden * itemsize` block per decode step, for as long as any request from that step is
still live.

`_compact_decode_input_history`
(`sglang_omni/scheduling/omni_scheduler.py:87-94`) is the only thing that breaks that sharing. Its
docstring states that a decode input row is a view of the batch snapshot it was written in, so a
request that leaves the running batch would keep every snapshot of its run alive while it waits, and
one copy gives it storage of its own. It is called from exactly one place,
`_add_request_to_queue` when the request is retracted (`omni_scheduler.py:2197-2200`). On the normal
path the history is dropped at terminal handling, `data.decode_input_embeds = None`
(`omni_scheduler.py:1677`).

`post_process_outputs` (`model_runner.py:239-262`) adds two more per-step clones,
`self.model._output_codes[:bs].detach().clone()` and
`self.model._output_embeds[:bs].detach().clone()`, with the note at `model_runner.py:250-251` that
per-row clones were a c32 decode-loop hot spot and rows must stay views of a snapshot rather than of
the reused graph buffers. Rows of the codes snapshot are appended to
`sched_req.data.output_codes`, which accumulates for the whole request, so the same sharing
argument applies: the `[bs, num_code_groups]` int64 block stays alive while any of its rows do.
Rows of the embeds snapshot go into `pending_feedback_queue`, which
`_write_feedback_buffers` pops each step (`model_runner.py:325`), so that queue is bounded.

`_decode_row_ids` (`model_runner.py:350-364`) keeps one cached `arange` of at least 64 elements.

---

## ASCII diagram

Startup, shipped profile, one GPU. All three stages live in one OS process.

```
                        host: MultiProcessPipelineRunner (mp_runner.py:506)
                                        |
                                        | spawn 1 process (one process-plan group "pipeline")
                                        v
+=============================================================================================+
| OS process "process-pipeline"   (stage_workers.py:403 stage_process_main)                   |
|                                                                                             |
|  construction order = config order (stage_workers.py:493-500), serialized per GPU by        |
|  gpu_startup_lock (stage_workers.py:883)                                                    |
|                                                                                             |
|  [1] preprocessing        gpu_id = None -> set_device NOT called                             |
|      ThreadedSimpleScheduler(max_concurrency=8)          stages.py:129-136                   |
|      device allocation at startup: NONE                                                      |
|                                                                                             |
|  [2] tts_engine           gpu_id = 0    -> set_device(0)                                     |
|      +-- talker weights                          bootstrap.py:159                            |
|      +-- KV pool  (589142 tok, K 31.46 + V 31.46 GB)  bootstrap.py:180                       |
|      +-- attention backend workspaces            bootstrap.py:181                            |
|      +-- persistent decode buffers, all [16, ...]  sglang_model.py:255-273, 886-956          |
|      +-- speech tokenizer COPY #1 (bf16)         engine_builder.py:123                       |
|      +-- ref-code batcher thread + private CUDA stream  request_builders.py:749, 889         |
|      +-- SGLang decode graphs, bs (1,2,4,8,12,16,24,32)  engine_factory.py:223               |
|      +-- predictor CUDA graphs, 11 keys, one shared pool  sglang_model.py:1258-1307          |
|                                                                                             |
|  [3] vocoder              gpu_id = 0    -> set_device(0)                                     |
|      +-- speech tokenizer COPY #2 (bf16)         stages.py:245                               |
|      +-- 3 graph holders (1 initial + 2 followup)  streaming_vocoder.py:568-594              |
|      |     16 frame counts x 4 batch sizes = 64 keys each                                    |
|      +-- 1 initial CUDA stream + 2 followup CUDA streams  streaming_vocoder.py:626-648       |
|      +-- warmup_now() captures all holders BEFORE readiness  stages.py:279                   |
+=============================================================================================+

GPU 0 budget accounting at startup
  mem_get_info free  ---------------------------------------------------------> t0
      pre_model_load_memory captured here (bootstrap.py:132)
  weights loaded
  KV pool = free_now - pre_model_load_memory*(1-0.85)      kv_cache_configurator.py:1789
          = 62.93 GiB  ->  589142 tokens at 114688 B/token
  "Memory pool end. avail mem=11.79 GB"                    kv_cache_configurator.py:294
      ^ everything below must fit in this residual:
        SGLang decode/prefill graphs, predictor graphs, vocoder tokenizer copy #2,
        vocoder decode graphs, all activations, all per-request buffers
```

Per request, streaming.

```
 HTTP payload
     |
     v
 [preprocessing thread pool, up to 8 concurrent]                  stages.py:129
     |  _prepare_qwen3_tts_base_request                           request_builders.py:1083
     |    cache lookup (host, 64 MiB adhoc / 512 MiB speaker)
     |    miss -> hook.encode_one                                 request_builders.py:923
     |        submit waveform ------------------> [ref-code batcher thread]
     |        extract_speaker_embedding (GPU,          |  drain <=8, wait 2 ms
     |          default stream)                       |  speech_tokenizer.encode(list)
     |        wait future (<=130 s) <-----------------+  on PRIVATE cuda stream
     |    build_voice_clone_inputs -> GPU [1, L, hidden] bf16
     |    build_embedding_cache_key_ids -> HOST float32 [L, hidden] copy
     |    _build_qwen3_tts_pad_embed -> one GPU projection forward
     |  store in module dict _PREPARED_REQUESTS, emit marker payload
     v
 [tts_engine, OmniScheduler]                                      engine_factory.py:419
     |  prefill: input_embeds cat of per-request slices           model_runner.py:366-397
     |  per decode step:
     |    _write_feedback_buffers -> stack into _decode_feedback_embedding.weight[:bs]
     |                            -> history = clone [bs, hidden]  model_runner.py:342
     |                               one row per request appended, block retained
     |    forward -> codec_head -> sample
     |    code_predictor_forward -> predictor graph replay (bucket >= bs)
     |                              writes into _output_codes / _output_embeds
     |    post_process_outputs -> clone [bs, groups] + [bs, hidden]  model_runner.py:253-254
     |                            code rows appended to data.output_codes (unbounded)
     |  stream_to=["vocoder"] -> in-process handoff (same_process_targets)
     v
 [vocoder]                                                        streaming_vocoder.py:399
     |  ingest chunk, should_decode at ramp thresholds 1,2,4 then stride 8
     |  first decode  -> initial queue  -> 1 thread, batch <=32, wait 2 ms
     |  later decodes -> priority queue -> 2 threads, batch <=8, wait 1 ms
     |  group plans by exact decoder_input shape                   streaming_vocoder.py:1623
     |  window = [max(0, emitted-16), end]  -> <=24 frames         streaming_vocoder.py:1048
     |  graph replay if (frames, batch) captured, else chunked_decode
     |  deltas -> pinned slot -> event -> clone to CPU             streaming_vocoder.py:1209
     v
 audio chunk out
```

---

## Constants that shape memory

| Name | Location | Value | What it protects, per its comment |
| --- | --- | --- | --- |
| `mem_fraction_static` | `sglang_omni/models/qwen3_tts/engine_builder.py:91` | 0.85 | No comment at the definition. Upstream defines the complement as runtime slack (`kv_cache_configurator.py:1765-1767`). |
| `max_running_requests` | `engine_builder.py:84` | 16 | No comment. Sizes every per-row model buffer (`sglang_model.py:255, 887`). |
| `max_queued_requests` | `engine_builder.py:85` | 16 | No comment. Feeds the coordinator in-flight cap (`mp_runner.py:66-78`). |
| `cuda_graph_max_bs` | `engine_builder.py:86` | 32 | No comment. Drives the decode graph ladder and, through it, the predictor ladder. |
| `torch_compile_max_bs` | `engine_builder.py:87` | 32 | No comment. Inert here, `enable_torch_compile` is False. |
| `max_prefill_tokens` | `engine_builder.py:92` | 8192 | No comment. |
| `context_length` | `engine_builder.py:41` | 8192 | No comment. |
| `disable_overlap_schedule` | `engine_builder.py:89` | True | No comment. |
| `QWEN3_TTS_PREFILL_CUDA_GRAPH_BS` | `engine_builder.py:29-36` | `(1,) + build_default_prefill_cuda_graph_bs(512)` | Commented: a 1-token prefill lands in bucket 4 and misses, 40.6 percent of measured prefills are exactly one token. Applied to CustomVoice only. |
| `request_build_max_workers` | `engine_builder.py:191` | 4 | No comment. |
| `request_build_max_pending` | `engine_builder.py:192` | 16 | No comment. |
| `preprocessing max_concurrency` | `stages.py:113`, note at `stages.py:125-128` | 8 | Commented: a serial executor would show the speech-tokenizer batcher batches of one, the default matches the batcher's `max_batch_size`. |
| `_Qwen3TTSRefCodeBatcher.max_batch_size` | `request_builders.py:743` | 8 | No comment at the definition. |
| `_Qwen3TTSRefCodeBatcher.max_batch_wait_ms` | `request_builders.py:744` | 2.0 | No comment at the definition. |
| ref encode future timeout | `request_builders.py:764`, `951` | 130.0 s | No comment. |
| `ReferenceEncodeService max_items` | `request_builders.py:1030` | 256 | No comment at the call site. |
| `ReferenceEncodeService max_bytes` | `request_builders.py:1031` | 64 MiB | No comment at the call site. Host-resident bound, entries are CPU tensors. |
| `ReferenceEncodeService timeout_s` | `request_builders.py:1032` | 130.0 | No comment at the call site. |
| `DEFAULT_SPEAKER_CACHE_BYTES` | `sglang_omni/scheduling/speaker_cache.py:15` | 512 MiB | No comment. Bounds the process-wide uploaded-voice artifact cache. |
| `QWEN3_TTS_DEFAULT_MAX_NEW_TOKENS` | `request_builders.py:43` | 2048 | No comment. Bounds decode steps, hence the retained history. |
| `vocoder max_batch_size` | `stages.py:223` | 8 | No comment. Non-streaming whole-utterance batch. |
| `vocoder max_batch_wait_ms` | `stages.py:224` | 2 | No comment. |
| `initial_max_batch_size` | `stages.py:232` | 32 | No comment at the definition. |
| `initial_batch_wait_ms` | `stages.py:233` | 2 | No comment at the definition. |
| `followup_max_batch_size` | `stages.py:234` | 8 | No comment at the definition. |
| `followup_batch_wait_ms` | `stages.py:235` | 1 | No comment at the definition. |
| `followup_worker_count` | `stages.py:236` | 2 | Commented at `streaming_vocoder.py:671-675`: each worker owns its graphs and stream so two threads cannot race on the static buffers. |
| `initial_cuda_graph` / `followup_cuda_graph` | `stages.py:236, 238` | True | No comment at the definition. |
| `enable_stateful_codec_decoder` | `stages.py:240` | False | No comment at the definition. When True it disables both decode graphs and async decode. |
| `suppress_bootstrap_max_streams` | `stages.py:242`, note at `streaming_vocoder.py:524-528` | 24 | Commented: the extra per-stream startup work is what the pipeline cannot spare near capacity, measured a win to 10 RPS and a regression at 20 RPS. |
| `DEFAULT_QWEN3_TTS_STREAM_STRIDE` | `streaming_vocoder.py:34` | 16 | No comment. |
| `DEFAULT_QWEN3_TTS_STREAM_FOLLOWUP_STRIDE` | `streaming_vocoder.py:35` | 8 | No comment. Sets the top of the captured fresh-frame span. |
| `DEFAULT_QWEN3_TTS_STREAM_INITIAL_FOLLOWUP_STRIDE` | `streaming_vocoder.py:36` | 8 | No comment. |
| `DEFAULT_QWEN3_TTS_INITIAL_CHUNK_FRAMES` | `streaming_vocoder.py:37` | 8 | No comment. Legacy branch only. |
| `DEFAULT_QWEN3_TTS_STREAM_CHUNK_RAMP` | `streaming_vocoder.py:38-41` | `(1, 2, 4)` | Commented: a one-frame first chunk emits audio after a single AR step instead of eight, and the ramp restores a full playback cushion within four chunks. |
| `DEFAULT_QWEN3_TTS_LEFT_CONTEXT_FRAMES` | `streaming_vocoder.py:42` | 16 | No comment. This is what bounds streaming decode activation independently of utterance length. |
| `_QWEN3_TTS_CODEBOOK_SIZE` | `streaming_vocoder.py:43` | 2048 | No comment at the definition. Used to clamp codes so an embedding lookup cannot device-assert (`streaming_vocoder.py:1073-1081`). |
| `_BOOTSTRAP_SILENCE_MAX_RMS` | `streaming_vocoder.py:44-50` | 1e-3 | Commented: fail-closed acoustic guard, -60 dBFS, calibrated against a corpus whose loudest first frame measured -103.7 dBFS RMS. |
| `_BOOTSTRAP_SILENCE_MAX_PEAK` | `streaming_vocoder.py:51` | 3.2e-3 | Same comment block, -50 dBFS. |
| `_Qwen3TTSInitialDecodeGraphs.batch_sizes` default | `streaming_vocoder.py:317` | `(1, 2, 4, 8)` | No comment at the definition. |
| decode graph warmup passes | `streaming_vocoder.py:349` | 2 | No comment. |
| `QTTS_PREDICTOR_GRAPH_ENV` | `sglang_model.py:52` | `SGLANG_OMNI_QTTS_PREDICTOR_GRAPH` | No comment. Default on (`sglang_model.py:66-68`). |
| `_PREDICTOR_GRAPH_MAX_LAZY_KEYS` | `sglang_model.py:53` | 32 | No comment at the definition. Caps lazily captured keys beyond the startup set before falling back to eager. |
| `_PREDICTOR_GRAPH_MAX_FAILURES` | `sglang_model.py:54` | 8 | No comment at the definition. Disables predictor graphs entirely after this many capture failures. |
| `_PREDICTOR_GRAPH_WARMUP_PASSES` | `sglang_model.py:55` | 2 | No comment. |
| `_PREDICTOR_TOP_K_LADDER` | `sglang_model.py:56-58` | `(4, 8, 16, 32, 50, 64, 128, 256, 512, 1024)` | Commented: 50 is on the ladder because it is the family checkpoint default, keeping the dominant signature's kernel width unchanged. Quantization shares graph keys across request top_k values. |
| `_row_ids_cache` floor | `model_runner.py:359` | `max(batch_size, 64)` | No comment at the definition. |
| `StreamQueue max_pending` | `stage_workers.py:800` | 4096 | No comment at the definition. |
| `CommConfig.slot_size_mb` | `sglang_omni/config/schema.py:89` | 512 | No comment at the field. |
| `CommConfig.credits` | `schema.py:90` | 2 | No comment at the field. |
| `CommConfig.cuda_ipc_slot_size_kb` | `schema.py:91` | 64 | No comment at the field. |
| `PlacementConfig.max_total_gpu_memory_fraction_per_gpu` | `schema.py:229` | 1.0 | No comment at the field. Checked in `placement.py:107-115`. |
| `MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO` | `/Users/ratish/sglang/.../kv_cache_configurator.py:155` | 3 | Commented at lines 150-154. Not on this model's path, Qwen3-TTS is not mambaish. |
| default `mem_fraction_static` floor when unset | `/Users/ratish/sglang/.../server_args.py:4969-4975` | 512 MB plus 1.5 MB per activation token, floor 10 GB above 60 GB of VRAM | Commented as constant metadata plus activation slack. Not exercised here, the builder sets the fraction explicitly. |

---

## Unverified

1. **The `qwen_tts` package.** It is not importable from the default `python3` and no copy exists on
   this machine (`find / -maxdepth 8 -name qwen_tts -type d` returned nothing). Everything reached
   through it is therefore unverified: `Qwen3TTSTokenizer.from_pretrained` and whether it honors the
   `dtype` argument, the internals of `speech_tokenizer.encode` and `.decode`, the identity of the
   codec architecture, `tokenizer.model.decoder`, `decoder.chunked_decode`, `decoder.total_upsample`,
   `tokenizer.get_output_sample_rate`, `Qwen3TTSSpeakerEncoder`, `Qwen3TTSModel`,
   `wrapper._normalize_audio_inputs`, `wrapper.create_voice_clone_prompt`,
   `model.build_voice_clone_inputs`, `model.extract_speaker_embedding`, and
   `wrapper._merge_generate_kwargs`.
2. **Whether the codec is Mimi.** The task calls it the Mimi encoder. Neither the omni tree nor its
   docs mention Mimi anywhere (case-insensitive grep over `sglang_omni/` and `docs/` returned
   nothing), and the package that would settle it is absent. The report calls it the speech
   tokenizer throughout.
3. **Checkpoint geometry.** `num_hidden_layers`, `hidden_size`, `num_key_value_heads`, `head_dim`,
   `v_head_dim`, `num_code_groups`, `vocab_size`, `code_predictor_config.vocab_size`,
   `num_quantizers` and `samples_per_frame` for `Qwen/Qwen3-TTS-12Hz-1.7B-Base` were not read, no
   local checkpoint is present. The 114688 bytes per token implied by the logged K and V sizes is
   arithmetic on the observed log, and the factorization `8 * (128 + 128) * 28 * 2` is one
   consistent reading of `pool_configurator.py:327-333`, not a verified fact.
4. **The generation config actually shipped with the checkpoint.**
   `_load_qwen3_tts_generate_defaults` (`stages.py:95-103`) reads `generation_config.json`, which
   was not available. The subtalker sampling values used for the startup predictor capture, and so
   whether 1 or 2 signatures are captured, depend on it. The report quotes the code fallbacks
   (`do_sample=True, top_k=50, top_p=1.0`) from `resolve_subtalker_sampling`.
5. **The GPU model and its total memory.** Inferred only from arithmetic on the two log lines. Not
   read from anything.
6. **`page_size`.** The profile never sets it and SGLang's default was not read, so the exact
   page alignment applied at `pool_configurator.py:423` is unconfirmed. It does not change the
   sizing rule.
7. **Whether the observed run actually used the shipped profile.** The placement log values match
   the shipped defaults exactly, but no launch command was provided. If a deployment passed
   `--preprocessing.process=...` or `--preprocessing.gpu`, the layout in section 1 would be two
   processes, `load_frontend=True`, and preprocessing would load its own frontend and its own
   speech tokenizer (`stages.py:139-184`).
8. **Runtime sizes of graph pools.** The bytes retained by
   `_predictor_graph_memory_pool` and by each vocoder holder's `graph_pool_handle` depend on the
   captured intermediates and were not measured. Only the explicitly named static buffers are
   quantified.
9. **`empty_cache` interaction.** `get_available_gpu_memory` empties the caching allocator before
   reading `mem_get_info` (`common.py:433-434`), but whether this happened between the vocoder's
   graph capture and any later measurement in the observed run is not determinable from the two log
   lines given.
