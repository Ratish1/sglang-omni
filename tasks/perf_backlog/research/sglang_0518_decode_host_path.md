# Research report: sglang v0.5.18 decode iteration host path (agent report, unverified)

Scenario: 16 running decode requests, no spec, tp=1, CUDA graphs on, overlap disabled, no logprobs, no grammar, page_size 1.

Layout at the tag: result processing is `managers/scheduler_components/batch_result_processor.py`, streaming is `managers/scheduler_components/output_streamer.py`, the graph runner is `model_executor/runner/decode_cuda_graph_runner.py` + `model_executor/runner_backend/full_cuda_graph_backend.py` + `model_executor/cuda_graph_buffer_registry.py` (`execute` / `load_batch` replace `replay` / `replay_prepare`), `recv_requests` is in `managers/scheduler_components/request_receiver.py`.

Stream: the whole event loop runs inside the schedule stream (`scheduler.py:1702`); the non overlap branch of `run_batch` never enters `forward_stream_ctx`.

## Execution order

1. Loop top, shutdown guard `scheduler.py:1721-1723`.
2. `recv_requests` `scheduler.py:1726` -> `request_receiver.py:75-102`, drain loop over the zmq socket with NOBLOCK, capped by `max_recv_per_poll` (`104-151`). No device work.
3. `process_input_requests` `scheduler.py:1876-1907`, loop over received requests, dispatcher. `flush_wrapper.check_pending()` at `1906`.
4. `get_next_batch_to_run` `scheduler.py:3015`:
   - 4b `_abort_on_waiting_timeout` `2816-2841` loop over waiting queue, gated by env `SGLANG_REQ_WAITING_TIMEOUT`.
   - 4c `_abort_on_running_timeout` `1628-1641` loop over rows, gated by env `SGLANG_REQ_RUNNING_TIMEOUT`.
   - 4d last batch merge `3065-3086`, only after a prefill (`filter_batch` `3082`, `merge_batch` `3086`).
   - 4f `get_new_batch_prefill` `3097` -> `_get_new_batch_prefill_raw` `3184`; steady decode exits at `3202-3206` when `batch_is_full or len(waiting_queue) == 0`.
   - 4g decode branch `3125-3128`: `update_running_batch`.
   - 4i `set_schedule_time_batch` `req_time_stats.py:1230`, loops rows only when tracing.
5. `update_running_batch` `scheduler.py:3481-3562`:
   - 5a `batch.filter_batch()` `schedule_batch.py:3125`; list comp over rows `3135-3140`; fast return when nothing finished `3150-3152`; otherwise pinned H2D of keep indices `3154-3158`, `index_select` per field `3166-3180`, `sampling_info.filter_batch` `sampling_batch_info.py:302-326` (penalizer filter + gather per sampling tensor).
   - 5b `check_decode_mem` `schedule_batch.py:2809-2814` -> `new_tokens_required_next_decode` `2782-2796`, generator over rows; `available_size()` is `len()` of tensors (`allocator/token.py:51-53`), no sync.
   - 5c retract path `3492-3550` (not steady state).
   - 5e `prepare_for_decode` `3561`.
6. `prepare_for_decode` `schedule_batch.py:3038-3122`: `cumulate_penalty_output_tokens` `3017-3036` only with penalizers (per row list comp + pinned H2D); `alloc_for_decode` `3069`; per row loop `3071-3074` (`decode_batch_idx`, `kv_committed_len`); `seq_lens + 1`, `seq_lens_cpu + 1`, `orig_seq_lens + 1` `3078-3082`; `seq_lens_sum = None`. `input_ids` not set here (relayed through FutureMap, `3061-3062`).
7. `alloc_for_decode` `mem_cache/allocation.py:512-565`: page 1 -> `alloc_token_slots` (`151-171`) -> slice of `free_pages` (`allocator/token.py:55-64`), no kernel; `locs = seq_lens_gpu.clone()` and `req_to_token_pool.write(...)` `544-552`; per row loop `561-562`.
8. `run_batch` `scheduler.py:3626`: bookkeeping `3632-3636`; `_profile_batch_predicate` `3641`; non overlap branch `3780-3794`: `resolve_forward_inputs`, `forward_batch_generation`, `_relay_forward_payload`, `batch.input_ids = None`. Never calls `copy_to_cpu`, never sets `copy_done`.
9. `resolve_forward_inputs` `overlap_utils.py:85-118`: `batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]`, one device gather, no host.
10. `TpModelWorker.forward_batch_generation` `tp_worker.py:574-668`: `ForwardBatch.init_new` `589-594`; `model_runner.forward` `609-612`; delay sample gate is overlap only (`629-645`); inline `model_runner.sample` `647-652`.
11. `ForwardBatch.init_new` `forward_batch_info.py:704-944`: `seq_lens_sum = int(seq_lens_cpu.sum())` `756-757` (CPU tensor, no sync); per row comps `lora_ids`, `rids` `800-801`; `token_type_ids` comp `985-987`; decode positions `clamp_position(batch.seq_lens)` `869-872` one kernel.
12. `ModelRunner.forward` `model_runner.py:1497-1590`; `no_copy_to_cpu = not get_schedule().disable_overlap_schedule` `1558`.
13. `_forward_raw` `1641-1737`: `can_run_graph` `1658-1662`; `decode_cuda_graph_runner.execute` `1672-1678`, early return.
14. `can_run_graph` `decode_cuda_graph_runner.py:648-719`: bucket check `681-696`; encoder-decoder truth read at `700-703` is a sync (not this model).
15. `execute` `1386-1452`: `load_batch` then `backend.replay`.
16. `load_batch` `1240-1379`: `needs_forward_metadata_init` true for plain decode (`forward_batch_info.py:622-641`); `_pad_to_bucket` `base_cuda_graph_runner.py:136-151`; `buffer_registry.fill_from` `1320-1327`; `build_replay_fb_view` `139-197` + `attn_backend.init_forward_metadata_out_graph(fb_view)` `1352-1363`; graph key `1374-1379`.
17. `CudaGraphBufferRegistry.fill_from` `cuda_graph_buffer_registry.py:382-472`: loops over slots (fixed ~7), padded tails reset per policy `416-422`, one `torch._foreach_copy_` per dtype pair `424-466` (input_ids, positions, out_cache_loc, req_pool_indices, seq_lens, mrope when present), `seq_lens_cpu` CPU to CPU copy (`600`), `post_fill` hooks `468-472`.
18. `init_forward_metadata_out_graph`: backend specific, UNVERIFIED cost.
19. `FullCudaGraphBackend.replay` `full_cuda_graph_backend.py:144-151`: one graph launch.
20. `ModelRunner.sample` `model_runner.py:1758-1792`: `_preprocess_logits` `1741-1756` (`update_regex_vocab_mask` returns at once without grammars `sampling_batch_info.py:240-242`; `apply_logits_bias` `283-300` applies penalizer + bias); sampler call `1777-1788`.
21. `Sampler.forward` `layers/sampler.py:98-248`: `sanitize_nan_logits` (async); greedy `130-137` argmax; sampled `191-216` (`div_` temperatures, softmax, `_sample_from_probs` `250-300`: flashinfer `top_k_top_p_sampling_from_probs` `281-286` or torch path `289-297`); returns a device tensor `248`.
22. `_relay_forward_payload` `scheduler.py:3869-3884` -> `FutureMap.stash` `overlap_utils.py:542-557`: `output_tokens_buf[indices] = ...` one device scatter.
23. `process_batch_result` `scheduler.py:3922-3956`: `flush_trace_batch` (tracing only), `publish_load_snapshot` (interval gated), `process_batch_result_decode`.
24. `process_batch_result_decode` `batch_result_processor.py:805-916`: `copy_done` branch `810-811` not taken in non overlap; `_normalize_decode_outputs` `825-830`; metrics `832`, `840-843`; `free_group_begin` `845`.
25. `_normalize_decode_outputs` `918-955`: `next_token_ids.tolist()` at `934`, THE device sync of the iteration (blocking pageable D2H of 16 ints); `[[t] for t in ids]` `938`.
26. Per request loop `847-904`: `output_ids.extend`, `set_last_decode_finish_time`, `update_finish_state` (`schedule_batch.py:1626-1666`: vocab boundary, str based finish `1542`, token based finish `1490-1512`, max_new_tokens), `_handle_finish_state_updated_req` `870`.
27. `_handle_finish_state_updated_req` `1011-1101`: finished row -> `release_kv_cache` `1096` (`mem_cache/common.py:195+`), frees deferred to the free group.
28. `stream_output` `output_streamer.py:103-113` -> `_stream_output_generation` `129-182`: four `any()` scans `136-147`, accept loop `162-171`.
29. `_GenerationStreamAccumulator.accept` `362-601`: stream interval gate `363-390` (non streaming rows emit every `DEFAULT_FORCE_STREAM_INTERVAL` tokens, else return at `392`); incremental detok bookkeeping `403-429`; `to_payload` `609-676` builds one `BatchTokenIDOutput`, `time_stats` pickled inline.
30. Send `output_streamer.py:178-182` -> `output_sender.py:12-29` -> `io_struct.py:2469-2474` `sock_send`: one zmq PUSH per iteration, msgpack encode (or pickle when `_USE_PICKLE_IPC`), non blocking.
31. `free_group_end` `907` -> `allocator/base.py:67-70`: one `torch.cat` + free for finished rows.
32. Decode metrics `909-916` -> `metrics_reporter.py:718+`; heavy branch gated by `decode_log_interval` `751-752`.
33. Tail `scheduler.py:3946-3956`: `_record_step_counters` (`all()` over rows `3964`), `log_batch_result_stats` (returns unless expert metrics), `_maybe_clear_mm_inputs` `2357-2366` per row loop every iteration, `maybe_send_health_check_signal`, `update_device_timer` (env gated).
34. `_maybe_report_active_ranks` `3822` (DP attn only).
35. End `1749-1751`: `last_batch = batch`.

## Row scaling actions (unconditional)
`filter_batch` comp `schedule_batch.py:3135-3140`; `new_tokens_required_next_decode` `2793`; `prepare_for_decode` loop `3072-3074`; `alloc_for_decode` loop `allocation.py:561-562`; `init_new` comps `forward_batch_info.py:800-801`, `985-987`; `.tolist()` + wrap `batch_result_processor.py:934, 938`; result loop `847-904` with `update_finish_state` and `_handle_finish_state_updated_req`; `output_streamer.py:136-147`, `162-171`, `accept` lists; `_maybe_clear_mm_inputs` `scheduler.py:2358-2366`; `_record_step_counters` `3964`; payload encode.
Conditional: timeouts, penalizer cumulate, grammar comps, filter_batch rebuilds when a row finished, logprobs comps, finished row KV release, tracing, retract, sampling mask, logprobs.
Not row scaling: `fill_from` slot loop (fixed list, one foreach copy per dtype pair).

## Device sync points
Unconditional, exactly one: `batch_result_processor.py:934` `next_token_ids.tolist()`.
Conditional: `copy_done.synchronize()` `810-811` (overlap or sync spec only); logprob `.tolist()` `941-952`; sampler `.cpu().tolist()` with `return_sampling_mask` `337`, `391-395`; encoder-decoder truth read `model_runner.py:1658` + `decode_cuda_graph_runner.py:700-703`; expert metrics `.item()` `metrics_reporter.py:975, 991`; `active_ranks_cpu.tolist()` `scheduler.py:3853`; `write_cache_indices` `.item()` fallback `allocation.py:88-91` extend only; attention backend `init_forward_metadata_out_graph` UNVERIFIED; sampling kernel internals UNVERIFIED.
Not syncs: `int(seq_lens_cpu.sum())`, `available_size()`, `_assert_async`, `sanitize_nan_logits`, `seq_lens_cpu` CPU copy, capture time synchronize.

## Overlap scheduler
`event_loop_overlap` `scheduler.py:1754-1821`: launch on `forward_stream` after `wait_stream(schedule_stream)` `3667-3671`; push `(batch.copy(), batch_result)` to `result_queue` `1801`; pop and process the previous iteration `1804-1806` (`1761-1764`). In `run_batch`: `seq_lens_cpu` from the future map `3663`; `_forward_isolation` `3676` (`sampling_info.copy_for_forward()`); `batch_record_buf` two iterations `3709-3719`, `record_batch_in_overlap` `3565`; publish `seq_lens + 1` `3697-3698`; `copy_done = Event()` `3721`; D2H on `copy_stream` after `wait_stream(forward_stream)` `3734-3740`; optional deferred sampling `delay_sample_func` (`tp_worker.py:629-645`, `launch_batch_sample_if_needed` `scheduler.py:3886-3920`). Result processing waits on `copy_done.synchronize()` `batch_result_processor.py:810-811`, per row skip of finished or retracted `850-856`. Flag: `ServerArgs.disable_overlap_schedule` `server_args.py:963-970`; `Scheduler.enable_overlap = not disable_overlap_schedule and not use_mlx()` `scheduler.py:433`; dispatch `4912` in `dispatch_event_loop` `4902`; `model_runner.py:1558`; forced True on MPS `server_args.py:4389`.

## Sampled ids
Sampler returns a device tensor (`sampler.py:248`), passed through `model_runner.py:1792`, `tp_worker.py:649-652`; stashed device side (`overlap_utils.py:554-556`), gathered device side next iteration (`106-107`). Host list only at `batch_result_processor.py:933-938`. Under overlap the same conversion runs on a pinned tensor from `GenerationBatchResult.copy_to_cpu` (`managers/utils.py:151`, `_async_d2h`).

## Verifier corrections (all mechanics confirmed, anchors fixed)
- `check_pending()` at scheduler.py:1907; last batch merge block 3067-3092, `merge_batch` at 3092; `get_new_batch_prefill` called at 3107 (def 3157); decode branch 3127-3133, `update_running_batch` call at 3130; `update_running_batch` 3481-3564; `prepare_for_decode` call at 3563; `set_schedule_time_batch` called at 3146.
- `alloc_for_decode` row loop allocation.py:562-563. `_profile_batch_predicate` at 3642; non overlap branch also calls `update_cache_from_scheduler` 3794 (no-op placeholder 4896-4899).
- tp_worker inline sample 649-653, delay gate 628-647. `token_type_ids` comp forward_batch_info.py:974-976 (in `_maybe_init_non_generation_fields`, called unconditionally at 807); `clamp_position` 870-872.
- `can_run_graph` `torch.all(encoder_lens > 0)` at decode_cuda_graph_runner.py:697-701, consumed at 716-721 and `model_runner.py:1658`.
- sampler call model_runner.py:1775-1787. Row loop also calls `_maybe_update_reasoning_tokens` 865 (fast return); `_handle_finish_state_updated_req` called at 869; `release_kv_cache` at 1097; `_mamba_prefix_cache_update` invoked at 1051 per row, returns at its first guard on non mamba.
- `accept` return at 391, detok block 405-429, `to_payload` 609-683 (`wrap_as_pickle(time_stats)` 624). `sock_send` uses flags=0, a blocking zmq send (blocks only at HWM).
- `_record_step_counters` `all()` at 3965. `_maybe_report_active_ranks` called 3841, def 3845, gated on DP attention and elastic EP; `active_ranks_cpu.tolist()` at 3856. `write_cache_indices` `.item()` at allocation.py:90-93.
- Overlap: `event_loop_overlap` 1754-1826; `forward_stream_ctx` 3665, `wait_stream` 3666, `resolve_forward_inputs` 3671; result queue push 1804; steady state `pop_and_process()` 1810-1812 (def 1760-1763); `seq_lens_cpu` from future map 3661; `_forward_isolation` 3673; `batch_record_buf` `[None]*2` at 1500-1501, extended 3703-3705; `record_batch_in_overlap` def 3566 called 3614.
- `FutureMap.stash` scatter overlap_utils.py:555-557. With SGLANG_IS_IN_CI the `_DEBUG_ASSERT` path adds an `_assert_async` + scatter at 108-111, no sync.
- `fill_from` decode slots from `build_decode_registry` 510-612: input_ids, positions, out_cache_loc, req_pool_indices, seq_lens, seq_lens_cpu (device cpu), mrope_positions; also registers global_num_tokens_gpu, global_num_tokens_for_logprob_gpu (and num_token_non_padded when enabled), skipped at 441-443 when the FB attribute is None. `_grouped_foreach_copy_` 47-68 buckets by (dst.dtype, src.dtype).

## Verifier extra answers
- triton_backend.py `init_forward_metadata_out_graph` 615-677, decode replay arm 667-677: `_apply_cuda_graph_metadata` 1209-1245 -> `_update_decode_kv_buffers` 449-501 -> `_fill_kv_indptr_and_indices` 429-447 (cumsum + `create_flashinfer_kv_indices_triton`), `get_num_kv_splits` 347-394 (fill_ or triton). No host read for a dense non SWA model on a standard pool; the `.item()` reads at 723 and 738 are unified pool only and bypassed when `seq_lens_sum` is set (the replay view always sets it, decode_cuda_graph_runner.py:173-177). About 2 to 3 launches.
- Default attention backend (`--attention-backend` unset): `_get_default_attn_backend` server_args.py:5870-5942 via overrides.py:2112-2126 at 5962. sm90 with CUDA >= 12.3: "fa3" (5897-5903, `is_hopper_with_cuda_12_3` utils/common.py:274-278). sm100 with CUDA >= 12.8: "trtllm_mha" (5904-5916), "fa4" when `has_asymmetric_kv`. triton only as fallback 5925.
