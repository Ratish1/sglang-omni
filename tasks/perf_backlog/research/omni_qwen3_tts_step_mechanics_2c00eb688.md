# Research report: omni Qwen3-TTS decode step mechanics at 2c00eb688 (agent report, unverified)

## Configuration that fixes the path
- Qwen3-TTS scheduler built without enable_async_decode / enable_overlap (engine_factory.py:477-505; omni_scheduler.py:188-189 defaults False) -> synchronous loop `_event_loop_normal` (omni_scheduler.py:1731-1736).
- extra_scheduler_kwargs adds only request_build_max_workers 4 (engine_builder.py:187-194).
- prefill_coalesce_requests default 0 -> get_new_batch_prefill passthrough (engine_builder.py:51, omni_scheduler.py:1342-1343).
- All three stages share one OS process and one asyncio loop (config.py:48,58,67 process="pipeline"; stage_workers.py:456-457,467,481-482).
- Non streaming gate: request_builders.py:1544-1550, `or not params.get("stream")): return []` -> no per step vocoder handoff; codes accumulate on device in data.output_codes and reach the vocoder once at finish via apply_sglang_qwen3_tts_result.

## Execution order (scheduler-tts_engine thread)
1. `_event_loop_normal` omni_scheduler.py:2288-2312.
2. `_process_admin_requests` 1864-1882 (queue drain).
3. `recv_requests` -> `_drain_local_inbox` 795-832 (tp 1).
4. `_take_deferred_request_payloads` 1250-1260 early out.
5. `process_input_requests` 834-907: four bookkeeping passes each step (`_drain_request_admission_results`, `_drain_request_build_results`, `_stage_request_build_payloads`, twice); takes `_request_admission_lock` RLock at 1046, 1067, 1130 (contended by 4 omni-request-build threads).
6. `get_next_batch_to_run` 1322-1334 -> upstream NextBatchPlan.
7. `get_new_batch_prefill` 1336-1343 passthrough.
8. `run_batch` -> `_run_batch` 1372-1410: `_emit_prefill_start_for_batch`, `_stamp_batch_launch`, `_build_sched_output`, `_model_runner.execute`, `_emit_prefill_end_for_batch`, `_emit_stream_output`, `_make_batch_result`.
9. `_emit_prefill_start_for_batch` 1524-1541: per row loop every step, dict alloc + 16 set membership checks, no early out.
10. `_stamp_batch_launch` 1386-1392.
11. `_build_sched_output` 1412-1421: per row, 16 SchedulerRequest dataclasses per step.
12. `ModelRunner.execute` model_runner/base.py:229-281.
13. `_execution_context` qwen3_tts/model_runner.py:28-39 -> base.py:193-202 -> sglang_execution.py:79-98: `resolve_forward_inputs` (device gather, overlap_utils.py:106-107) + `sampling_info.copy_for_forward()` (runs update_penalties).
14. `_build_forward_batch` base.py:385-423: set_device; capture_hidden_mode None (output_processor._capture_hidden False, engine_factory.py:241-243).
15. `_prepare_and_forward` base.py:425-478.
16. `before_decode` qwen3_tts/model_runner.py:59-70: `prepare_decode_buffers` (sglang_model.py:977-1083; identity cache returns at 1000-1001 on unchanged rids; on miss two 16 row loops + six torch.tensor H2D at 1063-1082) and `_write_feedback_buffers` (model_runner.py:288-348; three 16 row loops; `_peek_next_decode_inputs` talker_model_runner.py:453-471, `PendingTextTensorQueue` head O(1) pending_text_queue.py:88-91, `_decode_row` 437-451 view + assert no kernel, `_pop_next_decode_inputs` 473-477; device: 2 stack + add_ + clone + copy_; `_append_decode_input_history` 433-435 appends a row view per request per step to data.decode_input_embeds, never trimmed in decode, read only by `_generated_prefill_slice` 315-348 after retract).
17. `custom_decode_forward` base.py:640-648 None -> `tp_worker.forward_batch_generation`.
18. `ModelWorker.forward_batch_generation` model_worker.py:263-306 -> upstream `model_runner.forward` (graph decision upstream); `_record_prefill_cuda_graph_usage` returns for decode 314-316.
19. `_sample_next_token_ids` qwen3_tts/model_runner.py:102-115: `_install_semantic_sampling_seeds` 200-208 (slice); base.py:766-800: `_apply_repetition_penalty` no-op override 121-128 (base loop 925-945 skipped); `_apply_codec_suppress_tokens` 179-198 two device fills; `_install_sampling_seeds` base.py:802-832 short circuit 814-816; `wants_rollout_logprob = any(...)` base.py:776 16 row generator; upstream sample.
20. `post_decode` -> `_collect_codes` qwen3_tts/model_runner.py:81-88, 210-237: `_sample_positions` 264-286; `code_predictor_forward` sglang_model.py:1126-1148; `_stage_token_ids` base.py:121-134 async D2H into pinned ping pong pair (136-180) + Event.record.
21. `_ensure_next_token_ids` base.py:554-575 no-op.
22. `_publish_next_tokens` base.py:591-603 -> sglang_execution.py:100-122 `future_map.stash` device scatter; `batch.input_ids = None`.
23. exit `_execution_context` sglang_execution.py:96-98 restores sampling_info.
24. `_finalize` base.py:507-552 outside the context:
   - 24a `_resolve_host_token_ids` base.py:520 -> 182-187 `event.synchronize()`: THE omni device sync of the iteration.
   - 24b `SGLangOutputProcessor.process` base.py:524-526 -> scheduling/sglang_backend/output_processor.py:29-62: `ids.tolist()` on the pinned tensor (host only), 16 RequestOutput allocations 53-61.
   - 24c `post_process_outputs` qwen3_tts/model_runner.py:239-262: two D2D clones, 16 row loop appending views to data.output_codes and data.pending_feedback_queue (both unbounded in frames), `int(req_output.data)` host only.
   - 24d generation_steps loop base.py:529-541 (16 rows) + `on_generation_steps_advanced` base.py:499-505 per row no-op; `finalize_skip_rids` base.py:480-491 empty.
   - 24e `req_ids` / `req_id_to_index` base.py:542-543 two comprehensions.
25. `_emit_prefill_end_for_batch` omni_scheduler.py:1543-1572 early out 1556 in steady state.
26. `_emit_stream_output` 1423-1440: 16 row loop every step; builder request_builders.py:1538-1600 returns [] at 1550 in non streaming; streaming would `torch.cat` ref codes on first chunk (1580) and `codes.detach().to(long)` (1590), no sync.
27. `_make_batch_result` 1465-1481: swaps in `mr_output.host_token_ids` so upstream `.tolist()` is host only (note 1471-1473).
28. `process_batch_result` 1379-1384: upstream then omni 16 row loop (`skip_radix_cache_insert`). Upstream `batch_result_processor.py:933-937` `.tolist()` on the pinned tensor.
29. `OmniScheduler.stream_output` 1584-1693 injected as upstream output_streamer (665-670, called at batch_result_processor.py:906): 16 row loop, `req.finished()` check, body only for finishers, takes `_request_admission_lock` (1599-1618).
30. `last_batch = batch` 2310.

## (1) Row scaling actions (16 distinct passes per step on the omni side)
omni_scheduler.py:1530-1541; 1417-1420; sglang_model.py:987-999 (rid scan); 1011-1043 (miss only); qwen3_tts/model_runner.py:310-325; 337-338 (sparse only); 343-346; base.py:776; output_processor.py:53-61; qwen3_tts/model_runner.py:255-262; base.py:530-539; base.py:504-505; base.py:542-543; omni_scheduler.py:1432-1440; 1382-1384; 1591-1594.

## (2) Omni device sync per iteration
Exactly one: base.py:185 `event.synchronize()` in `_resolve_host_token_ids` (from `_finalize` base.py:520). Routed around: output_processor.py:38 tolist on pinned; upstream batch_result_processor.py:934 on pinned (omni_scheduler.py:1474-1476); model_runner.py:257 int() on host list; `_stage_token_ids` base.py:130-132 non_blocking + event.
Finish path (per finished request, scheduler thread): `apply_sglang_qwen3_tts_result` request_builders.py:1491-1522, 1535-1536: `torch.stack(data.output_codes, dim=0).to(long)` then `torch.cat([...]).cpu()`: blocking pageable D2H of the whole request's codes inside the decode loop, scheduler thread is the waiter; result enqueued omni_scheduler.py:1687-1693.

## (3) Threads in the pipeline process
main/asyncio (stage_workers.py:481-482; `_drain_outbox_external` runtime.py:1096-1147); scheduler-preprocessing (runtime.py:241-245; threaded_simple_scheduler.py:97-123); preprocessing pool x8 (threaded_simple_scheduler.py:89, max_concurrency 8 stages.py:108,129-136) GPU work; scheduler-tts_engine; omni-request-build x4 (omni_scheduler.py:238-243; engine_builder.py:190); qwen3-tts-ref-code (request_builders.py:751-756, own stream, voice clone only); scheduler-vocoder (streaming_simple_scheduler.py:124-145; non streaming runs `_vocode_payloads` inline, GPU work); qwen3-tts-vocoder-initial (streaming_vocoder.py:713-717,728, own stream; streaming only); qwen3-tts-vocoder-followup-0/1 (718-730, followup_worker_count 2 stages.py:235, one stream + graph holder each 1668-1678; streaming only); asyncio default executor threads (runtime.py:1102,1154).
Memory: tts_engine mem_fraction_static 0.85 (engine_builder.py:91) on gpu 0; vocoder gpu 0 (config.py:63,71), no gpu_memory_fraction, no total_reserve_bytes (config.py:47-76), so `set_per_process_memory_fraction` (stage_workers.py:838) not reached; all transients share one caching allocator.

## (4) Vocoder batch and graph decision
Defaults `create_vocoder_executor` stages.py:216-243: max_batch_size 8, max_batch_wait_ms 2, initial_max_batch_size 32, initial_batch_wait_ms 2, followup_max_batch_size 8, followup_batch_wait_ms 1, followup_worker_count 2, initial_cuda_graph True, followup_cuda_graph True. Stored streaming_vocoder.py:608-611.
Non streaming: SimpleScheduler collection max 8 / 2 ms (scheduling/streaming_vocoder.py:122-130 -> streaming_simple_scheduler.py:63-64; collector 249-308) on scheduler-vocoder; `_vocode_payloads` streaming_vocoder.py:1865-1895 -> `self._tokenizer.decode([{"audio_codes": item} ...])` whole utterance, batched, NO CUDA graph, no omni chunking (deterministic mode decodes one at a time).
Streaming initial: collector 1526-1549 called at 1551-1560 with max 32 / 2 ms. Followup: `_collect_followup_batch` 1686-1705, PriorityQueue by playback_deadline_s 1517-1524, max 8 / 1 ms, serialized by `_followup_collect_lock` 1680-1681.
Graph vs eager: `_Qwen3TTSInitialDecodeGraphs.decode` 379-396 (bucket = first of `_batch_sizes` >= batch, else None -> eager `chunked_decode` at 1190-1192); batch buckets (1,2,4,8) at 573/586 ((1,) deterministic); frame buckets `_decode_graph_frame_counts` 54-90 from left_context 16 (line 42), ramp (1,2,4) (line 41), steady stride 8, bumped schedule when bootstrap suppression on (549-567); capture at build `warmup_now` stages.py:279 -> 699-704. `_group_decode_plans` 1623-1633 groups by exact input shape, so an initial batch of up to 32 same shape rows goes eager above 8.

## (5) Finish detection and final codes
stop_token_ids = [codec_eos_token_id] request_builders.py:1404, eos_token_ids 1415; upstream update_finish_state + release_kv_cache; omni skips the EOS frame model_runner.py:257. Non streaming: `stream_output` 1631-1693 under `_request_admission_lock`, `on_request_finished` base.py:692 no-op, `data.output_ids = list(req.output_ids)`, `_flush_stream_output` returns 1460-1462, `_result_adapter` = `apply_sglang_qwen3_tts_result` -> `.cpu()` blocking pageable D2H. Handoff: runtime.py:1102-1109 -> `_route_result` 1169-1229 (stream_done first 1185-1190, payload 1219-1227) -> `_send_to_stage` 1262-1294 same process -> `LocalStageDispatcher.send_payload` local_dispatch.py:36-48 -> `receive_local_payload` -> `_execute` runtime.py:964-977 -> vocoder inbox; stream_done parked in `_pending_done` streaming_vocoder.py:1791-1797, discarded streaming_simple_scheduler.py:391-393.
Streaming: builder ships CUDA codes (request_builders.py:1590-1600, target vocoder); D2H on the vocoder thread in `_Qwen3TTSDecodeHandle.resolve` -> `_wait_and_release` 243-296 (`slot.output_transfer.synchronize()` = event.synchronize, utils/cuda_staging.py:164-171; deltas cloned), staged at 1210-1212.

## UNVERIFIED
Vocoder activation bytes per batch size; ref-code thread liveness per request type; whether the talker graph replays at bs 16 with cuda_graph_max_bs 32 (upstream decision).

## Verifier corrections (all mechanics confirmed unless stated)
- `pending_feedback_queue` is NOT unbounded: one append per step (model_runner.py:262) matched by one pop per step (model_runner.py:325 -> talker_model_runner.py:475), so it holds 0 or 1 frame. Only `output_codes` (model_runner.py:260, released with the data object at finish via `_detach_request_data` / `_close_completed_request` omni_scheduler.py:2621-2630) and `decode_input_embeds` (model_runner.py:343-346, cleared at finish by `data.decode_input_embeds = None` omni_scheduler.py:1667) grow with generated frames.
- Row passes: 14 of the 16 run every steady state iteration; sglang_model.py:1011-1043 is miss only, model_runner.py:337-338 sparse path only.
- `extra_scheduler_kwargs` (engine_builder.py:187-194) also passes stream_output_builder, request_build_max_pending 16, prefill_coalesce_requests, prefill_coalesce_wait_ms.
- `process_input_requests` makes five bookkeeping calls per step (836, 837, 838, 906, 907); steady state lock sites 973, 1046 (x2), 1130 (x2); 1067 only when a build completed.
- `_apply_codec_suppress_tokens`: two fills only when codec_eos lies inside the suppressed range (195-196), else one fill (198).
- `_write_feedback_buffers` dense path at 16 rows: 5 device ops: stack out=target (330), stack text (331), add_ (331), clone (342), copy_ row ids (348). `_decode_row` launches nothing. `_decode_row_ids` cache hit after the first call.
- Vocoder worker threads (initial + 2 followup) are created on every vocoder stage start (`on_serving_start` from `StreamingVocoderBase.start`, scheduling/streaming_vocoder.py:132-137); in non streaming they idle. Non streaming batch constants: stages.py:223-224 -> models/qwen3_tts/streaming_vocoder.py:679-686 -> scheduling/streaming_vocoder.py:122-130 -> streaming_simple_scheduler.py:63-64; collector `_collect_new_request_batch` 249-308 (deadline 261); dispatch 424-426 on scheduler-vocoder.
- Finish `.cpu()` at request_builders.py:1503-1506 (stack at 1499), called inside `stream_output` at 1648.
- omni-request-build threads issue no GPU work in this same process layout (registered prepared object). Preprocessing GPU work: request_builders.py:1226-1246 `.to(feedback_buffer.device)`.
- Anchor drift: `_event_loop_normal` def 2281; vocoder gpu 0 at config.py:72, process literals 50/60/69; preprocessing max_concurrency default stages.py:109; local payload enqueue runtime.py:964-969; `_stage_deltas` call 1209.
- Engine defaults (engine_builder.py:83-93): max_running_requests 16, cuda_graph_max_bs 32, disable_overlap_schedule True, mem_fraction_static 0.85, sampling_backend pytorch.
