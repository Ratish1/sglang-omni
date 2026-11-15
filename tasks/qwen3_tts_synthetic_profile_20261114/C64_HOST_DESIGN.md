# c64 host path: per-request decode state on the device (design, 11-15)

## Evidence

The evidence is the served SeedTTS c64 cell on main `bac76771a`, unseeded py-spy `--gil` (STATE, "GIL owners by function"):
- The pipeline process's lock is held by the talker scheduler 50 %, the vocoder threads 23 %, preprocessing 21 % and routing 5 %.
- Under nsys the scheduler waits for the lock as long as it holds it (4.4 against 3.7 ms a step).
- The talker stream runs kernels about 60 % of a 13 ms step.

Of the scheduler's lock time:
- `before_decode` is 16 %;
- `emit_stream_output` 7 %;
- `post_process_outputs` and `collect_codes` about 4 %.

Each of these walks the batch row by row in Python, for state that is per request and changes only at admission or once per step on the device.

A step at c64 restages that state almost every step. Requests are about 52 frames long at 68 req/s, so the batch changes nearly every step, and `prepare_decode_buffers`' composition cache (`sglang_model.py:1261-1279`) rarely hits.

## Flow on main

```
OmniScheduler._run_batch
└─ ModelRunner.execute                               model_runner/base.py:419
   └─ prepare_and_forward                            base.py:641
      └─ Qwen3TTSModelRunner.before_decode           models/qwen3_tts/model_runner.py:99
         ├─ Qwen3TTSTalker.prepare_decode_buffers    sglang_model.py:1250
         │    per row: 6 Python reads into 6 lists     :1289-1323
         │    6 pinned torch.tensor + 6 H2D copies     :1346-1357
         └─ write_feedback_buffers                   model_runner.py:322
              per row: peek_next_decode_inputs         qwen3_omni/talker_model_runner.py:695
                       decode_row x2 (device, dtype)   talker_model_runner.py:677
                       pop_next_decode_inputs          talker_model_runner.py:719
              torch.stack x2, add_, clone              model_runner.py:369-375
              per row: append_decode_input_history     model_runner.py:376-379
   └─ post_decode -> collect_codes                   model_runner.py:229
OmniScheduler.process_batch_result
└─ post_process_outputs                              model_runner.py:263
     codes_snap, embeds_snap clones                  :279-280
     per row: output_codes.append(view), pending_feedback_queue.append(view)   :287-304
└─ emit_stream_output -> stream_output_builder       request_builders.py:~2125-2200
     per row: .to(long), metadata dict, message      :2195
```

The feedback row each request feeds back is row `i` of last step's `output_embeds`. It is a slice of a tensor that was already batched on the device, taken apart into per-request views (`post_process_outputs`) and stacked back (`write_feedback_buffers`).

## SGLang's counterpart

`SamplingBatchInfo` (sglang v0.5.23):
- builds the per-request sampling tensors once when requests join (`sampling/sampling_batch_info.py:127` `from_schedule_batch`);
- indexes them on the device when requests leave (`:466` `filter_batch`, `value[keep_indices_device]`);
- concatenates when batches merge (`:550` `merge_batch`).

A decode step never rebuilds per-request tensors from Python. The KV mapping works the same way: a request owns a pool slot (`req_pool_idx`), and per-request device state is a table indexed by it.

## Design

One owner, `Qwen3TTSTalker`, holds slot tables sized by the request pool (`req_to_token.shape[0]`), indexed by `req_pool_idx`:
- **Sampling table** `[slots, 6]`: semantic seed, subtalker temperature, top_p, top_k, seed, do_sample. Written once when a request's prefill is built, in one H2D copy per prefill batch.
- **Feedback table** `[slots, hidden]`: the request's next feedback embedding. The predictor's output step scatters `output_embeds[:B]` into it by the batch's `req_pool_indices` (on the device, in `forward_batch`).

Each decode step:
- the sampling buffers become `table[req_pool_indices]`, one gather;
- the feedback buffer becomes `feedback_table[req_pool_indices] + text_rows`.

That leaves no Python per row for either.

Text rows:
- When every row's next text input is the pad embedding (`non_streaming_mode` prompts, and every streamed prompt past its trailing text), the add is a broadcast of one row.
- Requests still consuming trailing text keep the per-row path for that row only, until their text is exhausted.

History (`decode_input_embeds`) exists for retraction replay (`talker_model_runner.py:523` `generated_prefill_slice`). It is kept as a per-step device tensor plus each request's row index, appended once per step, not once per row.

Out of scope here, measured separately:
- the stream message per row (`stream_output_builder`, `OmniScheduler.emit_stream_output`), which `OmniScheduler` shares with other models;
- the vocoder's per-stream planning;
- preprocessing.

## Correctness gates

- Byte-identical seeded output against main: the same embeddings are added in the same order, so the feedback buffer holds the same bits. Seeded non-streaming SeedTTS EN, c1 and c16, every request's PCM compared.
- Unit tests:
  - the slot tables are written at admission and read by slot;
  - a request that leaves and a new one in its slot do not see each other's values;
  - retraction replays the same decode inputs.
- Qwen3-Omni shares `QwenTalkerModelRunner`'s helpers. Its path stays main's exact path: the slot tables live in the Qwen3-TTS talker only.

## Measurement

- Served unseeded SeedTTS Base and CustomVoice streaming at c1, c16 and c64, main and branch in alternating order (main, branch, main, branch), read against the same-card drift band.
- py-spy `--gil` at c64 to confirm `before_decode` and `post_process_outputs` leave the scheduler's lock time.
