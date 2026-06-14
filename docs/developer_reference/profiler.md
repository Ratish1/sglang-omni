# Request-level profiler

`sglang-omni` ships two complementary profilers that share the same `run_id`
and are controlled by the same HTTP surface:

- a **request-level event recorder** that writes a JSONL stream of
  per-request milestones (admission, preprocess, encoder, prefill, first
  token / first code chunk, hops, terminal response) — used to reconstruct
  a single request's end-to-end timeline and to aggregate stage / hop costs
  across a batch;
- a **torch profiler** that produces a Chrome trace of kernel-level CPU /
  CUDA activity — used to drill into a specific window once the event
  recorder has identified where the time is going.

Most diagnostics use the event recorder. The torch profiler is opt-in for
deeper kernel investigation.

## Event model

Every instrumentation point appends a single line of JSON to a per-process
JSONL file. The shape:

```jsonc
{
  "request_id": "req-123",
  "stage": "thinker",
  "event_name": "scheduler_first_emit",
  "timestamp_ns": 1717000000123456789,
  "run_id": "demo-run",
  "pid": 42,
  "metadata": {"chunk_id": 0}
}
```

Files are written under `<event_dir>/events_<stage>_<pid>.jsonl`. Multiple
co-located stages in the same OS process share **one** JSONL file — the
filename uses the first stage to start, and the per-event `stage` field
identifies the owner. The views layer merges files from every process by
`request_id`.

### Standard event names

The recorder always attaches the active `stage` name to every event, so the
same `scheduler_prefill_start` becomes "thinker prefill start" when emitted
from the thinker process and "talker prefill start" when emitted from the
talker process. `scheduler_queue_enter` marks a built request entering the
scheduler queue; `scheduler_prefill_start` is emitted later, when the request's
first executable prefill / extend batch is selected.

| Pipeline milestone | Concrete event | Source |
|---|---|---|
| Request admission | `request_admission` | `Coordinator._submit_request` |
| Preprocessing start / end | `preprocess_start` / `preprocess_end` | model preprocessor `__call__` |
| Encoder start / end | `encoder_start` / `encoder_end` (metadata `modality`, `batch_size`) | image / audio encoder executors |
| Aggregate ready | `stage_aggregate_ready` | `Stage._on_data_ready` after `InputHandler.receive` returns a merged payload |
| Thinker prefill start | `scheduler_prefill_start` (stage = thinker) | `OmniScheduler.run_batch` |
| Thinker first token | `stage_first_stream_chunk_sent` (stage = thinker) | `Stage._send_stream_to_target` / `_send_stream_to_coordinator` |
| First stream chunk to client | `stage_first_stream_chunk_sent` (terminal stage → coordinator) | same |
| Talker request build start / end | `scheduler_request_build_start` / `_end` (stage = talker) | `OmniScheduler.process_input_requests` |
| Talker prefill start | `scheduler_prefill_start` (stage = talker) | same |
| First code chunk | `stage_first_stream_chunk_sent` (stage = talker) | `Stage._send_stream_to_target` |
| Code2Wav first audio | `code2wav_first_audio` | `Code2WavScheduler._decode_and_emit` |
| Terminal response | `terminal_response` | `Coordinator._handle_completion` |

Supporting events used for finer-grained breakdown:

| Layer | Event | Notes |
|---|---|---|
| Coordinator | `coordinator_stream_received` | Each `StreamMessage` received on the coordinator |
| Stage | `stage_input_received` | Submit or relay payload accepted (metadata `from_stage`) |
| Stage | `stage_dispatch` | Scheduler inbox put |
| Stage | `stage_complete` | Scheduler result routed onward (metadata `terminal`, `next`) |
| Stage | `stage_hop_sent` | Payload `DataReadyMessage` sent to next stage |
| Stage | `stage_stream_chunk_sent` | Each stream chunk (metadata `to_stage`, `chunk_id`, `modality`) |
| Stage | `stage_stream_chunk_received` | Each stream chunk materialized and ready for the receiver scheduler, including coordinator terminal chunks |
| AR scheduler | `scheduler_queue_enter` | Built request entered the scheduler queue |
| AR scheduler | `scheduler_first_emit` | First `stream_output_builder` emission per request |
| MOSS-TTS Local AR | `moss_tts_local_collect_frame_start` / `_end` | `_collect_frame()` boundary after the backbone forward: frame decode, radix id build, feedback staging, and journal creation. Metadata includes `batch_size`, graph/eager path, fallback reason, emitted/chunked counts. |
| MOSS-TTS Local AR | `moss_tts_local_frame_decode_start` / `_end` | Local frame decoder boundary only, either CUDA graph replay or eager fallback. Metadata includes `used_frame_graph`, `frame_decode_path`, `frame_graph_max_bs`, and `fallback_reason`. |
| MOSS-TTS Local vocoder | `moss_tts_local_vocoder_batch_start` / `_end` | Batched code-to-waveform boundary. Metadata includes `batch_size` and `decode_count`. |
| MOSS-TTS Local vocoder | `moss_tts_local_vocoder_decode_start` / `_end` | Codec `decode_audio_codes(...)` boundary, excluding result packaging and waveform CPU materialization. |

MOSS-TTS Local frame events are emitted once per request for timeline
reconstruction, but each interval describes a batch-shared operation. Their
metadata carries `shared_batch_interval=true`; use p50/p95 and path/fallback
metadata for diagnosis, and avoid interpreting summed `total_ms` as unique GPU
time across all requests.

When torch profiling is active, MOSS-TTS Local also emits scoped trace ranges
inside `_collect_frame()`:

- `moss_tts_local.before_decode`
- `moss_tts_local.before_decode.prepare_active_rows`
- `moss_tts_local.before_decode.feedback_gather_copy`
- `moss_tts_local.before_decode.input_ids_write`
- `moss_tts_local.collect_frame.setup`
- `moss_tts_local.collect_frame.pool_rows`
- `moss_tts_local.collect_frame.param_gather`
- `moss_tts_local.collect_frame.sampling_state`
- `moss_tts_local.collect_frame.path_select`
- `moss_tts_local.frame_decode.cuda_graph`
- `moss_tts_local.frame_decode.eager`
- `moss_tts_local.collect_frame.graph_output_clone`
- `moss_tts_local.collect_frame.row_build`
- `moss_tts_local.collect_frame.radix_hash`
- `moss_tts_local.collect_frame.eager_feedback_embed`
- `moss_tts_local.collect_frame.emit_filter`
- `moss_tts_local.collect_frame.feedback_write`
- `moss_tts_local.collect_frame.feedback_write.all_emit_alias`
- `moss_tts_local.collect_frame.feedback_write.emit_index_tensor`
- `moss_tts_local.collect_frame.feedback_write.emit_row_select`
- `moss_tts_local.collect_frame.feedback_write.emit_rows_select`
- `moss_tts_local.collect_frame.feedback_write.emit_steps_select`
- `moss_tts_local.collect_frame.feedback_write.emit_embeds_select`
- `moss_tts_local.collect_frame.feedback_write.sampling_step_write`
- `moss_tts_local.collect_frame.feedback_write.feedback_embed_write`
- `moss_tts_local.collect_frame.audio_history_update`
- `moss_tts_local.collect_frame.journal`
- `moss_tts_local.collect_frame.journal.rows`
- `moss_tts_local.async_launch.radix_hash_publish`
- `moss_tts_local.async_resolve.restore_next_token_ids`

The base Omni model runner and scheduler also emit `record_function` ranges
with `omni_model_runner.*` and `omni_scheduler.*` prefixes around batch build,
forward, post hooks, async launch/resolve, event waits, stream emission, and
result processing. These give a trace ladder from scheduler batch selection
down to the MOSS frame loop.

The MOSS-TTS Local vocoder emits trace ranges for:

- `moss_tts_local.vocoder.prepare_codes`
- `moss_tts_local.vocoder.decode_audio_codes`
- `moss_tts_local.vocoder.wav_to_cpu`
- `moss_tts_local.vocoder.store_result`

These `torch.profiler.record_function` ranges are always entered in the debug
branch and are intended to split the post-backbone boundary into graph replay,
copy/snapshot work, tensor construction, radix hashing, feedback staging,
journal creation, async handoff, and code-to-waveform work.

The `feedback_write.*` scopes also separate the normal all-emitted path from the
chunked-prefill partial-emission path. In the normal path, request order already
matches pool-row order, so the runner can reuse `row_t`, `rows`, `gen_steps`,
and `embeds` directly instead of materializing an `emit_indices` tensor and
running multiple `index_select` operations. Metadata includes
`all_rows_emit=true/false` and `feedback_fast_path_enabled=true/false`.

Set `SGLANG_MOSS_TTS_LOCAL_FEEDBACK_FAST_PATH=0` before server startup to force
the indexed feedback-write path for A/B testing. The default is enabled because
the normal MOSS Local decode path usually emits every row in the batch.

Set `SGLANG_OMNI_NVTX_RANGES=1` before server startup to mirror these
`record_function` ranges into NVTX ranges. This is intended for Perfetto/Nsight
sync attribution when Chrome trace export drops user labels. Keep it off for
normal speed/WER runs.

Set `SGLANG_MOSS_TTS_LOCAL_VOCODER_DEEP_PROFILE=1` before server startup to
wrap checkpoint remote-code vocoder methods with additional ranges. The wrapper
is diagnostic-only and currently targets the processor `decode_audio_codes`
method plus common audio-tokenizer decode helpers when present. Use it only for
small n8/n16 profiling windows.

For trace export triage, use:

```bash
python scripts/debug/trace_summary.py \
  /path/to/trace.json.gz \
  --out-dir /path/to/reports
```

The script writes `trace_summary.txt`, `trace_summary.json`, and
`perfetto_sync_queries.sql`.

If Chrome trace export drops user `record_function` names, set
`SGLANG_MOSS_TTS_LOCAL_FINE_FRAME_EVENTS=1` for a scoped run. This emits
matching request-event intervals for the same scopes, using sanitized event
names such as `moss_tts_local_frame_decode_cuda_graph_start` /
`moss_tts_local_frame_decode_cuda_graph_end`. These are opt-in because they
multiply event volume in the per-frame loop. Like the broader MOSS Local frame
events, scope intervals are batch-shared and emitted once per active request, so
percentiles and relative deltas are meaningful but summed totals are not unique
GPU time.

For MOSS-TTS Local `torch.compile` experiments, keep the two compile surfaces
separate:

- the Qwen3 backbone uses SGLang's native compile path during decode CUDA graph
  capture and can be toggled with `--talker_torch_compile on`;
- the MOSS-specific frame-local decoder is outside SGLang's normal decode graph
  runner. Its seeded sampler is compiled before MOSS frame CUDA graph capture.

Use `SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs` unless deliberately
testing another mode; Inductor-owned cudagraphs should not be mixed into this
path while SGLang and MOSS Local already manage explicit CUDA graph capture.
For MOSS Local, keep compilation scoped to `sample_seeded_branchless`, the
seeded top-k/top-p sampler used 13 times per frame. Wider frame targets
(`logits` or `full`) failed direct code parity during the issue #752
investigation and should not be re-enabled without a new parity proof.

Custom callsites can call `sglang_omni.profiler.event_recorder.emit(...)` to
add domain-specific events. Events from inactive recorders are no-ops, so
instrumentation sites do not need to guard against the disabled case.

### Active-stage attribution

`emit(...)` accepts an explicit `stage=...` parameter; when the caller can't
plumb the stage name down (preprocessor `__call__`, encoder callables,
`OmniScheduler` / `Code2WavScheduler` internals), it can pass `stage=None`
and the recorder fills it in from the **per-thread / per-task active
stage**.

`Stage._run_scheduler` binds `set_active_stage(self.name)` on the scheduler
thread before invoking the scheduler. The binding uses both a
`threading.local` slot (for plain `threading.Thread` workers) and a
`contextvars.ContextVar` (so it propagates through `asyncio.to_thread` /
`loop.run_in_executor`, which copy contextvars but not thread-local).
Explicit `stage=...` on emit always wins; the active-stage binding is only
consulted when the caller passes `stage=None`.

To bind / unbind manually from your own thread:

```python
from sglang_omni.profiler.event_recorder import set_active_stage, reset_active_stage

token = set_active_stage("my_stage")
try:
    ...
finally:
    reset_active_stage(token)
```

`reset_active_stage(None)` is the "scrub" form (used by test fixtures) and
clears both the thread-local slot and the contextvar.

## Lifecycle

The recorder is process-local. It is started on every stage and on the
coordinator when `POST /start_profile` (or `POST /start_request_profile`)
is hit:

1. Launcher receives the HTTP request.
2. Coordinator starts its local recorder pointed at `<event_dir>`.
3. Launcher broadcasts `ProfilerStartMessage` over ZMQ to every stage,
   carrying both the torch trace template and the `event_dir`.
4. Each stage joins the per-process recorder. In a shared-process topology
   the first stage to call `start()` wins the filename; every subsequent
   stage in the same process writes to the same file and the per-event
   `stage` field disambiguates.
5. On `POST /stop_profile`, the recorder is closed everywhere; files
   remain on disk under `<event_dir>`.

`POST /stop_profile` and `POST /stop_request_profile` accept an optional
`run_id` field. When **omitted**, the request is a wildcard: every stage
stops whatever profiler session is currently active. When **set**, only
stages whose active run matches stop. This makes the common case (caller
didn't specify a run_id on either start or stop) work without ceremony.

The torch profiler and the event recorder share a `run_id`. Setting
`enable_torch=false` on the start request records JSONL events without
paying for a kernel trace.

## Generating reports

Use the views module directly:

```python
from sglang_omni.profiler.views import build_report
report = build_report("/tmp/profiles/demo-run/events")
print(report["request_count"], len(report["stage_breakdown"]))
```

…or via the CLI:

```bash
python -m sglang_omni.profiler /tmp/profiles/demo-run/events --format table
python -m sglang_omni.profiler /tmp/profiles/demo-run/events --format json --out report.json
```

The CLI / `build_report` returns three views derived from the same event
stream:

1. **Timeline** — per-request event list with `t_rel_ms` anchored at
   admission.
2. **Stage breakdown** — `(open_event, close_event)` interval durations
   aggregated per stage (count, total, avg, p50, p95, max). The same opener
   can participate in multiple pairs (e.g. `scheduler_prefill_start` closes
   against both `scheduler_first_emit` AND `stage_first_stream_chunk_sent`);
   every pair gets its own pending stack so a close event for pair A does
   not consume the opener of pair B.
3. **Hop breakdown** — `stage_hop_sent` / `stage_input_received` and
   `stage_stream_chunk_sent` / `stage_stream_chunk_received` durations per
   (source, destination, kind). Terminal stage stream chunks are paired the
   same way with destination `coordinator`.

Hop pairs match across processes by `(request_id, source_stage, dest_stage,
chunk_id?)`, so a single request's path through subprocesses can be
reconstructed even when each stage runs in its own process.

## Torch profiler

The torch profiler runs alongside the event recorder when
`enable_torch=true` (the default for `/start_profile`). It records
continuously between `start()` and `stop()` — no `schedule(...)`, no
`step()` requirement — and exports a Chrome trace `*.trace.json.gz` on stop.

The expensive introspection flags are opt-in via env vars so the default
trace stays small enough to load in `chrome://tracing` or
[`ui.perfetto.dev`](https://ui.perfetto.dev):

| Env var | Effect |
|---|---|
| `SGLANG_TORCH_PROFILER_RECORD_SHAPES=1` | Record input tensor shapes per op |
| `SGLANG_TORCH_PROFILER_PROFILE_MEMORY=1` | Track every CUDA caching-allocator alloc / free |
| `SGLANG_TORCH_PROFILER_WITH_STACK=1` | Record the Python (and C++) call stack per op |
| `SGLANG_TORCH_PROFILER_WITH_FLOPS=1` | Estimate FLOPs per op |

With all four off (the default), a typical 10-sample MMMU run produces a
trace in the tens of MB. With all four on, the same workload can produce a
multi-GB trace — only opt in when you need that specific information.

## HTTP surface

| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/start_profile` | `{"run_id": ?, "trace_path_template": ?, "event_dir": ?, "enable_torch": true \| false, "config": ?}` | Starts torch trace + event recorder. `run_id` auto-generated if omitted. |
| POST | `/stop_profile` | `{"run_id": ?}` | Stops torch trace + event recorder. Omitting `run_id` is a wildcard ("stop whatever's active"). |
| POST | `/start_request_profile` | `{"run_id": ?, "event_dir": ?}` | Event recorder only — no torch trace. Lower overhead; safer to leave on. |
| POST | `/stop_request_profile` | `{"run_id": ?}` | Same wildcard semantics as `/stop_profile`. |

Example: record cheap events on every request without a kernel trace:

```bash
curl -X POST http://localhost:8000/start_request_profile \
     -d '{"run_id":"demo","event_dir":"/tmp/profiles/demo/events"}'
# … run traffic …
curl -X POST http://localhost:8000/stop_request_profile -d '{}'
python -m sglang_omni.profiler /tmp/profiles/demo/events --format table
```

## Discipline

- **Profiling must never break serving.** The emitter swallows write
  errors and counts drops; the first failure is logged once.
- **Tensors and large blobs stay out of event metadata.** Keep metadata
  to small scalars (ids, counts, durations, modality, error strings). The
  recorder enforces this defensively: if a tensor / numpy array ends up
  in metadata, `_json_default` serializes a summary
  (`{"__tensor_summary__": true, "type": ..., "shape": [...], "dtype":
  "...", "device": "..."}`) instead of materializing the contents. 0-D
  tensors / numpy scalars still serialize as plain scalars.
- **Event naming.** Lowercase snake_case, prefix with the layer that
  owns the event (`stage_*`, `scheduler_*`, `encoder_*`, etc.). Use the
  stage name (not the event name) to distinguish "thinker prefill start"
  from "talker prefill start".
