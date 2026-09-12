Run this analysis in the Linux container. It reads existing reports: it does not launch a server, send requests, run pytest, or collect a new profile. An idle H100 is not required; report export uses CPU, RAM and disk. Do not run Nsight on the Mac.

The scripts were checked against NVIDIA's installed export-schema documentation and report queries. They have not been executed against the E5 SQLite exports, which were omitted from the downloaded archive. An unexpected schema or clock origin stops the script and preserves the diagnostic output; do not change a column name or subtract a guessed timestamp merely to make it run.

1. Copy `analyse_existing_e5.sh` and `inspect_sqlite.py` into the same directory in the container, for example:

   `/sgl-workspace/sglang-omni/tasks/qwen3_tts_e5_investigation_20260912/`

2. In the container, run exactly:

```bash
cd /sgl-workspace/sglang-omni
bash tasks/qwen3_tts_e5_investigation_20260912/analyse_existing_e5.sh \
  /sgl-workspace/sglang-omni/tmp/e5-22951f0ee/e5
```

The last argument is the existing directory containing `nsight/control/window.nsys-rep`, `nsight/early/window.nsys-rep`, and their `events_window/` directories. If your retained directory moved, change that one argument. Use the same `nsys` 2026.2.1 installation used for E5, or a newer version, and Python 3.11 or newer. There are no Python package installs.

The script prints a new output directory, exports each report sequentially into that directory, then reads each SQLite database in read-only mode. It preserves the input files and all earlier exports. It writes a compact return bundle and retains detailed compressed timelines in the container.

3. Read the outputs in this order.

| File, under each arm's output directory | Exact interpretation |
|---|---|
| `window.json` | Capture origin and the selected interval. Both arms use relative `[1 s, 19 s)`, an equal 18-second interior. Host event time becomes `timestamp_ns - epoch_ns`. |
| `thread_identity_candidates.csv` | Roles suggested by actual function names in readable sampled stacks. These are candidates, not an assignment inferred from a thread number. |
| `python_payload_examples.json`, `python_related_strings.json`, `metadata.json` | Raw stack/payload evidence and schemas needed to verify the candidates or decode encoded Python backtraces. If unreadable, leave the role unresolved. |
| `cuda_api_stats.csv` | Count and CPU duration distribution for each API, separately for each process/thread and runtime/driver table. |
| `gil_stats.csv` | Per-thread GIL wait/hold range durations, clipped to the selected interval. These totals are not per-request latency. |
| `kernel_stats.csv` | GPU duration and launch-to-start distributions, separated by submitting thread, stream and launch API. Queue durations overlap; do not sum them into wall time. |
| `kernel_join_status.json` | Work that could not be joined to a runtime launch. Driver-only or unmapped graph work must not be silently assigned to a thread. |
| `first_audio_paths.csv`, `first_audio_summary.json` | Requests with admission, chunk-0 receipt, explicit first-audio send and chunk-0 coordinator receipt all inside the interval. Incomplete paths are counted separately. |

4. Establish a thread identity table before interpreting the waits.

Create `thread_roles.csv` in the output root, with these columns:

```text
arm,pid,tid,role,stack_evidence,status
```

For both arms, identify the talker, reference-code encoder, preprocessing workers, initial vocoder and both followup workers. Copy an actual sampled stack as evidence. Use these functions to distinguish them:

| Role | Stack evidence |
|---|---|
| Initial vocoder | `streaming_vocoder.py` → `_run_initial_worker` / `_run_initial_batch` |
| Followup vocoder | `streaming_vocoder.py` → `_run_followup_worker` / `_run_followup_batch` |
| Reference-code encoder | `request_builders.py` → `_Qwen3TTSRefCodeBatcher._run`, calling tokenizer `encode`, or `_synchronize_outcomes` |
| Talker scheduler | `omni_scheduler.py` / `model_runner.py` → `_collect_codes`, `code_predictor_forward`, `_write_feedback_buffers` |
| Preprocessing | Preprocessing executor → request preparation / speaker embedding / reference-service future wait |

A generic `threading._bootstrap`, `queue.get`, or thread name `python` is insufficient. Followup workers share functions; use their distinct native TIDs and CUDA streams, not a made-up worker index. If the exported Python payload requires decoding, use its recorded NVTX payload schema and StringIds; do not guess from binary bytes. Keep the original SQLite files for this inspection. No additional collection is required to decode existing data.

Explicitly verify the readout's disputed control TID `1159764` and candidate TID `1165827`. Do not prefill them as initial vocoders. A different kernel mix is a reason to investigate their identity, not proof of it.

5. Recompute the comparison using verified identities.

For each verified initial worker, take the following values from its rows in the generated CSVs. Use the same 18-second interval on both arms:

- Number and CPU durations of `cudaStreamWaitEvent` calls.
- Number and CPU duration distribution of `cudaEventSynchronize`, separately from stream/device synchronization.
- GIL wait/hold totals and distributions. Do not divide these by every event synchronization and call the result per-output cost: event synchronization can belong to other operations or batched outputs.
- Kernel GPU duration and launch-to-start distribution, keeping direct launches and graph launches separate.
- Explicit first-audio request count from `first_audio_paths.csv`. This is a request count, not a decode-launch count.

Readiness waits are present in the base code. `post_process_outputs` records a code-ready event; the stream builder records a replacement event after prefix concatenation; the vocoder waits on the attached event. PyTorch's event wait issues `cudaStreamWaitEvent` even if the recorded event has already completed. Therefore “zero wait calls” does not mean “all inputs happened to be ready.” Resolve thread/capture attribution first.

6. Check for a moved timing boundary before blaming extra decoder work.

The optimization publishes codes earlier, before the predictor necessarily finishes. The control's predictor wait occurs before publication; part of that wait can appear after publication in the candidate. Therefore a larger code-receipt-to-first-audio interval does not, on its own, establish added decoder cost. Compare `prefill_to_audio_ms` (prefill-start to first-audio send), `preprocessing_ms`, and end-to-end first-audio latency on the same complete paths. Do not sum independently calculated percentiles; subtract timestamps per request, then aggregate. Keep input/cache-work differences and these different phase boundaries in the explanation.

7. Inspect individual slow paths; do not infer the critical path from averages.

Take the ten slowest complete paths in each arm's `first_audio_summary.json`, plus ordinary paths near the median in `first_audio_paths.csv`. The detailed files remain in the container:

```text
stage_events.jsonl.gz
selected_cuda_api.jsonl.gz
kernels.jsonl.gz
CUPTI_ACTIVITY_KIND_CUDA_EVENT.jsonl.gz             # if exported
CUPTI_ACTIVITY_KIND_SYNCHRONIZATION.jsonl.gz        # if exported
```

Use their `request_id` for stage events, and their relative nanosecond timestamps to inspect the same interval in the verified worker's CUDA timeline. CUDA records must be joined by process/context and correlation/event identities. A bare `correlationId` is not globally unique. Event handles can be reused: prefer `eventSyncId` plus process/context where available and verify temporal ordering. Do not join every occurrence of an event ID to every record with that ID. A nearby CPU event or kernel is not automatically that request's work.

Create one row per inspected request in `critical_paths.csv`:

```text
arm,request_id,worker_tid,code_received_ns,initial_enqueued_ns,worker_started_ns,plan_started_ns,required_ready_event,waited_ready_event,selected_end_frame,available_end_frame,decode_submit_ns,decode_gpu_start_ns,decode_gpu_end_ns,audio_d2h_end_ns,audio_sent_ns,evidence,unresolved
```

Populate a field only when directly linked by a request marker, stack plus an unambiguous operation, or a recorded CUDA correlation. Leave it empty and explain in `unresolved` otherwise. The `CUPTI_ACTIVITY_KIND_CUDA_EVENT.timestamp` field describes an event record; it is not by itself proof of the device completion time of all preceding work.

Use these rules for attributing a delay:

| Proposed cause | Evidence necessary |
|---|---|
| Worker queue/GIL/state-lock delay | Request enqueue and actual worker/plan start, with its CPU/GIL/lock interval. Receipt-to-first-audio alone cannot separate these. |
| Required first-input readiness | The exact first-plan input event, its producer work, and the corresponding consumer wait. |
| Unnecessary later-frame dependency | First-plan frame range versus available range, identity of the later event actually waited on, and evidence that waiting on it delayed useful work. Two chunks received before first audio is insufficient: the second might arrive after planning. |
| GPU execution/service delay | The same decode launch and GPU work, with predecessor/dependency completion established. Launch-to-start alone combines stream backlog, dependencies and scheduling. |
| Output routing delay | The decode's audio D2H completion and the same request's outbox enqueue/dequeue/send. A CPU event-synchronize duration alone includes wakeup effects. |

E5 added none of the request-specific initial enqueue/start/plan/selected-frame markers. It may therefore identify the correct threads and refute the original readout without being able to fill this table. That is a valid result. Do not invent a complete latency decomposition from these missing boundaries.

The current planner concatenates all retained chunks before slicing. That concatenation really reads the later chunks. Merely changing its wait to the first chunk's event would be unsafe. A later-frame dependency hypothesis must distinguish inputs required by the current concatenation from inputs logically required by the selected decode range; removing that dependency would require changing input selection and preserving matching readiness/lifetime, not deleting a wait.

8. Record the control abort accurately.

```bash
rg -n -m 3 'double free or corruption|Dead stage process' \
  /sgl-workspace/sglang-omni/tmp/e5-22951f0ee/e5/nsight/control/serve.log
```

It reports a native abort (`exit=-6`) near collection stop. Record its relation to the capture endpoint. Do not relabel it a normal shutdown or attribute it to early IDs, which that arm did not run. The existing evidence does not establish whether Nsight caused the abort.

9. Stop here and return the compact bundle, `thread_roles.csv` and `critical_paths.csv`.

If an export or analysis command fails, return the printed output directory's export/analysis log and `schema.sql`/`metadata.json`/`window.json` when present. Do not launch a new benchmark to repair an analysis failure.

If the critical-path table remains unresolved because request markers are absent, report exactly which fields are missing. The next collection should then add the same small diagnostic patch to both arms and capture only those missing boundaries, using the same workload. It must preserve selected tensor lifetimes/readiness and avoid `.item()`, `.cpu()`, tensor formatting, new synchronization or a changed GPU priority. A new capture is not part of this procedure.

The current E5 reports contain only control versus early IDs. After a justified fix is demonstrated, the nonblocking-copy stack needs a separate comparison against early IDs on the same base. It cannot be qualified from E5's two existing arms.
