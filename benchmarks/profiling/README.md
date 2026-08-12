# CUDA synchronization profiling

These tools support the Qwen3-TTS hidden host/device synchronization study.
They do not change transfer behavior; use H100 evidence from them to select a
repair.

## 1. Calibrate the Torch detector

Use the exact Python environment that will launch the server. Each invocation
must be a fresh process:

```bash
for mode in warn error; do
  for case in \
    item dtoh_pageable h2d_pageable \
    h2d_pinned_nonblocking dtoh_pinned_nonblocking \
    stream_synchronize device_synchronize; do
    python benchmarks/profiling/cuda_sync_debug_probe.py \
      "$case" "$mode" | tee -a "/tmp/cuda-sync-probe-${mode}.jsonl"
  done
done
```

Treat the observed matrix as authoritative for the installed Torch build.
Blocking scalar/D2H controls and explicit synchronization should produce a
warning or error where the build has coverage. Pinned nonblocking copies should
return without a detector hit. If positive controls are silent, a silent server
log does not prove the model has no synchronization.

## 2. Arm a bounded server window

After server readiness, graph capture, and out-of-window warmup, start a warning
discovery window without Torch Profiler overhead:

```bash
curl -fsS -X POST http://127.0.0.1:8000/start_profile \
  -H 'content-type: application/json' \
  -d '{
    "run_id": "q3tts-warn",
    "event_dir": "/tmp/q3tts-prof/q3tts-warn/events",
    "enable_torch": false,
    "config": {"cuda_sync_debug_mode": "warn"}
  }'
```

For a low-overhead CPU+CUDA trace, set `enable_torch` to `true` and supply an
absolute server-side `trace_path_template`. Send only the fixed target workload,
then stop with the matching run ID:

```bash
curl -fsS -X POST http://127.0.0.1:8000/stop_profile \
  -H 'content-type: application/json' \
  -d '{"run_id": "q3tts-warn"}'
```

The HTTP routes broadcast asynchronously. Before traffic, require a server log
line showing `applied=True` for the CUDA-owning PID and all expected colocated
stage joins. After stop, require the process-session stop log, `Trace exported`,
and a gzip whose size and mtime are stable on two polls. Do not retry the same
start request as an acknowledgement.

Use `SGLANG_TORCH_PROFILER_WITH_STACK=1` and
`SGLANG_TORCH_PROFILER_RECORD_SHAPES=1` only for a separate targeted stack pass.
Keep memory profiling in its own capture.

## 3. Analyze each process trace

```bash
python benchmarks/profiling/analyze_cuda_sync_trace.py \
  /tmp/q3tts-prof/q3tts-trace/*.trace.json.gz \
  --output-dir /tmp/q3tts-prof/q3tts-trace/analysis
```

The analyzer emits per-occurrence JSON plus aggregate JSON and CSV. It streams
the Chrome trace twice: once for synchronization/GPU correlation and once for
nested semantic, ATen, and Python attribution. It never compares timestamps
across trace files. A reported sync wait is not automatically wasted time;
inspect the correlation method, post-sync GPU bubble, host launch gap, and queue
horizon together.
