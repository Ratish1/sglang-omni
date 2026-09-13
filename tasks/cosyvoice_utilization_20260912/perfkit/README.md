# CosyVoice perfkit

`slice_trace.py` reads one Nsight SQLite export produced with the branch's pipeline annotations
(`SGLANG_OMNI_PIPELINE_NVTX=1`, `--trace=cuda,nvtx,osrt --cuda-graph-trace=node`) and writes
markdown and JSON ledgers. Standard library only, read-only, integer nanoseconds on the exported
Nsight clock. It runs on the Mac against the returned `trace.sqlite` files; nothing is launched.

```bash
python3 tasks/cosyvoice_utilization_20260912/perfkit/slice_trace.py \
  artifacts/cosyvoice/profile-streaming-en-c16/trace.sqlite \
  --md out/profile-streaming-en-c16.md --json out/profile-streaming-en-c16.json
```

Options: `--device` (CUDA kernel device id, default 0), `--start-ns/--end-ns` (explicit window;
default is first HTTP `received` to the last HTTP endpoint plus 1 ns), `--min-gap-ms` (gap ledger
threshold, default 0.5).

## Ledgers

| section | derived from | what it answers |
|---|---|---|
| AR thread | `ar/execute` ranges on the AR thread, nested `sampling`, `codec_ids_d2h`, `prefill`; kernels joined to their launch by (process, correlation) | host wall against GPU union per step by mode and batch; scheduler gap; eager against graph-node launches; GPU busy fraction while requests are admitted |
| Vocoder thread budget | innermost range on the vocoder thread, subtracted in the order d2h, hift, flow estimator, euler glue, packed glue, flow native, peer wait, payload collection, vocoder glue, idle | where the vocoder thread's time goes and how much GPU each state launched |
| Flow calls, HiFT calls | `flow/native`, `flow/packed_*`, `hift/inference` ranges with launched kernels and runtime API calls in the range | host per call, GPU union per call, host µs per launch, kernel count, mean kernel duration, GPU tail after host return, synchronizing API counts (`cudaStreamSynchronize`, `cudaMemcpy`, `cudaStreamWaitEvent`, `cudaEventQuery`) |
| Preprocessing | `preprocessing/request` and nested ranges | reference miss against hit, finalize lock wait, embedding preparation, cache key |
| Request timeline | HTTP, coordinator, `tts_engine` and `vocoder` marks per request | first-audio decomposition, AR span, drain after AR completion |
| Streaming hops | `vocoder/stream_step` participants, `vocoder/stream_delta` finals, `tts_engine/stage_stream_chunk_sent` chunk ids | per hop: ready time (chunk 0 for 28 tokens, chunk n for 28 plus 25 n), start, run, queue delay, time to PCM yield |
| GPU idle gaps | kernel union gaps intersected with the AR states (execute, active gap, inactive) and the vocoder states | what each thread was doing while the GPU idled, exactly by interval intersection |
| SM Active by kernels present | `GPU_METRICS` samples labeled with every stage whose kernels overlap the 100 µs sampling interval | spatial efficiency of each stage's kernels |

Attribution rules: a kernel belongs to the thread whose runtime API row shares its process and
correlation id (graph-replayed node kernels share the id of their `cudaGraphLaunch`); a kernel's
stage is the innermost annotated range containing that launch on that thread. Kernels without a
launch row stay unattributed and are counted. Ranges are never inferred from kernel names.

## Reading the numbers

- `gpu_over_host` well below 1 with `mean_kernel_us` near the launch cost is launch bound; the
  fix is fewer launches (graphs), not faster kernels.
- `sync_calls_per_call` above a handful means host tensors cross to the device inside the call;
  each is a pipeline drain.
- `host_us_per_kernel` on the vocoder thread rising while `ar_state` is `execute` is interpreter
  contention between the two Python loops.
- A large `queue_delay` on `final` hops is the pump loop starving finals; on batched follow-ups it
  is the serial vocoder falling behind the AR.
- The window mean of SM Active equals the conditional means weighted by coverage; use the
  conditional table to separate spatial from temporal loss.

## Capture protocol additions

1. One c16 capture with `--trace=cuda,nvtx,osrt,python-gil` to measure GIL wait and hold per
   thread; the perfkit ingests the `Waiting for GIL` and `Holding GIL` ranges when present.
2. One c16 capture with `--python-sampling=true --python-sampling-frequency=1000` to attribute the
   HiFT synchronization sites and the first multi-row prefill sampling stall to Python frames.
3. Add `mark("vocoder", "ingest", request_id, chunk_id, tokens)` at `ingest` so token order is
   observable; add `mark("ar", "chunk_emitted", request_id, chunk_id, tokens)` at `_emit_code_chunk`.
4. Long c16 cohort of at least 128 requests for a steady-state interior window; report both the
   full window and the interior window.
5. Profiler-off full runs use `/start_request_profile` (event recorder JSONL, no Torch profiler);
   the same op names appear there, so the request and hop ledgers can be built without Nsight.
   This is how a 300 s timeout is localized to the hop that never ran.
