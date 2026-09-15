# Stage 1: component profiles of every Fun-CosyVoice3 serving call

Written 2026-09-15 on upstream main `cc85ddaa9` (the #2169, #2170, #2171 stack merged; its tree equals
the stage 0 tree `2eefbc476`). Targets: SM Issue about 30 percent at c16, RTF p99 below 1, first audio
reasonable. Stage 0 baseline at stream c16: RTF p99 1.74, first audio p95 2.27 s, C50 72.4
(`../readouts/03_stage0_2eefbc476_20260915.md`).

## Why component profiles, not concurrency runs

A request's time is the sum of the calls that serve it plus the time it queues for them:

```text
request = reference encode + text prep            preprocessing
        + one prefill + one decode step per token  tts_engine (AR)
        + hops and a final (Flow, then HiFT)       vocoder
        + queueing in front of each
```

Every call has the same anatomy:

```text
wall = host dispatch (Python, kernel launches)
     + host blocked (syncs, blocking copies)
     + device tail after the last launch
device busy = union of the intervals the GPU executed this call's work
```

- busy / wall well below 1 with many launches: launch bound. Fewer launches is the fix (graph replay,
  fusion, dead work removed). Kernel choice does not move it.
- busy / wall well below 1 with blocked ms: sync bound. Removing the transfer or the read is the fix.
- busy / wall near 1: device bound. Less device work is the fix (layout without padding, recompute
  removed, better kernels, fusion of memory bound pointwise runs).

SM Issue at c16 is the time weighted busy of the calls that run, divided by wall, times how full each
kernel keeps the SM array. A call's anatomy at batch 1 and at the batch the workload forms, plus how
often the workload forms each, determines it. So every call is measured alone at those batches, the way
SGLang's `sglang.benchmark.one_batch` measures one prefill and single decode steps at fixed batch and
length, and the concurrency run is kept for the final A/B.

## The points, and where each comes from

From the stage 0 c16 call ledger (readout 03, section 6):

| point | shape | source |
|---|---|---|
| flow_hop_first_rows1 | 1 row, prompt 125 tokens, window 28 | first hop at c1; prompt p50 125 |
| flow_hop_first_rows16 | 16 rows, window 28 | c16 first hop cohort; hop rows max 16 |
| flow_hop_late_rows16 | 16 rows, window 378 (offset 275, hop 100) | hop length 100 is a third of hops |
| flow_hop_runaway_rows16 | 15 rows window 78, 1 row window 2,051 | widest rows reach 4,250 frames; pad ratio p95 5.7 |
| flow_final_rows1, flow_final_rows16 | 125 tokens | output tokens mean 125 |
| flow_final_long_rows16 | 500 tokens | tail of the output length |
| flow_buffered_rows16_graph and _eager | 16 rows, 125 tokens, one captured key | buffered groups run up to 15 rows |
| hift_hop_history0, 1000, 4000 | 50 new frames on that history | HiFT reruns the whole history each hop |
| hift_final_history1000 | finalize on 1,000 frames | the final flush |
| hift_batch_rows16_250 | 16 mels of 250 frames | buffered HiFT batches |
| preprocessing | shortest, median, longest reference: load 16k, load 24k, CAM++, S3 tokenizer, prompt mel; first call and repeat | a new reference length pays a first call |

Prompts are real SeedTTS references; generated tokens are random ids of the stated length. Numerics are
not measured here (stage 0 E2, E5, E6 did that).

### AR (`profile_ar.py`, its own process)

The engine comes from the serving factory (`create_sglang_tts_engine_executor`, bf16, 16 ONNX threads,
hop 25) and is never started. One step is the event loop body without the inbox poll:
`get_next_batch_to_run`, `run_batch`, `process_batch_result`. Requests enter through the real ingress
(`preprocess_cosyvoice3_payload`, then `process_input_requests`) from the first 400 SeedTTS en samples,
stream on.

| point | shape | why |
|---|---|---|
| ar_prefill_rows1_prompt{min, p50, max} | one request | prefill cost against prompt length |
| ar_prefill_rows16_median | 16 requests near the median prompt | c16 arrival cohort |
| ar_prefill_admit_of32_median | 32 waiting, one step admits what max_prefill_tokens allows | the prefill ceiling |
| ar_decode_rows{1, 16, 32}_graph and _eager | one decode step | replay against eager, launches per step |
| ar_decode_hop25_rows{1, 16, 32}_graph | 25 steps as one call | one hop, one stream chunk per request |
| ar_decode_rows32_graph_generated1000 | one step at 1,000 generated tokens | decode against KV length |

- Before each prefill repeat, the previous requests are aborted and stepped out and the radix cache
  is flushed, all outside the timed window. The measured step reuses no prefix and, like a served
  prefill, finishes no request.
- Decode requests hold stop tokens off with min_new_tokens equal to max_new_tokens.
- Eager clears `decode_cuda_graph_runner`, the state `disable_cuda_graph` leaves.
- `ar.md` lists the shape of every measured step (mode, rows, tokens), so a point that formed a
  different batch is visible.
- The KV pool is sized without the vocoder resident, so it is larger than in serving. Step cost does
  not depend on pool size.

## What each ledger holds

`trace_ledger.py` times the call without instrumentation (synchronized median of 5), then runs it once
under torch.profiler with a range around the call, around every module forward and around the Flow and
HiFT functions that are not modules (`pack_flow_inputs`, `prepare_flow_conditioning`,
`solve_flow_euler_packed`, `RowAttention.__call__`, `scatter_rows`, `gather_rows`, `_stft`, `_istft`
and the others listed in `profile_components.py`). From the Chrome trace:

| field | meaning |
|---|---|
| wall_ms_median | synchronized wall of the call without ranges |
| device_busy_ms, device_busy_share_of_wall | union of device intervals linked to this call's launches |
| launches, graph_launches | CUDA runtime launch events, of which graph replays |
| syncs, blocking_copies, host_blocked_ms | `cudaStreamSynchronize` / `cudaDeviceSynchronize` / `cudaEventSynchronize` and `cudaMemcpy`, with the host time inside them |
| ranges | per range: launches, self and inclusive device ms, pointwise kernels, syncs, blocked ms |
| top_kernels | kernel name, count, device ms, owning ranges |
| repeated_sequences | ranges whose instances launch the identical kernel sequence: graph or fusion candidates |
| trace_categories | the trace's category inventory, so a schema change is visible |

A device event belongs to the runtime event with the same `args.correlation`; a runtime event belongs
to the innermost range on its thread containing it. Profiled host time includes the ranges' own cost;
wall does not.

## Run

```bash
cd /sgl-workspace/sglang-omni
git fetch https://github.com/sgl-project/sglang-omni.git main
[ -d /sgl-workspace/wt/cosy-main ] || git worktree add --detach /sgl-workspace/wt/cosy-main cc85ddaa9
git -C /sgl-workspace/wt/cosy-main checkout --detach cc85ddaa9
git fetch https://github.com/Ratish1/sglang-omni.git analysis/cosyvoice-utilization-20260912
ANALYSIS=$(git rev-parse FETCH_HEAD)
[ -d /sgl-workspace/wt/cosyvoice-analysis ] || git worktree add --detach /sgl-workspace/wt/cosyvoice-analysis "$ANALYSIS"
git -C /sgl-workspace/wt/cosyvoice-analysis checkout --detach "$ANALYSIS"

S1=/sgl-workspace/wt/cosyvoice-analysis/tasks/cosyvoice_utilization_20260912/stage1
OUT=/sgl-workspace/wt/cosyvoice-analysis/artifacts/cosyvoice/stage1-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
cd /sgl-workspace/wt/cosy-main
COSYVOICE_PATH=/sgl-workspace/CosyVoice-utilization-20260912
export PYTHONPATH="/sgl-workspace/wt/cosy-main:$S1/../stage0:$S1:$COSYVOICE_PATH:$COSYVOICE_PATH/third_party/Matcha-TTS"
git rev-parse HEAD > "$OUT/head.txt"
python -c "import sglang_omni, cosyvoice, matcha.utils.audio; print(sglang_omni.__file__); print(cosyvoice.__file__); print(matcha.__file__)" | tee "$OUT/import_path.txt"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > "$OUT/gpus_before.csv"
nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv
uptime > "$OUT/host_load.txt"; nproc >> "$OUT/host_load.txt"
CUDA_VISIBLE_DEVICES=0 python "$S1/profile_components.py" --device cuda:0 \
  --components flow,hift,preprocess --out "$OUT" 2>&1 | tee "$OUT/components.log"
CUDA_VISIBLE_DEVICES=0 python "$S1/profile_ar.py" --device cuda:0 \
  --out "$OUT" 2>&1 | tee "$OUT/ar.log"
```

`PYTHONPATH` makes both scripts import the tree under test and the stage0 and stage1 helpers (a
script's own directory, not the working directory, is first on its path), plus the CosyVoice clone
and its Matcha-TTS submodule, which the cookbook requires. Preprocessing imports Matcha only when it
first computes a prompt mel, so the check line imports it up front instead of failing after the
engine boots. Each JSON records the `sglang_omni` file it loaded, which must be
under `/sgl-workspace/wt/cosy-main`. Host dispatch time depends on CPU contention, so `gpus_before.csv`
and `host_load.txt` record what else ran. The two scripts run as separate processes, one after the other.
GPU 0 must list no process. Outputs: `components.md` and `ar.md` (tables), `components.json` and
`ar.json` (every field), `traces/<point>.trace.json.gz` (the Chrome traces, for Perfetto or
`chrome://tracing`).

To view a trace, gunzip it and open the `.json` at https://ui.perfetto.dev. The Python thread's track
nests `call:<point>`, then `fn:` ranges (scheduler, runner and vocoder functions), then `mod:`
ranges (module forwards). Each `cudaLaunchKernel` or `cuLaunchKernel` has a flow arrow to its kernel
on the GPU stream track, and `gpu_user_annotation` repeats the same ranges on device time.

## Return

```bash
cd /sgl-workspace/wt/cosyvoice-analysis
tar -czf "$(basename "$OUT").tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
```
