# Higgs Torch Profiler H200 SeedTTS Runbook

Purpose: capture one authoritative full-stack torch profiler trace for Higgs TTS
on H200. This is not a WER run and not a command matrix. Run exactly this flow
first, inspect the trace, then iterate.

## Profile Contract

- Workload: full SeedTTS EN generation, no `--max-samples`.
- Concurrency: 16.
- Server mode: CUDA graph on, torch.compile off, async decode on.
- Profile window: only the full generation benchmark. Server startup, first
  request cold work, and ASR/WER are outside the torch-profiler window.
- Output: request events plus `trace_rank*.trace.json.gz`.

Do not run ASR/WER while torch profiling is active; it pollutes the trace with a
different model and a different serving path.

## Terminal 1: Start Higgs

```bash
cd /workspace/sglang-omni

export MODEL_PATH="${HIGGS_CKPT:-boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999}"
export PORT=8000
export PROFILE_ROOT=results/higgs_profiles
export SGLANG_TORCH_PROFILER_DIR="${PROFILE_ROOT}"

# Keep the trace loadable. Full-set c16 is already an extreme profile.
export SGLANG_TORCH_PROFILER_RECORD_SHAPES=1
unset SGLANG_TORCH_PROFILER_PROFILE_MEMORY
unset SGLANG_TORCH_PROFILER_WITH_STACK
unset SGLANG_TORCH_PROFILER_WITH_FLOPS

python -m benchmarks.dataset.prepare --dataset seedtts
mkdir -p "${PROFILE_ROOT}"

CUDA_VISIBLE_DEVICES=0 python -m sglang_omni.cli serve \
  --model-path "${MODEL_PATH}" \
  --port "${PORT}" \
  --talker-cuda-graph on \
  --talker-torch-compile off \
  --async-decode on
```

## Terminal 2: Run The Single Profile Test

```bash
cd /workspace/sglang-omni

export MODEL_PATH="${HIGGS_CKPT:-boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999}"
export PORT=8000
export PROFILE_ROOT=results/higgs_profiles
export RUN_ID="higgs-h200-seedtts-en-c16-$(date +%Y%m%d-%H%M%S)"

mkdir -p "results/${RUN_ID}"

# Warm up outside the profiler so the trace focuses on steady-state serving.
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model "${MODEL_PATH}" \
  --port "${PORT}" \
  --ref-format references \
  --output-dir "results/${RUN_ID}_warmup" \
  --lang en \
  --max-samples 16 \
  --concurrency 16 \
  --warmup 1 \
  --temperature 0.8 \
  --top-p 0.8 \
  --top-k 50 \
  --seed 123 \
  --disable-tqdm

curl -sS -X POST "http://localhost:${PORT}/start_profile" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\":\"${RUN_ID}\",\"enable_torch\":true}"

cleanup_profile() {
  curl -sS -X POST "http://localhost:${PORT}/stop_profile" \
    -H "Content-Type: application/json" \
    -d "{\"run_id\":\"${RUN_ID}\"}" >/dev/null || true
}
trap cleanup_profile EXIT

# Full SeedTTS EN set. Do not pass --max-samples here.
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model "${MODEL_PATH}" \
  --port "${PORT}" \
  --ref-format references \
  --output-dir "results/${RUN_ID}" \
  --lang en \
  --concurrency 16 \
  --warmup 0 \
  --temperature 0.8 \
  --top-p 0.8 \
  --top-k 50 \
  --seed 123 \
  --disable-tqdm

cleanup_profile
trap - EXIT

python -m sglang_omni.profiler \
  "${PROFILE_ROOT}/${RUN_ID}/events" \
  --format table | tee "results/${RUN_ID}/request_profile_table.txt"

for _ in $(seq 1 120); do
  if ls "${PROFILE_ROOT}/${RUN_ID}"/trace_rank*.trace.json.gz >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

ls -lh "${PROFILE_ROOT}/${RUN_ID}"/trace_rank*.trace.json*
ls -lh "results/${RUN_ID}/speed_results.json" \
       "results/${RUN_ID}/generated.json" \
       "results/${RUN_ID}/request_profile_table.txt"
```

## Pass Criteria

- `results/${RUN_ID}/speed_results.json` exists and reports the full EN set.
- `results/${RUN_ID}/generated.json` exists and has generated audio entries.
- `${PROFILE_ROOT}/${RUN_ID}/events/` contains JSONL event files.
- `${PROFILE_ROOT}/${RUN_ID}/trace_rank*.trace.json.gz` exists after gzip finishes.
- The trace contains these steady-state envelope labels:
  - `omni.scheduler.model_runner_execute`
  - `omni.model_runner.forward_batch_generation`
  - `higgs.runner.decode_pack_gpu`
  - `higgs.runner.decode_collect_d2h_async`
  - `higgs.vocoder.codec_decode_batch`

First inspection targets: CUDA graph replay kernels under
`omni.model_runner.forward_batch_generation`, post-decode D2H ranges, and
vocoder decode ranges. Inner `higgs.model.*` and `higgs.sampler.*` labels are
expected on prefill, eager fallback, or graph-capture paths; they are not a
required pass criterion for steady-state CUDA graph replay because replay does
not re-enter Python `forward`. Use the CUDA kernel rows under the replay
envelope for the steady-state backbone/head/sampler attribution. If the trace is
too large to load, keep the same run and rerun only after unsetting
`SGLANG_TORCH_PROFILER_RECORD_SHAPES`.
