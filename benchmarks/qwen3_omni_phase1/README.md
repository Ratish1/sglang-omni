# Qwen3-Omni Phase 1 H100 Matrix

Phase 1 turns Phase 0 instrumentation into a comparable H100 baseline matrix.
Do not optimize from a single smoke run. Capture the same profiler evidence for
each modality path, then rank bottlenecks across the matrix.

## Server

Use the FP8 colocated H100 config under test for text, audio, image, and
text-to-audio runs:

```bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/qwen3_phase1_profiles
export SGLANG_TORCH_PROFILER_RECORD_SHAPES=0
export SGLANG_TORCH_PROFILER_PROFILE_MEMORY=0
export SGLANG_TORCH_PROFILER_WITH_STACK=0
export SGLANG_TORCH_PROFILER_WITH_FLOPS=0

python -m sglang_omni.cli serve \
  --model-path marksverdhei/Qwen3-Omni-30B-A3B-FP8 \
  --config examples/configs/qwen3_omni_colocated_h100_fp8.yaml \
  --host 0.0.0.0 \
  --port 8000
```

For Video-MME and Video-AMME, use the video-context variant. Those benchmarks
produce prompts above 8k tokens with the standard `--video-fps 2
--video-max-frames 128 --video-max-pixels 401408` settings; the video config
raises preprocessing and thinker context to 32k so requests reach scheduler
admission.

```bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/qwen3_phase1_profiles
export SGLANG_TORCH_PROFILER_RECORD_SHAPES=0
export SGLANG_TORCH_PROFILER_PROFILE_MEMORY=0
export SGLANG_TORCH_PROFILER_WITH_STACK=0
export SGLANG_TORCH_PROFILER_WITH_FLOPS=0

python -m sglang_omni.cli serve \
  --model-path marksverdhei/Qwen3-Omni-30B-A3B-FP8 \
  --config examples/configs/qwen3_omni_colocated_h100_fp8_video.yaml \
  --host 0.0.0.0 \
  --port 8000
```

## Matrix Runs

For each case, start request profiling, run the benchmark, stop profiling, and
render a report.

```bash
run_profiled() {
  local label="$1"
  shift
  local run_id="phase1-${label}-$(date +%Y%m%d-%H%M%S)"
  local root="/tmp/qwen3_phase1_profiles/${run_id}"
  local event_dir="${root}/events"
  local status=0
  local profile_status=0
  local report_status=0

  mkdir -p "${root}/summary"

  if ! curl -sS -X POST http://localhost:8000/start_request_profile \
    -H 'Content-Type: application/json' \
    -d "{\"run_id\":\"${run_id}\",\"event_dir\":\"${event_dir}\"}" \
    -o "${root}/start_profile.json"; then
    echo "start_request_profile failed for ${run_id}" >&2
    return 1
  fi
  cat "${root}/start_profile.json"

  "$@" || status=$?

  if ! curl -sS -X POST http://localhost:8000/stop_request_profile \
    -H 'Content-Type: application/json' \
    -d "{\"run_id\":\"${run_id}\"}" \
    -o "${root}/stop_profile.json"; then
    profile_status=1
    echo "stop_request_profile failed for ${run_id}" >&2
  else
    cat "${root}/stop_profile.json"
  fi

  if ! python -m sglang_omni.profiler "${event_dir}" --format json \
    --out "${root}/summary/report.json"; then
    report_status=1
  fi
  if ! python -m sglang_omni.profiler "${event_dir}" --format table \
    > "${root}/summary/report.txt"; then
    report_status=1
  fi
  echo "${root}"
  if [ "${status}" -ne 0 ]; then
    return "${status}"
  fi
  if [ "${profile_status}" -ne 0 ]; then
    return "${profile_status}"
  fi
  return "${report_status}"
}
```

Recommended first-pass matrix:

```bash
# text -> text
run_profiled text \
  python -m benchmarks.eval.benchmark_omni_mmsu \
    --model qwen3-omni --port 8000 \
    --modalities text --max-samples 8 --max-concurrency 1 \
    --output-dir results/qwen3_phase1_text

# audio + text -> text
run_profiled audio_text \
  python -m benchmarks.eval.benchmark_omni_mmsu \
    --model qwen3-omni --port 8000 \
    --modalities text+audio --max-samples 8 --max-concurrency 1 \
    --output-dir results/qwen3_phase1_audio_text

# image + text -> text
run_profiled image_text \
  python -m benchmarks.eval.benchmark_omni_mmmu \
    --model qwen3-omni --port 8000 \
    --max-samples 8 --max-concurrency 1 \
    --output-dir results/qwen3_phase1_image_text
```

Restart with `examples/configs/qwen3_omni_colocated_h100_fp8_video.yaml` before
running the video cases:

```bash
# video + text -> text
run_profiled video_text \
  python -m benchmarks.eval.benchmark_omni_videomme \
    --model qwen3-omni --port 8000 \
    --output-dir results/qwen3_phase1_video_text \
    --max-samples 8 --max-concurrency 1 \
    --video-fps 2 --video-max-frames 128 --video-max-pixels 401408

# video + audio -> text
run_profiled video_audio_text \
  python -m benchmarks.eval.benchmark_omni_videoamme \
    --model qwen3-omni --port 8000 \
    --repo-id zhaochenyang20/Video_AMME_ci \
    --output-dir results/qwen3_phase1_video_audio_text \
    --max-samples 8 --max-concurrency 1 \
    --video-fps 2 --video-max-frames 128 --video-max-pixels 401408
```

Restart with `examples/configs/qwen3_omni_colocated_h100_fp8.yaml` before
running the text-to-audio case:

```bash
run_profiled text_audio_out \
  python -m benchmarks.eval.benchmark_omni_seedtts \
    --generate-only \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --output-dir results/qwen3_omni_phase1_seedtts \
    --model qwen3-omni --port 8000 \
    --max-samples 8 --max-concurrency 1
```

Increase `--max-concurrency` after the single-concurrency matrix is complete.
Use the same labels with suffixes such as `-c4`, `-c8`, and `-c16`.

## Aggregate Reports

Build one comparable matrix from all run directories:

```bash
python -m benchmarks.qwen3_omni_phase1.analyze_profiles \
  /tmp/qwen3_phase1_profiles/phase1-text-* \
  /tmp/qwen3_phase1_profiles/phase1-audio_text-* \
  /tmp/qwen3_phase1_profiles/phase1-image_text-* \
  /tmp/qwen3_phase1_profiles/phase1-video_text-* \
  /tmp/qwen3_phase1_profiles/phase1-video_audio_text-* \
  /tmp/qwen3_phase1_profiles/phase1-text_audio_out-* \
  --output-dir results/qwen3_omni_phase1_matrix
```

The analyzer writes:

- `phase1_matrix.json`
- `phase1_matrix.md`

Use `bottleneck_signals` to decide the first Phase 2 target. Large relay
transfers point to payload pruning/locality work. Near-context admissions point
to memory/admission modeling. Dominant talker/code2wav stream rows point to
streaming chunking and code2wav scheduling.
