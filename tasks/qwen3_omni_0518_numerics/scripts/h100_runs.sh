#!/usr/bin/env bash
# Server launch and benchmark commands for the arms in 00_plan.md section 4.
# Run inside the container of one stack. Every function takes a PORT and an
# optional list of extra serve flags. Results land under $OUT/<arm>/.
#
# Required environment:
#   OMNI_ROOT   omni checkout mounted in the container
#   OUT         result directory
#   GPU         CUDA_VISIBLE_DEVICES value for the worker (one GPU)
#
# Examples:
#   serve_fp8_colocated 31000
#   serve_fp8_colocated 31000 --stages.thinker.engine.moe_runner_backend triton
#   SGLANG_ENABLE_JIT_DEEPGEMM=0 serve_fp8_colocated 31000
#   bench_amme 31000 fp8_baseline_c16 16
#   bench_amme 31000 fp8_baseline_c1 1
#   backend_lines $OUT/logs/serve_fp8_31000.log
#   serve_bf16_thinker 31001
#   bench_mmsu 31001 bf16_baseline_c16 16
#   serve_bf16_disagg 31002     # needs two GPUs, GPU=0,1
#   bench_mme_talker 31002 bf16_disagg_c16 16

set -u

FP8_MODEL=marksverdhei/Qwen3-Omni-30B-A3B-FP8
BF16_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
FP8_CONFIG=examples/configs/qwen3_omni_colocated_h100_fp8.yaml
BF16_THINKER_CONFIG=examples/configs/qwen3_omni_mmmu_h100.yaml
THINKER_MAX_SEQ_LEN=32768

serve_fp8_colocated() {
  local port=$1
  shift
  cd "$OMNI_ROOT" && mkdir -p "$OUT/logs" \
  && CUDA_VISIBLE_DEVICES="$GPU" nohup sgl-omni serve \
      --model-path "$FP8_MODEL" --host 127.0.0.1 --port "$port" --model-name qwen3-omni \
      --config "$FP8_CONFIG" --colocate \
      --stages.thinker.factory-args.thinker-max-seq-len "$THINKER_MAX_SEQ_LEN" \
      "$@" > "$OUT/logs/serve_fp8_$port.log" 2>&1 &
  echo $! > "$OUT/logs/serve_$port.pid"
  wait_ready "$port"
}

serve_bf16_thinker() {
  local port=$1
  shift
  cd "$OMNI_ROOT" && mkdir -p "$OUT/logs" \
  && CUDA_VISIBLE_DEVICES="$GPU" nohup sgl-omni serve \
      --model-path "$BF16_MODEL" --host 127.0.0.1 --port "$port" --model-name qwen3-omni \
      --config "$BF16_THINKER_CONFIG" \
      "$@" > "$OUT/logs/serve_bf16_thinker_$port.log" 2>&1 &
  echo $! > "$OUT/logs/serve_$port.pid"
  wait_ready "$port"
}

serve_bf16_disagg() {
  local port=$1
  shift
  cd "$OMNI_ROOT" && mkdir -p "$OUT/logs" \
  && CUDA_VISIBLE_DEVICES="$GPU" nohup python examples/run_qwen3_omni_speech_server.py \
      --model-path "$BF16_MODEL" --port "$port" --model-name qwen3-omni \
      --thinker-max-seq-len "$THINKER_MAX_SEQ_LEN" \
      --gpu-thinker 0 --gpu-image-encoder 0 --gpu-audio-encoder 0 --gpu-talker 1 --gpu-code2wav 1 \
      --thinker-mem-fraction-static 0.82 --talker-mem-fraction-static 0.40 \
      "$@" > "$OUT/logs/serve_bf16_disagg_$port.log" 2>&1 &
  echo $! > "$OUT/logs/serve_$port.pid"
  wait_ready "$port"
}

wait_ready() {
  local port=$1
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
      echo "ready on $port"
      return 0
    fi
    sleep 5
  done
  echo "server on $port did not become ready" >&2
  return 1
}

stop_server() {
  local port=$1
  if [ -f "$OUT/logs/serve_$port.pid" ]; then
    kill "$(cat "$OUT/logs/serve_$port.pid")" 2>/dev/null || true
    rm -f "$OUT/logs/serve_$port.pid"
  fi
  sleep 15
}

RUN_BENCH="$OMNI_ROOT/tasks/qwen3_omni_0518_numerics/scripts/run_bench.py"

# Video-AMME, stage 9 settings (50 clips, 2 fps, 128 frames, 401408 pixels).
bench_amme() {
  local port=$1 arm=$2 conc=$3
  cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python "$RUN_BENCH" videoamme --port "$port" --out "$OUT/$arm" --concurrency "$conc"
}

# MMSU text only, stage 5 settings (mmsu-ci-2000).
bench_mmsu() {
  local port=$1 arm=$2 conc=$3
  cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python "$RUN_BENCH" mmsu --port "$port" --out "$OUT/$arm" --concurrency "$conc"
}

# Video-MME with the talker, stage 8 settings (20 clips, speech output, no inline WER).
bench_mme_talker() {
  local port=$1 arm=$2 conc=$3
  cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python "$RUN_BENCH" videomme-talker --port "$port" --out "$OUT/$arm" --concurrency "$conc"
}

# Arms of section 4.3 on the FP8 colocated server. Each arm restarts the server.
run_fp8_arms() {
  local port=31000
  for arm in baseline deepgemm_off moe_triton attn_triton audio_graph_off fp32_lm_head; do
    case $arm in
      baseline)        serve_fp8_colocated $port ;;
      deepgemm_off)    SGLANG_ENABLE_JIT_DEEPGEMM=0 serve_fp8_colocated $port ;;
      moe_triton)      serve_fp8_colocated $port --stages.thinker.engine.moe_runner_backend triton ;;
      attn_triton)     serve_fp8_colocated $port --stages.thinker.engine.attention_backend triton ;;
      audio_graph_off) serve_fp8_colocated $port --stages.audio_encoder.factory-args.enable-layer-cuda-graph false ;;
      fp32_lm_head)    serve_fp8_colocated $port --stages.thinker.engine.enable_fp32_lm_head true ;;
    esac || return 1
    bench_amme $port "fp8_${arm}_c1" 1
    for i in 1 2 3; do bench_amme $port "fp8_${arm}_c16_$i" 16; done
    stop_server $port
  done
}

# Arms of section 4.3 on the bf16 thinker-only server.
run_bf16_arms() {
  local port=31001
  for arm in baseline attn_triton moe_flashinfer_cutlass fp32_lm_head; do
    case $arm in
      baseline)               serve_bf16_thinker $port ;;
      attn_triton)            serve_bf16_thinker $port --stages.thinker.engine.attention_backend triton ;;
      moe_flashinfer_cutlass) serve_bf16_thinker $port --stages.thinker.engine.moe_runner_backend flashinfer_cutlass ;;
      fp32_lm_head)           serve_bf16_thinker $port --stages.thinker.engine.enable_fp32_lm_head true ;;
    esac || return 1
    bench_mmsu $port "bf16_${arm}_c1" 1
    bench_mmsu $port "bf16_${arm}_c16" 16
    stop_server $port
  done
}

# Stage 8 RTF sample: the first Video-MME clip (001-1) alone, ten times (section 4.5).
run_rtf_sample() {
  local port=$1
  for i in $(seq 1 10); do
    cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python "$RUN_BENCH" videomme-talker --port "$port" --out "$OUT/rtf_001_1_$i" --concurrency 1 --max-samples 1
  done
}

# MMSU answer-token logprobs on both stacks (section 4.4).
run_logprob_probe() {
  local port=$1 arm=$2
  cd "$OMNI_ROOT" && mkdir -p "$OUT/$arm" \
  && PYTHONPATH="$OMNI_ROOT" python "$OMNI_ROOT/tasks/qwen3_omni_0518_numerics/scripts/logprob_probe.py" --port "$port" --out "$OUT/$arm/mmsu_logprobs.json"
}

# Backend and capture lines from a server log (which kernels actually ran).
backend_lines() {
  grep -E "Configured SGLang backend policy|Config file not found|Down MoE config|audio layer CUDA graphs|Capture target decode CUDA graph begin|deferred finalize is|staying eager|Code2Wav CUDA graph runner" "$1" | cut -c1-400
}

# Compare two arms per sample (section 4.3 readout), one result file per attempt.
compare_arms() {
  local pre_glob=$1 post_glob=$2
  python "$OMNI_ROOT/tasks/qwen3_omni_0518_numerics/scripts/ci_artifacts.py" compare-local \
    --pre "$(ls $pre_glob | paste -sd, -)" --post "$(ls $post_glob | paste -sd, -)"
}
