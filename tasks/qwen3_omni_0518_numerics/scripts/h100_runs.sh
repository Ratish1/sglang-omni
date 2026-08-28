#!/usr/bin/env bash
# Server launch, benchmark and readout commands for 00_plan.md section 4.
# Everything runs inside one container (the current CI image) on this branch.
# Every serve function takes a PORT and an optional list of extra serve
# flags. Results land under $OUT/<arm>/.
#
# Required environment:
#   OMNI_ROOT   this checkout, mounted in the container
#   OUT         result directory
#   GPU         CUDA_VISIBLE_DEVICES value (one GPU, two for serve_bf16_disagg)
#
# Order (section 4.5):
#   source h100_runs.sh
#   check_install
#   run_unit_tests
#   serve_fp8_colocated 31000 && smoke_logprobs 31000 && stop_server 31000
#   run_kernel_ab                    # section 4.2
#   run_fp8_arms                     # sections 4.1 and 4.3, FP8 (stage 9 config)
#   run_bf16_arms                    # sections 4.1 and 4.3, bf16 (stage 5 config)
#   GPU=0,1 run_stage8               # sections 4.1 and 4.5, stage 8 config
#   compare_arms "$OUT/fp8_baseline_c16_*/videoamme/videoamme_results.json" \
#                "$OUT/fp8_moe_triton_c16_*/videoamme/videoamme_results.json"

set -u

FP8_MODEL=marksverdhei/Qwen3-Omni-30B-A3B-FP8
BF16_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
FP8_CONFIG=examples/configs/qwen3_omni_colocated_h100_fp8.yaml
BF16_THINKER_CONFIG=examples/configs/qwen3_omni_mmmu_h100.yaml
THINKER_MAX_SEQ_LEN=32768
TOP_LOGPROBS=5
SCRIPTS="$OMNI_ROOT/tasks/qwen3_omni_0518_numerics/scripts"
RUN_BENCH="$SCRIPTS/run_bench.py"

# The sgl-omni entry point imports the installed sglang_omni package, which
# is this checkout only when it was installed editable from $OMNI_ROOT.
# pytest imports the checkout directly (pythonpath = ["."] in pyproject.toml),
# so passing unit tests do not show which code the server runs. The import
# runs from / so the current directory cannot shadow the installed package.
check_install() {
  local location
  location=$(cd / && python -c 'import os, sglang_omni; print(os.path.dirname(os.path.dirname(os.path.abspath(sglang_omni.__file__))))')
  echo "sglang_omni imports from: $location"
  echo "OMNI_ROOT:                $(cd "$OMNI_ROOT" && pwd)"
  if [ "$location" != "$(cd "$OMNI_ROOT" && pwd)" ]; then
    echo "the server would not run this checkout; run: python -m pip install -e \"$OMNI_ROOT\" --no-deps" >&2
    return 1
  fi
  (cd "$OMNI_ROOT" && git rev-parse --short HEAD)
}

# Unit tests of the logprob path (runner, request data, decode stage, client,
# chat endpoint, benchmark records). The runner and request builder tests
# import sglang, so they only run here.
run_unit_tests() {
  cd "$OMNI_ROOT" && python -m pytest -q \
    tests/unit_test/model_runner/test_rollout_logprobs.py \
    tests/unit_test/scheduling/test_request_data.py \
    tests/unit_test/qwen3_omni/test_pipeline.py \
    tests/unit_test/qwen3_omni/test_request_builder_text_only.py \
    tests/unit_test/qwen3_omni/test_token_logprobs.py \
    tests/unit_test/qwen3_omni/test_streaming.py \
    tests/unit_test/qwen3_tts/test_pipeline.py \
    tests/unit_test/client/test_completion_rollout.py \
    tests/unit_test/serve/test_chat_logprobs.py \
    tests/unit_test/serve/test_generate_rollout.py \
    tests/unit_test/serve/test_openai_api.py \
    tests/unit_test/benchmarks/test_token_logprobs.py
}

serve_fp8_colocated() {
  local port=$1
  shift
  cd "$OMNI_ROOT" && mkdir -p "$OUT/logs" \
  && CUDA_VISIBLE_DEVICES="$GPU" nohup sgl-omni serve \
      --model-path "$FP8_MODEL" --host 127.0.0.1 --port "$port" --model-name qwen3-omni \
      --config "$FP8_CONFIG" --colocate \
      --preprocessing.factory.max_seq_len "$THINKER_MAX_SEQ_LEN" \
      --thinker.factory.max_seq_len "$THINKER_MAX_SEQ_LEN" \
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

# One chat request with logprobs against a running server. Prints the text,
# then one line per generated token: token, logprob, top alternatives.
smoke_logprobs() {
  local port=$1
  curl -s "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-omni","messages":[{"role":"user","content":"Reply with only the letter B."}],"max_tokens":4,"temperature":0.0,"logprobs":true,"top_logprobs":3}' \
  | python -c '
import json, sys
body = json.load(sys.stdin)
choice = body["choices"][0]
print(repr(choice["message"]["content"]))
if "logprobs" not in choice:
    print("no logprobs key in the choice (keys: %s): the server runs a sglang_omni without this branch, see check_install" % sorted(choice))
    sys.exit(1)
if choice["logprobs"] is None:
    print("logprobs is null although the request asked for it")
    sys.exit(1)
for item in choice["logprobs"]["content"]:
    top = [(t["token"], round(t["logprob"], 4)) for t in item["top_logprobs"]]
    print(repr(item["token"]), round(item["logprob"], 4), top)
'
}

# Video-AMME, stage 9 settings (50 clips, 2 fps, 128 frames, 401408 pixels).
bench_amme() {
  local port=$1 arm=$2 conc=$3
  cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python "$RUN_BENCH" videoamme \
    --port "$port" --out "$OUT/$arm" --concurrency "$conc" --top-logprobs "$TOP_LOGPROBS"
}

# MMSU text only, stage 5 settings (mmsu-ci-2000).
bench_mmsu() {
  local port=$1 arm=$2 conc=$3
  cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python "$RUN_BENCH" mmsu \
    --port "$port" --out "$OUT/$arm" --concurrency "$conc" --top-logprobs "$TOP_LOGPROBS"
}

# Video-MME with the talker, stage 8 settings (20 clips, speech output, no inline WER).
bench_mme_talker() {
  local port=$1 arm=$2 conc=$3
  cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python "$RUN_BENCH" videomme-talker \
    --port "$port" --out "$OUT/$arm" --concurrency "$conc" --top-logprobs "$TOP_LOGPROBS"
}

# Arms of section 4.3 on the FP8 colocated server (stage 9 config). The
# baseline arm is the section 4.1 reproduction. Each arm restarts the server
# and runs Video-AMME once at c1 and three times at c16.
run_fp8_arms() {
  local port=31000
  for arm in baseline deepgemm_off moe_triton attn_triton audio_graph_off fp32_lm_head; do
    case $arm in
      baseline)        serve_fp8_colocated $port ;;
      deepgemm_off)    SGLANG_ENABLE_JIT_DEEPGEMM=0 serve_fp8_colocated $port ;;
      moe_triton)      serve_fp8_colocated $port --thinker.engine.moe_runner_backend triton ;;
      attn_triton)     serve_fp8_colocated $port --thinker.engine.attention_backend triton ;;
      audio_graph_off) serve_fp8_colocated $port --audio_encoder.factory.enable_layer_cuda_graph false ;;
      fp32_lm_head)    serve_fp8_colocated $port --thinker.engine.enable_fp32_lm_head true ;;
    esac || return 1
    backend_lines "$OUT/logs/serve_fp8_$port.log" > "$OUT/logs/backend_fp8_$arm.txt"
    bench_amme $port "fp8_${arm}_c1" 1
    for i in 1 2 3; do bench_amme $port "fp8_${arm}_c16_$i" 16; done
    stop_server $port
  done
}

# Arms of section 4.3 on the bf16 thinker-only server (stage 5 config).
run_bf16_arms() {
  local port=31001
  for arm in baseline attn_triton moe_flashinfer_cutlass fp32_lm_head; do
    case $arm in
      baseline)               serve_bf16_thinker $port ;;
      attn_triton)            serve_bf16_thinker $port --thinker.engine.attention_backend triton ;;
      moe_flashinfer_cutlass) serve_bf16_thinker $port --thinker.engine.moe_runner_backend flashinfer_cutlass ;;
      fp32_lm_head)           serve_bf16_thinker $port --thinker.engine.enable_fp32_lm_head true ;;
    esac || return 1
    backend_lines "$OUT/logs/serve_bf16_thinker_$port.log" > "$OUT/logs/backend_bf16_$arm.txt"
    bench_mmsu $port "bf16_${arm}_c1" 1
    bench_mmsu $port "bf16_${arm}_c16" 16
    stop_server $port
  done
}

# Stage 8 config (bf16 disagg, two GPUs): three c16 attempts and the RTF sample.
run_stage8() {
  local port=31002
  serve_bf16_disagg $port || return 1
  backend_lines "$OUT/logs/serve_bf16_disagg_$port.log" > "$OUT/logs/backend_bf16_disagg.txt"
  for i in 1 2 3; do bench_mme_talker $port "bf16_disagg_c16_$i" 16; done
  run_rtf_sample $port
  stop_server $port
}

# Stage 8 RTF sample: the first Video-MME clip (001-1) alone, ten times (section 4.4).
run_rtf_sample() {
  local port=$1
  for i in $(seq 1 10); do
    cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python "$RUN_BENCH" videomme-talker \
      --port "$port" --out "$OUT/rtf_001_1_$i" --concurrency 1 --max-samples 1 --top-logprobs "$TOP_LOGPROBS"
  done
}

# Kernel-level A/B (section 4.2): two runs for determinism, then the backend
# pairs inside one run. AB_VIDEO may point at one CI clip for the hf_processor case.
run_kernel_ab() {
  local ab="$SCRIPTS/kernel_ab.py"
  mkdir -p "$OUT/ab"
  cd "$OMNI_ROOT" \
  && CUDA_VISIBLE_DEVICES="$GPU" python "$ab" run --seed 0 --out "$OUT/ab/run1.pt" --model-path "$FP8_MODEL" ${AB_VIDEO:+--video "$AB_VIDEO"} \
  && CUDA_VISIBLE_DEVICES="$GPU" python "$ab" run --seed 0 --out "$OUT/ab/run2.pt" --model-path "$FP8_MODEL" ${AB_VIDEO:+--video "$AB_VIDEO"} \
  && python "$ab" compare "$OUT/ab/run1.pt" "$OUT/ab/run2.pt" | tee "$OUT/ab/determinism.txt" \
  && python "$ab" pairs "$OUT/ab/run1.pt" | tee "$OUT/ab/backend_pairs.txt"
}

# Backend and capture lines from a server log (which kernels actually ran).
backend_lines() {
  grep -E "Configured SGLang backend policy|Config file not found|Down MoE config|audio layer CUDA graphs|Capture target decode CUDA graph begin|deferred finalize is|staying eager|Code2Wav CUDA graph runner" "$1" | cut -c1-400
}

# Compare two arms per sample (section 4.3 readout), one result file per
# attempt. The first glob is the reference arm, the second the arm under test.
compare_arms() {
  local ref_glob=$1 test_glob=$2
  python "$SCRIPTS/ci_artifacts.py" compare-local \
    --pre "$(ls $ref_glob | paste -sd, -)" --post "$(ls $test_glob | paste -sd, -)"
}
