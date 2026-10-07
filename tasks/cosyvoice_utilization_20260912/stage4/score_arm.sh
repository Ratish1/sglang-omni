#!/usr/bin/env bash
# WER (the benchmark's Qwen3-ASR server on the card) and speaker similarity (WavLM) of one
# ab_boot.sh arm's c16 pass, with the main tree's benchmark code. Run it only when no
# measured pair serves, and give each card its own ASR port.
#
# usage: score_arm.sh <arm out dir> <card> <asr port>
set -u
OUT=$1 CARD=$2 PORT=$3
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
cd /workspace/sglang-omni
echo "score start $(date +%T) card=$CARD asr_port=$PORT" >> "$OUT/progress.txt"
CUDA_VISIBLE_DEVICES=$CARD python3 -u -m benchmarks.eval.benchmark_tts_seedtts --transcribe-only \
  --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port "$PORT" --lang en \
  --output-dir "$OUT/c16" > "$OUT/score_wer.log" 2>&1
echo "wer rc $? $(date +%T)" >> "$OUT/progress.txt"
CUDA_VISIBLE_DEVICES=$CARD python3 -u -m benchmarks.eval.benchmark_tts_seedtts --similarity-only \
  --model $MODEL --lang en --output-dir "$OUT/c16" > "$OUT/score_sim.log" 2>&1
echo "sim rc $? $(date +%T)" >> "$OUT/progress.txt"
