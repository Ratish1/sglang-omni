#!/usr/bin/env bash
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
NV=/workspace/marks/tasks/qwen3_omni_perf_20260923/scripts/omni_pipeline_nvtx/sitecustomize.py
MAIN=/workspace/main
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
log "boot sy_smoke $(git -C $MAIN rev-parse --short HEAD) $(git -C $MAIN rev-parse --short=12 HEAD:sglang_omni) synthetic smoke start"
taskset -c 56-111,168-223 python3 $K/baseline.py --tree $MAIN --model omni_colocated --card 0 --port 28400 --out $O/sy_smoke \
  --loads 1 --passes none --serve-extra "$CONTEXT_32K" \
  --serve-env "OMNI_SYNTH_IGNORE_EOS=1 OMNI_PIPE_NVTX=1 OMNI_SITECUSTOMIZE_CHAIN=$NV PYTHONPATH=$K/synth_eos_probe:$MAIN" \
  --after "taskset -c 56-111,168-223 python3 $K/synth_regimes.py --port 28400 --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --out $O/sy_smoke/synth --regimes decode_c1,speech_c1,audio_1s_c16,image_c16,video_128f" \
  > $O/sy_smoke.driver.log 2>&1
log "boot sy_smoke rc $?"
