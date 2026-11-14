#!/usr/bin/env bash
# run_head.sh <commit> <card> [batch tiles]: check and stamps of one dev-branch head on one card,
# inside the container. Each step has a timeout, since a handoff bug shows up as a hang.
set -u
HEAD=$1
CARD=$2
BATCH_TILES=${3:-1}
B=/data/qwen3-tts-synthetic-20261114
R=$B/predictor_bench
T=$B/heads/$HEAD
K=$B/kit/tasks/qwen3_tts_synthetic_profile_20261114
cd $B/persistent && git fetch -q origin dev/qwen3-tts-predictor-persistent-20261114
[ -d $T ] || git worktree add -q --detach $T $HEAD
cd $T && echo "head $(git rev-parse --short HEAD) tree $(git rev-parse HEAD:sglang_omni)"
cd $K
export CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$T
timeout 600 python3 persistent_layer_check.py --out $R/h200_${HEAD}_check.json > $R/h200_${HEAD}_check.log 2>&1
echo "check exit $?"
timeout 600 python3 persistent_stamps.py --batch-tiles $BATCH_TILES --out $R/h200_${HEAD}_stamps.json > $R/h200_${HEAD}_stamps.log 2>&1
echo $? > $R/h200_${HEAD}.done
