#!/usr/bin/env bash
# After queue_snake_fp32_serving.sh: does SIM move with a vocoder's distance to the fp32 decoder? sim_truth_probe.py
# streams every SeedTTS EN reference clip's codes through main (with the fp32 truth), #3097 and #3097 with the fp32
# SnakeBeta as served; the SeedTTS scorer scores each arm against its clips; then each arm's paired SIM against the
# truth's and against main's.
set -u
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
P=$T/tasks/qwen3-tts-perf/box/chain_20261022/sim_truth_probe.py
OUT=$D/sim_truth
M=${M:?the main tree}
P2=${P2:?the 3097 tree}
PF=${PF:?the fix tree}
L=$D/queue.log
until grep -q queue_snake_fp32_serving_done $L; do sleep 20; done
echo "queue sim truth start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
probe() {
  local arm=$1 tree=$2 card=$3
  shift 3
  (cd $tree && TORCHINDUCTOR_CACHE_DIR=$D/inductor_sim_$arm PYTHONPATH=$tree timeout 3600 python3 $P run $OUT $arm --card $card "$@" \
    > $D/sim_truth_$arm.txt 2>&1)
  echo "sim truth $arm rc $? $(tail -n 1 $D/sim_truth_$arm.txt) $(date +%T)" >> $L
}
score() {
  local arm=$1 card=$2
  (cd $M && PYTHONPATH=$M SGLANG_OMNI_STRICT_PORT=1 python3 -m benchmarks.eval.benchmark_tts_seedtts \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --lang en --skip-gpu-cleanup --output-dir $OUT/$arm --similarity-only \
    --device cuda:$card > $D/sim_truth_score_$arm.txt 2>&1)
  echo "sim truth score $arm rc $? $(date +%T)" >> $L
}
probe main $M 2 --truth & probe fix $PF 3 & wait
probe d2 $P2 2 & score truth 3 & wait
score main 2 & score fix 3 & wait
score d2 2
(cd $M && PYTHONPATH=$M python3 $P compare $OUT main fix d2 > $D/sim_truth_compare.txt 2>&1)
echo "queue_sim_truth_done $(date +%T)" >> $L
