#!/usr/bin/env bash
# One same time A/B pair (arms start each cell together, ab_boot_sync.sh): arm A on card 0 and arm B on card 1 (ab_boot.sh, each pinned to its own
# NUMA 0 cores, GPU core dumps armed), then WER and speaker similarity of each arm's scored cells,
# one arm after the other (two similarity scorers race on the first download of their weights).
# usage (in the container): bash ab_pair.sh <out> <tree on card 0> <tree on card 1> [port base]
#   CELLS        ab_boot.sh cells (default seeded stream c1, stream c16, buffered c16)
#   EXTRA_ENV_0  extra server environment of the card 0 arm, EXTRA_ENV_1 of the card 1 arm
#   SERVE_ARGS_0 extra serve arguments of the card 0 arm, SERVE_ARGS_1 of the card 1 arm
#   SCORE_CELLS  cells to score (default "c16 b16")
#   ROOT         run root holding cores/ and the inductor caches (default /data/p20261008)
#   CARD_0       card of the first arm (default 0), CARD_1 of the second (default 1)
#   CPUS_0       cores of the first arm (default 0-15,64-79), CPUS_1 of the second
OUT=$1
TREE0=$2
TREE1=$3
BASE=${4:-31000}
ROOT=${ROOT:-/data/p20261008}
CARD_0=${CARD_0:-0}
CARD_1=${CARD_1:-1}
CPUS_0=${CPUS_0:-0-15,64-79}
CPUS_1=${CPUS_1:-16-31,80-95}
STAGE4=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4
IN_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
mkdir -p "$ROOT/cores"
CORE="CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 CUDA_COREDUMP_FILE=$ROOT/cores/core_%h_%p_%t"
export CELLS=${CELLS:-"c1:1:64:1234:1 c16:16:::1 b16:16:::0"}
ARM0=$OUT/card${CARD_0}_$TREE0
ARM1=$OUT/card${CARD_1}_$TREE1
export PEERS="$ARM0 $ARM1"
mkdir -p "$OUT"
echo "pair start $(date +%T) card$CARD_0=$TREE0 card$CARD_1=$TREE1 cells=$CELLS" >> "$OUT/progress.txt"
EXTRA_ENV="$CORE ${EXTRA_ENV_0:-}" SERVE_ARGS="${SERVE_ARGS_0:-}" bash $IN_DIR/ab_boot_sync.sh "/workspace/$TREE0" "$ARM0" "$CARD_0" "$BASE" \
  "$CPUS_0" "$ROOT/inductor_$TREE0" &
SERVE0=$!
EXTRA_ENV="$CORE ${EXTRA_ENV_1:-}" SERVE_ARGS="${SERVE_ARGS_1:-}" bash $IN_DIR/ab_boot_sync.sh "/workspace/$TREE1" "$ARM1" "$CARD_1" "$((BASE + 200))" \
  "$CPUS_1" "$ROOT/inductor_$TREE1" &
SERVE1=$!
wait $SERVE0 $SERVE1
echo "served $(date +%T)" >> "$OUT/progress.txt"

score() {
  local arm=$1 card=$2 port=$3 cpus=$4
  cd /workspace/sglang-omni
  for cell in ${SCORE_CELLS:-c16 b16}; do
    [ -d "$arm/$cell" ] || continue
    CUDA_VISIBLE_DEVICES=$card taskset -c "$cpus" python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
      --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port "$port" --lang en \
      --output-dir "$arm/$cell" > "$arm/score_wer_$cell.log" 2>&1
    echo "wer $cell rc $? $(date +%T)" >> "$arm/progress.txt"
    CUDA_VISIBLE_DEVICES=$card taskset -c "$cpus" python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
      --similarity-only --model $MODEL --lang en --output-dir "$arm/$cell" > "$arm/score_sim_$cell.log" 2>&1
    echo "sim $cell rc $? $(date +%T)" >> "$arm/progress.txt"
  done
}
score "$ARM0" "$CARD_0" "$((BASE + 400))" "$CPUS_0"
score "$ARM1" "$CARD_1" "$((BASE + 600))" "$CPUS_1"
echo "pair done $(date +%T)" >> "$OUT/progress.txt"
