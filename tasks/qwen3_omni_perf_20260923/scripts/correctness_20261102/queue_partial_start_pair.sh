set -u
# 11-02 colocated partial start against latest main: one server per arm from one tree, same time on cards 0 and 1, then
# swapped, at c16, c64 and c1 on the streamed full SeedTTS EN corpus, request events on, each boot scored on its card.
# Usage: queue_partial_start_pair.sh OUT TREE
OUT=$1
TREE=$2
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
ARM_ARGS="--talker_ar.factory.enable_partial_start true"
PORT=16000
mkdir -p $OUT
log() { echo "$(date -u +%T) $*" >> $OUT/queue.log; }
boot() {
  local dir=$1 card=$2 port=$3 conc=$4 extra=$5 cpus
  # note (ratish): both cards sit on NUMA node 1; each arm gets the same number of its CPUs, none shared.
  if [ "$card" = 0 ]; then cpus=56-69,168-181; else cpus=70-83,182-195; fi
  setsid env PIN_CPUS=$cpus PIN_NODE=1 EVENTS=1 SCORE=1 SGLANG_OMNI_STRICT_PORT=1 EXTRA_SERVE_ARGS="$extra" \
    bash $S/run_bench_boot.sh $TREE $dir $card $port h200 $conc seedtts_en > $dir.boot.log 2>&1 &
  echo $!
}
log "pair start: tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8); tools $(git -C /data/tools rev-parse --short HEAD)"
for conc in 16 64 1; do
  for order in 0 1; do
    round=c${conc}_o${order}
    mkdir -p $OUT/$round
    main_card=$order
    arm_card=$((1 - order))
    log "$round start: main card $main_card, partial start card $arm_card"
    boot $OUT/$round/main $main_card $PORT $conc "" > /dev/null
    boot $OUT/$round/partial $arm_card $((PORT + 300)) $conc "$ARM_ARGS" > /dev/null
    until grep -q '^done ' $OUT/$round/main/progress.txt 2>/dev/null && grep -q '^done ' $OUT/$round/partial/progress.txt 2>/dev/null; do
      sleep 10
    done
    for arm in main partial; do
      log "  $round $arm: $(grep -E 'gen seedtts_en rc|FAILED' $OUT/$round/$arm/progress.txt | cut -c1-80); $(python3 $S/state_durations.py $OUT/$round/$arm 2>/dev/null | tail -1 | cut -c1-240)"
    done
    PORT=$((PORT + 1000))
  done
done
log "pair done"
