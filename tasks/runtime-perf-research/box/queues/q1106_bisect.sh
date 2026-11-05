#!/usr/bin/env bash
# 11-06, node-0000, after q1106_cross2. The candidate beats #3451 as it stands by 3.38 ms (se 1.05) on h200 unseeded, every pair. The bisect arm is the simpler form, the Qwen3-TTS order with no
# added wait, and has only the seeded h100_bf16 read. Read it on the cell that separates the forms: against main, then directly against the candidate.
set -u
O=/data/rp1106
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main A=/workspace/p3a S=/workspace/p3s
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1106_cross2 done" $O/queue.log 2>/dev/null; do sleep 15; done
boot() {
  local name=$1 tree=$2 card=$3
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) omni_colocated card $card load $(tail -1 $O/cpu_load.log) start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pairs() {
  local tag=$1 first=$2 second=$3
  local index=1
  while [ $index -le 4 ]; do
    if [ $((index % 2)) = 1 ]; then
      boot ${tag}_m$index $first 0 & boot ${tag}_p$index $second 1 & wait
    else
      boot ${tag}_p$index $second 0 & boot ${tag}_m$index $first 1 & wait
    fi
    index=$((index + 1))
  done
  log "cell $tag done"
}
log "q1106_bisect start"
pairs bh $MAIN $S
pairs bc $A $S
log "q1106_bisect done"
