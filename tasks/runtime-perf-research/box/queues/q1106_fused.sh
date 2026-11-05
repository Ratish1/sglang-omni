#!/usr/bin/env bash
# 11-06, node-0000, after q1106_bucket: the text-only pipeline runs preprocessing in one process with the encoders and thinker. So #3452's OpenMP defaults and video threads
# reach the thinker's process there, which no other row reads. Text Video-MME on the CI thinker config, then MMSU at the CI's c96 and at c16, against main at the same time.
set -u
O=/data/rp1106
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main L=/workspace/lane
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1106_bucket done" $O/queue.log 2>/dev/null; do sleep 15; done
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 load $(tail -1 $O/cpu_load.log) start"; }
video() {
  local name=$1 tree=$2 card=$3
  note $name $tree "fused text video card $card"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_text_fused --card $card --port $(port $card) --out $O/$name \
    --bench videomme_text --loads 16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
mmsu() {
  local name=$1 tree=$2 card=$3
  note $name $tree "fused mmsu card $card"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_text_fused --card $card --port $(port $card) --out $O/$name \
    --bench mmsu --loads 96,16 --passes warm,clean,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1106_fused start"
video fv_m1 $MAIN 0 & video fv_l1 $L 1 & wait
video fv_l2 $L 0 & video fv_m2 $MAIN 1 & wait
log "cell fv done"
mmsu fu_m1 $MAIN 0 & mmsu fu_l1 $L 1 & wait
mmsu fu_l2 $L 0 & mmsu fu_m2 $MAIN 1 & wait
log "q1106_fused done"
