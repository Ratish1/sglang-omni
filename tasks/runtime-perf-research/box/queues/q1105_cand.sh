#!/usr/bin/env bash
# 11-05, node-0000, after q1105_a: #3452's candidate (the final form, hash and local audio off the loop, passive OpenMP in
# preprocessing) against main. Colocated speech at c64 and c16, Video-MME text-only and with audio, same time, each server on its
# card's socket, both card orders. Then two-card speech alone, on the full host and on one socket, where main's team does not spin.
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 C=/workspace/wk_fix
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1105_a done" $O/queue.log 2>/dev/null; do sleep 15; done
speech() {
  local name=$1 tree=$2 card=$3
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) seedtts card $card start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video() {
  local name=$1 tree=$2 card=$3 bench=$4
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) $bench card $card start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --bench $bench --loads 16 --passes clean,events --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
twocard() {
  local name=$1 tree=$2 cpuset=$3
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) two card cpus $cpuset start"
  taskset -c $cpuset python3 $K/baseline.py --tree $tree --model omni_speech --card 0 --port 28400 --out $O/$name \
    --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1105_cand start"
speech cs_m1 $MAIN 0 & speech cs_c1 $C 1 & wait
speech cs_c2 $C 0 & speech cs_m2 $MAIN 1 & wait
speech cs_m3 $MAIN 0 & speech cs_c3 $C 1 & wait
speech cs_c4 $C 0 & speech cs_m4 $MAIN 1 & wait
video ct_m1 $MAIN 0 videomme_text & video ct_c1 $C 1 videomme_text & wait
video ct_c2 $C 0 videomme_text & video ct_m2 $MAIN 1 videomme_text & wait
video cv_m1 $MAIN 0 videomme & video cv_c1 $C 1 videomme & wait
video cv_c2 $C 0 videomme & video cv_m2 $MAIN 1 videomme & wait
ALL=0-223
twocard ca_m1 $MAIN $ALL
twocard ca_c1 $C $ALL
twocard ca_c2 $C $ALL
twocard ca_m2 $MAIN $ALL
twocard co_m1 $MAIN $CPUS1
twocard co_c1 $C $CPUS1
twocard co_c2 $C $CPUS1
twocard co_m2 $MAIN $CPUS1
log "q1105_cand done"
