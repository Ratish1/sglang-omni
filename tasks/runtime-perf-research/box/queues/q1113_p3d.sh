#!/usr/bin/env bash
# 11-13, node-0000, after q1113_gpu2: #3451's final tree plus the talker decode masks written without a host sync (D). The probe
# found talker.py:1065 syncing on every reuse decode step and :1223, :1226, :1249 on full prepares, and the traces show the card
# switching 1,160 times a second. D's trace against the final tree's, its seeded c1 identity against main, and c16 boots s d d s.
set -u
O=/data/rp1113
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main P3S=/workspace/p3s P3D=/workspace/p3d
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 load $(tail -1 $O/cpu_load.log) start"; }
until grep -q "q1113_gpu2 done" $O/queue.log 2>/dev/null; do sleep 15; done
log "q1113_p3d start"
note tr_d $P3D "nsys gpuctxsw c16"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $P3D --model omni_colocated --card 0 --port 28400 --out $O/tr_d \
  --loads 16 --passes warm,nsys --nsys-session tr_d --nsys-trace cuda,nvtx,osrt --nsys-graph-trace node --nsys-gpuctxsw \
  --nsys-delay 20 --nsys-seconds 15 > $O/tr_d.driver.log 2>&1
log "boot tr_d rc $?"
note id_d $P3D "seeded c1"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $P3D --model omni_colocated --card 0 --port 28400 --out $O/id_d \
  --loads 1 --passes clean "--bench-extra=--seed 3451" > $O/id_d.driver.log 2>&1
log "boot id_d rc $?"
speech16() {
  local name=$1 tree=$2
  note $name $tree "c16"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech16 d16_s1 $P3S
speech16 d16_d1 $P3D
speech16 d16_d2 $P3D
speech16 d16_s2 $P3S
log "q1113_p3d done"
