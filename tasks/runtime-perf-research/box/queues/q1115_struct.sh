#!/usr/bin/env bash
# 11-15, node-0000: the restructured talker scheduler (STRUCT 7982b003a, tree 026f127dd410: one readiness split per
# step, parked rows, abort through the existing path, dense feedback write, step-count log) against #3451's pushed head
# (FOLDSYNC tree 48f296ad09d7 = 200783e92). Seeded c1 audio must be byte-identical; c16 same-time pairs, swapped.
set -u
O=/data/rp1115/struct
K=/workspace/kit2/tasks/runtime-perf-research/box
STRUCT=/workspace/struct FOLDSYNC=/workspace/foldsync
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167
CPUS1=56-111,168-223
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
colocated() {
  local name=$1 tree=$2 card=$3
  local cpus=$CPUS0 port=28400
  if [ $card = 1 ]; then cpus=$CPUS1; port=30400; else :; fi
  note $name $tree "colocated c16 card $card probe"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 16 --passes warm,events --omni-chunk-audio --probe --serve-env "OMNI_HOP_PROBE_TALKER_BATCHES=1" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
seeded() {
  local name=$1 tree=$2 card=$3
  local cpus=$CPUS0 port=28400
  if [ $card = 1 ]; then cpus=$CPUS1; port=30400; else :; fi
  note $name $tree "seeded c1 card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 1 --passes clean "--bench-extra=--seed 3451" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
  (cd $O/$name/c1_clean/bench/audio && find . -name "*.wav" | sort | xargs sha256sum > $O/$name/wav_sha256.txt)
}
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
log "q1115_struct start"
quiet_host
both "seeded id_t $STRUCT 0" "seeded id_s $FOLDSYNC 1"
log "identity $(python3 $K/wav_identity.py $O c1_clean id_s id_t 2>&1 | tail -1)"
quiet_host
both "colocated t1_t0 $STRUCT 0" "colocated t1_s1 $FOLDSYNC 1"
quiet_host
both "colocated t2_s0 $FOLDSYNC 0" "colocated t2_t1 $STRUCT 1"
python3 $K/chunk_gap_parts.py --cell c16_events $O/t1_t0 $O/t2_t1 $O/t1_s1 $O/t2_s0 > $O/chunk_gap_parts_T16.txt 2>&1
python3 $K/body_table.py $O c16 "restructure c16" head=t1_s1,t2_s0 struct=t1_t0,t2_t1 >> $O/queue.log 2>&1
for name in t1_t0 t2_t1; do
  log "$name parked log lines $(grep -c 'parked for thinker text' $O/$name/serve.log), errors $(grep -cE 'Traceback|ERROR' $O/$name/serve.log)"
done
log "q1115_struct done"
