#!/usr/bin/env bash
# 11-15, node-0000: #3451's final head (FINAL 17353ed22) against main 35890f10c for the PR body. Colocated c16 streamed
# and c16 not streamed, same-time pairs swapped across cards; then the final head on the two-card layout at c16 and c64,
# read against main's two-card boot from this lease (t_m).
set -u
O=/data/rp1115/final
K=/workspace/kit2/tasks/runtime-perf-research/box
FINAL=/workspace/final MAIN=/workspace/main
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
  local name=$1 tree=$2 card=$3 stream=$4
  local cpus=$CPUS0 port=28400
  if [ $card = 1 ]; then cpus=$CPUS1; port=30400; else :; fi
  local mode=""
  if [ $stream = off ]; then mode="--no-stream"; else :; fi
  note $name $tree "colocated c16 stream $stream card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 16 --passes warm,clean --omni-chunk-audio $mode > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
log "q1115_final start"
quiet_host
both "colocated s1_f0 $FINAL 0 on" "colocated s1_m1 $MAIN 1 on"
quiet_host
both "colocated s2_m0 $MAIN 0 on" "colocated s2_f1 $FINAL 1 on"
quiet_host
both "colocated n1_f0 $FINAL 0 off" "colocated n1_m1 $MAIN 1 off"
quiet_host
both "colocated n2_m0 $MAIN 0 off" "colocated n2_f1 $FINAL 1 off"
python3 $K/body_table.py $O c16 "c16 streamed" main=s1_m1,s2_m0 final=s1_f0,s2_f1 >> $O/queue.log 2>&1
python3 $K/body_table.py $O c16 "c16 not streamed" main=n1_m1,n2_m0 final=n1_f0,n2_f1 >> $O/queue.log 2>&1
quiet_host
note t_final $FINAL "two-card c16 c64"
python3 $K/baseline.py --tree $FINAL --model omni_speech --port 28400 --out $O/t_final --loads 16,64 --passes warm,clean \
  --omni-chunk-audio > $O/t_final.driver.log 2>&1
log "boot t_final rc $?"
log "q1115_final done"
