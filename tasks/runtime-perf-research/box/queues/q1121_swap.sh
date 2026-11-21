#!/usr/bin/env bash
# 11-21, node-0001, after q1121_both: the speech-alone pairs with the cards swapped. fe1 (graph change card 0, main card
# 1) read every row level but first audio p99 +11.5 %, and k1 runs both changes on card 0 too; card 0 carries the heavier
# first audio tail whichever arm runs on it. Main card 0 against the graph change (PFIX) card 1 at c16, then main card 0
# against both changes (PBOTH) card 1 at c1 and c16.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain PFIX=/workspace/pfix PBOTH=/workspace/pboth
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((28000 + 2000 * $1 + $2)); }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
alone() {
  local name=$1 tree=$2 card=$3 offset=$4 loads=$5
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  note $name $tree "seedtts loads $loads alone card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads $loads --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1121_both done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1121_swap start"
quiet_host
both "alone fe2_m0 $NMAIN 0 2400 16" "alone fe2_p1 $PFIX 1 2400 16"
python3 $K/body_table.py $O c16 "graph change against main, speech alone c16" main=fe1_m1,fe2_m0 graph=fe1_p0,fe2_p1 \
  >> $O/queue.log 2>&1
quiet_host
both "alone k2_m0 $NMAIN 0 2600 1,16" "alone k2_b1 $PBOTH 1 2600 1,16"
for cell in c1 c16; do
  python3 $K/body_table.py $O $cell "both changes against main, speech alone $cell" main=k1_m1,k2_m0 both=k1_b0,k2_b1 \
    >> $O/queue.log 2>&1
done
log "q1121_swap done"
