#!/usr/bin/env bash
# 11-15, node-0000, after q1115_followup: FOLDSYNC (3ccb9e882, the fold with the talker's decode buffer rebuild and its
# per-step repetition write made sync-free) against FOLD. If a changed row list costs 4 to 5 ms on one card because the
# rebuild's host syncs wait for the thinker's slice, FOLDSYNC's changed steps cost about the two-card 0.9 ms and its chunk
# gap falls; on two cards it changes little.
set -u
O=/data/rp1115
K=/workspace/kit2/tasks/runtime-perf-research/box
MAIN=/workspace/main FOLD=/workspace/fold FOLDSYNC=/workspace/foldsync
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167
CPUS1=56-111,168-223
ONE_CHUNK="--talker_ar.factory.enable_partial_start true --talker_ar.factory.enable_talker_start_topology true"
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1115_followup done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1115_syncfree start"
colocated() {
  local name=$1 tree=$2 card=$3 load=$4
  local cpus=$CPUS0 port=28400
  if [ $card = 1 ]; then cpus=$CPUS1; port=30400; else :; fi
  note $name $tree "colocated c$load card $card probe"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads $load --passes warm,events --omni-chunk-audio --probe --serve-env "OMNI_HOP_PROBE_TALKER_BATCHES=1" \
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
  local first="$1" second="$2"
  eval "$first" &
  local a=$!
  eval "$second" &
  local b=$!
  wait $a $b
}
both "colocated s1_s0 $FOLDSYNC 0 16" "colocated s1_f1 $FOLD 1 16"
both "colocated s2_f0 $FOLD 0 16" "colocated s2_s1 $FOLDSYNC 1 16"
both "colocated s3_s0 $FOLDSYNC 0 64" "colocated s3_f1 $FOLD 1 64"
both "colocated s4_f0 $FOLD 0 64" "colocated s4_s1 $FOLDSYNC 1 64"
both "seeded id_s $FOLDSYNC 0" "seeded id_m $MAIN 1"
log "identity $(python3 $K/wav_identity.py $O c1_clean id_m id_s 2>&1 | tail -1)"
python3 $K/chunk_gap_parts.py --cell c16_events $O/s1_s0 $O/s2_s1 $O/s1_f1 $O/s2_f0 > $O/chunk_gap_parts_S16.txt 2>&1
python3 $K/chunk_gap_parts.py --cell c64_events $O/s3_s0 $O/s4_s1 $O/s3_f1 $O/s4_f0 > $O/chunk_gap_parts_S64.txt 2>&1
twocard() {
  local name=$1 tree=$2 extra=$3
  note $name $tree "two-card c16 c64 probe $extra"
  python3 $K/baseline.py --tree $tree --model omni_speech --port 28400 --out $O/$name --loads 16,64 --passes warm,events \
    --omni-chunk-audio --probe --serve-env "OMNI_HOP_PROBE_TALKER_BATCHES=1" --serve-extra "$extra" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
twocard t_s $FOLDSYNC ""
twocard t_s1 $FOLDSYNC "$ONE_CHUNK"
log "q1115_syncfree done"
