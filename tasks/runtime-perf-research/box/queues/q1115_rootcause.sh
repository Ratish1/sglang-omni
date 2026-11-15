#!/usr/bin/env bash
# 11-15, node-0000, two H200 (card 0 NUMA 0, card 1 NUMA 1): why the folded #3451 head (FOLD 7342cfcff) raises the c16
# inter chunk mean. Hop probe with talker step records on every boot, chunk lengths recorded.
#   P: colocated same-time pairs, one server per card on its own socket, then swapped: main against FOLD, FIX against
#      FOLD, and FOLD against FOLD for the card and socket band. SeedTTS c16, warm and events.
#   T: the two-card layout (thinker on card 0, talker and code2wav on card 1, no time slicing between them): main, FIX,
#      FOLD (its two-card start is the 5-chunk partial start) and FOLD with the one-chunk start; c16 and c64.
set -u
O=/data/rp1115
K=/workspace/kit2/tasks/runtime-perf-research/box
MAIN=/workspace/main FIX=/workspace/fix FOLD=/workspace/fold
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167
CPUS1=56-111,168-223
ONE_CHUNK="--talker_ar.factory.enable_partial_start true --talker_ar.factory.enable_talker_start_topology true"
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "setup done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1115_rootcause start"
note fill $FOLD "card 0 boot only"
taskset -c $CPUS0 python3 $K/baseline.py --tree $FOLD --model omni_colocated --card 0 --port 28400 --out $O/fill \
  --loads 1 --passes none > $O/fill.driver.log 2>&1
log "boot fill rc $?"
colocated() {
  local name=$1 tree=$2 card=$3 cpus=$4 port=$5
  note $name $tree "colocated c16 card $card probe"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 16 --passes warm,events --omni-chunk-audio --probe --serve-env "OMNI_HOP_PROBE_TALKER_BATCHES=1" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pair() {
  colocated $1 $2 0 $CPUS0 28400 &
  local first=$!
  colocated $3 $4 1 $CPUS1 30400 &
  local second=$!
  wait $first $second
}
pair p1_m0 $MAIN p1_f1 $FOLD
pair p2_f0 $FOLD p2_m1 $MAIN
pair p3_x0 $FIX p3_f1 $FOLD
pair p4_f0 $FOLD p4_x1 $FIX
pair p5_f0 $FOLD p5_f1 $FOLD
python3 $K/chunk_gap_parts.py --cell c16_events $O/p1_m0 $O/p2_m1 $O/p3_x0 $O/p4_x1 $O/p1_f1 $O/p2_f0 $O/p3_f1 $O/p4_f0 \
  $O/p5_f0 $O/p5_f1 > $O/chunk_gap_parts_P.txt 2>&1
log "chunk gap parts P rc $?"
twocard() {
  local name=$1 tree=$2 extra=$3
  note $name $tree "two-card c16 c64 probe $extra"
  python3 $K/baseline.py --tree $tree --model omni_speech --port 28400 --out $O/$name --loads 16,64 --passes warm,events \
    --omni-chunk-audio --probe --serve-env "OMNI_HOP_PROBE_TALKER_BATCHES=1" --serve-extra "$extra" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
twocard t_m $MAIN ""
twocard t_x $FIX ""
twocard t_f $FOLD ""
twocard t_f1 $FOLD "$ONE_CHUNK"
for cell in c16_events c64_events; do
  python3 $K/chunk_gap_parts.py --cell $cell $O/t_m $O/t_x $O/t_f $O/t_f1 > $O/chunk_gap_parts_T_$cell.txt 2>&1
done
log "q1115_rootcause done"
