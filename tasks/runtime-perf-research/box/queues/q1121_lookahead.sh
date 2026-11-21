#!/usr/bin/env bash
# 11-21, node-0001, after q1121_profile: a what-if, not a candidate. The stack (STACK 6149c714f) against the stack with
# speech batches on the thinker's lookahead (LOOK 26375e856, lookahead_eligible no longer refuses a speech request),
# colocated, both probes on, same-time pairs on both cards, swapped: speech at 8/s on one seeded arrival schedule (cap
# 64) beside the looping Video-MME client (4 in flight), then speech alone closed c16. In the profile's first pair the
# stack's first chunk gap is the one gap longer than main's, its talker rows sit out more, and its thinker is no slower
# per speech token: the talker reaches the thinker's text head sooner, so the thinker's own cadence sets that gap.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
STACK=/workspace/tstack LOOK=/workspace/tlook
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
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
mixed() {
  local name=$1 tree=$2 card=$3 offset=$4
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside 4 videos card $card probes"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events "--bench-extra=--request-rate=8" \
    --probe --serve-env "$PROBES" > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $cpus python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
        --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
        --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 4 \
        --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1 &
      client=$!
      while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
      kill $client 2>/dev/null
      wait $client 2>/dev/null
      pass=$((pass + 1))
    done
  ) &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}
alone() {
  local name=$1 tree=$2 card=$3 offset=$4
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  note $name $tree "seedtts c16 alone card $card probes"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events \
    --probe --serve-env "$PROBES" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1121_profile done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1121_lookahead start"
quiet_host
both "mixed la1_l0 $LOOK 0 900" "mixed la1_s1 $STACK 1 900"
quiet_host
both "mixed la2_s0 $STACK 0 1100" "mixed la2_l1 $LOOK 1 1100"
python3 $K/body_table.py $O c64 "speech lookahead against the stack, 8/s beside video" stack=la1_s1,la2_s0 look=la1_l0,la2_l1 \
  >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O la1_l0 la2_l1 la1_s1 la2_s0 >> $O/queue.log 2>&1
python3 $K/chunk_gap_parts.py --cell c64_events $O/la1_s1 $O/la2_s0 $O/la1_l0 $O/la2_l1 > $O/chunk_gap_parts_lookahead.txt 2>&1
python3 $K/thinker_cadence.py --cell c64_events $O/la1_s1 $O/la2_s0 $O/la1_l0 $O/la2_l1 > $O/thinker_cadence_lookahead.txt 2>&1
log "lookahead eq read"
quiet_host
both "alone lc1_l0 $LOOK 0 1300" "alone lc1_s1 $STACK 1 1300"
quiet_host
both "alone lc2_s0 $STACK 0 1500" "alone lc2_l1 $LOOK 1 1500"
python3 $K/body_table.py $O c16 "speech lookahead against the stack, c16 alone" stack=lc1_s1,lc2_s0 look=lc1_l0,lc2_l1 \
  >> $O/queue.log 2>&1
python3 $K/chunk_gap_parts.py --cell c16_events $O/lc1_s1 $O/lc2_s0 $O/lc1_l0 $O/lc2_l1 > $O/chunk_gap_parts_lookahead_c16.txt 2>&1
log "q1121_lookahead done"
