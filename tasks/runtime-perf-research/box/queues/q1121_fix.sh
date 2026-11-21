#!/usr/bin/env bash
# 11-21, node-0001, after q1121_fixunits: the graph prefill change against its bases, same-time pairs on both cards,
# swapped. A video or image request whose chunk computes none of its visual rows (a prompt the radix cache holds, a text
# tail) ran the eager forward, about 70 ms with every speech row of its batch inside; it now replays the prefill graph.
#   A  stack with the change (NSFIX) against the stack (NSTACK), speech 8/s beside 4 looping videos, probes on
#   B  the change alone (PFIX) against main (NMAIN), the same cell
#   C  CI text Video-MME, the CI speech pipeline on one card, c1 and c16, two repeats (the second hits the radix cache)
#   D  stack with the change against the stack, closed c16 beside 8 videos
#   E  the change alone against main, speech alone c16 (no request carries visual inputs: the control)
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain PFIX=/workspace/pfix NSTACK=/workspace/nstack NSFIX=/workspace/nsfix
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_THINKER_PARTS=1"
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
  local name=$1 tree=$2 card=$3 offset=$4 loads=$5 videos=$6 bench=$7 probe=$8
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  local probe_args=()
  if [ $probe = probe ]; then probe_args=(--probe --serve-env "$PROBES"); else :; fi
  note $name $tree "seedtts loads $loads $bench beside $videos videos card $card $probe"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port \
    --out $O/$name --loads $loads --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events $bench \
    "${probe_args[@]}" > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $cpus python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
        --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
        --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency $videos \
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
video_text() {
  local name=$1 tree=$2 card=$3 offset=$4
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  note $name $tree "omni_ci_onecard text video c1 c16 two repeats card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card $card --port $port --out $O/$name \
    --bench videomme_text --loads 1,16 --passes clean,events --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
alone() {
  local name=$1 tree=$2 card=$3 offset=$4
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  note $name $tree "seedtts c16 alone card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1121_fixunits done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1121_fix start"
EQ="--bench-extra=--request-rate=8"
quiet_host
both "mixed fa1_f0 $NSFIX 0 200 64 4 $EQ probe" "mixed fa1_s1 $NSTACK 1 200 64 4 $EQ probe"
quiet_host
both "mixed fa2_s0 $NSTACK 0 400 64 4 $EQ probe" "mixed fa2_f1 $NSFIX 1 400 64 4 $EQ probe"
python3 $K/body_table.py $O c64 "stack with the change against the stack, 8/s beside video" stack=fa1_s1,fa2_s0 \
  change=fa1_f0,fa2_f1 >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O fa1_f0 fa2_f1 fa1_s1 fa2_s0 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O stack=fa1_s1,fa2_s0 change=fa1_f0,fa2_f1 > $O/fix_first_gap_stack.txt 2>&1
python3 $K/thinker_parts.py --cell c64_events $O/fa1_s1 $O/fa2_s0 $O/fa1_f0 $O/fa2_f1 > $O/fix_parts_stack.txt 2>&1
log "A read"
quiet_host
both "mixed fb1_p0 $PFIX 0 600 64 4 $EQ probe" "mixed fb1_m1 $NMAIN 1 600 64 4 $EQ probe"
quiet_host
both "mixed fb2_m0 $NMAIN 0 800 64 4 $EQ probe" "mixed fb2_p1 $PFIX 1 800 64 4 $EQ probe"
python3 $K/body_table.py $O c64 "change against main, 8/s beside video" main=fb1_m1,fb2_m0 change=fb1_p0,fb2_p1 \
  >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O fb1_p0 fb2_p1 fb1_m1 fb2_m0 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O main=fb1_m1,fb2_m0 change=fb1_p0,fb2_p1 > $O/fix_first_gap_main.txt 2>&1
python3 $K/thinker_parts.py --cell c64_events $O/fb1_m1 $O/fb2_m0 $O/fb1_p0 $O/fb2_p1 > $O/fix_parts_main.txt 2>&1
log "B read"
quiet_host
both "video_text fc1_p0 $PFIX 0 1000" "video_text fc1_m1 $NMAIN 1 1000"
quiet_host
both "video_text fc2_m0 $NMAIN 0 1200" "video_text fc2_p1 $PFIX 1 1200"
for cell in c1 c16; do
  python3 $K/video_reps.py $O ${cell}_clean "text Video-MME $cell" fc1_m1,fc2_m0 fc1_p0,fc2_p1 >> $O/queue.log 2>&1
done
python3 $K/videomme_text_identity.py --passes "main_c1=$O/fc1_m1/c1_clean/bench/rep*/videomme_results.json" \
  "main2_c1=$O/fc2_m0/c1_clean/bench/rep*/videomme_results.json" \
  "change_c1=$O/fc1_p0/c1_clean/bench/rep*/videomme_results.json" \
  "change2_c1=$O/fc2_p1/c1_clean/bench/rep*/videomme_results.json" > $O/fix_video_identity_c1.txt 2>&1
log "C read"
quiet_host
both "mixed fd1_f0 $NSFIX 0 1600 16 8 '' noprobe" "mixed fd1_s1 $NSTACK 1 1600 16 8 '' noprobe"
quiet_host
both "mixed fd2_s0 $NSTACK 0 1800 16 8 '' noprobe" "mixed fd2_f1 $NSFIX 1 1800 16 8 '' noprobe"
python3 $K/body_table.py $O c16 "stack with the change against the stack, closed c16 beside 8 videos" stack=fd1_s1,fd2_s0 \
  change=fd1_f0,fd2_f1 >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O fd1_f0 fd2_f1 fd1_s1 fd2_s0 >> $O/queue.log 2>&1
log "D read"
quiet_host
both "alone fe1_p0 $PFIX 0 2000" "alone fe1_m1 $NMAIN 1 2000"
python3 $K/body_table.py $O c16 "change against main, speech alone c16" main=fe1_m1 change=fe1_p0 >> $O/queue.log 2>&1
log "q1121_fix done"
