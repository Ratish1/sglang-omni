#!/usr/bin/env bash
# 11-15, node-0000, two H200s: #3452 rebuilt on main 35890f10c (LANE ecfebb1f3: requests read once, one run path for both lanes,
# Exception in place of BaseException) against main. First the stage alone, CPU only: main's serial stage and the lane's serial
# and 8-thread stage hash every output of 256 requests (Video-MME CI videos, speech, one request per input form), OMP 14 on both.
# Then same-time pairs, one card each on its own socket, swapped: text Video-MME CI c16 colocated (the video lane) and MMSU CI
# on the fused text pipeline (the serial lane).
set -u
O=/data/v1115
K=/workspace/kit2/tasks/runtime-perf-research/box
LANE=/workspace/vlane MAIN=/workspace/main
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167
CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
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
until grep -q "== done" $O/unit.log 2>/dev/null; do sleep 15; done
log "q1115_vlane start"
quiet_host

identity() {
  local name=$1 tree=$2 threads=$3 against=$4
  note $name $tree "stage identity threads $threads"
  (cd $tree && PYTHONPATH=$tree OMP_NUM_THREADS=14 taskset -c $CPUS0 python3 $K/preprocess_shared_identity.py \
    --threads $threads --image $O/form_image.png --out $O/$name.json $against > $O/$name.log 2>&1)
  log "boot $name rc $?"
}
identity id_main $MAIN 1 ""
identity id_lane $LANE 8 "--against $O/id_main.json"
log "identity done"

serve() {
  local name=$1 tree=$2 card=$3 model=$4 bench=$5
  local cpus=$CPUS0 port=28400
  if [ $card = 1 ]; then cpus=$CPUS1; port=30400; else :; fi
  note $name $tree "$model $bench card $card"
  if [ $bench = videomme_text ]; then
    taskset -c $cpus python3 $K/baseline.py --tree $tree --model $model --card $card --port $port --out $O/$name \
      --bench videomme_text --loads 16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  else
    taskset -c $cpus python3 $K/baseline.py --tree $tree --model $model --card $card --port $port --out $O/$name \
      --bench mmsu --loads 96,16 --passes warm,clean,events > $O/$name.driver.log 2>&1
  fi
  log "boot $name rc $?"
}
quiet_host
both "serve vp1_m0 $MAIN 0 omni_colocated videomme_text" "serve vp1_l1 $LANE 1 omni_colocated videomme_text"
quiet_host
both "serve vp2_l0 $LANE 0 omni_colocated videomme_text" "serve vp2_m1 $MAIN 1 omni_colocated videomme_text"
log "video done"
quiet_host
both "serve mp1_m0 $MAIN 0 omni_text_fused mmsu" "serve mp1_l1 $LANE 1 omni_text_fused mmsu"
quiet_host
both "serve mp2_l0 $LANE 0 omni_text_fused mmsu" "serve mp2_m1 $MAIN 1 omni_text_fused mmsu"
log "q1115_vlane done"
