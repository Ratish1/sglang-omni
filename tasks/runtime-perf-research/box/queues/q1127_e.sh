#!/usr/bin/env bash
# 11-27, node-0000, after the second review of #3452 at 91c2e2010 (tasks/pr3452_91c2e201/REVIEW.md). Its worse rows all
# scale with the videos running at once, and the equal load cell's read pass was the cached case (the looping video
# client repeats 50 videos, so from the second pass every video prompt is a prefix cache hit). Speech never waits behind
# a video at any video limit of 1 or more, so the limit is the lever.
#   1. Speech 8/s (seed 3451, cap 64) beside video 1/s (seed 7), colocated, thinker radix cache off so every video
#      prefills on every pass, warm and events passes read: main, the fixes at the limit 8 (default), 2 and 1,
#      M F8 F2 F1 F1 F2 F8 M.
#   2. Fresh servers, one text Video-MME c16 pass and one talker pass each, the fixes at the limit 2 and 1, 5 boots each,
#      read against main's 10 fresh boots of q1127_c (same node and day).
set -u
O=/data/v1127
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main FIX=/workspace/fix
CPUS=56-111,168-223
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
UNCACHED="--thinker.engine.disable_radix_cache true"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
limit_of() { case $1 in 8) echo "" ;; *) echo "--preprocessing.factory.max_running_video_requests $1" ;; esac; }
log "q1127_e start"

export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
video_loop() {
  local name=$1 tree=$2 driver=$3 port=$4
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $CPUS python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
      --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
      --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 8 --request-rate 1 \
      --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1 &
    client=$!
    while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
    kill $client 2>/dev/null
    wait $client 2>/dev/null
    pass=$((pass + 1))
  done
}
uncached() {
  local name=$1 tree=$2 port=$3 extra=$4
  note $name $tree "colocated, thinker radix cache off, speech 8/s beside video 1/s, $extra"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K $UNCACHED $extra" --omni-chunk-audio "--bench-extra=--request-rate=8" \
    --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $driver $port &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
  log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name)"
}
uncached u1_m $MAIN 20000 ""
uncached u2_f8 $FIX 20200 "$(limit_of 8)"
uncached u3_f2 $FIX 20400 "$(limit_of 2)"
uncached u4_f1 $FIX 20600 "$(limit_of 1)"
uncached u5_f1 $FIX 20800 "$(limit_of 1)"
uncached u6_f2 $FIX 21000 "$(limit_of 2)"
uncached u7_f8 $FIX 21200 "$(limit_of 8)"
uncached u8_m $MAIN 21400 ""
for limit in 8 2 1; do
  python3 $K/body_table.py $O c64 "the fixes at limit $limit against main, uncached, 8/s beside video 1/s, colocated" \
    main=u1_m,u8_m fix=$(ls -d $O/u*_f$limit | xargs -n1 basename | paste -sd,) >> $O/queue.log 2>&1
done
python3 $K/mixed_cell_load.py $O u1_m u8_m u2_f8 u7_f8 u3_f2 u6_f2 u4_f1 u5_f1 >> $O/queue.log 2>&1
log "uncached read"

fresh() {
  local name=$1 tree=$2 port=$3 extra=$4
  note $name $tree "CI speech pipeline, fresh server, text Video-MME c16 once, talker test once, $extra"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card 0 --port $port --out $O/$name \
    --serve-extra "$extra" --bench videomme_text --loads 16 --passes clean \
    --after "python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 1" > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(grep -o '\"accuracy\": [0-9.]*' $O/$name/talker/summary.jsonl 2>/dev/null | head -1)"
}
index=0
for limit in 2 1 1 2 2 1 1 2 2 1; do
  index=$((index + 1))
  fresh fl${index}_f$limit $FIX $((22000 + 200 * index)) "$(limit_of $limit)"
done
mains=$(ls -d $O/fr*_m | xargs -n1 basename | sed 's|$|/c16_clean|' | paste -sd,)
for limit in 2 1; do
  fixes=$(ls -d $O/fl*_f$limit | xargs -n1 basename | sed 's|$|/c16_clean|' | paste -sd,)
  python3 $K/videomme_answer_diff.py $O main=$mains fix$limit=$fixes > $O/gate_text_limit$limit.txt 2>&1
done
mains=$(ls -d $O/fr*_m | xargs -n1 basename | sed 's|$|/talker|' | paste -sd,)
for limit in 2 1; do
  fixes=$(ls -d $O/fl*_f$limit | xargs -n1 basename | sed 's|$|/talker|' | paste -sd,)
  python3 $K/videomme_answer_diff.py $O main=$mains fix$limit=$fixes > $O/gate_talker_limit$limit.txt 2>&1
done
log "limit gates read"
log "q1127_e done"
