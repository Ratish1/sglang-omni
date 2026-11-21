#!/usr/bin/env bash
# 11-21, node-0001, card 1. Check boot of the combined form with the index copy fix (OCVG 6c8168a28, tree
# 56d7e7db0fbc): the thinker moved CPU row indices to the device with blocking copies (post_prefill, the embed merge),
# which waited for the stream inside the turn (q1121_sync: 3.1/s each, holds of 200 to 218 ms on 6,000-token chunks).
# Speech 8/s beside video 1/s with the turn, thinker step and sync site probes: the sites must be gone and the long
# holds with them.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
OCVGP=/workspace/ocvg_prof2
OCVG_SHA=6c8168a28
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS=56-111,168-223
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
video_loop() {
  local name=$1 tree=$2 port=$3 driver=$4
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
log "q1121_check2 start"
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-one-context-visual-graph
git worktree add -q --detach $OCVGP FETCH_HEAD
if [ "$(git -C $OCVGP rev-parse --short=9 HEAD)" != "$OCVG_SHA" ] || ! git -C $OCVGP diff --quiet HEAD -- sglang_omni; then
  log "q1121_check2 stopped: ocvg_prof2 is not a clean $OCVG_SHA"
  exit 1
else :; fi
name=ck1_b port=31570
note $name $OCVGP "seedtts 8/s beside video 1/s card 1 turn, thinker step and sync site probes port $port"
taskset -c $CPUS python3 $K/baseline.py --tree $OCVGP --model omni_colocated --card 1 --port $port --out $O/$name \
  --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,events \
  --probe --serve-env "OMNI_HOP_PROBE_TURNS=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_SYNC_SITES=1" \
  > $O/$name.driver.log 2>&1 &
driver=$!
until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
video_loop $name $OCVGP $port $driver &
looper=$!
wait $driver
log "boot $name rc $?"
wait $looper
for cell in c64_warm c64_events; do
  python3 $K/turn_waits.py $O/$name --cell $cell > $O/$name.turns_$cell.txt 2>&1
  python3 $K/turn_hold_parts.py $O/$name --cell $cell > $O/$name.holds_$cell.txt 2>&1
  python3 $K/sync_window.py $O/$name --cell $cell --thread scheduler-thinker > $O/$name.sync_$cell.txt 2>&1
done
grep -h "waiter priority 1" $O/$name.turns_c64_*.txt >> $O/queue.log
grep -h "thinker holds handed" $O/$name.holds_c64_*.txt >> $O/queue.log
log "CK2 read"
log "q1121_check2 done"
