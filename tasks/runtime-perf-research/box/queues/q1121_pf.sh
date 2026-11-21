#!/usr/bin/env bash
# 11-21, node-0001, card 1 while card 0 measures. A replayed thinker prefill holds the host 10 to 11 ms in
# prepare_and_forward against 0.7 ms for a replayed decode (q1121_vp2 parts probe, both arms). In one context that is
# turn time the talker waits through. One boot of part 2's landing base (both changes and part 1, two processes,
# ed2980ce3) at speech 8/s beside video 1/s, py-spy record of every server process for 30 s of a pyspy pass; the
# thinker scheduler thread's samples under prepare_and_forward name the frames.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
BVGP=/workspace/bvg_prof
BVG_SHA=ed2980ce3
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
log "q1121_pf start"
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-both-changes-visual-graph
git worktree add -q --detach $BVGP FETCH_HEAD
if [ "$(git -C $BVGP rev-parse --short=9 HEAD)" != "$BVG_SHA" ] || ! git -C $BVGP diff --quiet HEAD -- sglang_omni; then
  log "q1121_pf stopped: bvg_prof is not a clean $BVG_SHA"
  exit 1
else :; fi
name=pf1_b port=31730
note $name $BVGP "seedtts 8/s beside video 1/s card 1 pyspy port $port"
taskset -c $CPUS python3 $K/baseline.py --tree $BVGP --model omni_colocated --card 1 --port $port --out $O/$name \
  --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,pyspy \
  --pyspy-delay 20 --pyspy-seconds 30 > $O/$name.driver.log 2>&1 &
driver=$!
until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
video_loop $name $BVGP $port $driver &
looper=$!
wait $driver
log "boot $name rc $?"
wait $looper
thinker_pid=$(sed -n 's/.*sglang_ar_startup stage=thinker .*pid=\([0-9]*\).*/\1/p' $O/$name/serve.log | head -1)
for record in $O/$name/c64_pyspy/pyspy_$thinker_pid.txt; do
  python3 $K/anchor_tree.py $record scheduler-thinker prepare_and_forward --top 40 > $O/$name.prepare_tree.txt 2>&1
done
log "PF read thinker pid $thinker_pid"
log "q1121_pf done"
