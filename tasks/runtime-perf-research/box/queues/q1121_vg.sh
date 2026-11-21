#!/usr/bin/env bash
# 11-21, node-0001, card 1 beside q1121_oc3 on card 0. Part 1 of the one-context design: every thinker prefill chunk on
# the prefill graph, deepstack rows included (PVG f71c367f0, tree 647e2722023f, on the graph change 09dae918c). The
# thinker's text model reads a zeroed deepstack buffer during prefill; the runner lays a chunk's deepstack rows over it
# and clears them before the next prefill, so a chunk that computes image or video rows replays the graph.
#   units   tests/unit_test/qwen3_omni on PVG
#   VT      CI layout text Video-MME c1 and c16, two repeats, accuracy and per-question identity against the graph
#           change (fc1_p0, fc2_p1) and main (fc1_m1, fc2_m0) from q1121_fix
#   VP      speech 8/s beside video 1/s with thinker step probes: steps over 30 ms in the warm pass (every video new)
#           and the events pass, against the graph change alone (PFIX) on the same card
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
PFIX=/workspace/pfix PVG=/workspace/pvg
PVG_SHA=f71c367f0
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1"
CPUS=56-111,168-223
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
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
probed_mixed() {
  local name=$1 tree=$2 port=$3
  note $name $tree "seedtts 8/s beside video 1/s card 1 probed port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 1 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,events \
    --probe --serve-env "$PROBES" > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $port $driver &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}
log "q1121_vg start, $(node_free)"
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-graph-prefill-visual-rows
git worktree add -q --detach $PVG FETCH_HEAD
log "tree pvg $(git -C $PVG rev-parse --short HEAD) $(git -C $PVG rev-parse --short=12 HEAD:sglang_omni)"
if [ "$(git -C $PVG rev-parse --short=9 HEAD)" != "$PVG_SHA" ]; then
  log "q1121_vg stopped: pvg is not $PVG_SHA"
  exit 1
else :; fi
mkdir -p $O/units_pvg
cd $PVG
PYTHONPATH=$PVG taskset -c $CPUS python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_omni > $O/units_pvg/units.log 2>&1
log "units pvg rc $? $(grep -E 'passed|failed' $O/units_pvg/units.log | tail -1)"
note vt1_g1 $PVG "omni_ci_onecard text video c1 c16 two repeats card 1"
taskset -c $CPUS python3 $K/baseline.py --tree $PVG --model omni_ci_onecard --card 1 --port 31790 --out $O/vt1_g1 \
  --bench videomme_text --loads 1,16 --passes clean,events --bench-repeats 2 > $O/vt1_g1.driver.log 2>&1
log "boot vt1_g1 rc $?"
for cell in c1 c16; do
  python3 $K/video_reps.py $O ${cell}_clean "text Video-MME $cell, graph change against part 1" fc1_p0,fc2_p1 vt1_g1 >> $O/queue.log 2>&1
done
python3 $K/videomme_text_identity.py --passes "main_c1=$O/fc1_m1/c1_clean/bench/rep*/videomme_results.json" \
  "graph_c1=$O/fc1_p0/c1_clean/bench/rep*/videomme_results.json" \
  "graph2_c1=$O/fc2_p1/c1_clean/bench/rep*/videomme_results.json" \
  "part1_c1=$O/vt1_g1/c1_clean/bench/rep*/videomme_results.json" > $O/vg_video_identity_c1.txt 2>&1
log "VT read"
probed_mixed vp1_g1 $PVG 31990
probed_mixed vp2_f1 $PFIX 31950
log "VP done"
log "q1121_vg done"
