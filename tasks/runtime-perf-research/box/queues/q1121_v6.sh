#!/usr/bin/env bash
# 11-21, node-0001, card 0, after q1121_v4's W3 traces. The receive-only transport change alone slows the talker's
# back-to-back steps in all four W1 boots (p25 7.73 / 7.60 against 6.84 / 6.94 ms, mean 9.43 / 9.39 against 9.02 / 9.04,
# rows 2.30 to 2.34 alike) while it adds no GPU work, and the graph change does the same (E4). Validation of the current
# heads (W4 to W7) waits for the mechanism. The steps it slows are the talker's unchanged-row-list steps (mean 9.57 /
# 9.42 against 8.53 / 8.66 ms; changed steps 9.38 against 9.20), and the thinker is not idler on main during its loop
# holds (only 37 to 40 of 126 holds overlap a thinker gap over 50 ms). Graph-level spans include time sliced out, so
# the traces collect context switches: q1121_v4 stops after W2 (its W3 is replaced here), then main, the transport
# change and the graph change are traced on card 0 one after another (one context switch session at a time), each
# with an events pass for the gate. Per trace: card residency per process, switches, the talker's end-of-step waits,
# talker steps and the thinker's eager steps.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain PFIX=/workspace/pfix PSHM2=/workspace/pshm2
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS=0-55,112-167
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
traced() {
  local name=$1 tree=$2 port=$3
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED beside video 1/s card 0 nsys then events port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" \
    --passes warm,nsys,events --nsys-session $name --nsys-trace cuda --nsys-graph-trace graph --nsys-gpuctxsw \
    --nsys-delay 20 --nsys-seconds 15 > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $port $driver &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
  log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name), $(node_free)"
  (cd $O/$name && python3 - c64_nsys/trace.sqlite serve.log < $K/talker_steps.py > talker_steps.txt 2>&1
    python3 - c64_nsys/trace.sqlite serve.log < $K/thinker_steps.py > thinker_steps.txt 2>&1
    python3 $K/ctxsw_read.py c64_nsys/trace.sqlite > ctxsw.txt 2>&1)
  log "trace read $name: $(grep -c . $O/$name/talker_steps.txt) talker, $(grep -c . $O/$name/thinker_steps.txt) thinker, $(grep -c . $O/$name/ctxsw.txt) ctxsw lines"
}
until grep -q "^boot w24_a .* start" $O/queue.log 2>/dev/null; do sleep 10; done
pkill -f "^bash /data/v1121/q1121_v4.sh"
log "q1121_v4 stopped during W2's last boot (it runs on); W3 moves here with context switches, W4 to W7 wait for the talker mechanism"
until ! pgrep -f "kit/tasks/runtime-perf-research/box/baseline.py" > /dev/null; do sleep 20; done
log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O w24_a)"
python3 $K/body_table.py $O c64 "stack with both changes against main, card 0 A B B A, speech 8/s and video 1/s" \
  base=w21_a,w24_a change=w22_b,w23_b >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O base=w21_a,w24_a change=w22_b,w23_b 2>&1 | grep gaps >> $O/queue.log
log "w2 read"
log "q1121_v6 start, $(node_free)"
traced c1_m $NMAIN 28170
traced c2_x $PSHM2 28370
traced c3_p $PFIX 28570
log "q1121_v6 done"
