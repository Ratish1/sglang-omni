#!/usr/bin/env bash
# 11-19, node-0001: the final tree (828ac76ea) read 5 % slower first audio on colocated speech beside video and 30 to
# 60 ms more c1 preprocessing than B (eb449662d) on boots that started at host load 57 to 71. The two trees differ only
# in the video thread's failure boundary. B and final alternate in one window, b f f b: colocated SeedTTS c16 beside 4
# videos (bv_*), then the one-card CI text Video-MME c1, two clean repeats and the events pass (bc_*). No taskset.
set -u
O=/data/v1119
K=/workspace/kit2/tasks/runtime-perf-research/box
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
tree_of() { case $1 in b) echo /workspace/vb ;; f) echo /workspace/vfinal2 ;; *) exit 2 ;; esac; }
mixed() {
  local name=$1 tree=$2 model=$3 extra=$4 port=$5
  note $name $tree "$model seedtts c16 beside text video"
  python3 $K/baseline.py --tree $tree --model $model --port $port --out $O/$name \
    --loads 16 --passes warm,events --serve-extra "$extra" > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree python3 $K/videomme_speed.py --host 127.0.0.1 --port $port --repo-id zhaochenyang20/Video_MME_ci \
        --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 4 \
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
log "q1119_bfband start"
for arm in b f f b; do
  quiet_host
  index=$([ -d $O/bv_${arm}1 ] && echo 2 || echo 1)
  mixed bv_${arm}$index $(tree_of $arm) omni_colocated "$CONTEXT_32K" 28600
done
for arm in b f f b; do
  quiet_host
  index=$([ -d $O/bc_${arm}1 ] && echo 2 || echo 1)
  name=bc_${arm}$index
  note $name $(tree_of $arm) "omni_ci_onecard text video c1"
  python3 $K/baseline.py --tree $(tree_of $arm) --model omni_ci_onecard --port 28400 --out $O/$name \
    --bench videomme_text --loads 1 --passes clean,events --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
done
log "q1119_bfband done"
