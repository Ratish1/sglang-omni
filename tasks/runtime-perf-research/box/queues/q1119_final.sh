#!/usr/bin/env bash
# 11-19, node-0001, after q1119_mmsu: the exact final tree (764495b0c, tree 33c0b6f6d291, B plus a note), no
# taskset: units, the one-CPU two-video probe, and two boots of each one-card cell the run measured on B: the
# one-card CI text Video-MME c1 c16 and talker test (of_*), SeedTTS c16 beside 4 videos on it (sf_*) and on the
# colocated config (cf_*), the colocated text Video-MME c16 (tf_*).
set -u
O=/data/v1119
K=/workspace/kit2/tasks/runtime-perf-research/box
F=/workspace/vfinal
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
until grep -q "q1119_mmsu done" $O/queue.log; do sleep 10; done
log "q1119_final start"
(cd $F && python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_omni tests/unit_test/preprocessing \
  tests/unit_test/utils tests/unit_test/minicpm_o tests/unit_test/ming_omni 2>&1 | tail -15) > $O/final_units.log
log "final units $(grep -E 'passed|failed' $O/final_units.log | tail -1)"
(cd $F && PYTHONPATH=$F python3 $K/video_share_probe.py 2>&1 | tail -3) > $O/final_probe.log
log "final probe $(cat $O/final_probe.log)"

for index in 1 2; do
  quiet_host
  name=of_$index
  note $name $F "omni_ci_onecard text video c1 c16, talker c16"
  python3 $K/baseline.py --tree $F --model omni_ci_onecard --port 28400 --out $O/$name \
    --bench videomme_text --loads 1,16 --passes clean,events --bench-repeats 2 \
    --after "python3 $K/stage_cpus.py \$(cat $O/$name/server.pgid) > $O/$name/stage_cpus.txt; python3 $K/videomme_talker_ci.py --port 28400 --out $O/$name/talker --passes 3" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
done

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
for index in 1 2; do
  quiet_host
  mixed sf_$index $F omni_ci_onecard "" 28500
done
for index in 1 2; do
  quiet_host
  mixed cf_$index $F omni_colocated "$CONTEXT_32K" 28600
done
for index in 1 2; do
  quiet_host
  name=tf_$index
  note $name $F "omni_colocated text video c16"
  python3 $K/baseline.py --tree $F --model omni_colocated --port 28700 --out $O/$name \
    --bench videomme_text --loads 16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
done
log "q1119_final done"
