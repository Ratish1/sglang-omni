#!/usr/bin/env bash
# 11-14, node-0000, after q1114_final: speech at a fixed 9 requests a second beside the looping Video-MME client under nsys
# (gpuctxsw, NVTX, graph replays), main once (trr_m) and the final stack (trr_x), so each owner's card wait is read at equal
# offered speech load. One 15 s window after 20 s of load.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
NV=/workspace/marks/tasks/qwen3_omni_perf_20260923/scripts/omni_pipeline_nvtx
MAIN=/workspace/main SFINAL=/workspace/sfinal
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1114_final done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1114_tracerate start"
traced() {
  local name=$1 tree=$2
  note $name $tree "seedtts c16 at 9/s beside text video, nsys"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,nsys "--bench-extra=--request-rate=9" \
    --nsys-session $name --nsys-trace cuda,nvtx,osrt --nsys-graph-trace graph --nsys-gpuctxsw --nsys-delay 20 --nsys-seconds 15 \
    --serve-env "OMNI_PIPE_NVTX=1 PYTHONPATH=$NV:$tree" > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:28400/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $CARD_CPUS python3 $K/videomme_speed.py --host 127.0.0.1 --port 28400 --repo-id zhaochenyang20/Video_MME_ci \
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
  python3 -c "import json; w=json.load(open('$O/$name/c16_nsys/nsys_window.json')); open('$O/$name/c16_nsys/window.txt','w').write(f\"{w['start']} {w['start'] + 15}\n\")"
}
traced trr_m $MAIN
traced trr_x $SFINAL
C=/workspace/marks/tasks/qwen3_omni_perf_20260923/scripts
for t in trr_m trr_x; do
  taskset -c 0-55 python3 $C/omni_census.py $O/$t/c16_nsys/trace.sqlite --window $O/$t/c16_nsys/window.txt --sections ADHLI \
    --sample-rate 24000 > $O/$t/census_ADHLI.txt 2>&1
done
log "q1114_tracerate done"
