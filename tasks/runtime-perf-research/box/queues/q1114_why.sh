#!/usr/bin/env bash
# 11-14, node-0000: speech at a fixed 9 requests a second beside the looping Video-MME client. trr_f: #3451 candidate
# (FIX 053a02a6f) under nsys, against trr_m, to name where the video answers' thinker decode time goes. mxr_w: #3452 alone
# (WAIT 0b7b20957) on main, against mxr_m, the pair that decides whether #3452 can merge before #3451.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
NV=/workspace/marks/tasks/qwen3_omni_perf_20260923/scripts/omni_pipeline_nvtx
FIX=/workspace/fix WAIT=/workspace/lane_wait
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
log "q1114_why start"
cell() {
  local name=$1 tree=$2 extra=$3 senv=${4:-}
  note $name $tree "seedtts c16 at 9/s beside text video"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=9" $extra \
    ${senv:+--serve-env "$senv"} > $O/$name.driver.log 2>&1 &
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
}
cell trr_f $FIX "--passes warm,nsys --nsys-session trr_f --nsys-trace cuda,nvtx,osrt --nsys-graph-trace graph --nsys-gpuctxsw --nsys-delay 20 --nsys-seconds 15" "OMNI_PIPE_NVTX=1 PYTHONPATH=$NV:$FIX"
python3 -c "import json; w=json.load(open('$O/trr_f/c16_nsys/nsys_window.json')); open('$O/trr_f/c16_nsys/window.txt','w').write(f\"{w['start']} {w['start'] + 15}\n\")"
cell mxr_w $WAIT "--passes warm,events"
taskset -c 0-55 python3 /workspace/marks/tasks/qwen3_omni_perf_20260923/scripts/omni_census.py $O/trr_f/c16_nsys/trace.sqlite \
  --window $O/trr_f/c16_nsys/window.txt --sections ADHLI --sample-rate 24000 > $O/trr_f/census_ADHLI.txt 2>&1
log "q1114_why done"
