#!/usr/bin/env bash
# 11-14, node-0000, after q1114_fix3: the final #3451 head (FINAL 2e0672ab4: #3451, the sync-free prompt build, relayed rows
# in pinned memory, the decode mask fix) and the final stack with #3452 (SFINAL f8313e115). Main is already on file for every
# cell: id_m3, m16, m64, mx2_m1/m2, mxr_m.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
FINAL=/workspace/final SFINAL=/workspace/sfinal
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1114_fix3 done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1114_final start"
log "units FINAL start"
(cd $FINAL && taskset -c $CARD_CPUS python3 -m pytest tests/unit_test/qwen3_omni -q -p no:cacheprovider > $O/units_final.log 2>&1)
log "units FINAL rc $? $(tail -1 $O/units_final.log)"
note id_fin $FINAL "seeded c1"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $FINAL --model omni_colocated --card 0 --port 28400 --out $O/id_fin \
  --loads 1 --passes clean "--bench-extra=--seed 3451" > $O/id_fin.driver.log 2>&1
log "boot id_fin rc $?"
(cd $O/id_fin/c1_clean/bench/audio && find . -name "*.wav" | sort | xargs sha256sum > $O/id_fin/wav_sha256.txt)
log "identity $(python3 $K/wav_identity.py $O c1_clean id_m3 id_fin 2>&1 | tail -1)"
speech() {
  local name=$1 load=$2 passes=$3 repeats=$4
  note $name $FINAL "c$load"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $FINAL --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads $load --passes $passes --bench-repeats $repeats > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech fin16 16 warm,clean,events 2
speech fin64 64 warm,events 1
mixed() {
  local name=$1 tree=$2 extra=$3
  note $name $tree "seedtts c16 beside text video $extra"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events $extra > $O/$name.driver.log 2>&1 &
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
mixed mxf_f $FINAL ""
mixed mxf_x $SFINAL ""
mixed mxrf_f $FINAL "--bench-extra=--request-rate=9"
mixed mxrf_x $SFINAL "--bench-extra=--request-rate=9"
log "q1114_final done"
