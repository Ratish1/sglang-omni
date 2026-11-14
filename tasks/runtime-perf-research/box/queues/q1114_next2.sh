#!/usr/bin/env bash
# 11-14, node-0000, one H200 (card 0, NUMA 1), after q1114_identity.
# R: main serves the looping questions (005-3, 017-1) and a stable control (006-1) at c1 and dumps their exact preprocessing
#    tensors; the HF bf16 thinker then answers the same bytes greedily, twice each.
# S: the lane at c16, then c64, each on a fresh boot so every prefill is cold and chunked where co-batching puts it, with
#    thinker_slice_probe checking each chunk's video embedding and DeepStack rows against the encoder rows the prompt fixes.
# M: speech c16 beside a looping Video-MME client (4 in flight), boots main, lane, lane, main, the speech client recording
#    each chunk's audio length (omni_seedtts_chunks.py), so the playback underrun is measured.
# T: the same cell on the lane, then main, under nsys with GPU context switches and the omni_pipeline_nvtx ranges (coord.recv
#    marks carry each chunk's samples), one 15 s window after 20 s of load.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
NV=/workspace/marks/tasks/qwen3_omni_perf_20260923/scripts/omni_pipeline_nvtx
MAIN=/workspace/main WAIT=/workspace/lane_wait
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
card_free() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  until [ "${used:-0}" -lt 1000 ]; do
    log "card busy ${used} MiB, waiting"
    sleep 30
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  done
}
until grep -q "q1114_identity done" $O/queue.log 2>/dev/null; do sleep 15; done
log "q1114_next2 start"

KEYS=$(cd $MAIN && python3 $K/question_keys.py $O/id_m1 005-3,017-1,006-1)
log "R keys 005-3,017-1,006-1 = $KEYS"
card_free
mkdir -p $O/ref_m1/hash
note ref_m1 $MAIN "c1 dump"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $MAIN --model omni_colocated --card 0 --port 28400 --out $O/ref_m1 \
  --bench videomme_text --loads 1 --passes clean --serve-extra "$CONTEXT_32K" \
  --serve-env "PYTHONPATH=$K/preproc_hash_probe:$MAIN OMNI_PREPROC_HASH=$O/ref_m1/hash/h OMNI_PREPROC_DUMP_KEYS=$KEYS" \
  > $O/ref_m1.driver.log 2>&1
log "boot ref_m1 rc $?"
card_free
log "hf reference start"
(cd $MAIN && taskset -c $CARD_CPUS python3 $K/hf_reference_greedy.py --dumps "$O/ref_m1/hash/h.*.pt" \
  --out $O/ref_m1/hf_reference.json --runs 2 > $O/ref_m1/hf_reference.log 2>&1)
log "hf reference rc $?"

slice() {
  local name=$1 loads=$2
  card_free
  mkdir -p $O/$name/slice
  note $name $WAIT "slice probe c$loads cold"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $WAIT --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --bench videomme_text --loads $loads --passes clean --serve-extra "$CONTEXT_32K" \
    --serve-env "PYTHONPATH=$K/thinker_slice_probe:$WAIT OMNI_SLICE_PROBE=$O/$name/slice/s" \
    --after "taskset -c $CARD_CPUS python3 $K/cobatch_probe.py --port 28400 --out $O/$name/cobatch.json" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
slice sl_w16 16
slice sl_w64 64
log "S done"

mixed() {
  local name=$1 tree=$2 extra=$3 senv=${4:-}
  card_free
  note $name $tree "seedtts c16 beside text video"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio $extra ${senv:+--serve-env "$senv"} > $O/$name.driver.log 2>&1 &
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
mixed mx2_m1 $MAIN "--passes warm,events"
mixed mx2_w1 $WAIT "--passes warm,events"
mixed mx2_w2 $WAIT "--passes warm,events"
mixed mx2_m2 $MAIN "--passes warm,events"
log "M done"

TRACE="--passes warm,nsys --nsys-trace cuda,nvtx,osrt --nsys-graph-trace graph --nsys-gpuctxsw --nsys-delay 20 --nsys-seconds 15"
mixed tr2_w $WAIT "$TRACE --nsys-session tr2_w" "OMNI_PIPE_NVTX=1 PYTHONPATH=$NV:$WAIT"
mixed tr2_m $MAIN "$TRACE --nsys-session tr2_m" "OMNI_PIPE_NVTX=1 PYTHONPATH=$NV:$MAIN"
log "q1114_next2 done"
