#!/usr/bin/env bash
# 11-14, node-0000, one H200 (card 0, NUMA 1), after q1114_synth. #3451 with the sync-free talker prompt build folded in
# (FIX 803329b97), and the lane with it (STACKFIX 48c10d0a1). Main runs once on this lease, only where nothing on file covers it.
# U: unit suite on FIX. I: seeded c1 SeedTTS, main and FIX, wav hashes kept. S: SeedTTS c16 (warm, clean x2, events) and c64
# (warm, events), main then FIX. M: speech c16 beside a looping Video-MME client on STACKFIX, against today's mx2_m*, mx2_w*.
# T: the same cell on STACKFIX under nsys gpuctxsw with NVTX, against tr2_m, tr2_w. Y: synthetic regimes on STACKFIX.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
NV=/workspace/marks/tasks/qwen3_omni_perf_20260923/scripts/omni_pipeline_nvtx
MAIN=/workspace/main FIX=/workspace/fix STACKFIX=/workspace/stackfix
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1114_synth done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1114_fix start"

log "units FIX start"
(cd $FIX && taskset -c $CARD_CPUS python3 -m pytest tests/unit_test/qwen3_omni -q -p no:cacheprovider > $O/units_fix.log 2>&1)
log "units FIX rc $? $(tail -1 $O/units_fix.log)"

identity() {
  local name=$1 tree=$2
  note $name $tree "seeded c1"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 1 --passes clean "--bench-extra=--seed 3451" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
  (cd $O/$name/c1_clean/bench/audio && find . -name "*.wav" | sort | xargs sha256sum > $O/$name/wav_sha256.txt)
}
identity id_m3 $MAIN
identity id_f $FIX
log "identity $(python3 $K/wav_identity.py $O c1_clean id_m3 id_f 2>&1 | tail -1)"

speech() {
  local name=$1 tree=$2 load=$3 passes=$4 repeats=$5
  note $name $tree "c$load"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads $load --passes $passes --bench-repeats $repeats > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech m16 $MAIN 16 warm,clean,events 2
speech f16 $FIX 16 warm,clean,events 2
speech m64 $MAIN 64 warm,events 1
speech f64 $FIX 64 warm,events 1

mixed() {
  local name=$1 tree=$2 extra=$3 senv=${4:-}
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
mixed mx3_x $STACKFIX "--passes warm,events"
TRACE="--passes warm,nsys --nsys-trace cuda,nvtx,osrt --nsys-graph-trace graph --nsys-gpuctxsw --nsys-delay 20 --nsys-seconds 15"
mixed tr3_x $STACKFIX "$TRACE --nsys-session tr3_x" "OMNI_PIPE_NVTX=1 PYTHONPATH=$NV:$STACKFIX"

log "boot sy_f $(git -C $STACKFIX rev-parse --short HEAD) $(git -C $STACKFIX rev-parse --short=12 HEAD:sglang_omni) synthetic regimes start"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $STACKFIX --model omni_colocated --card 0 --port 28400 --out $O/sy_f \
  --loads 1 --passes none --serve-extra "$CONTEXT_32K" \
  --nsys-session sy_f --nsys-trace cuda,nvtx,osrt --nsys-graph-trace graph \
  --serve-env "OMNI_SYNTH_IGNORE_EOS=1 OMNI_PIPE_NVTX=1 OMNI_SITECUSTOMIZE_CHAIN=$NV/sitecustomize.py PYTHONPATH=$K/synth_eos_probe:$STACKFIX" \
  --after "taskset -c $CARD_CPUS python3 $K/synth_regimes.py --port 28400 --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --out $O/sy_f/synth --nsys-session sy_f" \
  > $O/sy_f.driver.log 2>&1
log "boot sy_f rc $?"
for rep in $O/sy_f/synth/*/trace.nsys-rep $O/tr3_x/c16_nsys/trace.nsys-rep; do
  [ -f "${rep%.nsys-rep}.sqlite" ] || taskset -c 0-55 nsys export --type sqlite --force-overwrite=true --output "${rep%.nsys-rep}.sqlite" "$rep" > /dev/null 2>&1
done
log "q1114_fix done"
