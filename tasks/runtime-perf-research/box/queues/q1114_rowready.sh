#!/usr/bin/env bash
# 11-14, node-0000, after q1114_earlystart: per-row talker readiness (ROWREADY 439bb5e03 on the #3451 candidate) with the
# talker starting at one thinker chunk. Units; seeded c1 against main's hashes (each row's step inputs are unchanged, so
# bytes must match); c16 and c64 speech against m16/h16/es16 and m64/h64/es64; the equal-arrival mixed cell against eq_m/eq_f.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
ROW=/workspace/rowready
EARLY="--talker_ar.factory.enable_partial_start true --talker_ar.factory.enable_talker_start_topology true"
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1114_earlystart done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1114_rowready start"
log "units ROWREADY start"
(cd $ROW && taskset -c $CARD_CPUS python3 -m pytest tests/unit_test/qwen3_omni -q -p no:cacheprovider > $O/units_rowready.log 2>&1)
log "units ROWREADY rc $? $(tail -1 $O/units_rowready.log)"
note id_r $ROW "seeded c1 early start"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $ROW --model omni_colocated --card 0 --port 28400 --out $O/id_r \
  --loads 1 --serve-extra "$EARLY" --passes clean "--bench-extra=--seed 3451" > $O/id_r.driver.log 2>&1
log "boot id_r rc $?"
(cd $O/id_r/c1_clean/bench/audio && find . -name "*.wav" | sort | xargs sha256sum > $O/id_r/wav_sha256.txt)
log "identity $(python3 $K/wav_identity.py $O c1_clean id_m3 id_r 2>&1 | tail -1)"
speech() {
  local name=$1 load=$2
  note $name $ROW "c$load early start"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $ROW --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads $load --serve-extra "$EARLY" --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $? waiting $(grep -c 'talker rows waiting' $O/$name/serve.log) $(grep 'talker rows waiting' $O/$name/serve.log | tail -1 | cut -c1-200)"
}
speech rs16 16
speech rs64 64
for base in m es; do
  python3 $K/first_audio_states.py --main $O/${base}16 --branch $O/rs16 >> $O/queue.log 2>&1
  python3 $K/first_audio_states.py --cell c64_events --main $O/${base}64 --branch $O/rs64 >> $O/queue.log 2>&1
done
python3 $K/body_table.py $O c16 "row ready c16" main=m16 fix=h16 early=es16 row=rs16 >> $O/queue.log 2>&1
python3 $K/body_table.py $O c64 "row ready c64" main=m64 fix=h64 early=es64 row=rs64 >> $O/queue.log 2>&1
export OMNI_ARRIVAL_SEED=3451
note eq_r $ROW "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside text video, early start"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $ROW --model omni_colocated --card 0 --port 28400 --out $O/eq_r \
  --loads 64 --serve-extra "$CONTEXT_32K $EARLY" --omni-chunk-audio --passes warm,events "--bench-extra=--request-rate=8" \
  > $O/eq_r.driver.log 2>&1 &
driver=$!
until curl -sf --max-time 2 http://127.0.0.1:28400/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
(
  cd $ROW
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$ROW taskset -c $CARD_CPUS python3 $K/videomme_speed.py --host 127.0.0.1 --port 28400 --repo-id zhaochenyang20/Video_MME_ci \
      --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 4 \
      --output-dir $O/eq_r/video_pass$pass --disable-tqdm >> $O/eq_r.video.log 2>&1 &
    client=$!
    while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
    kill $client 2>/dev/null
    wait $client 2>/dev/null
    pass=$((pass + 1))
  done
) &
looper=$!
wait $driver
log "boot eq_r rc $? waiting $(grep -c 'talker rows waiting' $O/eq_r/serve.log)"
wait $looper
python3 $K/mixed_cell_load.py $O eq_m eq_f eq_r >> $O/queue.log 2>&1
log "q1114_rowready done"
