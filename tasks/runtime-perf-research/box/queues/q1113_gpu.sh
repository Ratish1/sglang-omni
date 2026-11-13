#!/usr/bin/env bash
# 11-13, node-0000, one H200 (card 0 on NUMA 1). After the first boots, in order of what decides:
# A: the serial preprocessing OpenMP profile on a quiet host (no server), the other socket, every team size and both policies.
# C2: one nsys trace with GPU context switches per #3451 arm (main, final), c16 events, the thinker reply's owner.
# C1: seeded c1 byte identity, main against #3451's final tree. C3: #3451 c16, boots m s s m. C4: #3451 c64, m s s m.
# B1: speech beside video, main, lane with the default wait policy (W), lane with passive (P): m w p p w m.
# B2: MMSU on the fused text pipeline, main against W: m w w m. B4: colocated text video, W against P, CPU from thread_cpu.
set -u
O=/data/rp1113
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main P3S=/workspace/p3s LANE=/workspace/lane WAIT=/workspace/lane_wait
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223 OTHER_CPUS=0-55,112-167
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 load $(tail -1 $O/cpu_load.log) start"; }
until grep -q "fill done" $O/queue.log 2>/dev/null; do sleep 15; done
log "q1113_gpu start"

profile() {
  local threads=$1 policy=$2
  local name=quiet_${threads}_${policy}_$(date -u +%H%M%S)
  log "omp $name start load $(tail -1 $O/cpu_load.log)"
  if [ "$policy" = unset ]; then
    (cd $WAIT && env -u OMP_WAIT_POLICY OMP_NUM_THREADS=$threads PYTHONPATH=$WAIT taskset -c $OTHER_CPUS \
      python3 $K/omp_serial_profile.py run --out $O/omp/$name.json > $O/omp/$name.log 2>&1)
  else
    (cd $WAIT && OMP_WAIT_POLICY=$policy OMP_NUM_THREADS=$threads PYTHONPATH=$WAIT taskset -c $OTHER_CPUS \
      python3 $K/omp_serial_profile.py run --out $O/omp/$name.json > $O/omp/$name.log 2>&1)
  fi
  log "omp $name rc $?"
}
for setting in 8:unset 8:PASSIVE 1:unset 2:unset 2:PASSIVE 4:unset 4:PASSIVE 14:unset 14:PASSIVE 28:unset 28:PASSIVE 112:unset 8:unset 8:PASSIVE; do
  profile ${setting%%:*} ${setting##*:}
done
log "cell A done"

trace() {
  local name=$1 tree=$2
  note $name $tree "nsys gpuctxsw c16"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --passes warm,nsys --nsys-session $name --nsys-trace cuda,nvtx,osrt --nsys-graph-trace node --nsys-gpuctxsw \
    --nsys-delay 20 --nsys-seconds 15 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
trace tr_m $MAIN
trace tr_s $P3S
log "cell C2 done"

identity() {
  local name=$1 tree=$2
  note $name $tree "seeded c1"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 1 --passes clean "--bench-extra=--seed 3451" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
identity id_m $MAIN
identity id_s $P3S
log "cell C1 done"

speech16() {
  local name=$1 tree=$2
  note $name $tree "c16"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech16 s16_m1 $MAIN
speech16 s16_s1 $P3S
speech16 s16_s2 $P3S
speech16 s16_m2 $MAIN
log "cell C3 done"

mixed() {
  local name=$1 tree=$2
  note $name $tree "seedtts beside text video"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --passes warm,events --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1 &
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
mixed mx_m1 $MAIN
mixed mx_w1 $WAIT
mixed mx_p1 $LANE
mixed mx_p2 $LANE
mixed mx_w2 $WAIT
mixed mx_m2 $MAIN
log "cell B1 done"

mmsu() {
  local name=$1 tree=$2
  note $name $tree "fused mmsu"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_text_fused --card 0 --port 28400 --out $O/$name \
    --bench mmsu --loads 96,16 --passes warm,clean,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
mmsu mu_m1 $MAIN
mmsu mu_w1 $WAIT
mmsu mu_w2 $WAIT
mmsu mu_m2 $MAIN
log "cell B2 done"

speech64() {
  local name=$1 tree=$2
  note $name $tree "c64"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 64 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech64 s64_m1 $MAIN
speech64 s64_s1 $P3S
speech64 s64_s2 $P3S
speech64 s64_m2 $MAIN
log "cell C4 done"

video() {
  local name=$1 tree=$2
  note $name $tree "colocated text video"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --bench videomme_text --loads 16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video vt_w1 $WAIT
video vt_p1 $LANE
video vt_p2 $LANE
video vt_w2 $WAIT
log "q1113_gpu done"
