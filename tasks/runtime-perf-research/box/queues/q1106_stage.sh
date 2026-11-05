#!/usr/bin/env bash
# 11-06, node-0000, after q1106_fill: the stage alone on CPU, closed loop on the Video-MME CI videos, with non-video requests on main's
# serial path and each video on its own thread. Thread count at 112, 16 and 8 CPUs under passive OpenMP, the team size the CPU split gives
# on the full host and on a socket, and what one more private preprocessor costs. Starts when the host is quiet.
set -u
O=/data/rp1106
K=/workspace/kit/tasks/runtime-perf-research/box
T=/workspace/lane
mkdir -p $O/stage
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1106_fill done" $O/queue.log 2>/dev/null; do sleep 15; done
wait_quiet() {
  local waited=0
  while [ $waited -lt 3600 ]; do
    if tail -3 $O/cpu_load.log | awk '{split($2,a,"/"); split($3,b,"/"); if (a[1]>25 || b[1]>25) bad=1} END {exit bad}'; then
      log "quiet after ${waited}s: $(tail -1 $O/cpu_load.log)"
      return
    else
      sleep 30
      waited=$((waited + 30))
    fi
  done
  log "not quiet after ${waited}s: $(tail -1 $O/cpu_load.log)"
}
cell() {
  local cpus=$1 tag=$2 omp=$3 threads=$4
  local name=${tag}_omp${omp}_k${threads}
  ( cd $T && OMP_NUM_THREADS=$omp OMP_WAIT_POLICY=PASSIVE PYTHONPATH=$T taskset -c $cpus python3 $K/preprocess_stage_bench.py --concurrency 16 \
    --processors $threads --out $O/stage/$name.json > $O/stage/$name.log 2>&1 )
  log "cell $name rc $? load $(tail -1 $O/cpu_load.log)"
}
log "q1106_stage start"
wait_quiet
( cd $T && PYTHONPATH=$T OMP_NUM_THREADS=8 taskset -c 56-63,168-175 python3 $K/preprocessor_cost.py --count 9 > $O/stage/preprocessor_cost.txt 2>&1 )
log "preprocessor cost rc $?"
C112=56-111,168-223 C16=56-63,168-175 C8=56-59,168-171
for threads in 2 4 6 8; do cell $C112 c112 8 $threads; done
cell $C112 c112 14 8
cell 0-223 c224 28 8
cell 0-223 c224 16 8
for threads in 4 8; do cell $C16 c16 8 $threads; done
cell $C16 c16 2 8
cell $C8 c8 1 8
log "q1106_stage done"
