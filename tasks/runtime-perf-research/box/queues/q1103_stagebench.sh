#!/usr/bin/env bash
# 11-03, after q1103_gate, node-0000, nothing serving: the preprocessing stage alone on the 50 Video-MME CI videos. Its limit
# under served pairs looks like decode CPU (media wait doubles while processor calls stay near 1.6 at once) and its peak memory
# has no bound (15.9 GB at c16). Videos in flight K from none down to 2, at 112, 56, 16 and 8 CPUs, c16 and c64, then processors
# N at K's settled value. One fresh process a cell, started under taskset. CPU only.
set -u
O=/data/rp1103n/stage
K=/workspace/kit_b/tasks/runtime-perf-research/box
T=/workspace/wk_bench
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> /data/rp1103n/queue.log; }
until grep -q "q1103_gate done" /data/rp1103n/queue.log 2>/dev/null; do sleep 20; done
log "q1103_stagebench start"
cell() {
  local cpus=$1 tag=$2 concurrency=$3 processors=$4 inflight=$5
  local name=${tag}_c${concurrency}_n${processors}_k${inflight}
  ( cd $T && OMP_NUM_THREADS=8 PYTHONPATH=$T taskset -c $cpus python3 $K/preprocess_stage_bench.py --concurrency $concurrency \
    --processors $processors --videos-in-flight $inflight --out $O/$name.json > $O/$name.log 2>&1 )
  log "cell $name rc $?"
}
for setting in "56-111,168-223 c112" "56-83,168-195 c56" "56-63,168-175 c16" "56-59,168-171 c8"; do
  set -- $setting
  for inflight in 0 8 6 4 2; do
    cell $1 $2 16 2 $inflight
  done
  cell $1 $2 64 2 0
  cell $1 $2 64 2 4
done
for processors in 1 3 4; do
  cell 56-111,168-223 c112 16 $processors 0
  cell 56-111,168-223 c112 16 $processors 4
done
log "q1103_stagebench done"
