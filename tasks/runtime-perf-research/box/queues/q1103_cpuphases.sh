#!/usr/bin/env bash
# 11-03, after q1103_identity, node-0000, nothing serving: a video request costs 5 to 7 CPU seconds against about 0.6 s of wall.
# Each phase's CPU, one request at a time, on the 50 Video-MME CI videos: OMP 8 as the colocated config writes, then with idle
# OpenMP threads not spinning, decode on 8 FFmpeg threads and on the reader alone, OMP 1, and the two-card default's 16.
set -u
O=/data/rp1103n/cpuphases
K=/workspace/kit_c/tasks/runtime-perf-research/box
T=/workspace/wk_bench
CPUS=56-111,168-223
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> /data/rp1103n/queue.log; }
until grep -q "q1103_identity done" /data/rp1103n/queue.log 2>/dev/null; do sleep 20; done
log "q1103_cpuphases start"
cell() {
  local name=$1 omp=$2 decode=$3 policy=$4
  ( cd $T && env OMP_NUM_THREADS=$omp ${policy:+OMP_WAIT_POLICY=$policy} PYTHONPATH=$T taskset -c $CPUS python3 $K/video_cpu_phases.py \
    --decode-threads $decode --out $O/$name.json > $O/$name.log 2>&1 )
  log "phases $name rc $?"
}
cell omp8_d16 8 16 ""
cell omp8_d16_again 8 16 ""
cell omp8_d16_passive 8 16 PASSIVE
cell omp8_d8 8 8 ""
cell omp8_d4 8 4 ""
cell omp1_d16 1 16 ""
cell omp16_d16 16 16 ""
log "q1103_cpuphases done"
