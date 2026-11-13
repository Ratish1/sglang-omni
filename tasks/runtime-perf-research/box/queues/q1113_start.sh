#!/usr/bin/env bash
# 11-13, node-0000, one H200 (card 0, NUMA 1, CPUs 56-111,168-223). Two independent lines: the card's first boots on upstream main
# 35890f10c and on #3451's final tree (compile, numbers not read), pinned to the card's socket; and the serial preprocessing OpenMP
# profile on the other socket, CPU only. The profile repeats its first two settings at the end to show drift from the fill.
set -u
O=/data/rp1113
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main LANE=/workspace/lane P3S=/workspace/p3s
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223 OTHER_CPUS=0-55,112-167
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
nohup setsid python3 $K/cpu_load_sampler.py $O/cpu_load.log > /dev/null 2>&1 < /dev/null &
fill_line() {
  for tree in main p3s; do
    log "fill $tree $(git -C /workspace/$tree rev-parse --short=12 HEAD:sglang_omni) start"
    taskset -c $CARD_CPUS python3 $K/baseline.py --tree /workspace/$tree --model omni_colocated --card 0 --port 28400 \
      --out $O/fill_$tree --loads 16 --passes warm > $O/fill_$tree.driver.log 2>&1
    log "fill $tree rc $?"
  done
  log "fill done"
}
profile_line() {
  mkdir -p $O/omp
  for setting in 8:unset 8:PASSIVE 1:unset 28:PASSIVE 28:unset 14:PASSIVE 14:unset 112:unset 8:PASSIVE 8:unset; do
    local threads=${setting%%:*} policy=${setting##*:}
    local name=omp_${threads}_${policy}_$(date -u +%H%M%S)
    log "profile $name start"
    if [ "$policy" = unset ]; then
      (cd $LANE && env -u OMP_WAIT_POLICY OMP_NUM_THREADS=$threads PYTHONPATH=$LANE taskset -c $OTHER_CPUS \
        python3 $K/omp_serial_profile.py run --out $O/omp/$name.json > $O/omp/$name.log 2>&1)
    else
      (cd $LANE && OMP_WAIT_POLICY=$policy OMP_NUM_THREADS=$threads PYTHONPATH=$LANE taskset -c $OTHER_CPUS \
        python3 $K/omp_serial_profile.py run --out $O/omp/$name.json > $O/omp/$name.log 2>&1)
    fi
    log "profile $name rc $?"
  done
  log "profile done"
}
log "q1113_start start"
fill_line &
fill_pid=$!
profile_line &
profile_pid=$!
wait $fill_pid $profile_pid
log "q1113_start done"
