#!/usr/bin/env bash
# 11-13, node-0000: the serial preprocessing OpenMP profile on the socket the card does not use, CPU only, after the first try in
# q1113_start failed on a missing constructor argument. Its first two settings repeat at the end to show drift from the card's boots.
set -u
O=/data/rp1113
K=/workspace/kit/tasks/runtime-perf-research/box
LANE=/workspace/lane
OTHER_CPUS=0-55,112-167
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "profile done" $O/queue.log 2>/dev/null; do sleep 5; done
mkdir -p $O/omp
log "q1113_omp start"
for setting in 8:unset 8:PASSIVE 1:unset 28:PASSIVE 28:unset 14:PASSIVE 14:unset 112:unset 8:PASSIVE 8:unset; do
  threads=${setting%%:*}
  policy=${setting##*:}
  name=omp_${threads}_${policy}_$(date -u +%H%M%S)
  log "omp $name start"
  if [ "$policy" = unset ]; then
    (cd $LANE && env -u OMP_WAIT_POLICY OMP_NUM_THREADS=$threads PYTHONPATH=$LANE taskset -c $OTHER_CPUS \
      python3 $K/omp_serial_profile.py run --out $O/omp/$name.json > $O/omp/$name.log 2>&1)
  else
    (cd $LANE && OMP_WAIT_POLICY=$policy OMP_NUM_THREADS=$threads PYTHONPATH=$LANE taskset -c $OTHER_CPUS \
      python3 $K/omp_serial_profile.py run --out $O/omp/$name.json > $O/omp/$name.log 2>&1)
  fi
  log "omp $name rc $?"
done
log "q1113_omp done"
