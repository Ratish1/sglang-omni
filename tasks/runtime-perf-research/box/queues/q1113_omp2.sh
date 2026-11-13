#!/usr/bin/env bash
# 11-13, node-0000, after q1113_omp: team sizes between 1 and 8 and the two-card lane's 28, both policies, plus repeats of 8 and 14
# unset, so the serial path's wall, wake cost and spin CPU are read across the team sizes that matter.
set -u
O=/data/rp1113
K=/workspace/kit/tasks/runtime-perf-research/box
LANE=/workspace/lane
OTHER_CPUS=0-55,112-167
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1113_omp done" $O/queue.log 2>/dev/null; do sleep 5; done
log "q1113_omp2 start"
for setting in 2:unset 2:PASSIVE 4:unset 4:PASSIVE 28:unset 28:PASSIVE 14:unset 8:unset; do
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
log "q1113_omp2 done"
