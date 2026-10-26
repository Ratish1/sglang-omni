#!/usr/bin/env bash
# test_a_preempting_interrupt_drops_its_turn_at_every_owner_of_the_target on four trees, CPU only as the unit job runs
# it: main and the fix (wait for the sink's turn 2 append) 50 times each; the probe (the sink's append event put 0.5 s
# after its text, a late queue feeder made certain) on main and on the fix 10 times each. The probe on main failing
# every run and on the fix passing every run shows the failure is the read racing the event, and the fix closes it.
set -u
D=${D:?the results directory}
TEST=tests/unit_test/pipeline/test_session_graph.py::test_a_preempting_interrupt_drops_its_turn_at_every_owner_of_the_target
L=$D/queue.log
mkdir -p $D
echo "queue session start $(date +%T) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
for arm in "main ${MAIN:?} 50" "fix ${FIX:?} 50" "probe ${PROBE:?} 10" "probe_fixed ${PROBE_FIXED:?} 10"; do
  set -- $arm
  passed=0
  failed=0
  for attempt in $(seq $3); do
    if (cd $2 && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$2 timeout 300 python3 -m pytest "$TEST" -q -p no:cacheprovider \
      > $D/${1}_last.log 2>&1); then
      passed=$((passed + 1))
    else
      failed=$((failed + 1))
      cp $D/${1}_last.log $D/${1}_fail_$attempt.log
    fi
  done
  echo "$1 $(git -C $2 rev-parse --short=9 HEAD) passed $passed failed $failed $(grep -h -m1 -oE 'assert .{0,80}' $D/${1}_fail_*.log 2>/dev/null | head -n 1) $(date +%T)" >> $L
done
(cd ${FIX} && CUDA_VISIBLE_DEVICES="" PYTHONPATH=${FIX} timeout 900 python3 -m pytest tests/unit_test/pipeline/test_session_graph.py -q \
  -p no:cacheprovider -m "not benchmark and not accelerator" > $D/fix_file.log 2>&1)
echo "fix whole file rc $? $(tail -n 1 $D/fix_file.log) $(date +%T)" >> $L
echo "queue_session_done $(date +%T)" >> $L
