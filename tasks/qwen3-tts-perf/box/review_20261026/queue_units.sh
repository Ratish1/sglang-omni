#!/usr/bin/env bash
# CI's two unit jobs on a tree as test.yaml runs them (CPU only, "not benchmark and not accelerator", the duration
# partition of 2), without -x and without the retry wrapper so every failing test shows; then each failing test alone
# 30 times and its file 10 times, which separates a deterministic failure from an order or timing one.
set -u
D=${D:?the results directory}
M=${M:?the tree}
L=$D/queue.log
mkdir -p $D
echo "queue units start $(date +%T) tree $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $M diff --quiet HEAD -- sglang_omni tests || echo DIRTY) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
run_partition() {
  (cd $M && CUDA_VISIBLE_DEVICES="" OMNI_CI_STAGE_LABEL=unit-test-$2 PYTHONPATH=$M timeout 3600 python3 -m pytest tests/ -v \
    -m "not benchmark and not accelerator" -p scripts.ci.utils.pytest_partition --auto-partition-id $1 \
    --auto-partition-size 2 --partition-model-file .github/unit_test_durations.json -p no:cacheprovider -rfE \
    > $D/partition_$2.log 2>&1)
  echo "partition $2 rc $? $(tail -n 1 $D/partition_$2.log) $(date +%T)" >> $L
  grep -E '^(FAILED|ERROR) ' $D/partition_$2.log | cut -c1-300 | sed "s/^/partition $2: /" >> $L
}
run_partition 0 a &
run_partition 1 b &
wait
grep -hE '^(FAILED|ERROR) ' $D/partition_a.log $D/partition_b.log | awk '{print $2}' | sort -u > $D/failing.txt
while read -r test_id; do
  passed=0
  failed=0
  for attempt in $(seq 30); do
    if (cd $M && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$M timeout 600 python3 -m pytest "$test_id" -q -p no:cacheprovider \
      > $D/alone_last.log 2>&1); then
      passed=$((passed + 1))
    else
      failed=$((failed + 1))
      cp $D/alone_last.log "$D/alone_fail_$(echo "$test_id" | tr '/:[]' '____')_$attempt.log"
    fi
  done
  echo "alone $test_id passed $passed failed $failed $(date +%T)" >> $L
  test_file=${test_id%%::*}
  passed=0
  failed=0
  for attempt in $(seq 10); do
    if (cd $M && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$M timeout 900 python3 -m pytest "$test_file" -q -p no:cacheprovider \
      -m "not benchmark and not accelerator" > $D/file_last.log 2>&1); then
      passed=$((passed + 1))
    else
      failed=$((failed + 1))
      cp $D/file_last.log "$D/file_fail_$(echo "$test_file" | tr '/:[]' '____')_$attempt.log"
    fi
  done
  echo "file $test_file passed $passed failed $failed $(date +%T)" >> $L
done < $D/failing.txt
echo "queue_units_done $(date +%T)" >> $L
