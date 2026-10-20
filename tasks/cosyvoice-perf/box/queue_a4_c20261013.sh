#!/usr/bin/env bash
# The review's F1 tests (#2782 head 5f9d8cdce plus the two cross-stream tests, a4_tests.patch) on three trees from
# main 204ba29a5 with a2 and a3 applied: t_a4 unchanged, t_a4_no_wait without the AR side's wait_event, t_a4_no_record
# without its record_stream. Each tree runs the three CUDA tests five times, then t_a4 runs the CosyVoice and
# preprocessing units. usage (in the container): bash queue_a4_c20261013.sh
set -u
R=/data/c20261013
P=$R/patches/m204ba29a5_20261013
OUT=$R/a4
mkdir -p $OUT
RB=sglang_omni/models/fun_cosyvoice3/request_builders.py
for tree in t_a4 t_a4_no_wait t_a4_no_record; do
  git -C /workspace/sglang-omni worktree add -q --detach /workspace/$tree 204ba29a5
  git -C /workspace/$tree apply -3 $P/a2_afd99d535.patch $P/a3_5f9d8cdce.patch $P/a4_tests.patch
  git -C /workspace/$tree add -A && git -C /workspace/$tree reset -q
done
sed -i '/stream.wait_event(prepared.ready_event)/d' /workspace/t_a4_no_wait/$RB
sed -i '/prepared.prompt_input_embeds.record_stream(stream)/d' /workspace/t_a4_no_record/$RB
for tree in t_a4 t_a4_no_wait t_a4_no_record; do
  echo "$tree changed=$(git -C /workspace/$tree status --short | wc -l)" \
    "wait=$(grep -c 'stream.wait_event(prepared.ready_event)' /workspace/$tree/$RB)" \
    "record=$(grep -c 'record_stream(stream)' /workspace/$tree/$RB)" >> $OUT/queue.txt
done
TESTS="tests/unit_test/fun_cosyvoice3/test_request_builders.py::test_cuda_preprocessing_builds_prompt_rows_without_a_host_wait"
TESTS="$TESTS tests/unit_test/fun_cosyvoice3/test_request_builders.py::test_cuda_prompt_rows_are_read_only_after_preprocessing_writes_them"
TESTS="$TESTS tests/unit_test/fun_cosyvoice3/test_request_builders.py::test_cuda_prompt_rows_stay_allocated_until_the_ar_stream_reads_them"
for tree in t_a4 t_a4_no_wait t_a4_no_record; do
  for round in 1 2 3 4 5; do
    (cd /workspace/$tree && PYTHONPATH=/workspace/$tree timeout 600 python3 -m pytest -q -p no:cacheprovider -rA $TESTS) \
      > $OUT/${tree}_$round.log 2>&1
    echo "$tree round $round rc $? $(grep -E '^(PASSED|FAILED)' $OUT/${tree}_$round.log | sed 's/.*:://' | tr '\n' ' ')" \
      >> $OUT/queue.txt
  done
done
(cd /workspace/t_a4 && PYTHONPATH=/workspace/t_a4 timeout 900 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/fun_cosyvoice3 tests/unit_test/preprocessing) > $OUT/units_t_a4.log 2>&1
echo "units t_a4 rc $? $(grep -E 'passed|failed' $OUT/units_t_a4.log | tail -1)" >> $OUT/queue.txt
echo "queue done $(date +%T)" >> $OUT/queue.txt
