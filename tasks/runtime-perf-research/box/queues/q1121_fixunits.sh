#!/usr/bin/env bash
# 11-21, node-0001, CPU only: the graph prefill change (PFIX 09dae918c on main e0482a08c) and its stacked trees by hash,
# the unit suites the diff reaches, the new tests on main's sources (they must fail), and each guarantee reverted alone
# (its test must fail).
set -u
O=/data/v1121/units
K=/workspace/kit/tasks/runtime-perf-research/box
PFIX=09dae918c109422a8de89e17b50332239848cedf
NSTACK=16517682307f1575cebdcbbdc16b0b48547b5101
NSFIX=065927d2f52b52333ef132937e36686e721c64fc
NMAIN=e0482a08cf48e6333952c974f3a5e3a4a3398361
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> /data/v1121/queue.log; }
log "q1121_fixunits start"
cd /workspace/sglang-omni
git fetch -q origin $PFIX $NSTACK $NSFIX
git worktree add -q --detach /workspace/pfix $PFIX
git worktree add -q --detach /workspace/nstack $NSTACK
git worktree add -q --detach /workspace/nsfix $NSFIX
git worktree add -q --detach /workspace/prefix $NMAIN
for tree in pfix nstack nsfix prefix nmain; do
  log "tree $tree $(git -C /workspace/$tree rev-parse --short HEAD) $(git -C /workspace/$tree rev-parse --short=12 HEAD:sglang_omni)"
done
SUITES="tests/unit_test/qwen3_omni tests/unit_test/minicpm_o/test_thinker_model_runner.py tests/unit_test/pipeline/test_scheduler.py"
cd /workspace/pfix
PYTHONPATH=/workspace/pfix python3 -m pytest -q -p no:cacheprovider $SUITES > $O/pfix_units.log 2>&1
log "units pfix rc $? $(grep -E 'passed|failed' $O/pfix_units.log | tail -1)"
cd /workspace/prefix
git checkout -q $PFIX -- tests/unit_test/qwen3_omni/test_thinker_prefill_contract.py
PYTHONPATH=/workspace/prefix python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/qwen3_omni/test_thinker_prefill_contract.py > $O/prefix_new_tests.log 2>&1
log "new tests on main's sources rc $? $(grep -E 'passed|failed' $O/prefix_new_tests.log | tail -1)"
git checkout -q -- .
git worktree add -q --detach /workspace/pmut $PFIX
cd /workspace/pmut
while read -r patch test; do
  git checkout -q -- .
  git apply $K/mutants/$patch.patch || { log "mutant $patch did not apply"; continue; }
  PYTHONPATH=/workspace/pmut python3 -m pytest -q -p no:cacheprovider "$test" > $O/$patch.log 2>&1
  rc=$?
  log "mutant $patch rc $rc ($([ $rc = 0 ] && echo SURVIVED || echo killed)) $(grep -E 'passed|failed' $O/$patch.log | tail -1)"
done <<'EOF'
graphprefill_visual_inputs_eager tests/unit_test/qwen3_omni/test_thinker_prefill_contract.py::test_a_chunk_past_every_visual_row_prefills_on_the_sidecar
graphprefill_split_deepstack_empty tests/unit_test/qwen3_omni/test_thinker_prefill_contract.py::test_a_chunk_past_every_visual_row_prefills_on_the_sidecar
EOF
git checkout -q -- .
log "q1121_fixunits done"
