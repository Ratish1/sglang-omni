#!/usr/bin/env bash
# 11-13, node-0000, after q1113_p3d, with no server on the card: the unit tests of each final head, the new tests shown failing on the
# code before each fix (copied into the older tree and run there), and three mutants of the lane's routing on a scratch tree.
set -u
O=/data/rp1113
K=/workspace/kit/tasks/runtime-perf-research/box
U=$O/units
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1113_p3d done" $O/queue.log 2>/dev/null; do sleep 15; done
mkdir -p $U
log "q1113_units start"
run() {
  local name=$1 tree=$2
  shift 2
  (cd $tree && timeout 3000 python3 -m pytest -q -p no:cacheprovider "$@" > $U/$name.txt 2>&1)
  log "units $name rc $? $(tail -1 $U/$name.txt | cut -c1-90)"
}
git -C /workspace/p3d fetch -q origin test/3p-decode-sync-20261113 && git -C /workspace/p3d checkout -q --detach FETCH_HEAD
log "p3d tests at $(git -C /workspace/p3d rev-parse --short HEAD) $(git -C /workspace/p3d rev-parse --short=12 HEAD:sglang_omni)"

run p3s_omni /workspace/p3s tests/unit_test/qwen3_omni
cp /workspace/p3s/tests/unit_test/qwen3_omni/test_logit_shaping.py /workspace/main/tests/unit_test/qwen3_omni/test_prefix_logit_shaping.py
run p3s_new_tests_on_main /workspace/main tests/unit_test/qwen3_omni/test_prefix_logit_shaping.py
rm -f /workspace/main/tests/unit_test/qwen3_omni/test_prefix_logit_shaping.py

run p3d_talker /workspace/p3d tests/unit_test/qwen3_omni/test_talker.py tests/unit_test/qwen3_omni/test_logit_shaping.py
cp /workspace/p3d/tests/unit_test/qwen3_omni/test_talker.py /workspace/p3s/tests/unit_test/qwen3_omni/test_prefix_talker.py
run p3d_new_test_on_p3s /workspace/p3s tests/unit_test/qwen3_omni/test_prefix_talker.py -k do_not_synchronize
rm -f /workspace/p3s/tests/unit_test/qwen3_omni/test_prefix_talker.py

run wait_omni /workspace/lane_wait tests/unit_test/qwen3_omni
run wait_pre /workspace/lane_wait tests/unit_test/preprocessing tests/unit_test/scheduling
cd /workspace/sglang-omni && git worktree add -q --detach /workspace/lane_mut $(git -C /workspace/lane_wait rev-parse HEAD) 2>/dev/null
for patch in lane_route_all lane_route_none lane_no_abort_check; do
  git -C /workspace/lane_mut checkout -q -- .
  git -C /workspace/lane_mut apply --unidiff-zero $K/mutants/$patch.patch && log "mutant $patch applied"
  run mutant_$patch /workspace/lane_mut tests/unit_test/qwen3_omni/test_pipeline.py -k "preprocessing or request_has_video"
done
git -C /workspace/lane_mut checkout -q -- .
log "q1113_units done"
