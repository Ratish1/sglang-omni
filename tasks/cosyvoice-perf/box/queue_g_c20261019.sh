#!/usr/bin/env bash
# 10-19, card 3, after queue f: the prefix layout busy-stream test (55ae12bea) at S2's head and with its fix reverted
# (the test file of 55ae12bea on 3e39412a1, the commit before 815bcdace), so the test is seen to fail without the
# pinned upload and pass with it.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261019/queue_g.pgid; exec bash queue_g_c20261019.sh'
D=/data/c20261019
TEST=tests/unit_test/fun_cosyvoice3/test_prefix_cache_gpu.py::test_prefix_layouts_upload_behind_queued_work_without_a_host_wait
log() { echo "$* $(date +%T)" >> $D/queue_g.txt; }

log "queue start"
cd /workspace/sglang-omni
git fetch -q origin perf/cosyvoice3-vocoder-overlap
[ -d /workspace/t_s2t ] || git worktree add -q --detach /workspace/t_s2t FETCH_HEAD
[ -d /workspace/t_c2pre ] || git worktree add -q --detach /workspace/t_c2pre 3e39412a1
git -C /workspace/t_c2pre checkout -q FETCH_HEAD -- tests/unit_test/fun_cosyvoice3/test_prefix_cache_gpu.py
for tree in t_s2t t_c2pre; do
  (
    cd /workspace/$tree
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/$tree taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
      "$TEST" > $D/s2/busy_test_$tree.log 2>&1
    rc=$?
    echo "busy test $tree $(git log --oneline -1 | cut -c1-9) rc $rc $(grep -E 'passed|failed' $D/s2/busy_test_$tree.log | tail -n 1) $(date +%T)" >> $D/queue_g.txt
  )
done
log "queue done"
