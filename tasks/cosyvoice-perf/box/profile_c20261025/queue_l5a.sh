#!/usr/bin/env bash
# 10-26 (UTC 10-25 evening), cards 0 and 1: #3133's rereview F1 (tasks/pr3133_rereview_20261026): the replay case's
# eager reference read the same sidecar rows as the graph, so attaching zeros passed. The fix composes the reference
# from each request's enqueued prompt rows over its extend window.
# Steps: the replay case on the fix (t_l5a, scratch/l5-anchor, card 0) beside the same case on the fix with zeros
# attached as the prefill rows (t_l5z, scratch/l5-zeroed-handoff, card 1).
# Decision rules, written before the run: the case passes on t_l5a and fails on t_l5z in assert_close.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261025b/l5a/queue.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261025/queue_l5a.sh'
D=/data/c20261025b
P=$D/l5a
CASE=tests/unit_test/fun_cosyvoice3/test_model_runner.py::test_cuda_prefill_graphs_replay_the_composed_rows_as_the_eager_forward
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
for pair in t_l5a:scratch/l5-anchor t_l5z:scratch/l5-zeroed-handoff; do
  name=${pair%%:*}
  git -C /workspace/sglang-omni fetch -q origin ${pair#*:}
  sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
  [ -d /workspace/$name ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-70) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) tests $(git -C /workspace/$name rev-parse HEAD:tests | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD || echo DIRTY)"
done
run() {
  name=$1 gpu=$2
  (cd /workspace/$name && CUDA_VISIBLE_DEVICES=$gpu TORCHINDUCTOR_CACHE_DIR=$P/inductor_$name \
    PYTHONPATH=/workspace/$name taskset -c $(cpus $gpu) \
    timeout 1800 python3 -m pytest $CASE -q -p no:cacheprovider -rfEs --durations=3 > $P/$name.log 2>&1
  log "$name rc $? $(grep -E 'passed|failed' $P/$name.log | tail -n 1)")
}
run t_l5a 0 &
run t_l5z 1
wait
grep -m3 -E "AssertionError|Mismatched|Greatest absolute" $P/t_l5z.log | cut -c1-200 | sed "s/^/    t_l5z: /" >> $P/queue.txt
log "queue l5a done"
