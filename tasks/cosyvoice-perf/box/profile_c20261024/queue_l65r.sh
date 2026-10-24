#!/usr/bin/env bash
# 10-24, cards 0 and 1, after queue_l4r: L6 and L5 with #3129's review fixes merged in (bundle $P/l65fix.bundle, not yet
# pushed). Units of the touched suite of each, one card each, then a default warm boot of L5 (the whole stack).
# Decision rule, written before the run: both suites pass, and the L5 boot serves 64 of 64 with the step at 7.40 GB,
# the prefix graphs near L6's 0.62 GB and the pool sized around them.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261024/l4r/queue65.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261024/queue_l65r.sh'
D=/data/c20261024
P=$D/l4r
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $P/queue65.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
for pair in t_l6r:perf/cosyvoice3-prefix-graph-ladder-v2 t_l5r:perf/cosyvoice3-prefill-graphs-v3; do
  name=${pair%%:*}
  branch=${pair#*:}
  git -C /workspace/sglang-omni fetch -q $P/l65fix.bundle $branch
  sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
  if [ -d /workspace/$name ]; then
    git -C /workspace/$name checkout -q --detach $sha
  else
    git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
  fi
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-70) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done
gpu=0
for name in t_l6r t_l5r; do
  (cd /workspace/$name && CUDA_VISIBLE_DEVICES=$gpu TORCHINDUCTOR_CACHE_DIR=$P/inductor_units_$name \
    PYTHONPATH=/workspace/$name:$COSYVOICE taskset -c $(cpus $gpu) \
    timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -rfEs \
    > $P/units_$name.log 2>&1; echo "units $name rc $? $(grep -E 'passed|failed' $P/units_$name.log | tail -n 1)" >> $P/queue65.txt) &
  gpu=1
done
wait
rm -rf $D/inductor_t_l5r
cp -a $D/inductor_t_l5q $D/inductor_t_l5r
bash $IN/warm_c20261021.sh t_l5r 1 $(cpus 1) 29700 $P/warm_t_l5r
log "warm t_l5r $(grep 'bench rc' $P/warm_t_l5r/progress.txt) $(grep -E 'Completed requests|Failed requests' $P/warm_t_l5r/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $P/warm_t_l5r/serve.log)"
grep -E 'Flow prefix cache|largest admitted step|prefix compile warmup|prefill graph' $P/warm_t_l5r/serve.log | sed 's/^.*\] //' | sed 's/^/    /' >> $P/queue65.txt
log "queue l65r done"
