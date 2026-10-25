#!/usr/bin/env bash
# 10-25, cards 0 and 1, after queue_l4t: L6 and L5 with #3129's rereview fixes merged in (bundle $P/l65t.bundle, not
# yet pushed). Units of the touched suite of each, one card each; the fixes change no CUDA sizing, so no boot.
# Decision rule, written before the run: both suites pass, each one case more than on 10-24 (L6 289, L5 293).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261025/l4t/queue65.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261025/queue_l65t.sh'
D=/data/c20261025
P=$D/l4t
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
log() { echo "$* $(date +%T)" >> $P/queue65.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
for pair in t_l6t:perf/cosyvoice3-prefix-graph-ladder-v2 t_l5t:perf/cosyvoice3-prefill-graphs-v3; do
  name=${pair%%:*}
  branch=${pair#*:}
  git -C /workspace/sglang-omni fetch -q $P/l65t.bundle $branch
  sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
  [ -d /workspace/$name ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-70) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done
gpu=0
for name in t_l6t t_l5t; do
  (cd /workspace/$name && CUDA_VISIBLE_DEVICES=$gpu TORCHINDUCTOR_CACHE_DIR=$P/inductor_units_$name \
    PYTHONPATH=/workspace/$name:$COSYVOICE taskset -c $(cpus $gpu) \
    timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -rfEs \
    > $P/units_$name.log 2>&1; echo "units $name rc $? $(grep -E 'passed|failed' $P/units_$name.log | tail -n 1) $(date +%T)" >> $P/queue65.txt) &
  gpu=1
done
wait
log "queue l65t done"
