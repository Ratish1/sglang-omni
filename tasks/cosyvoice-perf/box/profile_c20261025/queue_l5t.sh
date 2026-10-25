#!/usr/bin/env bash
# 10-25, cards 0 and 1: #3133's review F1 (tasks/pr3133_review_20261025): the new case runs the TTS engine on a two
# layer synthetic checkpoint in its own process and checks each CUDA prefill's graph replay against the eager prefill
# over the same composed rows. The head and a mutation come from $P/l5t.bundle.
# Steps: the touched file test_model_runner.py on the head (t_l5f, card 0) beside the new case on the pre-PR model
# files (t_l5pre: the head with the four fun_cosyvoice3 files of L6's head 0b9be5746, card 1); then the new case on
# the head with the forward's omni_prefill_rids removed (t_l5mut, card 1).
# Decision rules, written before the run: the file passes on t_l5f with the new case among the passed; the new case
# fails on t_l5pre (no prefill reaches the standard forward) and on t_l5mut (the forward refuses the keyword).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261025/l5t/queue.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261025/queue_l5t.sh'
D=/data/c20261025
P=$D/l5t
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
TEST=tests/unit_test/fun_cosyvoice3/test_model_runner.py
CASE=$TEST::test_cuda_prefill_graphs_replay_the_composed_rows_as_the_eager_forward
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-prefix-graph-ladder-v2
for pair in t_l5f:perf/cosyvoice3-prefill-graphs-v3 t_l5pre:perf/cosyvoice3-prefill-graphs-v3 t_l5mut:scratch/l5-no-prefill-rids; do
  name=${pair%%:*}
  git -C /workspace/sglang-omni fetch -q $P/l5t.bundle ${pair#*:}
  sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
  [ -d /workspace/$name ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
done
git -C /workspace/t_l5pre checkout -q 0b9be5746 -- sglang_omni/models/fun_cosyvoice3/__init__.py \
  sglang_omni/models/fun_cosyvoice3/engine_builder.py sglang_omni/models/fun_cosyvoice3/model_runner.py \
  sglang_omni/models/fun_cosyvoice3/sglang_model.py
for name in t_l5f t_l5pre t_l5mut; do
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-70) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done
run() {
  name=$1 gpu=$2 target=$3
  (cd /workspace/$name && CUDA_VISIBLE_DEVICES=$gpu TORCHINDUCTOR_CACHE_DIR=$P/inductor_$name \
    PYTHONPATH=/workspace/$name:$COSYVOICE taskset -c $(cpus $gpu) \
    timeout 3600 python3 -m pytest $target -q -p no:cacheprovider -rfEs --durations=3 > $P/$name.log 2>&1
  log "$name rc $? $(grep -E 'passed|failed' $P/$name.log | tail -n 1)")
}
run t_l5f 0 $TEST &
run t_l5pre 1 $CASE
run t_l5mut 1 $CASE
wait
for name in t_l5pre t_l5mut; do
  grep -m2 -E "AssertionError|TypeError|assert any" $P/$name.log | cut -c1-200 | sed "s/^/    $name: /" >> $P/queue.txt
done
log "queue l5t done"
