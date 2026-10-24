#!/usr/bin/env bash
# 10-24, cards 0 and 1: #3129's largest step measured over the non-streaming path too (ee659bfe5, bundle $P/l4n.bundle,
# not yet pushed). Units beside a default warm boot on card 1, then probe_l4_nonstream.py on card 1.
# Decision rules, written before the run: units pass; the default boot serves 64 of 64 and its startup line gives
# the streaming and the non-streaming peaks, the pool staying at 56960 frames when the streaming peak is the larger;
# at max_batch_size 4 the reserve is the non-streaming peak, and the real decode of the longest request alone and of
# a 4 request batch at the admission budget each peak at or under it.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261024/l4n/queue.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261024/queue_l4n.sh'
D=/data/c20261024
P=$D/l4n
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
export PAIR_HOME=$D
mkdir -p $P
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
git -C /workspace/sglang-omni fetch -q $P/l4n.bundle perf/cosyvoice3-prefix-pool-headroom
sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
if [ -d /workspace/t_l4n ]; then
  git -C /workspace/t_l4n checkout -q --detach $sha
else
  git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_l4n $sha
fi
log "tree t_l4n $(git -C /workspace/t_l4n log --oneline -1 | cut -c1-70) sglang_omni $(git -C /workspace/t_l4n rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/t_l4n diff --quiet HEAD -- sglang_omni || echo DIRTY)"
(cd /workspace/t_l4n && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$P/inductor_units \
  PYTHONPATH=/workspace/t_l4n:$COSYVOICE taskset -c $(cpus 0) \
  timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -rfEs \
  > $P/units.log 2>&1; echo "units t_l4n rc $? $(grep -E 'passed|failed' $P/units.log | tail -n 1)" >> $P/queue.txt) &
units=$!
rm -rf $D/inductor_t_l4n
cp -a $D/inductor_t_l4r $D/inductor_t_l4n
bash $IN/warm_c20261021.sh t_l4n 1 $(cpus 1) 29700 $P/warm_t_l4n
log "warm t_l4n $(grep 'bench rc' $P/warm_t_l4n/progress.txt) $(grep -E 'Completed requests|Failed requests' $P/warm_t_l4n/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $P/warm_t_l4n/serve.log)"
grep -E 'Flow prefix cache|largest admitted step' $P/warm_t_l4n/serve.log | sed 's/^.*\] //' | sed 's/^/    /' >> $P/queue.txt
log "probe start"
rm -rf $P/inductor_probe
cp -a $D/inductor_t_l4n $P/inductor_probe
(cd /workspace/t_l4n && CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=$P/inductor_probe \
  PYTHONPATH=/workspace/t_l4n:$COSYVOICE taskset -c $(cpus 1) \
  timeout 1200 python3 $IN/profile_c20261024/probe_l4_nonstream.py > $P/probe.log 2>&1)
log "probe rc $? tracebacks $(grep -c Traceback $P/probe.log)"
grep -E '^\{' $P/probe.log | sed 's/^/    /' >> $P/queue.txt
grep -E 'Flow prefix cache|largest admitted step' $P/probe.log | sed 's/^.*\] //' | sed 's/^/    /' >> $P/queue.txt
wait $units
log "queue l4n done"
