#!/usr/bin/env bash
# 10-24, cards 0 and 1: #3129's review fixes (the tokenizer's reference cap shared with its warmup, the warmup on every
# device, the throwaway prefix pool sized from the warmup's hops, the vocoder's memory read through torch.accelerator).
# Units of the touched suite on card 0 beside a warm boot of the new head on card 1, then the startup lines against
# t_l4p's warm boot (the head before the fixes).
# Decision rule, written before the run: units pass, the boot serves 64 of 64, and the reserve and pool lines match
# t_l4p's (prefix graphs 0.26 GB, engine headroom 11.79 GB, step 6.17 GB, pool about 50 GB); the throwaway pool grows
# from 256 to 384 frames (6 blocks), since it now holds the warmup's follow-up hop.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261024/l4q/queue.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261024/queue_l4q.sh'
D=/data/c20261024
P=$D/l4q
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
export PAIR_HOME=$D
mkdir -p $P
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-prefix-pool-headroom
sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
if [ -d /workspace/t_l4q ]; then
  git -C /workspace/t_l4q checkout -q --detach $sha
else
  git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_l4q $sha
fi
log "tree t_l4q $(git -C /workspace/t_l4q log --oneline -1 | cut -c1-70) sglang_omni $(git -C /workspace/t_l4q rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/t_l4q diff --quiet HEAD -- sglang_omni || echo DIRTY)"
(cd /workspace/t_l4q && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$P/inductor_units \
  PYTHONPATH=/workspace/t_l4q:$COSYVOICE taskset -c $(cpus 0) \
  timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -rfEs \
  > $P/units.log 2>&1) &
units=$!
rm -rf $D/inductor_t_l4q
cp -a $D/inductor_t_l4p $D/inductor_t_l4q
bash $IN/warm_c20261021.sh t_l4q 1 $(cpus 1) 29700 $P/warm_t_l4q
wait $units
log "units t_l4q rc $? $(grep -E 'passed|failed' $P/units.log | tail -n 1)"
log "warm t_l4q $(grep 'bench rc' $P/warm_t_l4q/progress.txt) $(grep -E 'Completed requests|Failed requests' $P/warm_t_l4q/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $P/warm_t_l4q/serve.log)"
for name in t_l4p t_l4q; do
  if [ $name = t_l4p ]; then
    serve_log=$D/l4p/warm_t_l4p/serve.log
  else
    serve_log=$P/warm_t_l4q/serve.log
  fi
  log "startup $name"
  grep -E 'Flow prefix cache|largest admitted step|prefix compile warmup' $serve_log | sed 's/^.*\] //' | sed 's/^/    /' >> $P/queue.txt
done
log "queue l4q done"
