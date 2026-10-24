#!/usr/bin/env bash
# 10-24, cards 0 and 1: #3129's fixes for its review (tasks/pr3129_review_20261024): the largest step measured with the
# longest reference prompt (F1), the pool charged its padding block, conv contexts and mapped bytes (F2), the share
# baseline read after a drain and a runtime headroom without the engine (F3, F1), a throwaway pool that holds no block
# serving without the cache (F4). The head comes from a bundle at $P/l4fix.bundle (same commits, not yet pushed).
# Steps: units of the touched suite on card 0 beside a default warm boot on card 1; the dedicated vocoder probe
# (probe_l4_review.py) on card 1; the pair p_l4r (main against the fixed head, c64, both orders).
# Decision rules, written before the run: units pass; the boot serves 64 of 64 with the step near 7.3 GB (rows of
# 4096 + 750 tokens against 4096) and the pool smaller than t_l4q's 50.4 GB by about the step's growth, the contexts
# (2 %) and the allocator's rounding; every probe case passes; p_l4r's c64 rows keep p_l4c's sign and stay inside
# p_l4c's spread plus p_aa2's band (req/s +6.9 ; +7.7 %, inter chunk p95 -12.2 ; -15.3 %).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261024/l4r/queue.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261024/queue_l4r.sh'
D=/data/c20261024
P=$D/l4r
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
MAIN=cd191df237f8c21e631133c6cf349164e6c1fcdf
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
mkdir -p $P
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
git -C /workspace/sglang-omni fetch -q $P/l4fix.bundle perf/cosyvoice3-prefix-pool-headroom
sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
if [ -d /workspace/t_l4r ]; then
  git -C /workspace/t_l4r checkout -q --detach $sha
else
  git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_l4r $sha
fi
log "tree t_l4r $(git -C /workspace/t_l4r log --oneline -1 | cut -c1-70) sglang_omni $(git -C /workspace/t_l4r rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/t_l4r diff --quiet HEAD -- sglang_omni || echo DIRTY)"
(cd /workspace/t_l4r && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$P/inductor_units \
  PYTHONPATH=/workspace/t_l4r:$COSYVOICE taskset -c $(cpus 0) \
  timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -rfEs \
  > $P/units.log 2>&1) &
units=$!
rm -rf $D/inductor_t_l4r
cp -a $D/inductor_t_l4q $D/inductor_t_l4r
bash $IN/warm_c20261021.sh t_l4r 1 $(cpus 1) 29700 $P/warm_t_l4r
wait $units
units_rc=$?
log "units t_l4r rc $units_rc $(grep -E 'passed|failed' $P/units.log | tail -n 1)"
log "warm t_l4r $(grep 'bench rc' $P/warm_t_l4r/progress.txt) $(grep -E 'Completed requests|Failed requests' $P/warm_t_l4r/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $P/warm_t_l4r/serve.log)"
for serve_log in $D/l4q/warm_t_l4q/serve.log $P/warm_t_l4r/serve.log; do
  log "startup $serve_log"
  grep -E 'Flow prefix cache|largest admitted step|prefix compile warmup' $serve_log | sed 's/^.*\] //' | sed 's/^/    /' >> $P/queue.txt
done
[ $units_rc = 0 ] || { log "units failed, stop"; exit 1; }
log "probe start"
rm -rf $P/inductor_probe
cp -a $D/inductor_t_l4r $P/inductor_probe
(cd /workspace/t_l4r && CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=$P/inductor_probe \
  PYTHONPATH=/workspace/t_l4r:$COSYVOICE taskset -c $(cpus 1) \
  timeout 1800 python3 $IN/profile_c20261024/probe_l4_review.py > $P/probe.log 2>&1)
log "probe rc $?"
grep -E '^\{' $P/probe.log | sed 's/^/    /' >> $P/queue.txt
grep -E 'Flow prefix cache|largest admitted step' $P/probe.log | sed 's/^.*\] //' | sed 's/^/    /' >> $P/queue.txt
(cd /workspace/sglang-omni && PAIR_CELLS=c64 python3 $IN/serve_pair_c20261021.py /workspace/t_mn \
  /workspace/t_l4r $P/p_l4r > $P/p_l4r.log 2>&1)
log "pair p_l4r rc $?"
python3 $IN/pooled_inter_chunk_c20261021.py $P/p_l4r c64 t_mn t_l4r > $P/p_l4r_pooled_c64.txt 2>&1
log "queue l4r done"
