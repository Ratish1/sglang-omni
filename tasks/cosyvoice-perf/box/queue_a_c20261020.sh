#!/usr/bin/env bash
# 10-20, cards 2 and 3 (CPUs 0-15,64-79 and 16-31,80-95, NUMA node 0). The PR candidate ported onto upstream main
# b98a28936 (prof/cosyvoice3-candidate ba5c7fccb, tree t_c20: S2 with its three launch fixes and fix 1, main merged)
# against main (t_m20). The touched suites, both warmed, both profiled at the same time (c16, c64) with every section's
# device and host time per finished request (section_ledger_c20261020.py) and each request's states, then the pair
# p_c20 (c1 seeded, c1u, c16, c64, r16, both orders).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_a.pgid; exec bash queue_a_c20261020.sh'
D=/data/c20261020
IN=$D/in
M=t_m20
C=t_c20
P=$D/prof_a
log() { echo "$* $(date +%T)" >> $D/queue_a.txt; }

log "queue start"
mkdir -p $P $D/units
cd /workspace/sglang-omni
git fetch -q origin prof/cosyvoice3-candidate
[ -d /workspace/$C ] || git worktree add -q --detach /workspace/$C FETCH_HEAD
for tree in $M $C; do
  log "tree $tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$tree rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

(
  cd /workspace/$C
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$C taskset -c 0-15,64-79 python3 -m pytest -q -p no:cacheprovider \
    tests/unit_test/fun_cosyvoice3 tests/unit_test/scheduling/test_streaming_vocoder.py \
    tests/unit_test/model_runner/test_sglang_execution.py tests/unit_test/qwen3_omni/test_talker_token_readback.py \
    tests/unit_test/qwen3_tts/test_decode_step_staging.py > $D/units/units_$C.log 2>&1
  rc=$?
  echo "units $C rc $rc $(grep -E 'passed|failed' $D/units/units_$C.log | tail -n 1) $(date +%T)" >> $D/queue_a.txt
)

bash $IN/warm_c20261020.sh $M 2 0-15,64-79 26300 $D/warm_$M &
bash $IN/warm_c20261020.sh $C 3 16-31,80-95 26500 $D/warm_$C &
wait
for tree in $M $C; do
  log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
done

PEER=$P/$C PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261020.sh $M 2 0-15,64-79 27600 \
  $P/$M > $P/run_$M.log 2>&1 &
PEER=$P/$M PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261020.sh $C 3 16-31,80-95 27800 \
  $P/$C > $P/run_$C.log 2>&1 &
wait
log "profiles $(tail -n 1 $P/$M/progress.txt) / $(tail -n 1 $P/$C/progress.txt)"
for cell in c16 c64; do
  python3 $IN/request_states_c20261019.py main=$P/$M/$cell.sqlite cand=$P/$C/$cell.sqlite > $P/states_$cell.txt 2>&1
done
log "analysis done"

(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261020.py /workspace/$M \
  /workspace/$C $D/p_c20 > $D/p_c20.log 2>&1)
log "pair p_c20 rc $?"
log "queue done"
