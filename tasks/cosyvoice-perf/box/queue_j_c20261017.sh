#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: the stack with the ranking's yield bound at two recent steps (t_pr2_720636a8b: item 1 +
# the ranking 8672556e1 + the AR loop on the top priority stream). Waits for queue_i, then card 3 runs the unit suites
# (the scheduler and stage fixtures and the accelerator stream test included) while card 2 warms the tree (its smoke
# test). Then one pair, c16 and c64 only, both orders: upstream main (t_main2) against the stack. c1 has no first hop
# competing with a final, so the bound cannot change it and p_pr's c1 rows stand; no scoring, the decode is item 1's.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_j.pgid; exec bash queue_j_c20261017.sh'
D=/data/c20261017
IN=$D/in
T=t_pr2_720636a8b
log() { echo "$* $(date +%T)" >> $D/queue_j.txt; }

log "queue wait"
until grep -q "queue done" $D/queue_i.txt; do sleep 30; done
log "queue start"
bash $IN/warm_c20261017.sh $T 2 0-15,64-79 21100 $D/smoke_pr2 &
WARM=$!
mkdir -p $D/units/$T
(cd /workspace/$T && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py \
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3 > $D/units/$T/units.log 2>&1)
log "units $T rc $? $(grep -E 'passed|failed' $D/units/$T/units.log | tail -n 1)"
wait $WARM
log "smoke pr2 $(grep 'bench rc' $D/smoke_pr2/progress.txt) tracebacks $(grep -c Traceback $D/smoke_pr2/serve.log)"
cd /workspace/t_main2
if grep -q "bench rc 0" $D/smoke_pr2/progress.txt && ! grep -q Traceback $D/smoke_pr2/serve.log; then
  PAIR_CELLS=c16,c64 python3 $IN/serve_pair_c20261017.py /workspace/t_main2 /workspace/$T $D/p_pr2 > $D/p_pr2.log 2>&1
  log "p_pr2 rc $?"
else
  log "pair skipped"
fi
log "queue done"
