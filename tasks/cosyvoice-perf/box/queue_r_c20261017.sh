#!/usr/bin/env bash
# 10-17 lease, card 3: the unit suites of the graphs PR's tree (t_gp_56b436580, the same source tree as the stacked
# perf/cosyvoice3-hift-decode-graphs head 34b753d96 on the priority branch). Waits for queue_q.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_r.pgid; exec bash queue_r_c20261017.sh'
D=/data/c20261017
G=t_gp_56b436580
log() { echo "$* $(date +%T)" >> $D/queue_r.txt; }

log "queue wait"
until grep -q "queue done" $D/queue_q.txt; do sleep 30; done
log "queue start"
mkdir -p $D/units/$G
(cd /workspace/$G && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py \
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3 > $D/units/$G/units.log 2>&1)
log "units $G rc $? $(grep -E 'passed|failed' $D/units/$G/units.log | tail -n 1)"
log "queue done"
