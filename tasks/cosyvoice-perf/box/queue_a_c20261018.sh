#!/usr/bin/env bash
# 10-18 lease, cards 2 and 3: PR B (t_hg3, perf/cosyvoice3-hift-decode-graphs 45f694edf, the five graph commits
# rebased onto upstream main 45588d502 with #2907; range-diff identical to the measured tree but for a context line).
# Card 3 runs PR B's unit suites, then warms it (its smoke test); card 2 warms main (t_main3) for later pairs. No pair:
# main's commits after #2907 do not reach CosyVoice's serving path, so `p_gp` (10-17) stands for PR B.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_a.pgid; exec bash queue_a_c20261018.sh'
D=/data/c20261018
IN=$D/in
T=t_hg3
log() { echo "$* $(date +%T)" >> $D/queue_a.txt; }

log "queue start"
bash $IN/warm_c20261018.sh t_main3 2 0-15,64-79 21000 $D/warm_main3 &
WARM=$!
mkdir -p $D/units/$T
(cd /workspace/$T && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py \
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3 > $D/units/$T/units.log 2>&1)
log "units $T rc $? $(grep -E 'passed|failed' $D/units/$T/units.log | tail -n 1) skipped $(grep -cE 'SKIP|skipped' $D/units/$T/units.log)"
bash $IN/warm_c20261018.sh $T 3 16-31,80-95 22000 $D/smoke_hg3
log "smoke hg3 $(grep 'bench rc' $D/smoke_hg3/progress.txt) tracebacks $(grep -c Traceback $D/smoke_hg3/serve.log) graphs $(grep -o 'hift decode graphs captured.*' $D/smoke_hg3/serve.log | head -1)"
wait $WARM
log "warm main3 $(grep 'bench rc' $D/warm_main3/progress.txt) tracebacks $(grep -c Traceback $D/warm_main3/serve.log)"
log "queue done"
