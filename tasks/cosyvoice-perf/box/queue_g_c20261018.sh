#!/usr/bin/env bash
# 10-18, card 3: #2910's review fixes (t_kp4, test/hift-kernel-principles f7e8880ac: the NVIDIA gate once inside the cuda branch, no implied Triton check, for HiFT's
# kernels and decode graphs, the empty frame tier guard, comments and docstrings). The unit suites, then the smoke
# test; the kernels are unchanged from t_kp2, so its decode and eager comparisons stand.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_g.pgid; exec bash queue_g_c20261018.sh'
D=/data/c20261018
IN=$D/in
T=t_kp4
log() { echo "$* $(date +%T)" >> $D/queue_g.txt; }

log "queue start"
mkdir -p $D/units/$T
(cd /workspace/$T && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py \
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3 > $D/units/$T/units.log 2>&1)
log "units $T rc $? $(grep -E 'passed|failed' $D/units/$T/units.log | tail -n 1) skipped $(grep -cE 'SKIP|skipped' $D/units/$T/units.log)"
bash $IN/warm_c20261018.sh $T 3 16-31,80-95 22800 $D/smoke_kp4
log "smoke kp4 $(grep 'bench rc' $D/smoke_kp4/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/smoke_kp4/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/smoke_kp4/serve.log) graphs $(grep -o 'hift decode graphs captured.*' $D/smoke_kp4/serve.log | head -1)"
log "queue done"
