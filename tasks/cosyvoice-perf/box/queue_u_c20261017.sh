#!/usr/bin/env bash
# 10-17 lease, card 3: #2907 after its review (t_pa3_3bb30b414: the loop's stream inlined into start() through the
# scheduler's device module, the stale vocoder note, the start() level test). The mutation tree (t_mut_af6016483: start()
# enters a null context in place of the loop's stream) runs the new test only, which must fail when the option is set.
# Then the head's unit suites and one served smoke test.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_u.pgid; exec bash queue_u_c20261017.sh'
D=/data/c20261017
IN=$D/in
T=t_pa3_3bb30b414
M=t_mut_af6016483
TEST=tests/unit_test/pipeline/test_scheduler.py::test_start_runs_the_loop_on_the_top_priority_stream_only_when_asked
log() { echo "$* $(date +%T)" >> $D/queue_u.txt; }

log "queue start"
mkdir -p $D/units/$T $D/units/$M
(cd /workspace/$M && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider $TEST \
  > $D/units/$M/units.log 2>&1)
log "mutation rc $? $(grep -E 'passed|failed' $D/units/$M/units.log | tail -n 1)"
(cd /workspace/$T && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py \
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3 > $D/units/$T/units.log 2>&1)
log "units $T rc $? $(grep -E 'passed|failed' $D/units/$T/units.log | tail -n 1) skipped $(grep -cE 'SKIP|skipped' $D/units/$T/units.log)"
bash $IN/warm_c20261017.sh $T 3 16-31,80-95 22600 $D/smoke_pa3
log "smoke pa3 $(grep 'bench rc' $D/smoke_pa3/progress.txt) tracebacks $(grep -c Traceback $D/smoke_pa3/serve.log)"
log "queue done"
