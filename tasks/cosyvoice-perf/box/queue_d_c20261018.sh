#!/usr/bin/env bash
# 10-18, cards 2 and 3: PR B with the kernel review (t_kp, test/hift-kernel-principles). Card 3 runs the unit suites,
# then the smoke test; card 2 meanwhile measures each tree's HiFT decode graph per row and frame tier and the eager
# whole utterance call (`hift_decode_tree_c20261018.py`), in the order t_hg3, t_kp, t_hg3, so t_hg3 against itself
# bounds the spread, and compares the outputs on the same seeded inputs.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_d.pgid; exec bash queue_d_c20261018.sh'
D=/data/c20261018
IN=$D/in
T=t_kp
log() { echo "$* $(date +%T)" >> $D/queue_d.txt; }

log "queue start"
mkdir -p $D/units/$T $D/tree
(
  (cd /workspace/$T && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$T python3 $IN/hift_kernel_registers_c20261018.py \
    > $D/tree/registers_kp.txt 2>&1)
  log "registers $T rc $?"
  for run in t_hg3.1 $T.1 t_hg3.2; do
    tree=${run%.*}
    (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$tree taskset -c 0-15,64-79 \
      python3 $IN/hift_decode_tree_c20261018.py $D/tree/$run.pt > $D/tree/$run.log 2>&1)
    log "tree $run rc $?"
  done
  python3 $IN/hift_decode_tree_c20261018.py --compare $D/tree/t_hg3.1.pt $D/tree/$T.1.pt > $D/tree/compare_kp.txt 2>&1
  python3 $IN/hift_decode_tree_c20261018.py --compare $D/tree/t_hg3.1.pt $D/tree/t_hg3.2.pt > $D/tree/compare_aa.txt 2>&1
  log "tree compare done"
) &
TREES=$!
(cd /workspace/$T && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py \
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3 > $D/units/$T/units.log 2>&1)
log "units $T rc $? $(grep -E 'passed|failed' $D/units/$T/units.log | tail -n 1) skipped $(grep -cE 'SKIP|skipped' $D/units/$T/units.log)"
bash $IN/warm_c20261018.sh $T 3 16-31,80-95 22200 $D/smoke_kp
log "smoke kp $(grep 'bench rc' $D/smoke_kp/progress.txt) tracebacks $(grep -c Traceback $D/smoke_kp/serve.log) graphs $(grep -o 'hift decode graphs captured.*' $D/smoke_kp/serve.log | head -1)"
wait $TREES
log "queue done"
