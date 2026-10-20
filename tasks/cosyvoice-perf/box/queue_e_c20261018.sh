#!/usr/bin/env bash
# 10-18, cards 2 and 3: PR B with the kernel review and the inverse STFT summed one bin at a time (t_kp2,
# test/hift-kernel-principles 3a3b447dd). Card 2: the kernels' registers, the inverse against #2910's kernel at the
# decode's shapes (`hift_inverse_restructure_c20261018.py`), each tree's decode graph per tier in the order t_hg3,
# t_kp2, t_hg3 with the outputs compared, then the eager call alternated four times per tree. Card 3: the unit
# suites, then the smoke test.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_e.pgid; exec bash queue_e_c20261018.sh'
D=/data/c20261018
IN=$D/in
T=t_kp2
log() { echo "$* $(date +%T)" >> $D/queue_e.txt; }

log "queue start"
mkdir -p $D/units/$T $D/tree2 $D/eager2
(
  cd /workspace/$T
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$T python3 $IN/hift_kernel_registers_c20261018.py \
    > $D/tree2/registers.txt 2>&1
  log "registers rc $?"
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$T taskset -c 0-15,64-79 \
    python3 $IN/hift_inverse_restructure_c20261018.py $D/tree2/inverse > $D/tree2/inverse.log 2>&1
  log "inverse rc $?"
  for run in t_hg3.1 $T.1 t_hg3.2; do
    tree=${run%.*}
    (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$tree taskset -c 0-15,64-79 \
      python3 $IN/hift_decode_tree_c20261018.py $D/tree2/$run.pt > $D/tree2/$run.log 2>&1)
    log "tree $run rc $?"
  done
  python3 $IN/hift_decode_tree_c20261018.py --compare $D/tree2/t_hg3.1.pt $D/tree2/$T.1.pt > $D/tree2/compare_kp2.txt 2>&1
  python3 $IN/hift_decode_tree_c20261018.py --compare $D/tree2/t_hg3.1.pt $D/tree2/t_hg3.2.pt > $D/tree2/compare_aa.txt 2>&1
  log "tree compare done"
) &
CARD2=$!
(cd /workspace/$T && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py \
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3 > $D/units/$T/units.log 2>&1)
log "units $T rc $? $(grep -E 'passed|failed' $D/units/$T/units.log | tail -n 1) skipped $(grep -cE 'SKIP|skipped' $D/units/$T/units.log)"
bash $IN/warm_c20261018.sh $T 3 16-31,80-95 22400 $D/smoke_kp2
log "smoke kp2 $(grep 'bench rc' $D/smoke_kp2/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/smoke_kp2/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/smoke_kp2/serve.log) graphs $(grep -o 'hift decode graphs captured.*' $D/smoke_kp2/serve.log | head -1)"
wait $CARD2
for i in 1 2 3 4; do
  for tree in t_hg3 $T; do
    (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$tree taskset -c 0-15,64-79 \
      python3 $IN/hift_eager_tree_c20261018.py 2> /dev/null | tail -n 1 >> $D/eager2/eager.txt)
  done
done
log "eager done"
log "queue done"
