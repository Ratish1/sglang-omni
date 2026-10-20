#!/usr/bin/env bash
# 10-18, card 3: #2910's review F1 (t_kp5, test/hift-kernel-principles 5d1e2317e: the Snake test with alphas from 0.125
# to 4). The touched test file, then the Snake mutations on t_kp4 (f7e8880ac, every alpha 1) and t_kp5.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_h.pgid; exec bash queue_h_c20261018.sh'
D=/data/c20261018
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_h.txt; }

log "queue start"
mkdir -p $D/units/t_kp5 $D/mutations
(cd /workspace/t_kp5 && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/fun_cosyvoice3/test_vocoder.py > $D/units/t_kp5/vocoder.log 2>&1)
log "units t_kp5 rc $? $(grep -E 'passed|failed' $D/units/t_kp5/vocoder.log | tail -n 1) skipped $(grep -cE 'SKIP|skipped' $D/units/t_kp5/vocoder.log)"
for T in t_kp4 t_kp5; do
  (cd /workspace/$T && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/$T taskset -c 16-31,80-95 \
    python3 $IN/hift_snake_mutations_c20261018.py > $D/mutations/$T.log 2>&1)
  log "mutations $T rc $? $(tr '\n' ' ' < $D/mutations/$T.log | tail -c 400)"
done
log "queue done"
