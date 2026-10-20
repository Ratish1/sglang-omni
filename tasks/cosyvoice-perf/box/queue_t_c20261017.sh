#!/usr/bin/env bash
# 10-17 lease, card 3: #2907's head after review (t_pa_a7345f6e7: the loop's stream through the scheduler's device
# module, the option renamed should_run_top_priority_stream, CosyVoice's opt in on its vocoder's own stream condition).
# On CUDA the calls are the measured ones (torch.cuda through the device module, device type cuda where is_cuda held),
# so the units and one served smoke test are the check, no pair.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_t.pgid; exec bash queue_t_c20261017.sh'
D=/data/c20261017
IN=$D/in
T=t_pa_a7345f6e7
log() { echo "$* $(date +%T)" >> $D/queue_t.txt; }

log "queue start"
mkdir -p $D/units/$T
(cd /workspace/$T && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py \
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3 > $D/units/$T/units.log 2>&1)
log "units $T rc $? $(grep -E 'passed|failed' $D/units/$T/units.log | tail -n 1)"
bash $IN/warm_c20261017.sh $T 3 16-31,80-95 22500 $D/smoke_pa
log "smoke pa $(grep 'bench rc' $D/smoke_pa/progress.txt) tracebacks $(grep -c Traceback $D/smoke_pa/serve.log)"
log "queue done"
