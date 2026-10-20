#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: the PR's head (t_pr3_0167dde34: item 1 less the runner's unreachable None path, the
# ranking at two recent steps, the AR loop on the top priority stream; against 720636a8b only that removal). Card 3
# runs its unit suites, then warms it (the smoke test); then the PR body's kit profile, upstream main (t_main2) on card
# 2 and the head on card 3 under the kit at the same time, c16 and c64 of 640 unseeded requests traced, no py-spy; then
# each cell's per request first audio timeline and card busy span.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_k.pgid; exec bash queue_k_c20261017.sh'
D=/data/c20261017
IN=$D/in
T=t_pr3_0167dde34
log() { echo "$* $(date +%T)" >> $D/queue_k.txt; }

log "queue start"
mkdir -p $D/units/$T
(cd /workspace/$T && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py \
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3 > $D/units/$T/units.log 2>&1)
log "units $T rc $? $(grep -E 'passed|failed' $D/units/$T/units.log | tail -n 1)"
bash $IN/warm_c20261017.sh $T 3 16-31,80-95 22100 $D/smoke_pr3
log "smoke pr3 $(grep 'bench rc' $D/smoke_pr3/progress.txt) tracebacks $(grep -c Traceback $D/smoke_pr3/serve.log)"
if grep -q "bench rc 0" $D/smoke_pr3/progress.txt && ! grep -q Traceback $D/smoke_pr3/serve.log; then
  mkdir -p $D/prof_pr
  PEER=$D/prof_pr/stack PYSPY=0 CELLS="c16:16:640 c64:64:640" bash $IN/cosy_profile_run_c20261017.sh t_main2 2 \
    0-15,64-79 21000 $D/prof_pr/main > $D/prof_pr/run_main.log 2>&1 &
  first=$!
  PEER=$D/prof_pr/main PYSPY=0 CELLS="c16:16:640 c64:64:640" bash $IN/cosy_profile_run_c20261017.sh $T 3 \
    16-31,80-95 22000 $D/prof_pr/stack > $D/prof_pr/run_stack.log 2>&1 &
  second=$!
  wait $first $second
  log "profiles $(tail -n 1 $D/prof_pr/main/progress.txt) / $(tail -n 1 $D/prof_pr/stack/progress.txt)"
  for cell in c16 c64; do
    python3 $IN/first_audio_timeline_v2_c20261017.py main=$D/prof_pr/main/$cell.sqlite \
      stack=$D/prof_pr/stack/$cell.sqlite > $D/prof_pr/timeline_$cell.txt 2>&1
    python3 $IN/card_busy_span_c20261017.py main=$D/prof_pr/main/$cell.sqlite stack=$D/prof_pr/stack/$cell.sqlite \
      > $D/prof_pr/card_busy_$cell.txt 2>&1
    log "analysis $cell done"
  done
else
  log "profile skipped"
fi
log "queue done"
