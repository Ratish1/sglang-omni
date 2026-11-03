#!/usr/bin/env bash
# 11-03, after q1103_wholdslot_b, node-0000: #3452's final form against main debd337b4. A request without a video holds a
# running slot from its start; a video request takes one only for its processor call; six request threads, two slots, the
# stage's defaults with no flag. Suites and three mutants first, while nothing serves. Then boots m f f m on SeedTTS (c64 and
# c16, warm and events), Video-MME text-only and with audio (c16, 32k, clean then events).
set -u
O=/data/rp1103
K=/workspace/kit_f/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 F=/workspace/wk_final
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
PRE=sglang_omni/models/qwen3_omni/components/preprocessor.py
TESTS="tests/unit_test/qwen3_omni/test_pipeline.py"
mkdir -p $O/units_final
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
wait_card() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do
    log "waiting, card used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 30
  done
}
mutant() {
  local name=$1 expression=$2 selector=$3
  rm -rf /tmp/wk_mutant_$name && git -C $F worktree add -q -f --detach /tmp/wk_mutant_$name HEAD
  sed -i "$expression" /tmp/wk_mutant_$name/$PRE
  git -C /tmp/wk_mutant_$name diff --stat > $O/units_final/mutant_$name.log 2>&1
  cd /tmp/wk_mutant_$name
  taskset -c 0-55 python3 -m pytest -q $TESTS -k "$selector" -p no:cacheprovider >> $O/units_final/mutant_$name.log 2>&1
  log "mutant $name rc $? (1 expected)"
}
speech() {
  local name=$1 tree=$2
  wait_card
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) seedtts start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name --loads 64,16 \
    --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video() {
  local name=$1 tree=$2 bench=$3
  wait_card
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) $bench start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name --bench $bench --loads 16 \
    --passes clean,events --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1103_wholdslot_b done" $O/queue.log 2>/dev/null; do sleep 20; done
wait_card
log "q1103_wfinal start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
cd $F
taskset -c 0-55 python3 -m pytest -q tests/unit_test/qwen3_omni tests/unit_test/scheduling/test_threaded_simple_scheduler.py \
  tests/unit_test/pipeline/test_stage.py tests/unit_test/pipeline/test_scheduler.py -p no:cacheprovider > $O/units_final/units.log 2>&1
log "suites final rc $?"
mutant decode_holds_slot 's/if request_videos is None:/if True:/' "video_decode_leaves_its_slot"
mutant video_unbounded 's/^            self.take_request_slot()$/            pass/' "runs_no_more_requests_than_its_slots"
mutant slot_kept 's/self.request_slots.release()/pass/' "failed_request_returns_its_slot"
speech fs_m1 $MAIN
speech fs_f1 $F
speech fs_f2 $F
speech fs_m2 $MAIN
video ft_m1 $MAIN videomme_text
video ft_f1 $F videomme_text
video ft_f2 $F videomme_text
video ft_m2 $MAIN videomme_text
video fv_m1 $MAIN videomme
video fv_f1 $F videomme
video fv_f2 $F videomme
video fv_m2 $MAIN videomme
log "q1103_wfinal done"
