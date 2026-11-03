#!/usr/bin/env bash
# 11-04, node-0000 on two cards: #3452's final form against main debd337b4. One event loop for every request's logic; two HF
# processor calls at once on private processors; a request without video holds a running slot from its start, a video takes
# one only for its processor call; up to the media pool's eight videos in flight. The stage's defaults, no flag. Suites and
# five mutants while nothing serves, then same-time pairs, both card orders, on SeedTTS, Video-MME text-only and with audio.
set -u
O=/data/rp1104
K=/workspace/kit_f/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 F=/workspace/wk_final
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
PRE=sglang_omni/models/qwen3_omni/components/preprocessor.py
TESTS=tests/unit_test/qwen3_omni/test_pipeline.py
mkdir -p $O/units
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
mutant() {
  local name=$1 expression=$2 selector=$3
  rm -rf /tmp/wk_mutant_$name && git -C $F worktree add -q -f --detach /tmp/wk_mutant_$name HEAD
  sed -i "$expression" /tmp/wk_mutant_$name/$PRE
  git -C /tmp/wk_mutant_$name diff --stat > $O/units/mutant_$name.log 2>&1
  cd /tmp/wk_mutant_$name
  taskset -c 0-55 python3 -m pytest -q $TESTS -k "$selector" -p no:cacheprovider >> $O/units/mutant_$name.log 2>&1
  log "mutant $name rc $? (1 expected)"
}
speech() {
  local name=$1 tree=$2 card=$3 port=$4
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) seedtts card $card start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video() {
  local name=$1 tree=$2 card=$3 port=$4 bench=$5
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) $bench card $card start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name --bench $bench --loads 16 \
    --passes clean,events --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1104_final start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
cd $F
taskset -c 0-55 python3 -m pytest -q tests/unit_test/qwen3_omni tests/unit_test/scheduling/test_threaded_simple_scheduler.py \
  tests/unit_test/preprocessing -p no:cacheprovider > $O/units/units.log 2>&1
log "suites final rc $?"
mutant decode_holds_slot 's/if request_videos is None:/if True:/' "video_decode_leaves_the_running_slot"
mutant video_unbounded 's/^            await self.take_running_slot()$/            pass/' "runs_no_more_requests_than_its_slots"
mutant no_arrival_order 's/^                await self.take_running_slot()$/                pass/' "without_video_run_in_arrival_order"
mutant slot_kept 's/self.request_limits.running_slots.release()/pass/' "failed_request_returns_its_slots"
mutant no_video_bound 's/await self.request_limits.video_slots.acquire()/pass/' "holds_a_video_slot"
log "units done"
speech fs_m1 $MAIN 0 28400 & speech fs_f1 $F 1 28800 & wait
speech fs_f2 $F 0 28400 & speech fs_m2 $MAIN 1 28800 & wait
video ft_m1 $MAIN 0 28400 videomme_text & video ft_f1 $F 1 28800 videomme_text & wait
video ft_f2 $F 0 28400 videomme_text & video ft_m2 $MAIN 1 28800 videomme_text & wait
video fv_m1 $MAIN 0 28400 videomme & video fv_f1 $F 1 28800 videomme & wait
video fv_f2 $F 0 28400 videomme & video fv_m2 $MAIN 1 28800 videomme & wait
log "q1104_final done"
