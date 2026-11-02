#!/usr/bin/env bash
# 11-02, node-0001 cards 0 and 1, every head on upstream main debd337b4. 3p (1f04aa0e6) against main, colocated, pinned,
# same-time pairs both card orders: c16 warm + events, read by request state with first_audio_states.py (its +0.76 % lean
# was never read by state), then c16 non-streamed warm + two clean passes (never run for 3p). Then the suites of 3p and
# workers (388254854) on these heads and each branch's new tests against main's sources.
set -u
O=/data/rp1102
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 P3=/workspace/p3_1f04aa0e6 WK=/workspace/wk_388254854
mkdir -p $O/units
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
log "q1102_3p start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot() {
  local name=$1 tree=$2 card=$3 port=$4; shift 4
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 16 "$@" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
boot pv_m1 $MAIN 0 25000 --passes warm,events & boot pv_p1 $P3 1 25400 --passes warm,events & wait
boot pv_p2 $P3 0 25000 --passes warm,events & boot pv_m2 $MAIN 1 25400 --passes warm,events & wait
cd $O
{
  echo "both orders"; taskset -c 0-55 python3 $K/first_audio_states.py --cell c16_events --main pv_m1,pv_m2 --branch pv_p1,pv_p2
  echo "main card 0, 3p card 1"; taskset -c 0-55 python3 $K/first_audio_states.py --cell c16_events --main pv_m1 --branch pv_p1
  echo "3p card 0, main card 1"; taskset -c 0-55 python3 $K/first_audio_states.py --cell c16_events --main pv_m2 --branch pv_p2
} > $O/pv_read.txt 2>&1
log "states read"
boot ns_m1 $MAIN 0 25000 --passes warm,clean --bench-repeats 2 --no-stream & boot ns_p1 $P3 1 25400 --passes warm,clean --bench-repeats 2 --no-stream & wait
boot ns_p2 $P3 0 25000 --passes warm,clean --bench-repeats 2 --no-stream & boot ns_m2 $MAIN 1 25400 --passes warm,clean --bench-repeats 2 --no-stream & wait
log "non-streamed done"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $P3 && taskset -c 0-55 python3 -m pytest -q tests/unit_test/qwen3_omni tests/unit_test/model_runner -p no:cacheprovider > $O/units/units_3p.log 2>&1
log "suites 3p rc $?"
cd $WK && taskset -c 0-55 python3 -m pytest -q tests/unit_test/qwen3_omni -p no:cacheprovider > $O/units/units_wk.log 2>&1
log "suites workers rc $?"
mkdir -p /tmp/on_main_3p /tmp/on_main_wk
cp $P3/tests/unit_test/qwen3_omni/test_logit_shaping.py /tmp/on_main_3p/
cp $WK/tests/unit_test/qwen3_omni/test_pipeline.py /tmp/on_main_wk/
cd $MAIN
taskset -c 0-55 python3 -m pytest -q /tmp/on_main_3p/test_logit_shaping.py -k do_not_synchronize -p no:cacheprovider --rootdir=$MAIN > $O/units/units_3p_on_main.log 2>&1
log "3p test on main rc $?"
taskset -c 0-55 python3 -m pytest -q /tmp/on_main_wk/test_pipeline.py -k "each_run_their_own_preprocessor or dispatch_follows_its_worker_count" -p no:cacheprovider --rootdir=$MAIN > $O/units/units_wk_on_main.log 2>&1
log "workers tests on main rc $?"
log "q1102_3p done"
