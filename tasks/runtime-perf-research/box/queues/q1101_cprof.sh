#!/usr/bin/env bash
# 11-01, after q1101_cap: why the talker cap breaks playback continuity at c64 colocated, then the cap on today's stack.
# 1. events pair, main (talker cap 32) card 0 and cap card 1 at the same time, hop probe with the talker's decode batches
#    (admission_trains.py, talker_cadence.py, first_audio_states.py), c64 warm then events.
# 2. one traced boot of marks + cap (prof/q3o-marks-cap c11699a06) on card 0 alone, c64, spans and GPU context switches;
#    if the nsys start times out the boot runs again without context switch collection.
# 3. the cap on the stack (test/cap-stack-20261101 3c6ceb520: 3p, request build, 3s, mixed admissions, cap) against main,
#    same-time pairs both card orders, c64 and c16 warm + two clean passes.
set -u
O=/data/rp1101
K=/workspace/kit_cp2/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 CAP=/workspace/cap_14069e285 MCAP=/workspace/mcap_c11699a06 STACK=/workspace/cstack_3c6ceb520
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
MARKS_ENV="SGLANG_ENABLE_NVTX_SCHEDULER=1 SGLANG_ENABLE_NVTX_OPERATIONS=1"
NSYS="--nsys-trace cuda,nvtx,osrt --nsys-graph-trace graph --nsys-delay 20 --nsys-seconds 10"
until grep -q "q1101_cap done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1101_cprof start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
ident() { echo "$(git -C $1 rev-parse --short HEAD) $(git -C $1 rev-parse --short HEAD:sglang_omni)"; }
events_boot() {
  local name=$1 tree=$2 card=$3 port=$4
  log "boot $name card $card port $port $(ident $tree) start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --passes warm,events --probe --serve-env "OMNI_HOP_PROBE_TALKER_BATCHES=1" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
trace_boot() {
  local name=$1 extra=$2
  log "boot $name card 0 port 23000 $(ident $MCAP) start"
  python3 $K/baseline.py --tree $MCAP --model omni_colocated --card 0 --port 23000 --out $O/$name \
    --loads 64 --passes warm,nsys --nsys-session $name $NSYS $extra --serve-env "$MARKS_ENV" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pair_boot() {
  local name=$1 tree=$2 card=$3 port=$4
  log "boot $name card $card port $port $(ident $tree) start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name --bench-repeats 2 \
    --loads 64,16 --passes warm,clean > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
R="taskset -c 0-55 python3"
events_boot ce_m1 $MAIN 0 24000 & events_boot ce_c1 $CAP 1 24400 & wait
{
  echo "== stalls (underrun), c64 events pass"
  $R $K/underrun.py main "$O/ce_m1/c64_events/bench/rep*/speed_results.json" cap "$O/ce_c1/c64_events/bench/rep*/speed_results.json"
  echo "== admission trains"
  $R $K/admission_trains.py --cell c64_events --arm main=$O/ce_m1 --arm cap=$O/ce_c1
  echo "== talker cadence"
  $R $K/talker_cadence.py --cell c64_events $O/ce_m1 $O/ce_c1
  echo "== first audio states"
  $R $K/first_audio_states.py --cell c64_events --main $O/ce_m1 --branch $O/ce_c1
} > $O/cprof_read_ce.txt 2>&1
log "cprof read ce done"
trace_boot tc_c1 --nsys-gpuctxsw
if grep -rqs "took too long" $O/tc_c1; then
  mv $O/tc_c1 $O/tc_c1_void_nsys_stall
  log "tc_c1 void (nsys start stalled), again without context switches"
  trace_boot tc_c1 ""
fi
{
  for run in t2_m1 tc_c1; do
    echo "== $run c64 talker decode intervals"
    (cd $K && $R talker_intervals.py $O/$run/c64_nsys/trace.sqlite $O/$run/serve.log)
  done
  echo "== tc_c1 c64 spans"
  $R $K/span_table.py $O/tc_c1/c64_nsys/trace.sqlite $O/tc_c1/serve.log --top 40
} > $O/cprof_read_tc.txt 2>&1
log "cprof read tc done"
pair_boot cs_m1 $MAIN 0 24000 & pair_boot cs_s1 $STACK 1 24400 & wait
pair_boot cs_s2 $STACK 0 24000 & pair_boot cs_m2 $MAIN 1 24400 & wait
log "q1101_cprof done"
