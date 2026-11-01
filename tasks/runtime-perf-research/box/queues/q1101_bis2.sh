#!/usr/bin/env bash
# 11-01, after q1101_bis: the stack without mixed admissions (test/cap-3p-rb-3s-20261101 315a2f414: the cap, 3p, the
# request build and 3s; the cap stack 3c6ceb520 minus the mixed admissions commit) against main, same-time pairs both
# card orders, colocated, pinned, c64 warm + two clean passes + an events pass, every boot with the talker batch probe.
set -u
O=/data/rp1101
K=/workspace/kit_bis2/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 NOMIX=/workspace/c3prb3s_315a2f414
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1101_bis done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1101_bis2 start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot() {
  local name=$1 tree=$2 card=$3 port=$4
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name --bench-repeats 2 \
    --loads 64 --passes warm,clean,events --probe --serve-env "OMNI_HOP_PROBE_TALKER_BATCHES=1" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
R="taskset -c 0-55 python3"
read_pair() {
  local main=$1 arm=$2
  {
    echo "== $main vs $arm"
    for a in $main $arm; do $R $K/underrun.py "$a clean" "$O/$a/c64_clean/bench/rep*/speed_results.json"; done
    for a in $main $arm; do $R $K/underrun.py "$a events" "$O/$a/c64_events/bench/speed_results.json"; done
    $R $K/admission_trains.py --cell c64_events --arm main=$O/$main --arm branch=$O/$arm
    $R $K/talker_cadence.py --cell c64_events $O/$main $O/$arm
  } >> $O/bis_read.txt 2>&1
}
boot bi_m5 $MAIN 0 24000 & boot bi_x1 $NOMIX 1 24400 & wait
read_pair bi_m5 bi_x1
boot bi_x2 $NOMIX 0 24000 & boot bi_m6 $MAIN 1 24400 & wait
read_pair bi_m6 bi_x2
log "q1101_bis2 done"
