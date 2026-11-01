#!/usr/bin/env bash
# 11-01, after q1101_cprof: which parts of the stack remove the talker cap's playback stalls. Each against main, same-time
# pairs both card orders, colocated, pinned, c64 warm + two clean passes + an events pass, every boot with the hop probe's
# talker batches (both arms alike). Arms: the cap with 3p (test/cap-3p-20261101 1588e4f0b, the suppress write's sync
# removed), the cap with 3p and the request build (test/cap-3p-rb-20261101 43e1c46a7, both waits the trace names).
set -u
O=/data/rp1101
K=/workspace/kit_bis/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 C3P=/workspace/c3p_1588e4f0b C3PRB=/workspace/c3prb_43e1c46a7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1101_cprof done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1101_bis start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
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
boot bi_m1 $MAIN 0 24000 & boot bi_p1 $C3P 1 24400 & wait
read_pair bi_m1 bi_p1
boot bi_p2 $C3P 0 24000 & boot bi_m2 $MAIN 1 24400 & wait
read_pair bi_m2 bi_p2
boot bi_m3 $MAIN 0 24000 & boot bi_r1 $C3PRB 1 24400 & wait
read_pair bi_m3 bi_r1
boot bi_r2 $C3PRB 0 24000 & boot bi_m4 $MAIN 1 24400 & wait
read_pair bi_m4 bi_r2
log "q1101_bis done"
