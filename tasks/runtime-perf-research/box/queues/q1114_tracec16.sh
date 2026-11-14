#!/usr/bin/env bash
# 11-14, node-0000, after q1114_rowfinal: SeedTTS c16 alone under nsys (gpuctxsw, NVTX, graph replays) on main (tc_m)
# and the folded #3451 head with the colocated one-chunk start (tc_r), one 15 s window after 20 s of load, so the card
# arbitration design reads switches a second, each owner's card wait and the talker step under thinker activity.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
NV=/workspace/marks/tasks/qwen3_omni_perf_20260923/scripts/omni_pipeline_nvtx
C=/workspace/marks/tasks/qwen3_omni_perf_20260923/scripts
MAIN=/workspace/main ROW=/workspace/rowready2
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1114_rowfinal done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1114_tracec16 start"
traced() {
  local name=$1 tree=$2
  note $name $tree "seedtts c16 nsys"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --passes warm,nsys --nsys-session $name --nsys-trace cuda,nvtx,osrt --nsys-graph-trace graph --nsys-gpuctxsw \
    --nsys-delay 20 --nsys-seconds 15 --serve-env "OMNI_PIPE_NVTX=1 PYTHONPATH=$NV:$tree" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
  python3 -c "import json; w=json.load(open('$O/$name/c16_nsys/nsys_window.json')); open('$O/$name/c16_nsys/window.txt','w').write(f\"{w['start']} {w['start'] + 15}\n\")"
}
traced tc_m $MAIN
traced tc_r $ROW
for t in tc_m tc_r; do
  taskset -c 0-55 python3 $C/omni_census.py $O/$t/c16_nsys/trace.sqlite --window $O/$t/c16_nsys/window.txt --sections ADI \
    --sample-rate 24000 > $O/$t/census_ADI.txt 2>&1
  log "census $t rc $?"
done
log "q1114_tracec16 done"
