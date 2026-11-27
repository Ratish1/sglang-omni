#!/usr/bin/env bash
# 11-27, node-0000, the last cells before the lease ends at 20:53 UTC: the ones #3452 reaches by no op (requests without
# video take main's serial path, the video lane's threads sit idle), read on today's main so the claim is a measurement.
#   1. Speech alone, no video, colocated, c16 and c64, warm and events passes, M F F M.
#   2. MMSU on the fused text pipeline (preprocessing in the thinker's process), all 2,000 CI questions, c96 and c16,
#      M F F M.
# A cell starts only before 20:20 UTC, so the box is idle and the results copied before the lease ends.
set -u
O=/data/v1127
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main FIX=/workspace/fix
CPUS=56-111,168-223
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
in_time() { [ "$(date -u +%H%M)" -lt 2020 ]; }
log "q1127_f start"
speech() {
  local name=$1 tree=$2 port=$3
  in_time || { log "skip $name, past 20:20"; return; }
  note $name $tree "colocated, speech alone c16 and c64"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --loads 16,64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
mmsu() {
  local name=$1 tree=$2 port=$3
  in_time || { log "skip $name, past 20:20"; return; }
  note $name $tree "fused text pipeline, MMSU c96 and c16"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_text_fused --card 0 --port $port --out $O/$name \
    --bench mmsu --loads 96,16 --passes clean > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech sa1_m $MAIN 25000
speech sa2_f $FIX 25200
speech sa3_f $FIX 25400
speech sa4_m $MAIN 25600
for cell in c16 c64; do
  python3 $K/body_table.py $O $cell "the fixes against main, speech alone, colocated, $cell" main=sa1_m,sa4_m \
    fix=sa2_f,sa3_f >> $O/queue.log 2>&1
done
mmsu mm1_m $MAIN 26000
mmsu mm2_f $FIX 26200
mmsu mm3_f $FIX 26400
mmsu mm4_m $MAIN 26600
log "q1127_f done"
