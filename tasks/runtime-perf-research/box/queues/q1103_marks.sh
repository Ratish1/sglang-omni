#!/usr/bin/env bash
# 11-03, after q1103_3451, node-0000, one H200: #3451 under fine marks (suppress write, sampler, predictor, stage tokens,
# emit, token wait, inbox counts per loop pass). The h200 config, c16 streamed, warm, then the trace. NVTX only over 40 s of
# a pass, boots m p p m, so the loop's own timing is as little perturbed as a trace allows; then one boot per arm with CUDA
# for 10 s, for the device time under each mark. Read with talker_loop.py and span_table.py.
set -u
O=/data/rp1103
K=/workspace/kit_m/tasks/runtime-perf-research/box
M=/workspace/marks_main P=/workspace/marks_3p
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
MARKS_ENV="SGLANG_ENABLE_NVTX_SCHEDULER=1 SGLANG_ENABLE_NVTX_OPERATIONS=1"
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
wait_card() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do
    log "waiting, card used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 30
  done
}
boot() {
  local name=$1 tree=$2 trace=$3 seconds=$4
  wait_card
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) trace $trace start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 26400 --out $O/$name \
    --loads 16 --passes warm,nsys --nsys-session $name --nsys-trace $trace --nsys-delay 15 --nsys-seconds $seconds \
    --serve-env "$MARKS_ENV" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1103_3451 done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1103_marks start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot tl_m1 $M nvtx 40
boot tl_p1 $P nvtx 40
boot tl_p2 $P nvtx 40
boot tl_m2 $M nvtx 40
cd $O
taskset -c 0-55 python3 $K/talker_loop.py tl_m1,tl_m2 tl_p1,tl_p2 > $O/read_tl.txt 2>&1
log "read tl"
boot tc_m1 $M cuda,nvtx,osrt 10
boot tc_p1 $P cuda,nvtx,osrt 10
cd $O
for arm in tc_m1 tc_p1; do
  taskset -c 0-55 python3 $K/span_table.py $arm/c16_nsys/trace.sqlite $arm/serve.log --top 60 > $O/read_$arm.txt 2>&1
done
log "q1103_marks done"
