#!/usr/bin/env bash
# 11-03, node-0001: the bisect arm #3451 never ran. r is 3p with a stream sync put back before the suppress write
# (6cf742af9, d999eaf0d405), so it keeps 3p's write and main's wait, against main debd337b4, on the setup where 3p reads first
# audio +3.9 ms (b_*): h100_bf16 config on the H200, seed 3451, warmup 16, boots m r r m m r r m, c16 streamed, warm, two
# clean passes, events. r level with main says the moved wait owns the cost; r still up says the write path does.
set -u
O=/data/rp1103w
K=/workspace/kit_t/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 R=/workspace/wait_6cf742af9
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
export CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
wait_card() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do
    log "waiting, card used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 30
  done
}
boot() {
  local name=$1 tree=$2
  wait_card
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model omni_colocated_h100_config --card 0 --port 27000 --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 "--bench-extra=--seed 3451 --warmup 16" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1103_wr start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot wr_m1 $MAIN
boot wr_r1 $R
boot wr_r2 $R
boot wr_m2 $MAIN
boot wr_m3 $MAIN
boot wr_r3 $R
boot wr_r4 $R
boot wr_m4 $MAIN
cd $O
taskset -c 0-55 python3 $K/first_audio_states.py --cell c16_events --main wr_m1,wr_m2,wr_m3,wr_m4 --branch wr_r1,wr_r2,wr_r3,wr_r4 > $O/read_wr.txt 2>&1
log "q1103_wr done"
