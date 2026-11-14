#!/usr/bin/env bash
# 11-14, node-0000, one H200 (card 0, NUMA 1). #3452 correctness on text Video-MME, the CI sample at the CI's video settings:
# every boot hashes each request's preprocessing tensors (preproc_hash_probe), runs c1 twice, and after its passes sends the
# looping questions alone and with other videos (cobatch_probe). The lane also runs c16 twice, so its video threads' tensors
# are compared with the serial c1 ones. Boots main, lane, lane, main for the cross-boot A/A.
set -u
O=/data/rp1114
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main WAIT=/workspace/lane_wait
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
card_free() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  until [ "${used:-0}" -lt 1000 ]; do
    log "card busy ${used} MiB, waiting"
    sleep 30
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  done
}
boot() {
  local name=$1 tree=$2 loads=$3 repeats=$4
  card_free
  mkdir -p $O/$name/hash
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) loads $loads start"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --bench videomme_text --loads $loads --passes clean --bench-repeats $repeats --serve-extra "$CONTEXT_32K" \
    --serve-env "PYTHONPATH=$K/preproc_hash_probe:$tree OMNI_PREPROC_HASH=$O/$name/hash/h" \
    --after "taskset -c $CARD_CPUS python3 $K/cobatch_probe.py --port 28400 --out $O/$name/cobatch.json" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1114_identity start"
boot id_m1 $MAIN 1 2
boot id_w1 $WAIT 1,16 2
boot id_w2 $WAIT 1 1
boot id_m2 $MAIN 1 1
log "q1114_identity done"
