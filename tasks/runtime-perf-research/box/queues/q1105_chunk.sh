#!/usr/bin/env bash
# 11-05, node-0000, after q1105_a2: is c1's 0.64 on Video-MME text a property of the unloaded thinker or one path among many? At c1
# there is no load and no batch, so only the prefill chunk size moves the numerics. Accuracy that scatters around the loaded 0.60
# means c1 was one draw; accuracy that stays 0.64 means load itself lowers it. Each size on main, c1, two clean and events.
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1105_a2 done" $O/queue.log 2>/dev/null; do sleep 15; done
boot() {
  local name=$1 card=$2 size=$3
  log "boot $name chunked_prefill_size $size card $card start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $MAIN --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --bench videomme_text --loads 1 --passes clean,events --bench-repeats 2 \
    --serve-extra "$CONTEXT_32K --thinker.engine.chunked_prefill_size $size" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1105_chunk start"
boot pc4096 0 4096 & boot pc6000 1 6000 & wait
boot pc7000 0 7000 & boot pc10000 1 10000 & wait
boot pc12000 0 12000 & boot pc16384 1 16384 & wait
log "q1105_chunk done"
