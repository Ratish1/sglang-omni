#!/usr/bin/env bash
# 11-05, node-0000, after q1105_omp: who moves Video-MME text accuracy at temperature 0 with identical inputs. Main's stage keeps
# the thinker near empty, so its video prefills run alone; the final form mixes them with decodes. Main and final at c16, final
# without mixed chunks, both at c1 to c8 for the same thinker load, and a float32 thinker as the numerics reference.
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 F=/workspace/wk_final
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
NO_MIX="--thinker.engine.enable_mixed_chunk false"
FP32="--thinker.engine.dtype float32"
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1105_omp done" $O/queue.log 2>/dev/null; do sleep 15; done
video() {
  local name=$1 tree=$2 card=$3 loads=$4 extra=${5:-}
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) loads $loads card $card [$extra] start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --bench videomme_text --loads $loads --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K $extra" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1105_acc start"
video vm1 $MAIN 0 16 & video vf1 $F 1 16 & wait
video vn1 $F 0 16 "$NO_MIX" & video vm2 $MAIN 1 16 & wait
video vf2 $F 0 16 & video vn2 $F 1 16 "$NO_MIX" & wait
video vfs $F 0 1,2,4,8 & video vms $MAIN 1 1,2,4,8 & wait
video vd1 $F 0 1,16 "$FP32" & video vd2 $F 1 16,1 "$FP32" & wait
log "q1105_acc done"
