#!/usr/bin/env bash
# 11-05, node-0000: Video-MME text, identical inputs. The deterministic flag, only to place a divergence, never a proposal: does it
# bring c16's answers back to c1's? Then chunk size 7000 again, to read whether its one boot was a boot state or the chunk size.
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
F=/workspace/wk_final MAIN=/workspace/main_debd337b4
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
video() {
  local name=$1 tree=$2 card=$3 loads=$4 extra=$5
  log "boot $name loads $loads card $card [$extra] start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --bench videomme_text --loads $loads --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K $extra" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1105_det start"
DET="--thinker.engine.enable_deterministic_inference true"
video dt1 $F 0 1,16 "$DET" & video dt2 $F 1 16,1 "$DET" & wait
video pd1 $MAIN 0 1 "--thinker.engine.chunked_prefill_size 7000" & video pd2 $MAIN 1 1 "--thinker.engine.chunked_prefill_size 7000" & wait
video pe1 $MAIN 0 1 "--thinker.engine.chunked_prefill_size 8192" & video pe2 $MAIN 1 1 "--thinker.engine.chunked_prefill_size 8192" & wait
log "q1105_det done"
