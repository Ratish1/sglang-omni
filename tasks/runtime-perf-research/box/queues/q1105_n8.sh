#!/usr/bin/env bash
# 11-05, node-0000: #3452's candidate with 2 processor calls against 8, the videos-in-flight bound. The stage alone flattens by 3 and
# passive threads cost nothing idle, so if 8 harms neither speech nor video the stage needs one bound, not two. Same time, own sockets.
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
C=/workspace/wk_fix
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
N8="--preprocessing.factory.max_concurrency 8"
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
speech() {
  local name=$1 card=$2 extra=${3:-}
  log "boot $name seedtts card $card [$extra] start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $C --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --loads 64,16 --passes warm,events --serve-extra "$extra" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video() {
  local name=$1 card=$2 extra=${3:-}
  log "boot $name videomme_text card $card [$extra] start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $C --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --bench videomme_text --loads 16 --passes clean,events --serve-extra "$CONTEXT_32K $extra" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1105_n8 start"
speech ns_2a 0 & speech ns_8a 1 "$N8" & wait
speech ns_8b 0 "$N8" & speech ns_2b 1 & wait
video nt_2a 0 & video nt_8a 1 "$N8" & wait
video nt_8b 0 "$N8" & video nt_2b 1 & wait
log "q1105_n8 done"
