#!/usr/bin/env bash
# 11-14, node-0000, one H200 (card 0, NUMA 1). Synthetic per-stage regimes (synth_regimes.py) under nsys: main 35890f10c once,
# then the stack of both PR heads 62417f890. Thinker requests decode exactly max_tokens (synth_eos_probe), NVTX ranges chained.
# Each regime is its own window; traces export after each boot, the census runs after both.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
NV=/workspace/marks/tasks/qwen3_omni_perf_20260923/scripts/omni_pipeline_nvtx/sitecustomize.py
MAIN=/workspace/main STACK=/workspace/stack
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
log "q1114_synth start"
boot() {
  local name=$1 tree=$2
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) synthetic regimes start"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 1 --passes none --serve-extra "$CONTEXT_32K" \
    --nsys-session $name --nsys-trace cuda,nvtx,osrt --nsys-graph-trace graph \
    --serve-env "OMNI_SYNTH_IGNORE_EOS=1 OMNI_PIPE_NVTX=1 OMNI_SITECUSTOMIZE_CHAIN=$NV PYTHONPATH=$K/synth_eos_probe:$tree" \
    --after "taskset -c $CARD_CPUS python3 $K/synth_regimes.py --port 28400 --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --out $O/$name/synth --nsys-session $name" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
  for rep in $O/$name/synth/*/trace.nsys-rep; do
    taskset -c 0-55 nsys export --type sqlite --force-overwrite=true --output "${rep%.nsys-rep}.sqlite" "$rep" > /dev/null 2>&1
  done
  log "export $name $(ls $O/$name/synth/*/trace.sqlite 2>/dev/null | wc -l) traces"
}
boot sy_m $MAIN
boot sy_x $STACK
log "q1114_synth done"
