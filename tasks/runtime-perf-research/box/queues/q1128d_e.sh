#!/usr/bin/env bash
# 11-28 second lease, node-0001, after q1128d_d: does the PR run out of GPU memory where main does not on the
# cookbook's colocated H20 config. Each card holds a ballast down to an H20's free memory (97871 MiB) and serves the
# H20 config's bytes (qwen3_omni_colocated_h20_on_h200.yaml), FIX 91c2e2010 against MAIN 0f70d9c7c, same-time pairs
# swapped: CI Video-MME text c16 x2 from a fresh boot (16 videos at once, the largest image encoder batches) and the
# talker test x1, every process's GPU memory sampled every 100 ms.
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main F=/workspace/fix
H20_FREE_MIB=97871
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((20000 + 4000 * $1 + $2)); }
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
gpu_sampler() {
  local name=$1 card=$2 driver=$3
  while kill -0 $driver 2>/dev/null; do
    nvidia-smi -i $card --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | sed "s/^/$(date +%s.%N | cut -c1-14),/" >> $O/$name/gpu_samples.csv
    sleep 0.1
  done
}
until grep -q "q1128d_d done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128d_e start"

h20() {
  local name=$1 tree=$2 card=$3 offset=$4
  local port=$(port_of $card $offset)
  mkdir -p $O/$name
  python3 $K/ballast.py --card $card --free-mib $H20_FREE_MIB --ready-file $O/$name/ballast.txt > $O/$name/ballast.log 2>&1 &
  local ballast=$!
  until [ -s $O/$name/ballast.txt ] || ! kill -0 $ballast 2>/dev/null; do sleep 2; done
  note $name $tree "H20 colocated bytes on card $card, $(cat $O/$name/ballast.txt 2>/dev/null), text c16 x2, talker x1"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated_h20_on_h200 --card $card \
    --port $port --out $O/$name --bench videomme_text --loads 16 --passes clean --bench-repeats 2 \
    --after "python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 1" > $O/$name.driver.log 2>&1 &
  local driver=$!
  gpu_sampler $name $card $driver &
  local sampler=$!
  wait $driver
  local rc=$?
  wait $sampler
  kill $ballast 2>/dev/null
  wait $ballast 2>/dev/null
  log "boot $name rc $rc oom $(grep -c -i -E 'out of memory|OutOfMemoryError' $O/$name/serve.log) failed $(grep -h -o -E '"failed": [0-9]+' $O/$name/c16_clean/bench/rep*/videomme_results.json $O/$name/talker/summary.jsonl 2>/dev/null | tr '\n' ' ')"
}
both "h20 h1_m $M 0 4100" "h20 h1_f $F 1 4100"
both "h20 h2_f $F 0 4300" "h20 h2_m $M 1 4300"
python3 $K/videomme_answer_diff.py $O main=h1_m/c16_clean,h2_m/c16_clean fix=h1_f/c16_clean,h2_f/c16_clean > $O/h20_text.txt 2>&1
log "h20 read"
log "q1128d_e done"
