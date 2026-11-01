#!/usr/bin/env bash
# Block 56 of 11-01: startup item E reopened, a GC census of the boot. Default 1.7B Base on card 0, one boot at a time:
# main cold into a fresh cache (fills it), then main warm twice with gc_census/sitecustomize.py on the path, which logs
# every generation 1 and 2 collection with its time per process; gc_split.py splits each warm boot's collections by
# the boot log's phases for the stage process and the launcher.
# usage: D=... T=<kit tree> MAIN=<tree> bash queue_block56.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block56
L=$W/queue.log
mkdir -p $W/logs
echo "tree $MAIN $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $MAIN diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue block56 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

boot() {
  local label=$1 port=$2 census=$3
  local path=$MAIN
  if [ -n "$census" ]; then
    mkdir -p $census
    path=$K/gc_census:$MAIN
  else
    :
  fi
  local began=$(date +%s.%N)
  (cd $MAIN && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$path GC_CENSUS_DIR=$census TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor TRITON_CACHE_DIR=$W/cache/triton \
    setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port \
    --model-path $M17 --tts_engine.gpu 0 --vocoder.gpu 0" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 1
  local pgid=$(cat $W/logs/serve_$label.pid)
  until kill -0 $pgid 2> /dev/null && curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null; then
      echo "$label died; $(grep -E 'Error|error' $W/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      return 1
    else
      sleep 1
    fi
  done
  echo "$label healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s $(date +%T)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

rm -rf $W/cache
boot cold 23000 ""
for round in 1 2; do
  boot warm_$round $((23000 + 200 * round)) $W/census_$round
  python3 $K/gc_census/gc_split.py $W/logs/serve_warm_$round.log $W/census_$round > $W/gc_split_$round.txt 2>&1
  cat $W/gc_split_$round.txt >> $L
done
echo "queue block56 done $(date +%T)" >> $L
