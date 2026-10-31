#!/usr/bin/env bash
# Block 42 of 10-31: main's startup attributed to functions (startup_20261031/PLAN.md). One card, nothing else on it.
# (1) python3 -X importtime of the launcher's import and of the Qwen3-TTS stage modules. (2) The default 1.7B Base server
# under py-spy (every process, idle samples included, 100 Hz, raw stacks), cold (empty inductor and Triton caches) then
# warm (the same caches): time to health, the server log's timeline, and the stacks reduced per process to inclusive and
# leaf seconds per function. Each server is stopped by TERM to the server process once healthy, so py-spy writes its file.
# usage: D=... T=<kit tree> MAIN=<tree> CARD=<card> PORT=<port> bash queue_block42.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
CARD=${CARD:?the card}
PORT=${PORT:?the first port}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block42
L=$W/queue.log
mkdir -p $W/logs
echo "tree $MAIN $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $MAIN diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue block42 start $(date +%T) kit $(git -C $T rev-parse HEAD) card $CARD; gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

importtime() {
  local name=$1 modules=$2
  (cd $MAIN && PYTHONPATH=$MAIN python3 -X importtime -c "import $modules" 2> $W/importtime_$name.txt > /dev/null)
  echo "importtime $name: $(awk -F'|' '/import time:/ && $2 ~ /[0-9]/ {if ($2 + 0 > top) top = $2 + 0} END {printf "%.1f s cumulative at the top", top / 1e6}' $W/importtime_$name.txt)" >> $L
  sort -t'|' -k2 -n -r $W/importtime_$name.txt | head -n 30 > $W/importtime_${name}_top.txt
}

importtime cli sglang_omni.cli
importtime stages "sglang_omni.models.qwen3_tts.engine_builder, sglang_omni.models.qwen3_tts.streaming_vocoder, sglang_omni.models.qwen3_tts.stages"

profile_boot() {
  local phase=$1 port=$2
  local began=$(date +%s.%N)
  (cd $MAIN && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$MAIN TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor TRITON_CACHE_DIR=$W/cache/triton \
    setsid bash -c "echo \$\$ > $W/logs/pyspy_$phase.pid; exec py-spy record --subprocesses --idle --rate 100 --format raw \
    -o $W/pyspy_$phase.txt -- python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port --model-path $M17 \
    --tts_engine.gpu $CARD --vocoder.gpu $CARD" > $W/logs/serve_$phase.log 2>&1 < /dev/null &)
  sleep 2
  local pyspy=$(cat $W/logs/pyspy_$phase.pid) waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pyspy 2> /dev/null || [ $waited -ge 1200 ]; then
      echo "$phase not healthy after $waited s; $(grep -E 'Error|error' $W/logs/serve_$phase.log | tail -n 2 | tr '\n' ' ')" >> $L
      kill -TERM -$pyspy 2> /dev/null
      return 1
    else
      sleep 1
      waited=$((waited + 1))
    fi
  done
  echo "$phase healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s $(date +%T)" >> $L
  local server=$(pgrep -P $pyspy -f sglang_omni.cli | head -n 1)
  kill -TERM $server
  for second in $(seq 180); do kill -0 $pyspy 2> /dev/null || break; sleep 1; done
  kill -TERM -$pyspy 2> /dev/null
  python3 $K/startup_20261031/pyspy_reduce.py $W/pyspy_$phase.txt --rate 100 > $W/pyspy_${phase}_reduced.txt 2>&1
  python3 $K/startup_20261031/boot_phases.py $W/logs/serve_$phase.log --min-gap 1.0 > $W/phases_$phase.txt 2>&1
  echo "$phase reduced: $(grep -E '^== ' $W/pyspy_${phase}_reduced.txt | cut -c1-140 | tr '\n' '#')" >> $L
}

rm -rf $W/cache
profile_boot cold $PORT
profile_boot warm $((PORT + 200))
echo "queue block42 done $(date +%T)" >> $L
