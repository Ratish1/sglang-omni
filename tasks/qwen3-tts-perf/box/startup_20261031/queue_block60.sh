#!/usr/bin/env bash
# Block 60 of 11-01: startup item E in SGLang's form. Arms on upstream main: main; vfreeze, SGLang's
# freeze_gc(enable_cudagraph_gc=False) around the vocoder's whole capture pass, as its graph runners take it; stage, E as
# built (collect and freeze before each stage's construction); both. Default 1.7B Base on card 0, one boot at a time:
# each tree cold into its own cache, then three rounds of warm boots (order reversed in the second) with
# gc_census/sitecustomize.py on the path; gc_split.py per boot (GC seconds by phase), boot_split.py over all of them.
# Waits for block 59 first.
# usage: D=... T=<kit tree> MAIN=<tree> VFREEZE=<tree> STAGE=<tree> BOTH=<tree> bash queue_block60.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
VFREEZE=${VFREEZE:?the vocoder freeze tree}
STAGE=${STAGE:?the stage freeze tree}
BOTH=${BOTH:?the tree with both}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block60
L=$W/queue.log
mkdir -p $W/logs
until grep -q "queue block59 done" $D/block59/queue.log 2> /dev/null; do sleep 20; done
for arm in MAIN VFREEZE STAGE BOTH; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block60 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

boot() {
  local label=$1 arm=$2 port=$3 census=$4
  local tree=${!arm}
  local path=$tree
  if [ -n "$census" ]; then
    mkdir -p $census
    path=$K/gc_census:$tree
  else
    :
  fi
  local began=$(date +%s.%N)
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$path GC_CENSUS_DIR=$census TORCHINDUCTOR_CACHE_DIR=$W/cache_${arm,,}/inductor TRITON_CACHE_DIR=$W/cache_${arm,,}/triton \
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

port=36000
for arm in MAIN VFREEZE STAGE BOTH; do
  boot cold_${arm,,} $arm $port ""
  port=$((port + 200))
done
for round in 1 2 3; do
  if [ $round -eq 2 ]; then
    order="BOTH STAGE VFREEZE MAIN"
  else
    order="MAIN VFREEZE STAGE BOTH"
  fi
  for arm in $order; do
    label=warm_${arm,,}_$round
    boot $label $arm $port $W/census_${arm,,}_$round
    python3 $K/gc_census/gc_split.py $W/logs/serve_$label.log $W/census_${arm,,}_$round > $W/gc_split_${arm,,}_$round.txt 2>&1
    echo "$label gc: $(grep -A7 '^== stage process' $W/gc_split_${arm,,}_$round.txt | grep -E '^(vocoder|engine|total) ' | tr '\n' '#')" >> $L
    port=$((port + 200))
  done
done
python3 $K/boot_split.py $W > $W/split.txt 2>&1
tail -n 6 $W/split.txt >> $L
echo "queue block60 done $(date +%T)" >> $L
