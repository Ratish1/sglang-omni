#!/usr/bin/env bash
# Block 145 of 11-05: the streaming c64 first chunk regression on the rows head, by the head's own components. Each arm is the head with
# one streaming-visible change put back: slots (one reserved slot), graphs (whole-utterance graphs captured), wait (the batch wait kept).
# The head itself and the fix tree (same healthy path as main) are the two ends. Each arm against main, same time, own sockets, request
# events on, both card orders, twice.
# usage: D=... T=<kit tree> MAIN=<tree> HEAD=<tree> FIX=<tree> SLOTS=<tree> GRAPHS=<tree> WAIT=<tree> HARNESS=<tree> [BLOCK=block145] bash queue_block145.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
HEAD=${HEAD:?the head}
FIX=${FIX:?the fix tree}
SLOTS=${SLOTS:?the slots arm}
GRAPHS=${GRAPHS:?the graphs arm}
WAIT=${WAIT:?the wait arm}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block145}
S=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
export CARD_CPUS="0=0-31,64-95;1=32-63,96-127"
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
mkdir -p $W
for arm in MAIN HEAD FIX SLOTS GRAPHS WAIT; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
# note: 9400, 9879 and 16000 are standing listeners on this host, so ports stay between them
port=10000
tail_arm() {
  local name=$1 tree=$2 card=$3 at=$4
  (cd $S && timeout 1200 python3 first_chunk_tail.py --tree $tree --card $card --port $at --root $W/$name --harness $HARNESS --concurrency 64 > $W/$name.txt 2>&1)
  echo "$name card $card rc $? $(date +%T) $(grep -E '^stream done' $W/$name.txt | cut -c1-110) keys $(grep -aom1 '#tokens: [0-9]*' $W/$name/serve.log)" >> $L
}
pair() {
  local arm=$1 tree=$2 n=$3 main_card=$4
  tail_arm main_${arm}_$n $MAIN $main_card $port &
  tail_arm ${arm}_$n $tree $((1 - main_card)) $((port + 200)) &
  wait
  port=$((port + 400))
  if [ $port -ge 15000 ]; then port=10000; else :; fi
}
for arm in slots graphs wait head fix; do
  case $arm in
    slots) tree=$SLOTS ;;
    graphs) tree=$GRAPHS ;;
    wait) tree=$WAIT ;;
    head) tree=$HEAD ;;
    fix) tree=$FIX ;;
  esac
  pair $arm $tree 0 0
  pair $arm $tree 1 1
  pair $arm $tree 2 0
  pair $arm $tree 3 1
  python3 $S/first_chunk_bisect_read.py $W $arm 2>&1 | cut -c1-140 >> $L
done
python3 $S/first_chunk_bisect_read.py $W slots graphs wait head fix > $W/bisect.txt 2>&1
echo "queue $BLOCK done $(date +%T)" >> $L
