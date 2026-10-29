#!/usr/bin/env bash
# Block 29 of 10-29: D (fix/qwen3-tts-reference-encoder-float32-exact-length 19f0f8de5, tree 4b8988068: the float32
# encoder and the exact-length encode, merged with main) against today's main on the cascade's 0.6B ICL sets, where its
# runaway effect was measured before A merged (10-28 block 3: length stops main 71 and 83 a run, A plus D 32 and 37).
# Seeded c16 every set, s = 1 to 4 with main on card 0, then s = 5 to 8 swapped; each server as the cascade profile
# starts it (static 0.08); scored on two lanes; pooled_pairs.py and babble_tally.py over the 8 seeds.
# usage: WAIT_PID=PID D=... T=... bash queue_block29.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/duplex_20261027
MAIN=/workspace/n29_main
REFENC=/workspace/n29_refenc
CASES=$D/block19/duplex/cases.jsonl
M06=Qwen/Qwen3-TTS-12Hz-0.6B-Base
W=$D/block29
L=$W/queue.log
mkdir -p $W/logs $W/runs
if [ -n "${WAIT_PID:-}" ]; then
  while [ -n "$(ps -o stat= -p $WAIT_PID 2> /dev/null | grep -v Z)" ]; do sleep 30; done
else
  :
fi
echo "queue block29 start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')D $(git -C $REFENC rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L

start_server() {
  local label=$1 tree=$2 port=$3
  shift 3
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $W/logs/serve_$label.pid) waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label not healthy after $waited s; $(grep -E 'Error|error' $W/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      kill -TERM -$pgid 2> /dev/null
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s, $(grep -o 'KV pool holds [0-9]* tokens' $W/logs/serve_$label.log | head -n 1) $(date +%T)" >> $L
}

stop_server() {
  local pgid=$(cat $W/logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

tts() {
  local card=$1
  echo "--model-path $M06 --config examples/configs/qwen3_tts_0_6b.yaml --mem-fraction-static 0.08 --tts_engine.gpu $card --vocoder.gpu $card"
}

draw() {
  local seed=$1 port_main=$2 port_d=$3
  local pids=()
  for spec in "main $port_main" "refenc $port_d"; do
    set -- $spec
    (cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py serve $CASES --base-url http://127.0.0.1:$2 --model $M06 \
      --label ${1}_s$seed --shape icl06 --out $W/runs/${1}_s${seed}_c16 --concurrency 16 \
      --seed $seed >> $W/logs/client_$1.log 2>&1) &
    pids+=($!)
  done
  wait ${pids[@]}
  echo "seed $seed: $(tail -n 1 $W/logs/client_main.log) / $(tail -n 1 $W/logs/client_refenc.log) $(date +%T)" >> $L
}

for spec in "o1 0 1 4770 4780 1 2 3 4" "o2 1 0 5170 5180 5 6 7 8"; do
  set -- $spec
  order=$1 card_main=$2 card_d=$3 port_main=$4 port_d=$5
  shift 5
  seeds="$*"
  start_server ${order}_main $MAIN $port_main $(tts $card_main) &
  a=$!
  start_server ${order}_refenc $REFENC $port_d $(tts $card_d) &
  b=$!
  wait $a $b
  for seed in $seeds; do
    draw $seed $port_main $port_d
  done
  stop_server ${order}_main
  stop_server ${order}_refenc
done

start_server asr0 $MAIN 5700 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 0 --mem-fraction-static 0.5 &
a=$!
start_server asr1 $MAIN 5720 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 1 --mem-fraction-static 0.5 &
b=$!
wait $a $b
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py score $W/runs/main_*_c16 --asr-port 5700 --sim-card 0 \
  >> $W/logs/score0.log 2>&1) &
a=$!
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py score $W/runs/refenc_*_c16 --asr-port 5720 --sim-card 1 \
  >> $W/logs/score1.log 2>&1) &
b=$!
wait $a
ra=$?
wait $b
echo "score rc $ra $? $(date +%T)" >> $L
stop_server asr0
stop_server asr1
dirs=""
for seed in 1 2 3 4 5 6 7 8; do dirs="$dirs $W/runs/main_s${seed}_c16 $W/runs/refenc_s${seed}_c16"; done
python3 $HERE/pooled_pairs.py refenc_s1_8 $dirs > $W/pooled.txt 2>&1
python3 $HERE/babble_tally.py main=$(ls -d $W/runs/main_*_c16 | paste -sd, -) refenc=$(ls -d $W/runs/refenc_*_c16 | paste -sd, -) \
  > $W/babble.txt 2>&1
echo "pooled $(cut -c1-160 $W/pooled.txt | tr '\n' ' ') $(date +%T)" >> $L
echo "queue block29 done $(date +%T)" >> $L
