#!/usr/bin/env bash
# Block 36 of 10-31: #3368 with the encoder's float32 weights given only where references are encoded, the engine and a
# frontend in its own process (NEW), against #3368's head that gave them in the shared tokenizer loader, so a vocoder in
# its own process held a float32 encoder it never runs (OLD, 139ee3649). Both heads at once, cards swapped per order:
# (1) the KV pool each one-process cascade profile sizes (0.6B static 0.08, 1.7B static 0.12), since the conversion now
#     runs later in the one-process startup;
# (2) every stage's GPU memory with the frontend, the engine and the vocoder each in its own process (the cookbook's
#     composed layout, fractions for an 80 GB card), read on the host by proc_memory.sh while the pair waits for
#     $W/mem_<order>.done;
# (3) seeded c1 non-streaming SeedTTS EN 1.7B Base, byte identity, and each server's silence-probe ids.
# usage: D=... T=... OLD=<tree> NEW=<tree> bash queue_block36.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
OLD=${OLD:?the old tree}
NEW=${NEW:?the new tree}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block36
L=$W/queue.log
export ARM_ROOT=$W/arms
mkdir -p $W/logs $ARM_ROOT
echo "queue block36 start $(date +%T) old $(git -C $OLD rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')new $(git -C $NEW rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L

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

profile() {
  local size=$1 card=$2
  if [ $size = 06 ]; then
    echo "--model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --config examples/configs/qwen3_tts_0_6b.yaml --mem-fraction-static 0.08 --tts_engine.gpu $card --vocoder.gpu $card"
  elif [ $size = 17 ]; then
    echo "--model-path $M17 --config examples/configs/qwen3_tts_1_7b.yaml --mem-fraction-static 0.12 --tts_engine.gpu $card --vocoder.gpu $card"
  else
    echo "--model-path $M17 --preprocessing.process tts_frontend --preprocessing.gpu $card --preprocessing.gpu_memory_fraction 0.05 --vocoder.process vocoder --vocoder.gpu $card --vocoder.gpu_memory_fraction 0.15 --tts_engine.gpu $card --tts_engine.gpu_memory_fraction 0.72 --tts_engine.engine.mem_fraction_static 0.72"
  fi
}

for spec in "06_o1 06 0 1 20000 20200" "06_o2 06 1 0 20400 20600" "17_o1 17 0 1 20800 21000" "17_o2 17 1 0 21200 21400"; do
  set -- $spec
  name=$1 size=$2 card_old=$3 card_new=$4 port_old=$5 port_new=$6
  start_server ${name}_old $OLD $port_old $(profile $size $card_old) &
  a=$!
  start_server ${name}_new $NEW $port_new $(profile $size $card_new) &
  b=$!
  wait $a $b
  stop_server ${name}_old
  stop_server ${name}_new
done

for spec in "split_o1 0 1 22000 22400" "split_o2 1 0 22800 23200"; do
  set -- $spec
  name=$1 card_old=$2 card_new=$3 port_old=$4 port_new=$5
  start_server ${name}_old $OLD $port_old $(profile split $card_old) &
  a=$!
  start_server ${name}_new $NEW $port_new $(profile split $card_new) &
  b=$!
  wait $a $b
  echo "$name healthy, waiting for the host memory read ($W/mem_$name.done) $(date +%T)" >> $L
  for second in $(seq 900); do [ -f $W/mem_$name.done ] && break; sleep 1; done
  stop_server ${name}_old
  stop_server ${name}_new
done

N1="--concurrency 1 --no-stream --seed 1234"
python3 $ARM --label c1n_old --tree $OLD --card 0 --port 26000 $N1 --barrier $W/barrier/c1n >> $W/logs/arm_c1n_old.log 2>&1 &
a=$!
python3 $ARM --label c1n_new --tree $NEW --card 1 --port 26200 $N1 --barrier $W/barrier/c1n >> $W/logs/arm_c1n_new.log 2>&1 &
b=$!
wait $a $b
echo "c1n: $(tail -n 1 $W/logs/arm_c1n_old.log | cut -c1-160) / $(tail -n 1 $W/logs/arm_c1n_new.log | cut -c1-160)" >> $L
echo "c1n identity: $(python3 $ARM --compare c1n_old c1n_new) $(date +%T)" >> $L
for label in c1n_old c1n_new; do
  echo "$label $(grep -o 'masks [0-9]* silence codec ids.*' $ARM_ROOT/$label/serve.log | head -n 1)" >> $L
done
echo "queue block36 done $(date +%T)" >> $L
