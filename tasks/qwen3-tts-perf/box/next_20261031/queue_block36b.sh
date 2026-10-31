#!/usr/bin/env bash
# Block 36b of 10-31: block 36's composed-layout part again. In block 36 the new head's server could not bind 22400,
# held by another session's MiniCPM-o server on 0.0.0.0 (the old head's +100 port 22100 too), and the health check
# answered from that server. Here a port that already answers is refused, and a healthy server must be ours.
# #3368's head that gives the encoder float32 weights in the shared tokenizer loader (OLD, 139ee3649) against the head
# that gives them only where references are encoded (NEW): with the frontend, the engine and the vocoder each in its
# own process, the engine's KV pool and every stage's GPU memory, read on the host by proc_memory.sh while the pair
# waits for $W/mem_<order>.done; both orders.
# usage: D=... T=... OLD=<tree> NEW=<tree> bash queue_block36b.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
OLD=${OLD:?the old tree}
NEW=${NEW:?the new tree}
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block36b
L=$W/queue.log
mkdir -p $W/logs
echo "queue block36b start $(date +%T) old $(git -C $OLD rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')new $(git -C $NEW rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L

start_server() {
  local label=$1 tree=$2 port=$3
  shift 3
  if (echo > /dev/tcp/127.0.0.1/$port) 2> /dev/null; then
    echo "$label port $port already answers, not started" >> $L
    return 1
  else
    :
  fi
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $W/logs/serve_$label.pid) waited=0
  until kill -0 $pgid 2> /dev/null && curl -sf http://127.0.0.1:$port/health > /dev/null; do
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

composed() {
  local card=$1
  echo "--model-path $M17 --preprocessing.process tts_frontend --preprocessing.gpu $card --preprocessing.gpu_memory_fraction 0.05 --vocoder.process vocoder --vocoder.gpu $card --vocoder.gpu_memory_fraction 0.15 --tts_engine.gpu $card --tts_engine.gpu_memory_fraction 0.72 --tts_engine.engine.mem_fraction_static 0.72"
}

for spec in "split_o1 0 1 12000 12400" "split_o2 1 0 13000 13400"; do
  set -- $spec
  name=$1 card_old=$2 card_new=$3 port_old=$4 port_new=$5
  start_server ${name}_old $OLD $port_old $(composed $card_old) &
  a=$!
  start_server ${name}_new $NEW $port_new $(composed $card_new) &
  b=$!
  wait $a $b
  echo "$name healthy, waiting for the host memory read ($W/mem_$name.done) $(date +%T)" >> $L
  for second in $(seq 900); do [ -f $W/mem_$name.done ] && break; sleep 1; done
  stop_server ${name}_old
  stop_server ${name}_new
done
echo "queue block36b done $(date +%T)" >> $L
