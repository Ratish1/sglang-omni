#!/usr/bin/env bash
# Block 32 of 10-30: D's cuBLAS workspace made on the device SGLang already set (fix/qwen3-tts-reference-encoder-float32-
# exact-length f0669a0d2, tree 8514748fe) against D's previous head (98b1c0d8a, tree 4b8988068, the explicit device
# context). The only difference is where the workspace is made, so the KV pool each server sizes must be equal. Each
# server as the cascade profile starts it (0.6B static 0.08, 1.7B static 0.12), both heads at once, both card orders.
# usage: D=... T=... bash queue_block32.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
OLD=/workspace/n30_d_old
NEW=/workspace/n30_d
W=$D/block32
L=$W/queue.log
mkdir -p $W/logs
echo "queue block32 start $(date +%T) old $(git -C $OLD rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')new $(git -C $NEW rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L

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
  else
    echo "--model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --config examples/configs/qwen3_tts_1_7b.yaml --mem-fraction-static 0.12 --tts_engine.gpu $card --vocoder.gpu $card"
  fi
}

for spec in "06_o1 06 0 1 6100 6110" "06_o2 06 1 0 6120 6130" "17_o1 17 0 1 6140 6150" "17_o2 17 1 0 6160 6170"; do
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
echo "queue block32 done $(date +%T)" >> $L
