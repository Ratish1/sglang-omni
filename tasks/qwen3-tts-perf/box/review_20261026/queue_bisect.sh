#!/usr/bin/env bash
# Which component makes our 0.6B ICL one-word clones run away (29 of 1,088 against official qwen-tts's 0 of 1,088 in
# queue_parity2.sh): the 1,088 "icl last 1" cases, the cascade's requests, through the probe tree (identical to main
# without its environment toggles) as shipped, with every difference from official at once (min_new_tokens 2, the
# plain predictor, talker and predictor graphs off, the radix cache off), and with each alone. Card 1, servers at
# --mem-fraction-static 0.12 beside the parity queue; one 300-frame cap.
set -u
D=${D:?the results directory}
P=${P:?the probe tree}
T=${T:?the kit tree}
CASES=${CASES:?the parity cases}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
CAP=300
MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
L=$D/queue.log
mkdir -p $D
grep '"set": "icl last 1"' $CASES > $D/cases.jsonl
echo "queue bisect start $(date +%T) probe $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) cases $(wc -l < $D/cases.jsonl)" >> $L
start_server() {
  local label=$1 port=$2
  shift 2
  (cd $P && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$P setsid bash -c "echo \$\$ > $D/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $D/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $D/serve_$label.pid)
  local waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label server not healthy after $waited s; $(grep -E 'Error|error' $D/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      kill -TERM -$pgid 2> /dev/null
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s, pgid $pgid $(date +%T)" >> $L
}
stop_server() {
  local pgid=$(cat $D/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do
    kill -0 $pgid 2> /dev/null || break
    sleep 1
  done
  kill -KILL -$pgid 2> /dev/null
  echo "$1 stopped $(date +%T)" >> $L
}
port=24000
GRAPHS_OFF="--tts_engine.engine.disable_cuda_graph true"
RADIX_OFF="--tts_engine.engine.disable_radix_cache true"
for arm in "B0_shipped||" \
  "B1_all_official|QWEN3_TTS_PROBE_MIN_NEW_TOKENS=2 QWEN3_TTS_PROBE_PLAIN_PREDICTOR=1 SGLANG_OMNI_QTTS_PREDICTOR_GRAPH=0|$GRAPHS_OFF $RADIX_OFF" \
  "B2_graphs_off|SGLANG_OMNI_QTTS_PREDICTOR_GRAPH=0|$GRAPHS_OFF" \
  "B3_radix_off||$RADIX_OFF" \
  "B5_min_new_tokens_2|QWEN3_TTS_PROBE_MIN_NEW_TOKENS=2|" \
  "B6_plain_predictor|QWEN3_TTS_PROBE_PLAIN_PREDICTOR=1|"; do
  label=${arm%%|*}
  rest=${arm#*|}
  env_vars=${rest%%|*}
  flags=${rest#*|}
  unset QWEN3_TTS_PROBE_MIN_NEW_TOKENS QWEN3_TTS_PROBE_PLAIN_PREDICTOR SGLANG_OMNI_QTTS_PREDICTOR_GRAPH
  for assignment in $env_vars; do
    export "$assignment"
  done
  echo "$label env [$env_vars] flags [$flags]" >> $L
  if start_server $label $port --model-path $MODEL --tts_engine.gpu $CARD --vocoder.gpu $CARD \
    --mem-fraction-static 0.12 $flags; then
    (cd $P && PYTHONPATH=$P python3 $HERE/parity.py serve $D/cases.jsonl --base-url http://127.0.0.1:$port \
      --model $MODEL --label $label --out $D/served_$label.jsonl --max-new-tokens $CAP >> $L 2>&1)
    echo "served $label rc $? $(date +%T)" >> $L
    stop_server $label
  else
    :
  fi
  port=$((port + 200))
done
echo "queue_bisect_done $(date +%T)" >> $L
