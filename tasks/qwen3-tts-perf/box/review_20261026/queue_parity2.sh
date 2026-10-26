#!/usr/bin/env bash
# queue_parity.sh with its arms side by side on card 1: the official qwen-tts arms (0.6B then 1.7B, batches of 128) in
# the background, our served arms (0.6B and 1.7B, as shipped and with the x-vector leading-silence mask off) in the
# foreground at --mem-fraction-static 0.12, the cascade profiles' own setting. Rates and lengths are compared once
# both are done; nothing here is timed, so sharing the card changes no result.
set -u
D=${D:?the results directory}
M=${M:?the main tree}
T=${T:?the kit tree}
REFPY=${REFPY:?the reference venv python}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
CAP=300
L=$D/queue.log
echo "queue parity2 start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) cases $(wc -l < $D/cases.jsonl)" >> $L
(
  for spec in "06 Qwen/Qwen3-TTS-12Hz-0.6B-Base" "17 Qwen/Qwen3-TTS-12Hz-1.7B-Base"; do
    set -- $spec
    (cd $D && $REFPY $HERE/parity_reference.py $D/cases.jsonl $D/official_$1.json --model $2 --card $CARD \
      --batch 128 --max-new-tokens $CAP > $D/official_$1.log 2>&1)
    echo "official $1 rc $? $(grep -c reached $D/official_$1.log) batches $(date +%T)" >> $L
  done
) &
OFFICIAL=$!
start_server() {
  local label=$1 port=$2
  shift 2
  (cd $M && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$M setsid bash -c "echo \$\$ > $D/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $D/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $D/serve_$label.pid)
  local waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label server not healthy after $waited s; $(tail -n 3 $D/serve_$label.log | tr '\n' ' ')" >> $L
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
port=22000
for spec in "06 Qwen/Qwen3-TTS-12Hz-0.6B-Base" "17 Qwen/Qwen3-TTS-12Hz-1.7B-Base"; do
  set -- $spec
  size=$1 model=$2
  for arm in "shipped" "nomask --tts_engine.factory.leading_silence_mask_frames 0"; do
    set -- $arm
    label=${size}_$1
    shift
    if start_server $label $port --model-path $model --tts_engine.gpu $CARD --vocoder.gpu $CARD \
      --mem-fraction-static 0.12 "$@"; then
      (cd $M && PYTHONPATH=$M python3 $HERE/parity.py serve $D/cases.jsonl --base-url http://127.0.0.1:$port \
        --model $model --label $label --out $D/served_$label.jsonl --max-new-tokens $CAP >> $L 2>&1)
      echo "served $label rc $? $(date +%T)" >> $L
      stop_server $label
    else
      :
    fi
    port=$((port + 200))
  done
done
wait $OFFICIAL
for size in 06 17; do
  (cd $M && PYTHONPATH=$M python3 $HERE/parity.py compare $D/official_$size.json $D/served_${size}_shipped.jsonl \
    $D/served_${size}_nomask.jsonl --max-new-tokens $CAP > $D/compare_$size.txt 2>&1)
  cat $D/compare_$size.txt >> $L
done
echo "queue_parity_done $(date +%T)" >> $L
