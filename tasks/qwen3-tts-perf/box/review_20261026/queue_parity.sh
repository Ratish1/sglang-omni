#!/usr/bin/env bash
# Runaways and natural lengths of Base voice clones, official qwen-tts 0.1.1 against our server on the cascade's own
# requests (parity.py, parity_reference.py): SeedTTS EN sentences, last words and last three words, ICL and
# x-vector-only, 0.6B and 1.7B, one 300-frame cap in every arm. Our server runs as shipped and with the x-vector
# leading-silence mask off (#2357, the one sampling change against official on this path). Card 1 only, one process
# at a time.
set -u
D=${D:?the results directory}
M=${M:?the main tree}
T=${T:?the kit tree}
REFPY=${REFPY:?the reference venv python}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
CAP=300
L=$D/queue.log
mkdir -p $D
until grep -q refenv_done /data/refenv_build/build.log 2> /dev/null; do
  sleep 20
done
echo "queue parity start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) $(grep '^refenv' /data/refenv_build/build.log) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
(cd $M && PYTHONPATH=$M python3 $HERE/parity.py cases $D >> $L 2>&1)
if [ ! -s $D/cases.jsonl ]; then
  echo "stopped: no cases" >> $L
  echo "queue_parity_done $(date +%T)" >> $L
  exit 1
else
  :
fi
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
for spec in "06 Qwen/Qwen3-TTS-12Hz-0.6B-Base 22000" "17 Qwen/Qwen3-TTS-12Hz-1.7B-Base 22400"; do
  set -- $spec
  size=$1 model=$2 port=$3
  (cd $D && $REFPY $HERE/parity_reference.py $D/cases.jsonl $D/official_$size.json --model $model --card $CARD \
    --max-new-tokens $CAP > $D/official_$size.log 2>&1)
  echo "official $size rc $? $(grep -c reached $D/official_$size.log) batches $(date +%T)" >> $L
  for arm in "shipped" "nomask --tts_engine.factory.leading_silence_mask_frames 0"; do
    set -- $arm
    label=${size}_$1
    shift
    if start_server $label $port --model-path $model --tts_engine.gpu $CARD --vocoder.gpu $CARD "$@"; then
      (cd $M && PYTHONPATH=$M python3 $HERE/parity.py serve $D/cases.jsonl --base-url http://127.0.0.1:$port \
        --model $model --label $label --out $D/served_$label.jsonl --max-new-tokens $CAP >> $L 2>&1)
      echo "served $label rc $? $(date +%T)" >> $L
      stop_server $label
    else
      :
    fi
    port=$((port + 200))
  done
  (cd $M && PYTHONPATH=$M python3 $HERE/parity.py compare $D/official_$size.json $D/served_${size}_shipped.jsonl \
    $D/served_${size}_nomask.jsonl --max-new-tokens $CAP > $D/compare_$size.txt 2>&1)
  cat $D/compare_$size.txt >> $L
done
echo "queue_parity_done $(date +%T)" >> $L
