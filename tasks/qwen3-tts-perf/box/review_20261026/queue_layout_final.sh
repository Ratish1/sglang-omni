#!/usr/bin/env bash
# The configuration queue_layout.sh points to, measured directly after it: the stream-text prompt (the clone default),
# a two-frame minimum (official min_new_tokens 2) and the x-vector leading-silence mask off; 0.6B and 1.7B, every case,
# both languages, default server settings; then transcribed, scored for similarity, and the summary rebuilt with every
# arm. Card 1 only.
set -u
D=${D:?the layout results directory}
P=${P:?the probe tree}
T=${T:?the kit tree}
CASES=${CASES:?the parity cases}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
CAP=300
L=$D/queue.log
until grep -q queue_layout_done $L; do
  sleep 30
done
echo "queue layout final start $(date +%T) probe $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
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
export QWEN3_TTS_PROBE_MIN_NEW_TOKENS=2
port=27000
for spec in "06 Qwen/Qwen3-TTS-12Hz-0.6B-Base" "17 Qwen/Qwen3-TTS-12Hz-1.7B-Base"; do
  set -- $spec
  label=${1}_min2_nomask
  echo "$label env [QWEN3_TTS_PROBE_MIN_NEW_TOKENS=2] flags [--tts_engine.factory.leading_silence_mask_frames 0]" >> $L
  if start_server $label $port --model-path $2 --tts_engine.gpu $CARD --vocoder.gpu $CARD \
    --tts_engine.factory.leading_silence_mask_frames 0; then
    for language in auto English; do
      (cd $P && PYTHONPATH=$P python3 $HERE/layout.py serve $CASES --base-url http://127.0.0.1:$port --model $2 \
        --label $label --language $language --out $D/served/${label}_$language --max-new-tokens $CAP >> $L 2>&1)
      echo "served $label $language rc $? $(date +%T)" >> $L
    done
    stop_server $label
  else
    :
  fi
  port=$((port + 200))
done
unset QWEN3_TTS_PROBE_MIN_NEW_TOKENS
export MASTER_PORT=27850
if start_server asr_final 27800 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu $CARD; then
  (cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $D/served/*_min2_nomask_* --asr-port 27800 >> $L 2>&1)
  echo "transcribed final rc $? $(date +%T)" >> $L
  stop_server asr_final
else
  :
fi
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $D/served/*_min2_nomask_* --sim-card $CARD >> $L 2>&1)
echo "similarity final rc $? $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py summary $D/official_*.json --served $D/served/* --max-new-tokens $CAP \
  > $D/summary.txt 2>&1)
echo "summary final rc $? $(date +%T)" >> $L
echo "queue_layout_final_done $(date +%T)" >> $L
