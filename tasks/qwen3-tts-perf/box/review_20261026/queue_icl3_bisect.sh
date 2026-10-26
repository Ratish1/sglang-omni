#!/usr/bin/env bash
# 0.6B ICL last three words, language English: ours runs away at 0.78 % against official qwen-tts's 0.39 % (five
# draws each) and against our own auto's 0.31 %, while official English and auto match. Main, one server per
# configuration, two English draws each at c16: default, the talker's radix cache off, CUDA graphs off; on the default
# server also two auto draws and one English draw at c1. Cap 300 as in the official draws. Card 1 only, after
# queue_branches.sh.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
M=${M:?the main tree}
CASES=${CASES:?the ICL last-three cases}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
L=$D/queue.log
mkdir -p $D/icl3
until grep -q queue_branches_done $L; do
  sleep 30
done
echo "queue icl3 bisect start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
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
      echo "$label server not healthy after $waited s; $(grep -E 'Error|error' $D/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      kill -TERM -$pgid 2> /dev/null
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s, pgid $pgid, card $(nvidia-smi -i $CARD --query-gpu=memory.used --format=csv,noheader) $(date +%T)" >> $L
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
draw() {
  local label=$1 concurrency=$2 language=${3:-English}
  (cd $M && PYTHONPATH=$M python3 $HERE/layout.py serve $CASES --base-url http://127.0.0.1:24000 \
    --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --label $label --language $language --out $D/icl3/$label --modes icl \
    --max-new-tokens 300 --concurrency $concurrency >> $L 2>&1)
  echo "icl3 $label rc $? $(date +%T)" >> $L
}
for config in "default|" "radix_off|--tts_engine.engine.disable_radix_cache true" \
  "graphs_off|--tts_engine.engine.disable_cuda_graph true"; do
  IFS='|' read -r name flags <<< "$config"
  if start_server icl3_$name 24000 --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --tts_engine.gpu $CARD \
    --vocoder.gpu $CARD $flags; then
    draw 06_${name}_1 16
    draw 06_${name}_2 16
    if [ $name = default ]; then
      draw 06_default_auto_1 16 auto
      draw 06_default_auto_2 16 auto
      draw 06_default_c1 1
    else
      :
    fi
    stop_server icl3_$name
  else
    :
  fi
done
echo "queue_icl3_bisect_done $(date +%T)" >> $L
