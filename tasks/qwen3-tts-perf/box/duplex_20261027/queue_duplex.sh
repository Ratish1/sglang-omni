#!/usr/bin/env bash
# Full-duplex A/B: the cascade's Qwen3-TTS requests (duplex_ab.py) against main and the first batch, each server
# started as the profiles start it (their config and static fraction), one arm a card, then the cards swapped.
# Per model pair: one request at a time on the English sentences (first audio as one conversation hears it), barge-in
# interrupts, then 16 at a time on every set. Then Qwen3-ASR on card 0 and WavLM on card 1 score the c16 clips.
set -u
D=${D:?the results directory}
MAIN=${MAIN:?the main tree}
FIRST=${FIRST:?the first-batch tree}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/duplex_20261027
CASES=$D/cases/cases.jsonl
L=$D/queue.log
mkdir -p $D/logs $D/runs
echo "queue duplex start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')first $(git -C $FIRST rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
start_server() {
  local label=$1 tree=$2 port=$3
  shift 3
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $D/logs/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $D/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $D/logs/serve_$label.pid)
  local waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label server not healthy after $waited s; $(grep -E 'Error|error' $D/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      kill -TERM -$pgid 2> /dev/null
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s, pgid $pgid, imports $(cd $tree && PYTHONPATH=$tree python3 -c 'import sglang_omni; print(sglang_omni.__file__)' 2> /dev/null), $(grep -o 'KV pool holds [0-9]* tokens, [0-9.]* GiB' $D/logs/serve_$label.log | head -n 1) $(date +%T)" >> $L
}
stop_server() {
  local pgid=$(cat $D/logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do
    kill -0 $pgid 2> /dev/null || break
    sleep 1
  done
  kill -KILL -$pgid 2> /dev/null
  echo "$1 stopped $(date +%T)" >> $L
}
serve_both() {
  local order=$1 model=$2 shape=$3 concurrency=$4 tag=$5
  shift 5
  for arm in main first; do
    local port=${PORT[$arm]}
    (cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py serve $CASES --base-url http://127.0.0.1:$port --model $model \
      --label ${arm}_$order --shape $shape --out $D/runs/${order}_${arm}_${shape}_$tag --concurrency $concurrency "$@" \
      >> $D/logs/client_${order}_${arm}.log 2>&1) &
  done
  wait
  echo "$order $shape $tag: $(tail -n 1 $D/logs/client_${order}_main.log) / $(tail -n 1 $D/logs/client_${order}_first.log) $(date +%T)" >> $L
}
declare -A PORT
for order in o1 o2; do
  if [ $order = o1 ]; then
    CARD_MAIN=0; CARD_FIRST=1; PORT=([main]=4610 [first]=4620)
  else
    CARD_MAIN=1; CARD_FIRST=0; PORT=([main]=4630 [first]=4640)
  fi
  for size in 06 17; do
    if [ $size = 06 ]; then
      model=Qwen/Qwen3-TTS-12Hz-0.6B-Base; config=examples/configs/qwen3_tts_0_6b.yaml; mem=0.08
    else
      model=Qwen/Qwen3-TTS-12Hz-1.7B-Base; config=examples/configs/qwen3_tts_1_7b.yaml; mem=0.12
    fi
    start_server ${order}_main_$size $MAIN ${PORT[main]} --model-path $model --config $config --mem-fraction-static $mem \
      --tts_engine.gpu $CARD_MAIN --vocoder.gpu $CARD_MAIN || exit 1
    start_server ${order}_first_$size $FIRST ${PORT[first]} --model-path $model --config $config --mem-fraction-static $mem \
      --tts_engine.gpu $CARD_FIRST --vocoder.gpu $CARD_FIRST || exit 1
    if [ $size = 06 ]; then
      serve_both $order $model icl06 1 c1 --sets "en sentence"
      serve_both $order $model icl06 16 interrupt --sets "en sentence" --interrupt
      serve_both $order $model icl06 16 c16
    else
      serve_both $order $model xvec17 1 c1 --sets "en sentence"
      serve_both $order $model icl17zh 1 c1 --sets "en sentence"
      serve_both $order $model xvec17 16 interrupt --sets "en sentence" --interrupt
      serve_both $order $model xvec17 16 c16
      serve_both $order $model icl17zh 16 c16
    fi
    stop_server ${order}_main_$size
    stop_server ${order}_first_$size
  done
done
start_server asr $MAIN 4700 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 0 --mem-fraction-static 0.5 || exit 1
DIRS=$(ls -d $D/runs/*_c16)
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py score $DIRS --asr-port 4700 --sim-card 1 >> $D/logs/score.log 2>&1)
echo "score rc $? $(tail -n 1 $D/logs/score.log) $(date +%T)" >> $L
stop_server asr
for run in icl06_c1 icl06_c16 icl06_interrupt xvec17_c1 xvec17_c16 xvec17_interrupt icl17zh_c1 icl17zh_c16; do
  for pair in "o1_main o1_first" "o2_main o2_first" "o1_main o2_main"; do
    set -- $pair
    echo "$D/runs/${1}_$run $D/runs/${2}_$run"
  done
done | xargs bash -c "cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py compare \"\$@\"" _ > $D/compare.txt 2>&1
echo "queue duplex done $(date +%T)" >> $L
