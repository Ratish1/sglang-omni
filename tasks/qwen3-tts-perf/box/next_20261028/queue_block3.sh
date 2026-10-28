#!/usr/bin/env bash
# Block 3 of 10-28: the full-duplex A/B of 10-27 (duplex_20261027/queue_duplex.sh) on the final heads: main against the
# first batch (the two-frame minimum merged with the encoder branch's final head), each server started as the cascade
# profiles start it. Both card orders at once on four cards (o1: main on card 0, first on card 1; o2: main on card 3,
# first on card 2), then Qwen3-ASR on card 0 and WavLM on card 1 score every c16 run of both orders, then the compare.
# Waits for block 2.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/duplex_20261027
REPO=/workspace/sglang-omni
MAIN=/workspace/n28_main
FIRST=/workspace/n28_first2
CASES=$D/duplex/cases.jsonl
L=$D/queue_3.log
mkdir -p $D/logs $D/runs
until grep -q "queue block2 done" $D/queue_2.log 2> /dev/null; do sleep 30; done
git -C $REPO fetch -q origin test/qwen3-tts-duplex-first-batch-1028
rm -rf $FIRST
git -C $REPO worktree prune
git -C $REPO worktree add -q --detach $FIRST 76f3a79e5
echo "queue block3 start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')first $(git -C $FIRST rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) cases $(wc -l < $CASES)" >> $L

start_server() {
  local label=$1 tree=$2 port=$3
  shift 3
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $D/logs/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $D/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $D/logs/serve_$label.pid) waited=0
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
  echo "$label healthy after $waited s, pgid $pgid, $(grep -o 'KV pool holds [0-9]* tokens, [0-9.]* GiB' $D/logs/serve_$label.log | head -n 1) $(date +%T)" >> $L
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
  local order=$1 model=$2 shape=$3 concurrency=$4 tag=$5 port_main=$6 port_first=$7
  shift 7
  local pids=()
  for arm in main first; do
    local port=$port_main
    if [ $arm = first ]; then port=$port_first; else :; fi
    (cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py serve $CASES --base-url http://127.0.0.1:$port --model $model \
      --label ${arm}_$order --shape $shape --out $D/runs/${order}_${arm}_${shape}_$tag --concurrency $concurrency "$@" \
      >> $D/logs/client_${order}_${arm}.log 2>&1) &
    pids+=($!)
  done
  wait ${pids[@]}
  echo "$order $shape $tag: $(tail -n 1 $D/logs/client_${order}_main.log) / $(tail -n 1 $D/logs/client_${order}_first.log) $(date +%T)" >> $L
}
run_order() {
  local order=$1 card_main=$2 card_first=$3 port_main=$4 port_first=$5
  for size in 06 17; do
    local model config mem
    if [ $size = 06 ]; then
      model=Qwen/Qwen3-TTS-12Hz-0.6B-Base; config=examples/configs/qwen3_tts_0_6b.yaml; mem=0.08
    else
      model=Qwen/Qwen3-TTS-12Hz-1.7B-Base; config=examples/configs/qwen3_tts_1_7b.yaml; mem=0.12
    fi
    start_server ${order}_main_$size $MAIN $port_main --model-path $model --config $config --mem-fraction-static $mem \
      --tts_engine.gpu $card_main --vocoder.gpu $card_main || return 1
    start_server ${order}_first_$size $FIRST $port_first --model-path $model --config $config --mem-fraction-static $mem \
      --tts_engine.gpu $card_first --vocoder.gpu $card_first || return 1
    if [ $size = 06 ]; then
      serve_both $order $model icl06 1 c1 $port_main $port_first --sets "en sentence"
      serve_both $order $model icl06 16 interrupt $port_main $port_first --sets "en sentence" --interrupt
      serve_both $order $model icl06 16 c16 $port_main $port_first
    else
      serve_both $order $model xvec17 1 c1 $port_main $port_first --sets "en sentence"
      serve_both $order $model icl17zh 1 c1 $port_main $port_first --sets "en sentence"
      serve_both $order $model xvec17 16 interrupt $port_main $port_first --sets "en sentence" --interrupt
      serve_both $order $model xvec17 16 c16 $port_main $port_first
      serve_both $order $model icl17zh 16 c16 $port_main $port_first
    fi
    stop_server ${order}_main_$size
    stop_server ${order}_first_$size
  done
  echo "order $order done $(date +%T)" >> $L
}
run_order o1 0 1 4610 4620 &
a=$!
run_order o2 3 2 5110 5120 &
b=$!
wait $a $b
start_server asr $MAIN 5600 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 0 --mem-fraction-static 0.5 || exit 1
DIRS=$(ls -d $D/runs/o*_c16)
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py score $DIRS --asr-port 5600 --sim-card 1 >> $D/logs/score.log 2>&1)
echo "score rc $? $(tail -n 1 $D/logs/score.log) $(date +%T)" >> $L
stop_server asr
for run in icl06_c1 icl06_c16 icl06_interrupt xvec17_c1 xvec17_c16 xvec17_interrupt icl17zh_c1 icl17zh_c16; do
  for pair in "o1_main o1_first" "o2_main o2_first" "o1_main o2_main"; do
    set -- $pair
    echo "$D/runs/${1}_$run $D/runs/${2}_$run"
  done
done | xargs bash -c "cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py compare \"\$@\"" _ > $D/compare_duplex.txt 2>&1
echo "queue block3 done $(date +%T)" >> $L
