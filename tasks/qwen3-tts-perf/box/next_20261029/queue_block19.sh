#!/usr/bin/env bash
# Block 19 of 10-29, after block 18 (WAIT_PID): the full-duplex A/B (block 3 of 10-28, duplex_20261027/duplex_ab.py) for
# the speedups branch, on the shapes its codes can move. The compiled encode changes main's bf16 reference codes (41 %
# differ, slightly closer to official's float32), so ICL clones can change; x-vector clones carry no reference codes, and
# the silence probe encodes eagerly through the integer padding, which is exact. Each server as the cascade profiles
# start it (0.6B static fraction 0.08, 1.7B 0.12). Order o1: main on card 0, the branch on card 1; then o2 swapped.
# Per order: 0.6B ICL c1 sentences and c16 every set, 1.7B Chinese clone c1 sentences and c16. Then Qwen3-ASR on card 0
# and WavLM on card 1 score every c16 run; compare per order and main against main.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/duplex_20261027
MAIN=/workspace/n29_main
BRANCH=/workspace/n29_compile
W=$D/block19
CASES=$W/duplex/cases.jsonl
L=$W/queue.log
mkdir -p $W/logs $W/runs
if [ -n "${WAIT_PID:-}" ]; then
  while [ -n "$(ps -o stat= -p $WAIT_PID 2> /dev/null | grep -v Z)" ]; do sleep 30; done
else
  :
fi
echo "queue block19 start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')branch $(git -C $BRANCH rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py cases $W/duplex --profiles examples/streaming_cascade/profiles >> $W/logs/cases.log 2>&1)
echo "cases $(wc -l < $CASES) $(date +%T)" >> $L

start_server() {
  local label=$1 tree=$2 port=$3
  shift 3
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $W/logs/serve_$label.pid) waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label server not healthy after $waited s; $(grep -E 'Error|error' $W/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      kill -TERM -$pgid 2> /dev/null
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s, $(grep -o 'KV pool holds [0-9]* tokens, [0-9.]* GiB' $W/logs/serve_$label.log | head -n 1) $(date +%T)" >> $L
}

stop_server() {
  local pgid=$(cat $W/logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

serve_both() {
  local order=$1 model=$2 shape=$3 concurrency=$4 tag=$5 port_main=$6 port_branch=$7
  shift 7
  local pids=()
  for arm in main branch; do
    local port=$port_main
    if [ $arm = branch ]; then port=$port_branch; else :; fi
    (cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py serve $CASES --base-url http://127.0.0.1:$port --model $model \
      --label ${arm}_$order --shape $shape --out $W/runs/${order}_${arm}_${shape}_$tag --concurrency $concurrency "$@" \
      >> $W/logs/client_${order}_${arm}.log 2>&1) &
    pids+=($!)
  done
  wait ${pids[@]}
  echo "$order $shape $tag: $(tail -n 1 $W/logs/client_${order}_main.log) / $(tail -n 1 $W/logs/client_${order}_branch.log) $(date +%T)" >> $L
}

run_order() {
  local order=$1 card_main=$2 card_branch=$3 port_main=$4 port_branch=$5
  for size in 06 17; do
    local model config mem
    if [ $size = 06 ]; then
      model=Qwen/Qwen3-TTS-12Hz-0.6B-Base; config=examples/configs/qwen3_tts_0_6b.yaml; mem=0.08
    else
      model=Qwen/Qwen3-TTS-12Hz-1.7B-Base; config=examples/configs/qwen3_tts_1_7b.yaml; mem=0.12
    fi
    start_server ${order}_main_$size $MAIN $port_main --model-path $model --config $config --mem-fraction-static $mem \
      --tts_engine.gpu $card_main --vocoder.gpu $card_main &
    local a=$!
    start_server ${order}_branch_$size $BRANCH $port_branch --model-path $model --config $config --mem-fraction-static $mem \
      --tts_engine.gpu $card_branch --vocoder.gpu $card_branch &
    local b=$!
    wait $a $b
    if [ $size = 06 ]; then
      serve_both $order $model icl06 1 c1 $port_main $port_branch --sets "en sentence"
      serve_both $order $model icl06 16 c16 $port_main $port_branch
    else
      serve_both $order $model icl17zh 1 c1 $port_main $port_branch --sets "en sentence"
      serve_both $order $model icl17zh 16 c16 $port_main $port_branch
    fi
    stop_server ${order}_main_$size
    stop_server ${order}_branch_$size
  done
  echo "order $order done $(date +%T)" >> $L
}

run_order o1 0 1 4610 4620
run_order o2 1 0 5110 5120
start_server asr $MAIN 5600 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 0 --mem-fraction-static 0.5 || exit 1
DIRS=$(ls -d $W/runs/o*_c16)
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py score $DIRS --asr-port 5600 --sim-card 1 >> $W/logs/score.log 2>&1)
echo "score rc $? $(tail -n 1 $W/logs/score.log) $(date +%T)" >> $L
stop_server asr
for run in icl06_c1 icl06_c16 icl17zh_c1 icl17zh_c16; do
  for pair in "o1_main o1_branch" "o2_main o2_branch" "o1_main o2_main"; do
    set -- $pair
    echo "$W/runs/${1}_$run $W/runs/${2}_$run"
  done
done | xargs -n 2 bash -c "cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py compare \"\$@\"" _ > $W/compare_duplex.txt 2>&1
echo "queue block19 done $(date +%T)" >> $L
