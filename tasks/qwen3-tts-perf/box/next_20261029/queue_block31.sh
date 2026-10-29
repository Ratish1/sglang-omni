#!/usr/bin/env bash
# Block 31 of 10-29: D (19f0f8de5) against today's main (08cecee99, A merged) on the cascade shapes block 29 did not
# run, for D's PR body on its landing base: 0.6B ICL c1 sentences (first audio with a new reference encoded in
# float32); 1.7B x-vector c1 sentences and c16 every set; 1.7B Chinese clone c1 sentences and c16. block 24's
# duplex_ab.py runs, unseeded, each server as the cascade profiles start it (0.6B static 0.08, 1.7B 0.12), order o1
# (main card 0) then o2 swapped; Qwen3-ASR and WavLM score the c16 runs; compare per order and main against main.
# usage: WAIT_PID=PID D=... T=... bash queue_block31.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/duplex_20261027
MAIN=/workspace/n29_main
REFENC=/workspace/n29_refenc
W=$D/block31
CASES=$D/block19/duplex/cases.jsonl
L=$W/queue.log
mkdir -p $W/logs $W/runs
if [ -n "${WAIT_PID:-}" ]; then
  while [ -n "$(ps -o stat= -p $WAIT_PID 2> /dev/null | grep -v Z)" ]; do sleep 30; done
else
  :
fi
echo "queue block31 start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')D $(git -C $REFENC rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L

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

serve_both() {
  local order=$1 model=$2 shape=$3 concurrency=$4 tag=$5 port_main=$6 port_d=$7
  shift 7
  local pids=()
  for arm in main refenc; do
    local port=$port_main
    if [ $arm = refenc ]; then port=$port_d; else :; fi
    (cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py serve $CASES --base-url http://127.0.0.1:$port --model $model \
      --label ${arm}_$order --shape $shape --out $W/runs/${order}_${arm}_${shape}_$tag --concurrency $concurrency "$@" \
      >> $W/logs/client_${order}_${arm}.log 2>&1) &
    pids+=($!)
  done
  wait ${pids[@]}
  echo "$order $shape $tag: $(tail -n 1 $W/logs/client_${order}_main.log) / $(tail -n 1 $W/logs/client_${order}_refenc.log) $(date +%T)" >> $L
}

run_order() {
  local order=$1 card_main=$2 card_d=$3 port_main=$4 port_d=$5
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
    start_server ${order}_refenc_$size $REFENC $port_d --model-path $model --config $config --mem-fraction-static $mem \
      --tts_engine.gpu $card_d --vocoder.gpu $card_d &
    local b=$!
    wait $a $b
    if [ $size = 06 ]; then
      serve_both $order $model icl06 1 c1 $port_main $port_d --sets "en sentence"
    else
      serve_both $order $model xvec17 1 c1 $port_main $port_d --sets "en sentence"
      serve_both $order $model icl17zh 1 c1 $port_main $port_d --sets "en sentence"
      serve_both $order $model xvec17 16 c16 $port_main $port_d
      serve_both $order $model icl17zh 16 c16 $port_main $port_d
    fi
    stop_server ${order}_main_$size
    stop_server ${order}_refenc_$size
  done
  echo "order $order done $(date +%T)" >> $L
}

run_order o1 0 1 4790 4795
run_order o2 1 0 5190 5195
start_server asr0 $MAIN 5740 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 0 --mem-fraction-static 0.5 &
a=$!
start_server asr1 $MAIN 5760 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 1 --mem-fraction-static 0.5 &
b=$!
wait $a $b
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py score $W/runs/o*_main_*_c16 --asr-port 5740 --sim-card 0 \
  >> $W/logs/score0.log 2>&1) &
a=$!
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py score $W/runs/o*_refenc_*_c16 --asr-port 5760 --sim-card 1 \
  >> $W/logs/score1.log 2>&1) &
b=$!
wait $a
ra=$?
wait $b
echo "score rc $ra $? $(date +%T)" >> $L
stop_server asr0
stop_server asr1
for run in icl06_c1 xvec17_c1 icl17zh_c1 xvec17_c16 icl17zh_c16; do
  for pair in "o1_main o1_refenc" "o2_main o2_refenc" "o1_main o2_main"; do
    set -- $pair
    if [ -d $W/runs/${1}_$run ] && [ -d $W/runs/${2}_$run ]; then echo "$W/runs/${1}_$run $W/runs/${2}_$run"; else :; fi
  done
done | xargs -n 2 bash -c "cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py compare \"\$@\"" _ > $W/compare_duplex.txt 2>&1
echo "queue block31 done $(date +%T)" >> $L
