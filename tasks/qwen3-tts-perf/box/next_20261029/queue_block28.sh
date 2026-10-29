#!/usr/bin/env bash
# Block 28 of 10-29: the talker priority PR head (perf/qwen3-tts-talker-priority-stream 7990c2f65, a comment past
# block 23's 92b153edc) against main, the rest of the standard SeedTTS A/B and the babble. Part A, SeedTTS EN 1.7B Base
# with serve_arm.py at its barrier: seeded c1 non-streaming and streaming (seed 1234, main card 0, the branch card 1;
# byte identity at a fixed batch of one, the discriminator for a defect beyond batch composition, and the c1 speed),
# then c64 non-streaming and streaming in both card orders, scored. Part B, the babble rate where it is highest (1.7B
# x-vector, blocks 24 to 27: main 1 in 10,552 natural stops, the branch 3 in 10,551): duplex_ab.py every set at c16,
# unseeded as serving runs, 8 runs an arm with main on card 0 then 8 swapped, each server as the cascade profile starts
# it (static 0.12); scored on two lanes; babble_tally.py over the 16 runs an arm.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
REPO=/workspace/sglang-omni
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
HERE=$T/tasks/qwen3-tts-perf/box/duplex_20261027
MAIN=/workspace/n29_main
BRANCH=/workspace/n29_prio4
CASES=$D/block19/duplex/cases.jsonl
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block28
L=$W/queue.log
export ARM_ROOT=$W/arms
mkdir -p $ARM_ROOT $W/logs $W/barrier $W/runs
git -C $REPO fetch -q origin perf/qwen3-tts-talker-priority-stream
rm -rf $BRANCH
git -C $REPO worktree prune
git -C $REPO worktree add -q --detach $BRANCH 7990c2f65
echo "queue block28 start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')branch $(git -C $BRANCH rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L

arm() {
  local barrier=$1 label=$2 tree=$3 card=$4 port=$5
  shift 5
  python3 $ARM --label $label --tree $tree --card $card --port $port --barrier $W/barrier/$barrier "$@" >> $W/logs/arm_$label.log 2>&1
  echo "arm $label rc $? $(grep -h -E 'Completed requests|Failed requests|RTF mean|WER \(corpus|Max token hits|Throughput|TTFC mean|TTFC p95|Max underrun p95' $ARM_ROOT/$label/bench.log $ARM_ROOT/$label/score*.log 2> /dev/null | tr -s ' ' | sort -u | tr '\n' ';') $(grep -h -o -E 'speaker similarity: [0-9.]+' $ARM_ROOT/$label/score*.log 2> /dev/null | tail -1) $(grep -c -E 'device-side assert|CUDA error' $ARM_ROOT/$label/serve.log) $(date +%T)" >> $L
}

pair() {
  local name=$1
  arm $name $2 &
  local a=$!
  arm $name $3 &
  local b=$!
  wait $a $b
}

N1="--concurrency 1 --no-stream --seed 1234"
S1="--concurrency 1 --seed 1234"
N64="--concurrency 64 --no-stream --score"
S64="--concurrency 64 --score"
pair p_c1n "p_c1n_main $MAIN 0 31000 $N1" "p_c1n_prio $BRANCH 1 31500 $N1"
echo "c1 non-streaming: $(python3 $ARM --compare p_c1n_main p_c1n_prio) $(date +%T)" >> $L
pair p_c1s "p_c1s_main $MAIN 0 32000 $S1" "p_c1s_prio $BRANCH 1 32500 $S1"
echo "c1 streaming: $(python3 $ARM --compare p_c1s_main p_c1s_prio) $(date +%T)" >> $L
pair p_c64n_a "p_c64n_a_main $MAIN 0 33000 $N64" "p_c64n_a_prio $BRANCH 1 33500 $N64"
pair p_c64n_b "p_c64n_b_prio $BRANCH 0 34000 $N64" "p_c64n_b_main $MAIN 1 34500 $N64"
pair p_c64s_a "p_c64s_a_main $MAIN 0 35000 $S64" "p_c64s_a_prio $BRANCH 1 35500 $S64"
pair p_c64s_b "p_c64s_b_prio $BRANCH 0 36000 $S64" "p_c64s_b_main $MAIN 1 36500 $S64"
echo "part A done $(date +%T)" >> $L

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
  echo "$label healthy after $waited s $(date +%T)" >> $L
}

stop_server() {
  local pgid=$(cat $W/logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

tts() {
  local card=$1
  echo "--model-path $M17 --config examples/configs/qwen3_tts_1_7b.yaml --mem-fraction-static 0.12 --tts_engine.gpu $card --vocoder.gpu $card"
}

draw() {
  local order=$1 run=$2 port_main=$3 port_branch=$4
  local pids=()
  for spec in "main $port_main" "branch $port_branch"; do
    set -- $spec
    (cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py serve $CASES --base-url http://127.0.0.1:$2 --model $M17 \
      --label ${1}_${order}_r$run --shape xvec17 --out $W/runs/${order}_${1}_r${run}_c16 --concurrency 16 \
      >> $W/logs/client_$1.log 2>&1) &
    pids+=($!)
  done
  wait ${pids[@]}
  echo "$order run $run: $(tail -n 1 $W/logs/client_main.log) / $(tail -n 1 $W/logs/client_branch.log) $(date +%T)" >> $L
}

for spec in "o1 0 1 4750 4760" "o2 1 0 5150 5160"; do
  set -- $spec
  order=$1 card_main=$2 card_branch=$3 port_main=$4 port_branch=$5
  start_server ${order}_main $MAIN $port_main $(tts $card_main) &
  a=$!
  start_server ${order}_branch $BRANCH $port_branch $(tts $card_branch) &
  b=$!
  wait $a $b
  for run in 1 2 3 4 5 6 7 8; do
    draw $order $run $port_main $port_branch
  done
  stop_server ${order}_main
  stop_server ${order}_branch
done

start_server asr0 $MAIN 5660 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 0 --mem-fraction-static 0.5 &
a=$!
start_server asr1 $MAIN 5680 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 1 --mem-fraction-static 0.5 &
b=$!
wait $a $b
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py score $W/runs/*_main_*_c16 --asr-port 5660 --sim-card 0 \
  >> $W/logs/score0.log 2>&1) &
a=$!
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py score $W/runs/*_branch_*_c16 --asr-port 5680 --sim-card 1 \
  >> $W/logs/score1.log 2>&1) &
b=$!
wait $a
ra=$?
wait $b
echo "score rc $ra $? $(date +%T)" >> $L
stop_server asr0
stop_server asr1
python3 $HERE/babble_tally.py main=$(ls -d $W/runs/*_main_*_c16 | paste -sd, -) branch=$(ls -d $W/runs/*_branch_*_c16 | paste -sd, -) \
  > $W/babble.txt 2>&1
echo "babble rc $? $(grep 'all sets' $W/babble.txt | tr '\n' ' ') $(date +%T)" >> $L
echo "queue block28 done $(date +%T)" >> $L
