#!/usr/bin/env bash
# Block 25 of 10-29: bound block 24's 0.6B ICL length-stop lean on the talker priority branch (c16, every set: main 73
# and 69, the branch 76 and 93 in 3,259 a run). Seeded draws: every request of the three sets carries seed s (s = 1 to
# 4), so both arms draw from the same sampling stream and differ only through timing and batch composition. 0.6B Base
# at the profile's start (static 0.08), main on card 0, the branch (perf/qwen3-tts-talker-priority-stream 96c8120be,
# sglang_omni tree b74e6db8c) on card 1, c16; then main against main for seeds 1 and 2 as the seeded band. Qwen3-ASR on
# card 0 and WavLM on card 1 score every run.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/duplex_20261027
MAIN=/workspace/n29_main
BRANCH=/workspace/n29_prio3
CASES=$D/block19/duplex/cases.jsonl
M06=Qwen/Qwen3-TTS-12Hz-0.6B-Base
W=$D/block25
L=$W/queue.log
mkdir -p $W/logs $W/runs
if [ -n "${WAIT_PID:-}" ]; then
  while [ -n "$(ps -o stat= -p $WAIT_PID 2> /dev/null | grep -v Z)" ]; do sleep 30; done
else
  :
fi
echo "queue block25 start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')branch $(git -C $BRANCH rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L

start_server() {
  local label=$1 tree=$2 port=$3 card=$4
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port --model-path $M06 --config examples/configs/qwen3_tts_0_6b.yaml \
    --mem-fraction-static 0.08 --tts_engine.gpu $card --vocoder.gpu $card" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $W/logs/serve_$label.pid) waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then echo "$label not healthy" >> $L; kill -TERM -$pgid 2> /dev/null; return 1; else sleep 5; waited=$((waited + 5)); fi
  done
  echo "$label healthy after $waited s $(date +%T)" >> $L
}

stop_server() {
  local pgid=$(cat $W/logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

draw() {
  local first=$1 second=$2 seed=$3
  local pids=()
  for spec in "$first 4710" "$second 4720"; do
    set -- $spec
    (cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py serve $CASES --base-url http://127.0.0.1:$2 --model $M06 \
      --label ${1}_s$seed --shape icl06 --out $W/runs/${1}_s${seed}_c16 --concurrency 16 \
      --seed $seed >> $W/logs/client_$1.log 2>&1) &
    pids+=($!)
  done
  wait ${pids[@]}
  echo "seed $seed $first / $second: $(tail -n 1 $W/logs/client_$first.log) / $(tail -n 1 $W/logs/client_$second.log) $(date +%T)" >> $L
}

start_server main $MAIN 4710 0 &
a=$!
start_server branch $BRANCH 4720 1 &
b=$!
wait $a $b
for seed in 1 2 3 4; do
  draw main branch $seed
done
stop_server branch
start_server main2 $MAIN 4720 1 || exit 1
for seed in 1 2; do
  draw band1 band2 $seed
done
stop_server main
stop_server main2

(cd $MAIN && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$MAIN setsid bash -c "echo \$\$ > $W/logs/serve_asr.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port 5600 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 0 --mem-fraction-static 0.5" > $W/logs/serve_asr.log 2>&1 < /dev/null &)
waited=0
until curl -sf http://127.0.0.1:5600/health > /dev/null; do sleep 5; waited=$((waited + 5)); [ $waited -ge 900 ] && { echo "asr not healthy" >> $L; exit 1; }; done
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py score $W/runs/*_c16 --asr-port 5600 --sim-card 1 >> $W/logs/score.log 2>&1)
echo "score rc $? $(date +%T)" >> $L
stop_server asr
for pair in "main_s1 branch_s1" "main_s2 branch_s2" "main_s3 branch_s3" "main_s4 branch_s4" "band1_s1 band2_s1" "band1_s2 band2_s2"; do
  set -- $pair
  echo "$W/runs/${1}_c16 $W/runs/${2}_c16"
done | xargs -n 2 bash -c "cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py compare \"\$@\"" _ > $W/compare_seeded.txt 2>&1
echo "queue block25 done $(date +%T)" >> $L
