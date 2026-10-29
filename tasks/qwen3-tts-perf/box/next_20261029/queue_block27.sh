#!/usr/bin/env bash
# Block 27 of 10-29: bound block 25's pooled WER lean on the talker priority branch (paired WER points over seeds 1 to
# 4: last word +0.45 +- 0.64, last three +0.27 +- 0.36, sentence +0.23 +- 0.45; the band over seeds 1 and 2: +0.25,
# -0.05, -0.01) and its babble outputs in one powered run: seeds 5 to 12 main against the branch, then seeds 3 to 6
# main against main, 0.6B ICL every set at c16 as block 25 (main card 0, the branch card 1, static 0.08); scored on
# two lanes (Qwen3-ASR and WavLM on each card); then pooled_pairs.py over blocks 25 and 27 together.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/duplex_20261027
MAIN=/workspace/n29_main
BRANCH=/workspace/n29_prio3
CASES=$D/block19/duplex/cases.jsonl
M06=Qwen/Qwen3-TTS-12Hz-0.6B-Base
W=$D/block27
L=$W/queue.log
mkdir -p $W/logs $W/runs
echo "queue block27 start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')branch $(git -C $BRANCH rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L

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
  echo "--model-path $M06 --config examples/configs/qwen3_tts_0_6b.yaml --mem-fraction-static 0.08 --tts_engine.gpu $card --vocoder.gpu $card"
}

draw() {
  local first=$1 second=$2 seed=$3
  local pids=()
  for spec in "$first 4730" "$second 4740"; do
    set -- $spec
    (cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py serve $CASES --base-url http://127.0.0.1:$2 --model $M06 \
      --label ${1}_s$seed --shape icl06 --out $W/runs/${1}_s${seed}_c16 --concurrency 16 \
      --seed $seed >> $W/logs/client_$1.log 2>&1) &
    pids+=($!)
  done
  wait ${pids[@]}
  echo "seed $seed $first / $second: $(tail -n 1 $W/logs/client_$first.log) / $(tail -n 1 $W/logs/client_$second.log) $(date +%T)" >> $L
}

start_server main $MAIN 4730 $(tts 0) &
a=$!
start_server branch $BRANCH 4740 $(tts 1) &
b=$!
wait $a $b
for seed in 5 6 7 8 9 10 11 12; do
  draw main branch $seed
done
stop_server branch
start_server main2 $MAIN 4740 $(tts 1) || exit 1
for seed in 3 4 5 6; do
  draw band1 band2 $seed
done
stop_server main
stop_server main2

start_server asr0 $MAIN 5620 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 0 --mem-fraction-static 0.5 &
a=$!
start_server asr1 $MAIN 5640 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 1 --mem-fraction-static 0.5 &
b=$!
wait $a $b
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py score $W/runs/main_*_c16 $W/runs/band1_*_c16 --asr-port 5620 \
  --sim-card 0 >> $W/logs/score0.log 2>&1) &
a=$!
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/duplex_ab.py score $W/runs/branch_*_c16 $W/runs/band2_*_c16 --asr-port 5640 \
  --sim-card 1 >> $W/logs/score1.log 2>&1) &
b=$!
wait $a
ra=$?
wait $b
echo "score rc $ra $? $(date +%T)" >> $L
stop_server asr0
stop_server asr1

R25=$D/block25/runs
R27=$W/runs
branch_dirs=""
for seed in 1 2 3 4; do branch_dirs="$branch_dirs $R25/main_s${seed}_c16 $R25/branch_s${seed}_c16"; done
for seed in 5 6 7 8 9 10 11 12; do branch_dirs="$branch_dirs $R27/main_s${seed}_c16 $R27/branch_s${seed}_c16"; done
band_dirs=""
for seed in 1 2; do band_dirs="$band_dirs $R25/band1_s${seed}_c16 $R25/band2_s${seed}_c16"; done
for seed in 3 4 5 6; do band_dirs="$band_dirs $R27/band1_s${seed}_c16 $R27/band2_s${seed}_c16"; done
python3 $HERE/pooled_pairs.py branch_s1_12 $branch_dirs -- band_s1_6 $band_dirs > $W/pooled.txt 2>&1
echo "queue block27 done $(date +%T)" >> $L
