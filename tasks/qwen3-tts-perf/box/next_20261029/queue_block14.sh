#!/usr/bin/env bash
# Block 14 of 10-29, after block 13 (WAIT_PID): which half of D closes the CI's ICL speaker loss. The case
# common_voice_en_10119832-common_voice_en_10119840 (ICL sentence) drawn 300 times as the SeedTTS CI client sends it
# (c16, no seed, no max_new_tokens), scored with the CI's WavLM similarity, on main, exact (bf16 encoder), fp32 and D.
# 10-27: the stack before the exact-length encode 39 to 59 of 300 below SIM 50, with it 6, official float32 7.
# Card 0: main, then exact. Card 1: refenc, then fp32.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/bisect_20261027
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
IDS=common_voice_en_10119832-common_voice_en_10119840_icl_sentence
W=$D/block14
L=$W/queue.log
mkdir -p $W
if [ -n "${WAIT_PID:-}" ]; then
  while [ -n "$(ps -o stat= -p $WAIT_PID 2> /dev/null | grep -v Z)" ]; do sleep 30; done
else
  :
fi
echo "queue block14 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd /workspace/n29_main && PYTHONPATH=/workspace/n29_main python3 $T/tasks/qwen3-tts-perf/box/review_20261026/parity.py cases $W/parity >> $L 2>&1)
echo "parity cases $(wc -l < $W/parity/cases.jsonl) $(date +%T)" >> $L

draw() {
  local label=$1 card=$2 port=$3
  local tree=/workspace/n29_$label
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $W/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port --model-path $MODEL --tts_engine.gpu $card --vocoder.gpu $card" > $W/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $W/serve_$label.pid) waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then echo "$label not healthy" >> $L; kill -TERM -$pgid 2> /dev/null; return 1; else sleep 5; waited=$((waited + 5)); fi
  done
  echo "$label healthy after $waited s, tree $(git -C $tree rev-parse HEAD:sglang_omni) $(date +%T)" >> $L
  (cd $tree && PYTHONPATH=$tree python3 $HERE/speaker_loss.py serve $W/parity/cases.jsonl --ids $IDS \
    --base-url http://127.0.0.1:$port --model $MODEL --out $W/$label --draws 300 --concurrency 16 >> $L 2>&1)
  echo "$label served rc $? $(date +%T)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
  (cd $tree && PYTHONPATH=$tree python3 $HERE/speaker_loss.py score $W/$label --card $card > $W/summary_$label.txt 2>&1)
  echo "$label scored rc $? $(grep -i -E 'below|collapsed' $W/summary_$label.txt | head -2 | tr '\n' ' ') $(date +%T)" >> $L
}

(draw main 0 18000; draw exact 0 18000) &
zero=$!
(draw refenc 1 19000; draw fp32 1 19000) &
one=$!
wait $zero $one
echo "queue block14 done $(date +%T)" >> $L
