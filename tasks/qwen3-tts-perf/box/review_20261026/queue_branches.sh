#!/usr/bin/env bash
# The three fix branches against their base (upstream main 65868af05), default server settings, one arm at a time on
# card 1: the two-frame minimum (A), per-task frame ceilings (C), the x-vector leading-silence mask removed (B1).
# 1. 0.6B on the cascade's requests (every parity case, language English, streaming PCM, no max_new_tokens), each arm,
#    then transcribed: empty responses, runaways and where they stop, natural stops cut by the default cap, WER.
# 2. 1.7B SeedTTS EN with references (ICL), non-streaming c16, main, A, C, B1, main; WER, similarity, RTF.
# 3. SeedTTS EN without reference text (x-vector), non-streaming c16, 1.7B and 0.6B, main and B1.
# 4. 1.7B SeedTTS EN ICL, non-streaming c1, main, A, C.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
M=${M:?the main tree}
A=${A:?the two-frame minimum tree}
C=${C:?the frame ceilings tree}
B1=${B1:?the mask removal tree}
CASES=${CASES:?the parity cases}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
export ARM_ROOT=$D/arms
CARD=1
L=$D/queue.log
mkdir -p $D/served $ARM_ROOT
until grep -q queue_layout_score2_done /data/layout_20261026/queue.log; do
  sleep 30
done
echo "queue branches start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')A $(git -C $A rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')C $(git -C $C rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')B1 $(git -C $B1 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
start_server() {
  local label=$1 tree=$2 port=$3
  shift 3
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $D/serve_$label.pid; exec python3 \
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
for arm in "main $M" "A $A" "C $C" "B1 $B1"; do
  set -- $arm
  label=06_$1
  if start_server $label $2 23000 --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --tts_engine.gpu $CARD --vocoder.gpu $CARD; then
    (cd $2 && PYTHONPATH=$2 python3 $HERE/layout.py serve $CASES --base-url http://127.0.0.1:23000 \
      --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --label $label --language English --out $D/served/$label \
      --max-new-tokens 0 >> $L 2>&1)
    echo "served $label rc $? $(date +%T)" >> $L
    stop_server $label
  else
    :
  fi
done
export MASTER_PORT=23850
if start_server asr $M 23800 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu $CARD; then
  for label in 06_main 06_A 06_C 06_B1; do
    (cd $M && PYTHONPATH=$M python3 $HERE/layout.py score $D/served/$label --asr-port 23800 >> $L 2>&1)
    echo "transcribed $label rc $? $(date +%T)" >> $L
  done
  stop_server asr
else
  :
fi
unset MASTER_PORT
run_arm() {
  local label=$1 tree=$2
  shift 2
  python3 $ARM --label $label --tree $tree --card $CARD --port 21000 --no-stream --score "$@" >> $L 2>&1
  echo "arm $label rc $? $(date +%T)" >> $L
}
run_arm icl_c16_main1 $M --concurrency 16
run_arm icl_c16_A $A --concurrency 16
run_arm icl_c16_C $C --concurrency 16
run_arm icl_c16_B1 $B1 --concurrency 16
run_arm icl_c16_main2 $M --concurrency 16
run_arm xvec17_c16_main $M --concurrency 16 --bench-args "--no-ref-text"
run_arm xvec17_c16_B1 $B1 --concurrency 16 --bench-args "--no-ref-text"
run_arm xvec06_c16_main $M --concurrency 16 --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --bench-args "--no-ref-text"
run_arm xvec06_c16_B1 $B1 --concurrency 16 --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --bench-args "--no-ref-text"
run_arm icl_c1_main $M --concurrency 1
run_arm icl_c1_A $A --concurrency 1
run_arm icl_c1_C $C --concurrency 1
echo "queue_branches_done $(date +%T)" >> $L
