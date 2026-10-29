#!/usr/bin/env bash
# Block 13 of 10-29, after block 12 (WAIT_PID): the CustomVoice and VoiceDesign limit branch against main.
# 1. The standard SeedTTS EN A/B on Base (the branch cannot change a Base output): non-streaming c16 and streaming c16,
#    both card orders, scored on each card.
# 2. Where it acts: every Chinese CustomVoice case (every speaker plain, ryan and vivian slow) and every Chinese
#    VoiceDesign case (plain, slow), seeded, non-streaming, c32, no max_new_tokens; main on card 0 and the branch on
#    card 1 at once, one round (an outputs check: block 7 measured main's cuts, and the speed pairs are part 1);
#    every length stop and every natural stop past #3157's budget kept.
# 3. One Qwen3-ASR server on card 0 (64 long transcriptions at once) scores the kept outputs; compare per task.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box/next_20261029
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
MAIN=/workspace/n29_main
CV=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
VD=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
W=$D/block13
L=$W/queue.log
export ARM_ROOT=$W/arms
mkdir -p $ARM_ROOT $W/logs $W/scan
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 $WAIT_PID 2> /dev/null; do sleep 30; done
else
  :
fi
echo "queue block13 start $(date +%T) kit $(git -C $T rev-parse HEAD) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')limit $(git -C /workspace/n29_limit rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
(cd $MAIN && PYTHONPATH=$MAIN python3 $K/limit_ab.py cases $W/cases --custom-voice-model $CV >> $L 2>&1)

arm() {
  local label=$1 tree=$2 card=$3 port=$4
  shift 4
  python3 $ARM --label $label --tree /workspace/n29_$tree --card $card --port $port "$@" >> $W/logs/arm_$label.log 2>&1
  echo "arm $label rc $? $(grep -h -E 'RTF mean|WER \(corpus|Max token hits|Throughput' $ARM_ROOT/$label/bench.log $ARM_ROOT/$label/score*.log 2> /dev/null | tr -s ' ' | sort -u | tr '\n' ';') $(grep -h -o -E 'speaker similarity: [0-9.]+' $ARM_ROOT/$label/score*.log 2> /dev/null | tail -1) $(date +%T)" >> $L
}

pair() {
  arm $1 &
  local a=$!
  arm $2 &
  local b=$!
  wait $a $b
}

pair "c16n_a_main main 0 11000 --concurrency 16 --no-stream --score" "c16n_a_limit limit 1 11500 --concurrency 16 --no-stream --score"
pair "c16n_b_limit limit 0 12000 --concurrency 16 --no-stream --score" "c16n_b_main main 1 12500 --concurrency 16 --no-stream --score"
pair "c16s_a_main main 0 13000 --concurrency 16 --score" "c16s_a_limit limit 1 13500 --concurrency 16 --score"
pair "c16s_b_limit limit 0 14000 --concurrency 16 --score" "c16s_b_main main 1 14500 --concurrency 16 --score"
echo "seedtts pairs done $(date +%T)" >> $L

start_server() {
  local label=$1 tree=$2 port=$3
  shift 3
  (cd /workspace/n29_$tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=/workspace/n29_$tree setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $W/logs/serve_$label.pid) waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then echo "$label not healthy" >> $L; return 1; else sleep 5; waited=$((waited + 5)); fi
  done
  echo "$label healthy after $waited s $(date +%T)" >> $L
}

stop_server() {
  local pgid=$(cat $W/logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

scan_lane() {
  local tree=$1 card=$2 port=$3 round=$4
  for task in customvoice voicedesign; do
    if [ $task = customvoice ]; then model=$CV; else model=$VD; fi
    local label=${task}_${round}_$tree
    start_server $label $tree $port --model-path $model --tts_engine.gpu $card --vocoder.gpu $card || continue
    (cd /workspace/n29_$tree && PYTHONPATH=/workspace/n29_$tree python3 $K/limit_ab.py scan $W/cases/${task}_zh.jsonl \
      --base-url http://127.0.0.1:$port --model $model --label $label --out $W/scan/$label.jsonl --concurrency 32 >> $L 2>&1)
    echo "scan $label rc $? $(date +%T)" >> $L
    stop_server $label
    port=$((port + 2000))
  done
}

for round in a; do
  if [ $round = a ]; then main_card=0 limit_card=1; else main_card=1 limit_card=0; fi
  scan_lane main $main_card 15000 $round &
  first=$!
  scan_lane limit $limit_card 15500 $round &
  second=$!
  wait $first $second
done

if start_server asr main 24300 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 0 --audio_chunking.max_concurrent_long_audio_requests 64; then
  for result in $W/scan/*.jsonl; do
    case $result in
      *.scored.jsonl) continue ;;
      *) ;;
    esac
    (cd $MAIN && PYTHONPATH=$MAIN python3 $K/limit_ab.py score $result --asr-port 24300 --concurrency 64 >> $L 2>&1)
  done
  stop_server asr
  for round in a; do
    for task in customvoice voicedesign; do
      (cd $MAIN && PYTHONPATH=$MAIN python3 $K/limit_ab.py compare $W/scan/${task}_${round}_main.scored.jsonl \
        $W/scan/${task}_${round}_limit.scored.jsonl > $W/compare_${task}_${round}.txt 2>&1)
      echo "compare $task $round rc $? $(date +%T)" >> $L
    done
  done
else
  echo "asr not started, nothing scored" >> $L
fi
echo "queue block13 done $(date +%T)" >> $L
