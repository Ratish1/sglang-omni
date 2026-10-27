#!/usr/bin/env bash
# The reference codec encoder in its checkpoint dtype (float32) against bf16, on 1.7B Base: after
# queue_identity_3176.sh, the three speaker-loss cases drawn 100 times each on `probe/qwen3-tts-fp32-reference-encoder`
# with QWEN3_TTS_PROBE_ENCODER_FP32=1 and on official qwen-tts with --encoder-fp32, scored like queue_speaker_loss.sh;
# then SeedTTS EN non-streaming c16, unseeded, main and the fp32 probe as main, fp32, fp32, main, each scored. Card 0.
set -u
D=${D:?the results directory}
AFTER=${AFTER:?the identity directory, for its done line}
TF=${TF:?the teacher-forced directory, for its cases}
MAIN=${MAIN:?the main tree}
P=${P:?the fp32 probe tree}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/bisect_20261027
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
PORT=28800
IDS="common_voice_en_10119832-common_voice_en_10119840_icl_sentence common_voice_en_10119832-common_voice_en_10119847_icl_sentence common_voice_en_15735839-common_voice_en_15735837_icl_sentence"
L=$D/queue.log
mkdir -p $D
until grep -q queue_identity_3176_done $AFTER/queue.log 2> /dev/null; do
  sleep 20
done
echo "queue fp32 encoder start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')probe $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $P && QWEN3_TTS_PROBE_ENCODER_FP32=1 SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$P setsid bash -c "echo \$\$ > $D/serve.pid; \
  exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT --model-path $MODEL \
  --tts_engine.gpu 0 --vocoder.gpu 0" > $D/serve.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $D/serve.pid)
WAITED=0
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $WAITED -ge 1800 ]; then
    echo "server not healthy after $WAITED s; $(grep -E 'Error|error' $D/serve.log | tail -n 2 | tr '\n' ' ')" >> $L
    kill -TERM -$PGID 2> /dev/null
    echo "queue_fp32_encoder_done $(date +%T)" >> $L
    exit 1
  else
    sleep 5
    WAITED=$((WAITED + 5))
  fi
done
echo "healthy after $WAITED s, pgid $PGID $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/speaker_loss.py serve $TF/cases.jsonl --ids $IDS --base-url http://127.0.0.1:$PORT \
  --model $MODEL --out $D/ours_fp32 --draws 100 --concurrency 16 >> $L 2>&1)
echo "served rc $? $(date +%T)" >> $L
kill -TERM -$PGID 2> /dev/null
for second in $(seq 120); do
  kill -0 $PGID 2> /dev/null || break
  sleep 1
done
kill -KILL -$PGID 2> /dev/null
echo "stopped $(date +%T)" >> $L
/data/refenv/bin/python $HERE/speaker_loss_reference.py $TF/cases.jsonl --ids $IDS --model $MODEL --out $D/official_fp32 \
  --draws 100 --batch 16 --card 0 --encoder-fp32 >> $L 2>&1
echo "official rc $? $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/speaker_loss.py score $D/ours_fp32 $D/official_fp32 --card 0 > $D/speaker_summary.txt 2>&1)
echo "scored rc $? $(date +%T)" >> $L
port=29000
for spec in "c16n_main1 $MAIN 0" "c16n_fp32_1 $P 1" "c16n_fp32_2 $P 1" "c16n_main2 $MAIN 0"; do
  set -- $spec
  if [ $3 = 1 ]; then
    QWEN3_TTS_PROBE_ENCODER_FP32=1 ARM_ROOT=$D/arms python3 $ARM --label $1 --tree $2 --card 0 --port $port \
      --concurrency 16 --no-stream --score >> $L 2>&1
  else
    ARM_ROOT=$D/arms python3 $ARM --label $1 --tree $2 --card 0 --port $port --concurrency 16 --no-stream --score >> $L 2>&1
  fi
  echo "arm $1 rc $? $(date +%T)" >> $L
  port=$((port + 20))
done
echo "queue_fp32_encoder_done $(date +%T)" >> $L
