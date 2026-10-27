#!/usr/bin/env bash
# The ICL difference left at float32: after queue_stack_ab.sh, the CI's collapsing case drawn 300 times on the stack
# (its codec encoder loads in the checkpoint's float32) and 300 times on official qwen-tts with --encoder-fp32, both
# scored with the CI's WavLM similarity; the stack's codes logged by nothing (a plain tree), so this bounds the rate
# only. Card 0.
set -u
D=${D:?the results directory}
AFTER=${AFTER:?the stack A/B directory, for its done line}
TF=${TF:?the teacher-forced directory, for its cases}
STACK=${STACK:?the stack tree}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/bisect_20261027
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
PORT=31000
IDS=common_voice_en_10119832-common_voice_en_10119840_icl_sentence
L=$D/queue.log
mkdir -p $D
until grep -q queue_stack_ab_done $AFTER/queue.log 2> /dev/null; do
  sleep 20
done
echo "queue icl residual start $(date +%T) stack $(git -C $STACK rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $STACK && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$STACK setsid bash -c "echo \$\$ > $D/serve.pid; \
  exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT --model-path $MODEL \
  --tts_engine.gpu 0 --vocoder.gpu 0" > $D/serve.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $D/serve.pid)
WAITED=0
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $WAITED -ge 1800 ]; then
    echo "server not healthy after $WAITED s; $(grep -E 'Error|error' $D/serve.log | tail -n 2 | tr '\n' ' ')" >> $L
    kill -TERM -$PGID 2> /dev/null
    echo "queue_icl_residual_done $(date +%T)" >> $L
    exit 1
  else
    sleep 5
    WAITED=$((WAITED + 5))
  fi
done
echo "healthy after $WAITED s, pgid $PGID $(date +%T)" >> $L
(cd $STACK && PYTHONPATH=$STACK python3 $HERE/speaker_loss.py serve $TF/cases.jsonl --ids $IDS \
  --base-url http://127.0.0.1:$PORT --model $MODEL --out $D/stack --draws 300 --concurrency 16 >> $L 2>&1)
echo "served rc $? $(date +%T)" >> $L
kill -TERM -$PGID 2> /dev/null
for second in $(seq 120); do
  kill -0 $PGID 2> /dev/null || break
  sleep 1
done
kill -KILL -$PGID 2> /dev/null
echo "stopped $(date +%T)" >> $L
/data/refenv/bin/python $HERE/speaker_loss_reference.py $TF/cases.jsonl --ids $IDS --model $MODEL --out $D/official_fp32 \
  --draws 300 --batch 16 --card 0 --encoder-fp32 --seed 1000 >> $L 2>&1
echo "official rc $? $(date +%T)" >> $L
(cd $STACK && PYTHONPATH=$STACK python3 $HERE/speaker_loss.py score $D/stack $D/official_fp32 --card 0 > $D/summary.txt 2>&1)
echo "scored rc $? $(date +%T)" >> $L
echo "queue_icl_residual_done $(date +%T)" >> $L
