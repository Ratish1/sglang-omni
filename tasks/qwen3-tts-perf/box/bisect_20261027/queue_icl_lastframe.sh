#!/usr/bin/env bash
# Whether our reference codes explain the ICL speaker loss: the CI's collapsing case's reference encoded in the serving
# image by the stack tree (float32 encoder) three ways (ref_codes_case.py: our graphs as served, our eager encode, the
# official tokenizer's encode), then official qwen-tts with --encoder-fp32 drawing the case 300 times each with its own
# codes, with our served codes in place, and with only our last frame in place, all scored with the CI's WavLM
# similarity. Waits for AFTER_LOG to hold AFTER_LINE. Card 0.
set -u
D=${D:?the results directory}
TF=${TF:?the teacher-forced directory, for its cases}
STACK=${STACK:?the stack tree}
T=${T:?the kit tree}
AFTER_LOG=${AFTER_LOG:?the log to wait on}
AFTER_LINE=${AFTER_LINE:?the line to wait for}
HERE=$T/tasks/qwen3-tts-perf/box/bisect_20261027
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ID=common_voice_en_10119832-common_voice_en_10119840_icl_sentence
L=$D/queue.log
mkdir -p $D
until grep -q "$AFTER_LINE" $AFTER_LOG 2> /dev/null; do
  sleep 20
done
echo "queue icl lastframe start $(date +%T) stack $(git -C $STACK rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
REF=$(python3 -c "import json, sys; print(next(c['ref_audio'] for c in map(json.loads, open(sys.argv[1])) if c['id'] == sys.argv[2]))" $TF/cases.jsonl $ID)
echo "reference $REF" >> $L
(cd $STACK && PYTHONPATH=$STACK python3 $HERE/ref_codes_case.py $REF --model $MODEL --out $D/codes.pt --card 0 2>&1 \
  | grep -v -E "Warning|warn" >> $L)
echo "codes rc $? $(date +%T)" >> $L
for arm in "official|" "ours_codes|--ref-codes $D/codes.pt" "ours_last_frame|--ref-codes $D/codes.pt --last-frame-only"; do
  IFS='|' read -r label flags <<< "$arm"
  /data/refenv/bin/python $HERE/speaker_loss_reference.py $TF/cases.jsonl --ids $ID --model $MODEL --out $D/$label \
    --draws 300 --batch 16 --card 0 --encoder-fp32 --seed 1000 $flags 2>&1 | grep -v -E "draws [0-9]+-|Warning|warn" >> $L
  echo "$label rc $? $(date +%T)" >> $L
done
(cd $STACK && PYTHONPATH=$STACK python3 $HERE/speaker_loss.py score $D/official $D/ours_codes $D/ours_last_frame \
  --card 0 > $D/summary.txt 2>&1)
echo "scored rc $? $(date +%T)" >> $L
echo "queue_icl_lastframe_done $(date +%T)" >> $L
