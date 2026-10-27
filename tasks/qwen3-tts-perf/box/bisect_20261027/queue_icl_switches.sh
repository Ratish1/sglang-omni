#!/usr/bin/env bash
# Which of our components makes the CI's ICL case lose its speaker more often than official at float32: the stack
# tree (its codec encoder in the checkpoint's float32) served once per configuration, the case drawn 300 times each as
# the SeedTTS CI client sends it, scored with the CI's WavLM similarity: default c16, default c1, the talker's radix
# cache off, the talker's CUDA graphs off, the predictor graph off, PyTorch sampling for the talker. Waits for
# AFTER_LOG to hold AFTER_LINE. Card 0.
set -u
D=${D:?the results directory}
TF=${TF:?the teacher-forced directory, for its cases}
STACK=${STACK:?the stack tree}
T=${T:?the kit tree}
AFTER_LOG=${AFTER_LOG:?the log to wait on}
AFTER_LINE=${AFTER_LINE:?the line to wait for}
HERE=$T/tasks/qwen3-tts-perf/box/bisect_20261027
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
IDS=common_voice_en_10119832-common_voice_en_10119840_icl_sentence
L=$D/queue.log
mkdir -p $D
until grep -q "$AFTER_LINE" $AFTER_LOG 2> /dev/null; do
  sleep 20
done
echo "queue icl switches start $(date +%T) stack $(git -C $STACK rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
port=31200
for config in "default_c16|16||" "default_c1|1||" "radix_off|16||--tts_engine.engine.disable_radix_cache true" \
  "graphs_off|16||--tts_engine.engine.disable_cuda_graph true" "predictor_graph_off|16|SGLANG_OMNI_QTTS_PREDICTOR_GRAPH=0|" \
  "sampling_pytorch|16||--tts_engine.engine.sampling_backend pytorch"; do
  IFS='|' read -r label concurrency env flags <<< "$config"
  echo "$label concurrency $concurrency env [$env] flags [$flags]" >> $L
  (cd $STACK && env $env SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$STACK setsid bash -c "echo \$\$ > $D/serve_$label.pid; \
    exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port --model-path $MODEL \
    --tts_engine.gpu 0 --vocoder.gpu 0 $flags" > $D/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  pgid=$(cat $D/serve_$label.pid)
  waited=0
  healthy=1
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label not healthy after $waited s; $(grep -E 'Error|error' $D/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      kill -TERM -$pgid 2> /dev/null
      healthy=0
      break
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  if [ $healthy = 1 ]; then
    echo "$label healthy after $waited s $(date +%T)" >> $L
    (cd $STACK && PYTHONPATH=$STACK python3 $HERE/speaker_loss.py serve $TF/cases.jsonl --ids $IDS \
      --base-url http://127.0.0.1:$port --model $MODEL --out $D/$label --draws 300 --concurrency $concurrency >> $L 2>&1)
    echo "$label served rc $? $(date +%T)" >> $L
    kill -TERM -$pgid 2> /dev/null
    for second in $(seq 120); do
      kill -0 $pgid 2> /dev/null || break
      sleep 1
    done
    kill -KILL -$pgid 2> /dev/null
  else
    :
  fi
  port=$((port + 20))
done
(cd $STACK && PYTHONPATH=$STACK python3 $HERE/speaker_loss.py score $D/default_c16 $D/default_c1 $D/radix_off $D/graphs_off \
  $D/predictor_graph_off $D/sampling_pytorch --card 0 > $D/summary.txt 2>&1)
echo "scored rc $? $(date +%T)" >> $L
echo "queue_icl_switches_done $(date +%T)" >> $L
