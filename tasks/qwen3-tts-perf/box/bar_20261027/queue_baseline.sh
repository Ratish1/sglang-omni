#!/usr/bin/env bash
# Our server against Pipeshift's bar with Nari Labs' client unmodified (`nari-labs/benchmarks` `tts_bench` at a pinned
# commit): one tree, 1.7B CustomVoice at default settings, Ryan, English, Poisson arrivals at each rate in RATES, 30 s
# warm-up and 120 s measured, one seed per call; then bar_summary.py over the rates. Waits for AFTER's done line. Card 0.
set -u
D=${D:?the results directory}
TREE=${TREE:?the server tree}
T=${T:?the kit tree}
LABEL=${LABEL:?the arm label}
SEED=${SEED:-0}
RATES=${RATES:-"1 4 8 12 20 30 40"}
AFTER_LOG=${AFTER_LOG:-}
AFTER_LINE=${AFTER_LINE:-}
CLIENT_COMMIT=a2d65e1
PORT=${PORT:-30000}
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
HERE=$T/tasks/qwen3-tts-perf/box/bar_20261027
L=$D/queue.log
mkdir -p $D/$LABEL
if [ -n "$AFTER_LOG" ]; then
  until grep -q "$AFTER_LINE" $AFTER_LOG 2> /dev/null; do
    sleep 20
  done
else
  :
fi
if [ ! -f /data/tts_bench_src/tts_bench/datasets/seed-tts-eval.jsonl ]; then
  git clone -q https://github.com/nari-labs/benchmarks /data/tts_bench_src && git -C /data/tts_bench_src checkout -q $CLIENT_COMMIT
  (cd /data/tts_bench_src/tts_bench && uv sync -q --all-groups && uv run -q bench dataset prepare) >> $L 2>&1
else
  :
fi
echo "queue baseline $LABEL seed $SEED start $(date +%T) tree $(git -C $TREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')client $(git -C /data/tts_bench_src rev-parse --short HEAD) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $TREE && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$TREE setsid bash -c "echo \$\$ > $D/$LABEL/serve.pid; exec python3 \
  -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT --model-path $MODEL --tts_engine.gpu 0 --vocoder.gpu 0" \
  > $D/$LABEL/serve.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $D/$LABEL/serve.pid)
WAITED=0
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $WAITED -ge 1800 ]; then
    echo "$LABEL server not healthy after $WAITED s; $(grep -E 'Error|error' $D/$LABEL/serve.log | tail -n 2 | tr '\n' ' ')" >> $L
    kill -TERM -$PGID 2> /dev/null
    echo "queue_baseline_${LABEL}_done $(date +%T)" >> $L
    exit 1
  else
    sleep 5
    WAITED=$((WAITED + 5))
  fi
done
echo "$LABEL healthy after $WAITED s, pgid $PGID $(date +%T)" >> $L
for rate in $RATES; do
  (cd /data/tts_bench_src/tts_bench && uv run -q bench run --target sglang-omni --base-url http://127.0.0.1:$PORT \
    --model $MODEL --voice Ryan --language English --dataset datasets/seed-tts-eval.jsonl --rps $rate \
    --warmup 30s --duration 120s --arrival poisson --seed $SEED --timeout 120s --max-in-flight 4096 \
    --output $D/$LABEL/seed${SEED}_rps$rate >> $D/$LABEL/bench.log 2>&1)
  echo "$LABEL rps $rate rc $? $(date +%T)" >> $L
  sleep 30
done
kill -TERM -$PGID 2> /dev/null
for second in $(seq 120); do
  kill -0 $PGID 2> /dev/null || break
  sleep 1
done
kill -KILL -$PGID 2> /dev/null
python3 $HERE/bar_summary.py $D/$LABEL > $D/$LABEL/summary_seed$SEED.txt 2>&1
echo "summary rc $? $(date +%T)" >> $L
echo "queue_baseline_${LABEL}_done $(date +%T)" >> $L
