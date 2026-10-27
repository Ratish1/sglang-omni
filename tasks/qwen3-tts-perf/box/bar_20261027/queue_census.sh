#!/usr/bin/env bash
# The first-audio path at low load, component by component: each tree in ARMS ("label:tree ...") served under Nsight
# Systems with the pipeline_nvtx probe (every stage hop, scheduler phase, model-runner phase and vocoder step, joined by
# request id), 1.7B CustomVoice at default settings, Nari's client unmodified at 1 req/s (Ryan, English, Poisson, 20 s
# warm-up, 60 s measured, seed 0); then pipeline_census.py sections A (probe coverage), B (components), C (hops) and F
# (first-chunk critical path per request) over the whole report; NSYS_ARGS and SECTIONS override the trace and the
# sections (python-gil with E for the interpreter lock, node graph tracing with G for line attribution). Waits for
# AFTER_LOG to hold AFTER_LINE. Card 0.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
ARMS=${ARMS:?label:tree pairs}
AFTER_LOG=${AFTER_LOG:?the log to wait on}
AFTER_LINE=${AFTER_LINE:?the line to wait for}
PORT=${PORT:-33000}
S=$T/tasks/omni_step_profiling_20260918/scripts
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
NSYS_ARGS=${NSYS_ARGS:-"--trace=cuda,nvtx,osrt --cuda-graph-trace=graph --sample=none --cpuctxsw=none"}
SECTIONS=${SECTIONS:-ABCF}
L=$D/queue.log
mkdir -p $D
until grep -q "$AFTER_LINE" $AFTER_LOG 2> /dev/null; do
  sleep 20
done
echo "queue census start $(date +%T) kit $(git -C $T rev-parse HEAD) $(nsys --version 2>&1 | tail -n 1)" >> $L
for arm in $ARMS; do
  label=${arm%%:*}
  tree=${arm#*:}
  out=$D/$label
  mkdir -p $out
  echo "$label tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(date +%T)" >> $L
  (cd $tree && env PYTHONPATH=$tree:$S/pipeline_nvtx OMNI_PIPE_NVTX=1 SGLANG_OMNI_STRICT_PORT=1 setsid bash -c \
    "echo \$\$ > $out/serve.pid; exec nsys profile -o $out/serve --force-overwrite=true $NSYS_ARGS \
    python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT --model-path $MODEL \
    --tts_engine.gpu 0 --vocoder.gpu 0" > $out/serve.log 2>&1 < /dev/null &)
  sleep 2
  pgid=$(cat $out/serve.pid)
  waited=0
  healthy=1
  until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label not healthy after $waited s; $(grep -E 'Error|error' $out/serve.log | tail -n 2 | tr '\n' ' ')" >> $L
      healthy=0
      break
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  if [ $healthy = 1 ]; then
    echo "$label healthy after $waited s $(date +%T)" >> $L
    (cd /data/tts_bench_src/tts_bench && uv run -q bench run --target sglang-omni --base-url http://127.0.0.1:$PORT \
      --model $MODEL --voice Ryan --language English --dataset datasets/seed-tts-eval.jsonl --rps 1 \
      --warmup 20s --duration 60s --arrival poisson --seed 0 --timeout 120s --max-in-flight 4096 \
      --output $out/rps1 > $out/bench.log 2>&1)
    echo "$label load rc $? $(date +%T)" >> $L
  else
    :
  fi
  serve=$(for p in $(pgrep -f "sglang_omni.cli serve.*--port $PORT"); do
    if [[ "$(cat /proc/$p/comm)" == python* ]]; then echo $p; fi; done | head -1)
  kill -TERM $serve 2> /dev/null
  for second in $(seq 600); do
    kill -0 $pgid 2> /dev/null || break
    sleep 1
  done
  kill -KILL -$pgid 2> /dev/null
  echo "$label stopped $(date +%T)" >> $L
  if [ $healthy = 1 ]; then
    nsys export --type sqlite --force-overwrite=true -o $out/serve.sqlite $out/serve.nsys-rep > $out/export.log 2>&1
    echo "$label exported rc $? $(date +%T)" >> $L
    (cd $S && python3 pipeline_census.py $out/serve.sqlite --sections $SECTIONS > $out/census.txt 2>&1)
    echo "$label census rc $? $(date +%T)" >> $L
  else
    :
  fi
  port=$((PORT + 20))
  PORT=$port
done
echo "queue_census_done $(date +%T)" >> $L
