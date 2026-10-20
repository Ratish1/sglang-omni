#!/usr/bin/env bash
# GIL holders of the Fun-CosyVoice3 pipeline process at stream c16 and c64, without nsys: the server
# in its own process group, a warmup, then py-spy (--gil --threads, blocking, 50 Hz, 20 s) on the
# process whose stacks run the OmniScheduler loop while each cell runs, then the stage profiles.
# usage (in the container): bash pyspy_run_c20261010.sh <tree> <card> <cpus> <port> <out>
TREE=$1 CARD=$2 CPUS=$3 PORT=$4 R=$5
IN=/data/c20261010/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
mkdir -p $R
cd /workspace/$TREE
echo "start $(date +%T) tree $TREE card $CARD cpus $CPUS port $PORT" > $R/progress.txt
setsid env CUDA_VISIBLE_DEVICES=$CARD SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=/workspace/$TREE \
  TORCHINDUCTOR_CACHE_DIR=/data/c20261010/inductor_$TREE \
  taskset -c $CPUS python3 -m sglang_omni.cli serve --model-path $MODEL --host 127.0.0.1 \
  --port $PORT > $R/serve.log 2>&1 < /dev/null &
SERVE_PID=$!
echo "serve pgid $SERVE_PID" >> $R/progress.txt
for i in $(seq 1 360); do
  if curl -sf http://127.0.0.1:$PORT/health > /dev/null; then
    echo "healthy $(date +%T)" >> $R/progress.txt
    break
  fi
  if ! kill -0 $SERVE_PID 2> /dev/null; then
    echo "server died $(date +%T)" >> $R/progress.txt
    exit 1
  fi
  sleep 5
done

bench() {
  local name=$1 concurrency=$2 samples=$3
  (cd /workspace/sglang-omni && taskset -c $CPUS env CUDA_VISIBLE_DEVICES= python3 -m benchmarks.eval.benchmark_tts_seedtts \
    --model $MODEL --port $PORT --use-existing-server --generate-only --stream --lang en --warmup 1 \
    --max-concurrency $concurrency --max-samples $samples --output-dir $R/bench_$name) > $R/bench_$name.log 2>&1
  echo "bench $name rc $? $(date +%T)" >> $R/progress.txt
}

bench warm 16 64
PIPE_PID=""
for pid in $(ps -o pid= -g $SERVE_PID); do
  if [[ "$(cat /proc/$pid/comm)" == python* ]] && py-spy dump --pid $pid 2> /dev/null | grep -q "event_loop_"; then
    PIPE_PID=$pid
    break
  fi
done
echo "pipeline pid ${PIPE_PID:-none}" >> $R/progress.txt
for cell in c16:16:640 c64:64:768; do
  IFS=: read -r name concurrency samples <<< "$cell"
  bench $name $concurrency $samples &
  BENCH_PID=$!
  sleep 8
  if [ -n "$PIPE_PID" ]; then
    timeout 90 py-spy record --pid $PIPE_PID --rate 50 --duration 20 --gil --threads \
      --format raw --output $R/pyspy_$name.txt > $R/pyspy_$name.log 2>&1
    echo "pyspy $name rc $? $(date +%T)" >> $R/progress.txt
  fi
  wait $BENCH_PID
done
kill -TERM -- -$SERVE_PID
sleep 20
kill -KILL -- -$SERVE_PID 2> /dev/null
echo "server stopped $(date +%T)" >> $R/progress.txt
for name in c16 c64; do
  [ -s $R/pyspy_$name.txt ] || continue
  python3 /data/box/pyspy_thread_profile.py $R/pyspy_$name.txt > $R/pyspy_engine_$name.txt 2>&1
  python3 $IN/pyspy_stage_profile.py $R/pyspy_$name.txt --thread-has run_step \
    --phases run_step,select_step_participants,ingest,hop_batch_with_prefix,hop_batch_prefix,leftover_batch,hift_step,hift_group,prepare_flow_conditioning,pack_flow_inputs,split_generated_mels,run \
    --expand run_step,hop_batch_with_prefix,hift_step,hift_group,pack_flow_inputs,prepare_flow_conditioning \
    > $R/pyspy_vocoder_$name.txt 2>&1
done
echo "done $(date +%T)" >> $R/progress.txt
