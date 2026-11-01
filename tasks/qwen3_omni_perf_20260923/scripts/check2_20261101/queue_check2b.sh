set -u
# 11-01, after check 5: check 2's c64 window ended with code2wav (its own process on card 0 in the two-card speech
# default) no longer acknowledging the talker's CUDA IPC transfers, under node-level tracing. The same layout and
# tree untraced: a warm pass, then two full SeedTTS EN c64 passes; any stage failure or ack timeout is the defect
# reproducing without the profiler.
Q=/data/c2b_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
TREE=/data/ab/main
PORT=33500
PIN=0-31,64-95
mkdir -p $Q
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
until grep -q "check5 done" /data/c5_1101/queue.log 2>/dev/null; do sleep 10; done
log "check2b start: tools $(git -C /data/tools rev-parse --short HEAD); tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8); apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | tr '\n' ';')]"
cd $TREE || exit 1
env PYTHONPATH=$TREE setsid taskset -c $PIN python3 -u -m sglang_omni.cli serve --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --host 127.0.0.1 --port $PORT > $Q/serve.log 2>&1 &
SERVE=$!
healthy=0
for _ in $(seq 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then healthy=1; break; fi
  kill -0 $SERVE 2>/dev/null || break
  sleep 5
done
if [ $healthy = 0 ]; then
  log "server not healthy; tail: $(tail -3 $Q/serve.log | tr '\n' ' ')"
  exit 1
else
  log "healthy"
fi
PYTHONPATH=$TREE taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency 16 \
  --out $Q/warm --max-samples 64 > $Q/warm.log 2>&1
log "warm rc $?"
for pass in 1 2; do
  PYTHONPATH=$TREE taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency 64 \
    --out $Q/c64_$pass > $Q/gen_c64_$pass.log 2>&1
  log "c64 pass $pass rc $?; $(tail -1 $Q/gen_c64_$pass.log | cut -c1-200)"
  log "  failures so far: stage failure $(grep -c 'Stage process failure' $Q/serve.log), ack crashes $(grep -c 'comm ack .* task crashed' $Q/serve.log), stuck $(grep -c 'Terminating stuck process' $Q/serve.log)"
done
kill -TERM -- -$SERVE 2>/dev/null
for _ in $(seq 60); do kill -0 $SERVE 2>/dev/null || break; sleep 5; done
log "server ended; apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | tr '\n' ';')]"
log "check2b done"
