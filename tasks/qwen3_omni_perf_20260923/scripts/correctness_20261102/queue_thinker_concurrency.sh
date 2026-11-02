set -u
# 11-02 test_thinker_concurrency.py on two H200s: the repo's H200 profile on card 0, HF's thinkers on card 1, 110 requests
# at CONCURRENCY (64 by default); SERVE_EXTRA adds server flags. Usage: queue_thinker_concurrency.sh OUT TREE
OUT=$1
TREE=$2
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
# note (ratish): below the host's ephemeral range, where another process can hold a port a server asks for.
PORT=${SERVE_PORT:-27700}
mkdir -p $OUT/report
log() { echo "$(date -u +%T) $*" >> $OUT/queue.log; }
log "thinker check start: tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8); cards $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
(cd $TREE && exec setsid env SGLANG_OMNI_STRICT_PORT=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TREE python3 -u -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT \
  --model-path $MODEL --config examples/configs/qwen3_omni_colocated_h200.yaml --colocate \
  --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768 ${SERVE_EXTRA:-}) > $OUT/serve.log 2>&1 &
SERVE_PID=$!
healthy=1
for _ in $(seq 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then
    healthy=0
    break
  elif ! ps -o stat= -g $SERVE_PID 2>/dev/null | grep -qv '^Z'; then
    break
  else
    sleep 5
  fi
done
if [ $healthy = 0 ]; then
  log "healthy; cards $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
  (cd $TREE && QWEN3_OMNI_ACCURACY_MODEL=$MODEL QWEN3_OMNI_ACCURACY_REPORT_DIR=$OUT/report \
    QWEN3_OMNI_ACCURACY_SERVER=http://127.0.0.1:$PORT QWEN3_OMNI_ACCURACY_CONCURRENCY=${CONCURRENCY:-64} \
    QWEN3_OMNI_ACCURACY_REFERENCE_DEVICE=cuda:1 QWEN3_OMNI_ACCURACY_REFERENCE_CACHE=$OUT/reference.json \
    PYTHONPATH=$TREE python3 -m pytest -s -q -x -p no:cacheprovider \
    tests/test_model/qwen3_omni_component_accuracy/test_thinker_concurrency.py) > $OUT/test.log 2>&1
  log "test rc $?: $(tail -1 $OUT/test.log | cut -c1-140)"
else
  log "server not healthy: $(tail -3 $OUT/serve.log | tr '\n' ' ' | cut -c1-400)"
fi
kill -TERM -- -$SERVE_PID 2>/dev/null
for _ in $(seq 60); do kill -0 $SERVE_PID 2>/dev/null || break; sleep 5; done
kill -KILL -- -$SERVE_PID 2>/dev/null
log "thinker check done"
