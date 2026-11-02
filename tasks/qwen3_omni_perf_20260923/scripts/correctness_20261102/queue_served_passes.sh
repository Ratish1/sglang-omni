set -u
# 11-02 served SeedTTS EN against HF's own cells: one boot of TREE on the repo's H200 profile on card 0, PASSES passes of the
# full corpus at CONC, each scored by queue_score.sh, then the passes merged and paired by sample against HF_RUN.
# Usage: queue_served_passes.sh OUT TREE PASSES CONC HF_RUN
OUT=$1
TREE=$2
PASSES=$3
CONC=$4
HF_RUN=$5
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
C=$S/correctness_20261102
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
PORT=38100
mkdir -p $OUT
log() { echo "$(date -u +%T) $*" >> $OUT/queue.log; }
log "served passes start: tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8); tools $(git -C /data/tools rev-parse --short HEAD); c$CONC x $PASSES"
(cd $TREE && exec setsid env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TREE python3 -u -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT \
  --model-path $MODEL --config examples/configs/qwen3_omni_colocated_h200.yaml --colocate \
  --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768) > $OUT/serve.log 2>&1 &
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
RUNS=""
if [ $healthy = 0 ]; then
  for pass in $(seq $PASSES); do
    (cd $TREE && PYTHONPATH=$TREE python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency $CONC --out $OUT/p$pass) > $OUT/gen_p$pass.log 2>&1
    log "pass $pass rc $?: $(tail -1 $OUT/gen_p$pass.log | cut -c1-140)"
    RUNS="$RUNS $OUT/p$pass"
  done
else
  log "server not healthy: $(tail -3 $OUT/serve.log | tr '\n' ' ' | cut -c1-400)"
fi
kill -TERM -- -$SERVE_PID 2>/dev/null
for _ in $(seq 60); do kill -0 $SERVE_PID 2>/dev/null || break; sleep 5; done
kill -KILL -- -$SERVE_PID 2>/dev/null
sleep 5
if [ -n "$RUNS" ]; then
  CUDA_VISIBLE_DEVICES=0 bash $C/queue_score.sh $OUT/score.log $TREE $RUNS
  python3 $C/merge_passes.py $OUT/merged $RUNS >> $OUT/queue.log 2>&1
  python3 $C/cells_paired.py $OUT/merged $HF_RUN $OUT/paired.json >> $OUT/queue.log 2>&1
else
  :
fi
log "served passes done"
