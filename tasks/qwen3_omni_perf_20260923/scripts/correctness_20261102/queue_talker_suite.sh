set -u
# 11-02 the served talker against HF on one H200: one colocated boot of TREE with the talker capture, smaller budgets so
# HF's talker fits beside the server. test_talker.py (stage 6's requests teacher-forced through HF FP32 and BF16), then
# SeedTTS EN at c16 captured, then talker_draw_calibration.py over those captures.
# Usage: queue_talker_suite.sh OUT TREE PASSES [media]
# With media, test_talker.py runs the image and video stages' requests in place of stage 6's and the SeedTTS 60.
OUT=$1
TREE=$2
PASSES=$3
if [ "${4:-}" = media ]; then
  REQUESTS=QWEN3_OMNI_ACCURACY_TALKER_MEDIA=1
  PROFILE=qwen3_omni_colocated_media.yaml
else
  REQUESTS=QWEN3_OMNI_ACCURACY_SEEDTTS_IDS=/data/sample_ids_60.txt
  PROFILE=qwen3_omni_colocated_accuracy.yaml
fi
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
C=$S/correctness_20261102
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
PORT=37200
mkdir -p $OUT/capture $OUT/capture_seedtts $OUT/report
log() { echo "$(date -u +%T) $*" >> $OUT/queue.log; }
log "talker suite start: tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8); tools $(git -C /data/tools rev-parse --short HEAD); gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
(cd $TREE && exec setsid env PYTHONPATH=$TREE SGLANG_OMNI_TALKER_CAPTURE_DIR=$OUT/capture \
  python3 -u -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT --model-path $MODEL \
  --config $C/$PROFILE --colocate \
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
if [ $healthy = 0 ]; then
  log "healthy; gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  (cd $TREE && QWEN3_OMNI_ACCURACY_MODEL=$MODEL QWEN3_OMNI_ACCURACY_REPORT_DIR=$OUT/report \
    QWEN3_OMNI_ACCURACY_SERVER=http://127.0.0.1:$PORT QWEN3_OMNI_ACCURACY_TALKER_CAPTURE=$OUT/capture \
    env $REQUESTS QWEN3_OMNI_ACCURACY_CONCURRENCY=${CONCURRENCY:-64} \
    PYTHONPATH=$TREE python3 -m pytest -s -q -x -p no:cacheprovider \
    tests/test_model/qwen3_omni_component_accuracy/test_talker.py) > $OUT/test_talker.log 2>&1
  log "test_talker rc $?: $(grep -E '^\[talker\]' $OUT/test_talker.log | cut -c1-500)"
  log "$(tail -1 $OUT/test_talker.log | cut -c1-160)"
  for pass in $(seq $PASSES); do
    PYTHONPATH=$TREE python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency 16 \
      --out $OUT/seedtts_p$pass --sample-ids /data/sample_ids_60.txt > $OUT/gen_p$pass.log 2>&1
    log "seedtts pass $pass rc $?"
  done
else
  log "server not healthy: $(tail -3 $OUT/serve.log | tr '\n' ' ' | cut -c1-400)"
fi
kill -TERM -- -$SERVE_PID 2>/dev/null
for _ in $(seq 60); do kill -0 $SERVE_PID 2>/dev/null || break; sleep 5; done
kill -KILL -- -$SERVE_PID 2>/dev/null
(cd $TREE && PYTHONPATH=$TREE python3 $C/talker_draw_calibration.py $OUT/draw_calibration.json $OUT/capture) > $OUT/draw_calibration.log 2>&1
log "draw calibration rc $?: $(tr -d '\n' < $OUT/draw_calibration.json | cut -c1-600)"
log "talker suite done"
