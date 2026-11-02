set -u
# 11-02 the model card's image preparation against main on the full MMMU validation set (900), text answers, c32, one
# server per pass on one H200, boots in the order main image image main. Both trees are upstream main plus at most the
# fix. mmmu_pair_report.py reads the passes answer by answer.
Q=/data/image_ab_1102
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
C=$S/correctness_20261102
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
PORT=37600
mkdir -p $Q
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }

boot() {
  local label=$1 tree=$2 out=$Q/$1
  mkdir -p $out
  log "$label start: tree $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse HEAD:sglang_omni | cut -c1-8); gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  (cd $tree && exec setsid env PYTHONPATH=$tree python3 -u -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT \
    --model-path $MODEL --config $C/qwen3_omni_colocated_small.yaml --colocate \
    --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768) > $out/serve.log 2>&1 &
  local pid=$!
  local healthy=1
  for _ in $(seq 360); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then
      healthy=0
      break
    elif ! ps -o stat= -g $pid 2>/dev/null | grep -qv '^Z'; then
      break
    else
      sleep 5
    fi
  done
  if [ $healthy = 0 ]; then
    (cd $tree && PYTHONPATH=$tree python3 $S/run_bench.py gen --arm mmmu --port $PORT --concurrency 32 --out $out/c32) \
      > $out/gen.log 2>&1
    log "$label gen rc $?; $(tail -1 $out/gen.log | cut -c1-120)"
  else
    log "$label not healthy: $(tail -2 $out/serve.log | tr '\n' ' ' | cut -c1-300)"
  fi
  kill -TERM -- -$pid 2>/dev/null
  for _ in $(seq 60); do kill -0 $pid 2>/dev/null || break; sleep 5; done
  kill -KILL -- -$pid 2>/dev/null
  sleep 5
}

log "image ab start: tools $(git -C /data/tools rev-parse --short HEAD)"
boot main_a /data/ab/main
boot image_a /data/ab/image
boot image_b /data/ab/image
boot main_b /data/ab/main
python3 $C/mmmu_pair_report.py $Q > $Q/report.txt 2>&1
while read -r line; do log "  $line"; done < $Q/report.txt
log "image ab done"
