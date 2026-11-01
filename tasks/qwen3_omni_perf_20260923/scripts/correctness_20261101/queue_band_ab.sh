set -u
# 11-01 does the linear 24 to 16 kHz resampler cause the talker's babble? Main against main plus the band-limited
# resampler (test/qwen3-omni-band-limited-ab-20261101), colocated default, boots main_a band_a band_b main_b. Per boot a
# 64-sample warm pass, then PASSES passes over the 60 samples of /data/rows_1101/sample_ids.txt (every sample that went
# above 50 % WER in the census or the per-row pair, then 24 controls) at c16, unseeded; the server stopped, every pass
# scored by Qwen3-ASR and split by bad_output_census.py.
Q=/data/band_ab_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
C=$S/correctness_20261101
PIN=0-31,64-95
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
ASR_MODEL=Qwen/Qwen3-ASR-1.7B
PORT=36400
ASR_PORT=36600
PASSES=${PASSES:-10}
IDS=$Q/sample_ids.txt
mkdir -p $Q
cp /data/rows_1101/sample_ids.txt $IDS
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }

if [ ! -d /data/ab/band ]; then
  git -C /data/ab/main fetch -q origin test/qwen3-omni-band-limited-ab-20261101
  git -C /data/ab/main worktree add -q --detach /data/ab/band FETCH_HEAD
fi

serve() {
  local out=$1 tree=$2 port=$3
  shift 3
  for _ in $(seq 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | sort -n | tail -1)
    [ "$used" -lt 100 ] && break
    sleep 2
  done
  (cd $tree && exec setsid env PYTHONPATH=$tree taskset -c $PIN python3 -u -m sglang_omni.cli serve \
    --host 127.0.0.1 --port $port "$@") > $out 2>&1 &
  SERVE_PID=$!
  for _ in $(seq 360); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; then
      return 0
    elif ! ps -o stat= -g $SERVE_PID 2>/dev/null | grep -qv '^Z'; then
      return 1
    else
      sleep 5
    fi
  done
  return 1
}

stop() {
  kill -TERM -- -$SERVE_PID 2>/dev/null
  for _ in $(seq 60); do kill -0 $SERVE_PID 2>/dev/null || break; sleep 5; done
  kill -KILL -- -$SERVE_PID 2>/dev/null
  sleep 5
}

boot() {
  local label=$1 tree=$2 out=$Q/$1
  mkdir -p $out
  log "$label start: tree $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse HEAD:sglang_omni | cut -c1-8)"
  if ! serve $out/serve.log $tree $PORT --model-path $MODEL \
    --config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate \
    --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768; then
    log "$label not healthy: $(tail -2 $out/serve.log | tr '\n' ' ')"
    stop
    return 1
  fi
  PYTHONPATH=$tree taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency 16 \
    --out $out/warm --max-samples 64 > $out/warm.log 2>&1
  for pass in $(seq $PASSES); do
    PYTHONPATH=$tree taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency 16 \
      --out $out/p$pass --sample-ids $IDS > $out/gen_p$pass.log 2>&1
  done
  log "$label generated $PASSES passes"
  stop
  if serve $out/asr.log $tree $ASR_PORT --model-path $ASR_MODEL; then
    for pass in $(seq $PASSES); do
      PYTHONPATH=$tree taskset -c $PIN python3 $S/run_bench.py score --arm seedtts_en --asr-port $ASR_PORT \
        --out $out/p$pass > $out/score_p$pass.log 2>&1
    done
  else
    log "$label asr not healthy"
  fi
  stop
  for pass in $(seq $PASSES); do
    (cd /data/ab/main && PYTHONPATH=/data/ab/main python3 $C/bad_output_census.py en $out/p$pass/seedtts_en \
      > $out/census_p$pass.txt 2>&1)
    find $out/p$pass/seedtts_en/audio -name '*.wav' -delete
  done
  log "$label bad over $PASSES passes: thinker text $(grep -h '^   thinker text' $out/census_p*.txt | awk '{s += $NF} END {print s}'), speech $(grep -h '^   speech' $out/census_p*.txt | awk '{s += $NF} END {print s}')"
}

log "band ab start: tools $(git -C /data/tools rev-parse --short HEAD), $(wc -w < $IDS) samples, $PASSES passes a boot"
for label in main_a band_a band_b main_b; do
  case $label in
    main_*) boot $label /data/ab/main ;;
    *) boot $label /data/ab/band ;;
  esac
done
log "band ab done"
