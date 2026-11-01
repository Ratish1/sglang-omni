set -u
# 11-01 bad-output census on upstream main: which stage owns each SeedTTS EN word error and each bad sample (WER above
# 0.5). One boot per layout (the colocated H100 BF16 default, then the two-card speech default); per boot a 64-sample
# warm pass, then the full corpus at c64 twice and c16 twice, run_bench.py recording the thinker's reply text. The
# server stopped, the Qwen3-ASR server scores every pass, bad_output_census.py splits the errors by owner, and the
# bad samples' WAVs are kept in keep/ (the rest deleted).
Q=/data/census_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
C=$S/correctness_20261101
TREE=/data/ab/main
PIN=0-31,64-95
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
ASR_MODEL=Qwen/Qwen3-ASR-1.7B
PORT=34400
ASR_PORT=34600
PASSES="c64_1 c64_2 c16_1 c16_2"
mkdir -p $Q
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }

serve() {
  local out=$1 port=$2
  shift 2
  for _ in $(seq 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | sort -n | tail -1)
    [ "$used" -lt 100 ] && break
    sleep 2
  done
  (cd $TREE && exec setsid env PYTHONPATH=$TREE taskset -c $PIN python3 -u -m sglang_omni.cli serve \
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
  local label=$1 out=$Q/$1
  shift
  mkdir -p $out
  log "$label start: tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8)"
  local healthy=1
  for attempt in 1 2 3; do
    if serve $out/serve.log $PORT --model-path $MODEL "$@"; then
      healthy=0
      break
    else
      log "$label attempt $attempt not healthy: $(grep -o 'minimum viable[^.]*' $out/serve.log | head -1)"
      cp $out/serve.log $out/serve_attempt$attempt.log
      stop
    fi
  done
  if [ $healthy != 0 ]; then
    log "$label not healthy"
    return 1
  fi
  PYTHONPATH=$TREE taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency 16 \
    --out $out/warm --max-samples 64 > $out/warm.log 2>&1
  for name in $PASSES; do
    local conc=${name%_*}
    PYTHONPATH=$TREE taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $PORT \
      --concurrency ${conc#c} --out $out/$name > $out/gen_$name.log 2>&1
    log "$label $name rc $?; $(tail -1 $out/gen_$name.log | cut -c1-100)"
  done
  stop
  log "$label stopped; failures: stage $(grep -c 'Stage process failure' $out/serve.log), stuck $(grep -c 'Terminating stuck process' $out/serve.log)"
  if serve $out/asr.log $ASR_PORT --model-path $ASR_MODEL; then
    for name in $PASSES; do
      PYTHONPATH=$TREE taskset -c $PIN python3 $S/run_bench.py score --arm seedtts_en --asr-port $ASR_PORT \
        --out $out/$name > $out/score_$name.log 2>&1
      log "$label score $name rc $?"
    done
  else
    log "$label asr not healthy"
  fi
  stop
  for name in $PASSES; do
    local pass_dir=$out/$name/seedtts_en
    (cd $TREE && python3 $C/bad_output_census.py en $pass_dir > $out/census_$name.txt 2>&1)
    head -3 $out/census_$name.txt | while read -r line; do log "  $label $name $line"; done
    mkdir -p $out/keep/$name
    python3 -c "
import json, shutil, sys
rows = json.load(open('$pass_dir/wer_results.json'))['per_sample']
for row in rows:
    if row['wer'] > 0.5:
        shutil.copy('$pass_dir/audio/' + row['id'] + '.wav', '$out/keep/$name/')
"
    find $pass_dir/audio -name '*.wav' -delete
  done
}

log "census start: tools $(git -C /data/tools rev-parse --short HEAD)"
boot colocated --config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate \
  --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768
boot two_card
log "census done"
