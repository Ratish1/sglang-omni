set -u
# 11-01 per-row readiness (design 16) against upstream main on the two-card speech default (partial start on, the gate's
# 21.7 % of the c16 wall). The layout takes both cards, so the arms run one after another, main branch branch main.
# Per boot: talker_gate_probe on the server's PYTHONPATH, a 64-sample warm pass, the full SeedTTS EN corpus at c16 then
# c64, the server stopped, then the Qwen3-ASR server scores each pass (WER) and the similarity model reads it (SIM).
Q=/data/pp_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
P=$S/perrow_20261101
PIN=0-31,64-95
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
ASR_MODEL=Qwen/Qwen3-ASR-1.7B
PORT=34000
ASR_PORT=34200
mkdir -p $Q
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }

serve() {
  local out=$1 tree=$2 model=$3 port=$4 prefix=$5
  # note (ratish): a card another process still holds memory on leaves the thinker short of its KV pool.
  for _ in $(seq 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | sort -n | tail -1)
    [ "$used" -lt 100 ] && break
    sleep 2
  done
  log "  cards before $(basename $out): $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ';')"
  (cd $tree && exec setsid env PYTHONPATH=$prefix$tree taskset -c $PIN python3 -u -m sglang_omni.cli serve \
    --model-path $model --host 127.0.0.1 --port $port) > $out 2>&1 &
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
  log "$label start: tree $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse HEAD:sglang_omni | cut -c1-8); apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | tr '\n' ';')]"
  if serve $out/serve.log $tree $MODEL $PORT $P/talker_gate_probe:; then
    log "$label healthy"
    PYTHONPATH=$tree taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency 16 \
      --out $out/warm --max-samples 64 > $out/warm.log 2>&1
    for conc in 16 64; do
      local begin end
      begin=$(date +%s.%N)
      PYTHONPATH=$tree taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency $conc \
        --out $out/c$conc > $out/gen_c$conc.log 2>&1
      local rc=$?
      end=$(date +%s.%N)
      log "$label c$conc rc $rc; $(tail -1 $out/gen_c$conc.log | cut -c1-120)"
      log "window ${label}_c$conc $begin $end"
    done
  else
    log "$label not healthy; tail: $(tail -3 $out/serve.log | tr '\n' ' ')"
    stop
    return 1
  fi
  stop
  log "$label stopped; failures: stage $(grep -c 'Stage process failure' $out/serve.log), stuck $(grep -c 'Terminating stuck process' $out/serve.log)"
  if serve $out/asr.log $tree $ASR_MODEL $ASR_PORT ""; then
    for conc in 16 64; do
      PYTHONPATH=$tree taskset -c $PIN python3 $S/run_bench.py score --arm seedtts_en --asr-port $ASR_PORT --out $out/c$conc \
        > $out/score_c$conc.log 2>&1
      log "$label score c$conc rc $?"
    done
  else
    log "$label asr not healthy"
  fi
  stop
  for conc in 16 64; do
    PYTHONPATH=$tree taskset -c $PIN python3 $S/run_bench.py sim --arm seedtts_en --out $out/c$conc > $out/sim_c$conc.log 2>&1
    log "$label sim c$conc rc $?"
  done
  find $out -name '*.wav' -path '*/audio/*' -delete
  python3 $P/gate_probe_report.py $out/serve.log $Q/queue.log $label > $out/gate.txt 2>&1
  while read -r line; do log "  $line"; done < $out/gate.txt
}

log "pair start: tools $(git -C /data/tools rev-parse --short HEAD); boots ${BOOTS:-main_a perrow_a perrow_b main_b}"
for label in ${BOOTS:-main_a perrow_a perrow_b main_b}; do
  case $label in
    main_*) boot $label /data/ab/main ;;
    *) boot $label /data/ab/perrow ;;
  esac
done
log "pair done"
