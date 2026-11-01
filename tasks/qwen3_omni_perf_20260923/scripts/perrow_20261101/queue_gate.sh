set -u
# 11-01 per-row readiness, audit E9 step 1 (DECISIONS "Colocated item ownership" 2): how often the talker's all-rows
# decode gate defers a step on upstream main with partial start on, colocated (card 0, --talker_ar.factory.
# enable_partial_start true) and in the two-card speech default (partial start on by default). talker_gate_probe on
# the server's PYTHONPATH; per boot a 64-sample warm pass, then the full SeedTTS EN corpus at c16 and at c64, each pass
# window logged in epoch seconds for gate_probe_report.py. The colocated default cannot defer (a request builds only
# after its thinker stream closed, request_builders.py:1326), so it is not run.
Q=/data/g_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
P=$S/perrow_20261101
TREE=/data/ab/main
PIN=0-31,64-95
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
mkdir -p $Q
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }

boot() {
  local label=$1 port=$2; shift 2
  mkdir -p $Q/$label
  (cd $TREE && exec setsid env PYTHONPATH=$P/talker_gate_probe:$TREE taskset -c $PIN python3 -u -m sglang_omni.cli serve \
    --model-path $MODEL "$@" --host 127.0.0.1 --port $port) > $Q/$label/serve.log 2>&1 &
  echo $! > $Q/$label/serve.pid
  for _ in $(seq 360); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; then
      log "$label healthy"
      return 0
    else
      sleep 5
    fi
  done
  log "$label not healthy; tail: $(tail -3 $Q/$label/serve.log | tr '\n' ' ')"
  return 1
}

passes() {
  local label=$1 port=$2
  PYTHONPATH=$TREE taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $port --concurrency 16 \
    --out $Q/$label/warm --max-samples 64 > $Q/$label/warm.log 2>&1
  log "$label warm rc $?"
  for conc in 16 64; do
    local begin end
    begin=$(date +%s.%N)
    PYTHONPATH=$TREE taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $port --concurrency $conc \
      --out $Q/$label/c$conc > $Q/$label/gen_c$conc.log 2>&1
    local rc=$?
    end=$(date +%s.%N)
    log "$label c$conc rc $rc; $(tail -1 $Q/$label/gen_c$conc.log | cut -c1-200)"
    log "window ${label}_c$conc $begin $end"
  done
}

stop() {
  local label=$1 pid
  pid=$(cat $Q/$label/serve.pid)
  kill -TERM -- -$pid 2>/dev/null
  for _ in $(seq 60); do kill -0 $pid 2>/dev/null || break; sleep 5; done
  log "$label stopped; failures: stage $(grep -c 'Stage process failure' $Q/$label/serve.log), stuck $(grep -c 'Terminating stuck process' $Q/$label/serve.log); apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | tr '\n' ';')]"
}

log "gate start: tools $(git -C /data/tools rev-parse --short HEAD); tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8); apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | tr '\n' ';')]"
# note (ratish): ports 200 apart, as a server also binds its port plus 100.
if boot colocated_partial 33700 --config examples/configs/qwen3_omni_colocated_h100_bf16.yaml \
  --colocate --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768 \
  --talker_ar.factory.enable_partial_start true; then
  passes colocated_partial 33700
else
  log "colocated_partial passes skipped"
fi
stop colocated_partial
if boot two_card 33900; then
  passes two_card 33900
else
  log "two_card passes skipped"
fi
stop two_card
for label in colocated_partial two_card; do
  python3 $P/gate_probe_report.py $Q/$label/serve.log $Q/queue.log $label > $Q/$label/gate.txt 2>&1
  while read -r line; do log "  $label $line"; done < $Q/$label/gate.txt
done
log "gate done"
