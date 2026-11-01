set -u
# 11-01 check 2 (ARCHITECTURE_REVIEW_20261101.md section 7): the talker's decode step on upstream main, alone on its
# card. The two-card speech default puts only talker_ar on card 1 (config.py speech_stages: code2wav on the thinker's
# card). One nsys launch session wraps the server; a capture window opens only around each measured pass (c16, then
# c64), node-traced graphs, no context switch collection. talker_step_census.py reads card 1 per window.
Q=/data/c2_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
TREE=/data/ab/main
PORT=33100
PIN=0-31,64-95
SESSION=q3oc2
mkdir -p $Q
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
cards() { nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | tr '\n' ';'; }
log "check2 start: tools $(git -C /data/tools rev-parse --short HEAD); tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C $TREE status --porcelain -- sglang_omni | wc -l) dirty; apps [$(cards)]"
cd $TREE || exit 1
env PYTHONPATH=$TREE setsid taskset -c $PIN nsys launch --session-new=$SESSION --trace=cuda,nvtx \
  --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  python3 -u -m sglang_omni.cli serve --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --host 127.0.0.1 --port $PORT \
  > $Q/serve.log 2>&1 &
WRAP=$!
log "launched wrapper pid $WRAP"
healthy=0
for _ in $(seq 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then healthy=1; break; fi
  kill -0 $WRAP 2>/dev/null || break
  sleep 5
done
if [ $healthy = 0 ]; then
  log "server not healthy; tail: $(tail -3 $Q/serve.log | tr '\n' ' ')"
  exit 1
else
  log "healthy; apps [$(cards)]"
fi
grep -E "KV Cache is allocated|stage=.*gpu_id|placement|Captured Qwen3-Omni predictor" $Q/serve.log | tail -12 | while read -r line; do log "  $line"; done
PYTHONPATH=$TREE taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency 16 \
  --out $Q/warm --max-samples 64 > $Q/warm.log 2>&1
log "warm rc $?"
offset=64
for conc in 16 64; do
  samples=$((conc * 6))
  nsys start --session=$SESSION -o $Q/c$conc --force-overwrite=true >> $Q/nsys_ctl.log 2>&1
  log "c$conc window open"
  PYTHONPATH=$TREE taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency $conc \
    --out $Q/run_c$conc --max-samples $((offset + samples)) --skip-samples $offset > $Q/gen_c$conc.log 2>&1
  rc=$?
  nsys stop --session=$SESSION >> $Q/nsys_ctl.log 2>&1
  log "c$conc rc $rc, window closed; $(grep -hE 'qps|throughput' $Q/gen_c$conc.log | tail -2 | tr '\n' ' ')"
  offset=$((offset + samples))
done
SERVE=$(pgrep -f -- "--port $PORT" | while read -r p; do [[ "$(cat /proc/$p/comm 2>/dev/null)" == python* ]] && echo $p; done | head -1)
log "stopping serve pid ${SERVE:-none}"
[ -n "$SERVE" ] && kill -TERM $SERVE
for _ in $(seq 60); do kill -0 $WRAP 2>/dev/null || break; sleep 5; done
log "server ended; apps [$(cards)]"
for conc in 16 64; do
  if [ -f $Q/c$conc.nsys-rep ]; then
    nsys export --type sqlite --force-overwrite=true -o $Q/c$conc.sqlite $Q/c$conc.nsys-rep > $Q/export_c$conc.log 2>&1
    python3 $S/check2_20261101/talker_step_census.py $Q/c$conc.sqlite --device 1 > $Q/census_c$conc.txt 2>&1
    log "c$conc census rc $?"
    head -14 $Q/census_c$conc.txt | while read -r line; do log "  $line"; done
  else
    log "c$conc: no report"
  fi
done
log "check2 done"
