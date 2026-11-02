set -u
# 11-02 our thinker layer by layer against HF (served_thinker_dump.py). One colocated boot of TREE, thinker eager,
# radix cache off so a prefill's rows are the whole prompt, SGLang's dumper on, the ledger's requests one at a time on
# HF's processor outputs. Usage: queue_thinker_dump.sh OUT TREE COUNT LEDGER_PREFIX
OUT=$1
TREE=$2
COUNT=$3
LEDGER=$4
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
C=$S/correctness_20261102
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
PORT=37000
mkdir -p $OUT
log() { echo "$(date -u +%T) $*" >> $OUT/queue.log; }
log "dump start: tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8); tools $(git -C /data/tools rev-parse --short HEAD); gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
(cd $TREE && exec setsid env PYTHONPATH=$C/dump_env:$TREE DUMPER_BASE=$OUT/dumps DUMPER_ENABLE=1 \
  DUMPER_ENABLE_OUTPUT_CONSOLE=0 DUMPER_NON_INTRUSIVE_MODE=all DUMPER_EXP_NAME=thinker \
  DUMPER_FILTER="match(r'^non_intrusive__(model\.layers\.\d+\.(self_attn\.inputs\.hidden_states|mlp\.inputs\.0)|logits_processor\.inputs\.\d+)$', name)" \
  python3 -u -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT --model-path $MODEL \
  --config $C/qwen3_omni_colocated_accuracy.yaml --colocate \
  --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768 \
  --thinker.engine.disable_cuda_graph true --thinker.engine.disable_radix_cache true) > $OUT/serve.log 2>&1 &
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
  (cd $TREE && PYTHONPATH=$TREE python3 $C/served_thinker_dump.py send http://127.0.0.1:$PORT $OUT/replies.json \
    /data/sample_ids_60.txt $COUNT) > $OUT/send.log 2>&1
  log "send rc $?; $(tail -1 $OUT/send.log | cut -c1-160)"
else
  log "server not healthy: $(tail -3 $OUT/serve.log | tr '\n' ' ' | cut -c1-400)"
fi
kill -TERM -- -$SERVE_PID 2>/dev/null
for _ in $(seq 60); do kill -0 $SERVE_PID 2>/dev/null || break; sleep 5; done
kill -KILL -- -$SERVE_PID 2>/dev/null
log "stopped; dump dirs: $(for d in $OUT/dumps/*; do echo "$(basename $d)=$(find $d -name '*.pt' | wc -l)"; done | tr '\n' ' ')"
BIGGEST=$(for d in $OUT/dumps/*; do echo "$(find $d -name '*.pt' | wc -l) $d"; done | sort -n | tail -1 | cut -d' ' -f2)
(cd $TREE && PYTHONPATH=$TREE python3 $C/served_thinker_dump.py compare $OUT/compare.json $OUT/replies.json $BIGGEST $LEDGER) > $OUT/compare.log 2>&1
log "compare rc $?"
grep -E "overlap|top1|steps|layer_" $OUT/compare.log | while read -r line; do log "  $line"; done
log "dump done"
