set -u
# 11-02 thinker_media_rows.py: HF's rows on card 1 once WAIT_HF_LOG holds its text, then one colocated boot of TREE on card 0
# once WAIT_SERVER_LOG holds its text: thinker eager, radix cache off, SGLang's dumper on the chosen layers.
# Usage: queue_media_rows.sh OUT TREE WAIT_HF_LOG WAIT_HF_TEXT WAIT_SERVER_LOG WAIT_SERVER_TEXT
OUT=$1
TREE=$2
WAIT_HF_LOG=$3
WAIT_HF_TEXT=$4
WAIT_SERVER_LOG=$5
WAIT_SERVER_TEXT=$6
C=/data/tools/tasks/qwen3_omni_perf_20260923/scripts/correctness_20261102
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
# note (ratish): below the host's ephemeral range, where another process can hold a port a server asks for.
PORT=${SERVE_PORT:-28200}
mkdir -p $OUT
log() { echo "$(date -u +%T) $*" >> $OUT/queue.log; }
until grep -q "$WAIT_HF_TEXT" "$WAIT_HF_LOG" 2>/dev/null; do sleep 10; done
if grep -q 'hf done' $OUT/hf.log 2>/dev/null; then
  log "hf rows kept from an earlier run"
else
  log "hf start: tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8); tools $(git -C /data/tools rev-parse --short HEAD)"
  (cd $TREE && PYTHONPATH=$TREE python3 -u $C/thinker_media_rows.py hf $OUT cuda:1) > $OUT/hf.log 2>&1
  log "hf rc $?: $(tail -1 $OUT/hf.log | cut -c1-120)"
fi
until grep -q "$WAIT_SERVER_TEXT" "$WAIT_SERVER_LOG" 2>/dev/null; do sleep 10; done
log "server start"
(cd $TREE && exec setsid env SGLANG_OMNI_STRICT_PORT=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$C/dump_env:$TREE DUMPER_BASE=$OUT/dumps DUMPER_ENABLE=1 \
  DUMPER_ENABLE_OUTPUT_CONSOLE=0 DUMPER_NON_INTRUSIVE_MODE=all DUMPER_EXP_NAME=thinker \
  DUMPER_FILTER="match(r'^non_intrusive__model\.layers\.(0|1|2|3|12|23|24|47)\.(self_attn\.inputs\.hidden_states|mlp\.inputs\.0)$', name)" \
  python3 -u -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT --model-path $MODEL \
  --config examples/configs/qwen3_omni_colocated_h200.yaml --colocate \
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
  (cd $TREE && PYTHONPATH=$TREE python3 -u $C/thinker_media_rows.py send http://127.0.0.1:$PORT $OUT) > $OUT/send.log 2>&1
  log "send rc $?: $(tail -1 $OUT/send.log | cut -c1-140)"
else
  log "server not healthy: $(tail -3 $OUT/serve.log | tr '\n' ' ' | cut -c1-400)"
fi
kill -TERM -- -$SERVE_PID 2>/dev/null
for _ in $(seq 60); do kill -0 $SERVE_PID 2>/dev/null || break; sleep 5; done
kill -KILL -- -$SERVE_PID 2>/dev/null
log "stopped; dump dirs: $(for d in $OUT/dumps/*; do echo "$(basename $d)=$(find $d -name '*.pt' | wc -l)"; done | tr '\n' ' ')"
# note (ratish): the talker's warmups dump more files, the thinker's long prompts more bytes.
BIGGEST=$(du -s $OUT/dumps/* | sort -n | tail -1 | cut -f2)
(cd $TREE && PYTHONPATH=$TREE python3 -u $C/thinker_media_rows.py compare $OUT $BIGGEST) > $OUT/compare.log 2>&1
log "compare rc $?"
log "media rows done"
