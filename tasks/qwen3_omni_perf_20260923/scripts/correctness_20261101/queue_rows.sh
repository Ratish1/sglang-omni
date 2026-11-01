set -u
# 11-01 correctness plan row 10: our talker prefill rows against HF's. One colocated boot of the capture branch
# (main f4fbb735d plus the talker capture), the census's and the pair's bad SeedTTS EN samples plus 24 controls captured
# at c16, the server stopped, then
# prompt_rows_parity.py builds HF's rows in bfloat16 (floor) and float32 (truth) across cards 0 and 1 and compares.
Q=/data/rows_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
C=$S/correctness_20261101
TREE=/data/ab/capture
PIN=0-31,64-95
MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
PORT=34800
mkdir -p $Q/captures
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }

if [ ! -d $TREE ]; then
  git -C /data/ab/main fetch -q origin test/qwen3-omni-talker-capture-20261101
  git -C /data/ab/main worktree add -q --detach $TREE FETCH_HEAD
fi
log "rows start: tree $(git -C $TREE rev-parse --short HEAD); tools $(git -C /data/tools rev-parse --short HEAD)"
for _ in $(seq 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | sort -n | tail -1)
  [ "$used" -lt 100 ] && break
  sleep 2
done
(cd $TREE && exec setsid env PYTHONPATH=$TREE SGLANG_OMNI_TALKER_CAPTURE_DIR=$Q/captures taskset -c $PIN python3 -u \
  -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT --model-path $MODEL \
  --config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate \
  --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768) > $Q/serve.log 2>&1 &
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
# note (ratish): every sample above 50 % WER in the census and the per-row pair, then the corpus's first 24 as controls.
python3 -c "
import glob, json
bad, controls = [], []
for path in sorted(glob.glob('/data/census_1101/*/c*/seedtts_en/wer_results.json') + glob.glob('/data/pp_1101/*/c*/seedtts_en/wer_results.json')):
    rows = json.load(open(path))['per_sample']
    bad += [row['id'] for row in rows if row['wer'] > 0.5 and row['id'] not in bad]
    controls = controls or [row['id'] for row in rows[:24]]
print(' '.join(bad + [sample_id for sample_id in controls if sample_id not in bad]))
" > $Q/sample_ids.txt
log "samples: $(wc -w < $Q/sample_ids.txt)"
if [ $healthy = 0 ]; then
  PYTHONPATH=$TREE taskset -c $PIN python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency 16 \
    --out $Q/gen --sample-ids $Q/sample_ids.txt > $Q/gen.log 2>&1
  log "gen rc $?; captures $(ls $Q/captures | wc -l)"
else
  log "server not healthy: $(tail -2 $Q/serve.log | tr '\n' ' ')"
fi
kill -TERM -- -$SERVE_PID 2>/dev/null
for _ in $(seq 60); do kill -0 $SERVE_PID 2>/dev/null || break; sleep 5; done
kill -KILL -- -$SERVE_PID 2>/dev/null
sleep 10
log "server stopped"
for dtype in bfloat16 float32; do
  (cd $TREE && PYTHONPATH=$TREE python3 $C/prompt_rows_parity.py build $dtype $Q/$dtype.pt $Q/captures/*.pt) \
    > $Q/build_$dtype.log 2>&1
  log "build $dtype rc $?; $(grep -c '^built' $Q/build_$dtype.log) built; $(tail -1 $Q/build_$dtype.log | cut -c1-150)"
done
(cd $TREE && PYTHONPATH=$TREE python3 $C/prompt_rows_parity.py compare $Q/float32.pt $Q/bfloat16.pt $Q/captures/*.pt) \
  > $Q/compare.txt 2>&1
grep '^==\|^FAIL' $Q/compare.txt | while read -r line; do log "  $line"; done
log "rows done"
