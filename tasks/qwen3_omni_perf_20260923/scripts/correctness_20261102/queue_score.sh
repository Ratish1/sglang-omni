set -u
# 11-02 score finished passes: each RUN directory holds seedtts_en/{generated.json,reply_text.json,audio}. One Qwen3-ASR
# server, WER per pass, the speaker similarity, the census by owner; WAVs kept only for samples above 50 % WER.
# Usage: queue_score.sh LOG TREE RUN [RUN ...]
LOG=$1
TREE=$2
shift 2
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
C=$S/correctness_20261101
ASR_MODEL=Qwen/Qwen3-ASR-1.7B
PORT=37400
log() { echo "$(date -u +%T) $*" >> $LOG; }
(cd $TREE && exec setsid env PYTHONPATH=$TREE python3 -u -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT \
  --model-path $ASR_MODEL) > $LOG.asr 2>&1 &
SERVE_PID=$!
healthy=1
for _ in $(seq 120); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then
    healthy=0
    break
  elif ! ps -o stat= -g $SERVE_PID 2>/dev/null | grep -qv '^Z'; then
    break
  else
    sleep 5
  fi
done
if [ $healthy != 0 ]; then
  log "asr not healthy: $(tail -2 $LOG.asr | tr '\n' ' ' | cut -c1-300)"
else
  for run in "$@"; do
    PYTHONPATH=$TREE python3 $S/run_bench.py score --arm seedtts_en --asr-port $PORT --out $run > $run/score.log 2>&1
    log "$run score rc $?"
  done
fi
kill -TERM -- -$SERVE_PID 2>/dev/null
for _ in $(seq 40); do kill -0 $SERVE_PID 2>/dev/null || break; sleep 3; done
kill -KILL -- -$SERVE_PID 2>/dev/null
for run in "$@"; do
  # note (ratish): a repeated sample's id carries its pass; the scorer looks its reference up by the bare id.
  mkdir -p $run/sim/seedtts_en
  python3 - "$run" <<'PY'
import json, sys
run = sys.argv[1]
rows = json.load(open(f"{run}/seedtts_en/generated.json"))
for row in rows:
    row["sample_id"] = row["sample_id"].split("__r")[0]
json.dump(rows, open(f"{run}/sim/seedtts_en/generated.json", "w"))
PY
  PYTHONPATH=$TREE python3 $S/run_bench.py sim --arm seedtts_en --out $run/sim > $run/sim.log 2>&1
  log "$run sim rc $?: $(python3 -c "import json;d=json.load(open('$run/sim/seedtts_en/sim_summary.json'));print(d)" 2>/dev/null | cut -c1-260)"
  (cd $TREE && PYTHONPATH=$TREE python3 $C/bad_output_census.py en $run/seedtts_en > $run/census.txt 2>&1)
  log "$run: $(head -3 $run/census.txt | cut -c1-200 | tr '\n' ' ')"
  python3 - "$run" <<'PY'
import json, os, sys
run = sys.argv[1] + "/seedtts_en"
rows = json.load(open(f"{run}/wer_results.json"))["per_sample"]
bad = {row["id"] for row in rows if row["wer"] > 0.5}
for name in os.listdir(f"{run}/audio"):
    if name[:-4] not in bad:
        os.remove(f"{run}/audio/{name}")
PY
done
log "score done"
