#!/usr/bin/env bash
# One Fun-CosyVoice3 serve, clean or under nsys, and a list of SeedTTS cells run one after
# another against it. Each cell writes its own client log, whose "Benchmarking N requests"
# and "Results saved" stamps cut that cell's window from the DCGM log and the nsys report.
# Only the serve is sent TERM; nsys finalizes itself.
#
# usage: run_census_boot.sh <tree> <out dir> <card> <port> <cells>
#   cells     "mode:concurrency[:samples] ...", samples empty for the whole English split
#   NSYS_ARGS profile flags, "none" (the default) for a clean boot
#   PROBE=1   the cosy_nvtx probe on the server's path (LINES= passes OMNI_PIPE_LINES)
#   SERVE_ARGS extra serve arguments
#   SEED      sampling seed of every request, empty for none                  1234
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4 CELLS=$5
S=$(cd "$(dirname "$0")" && pwd)
T=$(dirname "$S")
PY=python3
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
URL=http://127.0.0.1:$PORT
SEED=${SEED-1234}
NSYS_ARGS=${NSYS_ARGS:-none}
mkdir -p "$OUT"
git -C "$TREE" rev-parse HEAD > "$OUT/head.txt"
git -C "$T" rev-parse HEAD > "$OUT/tools_head.txt"
(cd "$TREE" && PYTHONPATH="$TREE" $PY -c "import sglang_omni; print(sglang_omni.__file__)") > "$OUT/import_path.txt"
grep -q "^$TREE/" "$OUT/import_path.txt" || { echo "server would import $(cat "$OUT/import_path.txt")" > "$OUT/FAILED"; exit 1; }
nvidia-smi > "$OUT/gpus_before.txt"
echo "start $(date +%T) cells=$CELLS nsys=$NSYS_ARGS probe=${PROBE:-0} lines=${LINES:-}" > "$OUT/progress.txt"

if [ "$NSYS_ARGS" = none ]; then
  WRAP=""
else
  WRAP="nsys profile -o $OUT/serve --force-overwrite=true $NSYS_ARGS"
fi
PROBE_ENV=""
PROBE_PATH=""
if [ "${PROBE:-0}" = 1 ]; then
  PROBE_ENV="OMNI_PIPE_NVTX=1 OMNI_PIPE_LINES=${LINES:-}"
  PROBE_PATH=":$S/cosy_nvtx"
elif [ "${COUNT:-0}" = 1 ]; then
  PROBE_ENV="OMNI_PREFIX_COUNT=1"
  PROBE_PATH=":$S/prefix_count"
fi
if [ -n "$SEED" ]; then
  printf '{"seed": %d, "max_new_tokens": null}\n' "$SEED" > "$OUT/generation.json"
else
  printf '{"max_new_tokens": null}\n' > "$OUT/generation.json"
fi

(while true; do echo "$(date +%s) $(cat /proc/loadavg)"; sleep 30; done) > "$OUT/loadavg.log" 2>&1 &
LOADS=$!
nvidia-smi dmon -i "$CARD" -s pucm -d 1 > "$OUT/dmon.log" 2>&1 &
DMON=$!
(while true; do echo "$(date +%s) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$CARD" | tr '\n' ';')"; sleep 2; done) > "$OUT/pids.log" 2>&1 &
PIDS=$!
(cd "$TREE" && env CUDA_VISIBLE_DEVICES=$CARD SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH="$TREE$PROBE_PATH" $PROBE_ENV \
  $WRAP $PY -u -m sglang_omni.cli serve --model-path $MODEL --host 127.0.0.1 --port $PORT ${SERVE_ARGS:-} \
  > "$OUT/serve.log" 2>&1) &
LAUNCH=$!

healthy=0
for _ in $(seq 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' $URL/health)" = 200 ]; then healthy=1; break; fi
  kill -0 $LAUNCH 2>/dev/null || break
  sleep 5
done
if [ $healthy = 1 ]; then
  echo "healthy $(date +%T)" >> "$OUT/progress.txt"
  for cell in $CELLS; do
    IFS=: read -r mode conc samples <<< "$cell"
    name="$mode-c$conc${samples:+-n$samples}"
    (cd "$T/../.." && $PY -u "$T/diagnostics/run_seedtts.py" --mode "$mode" --lang en \
      --concurrency "$conc" --warmup 1 ${samples:+--samples $samples} --model $MODEL \
      --base-url $URL --generation-json "$OUT/generation.json" --ready-timeout 900 \
      --output "$OUT/$name") > "$OUT/$name.log" 2>&1
    echo "cell $name rc $? $(date +%T)" >> "$OUT/progress.txt"
  done
else
  echo "server not healthy" > "$OUT/FAILED"
fi

SERVE_PID=$(for p in $(pgrep -f "sglang_omni.cli serve.*--port $PORT"); do
  if [[ "$(cat /proc/$p/comm)" == python* ]]; then echo $p; fi; done | head -1)
echo "serve pid $SERVE_PID" >> "$OUT/progress.txt"
[ -n "$SERVE_PID" ] && kill -TERM "$SERVE_PID"
for _ in $(seq 180); do
  kill -0 $LAUNCH 2>/dev/null || break
  sleep 5
done
kill -0 $LAUNCH 2>/dev/null && echo "serve still running after 15 min" >> "$OUT/progress.txt"
echo "serve ended $(date +%T)" >> "$OUT/progress.txt"
kill $LOADS $DMON $PIDS 2>/dev/null
nvidia-smi > "$OUT/gpus_after.txt"
[ -f "$OUT/FAILED" ] && exit 1
if [ "$NSYS_ARGS" != none ]; then
  nsys export --type sqlite --force-overwrite=true -o "$OUT/serve.sqlite" "$OUT/serve.nsys-rep" > "$OUT/export.log" 2>&1
  echo "exported $(date +%T)" >> "$OUT/progress.txt"
fi
echo "done $(date +%T)" >> "$OUT/progress.txt"
