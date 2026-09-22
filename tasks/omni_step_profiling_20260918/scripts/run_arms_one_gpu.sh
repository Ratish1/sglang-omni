#!/usr/bin/env bash
# Server arms on ONE card, one boot after another, PASSES passes over the arm list so drift
# shows as the spread between passes. Each boot: provenance, serve with the arm's args,
# startup seconds, which vocoder graph runners captured or were disabled, the seed-tts
# stream benchmark (full English corpus, warmup 1), stop by its own process group, and with
# QUALITY=1 WER and speaker similarity on the boot's WAVs. The
# pids seen on the card are logged; more than one voids the boot.
# usage: run_arms_one_gpu.sh <out dir> <card> <passes> <concurrency> <arms file>
#   arms file: one arm per line, "label|tree|serve args"
set -u
OUT=$1 CARD=$2 PASSES=$3 CONC=$4 ARMS=$5
PY=python3
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
META=zhaochenyang20/seed-tts-eval-arrow
BENCH_TREE=$(head -1 "$ARMS" | cut -d'|' -f2)
#PORT lets cells run on several cards at once; BENCH_ARGS picks the mode
# (default streaming; "" is non streaming; "--meta <list>" is another corpus)
PORT=${PORT:-8301}
BENCH_ARGS=${BENCH_ARGS---stream}
mkdir -p "$OUT"
md5sum "$0" "$ARMS" > "$OUT/md5.txt"
cp "$ARMS" "$OUT/arms.txt"

for pass in $(seq "$PASSES"); do
  #the arm list is read on fd 3 so nothing in the loop consumes it
  while IFS='|' read -r label tree serve_args <&3; do
    [ -n "$label" ] || continue
    d=$OUT/pass${pass}_$label; mkdir -p "$d"
    echo "start $(date +%T) card $CARD args: $serve_args" > "$d/progress.txt"
    git -C "$tree" rev-parse HEAD > "$d/head.txt"
    (cd "$tree" && CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$tree $PY -c \
      "import sglang_omni; print(sglang_omni.__file__)") > "$d/import_path.txt" 2>&1
    nvidia-smi dmon -i "$CARD" -s pucvm -d 1 > "$d/dmon.log" 2>&1 &
    dmon=$!
    uuid=$(nvidia-smi -i "$CARD" --query-gpu=uuid --format=csv,noheader)
    (while true; do nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$uuid" | sed "s/^/$(date +%T) /"; sleep 2; done) > "$d/apps.csv" 2>&1 &
    apps=$!
    began=$(date +%s)
    #PROBE_PATH adds a sitecustomize probe directory behind the tree
    (cd "$tree" && setsid bash -c "echo \$\$ > $d/server.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD \
      OMNI_FLOW_PROBE=$d/flow PYTHONPATH=$tree${PROBE_PATH:+:$PROBE_PATH} $PY -u -m sglang_omni.cli serve --model-path $MODEL --port $PORT $serve_args" \
      > "$d/serve.log" 2>&1 &)
    healthy=0
    for _ in $(seq 360); do
      if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then healthy=1; break; fi
      kill -0 -- -"$(cat "$d/server.pgid" 2>/dev/null)" 2>/dev/null || break
      sleep 5
    done
    echo "healthy=$healthy startup_s=$(( $(date +%s) - began ))" >> "$d/progress.txt"
    echo "vocoder runners captured $(grep -c 'Codec graphs captured' "$d/serve.log"), disabled $(grep -c 'capture disabled' "$d/serve.log"), fused modules line: $(grep -c 'fused SnakeBeta modules' "$d/serve.log")" >> "$d/progress.txt"
    if [ $healthy = 1 ]; then
      (cd "$BENCH_TREE" && PYTHONPATH=$BENCH_TREE timeout 3600 $PY -m benchmarks.eval.benchmark_tts_seedtts \
        --model $MODEL --meta $META --lang en --use-existing-server --host 127.0.0.1 --port $PORT \
        --warmup 1 --generate-only --concurrency "$CONC" --output-dir "$d/bench" $BENCH_ARGS) > "$d/bench.log" 2>&1
      echo "bench rc $? $(date +%T)" >> "$d/progress.txt"
    fi
    kill -TERM -- -"$(cat "$d/server.pgid")" 2>/dev/null
    sleep 15
    kill -0 -- -"$(cat "$d/server.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$d/server.pgid")"
    kill $dmon $apps 2>/dev/null
    echo "pids on the card: $(awk '{print $3}' "$d/apps.csv" | sort -u | tr '\n' ' ')" >> "$d/progress.txt"
    for _ in $(seq 60); do
      [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CARD")" -lt 100 ] && break
      sleep 2
    done
    #QUALITY=1 scores WER and speaker similarity on the boot's WAVs once the server is gone
    if [ "${QUALITY:-0}" = 1 ] && [ $healthy = 1 ]; then
      (cd "$BENCH_TREE" && CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$BENCH_TREE $PY -m benchmarks.eval.benchmark_tts_seedtts \
        --model $MODEL --meta $META --lang en --port $((PORT + 100)) --skip-gpu-cleanup \
        --transcribe-only --output-dir "$d/bench") > "$d/wer.log" 2>&1
      echo "wer rc $? $(date +%T)" >> "$d/progress.txt"
      #--skip-gpu-cleanup returns before the ASR server frees the card
      for _ in $(seq 60); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CARD")" -lt 100 ] && break
        sleep 2
      done
      #the WavLM scorer's attention grows with the square of clip length, so a runaway
      # clip does not fit on the card; SIM skips a missing WAV, the same >50% WER
      # exclusion the WER corpus figure uses, and the count lands in the log
      $PY - "$d/bench" >> "$d/progress.txt" <<'PYEOF'
import json, os, sys
bench = sys.argv[1]
wer = json.load(open(os.path.join(bench, "wer_results.json")))["per_sample"]
above = {row["id"] for row in wer if row.get("wer") is not None and row["wer"] > 0.5}
for entry in json.load(open(os.path.join(bench, "generated.json"))):
    if entry.get("sample_id") in above and os.path.isfile(entry.get("wav_path") or ""):
        os.remove(entry["wav_path"])
print(f"sim excludes {len(above)} samples above 50% WER")
PYEOF
      (cd "$BENCH_TREE" && CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$BENCH_TREE $PY -m benchmarks.eval.benchmark_tts_seedtts \
        --model $MODEL --meta $META --lang en --similarity-only --output-dir "$d/bench") > "$d/sim.log" 2>&1
      echo "sim rc $? $(date +%T)" >> "$d/progress.txt"
    fi
    rm -rf "$d/bench/audio"
    echo "done $(date +%T)" >> "$d/progress.txt"
  done 3< "$ARMS"
done
echo "all done $(date +%T)" > "$OUT/DONE"
