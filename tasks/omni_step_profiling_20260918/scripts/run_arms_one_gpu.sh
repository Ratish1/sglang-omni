#!/usr/bin/env bash
# Server arms on ONE card, one boot after another, PASSES passes over the arm list so drift
# shows as the spread between passes. Each boot: provenance, serve with the arm's args,
# startup seconds, which vocoder graph runners captured or were disabled, the seed-tts
# stream benchmark (full English corpus, warmup 1), stop by its own process group. The
# pids seen on the card are logged; more than one voids the boot.
# usage: run_arms_one_gpu.sh <out dir> <card> <passes> <concurrency> <arms file>
#   arms file: one arm per line, "label|tree|serve args"
set -u
OUT=$1 CARD=$2 PASSES=$3 CONC=$4 ARMS=$5
PY=/workspace/sglang-omni/.venv/bin/python
MODEL=/data/ratish/models/Qwen3-TTS-12Hz-1.7B-Base
META=zhaochenyang20/seed-tts-eval-arrow
BENCH_TREE=$(head -1 "$ARMS" | cut -d'|' -f2)
# note(ratish): PORT lets cells run on several cards at once; BENCH_ARGS picks the mode
# (default streaming; "" is non streaming; "--meta <list>" is another corpus)
PORT=${PORT:-8301}
BENCH_ARGS=${BENCH_ARGS---stream}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$OUT"
md5sum "$0" "$ARMS" > "$OUT/md5.txt"
cp "$ARMS" "$OUT/arms.txt"

for pass in $(seq "$PASSES"); do
  # note(ratish): the arm list is read on fd 3 so nothing in the loop consumes it
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
    # note(ratish): PROBE_PATH adds a sitecustomize probe directory behind the tree
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
      rm -rf "$d/bench/audio"
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
    echo "done $(date +%T)" >> "$d/progress.txt"
  done 3< "$ARMS"
done
echo "all done $(date +%T)" > "$OUT/DONE"
