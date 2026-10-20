#!/usr/bin/env bash
# GIL ownership of the pipeline process under stream c16: one server (ab_boot.sh, c16 cell only) and,
# 20 s into the cell, two py-spy records of 25 s each on the stage process that holds the card:
# --gil (only the thread holding the GIL is sampled) and all non idle threads.
# usage (in the container): bash pyspy_run.sh <tree> <card> <port> <out>
TREE=$1
CARD=$2
PORT=$3
OUT=$4
CPUS=$([ "$CARD" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95)
mkdir -p "$OUT"
CELLS="c16:16:::1" bash /workspace/tools/tasks/cosyvoice_utilization_20260912/stage4/ab_boot.sh \
  "/workspace/$TREE" "$OUT" "$CARD" "$PORT" "$CPUS" "/data/p20261008/inductor_$TREE" &
BOOT=$!
until grep -q "healthy" "$OUT/progress.txt" 2>/dev/null || [ -e "$OUT/FAILED" ]; do sleep 2; done
if [ -e "$OUT/FAILED" ]; then
  wait $BOOT
  exit 1
else
  :
fi
sleep 20
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v c="$CARD" '$1 == c {print $2}')
STAGE=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$UUID" '$1 == u {print $2}' | head -1)
echo "stage pid $STAGE $(date +%T)" >> "$OUT/progress.txt"
taskset -c "$CPUS" py-spy record --pid "$STAGE" --gil --rate 100 --duration 25 --format raw \
  --output "$OUT/pyspy_gil.txt" > "$OUT/pyspy_gil.log" 2>&1
taskset -c "$CPUS" py-spy record --pid "$STAGE" --rate 100 --duration 25 --format raw \
  --output "$OUT/pyspy_threads.txt" > "$OUT/pyspy_threads.log" 2>&1
echo "pyspy done $(date +%T)" >> "$OUT/progress.txt"
wait $BOOT
