#!/usr/bin/env bash
# The omni-gpu-deep-dive trace pair (mapping, then formal) of vocoder cases on one tree and card.
# The BBuf backend's tables are made on the Mac from the synced traces.
# usage (in the container): bash trace_pairs.sh <tree> <card> <out> <case>...
set -u
TREE=$1
CARD=$2
OUT=$3
shift 3
CPUS=$([ "$CARD" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95)
mkdir -p "$OUT"
cd "/workspace/$TREE"
# a script run by path imports from its own directory first, so the tree goes on PYTHONPATH
export PYTHONPATH=/workspace/$TREE
python3 -c "import sglang_omni; print(sglang_omni.__file__)" > "$OUT/import_path.txt"
grep -q "^/workspace/$TREE/" "$OUT/import_path.txt" || { echo "imports $(cat "$OUT/import_path.txt")" > "$OUT/FAILED"; exit 1; }
echo "tree $(git log --oneline -1) card $CARD start $(date +%T)" >> "$OUT/progress.txt"
for side in mapping formal; do
  CUDA_VISIBLE_DEVICES=$CARD TORCHINDUCTOR_CACHE_DIR=/data/p20261008/inductor_$TREE taskset -c "$CPUS" \
    python3 -u /workspace/tools/tasks/cosyvoice_utilization_20260912/stage4/vocoder_trace_pair.py \
    --side $side --cases "$@" --out "$OUT" > "$OUT/$side.log" 2>&1
  echo "$side exit $? $(date +%T)" >> "$OUT/progress.txt"
done
echo "done $(date +%T)" >> "$OUT/progress.txt"
