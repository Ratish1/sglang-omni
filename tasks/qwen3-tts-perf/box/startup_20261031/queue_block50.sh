#!/usr/bin/env bash
# Block 50 of 10-31: PR B against PR A on seeded streaming c16 (seeds 1 and 2, both card orders), to bound the codec's
# numerics change on speaker similarity and WER; then every request's SIM paired over the four seeded runs.
# Waits for block 49 first.
# usage: D=... T=<kit tree> BASE=<tree> HEAD=<tree> bash queue_block50.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the PR A tree}
HEAD=${HEAD:?the PR B tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block50
L=$W/queue.log
mkdir -p $W
until grep -q "queue block49 done" $D/block49/queue.log 2> /dev/null; do sleep 20; done
for tree in $BASE $HEAD; do
  echo "tree $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block50 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $K/next_20261031 && python3 seeded_stream_ab.py --a base=$BASE --b head=$HEAD --root $W/ab --port 13000 > $W/ab.txt 2>&1)
echo "ab: $(grep -E '^s16seed|paired' $W/ab.txt | cut -c1-400 | tr '\n' '#')" >> $L
(cd $K/next_20261031 && python3 - $W/ab >> $L 2>&1 <<'PY'
import statistics
import sys
from pathlib import Path
from seeded_tail import per_sample
root = Path(sys.argv[1])
deltas = []
for run in sorted(path.name for path in root.glob("s16seed*_o[12]")):
    base = per_sample(root / run / "base", "similarity_results.json", "speaker_similarity")
    head = per_sample(root / run / "head", "similarity_results.json", "speaker_similarity")
    deltas += [head[key] - base[key] for key in sorted(set(base) & set(head))]
half_width = 1.96 * statistics.pstdev(deltas) / len(deltas) ** 0.5
print(f"pooled seeded paired SIM (head minus base) {statistics.mean(deltas):+.3f} +- {half_width:.3f} over {len(deltas)} pairs; identical {sum(delta == 0 for delta in deltas)}")
PY
)
echo "queue block50 done $(date +%T)" >> $L
