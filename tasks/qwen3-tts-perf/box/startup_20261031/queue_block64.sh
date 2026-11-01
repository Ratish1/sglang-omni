#!/usr/bin/env bash
# Block 64 of 11-01: block 63's seeded runs (main card 0, the reference encoder fix card 1, seed 1234, concurrency 1)
# differ on 98 files, every prompt of 55 reference clips whose codes the fix moves. All four runs scored as the A/B
# scores them (seeded_tail.score_card: ASR transcription then speaker similarity, card 0), then WER and speaker
# similarity paired per prompt on the moved files and checked equal on the rest. Waits for block 63.
# usage: D=... T=<kit tree> MAIN=<tree> bash queue_block64.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
K=$T/tasks/qwen3-tts-perf/box
R=$D/block63/identity
W=$D/block64
L=$W/queue.log
mkdir -p $W
until grep -q "queue block63 done" $D/block63/queue.log 2> /dev/null; do sleep 20; done
echo "queue block64 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $K/next_20261031 && python3 -c "
from pathlib import Path
from seeded_tail import score_card
root = Path('$R')
score_card(0, 19000, '$MAIN', [root / run / arm for run in ('c1n', 'c1s') for arm in ('main', 'branch')], Path('$W'))
" > $W/score.log 2>&1)
echo "scored: $(grep -c scored $W/score.log) runs" >> $L
python3 - $R >> $L 2>&1 <<'EOF'
import csv
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])


def load(run, arm):
    bench = root / run / arm / "bench"
    sims = {r["id"]: r["speaker_similarity"] for r in json.loads((bench / "similarity_results.json").read_text())["per_sample"]}
    with (bench / "wer_results.csv").open() as handle:
        wer = {r["id"]: r for r in csv.DictReader(handle)}
    hashes = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in (bench / "audio").glob("*.wav")}
    return sims, wer, hashes


def interval(deltas):
    rng = random.Random(0)
    means = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
    return means[50], means[1949]


for run in ("c1n", "c1s"):
    (sims_a, wer_a, hash_a), (sims_b, wer_b, hash_b) = load(run, "main"), load(run, "branch")
    moved = sorted(key for key in hash_a if hash_a[key] != hash_b.get(key))
    same = [key for key in hash_a if hash_a[key] == hash_b.get(key)]
    sim_equal = sum(sims_a[key] == sims_b[key] for key in same)
    deltas = [sims_b[key] - sims_a[key] for key in moved]
    low, high = interval(deltas)
    errors = lambda table, keys: sum(int(table[k]["substitutions"]) + int(table[k]["deletions"]) + int(table[k]["insertions"]) for k in keys)
    words = sum(int(wer_a[k]["substitutions"]) + int(wer_a[k]["deletions"]) + int(wer_a[k]["hits"]) for k in moved)
    print(f"{run}: moved {len(moved)}, unmoved {len(same)} (SIM equal on {sim_equal}); on the moved, SIM main {statistics.mean(sims_a[k] for k in moved):.2f} "
          f"fix {statistics.mean(sims_b[k] for k in moved):.2f}, paired {statistics.mean(deltas):+.2f} (95 % {low:+.2f} to {high:+.2f}), "
          f"below 40 {sum(sims_a[k] < 40 for k in moved)} / {sum(sims_b[k] < 40 for k in moved)}, "
          f"WER main {100 * errors(wer_a, moved) / words:.2f} % fix {100 * errors(wer_b, moved) / words:.2f} % over {words} words, "
          f"above 50 % {sum(float(wer_a[k]['wer']) > 0.5 for k in moved)} / {sum(float(wer_b[k]['wer']) > 0.5 for k in moved)}")
    print(f"{run}: whole run SIM main {statistics.mean(sims_a.values()):.3f} fix {statistics.mean(sims_b.values()):.3f}")
EOF
echo "queue block64 done $(date +%T)" >> $L
