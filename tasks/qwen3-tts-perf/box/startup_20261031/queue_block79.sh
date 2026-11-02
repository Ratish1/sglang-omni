#!/usr/bin/env bash
# Block 79 of 11-02: #3418's final head (the ring step's rows in one launch) against main. Waits for block 78. The
# qwen3_tts suite and the kernels' tests; seeded concurrency 1 main (card 0) against the PR (card 1), every file by
# SHA-256, all four runs scored, WER and SIM paired on the moved files; the standard SeedTTS A/B, both card orders;
# boots, each tree cold into its own cache then three warm rounds.
# usage: D=... T=<kit tree> MAIN=<tree> PR=<tree> bash queue_block73.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
PR=${PR:?the PR tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block79
L=$W/queue.log
mkdir -p $W/logs
until grep -q "queue block78 done" $D/block78/queue.log; do sleep 5; done
for arm in MAIN PR; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block79 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $PR && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 3600 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts \
  tests/unit_test/kernels > $W/suite.log 2>&1)
echo "suite: $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1)" >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $MAIN --branch $PR --root $W/identity --port 20000 > $W/identity.txt 2>&1)
echo "identity: $(grep -E 'byte identical|error lines|Error' $W/identity.txt | cut -c1-120 | tr '\n' '#')" >> $L
(cd $K/next_20261031 && python3 -c "
from pathlib import Path
from seeded_tail import score_card
root = Path('$W/identity')
score_card(0, 19000, '$MAIN', [root / run / arm for run in ('c1n', 'c1s') for arm in ('main', 'branch')], Path('$W'))
" > $W/score.log 2>&1)
python3 - $W/identity >> $L 2>&1 <<'EOF'
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
    every = sorted(hash_a)
    errors = lambda table, keys: sum(int(table[k]["substitutions"]) + int(table[k]["deletions"]) + int(table[k]["insertions"]) for k in keys)
    words = lambda keys: sum(int(wer_a[k]["substitutions"]) + int(wer_a[k]["deletions"]) + int(wer_a[k]["hits"]) for k in keys)
    print(f"{run}: all {len(every)}; SIM main {statistics.mean(sims_a[k] for k in every):.2f} PR {statistics.mean(sims_b[k] for k in every):.2f}; "
          f"WER {100 * errors(wer_a, every) / words(every):.2f} / {100 * errors(wer_b, every) / words(every):.2f} % over {words(every)} words")
    moved = sorted(key for key in hash_a if hash_a[key] != hash_b.get(key))
    if not moved:
        print(f"{run}: no moved file")
        continue
    else:
        pass
    deltas = [sims_b[key] - sims_a[key] for key in moved]
    low, high = interval(deltas)
    print(f"{run}: moved {len(moved)}; SIM main {statistics.mean(sims_a[k] for k in moved):.2f} PR "
          f"{statistics.mean(sims_b[k] for k in moved):.2f}, paired {statistics.mean(deltas):+.2f} (95 % {low:+.2f} to {high:+.2f}), "
          f"below 40 {sum(sims_a[k] < 40 for k in moved)} / {sum(sims_b[k] < 40 for k in moved)}; WER {100 * errors(wer_a, moved) / words(moved):.2f} / "
          f"{100 * errors(wer_b, moved) / words(moved):.2f} % over {words(moved)} words, above 50 % {sum(float(wer_a[k]['wer']) > 0.5 for k in moved)} / "
          f"{sum(float(wer_b[k]['wer']) > 0.5 for k in moved)}")
EOF
(cd $K/next_20261031 && python3 ab_runs.py --a main=$MAIN --b pr=$PR --root $W/ab --port 21000 > $W/ab.txt 2>&1)
echo "ab: $(grep -E '^(n16|s16)_o[12]|paired|Error' $W/ab.txt | cut -c1-400 | tr '\n' '#')" >> $L

boot() {
  local label=$1 arm=$2 port=$3
  local tree=${!arm}
  local began=$(date +%s.%N)
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/boot_${arm,,}/inductor TRITON_CACHE_DIR=$W/boot_${arm,,}/triton \
    setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port \
    --model-path $M17" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 1
  local pgid=$(cat $W/logs/serve_$label.pid)
  until kill -0 $pgid 2> /dev/null && curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null; then
      echo "$label died; $(grep -E 'Error|error' $W/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      return 1
    else
      sleep 1
    fi
  done
  echo "$label healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s $(date +%T)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}
port=24000
for arm in MAIN PR; do
  boot cold_${arm,,} $arm $port
  port=$((port + 200))
done
for round in 1 2 3; do
  if [ $round -eq 2 ]; then
    order="PR MAIN"
  else
    order="MAIN PR"
  fi
  for arm in $order; do
    boot warm_${arm,,}_$round $arm $port
    port=$((port + 200))
  done
done
python3 $K/startup_20261031/boot_split.py $W > $W/split.txt 2>&1
tail -n 4 $W/split.txt >> $L
for log in $W/logs/serve_*.log; do echo "$(basename $log): $(grep -m1 -oE '#tokens: [0-9]+' $log)" >> $L; done
echo "queue block79 done $(date +%T)" >> $L
