#!/usr/bin/env bash
# Block 99 of 11-02: does quality survive FP8 weights, before any FP8 kernel is written. fp8_weights_sim.py rounds the talker's
# and predictor's GEMM weights to the e4m3 grid and stores them as bf16. The chain base serves the original checkpoint on one
# card and the rounded one on the other, at the same time, SeedTTS EN streaming at concurrency 16, all 1,088 prompts, scored
# for WER and SIM, then swapped. Activations stay bf16, so a pass means FP8 weights are worth building and nothing more.
# usage: D=... T=<kit tree> BASE=<tree> bash queue_block99.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block99
L=$W/queue.log
SIM=/data/fp8sim/Qwen3-TTS-12Hz-1.7B-Base
ORIGINAL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
mkdir -p $W
echo "tree BASE $BASE $(git -C $BASE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
echo "queue block99 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
SOURCE=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
python3 $K/startup_20261031/fp8_weights_sim.py --source $SOURCE --out $SIM > $W/sim.txt 2>&1
echo "sim exit $? $(date +%T)" >> $L
cat $W/sim.txt | cut -c1-120 >> $L

run_arm() {
  local label=$1 model=$2 card=$3 port=$4 barrier=$5
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms TORCHINDUCTOR_CACHE_DIR=$D/block96/cache/inductor TRITON_CACHE_DIR=$D/block96/cache/triton timeout 3000 python3 serve_arm.py \
    --label $label --tree $BASE --card $card --port $port --concurrency 16 --score --barrier $barrier --parties 2 --model $model > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T); $(grep -hE 'RTF mean|Throughput \(req/s\)|TTFC mean' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-110); $(grep -hE 'WER \(corpus' $W/arms/$label/score--transcribe-only.log 2>/dev/null | tr -s ' ' | cut -c1-50); $(grep -hE 'SeedTTS speaker similarity' $W/arms/$label/score--similarity-only.log 2>/dev/null | tr -s ' ' | cut -c1-60); tracebacks $(grep -c Traceback $W/arms/$label/serve.log)" >> $L
}
run_arm base_o0 $ORIGINAL 2 21000 $W/barrier_o0 &
run_arm fp8_o0 $SIM 3 21800 $W/barrier_o0 &
wait
run_arm base_o1 $ORIGINAL 3 22600 $W/barrier_o1 &
run_arm fp8_o1 $SIM 2 23400 $W/barrier_o1 &
wait
python3 - $W >> $L 2>&1 <<'EOF'
import csv
import json
import random
import statistics
import sys

root = sys.argv[1]
for order in ("o0", "o1"):
    base = {r["id"]: r for r in json.load(open(f"{root}/arms/base_{order}/bench/similarity_results.json"))["per_sample"]}
    arm = {r["id"]: r for r in json.load(open(f"{root}/arms/fp8_{order}/bench/similarity_results.json"))["per_sample"]}
    keys = sorted(set(base) & set(arm))
    deltas = [arm[k]["speaker_similarity"] - base[k]["speaker_similarity"] for k in keys]
    rng = random.Random(0)
    means = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
    a = list(csv.DictReader(open(f"{root}/arms/base_{order}/bench/wer_results.csv")))
    b = list(csv.DictReader(open(f"{root}/arms/fp8_{order}/bench/wer_results.csv")))
    errors = lambda rows: sum(int(r["substitutions"]) + int(r["deletions"]) + int(r["insertions"]) for r in rows)
    words = sum(int(r["substitutions"]) + int(r["deletions"]) + int(r["hits"]) for r in a)
    print(f"{order}: SIM {statistics.mean(base[k]['speaker_similarity'] for k in keys):.2f} / {statistics.mean(arm[k]['speaker_similarity'] for k in keys):.2f}, paired {statistics.mean(deltas):+.3f} (95 % {means[50]:+.3f} to {means[1949]:+.3f}), below 40 {sum(base[k]['speaker_similarity'] < 40 for k in keys)} / {sum(arm[k]['speaker_similarity'] < 40 for k in keys)}; WER errors {errors(a)} / {errors(b)} over {words} words, above 50 % {sum(float(r['wer']) > 0.5 for r in a)} / {sum(float(r['wer']) > 0.5 for r in b)}")
EOF
echo "queue block99 done $(date +%T)" >> $L
