#!/usr/bin/env bash
# Block 112 of 11-02: does quality survive FP8 KV, before any fused store is written? The host path head on SGLang's stock
# FP8 KV path against the same head on bf16, SeedTTS EN streaming at concurrency 16, all 1,088 prompts, scored, same time, both
# card orders. The speed side is already read (DESIGNS item 2: 4 launches a layer through this path, fails the c1 bar), so this
# reads WER and SIM only. A pass means a fused FP8 store is worth building, and nothing more.
# usage: D=... T=<kit tree> TREE=<tree> [BLOCK=block112] [CARD_A=2] [CARD_B=3] [PORT=52000] bash queue_block112.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
TREE=${TREE:?the tree}
BLOCK=${BLOCK:-block112}
CARD_A=${CARD_A:-2}
CARD_B=${CARD_B:-3}
port=${PORT:-52000}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
echo "tree $TREE $(git -C $TREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $TREE diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
FP8="--tts_engine.engine.kv_cache_dtype fp8_e4m3"
run_arm() {
  local label=$1 serve_args=$2 card=$3 port=$4 barrier=$5 cache=$6
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms TORCHINDUCTOR_CACHE_DIR=$W/cache_$cache/inductor TRITON_CACHE_DIR=$W/cache_$cache/triton timeout 3000 python3 serve_arm.py \
    --label $label --tree $TREE --card $card --port $port --concurrency 16 --score --serve-args "$serve_args" --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T); $(grep -hE 'RTF mean|Throughput \(req/s\)|TTFC mean' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-110); $(grep -hE 'WER \(corpus' $W/arms/$label/score--transcribe-only.log 2>/dev/null | tr -s ' ' | cut -c1-50); $(grep -hE 'SeedTTS speaker similarity' $W/arms/$label/score--similarity-only.log 2>/dev/null | tr -s ' ' | cut -c1-60); tracebacks $(grep -c Traceback $W/arms/$label/serve.log)" >> $L
}
run_arm base_o0 "" $CARD_A $port $W/barrier_o0 base &
run_arm fp8_o0 "$FP8" $CARD_B $((port + 400)) $W/barrier_o0 fp8 &
wait
port=$((port + 800))
run_arm base_o1 "" $CARD_B $port $W/barrier_o1 base &
run_arm fp8_o1 "$FP8" $CARD_A $((port + 400)) $W/barrier_o1 fp8 &
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
echo "queue $BLOCK done $(date +%T)" >> $L
