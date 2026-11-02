#!/usr/bin/env bash
# Block 85 of 11-02: non-streaming requests replayed per cohort on the codec state against #3418's final head. The qwen3_tts suite,
# then SeedTTS EN 1.7B Base non-streaming on all 1,088 prompts, one server per arm at the same time, at concurrency 1, then 64 and
# 16 in both card orders and concurrency 1 once. Speed and the server's boot, KV pool and errors; nothing scored.
# usage: D=... T=<kit tree> PR=<tree> WHOLE=<tree> bash queue_block85.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
PR=${PR:?the base tree}
WHOLE=${WHOLE:?the tree under test}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block85
L=$W/queue.log
mkdir -p $W
for arm in PR WHOLE; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block85 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $WHOLE && CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 3000 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts > $W/suite.log 2>&1)
echo "suite: $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 12 | cut -c1-160 >> $L

run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 concurrency=$5 barrier=$6 cache=$7
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms TORCHINDUCTOR_CACHE_DIR=$W/cache_$cache/inductor TRITON_CACHE_DIR=$W/cache_$cache/triton timeout 1800 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency $concurrency --no-stream --barrier $barrier --parties 2 \
    > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null) s; $(python3 - $W/arms/$label/bench/speed_results.json <<'EOF'
import json
import sys
try:
    s = json.load(open(sys.argv[1]))["summary"]
    print(f"done {s['completed_requests']}/{s['total_requests']} req/s {s['throughput_qps']:.2f} RTF {s['rtf_mean']:.4f} latency mean {1000 * s['latency_mean_s']:.1f} p50 {1000 * s['latency_median_s']:.1f} p95 {1000 * s['latency_p95_s']:.1f} ms")
except Exception as exc:
    print("no summary", exc)
EOF
); $(grep -m1 -oE '#tokens: [0-9]+' $W/arms/$label/serve.log); errors $(grep -cE 'Traceback|OutOfMemory|out of memory' $W/arms/$label/serve.log); fallbacks $(grep -c 'left-context decoder' $W/arms/$label/serve.log)" >> $L
}

port=25000
for pass in "1 0" "64 0" "64 1" "16 0" "16 1"; do
  set -- $pass
  concurrency=$1
  order=$2
  run_arm c${concurrency}_o${order}_pr $PR $order $port $concurrency $W/barrier_c${concurrency}_o${order} pr &
  run_arm c${concurrency}_o${order}_whole $WHOLE $((1 - order)) $((port + 200)) $concurrency $W/barrier_c${concurrency}_o${order} whole &
  wait
  port=$((port + 400))
done
echo "queue block85 done $(date +%T)" >> $L
