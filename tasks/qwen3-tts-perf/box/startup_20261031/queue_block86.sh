#!/usr/bin/env bash
# Block 86 of 11-02: what follow-up cohort size buys streaming. Waits for block 87. #3418's final head, SeedTTS EN 1.7B
# Base streaming at concurrency 64, every prompt, the default server against one flag change at the same time on the
# other card, each in both card orders: the follow-up gather window 12 and 24 ms (default 4), and the 24 ms window with
# bucket 16 (default 8). Speed only; rows get one frame per engine step, so a longer window lines their decodes up.
# usage: D=... T=<kit tree> PR=<tree> bash queue_block86.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
PR=${PR:?the tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block86
L=$W/queue.log
mkdir -p $W
until grep -q "queue block87 done" $D/block87/queue.log 2> /dev/null; do sleep 5; done
echo "tree $PR $(git -C $PR rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
echo "queue block86 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L

run_arm() {
  local label=$1 card=$2 port=$3 barrier=$4 cache=$5 args=$6
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms TORCHINDUCTOR_CACHE_DIR=$W/cache_$cache/inductor TRITON_CACHE_DIR=$W/cache_$cache/triton timeout 1800 python3 serve_arm.py \
    --label $label --tree $PR --card $card --port $port --concurrency 64 --barrier $barrier --parties 2 "--serve-args=$args" \
    > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null) s; $(python3 - $W/arms/$label/bench/speed_results.json <<'EOF'
import json
import sys
try:
    s = json.load(open(sys.argv[1]))["summary"]
    print(f"done {s['completed_requests']}/{s['total_requests']} req/s {s['throughput_qps']:.2f} RTF {s['rtf_mean']:.4f} first chunk mean {1000 * s['audio_ttfp_mean_s']:.1f} p50 {1000 * s['audio_ttfp_median_s']:.1f} p95 {1000 * s['audio_ttfp_p95_s']:.1f} ms underrun mean {1000 * s['max_playback_underrun_mean_s']:.1f} ms p99 {1000 * s['max_playback_underrun_p99_s']:.1f} ms inter chunk p95 {1000 * s['inter_chunk_p95_s']:.1f}")
except Exception as exc:
    print("no summary", exc)
EOF
); errors $(grep -cE 'Traceback|OutOfMemory|out of memory' $W/arms/$label/serve.log)" >> $L
}

port=26000
for pass in "w12 --vocoder.factory.followup_batch_wait_ms=12" "w24 --vocoder.factory.followup_batch_wait_ms=24" "w24b16 --vocoder.factory.followup_batch_wait_ms=24 --vocoder.factory.followup_max_batch_size=16"; do
  name=${pass%% *}
  flags=${pass#* }
  flags=${flags//=/ }
  for order in 0 1; do
    run_arm ${name}_o${order}_base $order $port $W/barrier_${name}_o${order} base "" &
    run_arm ${name}_o${order}_flag $((1 - order)) $((port + 200)) $W/barrier_${name}_o${order} $name "$flags" &
    wait
    port=$((port + 400))
  done
done
echo "queue block86 done $(date +%T)" >> $L
