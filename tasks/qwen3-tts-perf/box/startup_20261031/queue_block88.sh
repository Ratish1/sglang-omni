#!/usr/bin/env bash
# Block 88 of 11-02: only the numbers no earlier block holds. Card 0: the cohort replay head, non-streaming SeedTTS EN at
# concurrency 64, 16, 1, read against the base's blocks 83 and 85 (c64 43.74 to 44.80, c16 31.29, 31.52, c1 3.98 req/s).
# Card 1: #3418's head, streaming at concurrency 64 with a 24 ms follow-up gather window, read against block 80 (70.5).
# usage: D=... T=<kit tree> PR=<tree> WHOLE=<tree> bash queue_block88.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
PR=${PR:?the base tree}
WHOLE=${WHOLE:?the tree under test}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block88
L=$W/queue.log
mkdir -p $W
echo "tree WHOLE $WHOLE $(git -C $WHOLE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
echo "queue block88 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L

run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 concurrency=$5 cache=$6 flags=$7 args=$8
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms TORCHINDUCTOR_CACHE_DIR=$D/block85/cache_$cache/inductor TRITON_CACHE_DIR=$D/block85/cache_$cache/triton timeout 1800 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency $concurrency $flags "--serve-args=$args" > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null) s; $(python3 - $W/arms/$label/bench/speed_results.json <<'EOF'
import json
import sys
try:
    s = json.load(open(sys.argv[1]))["summary"]
    extra = "" if "audio_ttfp_mean_s" not in s else f" first chunk mean {1000 * s['audio_ttfp_mean_s']:.1f} p50 {1000 * s['audio_ttfp_median_s']:.1f} p95 {1000 * s['audio_ttfp_p95_s']:.1f} ms underrun p99 {1000 * s['max_playback_underrun_p99_s']:.1f} ms"
    print(f"done {s['completed_requests']}/{s['total_requests']} req/s {s['throughput_qps']:.2f} RTF {s['rtf_mean']:.4f} latency mean {1000 * s['latency_mean_s']:.1f} p50 {1000 * s['latency_median_s']:.1f} p95 {1000 * s['latency_p95_s']:.1f} ms{extra}")
except Exception as exc:
    print("no summary", exc)
EOF
); $(grep -m1 -oE '#tokens: [0-9]+' $W/arms/$label/serve.log); tracebacks $(grep -c 'Traceback' $W/arms/$label/serve.log); left graphs $(grep -c 'using the whole-utterance decode' $W/arms/$label/serve.log)" >> $L
}

(
  run_arm whole_c64 $WHOLE 0 20000 64 whole --no-stream ""
  run_arm whole_c16 $WHOLE 0 20200 16 whole --no-stream ""
  run_arm whole_c1 $WHOLE 0 20400 1 whole --no-stream ""
) &
run_arm stream_w24 $PR 1 20600 64 pr "" "--vocoder.factory.followup_batch_wait_ms 24" &
wait
echo "queue block88 done $(date +%T)" >> $L
