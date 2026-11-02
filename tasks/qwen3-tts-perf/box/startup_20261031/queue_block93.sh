#!/usr/bin/env bash
# Block 93 of 11-02: what comes after the whole request head, new numbers only. Waits for block 92. Card 0: non-streaming at
# concurrency 64 with prefill coalescing (4 requests, 10 ms hold), then non-streaming at 128 and 256 with running requests,
# queue and codec slots all N, read against block 80's flat 44. Card 1: streaming at concurrency 64 with the same coalescing,
# read against block 80's 70.5 req/s and first chunk. Speed and first chunk only, nothing scored.
# usage: D=... T=<kit tree> WHOLE=<tree> bash queue_block93.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
WHOLE=${WHOLE:?the tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block93
L=$W/queue.log
mkdir -p $W
until grep -q "queue block92 done" $D/block92/queue.log 2> /dev/null; do sleep 5; done
echo "tree $WHOLE $(git -C $WHOLE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
echo "queue block93 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L

run_arm() {
  local label=$1 card=$2 port=$3 concurrency=$4 flags=$5 args=$6
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 1800 python3 serve_arm.py --label $label --tree $WHOLE --card $card --port $port --concurrency $concurrency $flags "--serve-args=$args" > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null) s; $(python3 - $W/arms/$label/bench/speed_results.json <<'EOF'
import json
import sys
try:
    s = json.load(open(sys.argv[1]))["summary"]
    extra = "" if "audio_ttfp_mean_s" not in s else f" first chunk mean {1000 * s['audio_ttfp_mean_s']:.1f} p50 {1000 * s['audio_ttfp_median_s']:.1f} p95 {1000 * s['audio_ttfp_p95_s']:.1f} ms underrun p99 {1000 * s['max_playback_underrun_p99_s']:.1f} ms"
    print(f"done {s['completed_requests']}/{s['total_requests']} req/s {s['throughput_qps']:.2f} RTF {s['rtf_mean']:.4f} latency mean {1000 * s['latency_mean_s']:.1f} p95 {1000 * s['latency_p95_s']:.1f} ms{extra}")
except Exception as exc:
    print("no summary", exc)
EOF
); $(grep -m1 -oE '#tokens: [0-9]+' $W/arms/$label/serve.log); tracebacks $(grep -c 'Traceback' $W/arms/$label/serve.log)" >> $L
}

COALESCE="--tts_engine.factory.prefill_coalesce_requests 4 --tts_engine.factory.prefill_coalesce_wait_ms 10"
(
  run_arm n64_coalesce 0 20000 64 --no-stream "$COALESCE"
  for n in 128 256; do
    run_arm n$n 0 $((20000 + n)) $n --no-stream "--tts_engine.engine.max_running_requests $n --tts_engine.engine.max_queued_requests $n --vocoder.factory.codec_state_slots $n"
  done
) &
run_arm s64_coalesce 1 21000 64 "" "$COALESCE" &
wait
echo "queue block93 done $(date +%T)" >> $L
