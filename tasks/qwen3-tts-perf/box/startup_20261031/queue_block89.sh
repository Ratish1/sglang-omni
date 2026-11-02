#!/usr/bin/env bash
# Block 89 of 11-02: non-streaming requests as one plan through the codec's existing window path. The touched test file;
# card 0: speed at concurrency 64, 16, 1, read against block 88's larger form and blocks 83 and 85's base; card 1: seeded
# files of both modes hashed against block 79's, non-streaming scored and paired against block 79's scores.
# usage: D=... T=<kit tree> WHOLE=<tree> bash queue_block89.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
WHOLE=${WHOLE:?the tree under test}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block89
L=$W/queue.log
mkdir -p $W
echo "tree WHOLE $WHOLE $(git -C $WHOLE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $WHOLE diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue block89 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $WHOLE && CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 3000 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_pipeline.py > $W/suite.log 2>&1)
echo "suite: $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 12 | cut -c1-160 >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/block85/cache_whole/inductor TRITON_CACHE_DIR=$D/block85/cache_whole/triton

run_arm() {
  local label=$1 port=$2 concurrency=$3
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 1800 python3 serve_arm.py --label $label --tree $WHOLE --card 0 --port $port --concurrency $concurrency --no-stream > $W/$label.out 2>&1)
  echo "$label card 0 exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null) s; $(python3 - $W/arms/$label/bench/speed_results.json <<'EOF'
import json
import sys
try:
    s = json.load(open(sys.argv[1]))["summary"]
    print(f"done {s['completed_requests']}/{s['total_requests']} req/s {s['throughput_qps']:.2f} RTF {s['rtf_mean']:.4f} latency mean {1000 * s['latency_mean_s']:.1f} p50 {1000 * s['latency_median_s']:.1f} p95 {1000 * s['latency_p95_s']:.1f} ms")
except Exception as exc:
    print("no summary", exc)
EOF
); $(grep -m1 -oE '#tokens: [0-9]+' $W/arms/$label/serve.log); tracebacks $(grep -c 'Traceback' $W/arms/$label/serve.log); left context $(grep -c 'left-context decoder' $W/arms/$label/serve.log)" >> $L
}

(
  run_arm whole_c64 20000 64
  run_arm whole_c16 20200 16
  run_arm whole_c1 20400 1
) &
(cd $K/startup_20261031 && python3 whole_seeded_quality.py --tree $WHOLE --base-root $D/block79/identity --root $W/seeded --card 1 --port 20800 > $W/seeded.txt 2>&1; grep -E "byte identical|c1n SIM|c1n WER|worst SIM|error lines|Error" $W/seeded.txt | cut -c1-210 >> $L) &
wait
echo "queue block89 done $(date +%T)" >> $L
