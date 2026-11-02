#!/usr/bin/env bash
# Block 80 of 11-02: what concurrency buys, on #3418's final head. Waits for block 79. SeedTTS EN 1.7B Base on one card,
# running requests, queue and codec slots all N, at concurrency N, streaming then non-streaming, N 64, 128, 256; one boot
# per N and mode. Throughput, RTF, first chunk, failures and the server's KV pool per run. Speed only, nothing scored.
# usage: D=... T=<kit tree> PR=<tree> bash queue_block80.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
PR=${PR:?the tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block80
L=$W/queue.log
mkdir -p $W
until grep -q "queue block79 done" $D/block79/queue.log; do sleep 5; done
echo "tree $PR $(git -C $PR rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $PR diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue block80 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
port=22000
for n in 64 128 256; do
  for mode in stream nostream; do
    label=n${n}_${mode}
    flags=""
    if [ $mode = nostream ]; then flags="--no-stream"; else :; fi
    (cd $K/inline_20261026 && ARM_ROOT=$W/arms TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor TRITON_CACHE_DIR=$W/cache/triton timeout 3000 python3 serve_arm.py \
      --label $label --tree $PR --card 0 --port $port --concurrency $n $flags \
      "--serve-args=--tts_engine.engine.max_running_requests $n --tts_engine.engine.max_queued_requests $n --vocoder.factory.codec_state_slots $n" \
      > $W/$label.out 2>&1)
    rc=$?
    summary=$(grep -E "Completed requests|Failed requests|RTF mean|Throughput|TTFC mean|TTFC p95|Latency mean|Latency p99" $W/arms/$label/bench/*.log $W/arms/$label/bench.log 2>/dev/null | sed -E 's/.*:([A-Za-z].*)/\1/' | tr -s ' ' | tr '\n' ';' | cut -c1-300)
    echo "$label exit $rc $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null) s; $summary; $(grep -m1 -oE '#tokens: [0-9]+' $W/arms/$label/serve.log); errors $(grep -cE 'Traceback|OutOfMemory|out of memory' $W/arms/$label/serve.log)" >> $L
    port=$((port + 200))
  done
done
echo "queue block80 done $(date +%T)" >> $L
