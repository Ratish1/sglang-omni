#!/usr/bin/env bash
# Block 98 of 11-02: enough lock samples to rank a flat host profile. The chain base, streaming SeedTTS EN at concurrency 64,
# py-spy on the lock holder, thread names kept, three rounds on each of cards 0 and 1, then all rounds reduced together, the
# stage process alone (the API server has its own lock). Shares only; nothing about held time or speed is read from these.
# usage: D=... T=<kit tree> BASE=<tree> bash queue_block98.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the base tree}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/block98
L=$W/queue.log
mkdir -p $W
echo "tree BASE $BASE $(git -C $BASE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
echo "queue block98 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/block96/cache/inductor TRITON_CACHE_DIR=$D/block96/cache/triton
port=20000
for round in 1 2 3; do
  (cd $K && python3 gil_owners.py --tree $BASE --root $W/r${round}_c0 --card 0 --port $port > $W/r${round}_c0.txt 2>&1) &
  (cd $K && python3 gil_owners.py --tree $BASE --root $W/r${round}_c1 --card 1 --port $((port + 400)) > $W/r${round}_c1.txt 2>&1) &
  wait
  echo "round $round done $(date +%T)" >> $L
  port=$((port + 800))
done
(cd $K && python3 gil_owners.py --tree $BASE --root $W --card 0 --port 1 --reduce-glob "$W/r*/gil_*.txt" > $W/reduced.txt 2>&1)
grep -E "^stage process|^[a-zA-Z_0-9N-]+: [0-9.]+ %" $W/reduced.txt | head -n 10 | cut -c1-110 >> $L
echo "queue block98 done $(date +%T)" >> $L
