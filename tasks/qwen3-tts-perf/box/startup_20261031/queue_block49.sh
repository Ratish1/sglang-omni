#!/usr/bin/env bash
# Block 49 of 10-31: startup item 4, the 64 whole-utterance decoder graphs, on main, card 0, one fresh process per run:
# two warm-ups and torch's default attention twice (the band), one warm-up, flash attention with two and with one.
# Then each run's waveforms against the first. Waits for block 48 first.
# usage: D=... T=<kit tree> MAIN=<tree> bash queue_block49.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/block49
L=$W/queue.log
mkdir -p $W
until grep -q "queue block48 done" $D/block48/queue.log 2> /dev/null; do sleep 20; done
echo "tree $MAIN $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $MAIN diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue block49 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
run() {
  local name=$1
  shift
  (cd $MAIN && PYTHONPATH=$MAIN timeout 900 python3 $K/decoder_graphs_probe.py --out $W/$name "$@" > $W/$name.log 2>&1)
  echo "$name exit $? $(tail -n 1 $W/$name.log | cut -c1-300)" >> $L
}
run default_2a --warmups 2 --attention default
run default_2b --warmups 2 --attention default
run default_1 --warmups 1 --attention default
run flash_2 --warmups 2 --attention flash
run flash_1 --warmups 1 --attention flash
(cd $MAIN && python3 - $W >> $L 2>&1 <<'PY'
import sys
import torch
directory = sys.argv[1]
reference = torch.load(f"{directory}/default_2a.pt")
for name in ("default_2b", "default_1", "flash_2", "flash_1"):
    other = torch.load(f"{directory}/{name}.pt")
    equal = sum(torch.equal(other[key], reference[key]) for key in reference)
    relative = max(float((other[key] - reference[key]).norm() / reference[key].norm()) for key in reference)
    print(f"{name} against default_2a: bit-equal {equal}/{len(reference)}, worst relative L2 {relative:.2e}")
PY
)
echo "queue block49 done $(date +%T)" >> $L
