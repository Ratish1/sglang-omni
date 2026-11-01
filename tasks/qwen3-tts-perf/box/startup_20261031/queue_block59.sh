#!/usr/bin/env bash
# Block 59 of 11-01: the three PRs as they now stand (#3403 narrowed to the opaque ops, each branch with upstream main
# merged) held to byte identity against that main: seeded_identity.py per PR, concurrency 1, seed 1234, non-streaming
# then streaming, every request's audio by SHA-256. Then the warm-up PR's codec captures on main's codec form
# (codec_forms.py, card 0, one cache per tree filled cold, then warm main, PR, PR, main: capture seconds, compiled
# replay times, bits per key against the first main). Then the warm-up PR's whole head against main on the standard
# SeedTTS A/B, and main against main on the same A/B, the same-node band every PR's rows are read against.
# usage: D=... T=<kit tree> MAIN=<tree> CODEC=<tree> IMPORTS=<tree> WARMUPS=<tree> bash queue_block59.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
CODEC=${CODEC:?the codec PR tree}
IMPORTS=${IMPORTS:?the imports PR tree}
WARMUPS=${WARMUPS:?the warm-ups PR tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block59
L=$W/queue.log
mkdir -p $W
for arm in MAIN CODEC IMPORTS WARMUPS; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block59 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
port=26000
for arm in CODEC IMPORTS WARMUPS; do
  (cd $K/next_20261031 && python3 seeded_identity.py --main $MAIN --branch ${!arm} --root $W/identity_${arm,,} --port $port > $W/identity_${arm,,}.txt 2>&1)
  echo "${arm,,}: $(grep -E 'byte identical|error lines' $W/identity_${arm,,}.txt | tr '\n' '#')" >> $L
  port=$((port + 1000))
done
forms() {
  local name=$1 tree=$2 cache=$3 phase=$4
  (cd $tree && PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/forms/cache_$cache/inductor TRITON_CACHE_DIR=$W/forms/cache_$cache/triton \
    timeout 1800 python3 $K/startup_20261031/codec_forms.py --out $W/forms/${name}_$phase > $W/forms/${name}_$phase.log 2>&1)
  echo "forms $name $phase exit $? $(date +%T)" >> $L
}
mkdir -p $W/forms
forms main_a $MAIN main cold
forms warmups_a $WARMUPS warmups cold
forms main_a $MAIN main warm
forms warmups_a $WARMUPS warmups warm
forms warmups_b $WARMUPS warmups warm
forms main_b $MAIN main warm
python3 $K/startup_20261031/compare_forms.py $W/forms main_a main_b warmups_a warmups_b > $W/forms/compare.txt 2>&1
head -n 5 $W/forms/compare.txt >> $L
(cd $K/next_20261031 && python3 ab_runs.py --a main=$MAIN --b warmups=$WARMUPS --root $W/ab_warmups --port 29000 > $W/ab_warmups.txt 2>&1)
echo "ab_warmups: $(grep -E '^(n16|s16)_o[12]|paired' $W/ab_warmups.txt | cut -c1-400 | tr '\n' '#')" >> $L
(cd $K/next_20261031 && python3 ab_runs.py --a maina=$MAIN --b mainb=$MAIN --root $W/band --port 34000 > $W/band.txt 2>&1)
echo "band: $(grep -E '^(n16|s16)_o[12]|paired' $W/band.txt | cut -c1-400 | tr '\n' '#')" >> $L
echo "queue block59 done $(date +%T)" >> $L
