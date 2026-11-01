#!/usr/bin/env bash
# Block 59b of 11-01: block 59's seeded identity pairs differ on 63 of 666 reference clips (every prompt of each), on
# main e354ea41f and not on 0e72eba6a (block 57), whose difference includes the reference encoder's compile. The
# reference encoder alone (refenc_codes.py, built as the engine builds it, every SeedTTS EN clip): card 0 into a fresh
# cache, again on that cache, into a second fresh cache; card 1 into a fresh cache and on card 0's cache; eager on both
# cards; codes compared per clip and against the served differences. Then main against main served (seeded_identity.py,
# cards 0 and 1, the servers' default caches as in block 59). Waits for block 59 first.
# usage: D=... T=<kit tree> MAIN=<tree> bash queue_block59b.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block59b
L=$W/queue.log
mkdir -p $W
until grep -q "queue block59 done" $D/block59/queue.log 2> /dev/null; do sleep 20; done
echo "tree MAIN $MAIN $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $MAIN diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue block59b start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
probe() {
  local name=$1 gpu=$2 cache=$3 flag=${4:-}
  (cd $MAIN && HF_HUB_OFFLINE=1 PYTHONPATH=$MAIN TORCHINDUCTOR_CACHE_DIR=$W/$cache/inductor TRITON_CACHE_DIR=$W/$cache/triton \
    timeout 1200 python3 $K/next_20261031/refenc_codes.py --gpu $gpu --out $W/$name $flag > $W/$name.log 2>&1)
  echo "$name exit $? $(date +%T): $(tail -n 1 $W/$name.log)" >> $L
}
probe card0_cacheA 0 cacheA
probe card0_cacheA_again 0 cacheA
probe card0_cacheB 0 cacheB
probe card1_cacheC 1 cacheC
probe card1_cacheA 1 cacheA
probe card0_eager 0 cacheE --eager
probe card1_eager 1 cacheE --eager
(cd $MAIN && HF_HUB_OFFLINE=1 PYTHONPATH=$MAIN python3 $K/next_20261031/refenc_codes.py --compare $W/card0_cacheA.pt \
  $W/card0_cacheA_again.pt $W/card0_cacheB.pt $W/card1_cacheC.pt $W/card1_cacheA.pt $W/card0_eager.pt $W/card1_eager.pt \
  --identity $D/block59/identity_codec > $W/compare.txt 2>&1)
grep -vE "INFO|WARN|Using the latest|Found the latest" $W/compare.txt >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $MAIN --branch $MAIN --root $W/identity_aa --port 40000 > $W/identity_aa.txt 2>&1)
echo "aa: $(grep -E 'byte identical|error lines' $W/identity_aa.txt | cut -c1-200 | tr '\n' '#')" >> $L
(cd $MAIN && HF_HUB_OFFLINE=1 PYTHONPATH=$MAIN python3 $K/next_20261031/identity_by_bucket.py --root $W/identity_aa 2>&1 | grep -vE "INFO|Using the latest|Found the latest" | head -n 3 >> $L)
echo "queue block59b done $(date +%T)" >> $L
