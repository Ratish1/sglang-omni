#!/usr/bin/env bash
# Block 63 of 11-01: the reference encoder fix's A/B (block 61) is level, but its streaming runs had 5 and 6 requests
# below SIM 40 against main's 2 and 3. The low prompts' reference codes equal main's compiled states in the encoder
# probes. Seeded at concurrency 1, main (card 0) against the fix (card 1), non-streaming and streaming: every file by
# SHA-256, the differing files grouped by reference clip, and the 11 prompts the fix had below 40 checked one by one.
# Waits for block 62.
# usage: D=... T=<kit tree> MAIN=<tree> FIX=<tree> bash queue_block63.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
FIX=${FIX:?the fix tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block63
L=$W/queue.log
mkdir -p $W
until grep -q "queue block62 done" $D/block62/queue.log 2> /dev/null; do sleep 20; done
for arm in MAIN FIX; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block63 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $MAIN --branch $FIX --root $W/identity --port 20000 > $W/identity.txt 2>&1)
echo "identity: $(grep -E 'byte identical|error lines|Error' $W/identity.txt | cut -c1-120 | tr '\n' '#')" >> $L
(cd $MAIN && HF_HUB_OFFLINE=1 PYTHONPATH=$MAIN python3 $K/next_20261031/identity_by_bucket.py --root $W/identity 2>&1 | grep -vE "INFO|Using the latest|Found the latest" | head -n 12 >> $L)
python3 - $W/identity >> $L 2>&1 <<'EOF'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
low = ["15735839-common_voice_en_15735837", "17369337-common_voice_en_17369336", "17406452-common_voice_en_17406455",
       "17861489-common_voice_en_17861487", "19599129-common_voice_en_19599130", "19670674-common_voice_en_19670675",
       "21538043-common_voice_en_21538044", "22575787-common_voice_en_22575799", "23733663-common_voice_en_23733664",
       "29495818-common_voice_en_29495898", "31499885-common_voice_en_31499889"]
for run in ("c1n", "c1s"):
    for prompt in low:
        name = f"common_voice_en_{prompt}.wav"
        first = hashlib.sha256((root / run / "main" / "bench" / "audio" / name).read_bytes()).hexdigest()
        second = hashlib.sha256((root / run / "branch" / "bench" / "audio" / name).read_bytes()).hexdigest()
        print(f"{run} {prompt}: {'identical' if first == second else 'DIFFERENT'}")
EOF
echo "queue block63 done $(date +%T)" >> $L
