#!/usr/bin/env bash
# Block 117 of 11-03: the first-principles review of #3477. The cleanups move no value, so 1.7B reads byte identical to the
# PR's head in both modes or they stop. Long non-streaming requests beside streams on 1.7B, against main, the one scenario
# its bounded pieces exist for and never served. Then 0.6B, whose speech tokenizer is byte for byte 1.7B's, so a size gate
# has no decoder reason: seeded c1 against main, the same talker codes through both decodes, scored; non-streaming c64 and
# c16 against main in both card orders. The gate stays out if WER and SIM hold on the seeded read and speed gains.
# usage: D=... T=<kit tree> LAST=<tree> MAIN=<tree> NEW=<tree> HARNESS=<tree> [BLOCK=block117] [PORT=10000] bash queue_block117.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
LAST=${LAST:?the PR head}
MAIN=${MAIN:?main}
NEW=${NEW:?the tree under test}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block117}
PORT=${PORT:-10000}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
SMALL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
mkdir -p $W
for arm in LAST MAIN NEW; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $NEW && TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 2400 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts > $W/suite.log 2>&1)
echo "suite rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 16 | cut -c1-170 >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
(cd $K/next_20261031 && python3 seeded_identity.py --main $LAST --branch $NEW --root $W/identity --port $PORT > $W/identity.txt 2>&1)
echo "identity 1.7B rc $? $(date +%T): $(grep -E 'byte identical|error lines|Error|Traceback' $W/identity.txt | cut -c1-120 | tr '\n' '#')" >> $L
(cd $K/startup_20261031 && timeout 3000 python3 mixed_long.py --a main=$MAIN --b new=$NEW --harness $HARNESS --root $W/mixed_long --port $((PORT + 800)) > $W/mixed_long.txt 2>&1)
echo "mixed long rc $? $(date +%T)" >> $L
grep -E ' card [01]: stream done|^  [a-z]+: [0-9]+ of |Traceback|Error' $W/mixed_long.txt | cut -c1-300 >> $L
python3 -c "from huggingface_hub import snapshot_download as s; print(s('$SMALL'))" > $W/small_download.log 2>&1
echo "0.6B download rc $? $(date +%T)" >> $L
export BOX_TTS_MODEL=$SMALL
(cd $K/startup_20261031 && timeout 3000 python3 whole_seeded_pair.py --base $MAIN --arm $NEW --root $W/small_seeded --port $((PORT + 2000)) > $W/small_seeded.txt 2>&1)
echo "0.6B seeded rc $? $(date +%T)" >> $L
grep -E 'seeded c1|byte identical|c1n SIM|c1n WER|worst SIM|error lines|Traceback|Error' $W/small_seeded.txt | cut -c1-200 >> $L
port=$((PORT + 3000))
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 concurrency=$5 barrier=$6 cache=$7
  (cd $K/inline_20261026 && ARM_ROOT=$W/small_arms TORCHINDUCTOR_CACHE_DIR=$D/cache_$cache/inductor TRITON_CACHE_DIR=$D/cache_$cache/triton timeout 2400 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency $concurrency --no-stream --model $SMALL --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/small_arms/$label/startup.txt 2>/dev/null); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency mean|Latency p95' $W/small_arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-170); tracebacks $(grep -c Traceback $W/small_arms/$label/serve.log)" >> $L
}
for concurrency in 64 16; do
  run_arm main_c${concurrency}_o0 $MAIN 0 $port $concurrency $W/barrier_c${concurrency}_o0 main &
  run_arm new_c${concurrency}_o0 $NEW 1 $((port + 400)) $concurrency $W/barrier_c${concurrency}_o0 new &
  wait
  port=$((port + 800))
  run_arm main_c${concurrency}_o1 $MAIN 1 $port $concurrency $W/barrier_c${concurrency}_o1 main &
  run_arm new_c${concurrency}_o1 $NEW 0 $((port + 400)) $concurrency $W/barrier_c${concurrency}_o1 new &
  wait
  port=$((port + 800))
done
echo "queue $BLOCK done $(date +%T)" >> $L
