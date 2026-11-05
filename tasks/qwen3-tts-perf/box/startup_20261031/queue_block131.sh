#!/usr/bin/env bash
# Block 131 of 11-05: whole requests as their own rows, the final read against main and against the previous head. Suite. Real
# codec with a replay failed. Seeded identity against the last head, and against main for streams. Non-streaming c64 and c16 and
# streaming c64 against main, both orders. First chunks beside 8 long clients against main, two rounds of both orders, and beside
# 64 long clients against the last head. 0.6B non-streaming c64 and c16 against main, and its seeded bytes against the last head.
# CustomVoice and VoiceDesign by block 94's driver.
# usage: D=... T=<kit tree> MAIN=<tree> LAST=<tree> ARM=<tree> HARNESS=<tree> [BLOCK=block131] [PORT=4000] bash queue_block131.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
LAST=${LAST:?the last head}
ARM=${ARM:?the tree under test}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block131}
PORT=${PORT:-4000}
K=$T/tasks/qwen3-tts-perf/box
S=$K/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
SMALL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
mkdir -p $W
for arm in MAIN LAST ARM; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $ARM && TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 2400 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts tests/unit_test/pipeline/test_stage.py > $W/suite.log 2>&1)
echo "suite rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 16 | cut -c1-170 >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
export CARD_CPUS="0=0-31,64-95;1=32-63,96-127"
for cases in "0 1 2" 3 4; do
  name=fault_$(echo $cases | tr -d ' ')
  (cd $ARM && PYTHONPATH=$ARM CUDA_VISIBLE_DEVICES=0 timeout 1500 python3 $S/whole_fault_read.py --only-case $cases > $W/$name.log 2>&1)
  echo "$name rc $? $(date +%T)" >> $L
  grep -E '^\{"(stream_slots|case)"' $W/$name.log | cut -c1-280 >> $L
done
(cd $K/next_20261031 && python3 seeded_identity.py --main $LAST --branch $ARM --root $W/identity_last --port $PORT > $W/identity_last.txt 2>&1)
echo "identity last rc $? $(date +%T): $(grep -E 'byte identical|error lines|Error|Traceback' $W/identity_last.txt | cut -c1-120 | tr '\n' '#')" >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $MAIN --branch $ARM --root $W/identity_main --port $((PORT + 800)) > $W/identity_main.txt 2>&1)
echo "identity main rc $? $(date +%T): $(grep -E 'byte identical|error lines|Error|Traceback' $W/identity_main.txt | cut -c1-120 | tr '\n' '#')" >> $L
port=$((PORT + 1600))
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5 concurrency=$6 flags=$7 model=$8
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py --label $label --tree $tree --card $card --port $port --concurrency $concurrency $flags --model $model --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency p95|TTFC mean|TTFC p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-190); tokens $(grep -m1 -oE '#tokens: [0-9]+' $W/arms/$label/serve.log); memory $(cut -c1-60 $W/arms/$label/memory.txt 2>/dev/null); tracebacks $(grep -c Traceback $W/arms/$label/serve.log 2>/dev/null) left context $(grep -c left-context $W/arms/$label/serve.log 2>/dev/null)" >> $L
}
pair() {
  local name=$1 concurrency=$2 flags=$3 model=$4
  run_arm main_${name}_o0 $MAIN 0 $port $W/barrier_${name}_o0 $concurrency "$flags" $model &
  run_arm head_${name}_o0 $ARM 1 $((port + 400)) $W/barrier_${name}_o0 $concurrency "$flags" $model &
  wait
  port=$((port + 800))
  run_arm main_${name}_o1 $MAIN 1 $port $W/barrier_${name}_o1 $concurrency "$flags" $model &
  run_arm head_${name}_o1 $ARM 0 $((port + 400)) $W/barrier_${name}_o1 $concurrency "$flags" $model &
  wait
  port=$((port + 800))
}
LARGE=Qwen/Qwen3-TTS-12Hz-1.7B-Base
pair n64 64 --no-stream $LARGE
pair n16 16 --no-stream $LARGE
pair s64 64 "" $LARGE
for round in r1 r2; do
  (cd $S && timeout 2400 python3 mixed_long.py --a main=$MAIN --b head=$ARM --harness $HARNESS --root $W/mixed_$round --port $port --long-requests 32 > $W/mixed_$round.txt 2>&1)
  echo "mixed $round rc $? $(date +%T)" >> $L
  grep -vE "^(o1|o2) (main|head): card" $W/mixed_$round.txt | cut -c1-300 >> $L
  port=$((port + 800))
done
(cd $S && timeout 2400 python3 mixed_long.py --a last=$LAST --b head=$ARM --harness $HARNESS --root $W/mixed_burst --port $port --long-concurrency 64 --long-requests 192 > $W/mixed_burst.txt 2>&1)
echo "mixed burst rc $? $(date +%T)" >> $L
grep -vE "^(o1|o2) (last|head): card" $W/mixed_burst.txt | cut -c1-300 >> $L
port=$((port + 800))
pair small_n64 64 --no-stream $SMALL
pair small_n16 16 --no-stream $SMALL
(cd $S && BOX_TTS_MODEL=$SMALL timeout 3000 python3 whole_seeded_pair.py --base $LAST --arm $ARM --root $W/small_seed --port $port --seed 1234 --no-streaming > $W/small_seed.txt 2>&1)
echo "small seeded rc $? $(date +%T)" >> $L
grep -E 'byte identical|SIM over|WER over|Traceback' $W/small_seed.txt | cut -c1-230 >> $L
D=$W T=$T PR=$MAIN WHOLE=$ARM bash $S/queue_block94.sh > $W/voice.out 2>&1
echo "voice rc $? $(date +%T)" >> $L
cut -c1-300 $W/block94/queue.log >> $L
echo "queue $BLOCK done $(date +%T)" >> $L
