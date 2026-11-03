#!/usr/bin/env bash
# Block 116 of 11-03: why the joined predictor read +1.6, +1.5 % at c64 against a written +5 %. One SeedTTS c64 census of the
# joined head under nsys, read against block 103's seed_c64, then join_ledger.py: the share of prefill rows that join, what a
# joined step costs against a decode and a prefill apart, what the empty poll costs. Decides fix, widen to the loop top, or
# drop. Then the touched suites on the same head, which holds every test double of the branch below it.
# usage: D=... T=<kit tree> TREE=<tree> [BLOCK=block116] [CARD=0] [PORT=14000] bash queue_block116.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
TREE=${TREE:?the tree}
BLOCK=${BLOCK:-block116}
CARD=${CARD:-0}
PORT=${PORT:-14000}
S=$T/tasks/omni_step_profiling_20260918/scripts
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
echo "tree $TREE $(git -C $TREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $TREE diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
out=$W/seed_c64
PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py \
  NSYS_ARGS="--trace=cuda,nvtx --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
  bash $S/run_nsys_boot.sh $TREE $out $CARD $PORT 64 640 > $out.out 2>&1
echo "seed_c64 rc $? $(date +%T); probe patch failures $(grep -c 'pipeline_nvtx patch' $out/serve.log); $(grep -hE 'RTF mean|Throughput \(req/s\)|TTFC mean' $out/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-140)" >> $L
(cd $S && python3 join_ledger.py $out/serve.sqlite --bench-log $out/bench.log > $W/join_ledger.txt 2>&1)
echo "ledger rc $?" >> $L
cat $W/join_ledger.txt | cut -c1-170 >> $L
(cd $TREE && TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 3000 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts tests/unit_test/pipeline tests/unit_test/scheduling > $W/suite.log 2>&1)
echo "suite rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 24 | cut -c1-170 >> $L
echo "queue $BLOCK done $(date +%T)" >> $L
