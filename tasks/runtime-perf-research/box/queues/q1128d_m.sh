#!/usr/bin/env bash
# 11-28 second lease, node-0001, after q1128d_l: DEV3 (dev/qwen3-omni-preprocessing-request-threads a83177b79, source tree
# 67ed98c669b2: dev2 plus a copy of the HF processor for every thread that runs it but one, made before serving).
#   1. The two new tests on the PR head (must fail), then test_pipeline.py and test_preprocessor.py whole on dev3.
#   2. Byte identity: main's serial stage against dev3 with 8 video threads, on 112 and on 16 CPUs (preprocess_shared_identity.py).
#   3. CI one card, CI Video-MME text c16 x2 from a fresh boot, DEV2 against DEV3, swapped: startup, preprocessing host
#      memory and speed.
#   4. Fused MMSU c96 and c16 with --preprocessing.factory.max_concurrency 4 (request threads take copies), MAIN against DEV3.
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main V=/workspace/dev2 W=/workspace/dev3 P=/workspace/prefix
DEV3_COMMIT=a83177b79
DEV3_TREE=67ed98c669b2
CPUS=56-111,168-223
SMALL=56-63,168-175
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((20000 + 4000 * $1 + $2)); }
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
run() {
  local tree=$1 cpus=$2 threads=$3; shift 3
  (cd $tree && PYTHONPATH=$tree OMP_NUM_THREADS=$threads taskset -c $cpus python3 "$@")
}
until grep -q "q1128d_l done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128d_m start"
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-preprocessing-request-threads
git worktree add -q --detach $W $DEV3_COMMIT
log "tree dev3 $(git -C $W rev-parse --short HEAD) $(git -C $W rev-parse --short=12 HEAD:sglang_omni) want $DEV3_TREE"
if [ "$(git -C $W rev-parse --short=12 HEAD:sglang_omni)" != "$DEV3_TREE" ]; then log "dev3 tree mismatch, stop"; exit 1; else :; fi

NEW="test_thread_processors_give_each_calling_thread_its_own_copy_then_the_shared_one or test_preprocessing_executor_copies_the_processor_for_every_calling_thread_but_one"
git -C $P checkout -q -- tests
cp $W/tests/unit_test/qwen3_omni/test_pipeline.py $P/tests/unit_test/qwen3_omni/test_pipeline.py
(cd $P && python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_omni/test_pipeline.py -k "$NEW" > $O/unit_prefix3.log 2>&1)
log "unit prefix3 (must fail) rc $? $(tail -1 $O/unit_prefix3.log)"
git -C $P checkout -q -- tests
(cd $W && python3 -c "import torch; torch.cuda.set_device(1)" && python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/qwen3_omni/test_pipeline.py tests/unit_test/qwen3_omni/test_preprocessor.py > $O/unit_dev3.log 2>&1)
log "unit dev3 rc $? $(tail -1 $O/unit_dev3.log)"

python3 -c "import numpy as np; from PIL import Image; Image.fromarray(np.random.default_rng(0).integers(0, 255, (480, 640, 3)).astype('uint8')).save('$O/id3.png')"
run $M $CPUS 112 $K/preprocess_shared_identity.py --threads 1 --image $O/id3.png --out $O/id3_main.json > $O/id3_main.log 2>&1
log "identity main rc $? $(grep -E '^serial' $O/id3_main.log | head -1)"
run $W $CPUS 112 $K/preprocess_shared_identity.py --threads 8 --image $O/id3.png --out $O/id3_dev3.json \
  --against $O/id3_main.json > $O/id3_dev3.log 2>&1
log "identity dev3 112 cpus rc $? $(grep -E 'differ' $O/id3_dev3.log | sed 's/.*: //' | tr '\n' ' ')"
run $W $SMALL 8 $K/preprocess_shared_identity.py --threads 8 --image $O/id3.png --out $O/id3_dev3_16.json \
  --against $O/id3_main.json > $O/id3_dev3_16.log 2>&1
log "identity dev3 16 cpus rc $? $(grep -E 'differ' $O/id3_dev3_16.log | sed 's/.*: //' | tr '\n' ' ')"

speed() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "CI one card, text c16 x2 from a fresh boot, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card $card \
    --port $(port_of $card $offset) --out $O/$name --bench videomme_text --loads 16 --passes clean --bench-repeats 2 \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(cat $O/$name/startup.txt 2>/dev/null | tr '\n' ' ' | cut -c1-120)"
}
both "speed s3a_v $V 0 1000" "speed s3a_w $W 1 1000"
both "speed s3b_w $W 0 1200" "speed s3b_v $V 1 1200"
log "dev3 speed read"

mmsu() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "fused text pipeline, MMSU c96 and c16, max_concurrency 4, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_text_fused --card $card \
    --port $(port_of $card $offset) --out $O/$name --serve-extra "--preprocessing.factory.max_concurrency 4" \
    --bench mmsu --loads 96,16 --passes clean > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(grep -m1 'Throughput:' $O/$name/c96_clean/bench.log 2>/dev/null)"
}
both "mmsu m3a_m $M 0 1400" "mmsu m3a_w $W 1 1400"
both "mmsu m3b_w $W 0 1600" "mmsu m3b_m $M 1 1600"
log "dev3 mmsu read"
log "q1128d_m done"
