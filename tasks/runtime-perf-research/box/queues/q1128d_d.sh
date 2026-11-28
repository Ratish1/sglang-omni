#!/usr/bin/env bash
# 11-28 second lease, node-0001, after q1128d_c: DEV (the PR head plus max_concurrency threads for requests without
# video again, dev/qwen3-omni-preprocessing-request-threads 296082f50, source tree 8ddf8f6b6c24) against MAIN 0f70d9c7c.
#   1. The new dispatch tests on the PR head (copied onto a worktree at 91c2e2010), which must fail there, then the
#      preprocessing scheduler and video lane tests of test_pipeline.py on the dev tree. CPU only, between cells.
#   2. MMSU (2,000 questions) on the fused text pipeline, c96 and c16, same-time pairs swapped: dev with
#      --preprocessing.factory.max_concurrency 4 against main with it, then dev default against main default.
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main D=/workspace/dev P=/workspace/prefix
DEV_COMMIT=296082f50
DEV_TREE=8ddf8f6b6c24
TESTS=tests/unit_test/qwen3_omni/test_pipeline.py
NEW_TESTS="test_preprocessing_executor_threads_requests_without_video_above_one or test_preprocessing_request_threads_keep_results_errors_and_running_abort"
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
until grep -q "q1128d_c done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128d_d start"
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-preprocessing-request-threads
git worktree add -q --detach $D $DEV_COMMIT
git worktree add -q --detach $P 91c2e2010bed
log "tree dev $(git -C $D rev-parse --short HEAD) $(git -C $D rev-parse --short=12 HEAD:sglang_omni) want $DEV_TREE"
if [ "$(git -C $D rev-parse --short=12 HEAD:sglang_omni)" != "$DEV_TREE" ]; then log "dev tree mismatch, stop"; exit 1; else :; fi

cp $D/$TESTS $P/$TESTS
cd $P && python3 -m pytest -q -p no:cacheprovider $TESTS -k "$NEW_TESTS" > $O/unit_prefix.log 2>&1
log "unit prefix (must fail) rc $? $(tail -1 $O/unit_prefix.log)"
cd $D && python3 -m pytest -q -p no:cacheprovider $TESTS -k "preprocessing or video_lane or read_request_inputs" > $O/unit_dev.log 2>&1
log "unit dev rc $? $(tail -1 $O/unit_dev.log)"

mmsu() {
  local name=$1 tree=$2 card=$3 offset=$4 extra=$5
  note $name $tree "fused text pipeline, MMSU c96 and c16, $extra, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_text_fused --card $card \
    --port $(port_of $card $offset) --out $O/$name --serve-extra "$extra" --bench mmsu --loads 96,16 --passes clean \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(grep -m1 'Throughput:' $O/$name/c96_clean/bench.log 2>/dev/null)"
}
OPT="--preprocessing.factory.max_concurrency 4"
both "mmsu mo1_m $M 0 3500 '$OPT'" "mmsu mo1_v $D 1 3500 '$OPT'"
both "mmsu mo2_v $D 0 3700 '$OPT'" "mmsu mo2_m $M 1 3700 '$OPT'"
both "mmsu md1_m $M 0 3900 ''" "mmsu md1_v $D 1 3900 ''"
both "mmsu md2_v $D 0 3800 ''" "mmsu md2_m $M 1 3800 ''"
log "mmsu dev read"
log "q1128d_d done"
