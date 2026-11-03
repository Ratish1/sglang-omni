#!/usr/bin/env bash
# 11-03, node-0001, the second H200, beside q1103_marks on node-0000. #3452 after its review fixes (4e8763b4a): suites, the
# new context test on main's sources, the serial test on a mutant; #3451's test fix (397bbbe39). Then #3452's first read by
# request state on the colocated config, boots m w w m, c64 and c16, warm and events. Then #3451 under the fine marks on
# the setup where its first audio cost shows (h100_bf16 config, seed 3451, warmup 16), boots m p p m, NVTX over 40 s.
set -u
O=/data/rp1103w
K=/workspace/kit_w/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 WK=/workspace/wk_4e8763b4a P3F=/workspace/p3fix_397bbbe39 M=/workspace/marks_main P=/workspace/marks_3p
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
export CUDA_DEVICE_ORDER=PCI_BUS_ID
MARKS_ENV="SGLANG_ENABLE_NVTX_SCHEDULER=1 SGLANG_ENABLE_NVTX_OPERATIONS=1"
mkdir -p $O/units
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
wait_card() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do
    log "waiting, card used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 30
  done
}
log "q1103_wk start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
cd $WK
taskset -c 0-55 python3 -m pytest -q tests/unit_test/qwen3_omni tests/unit_test/scheduling/test_threaded_simple_scheduler.py \
  tests/unit_test/pipeline/test_stage.py tests/unit_test/pipeline/test_scheduler.py -p no:cacheprovider > $O/units/units_wk.log 2>&1
log "suites workers rc $?"
mkdir -p /tmp/ctx_on_main && cp $WK/tests/unit_test/scheduling/test_threaded_simple_scheduler.py /tmp/ctx_on_main/
cd $MAIN
taskset -c 0-55 python3 -m pytest -q /tmp/ctx_on_main/test_threaded_simple_scheduler.py -k bound_to_the_scheduler_thread -p no:cacheprovider \
  --rootdir=$MAIN > $O/units/context_test_on_main.log 2>&1
log "context test on main rc $? (1 expected)"
rm -rf /tmp/wk_mutant && git -C $WK worktree add -q -f /tmp/wk_mutant HEAD
sed -i 's/if max_concurrency <= 1:/if max_concurrency <= 0:/; s/for _ in range(max_concurrency):/for _ in range(2):/; s/        max_concurrency=max_concurrency,\n        thread_initializer/&/' /tmp/wk_mutant/sglang_omni/models/qwen3_omni/stages.py
sed -i '/preprocess_on_thread,/{n;s/max_concurrency=max_concurrency/max_concurrency=2/}' /tmp/wk_mutant/sglang_omni/models/qwen3_omni/stages.py
git -C /tmp/wk_mutant diff --stat > $O/units/serial_test_mutant.log 2>&1
cd /tmp/wk_mutant
taskset -c 0-55 python3 -m pytest -q tests/unit_test/qwen3_omni/test_pipeline.py -k one_worker_runs_requests_one_at_a_time -p no:cacheprovider >> $O/units/serial_test_mutant.log 2>&1
log "serial test on the two worker mutant rc $? (1 expected)"
cd $P3F
taskset -c 0-55 python3 -m pytest -q tests/unit_test/qwen3_omni/test_logit_shaping.py -p no:cacheprovider > $O/units/units_3p_testfix.log 2>&1
log "3p test file rc $?"

boot() {
  local name=$1 tree=$2; shift 2
  wait_card
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) load $(cut -d' ' -f1 /proc/loadavg) start"
  python3 $K/baseline.py --tree $tree --card 0 --port 26800 --out $O/$name "$@" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
A="--model omni_colocated --loads 64,16 --passes warm,events"
boot fill_a $MAIN --model omni_colocated --loads 16 --passes warm
boot wk_m1 $MAIN $A
boot wk_w1 $WK $A
boot wk_w2 $WK $A
boot wk_m2 $MAIN $A
log "workers by state done"
T="--model omni_colocated_h100_config --loads 16 --passes warm,nsys --nsys-trace nvtx --nsys-delay 15 --nsys-seconds 40"
R="--bench-extra=--seed 3451 --warmup 16"
boot fill_b $MAIN --model omni_colocated_h100_config --loads 16 --passes warm
boot tb_m1 $M $T --nsys-session tb_m1 --serve-env "$MARKS_ENV" "$R"
boot tb_p1 $P $T --nsys-session tb_p1 --serve-env "$MARKS_ENV" "$R"
boot tb_p2 $P $T --nsys-session tb_p2 --serve-env "$MARKS_ENV" "$R"
boot tb_m2 $M $T --nsys-session tb_m2 --serve-env "$MARKS_ENV" "$R"
cd $O
taskset -c 0-55 python3 $K/talker_loop.py tb_m1,tb_m2 tb_p1,tb_p2 > $O/read_tb.txt 2>&1
log "q1103_wk done"
