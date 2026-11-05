#!/usr/bin/env bash
# Block 136 of 11-05: every runner on a capture stream torn down together. The new tests against main's source must fail, the suite
# on the fix must pass. The probes that faulted on main, on the fix: cold and tail after the window went down, capture failures,
# and the warm runner on its own stream staying up. A real server through one failed replay of each runner, main on card 0, the
# fix on card 1, in the same order.
# usage: D=... T=<kit tree> MAIN=<tree> FIX=<tree> PREFIX=<fix's tests on main's source> HARNESS=<tree> [BLOCK=block136] [PORT=4100] bash queue_block136.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
FIX=${FIX:?the fix}
PREFIX=${PREFIX:?the fix tests on main source}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block136}
PORT=${PORT:-4100}
S=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
for arm in MAIN FIX PREFIX; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree write-tree 2>/dev/null | cut -c1-9)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $PREFIX && timeout 600 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_incremental_codec_cuda_graph.py -k "every_runner_on_its_pool" > $W/prefix.log 2>&1)
echo "prefix rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/prefix.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED|^ERROR" $W/prefix.log | cut -c1-170 >> $L
(cd $FIX && TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 2400 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts tests/unit_test/pipeline/test_stage.py > $W/suite.log 2>&1)
echo "suite rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 16 | cut -c1-170 >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
probe() {
  local tree=$1 card=$2 name=$3
  shift 3
  (cd $tree && PYTHONPATH=$tree CUDA_VISIBLE_DEVICES=$card timeout 900 python3 $S/eager_decode_probe.py "$@" > $W/$name.log 2>&1)
  local rc=$?
  echo "$name rc $rc $(date +%T) $(grep -a '^PROBE ' $W/$name.log | sed -E 's/^PROBE //' | python3 -c 'import json,sys; [print({k:v for k,v in json.loads(l).items() if k in ("outcome","steps_done","enabled","enabled_after","reserved_mib","reserved_mib_after","tail_missed","step0_rms","step2_rms")}) for l in sys.stdin]')" >> $L
}
fault() {
  local tree=$1 card=$2 port=$3 name=$4 mode=$5
  (cd $HARNESS && timeout 1500 python3 $S/graph_fault_read.py --tree $tree --harness $HARNESS --card $card --port $port --mode $mode --replay 3 --out $W/$name > $W/$name.txt 2>&1)
  echo "$name rc $? $(date +%T)" >> $L
  grep -aE '^(ROUND|FAULT) ' $W/$name.txt | cut -c1-1200 >> $L
}
probe $FIX 1 fix_cold_alive --state alive --slot stream --frames 2 --path launch
probe $FIX 1 fix_cold_torn --state torn --slot stream --frames 2 --path launch
probe $FIX 1 fix_window_torn --state torn --slot stream --frames 64 --path launch
probe $FIX 1 fix_tail_torn --state torn --slot stream --frames 40 --path tail
probe $FIX 1 fix_warm_torn --state torn --slot stream --frames 8 --path warm
probe $FIX 1 fix_capture_fail_window --state capture_fail --slot stream --frames 2 --path launch
probe $FIX 1 fix_capture_fail_tail --state capture_fail --fail-runner tail --slot stream --frames 2 --path launch
for mode in tail window cold warm; do
  fault $MAIN 0 $PORT main_fault_$mode $mode &
  fault $FIX 1 $((PORT + 300)) fix_fault_$mode $mode &
  wait
  PORT=$((PORT + 600))
done
echo "queue $BLOCK done $(date +%T)" >> $L
