#!/usr/bin/env bash
# Block 135 of 11-05: a capture failure at startup with no later capture to re-allocate the freed workspace, on main. The suite and the probes on the rows tree.
# usage: D=... T=<kit tree> MAIN=<tree> ARM=<tree> [BLOCK=block135] bash queue_block135.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
ARM=${ARM:?the tree under test}
BLOCK=${BLOCK:-block135}
S=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ') arm $(git -C $ARM rev-parse HEAD HEAD:sglang_omni | tr '\n' ' '); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
probe() {
  local tree=$1 card=$2 name=$3
  shift 3
  (cd $tree && PYTHONPATH=$tree CUDA_VISIBLE_DEVICES=$card timeout 900 python3 $S/eager_decode_probe.py "$@" > $W/$name.log 2>&1)
  echo "$name rc $? $(date +%T) $(grep -a '^PROBE ' $W/$name.log | sed -E 's/^PROBE //' | python3 -c 'import json,sys; [print({k:v for k,v in json.loads(l).items() if k in ("outcome","steps_done","footprint_mib","enabled","step0_rms")}) for l in sys.stdin]' 2>/dev/null | cut -c1-320)" >> $L
}
(
  probe $MAIN 0 main_cold_capture_fail_no_tail --state capture_fail --slot stream --frames 2 --path launch --no-tail
  probe $MAIN 0 main_cold_tail_capture_fail --state capture_fail --fail-runner tail --slot stream --frames 2 --path launch
  probe $MAIN 0 main_window_tail_capture_fail --state capture_fail --fail-runner tail --slot stream --frames 64 --path launch
) &
(
  (cd $ARM && TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton CUDA_VISIBLE_DEVICES=1 timeout 2400 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts tests/unit_test/pipeline/test_stage.py > $W/suite.log 2>&1)
  echo "suite rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1 | cut -c1-90)" >> $L
  grep -E "^FAILED|^ERROR" $W/suite.log | head -n 16 | cut -c1-170 >> $L
) &
wait
echo "queue $BLOCK done $(date +%T)" >> $L
