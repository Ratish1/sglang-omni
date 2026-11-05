#!/usr/bin/env bash
# Block 134 of 11-05: does tearing one runner down free the cuBLAS workspace its siblings' graphs read. Every scenario in its own process, with
# CUDA_LAUNCH_BLOCKING=1 when it fails, on main on card 0 and the rows tree on card 1. Controls that should each remove the fault if so: no empty_cache,
# a capture stream per runner, a warm replay on its own stream. Direct: free the workspaces with every graph alive. Reachable: a window capture failing partway.
# usage: D=... T=<kit tree> MAIN=<tree> ARM=<tree> [BLOCK=block134] bash queue_block134.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
ARM=${ARM:?the tree under test}
BLOCK=${BLOCK:-block134}
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
  local rc=$?
  echo "$name rc $rc $(date +%T) $(grep -a '^PROBE ' $W/$name.log | sed -E 's/^PROBE //' | python3 -c 'import json,sys; [print({k:v for k,v in json.loads(l).items() if k in ("outcome","steps_done","footprint_mib","reserved_mib","window_enabled_after")}) for l in sys.stdin]' 2>/dev/null | cut -c1-300)" >> $L
  if grep -aq '"outcome": "ok"' $W/$name.log; then
    :
  else
    (cd $tree && PYTHONPATH=$tree CUDA_VISIBLE_DEVICES=$card CUDA_LAUNCH_BLOCKING=1 timeout 900 python3 $S/eager_decode_probe.py "$@" > $W/${name}_blocking.log 2>&1)
    echo "$name blocking rc $? $(grep -a '^PROBE ' $W/${name}_blocking.log | sed -E 's/^PROBE //' | python3 -c 'import json,sys; [print(json.loads(l).get("outcome"), "|", json.loads(l).get("traceback","")[-420:].replace(chr(10)," / ")) for l in sys.stdin]' 2>/dev/null | cut -c1-620)" >> $L
  fi
}
(
  probe $MAIN 0 main_cold_alive --state alive --slot stream --frames 2 --path launch
  probe $MAIN 0 main_cold_torn --state torn --slot stream --frames 2 --path launch
  probe $MAIN 0 main_cold_torn_no_empty_cache --state torn --slot stream --frames 2 --path launch --no-empty-cache
  probe $MAIN 0 main_cold_torn_own_capture_streams --state torn --slot stream --frames 2 --path launch --own-capture-streams
  probe $MAIN 0 main_cold_clear --state clear --slot stream --frames 2 --path launch
  probe $MAIN 0 main_cold_capture_fail --state capture_fail --slot stream --frames 2 --path launch
  probe $MAIN 0 main_tail_torn --state torn --slot stream --frames 40 --path tail
  probe $MAIN 0 main_tail_alive --state alive --slot stream --frames 40 --path tail
  probe $MAIN 0 main_warm_torn --state torn --slot stream --frames 8 --path warm
) &
(
  probe $ARM 1 rows_cold_alive --state alive --slot stream --frames 2 --path launch
  probe $ARM 1 rows_cold_torn --state torn --slot stream --frames 2 --path launch
  probe $ARM 1 rows_cold_torn_own_capture_streams --state torn --slot stream --frames 2 --path launch --own-capture-streams
  probe $ARM 1 rows_cold_clear --state clear --slot stream --frames 2 --path launch
  probe $ARM 1 rows_cold_capture_fail --state capture_fail --slot stream --frames 2 --path launch
) &
wait
echo "queue $BLOCK done $(date +%T)" >> $L
