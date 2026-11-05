#!/usr/bin/env bash
# Block 132 of 11-05: where the illegal address comes from. Every scenario in its own process, once as is, and with CUDA_LAUNCH_BLOCKING=1
# when it fails, on the rows tree on card 0 and main on card 1, alive against torn window graphs, stream against reserved slot, width in place
# and past the rings, eager against launched, pinned against pageable.
# usage: D=... T=<kit tree> MAIN=<tree> ARM=<tree> [BLOCK=block132] bash queue_block132.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
ARM=${ARM:?the tree under test}
BLOCK=${BLOCK:-block132}
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
  echo "$name rc $rc $(date +%T) $(grep -a '^PROBE ' $W/$name.log | sed -E 's/^PROBE //' | python3 -c 'import json,sys; [print({k:v for k,v in json.loads(l).items() if k!="traceback"}) for l in sys.stdin]' 2>/dev/null | cut -c1-360)" >> $L
  if grep -aq '"outcome": "ok"' $W/$name.log; then
    :
  else
    (cd $tree && PYTHONPATH=$tree CUDA_VISIBLE_DEVICES=$card CUDA_LAUNCH_BLOCKING=1 timeout 900 python3 $S/eager_decode_probe.py "$@" > $W/${name}_blocking.log 2>&1)
    echo "$name blocking rc $? $(grep -a '^PROBE ' $W/${name}_blocking.log | sed -E 's/^PROBE //' | python3 -c 'import json,sys; [print(json.loads(l).get("outcome"), "|", json.loads(l).get("traceback","")[-700:].replace(chr(10)," / ")) for l in sys.stdin]' 2>/dev/null | cut -c1-900)" >> $L
  fi
}
(
  probe $ARM 0 rows_direct_alive_stream_64 --state alive --slot stream --frames 64 --path direct
  probe $ARM 0 rows_direct_torn_stream_64 --state torn --slot stream --frames 64 --path direct
  probe $ARM 0 rows_direct_alive_reserved_64 --state alive --slot reserved --frames 64 --path direct
  probe $ARM 0 rows_direct_torn_reserved_64 --state torn --slot reserved --frames 64 --path direct
  probe $ARM 0 rows_direct_alive_stream_8 --state alive --slot stream --frames 8 --path direct
  probe $ARM 0 rows_direct_torn_stream_8 --state torn --slot stream --frames 8 --path direct
  probe $ARM 0 rows_launch_torn_stream_64 --state torn --slot stream --frames 64 --path launch
  probe $ARM 0 rows_launch_torn_stream_64_pageable --state torn --slot stream --frames 64 --path launch --pageable
  probe $ARM 0 rows_launch_alive_stream_64_pageable --state alive --slot stream --frames 64 --path launch --pageable
  probe $ARM 0 rows_launch_torn_reserved_64_pageable --state torn --slot reserved --frames 64 --path launch --pageable
) &
(
  probe $MAIN 1 main_direct_alive_stream_64 --state alive --slot stream --frames 64 --path direct
  probe $MAIN 1 main_direct_torn_stream_64 --state torn --slot stream --frames 64 --path direct
  probe $MAIN 1 main_direct_alive_stream_8 --state alive --slot stream --frames 8 --path direct
  probe $MAIN 1 main_direct_torn_stream_8 --state torn --slot stream --frames 8 --path direct
  probe $MAIN 1 main_launch_torn_stream_64 --state torn --slot stream --frames 64 --path launch
  probe $MAIN 1 main_launch_torn_stream_64_pageable --state torn --slot stream --frames 64 --path launch --pageable
) &
wait
echo "queue $BLOCK done $(date +%T)" >> $L
