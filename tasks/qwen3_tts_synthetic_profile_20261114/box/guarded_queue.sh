#!/usr/bin/env bash
# guarded_queue.sh <card> <head>...: on the host, runs each head with run_head.sh once no other
# session holds a GPU, and repeats a head whose Triton reference at 1 row is slower than
# TRITON_CEILING_US (135 on H200, where a lone card measures 129 to 131; 145 on H100, 139).
CARD=$1
shift
CEILING=${TRITON_CEILING_US:-135}
C=sglang-omni-qwen3-tts-predictor-20261114
R=$HOME/qwen3-tts-synthetic-20261114/predictor_bench
BOX=/data/qwen3-tts-synthetic-20261114/kit/tasks/qwen3_tts_synthetic_profile_20261114/box
OURS="persistent_layer_check|persistent_stamps|persistent_ncu|predictor_bench|latency_probe|small_gemm_probe"
foreign() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    ps -o cmd= -p $p | grep -qE "$OURS" || echo $p
  done
}
for HEAD in "$@"; do
  for attempt in 1 2 3 4 5 6; do
    until [ -z "$(foreign)" ]; do sleep 15; done
    echo "$(date +%H:%M:%S) start $HEAD card $CARD attempt $attempt"
    rm -f $R/h200_${HEAD}_check.* $R/h200_${HEAD}_stamps.* $R/h200_${HEAD}.done
    podman exec $C bash $BOX/run_head.sh $HEAD $CARD
    TRITON=$(podman exec $C python3 -c "import json; print(json.load(open('/data/qwen3-tts-synthetic-20261114/predictor_bench/h200_${HEAD}_check.json'))['1']['triton 5 layers us'])" 2>/dev/null)
    echo "$(date +%H:%M:%S) done $HEAD triton reference ${TRITON:-missing}"
    [ -z "$TRITON" ] && break
    python3 -c "import sys; sys.exit(0 if float('$TRITON') <= $CEILING else 1)" && break
    echo "$(date +%H:%M:%S) contended, repeating $HEAD"
  done
done
echo "$(date +%H:%M:%S) queue done"
