#!/usr/bin/env bash
# The serving transient by vocoder step: one server with the count probe's memory fields over the
# heaviest cells (stream c64, buffered c32, long form c16), GPU core dumps armed.
# usage (in the container): bash memory_run.sh <tree> <card> <port> <out> [pool GB]
#   ALLOC_CONF  PYTORCH_CUDA_ALLOC_CONF for the server, e.g. expandable_segments:True
TREE=$1
CARD=$2
PORT=$3
OUT=$4
POOL_GB=${5:-}
CPUS=$([ "$CARD" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95)
PROBE=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4/prefix_count
export EXTRA_ENV="PYTHONPATH=/workspace/$TREE:$PROBE OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1 \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
CUDA_COREDUMP_FILE=/data/p20261008/cores/core_%h_%p_%t${ALLOC_CONF:+ PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF}"
export CELLS="s64:64:::1 b32:32:::0 L16:16:::1:/data/p20261008/longform/meta.lst"
export SERVE_ARGS=${POOL_GB:+--vocoder.factory.flow_prefix_cache_gb $POOL_GB}
bash /workspace/tools/tasks/cosyvoice_utilization_20260912/stage4/ab_boot.sh "/workspace/$TREE" "$OUT" \
  "$CARD" "$PORT" "$CPUS" "/data/p20261008/inductor_$TREE"
