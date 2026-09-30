#!/usr/bin/env bash
# Host side DCGM counter sampler for one GPU: every line gets the host epoch in front.
# DCGM runs as root on the node and reads the counters the driver keeps from users
# (RmProfilingAdminOnly), so this works where nsys --gpu-metrics and ncu are refused.
# usage: dcgm_sample.sh <dcgm gpu id> <interval ms> <out file>
set -u
GPU=$1 MS=$2 OUT=$3
FIELDS=1001,1002,1003,1004,1005,1007,1008,1014,1016
dcgmi dmon -e "$FIELDS" -i "$GPU" -d "$MS" 2>&1 | while IFS= read -r line; do
  printf '%s %s\n' "$(date +%s.%N)" "$line"
done > "$OUT"
