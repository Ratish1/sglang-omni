#!/usr/bin/env bash
# 10-17 heartbeat, run on the Mac: every 3 minutes one line with cards 2 and 3 (utilization, memory), the queue's last
# line and the pair's progress, or the ssh failure.
# usage: bash heartbeat_c20261017.sh <queue name, e.g. queue_b> <pair dir, e.g. p_a3>
Q=$1 P=$2
while true; do
  line=$(ssh -o ConnectTimeout=15 -o BatchMode=yes ratish1@85.234.79.62 "
    gpus=\$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' '\$1>=2{printf \"g%s %s%% %sMiB \", \$1, \$2, \$3}')
    queue=\$(tail -1 ~/c20261017/$Q.txt 2> /dev/null)
    cells=\$(ls -d ~/c20261017/$P/*/gpu*/* 2> /dev/null | wc -l)
    echo \"\$gpus| queue: \${queue:-not started} | pair cells: \$cells\"" 2>&1) \
    || line="SSH FAIL: $line"
  echo "$(date +%T) $line"
  case "$line" in *"queue done"*) exit 0 ;; esac
  sleep 180
done
