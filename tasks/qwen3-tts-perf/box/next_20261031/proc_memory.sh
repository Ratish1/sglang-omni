#!/usr/bin/env bash
# Run on the host, not in the container: the container's nvidia-smi lists no stage processes. Maps every GPU process to
# its container pid (the last NSpid field) and then to the stage group a server log says spawned it, and prints
# "<log> <stage> <container pid> <MiB>".
# usage: bash proc_memory.sh <serve.log> [<serve.log> ...]
set -u
apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
for log in "$@"; do
  echo "$apps" | while IFS=', ' read -r host_pid mib; do
    container_pid=$(awk '/^NSpid/ {print $NF}' /proc/$host_pid/status 2> /dev/null)
    stage=$(grep -oE "StageGroup [a-z_]+: spawned 1 process\(es\) \(pids=\[$container_pid\]\)" "$log" | awk '{print $2}' | tr -d ':')
    if [ -n "$container_pid" ] && [ -n "$stage" ]; then
      echo "$(basename "$log") $stage $container_pid $mib"
    else
      :
    fi
  done
done
