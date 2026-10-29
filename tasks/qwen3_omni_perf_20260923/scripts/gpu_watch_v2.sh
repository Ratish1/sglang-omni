# Logs, in UTC, every change in the set of GPU processes on our cards that belong to another container than ours, with a heartbeat each minute. A pid that exits before its cgroup is read is skipped: it cannot be classified, and a foreign workload outlives one sample.
OURS=$(podman inspect -f "{{.Id}}" sglang-omni-qwen3-omni-perf)
last=""
beat=0
while true; do
  foreign=""
  for line in $(nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id --format=csv,noheader,nounits | tr -d " "); do
    pid=${line%%,*}
    cgroup=$(tail -1 /proc/$pid/cgroup 2>/dev/null)
    if [ -z "$cgroup" ]; then
      continue
    fi
    case "$cgroup" in
      *"$OURS"*) ;;
      *) foreign="$foreign $line@$(echo "$cgroup" | grep -oE "[0-9a-f]{64}" | cut -c1-12)" ;;
    esac
  done
  state=${foreign:-clean}
  now=$(date -u +%s)
  if [ "$state" != "$last" ] || [ $((now - beat)) -ge 60 ]; then
    echo "$(date -u +%T) ${foreign:+FOREIGN}${foreign:-clean}"
    last=$state
    beat=$now
  fi
  sleep 5
done
