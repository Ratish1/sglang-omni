#!/usr/bin/env bash
# 11-19, node-0001: MMSU on the fused text path leaned -7 to -9 % req/s at c16 on B against main (mm_*), all of it in
# the preprocessing queue wait and its serial work. Bisect by commit: main, the lane commit alone (b1f62fbba, main's
# OMP), B (eb449662d), the PR head (cea7dde64, the config's OMP split), boots m l b h h b l m, c96 and c16, clean
# and events, no taskset.
set -u
O=/data/v1119
K=/workspace/kit2/tasks/runtime-perf-research/box
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
tree_of() { case $1 in m) echo /workspace/main ;; l) echo /workspace/vlane1 ;; b) echo /workspace/vb ;; h) echo /workspace/vhead ;; *) exit 2 ;; esac; }
log "q1119_mmsubisect start"
for arm in m l b h h b l m; do
  quiet_host
  index=$([ -d $O/mb_${arm}1 ] && echo 2 || echo 1)
  name=mb_${arm}$index
  note $name $(tree_of $arm) "omni_text_fused mmsu c96 c16"
  python3 $K/baseline.py --tree $(tree_of $arm) --model omni_text_fused --port 28800 --out $O/$name \
    --bench mmsu --loads 96,16 --passes clean,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
done
log "q1119_mmsubisect done"
