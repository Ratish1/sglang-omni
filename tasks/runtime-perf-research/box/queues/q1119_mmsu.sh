#!/usr/bin/env bash
# 11-19, node-0001, after q1119_coltext: MMSU (all 2,000 CI questions) on the fused text-path config, where
# preprocessing shares one process with the encoders and the thinker, c96 and c16, clean and events, main against
# B (eb449662d), boots m b b m, no taskset. B runs requests without video on main's team there.
set -u
O=/data/v1119
K=/workspace/kit2/tasks/runtime-perf-research/box
MAIN=/workspace/main B=/workspace/vb
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
until grep -q "q1119_coltext done" $O/queue.log; do sleep 10; done
log "q1119_mmsu start"
for arm in m1:$MAIN b1:$B b2:$B m2:$MAIN; do
  quiet_host
  name=mm_${arm%%:*}
  note $name ${arm#*:} "omni_text_fused mmsu c96 c16"
  python3 $K/baseline.py --tree ${arm#*:} --model omni_text_fused --port 28800 --out $O/$name \
    --bench mmsu --loads 96,16 --passes clean,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
done
log "q1119_mmsu done"
