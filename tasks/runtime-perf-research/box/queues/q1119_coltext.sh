#!/usr/bin/env bash
# 11-19, node-0001, after q1119_stageshare: the colocated config's text Video-MME c16 (CI sample and video settings,
# two clean repeats and the events pass a boot), main against B (eb449662d), boots m b b m, no taskset; then the
# skip reasons of B's unit suites.
set -u
O=/data/v1119
K=/workspace/kit2/tasks/runtime-perf-research/box
MAIN=/workspace/main B=/workspace/vb
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
until grep -q "q1119_stageshare done" $O/queue.log; do sleep 10; done
log "q1119_coltext start"
for arm in m1:$MAIN b1:$B b2:$B m2:$MAIN; do
  quiet_host
  name=ct_${arm%%:*}
  note $name ${arm#*:} "omni_colocated text video c16"
  python3 $K/baseline.py --tree ${arm#*:} --model omni_colocated --port 28700 --out $O/$name \
    --bench videomme_text --loads 16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
done
(cd $B && python3 -m pytest -q -p no:cacheprovider -rs tests/unit_test/qwen3_omni tests/unit_test/preprocessing \
  tests/unit_test/utils tests/unit_test/minicpm_o tests/unit_test/ming_omni 2>&1 | grep -E "^SKIPPED|passed|failed") > $O/b_units_skips.log
log "units skips $(grep -c SKIPPED $O/b_units_skips.log) lines"
log "q1119_coltext done"
