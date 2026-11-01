set -u
# 11-01 predictor O4 prototype (design 15) on card 0: gates 1 and 2, bit identity of the persistent pass to main's
# fused layers and both paths' step replay times at 1, 12, 31 and 64 rows.
Q=/data/p_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts/predictor_20261101
TREE=/data/ab/main
PIN=0-15,64-79
mkdir -p $Q
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
log "persistent start: tools $(git -C /data/tools rev-parse --short HEAD); tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8); card 0 apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | grep -i 0A:00 | tr '\n' ';')]"
cd $S
PYTHONPATH=$TREE taskset -c $PIN python3 persistent_pass_bench.py --card 0 --batch ${BATCHES:-1 12 31 64} > $Q/persistent.out 2>&1
log "persistent bench rc $?"
while read -r line; do log "  $line"; done < <(grep -vE "^\s*$" $Q/persistent.out | tail -60)
log "persistent done"
