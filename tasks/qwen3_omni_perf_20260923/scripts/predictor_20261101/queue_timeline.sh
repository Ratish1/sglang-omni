set -u
# 11-01 predictor O4 prototype on card 0: where the persistent pass's time goes, per phase and barrier, from each
# program's global timer stamps, with and without the L2 prefetch.
Q=/data/p_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts/predictor_20261101
TREE=/data/ab/main
PIN=0-15,64-79
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
log "timeline start: tools $(git -C /data/tools rev-parse --short HEAD); card 0 apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | grep -i 0A:00 | tr '\n' ';')]"
cd $S
PYTHONPATH=$TREE taskset -c $PIN python3 -u persistent_pass_timeline.py --card 0 --batch ${BATCHES:-1 12 31 64} > $Q/timeline.out 2>&1
log "timeline rc $?"
log "timeline done"
