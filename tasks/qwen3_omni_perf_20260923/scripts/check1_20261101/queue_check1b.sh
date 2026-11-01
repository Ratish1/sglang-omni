set -u
# The MoE probe again after queue_check1.sh's one_batch has left card 0: its first run used SGLang 0.5.21's
# initialize_model_parallel widths, which 0.5.22 reads from the published topology instead.
Q=/data/c1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
PIN=0-15,64-79
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
until grep -q "check1 done" $Q/queue.log; do sleep 5; done
apps=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | grep -i "0A:00" | tr '\n' ';')
log "moe2 start: tools $(git -C /data/tools rev-parse --short HEAD); card 0 apps: [$apps]"
taskset -c $PIN python3 $S/moe_config_probe.py --label h100_triton380 --bandwidth-tbps 3.10 > $Q/moe2.out 2> $Q/moe2.err
rc=$?
apps=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | grep -i "0A:00" | tr '\n' ';')
log "moe2 rc=$rc; card 0 apps after: [$apps]"
grep -E "^tokens|^h100|Fallback|not found|Using MoE|TMA|Error" $Q/moe2.out $Q/moe2.err | while read -r line; do log "  $line"; done
log "check1b done"
