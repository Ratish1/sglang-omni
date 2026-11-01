set -u
# 11-01 predictor (DECISIONS "Colocated item ownership" 1), O4's two preconditions on card 0: the served predictor step
# alone at Qwen3-Omni's rows (the fused layers bench on upstream main, both graph replay modes), and a cooperative
# Triton launch with device-wide barriers captured in a CUDA graph and replayed correctly.
Q=/data/p_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts/predictor_20261101
TREE=/data/ab/main
PIN=0-15,64-79
mkdir -p $Q
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
log "feasibility start: tools $(git -C /data/tools rev-parse --short HEAD); tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8); card 0 apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | grep -i 0A:00 | tr '\n' ';')]"
taskset -c $PIN python3 $S/coop_graph_probe.py > $Q/coop.out 2>&1
log "coop rc $?"
while read -r line; do log "  $line"; done < <(grep -vE "^\s*$" $Q/coop.out | tail -8)
PYTHONPATH=$TREE taskset -c $PIN python3 $S/predictor_layers_bench.py --card 0 --batch 1 12 31 64 > $Q/layers.out 2>&1
log "layers bench rc $?"
while read -r line; do log "  $line"; done < <(grep -vE "^\s*$" $Q/layers.out | tail -40)
log "feasibility done"
