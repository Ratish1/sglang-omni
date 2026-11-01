set -u
# 11-01 item 6, after check 5b: the served thinker MoE call per kernel on H100 (router, align, gate-up GEMM, SwiGLU, down
# GEMM, top-k sum) at decode counts, each GEMM against its byte floor at card 0's measured 3.10 TB/s. The per-kernel
# split was read on H200 only (OPTIMIZATIONS.md item 4); it decides whether item 6 targets the GEMMs' bandwidth or the
# non-GEMM launches around them.
Q=/data/i6_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
PIN=0-15,64-79
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
until grep -q "check5b done" /data/c5b_1101/queue.log 2>/dev/null; do sleep 10; done
log "breakdown start: tools $(git -C /data/tools rev-parse --short HEAD); card 0 apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | grep -i 0A:00 | tr '\n' ';')]"
taskset -c $PIN python3 $S/moe_decode_breakdown.py --bandwidth-tbps 3.10 --tokens 1 2 4 8 16 > $Q/breakdown.out 2> $Q/breakdown.err
log "breakdown rc $?"
while read -r line; do log "  $line"; done < $Q/breakdown.out
log "breakdown done"
