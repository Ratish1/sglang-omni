set -u
# 11-01 item 6: the thinker MoE call on every bf16 runner SGLang 0.5.22 offers at decode and prefill shapes, on card 0
# after check 2b, against the byte floor at card 0's measured 3.10 TB/s. Item 6's runner comparison
# (items/25/PLAN.md section 10) was read on SGLang 0.5.21's FlashInfer.
Q=/data/i6_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
PIN=0-15,64-79
mkdir -p $Q
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
until grep -q "check2b done" /data/c2b_1101/queue.log 2>/dev/null; do sleep 10; done
log "runners start: tools $(git -C /data/tools rev-parse --short HEAD); flashinfer $(python3 -c 'import flashinfer; print(flashinfer.__version__)' 2>&1); card 0 apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | grep -i 0A:00 | tr '\n' ';')]"
taskset -c $PIN python3 $S/moe_runner_study.py --bandwidth-tbps 3.10 --tokens 1 2 3 4 6 8 12 16 32 64 128 > $Q/runners.out 2> $Q/runners.err
log "runners rc $?; card 0 apps after [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | grep -i 0A:00 | tr '\n' ';')]"
while read -r line; do log "  $line"; done < $Q/runners.out
log "runners done"
