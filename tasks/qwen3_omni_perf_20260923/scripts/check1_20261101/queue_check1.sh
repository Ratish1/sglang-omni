set -u
# 11-01 check 1 (ARCHITECTURE_REVIEW_20261101.md section 7): the thinker's device cost alone on the target H100 with
# today's image (Triton 3.8.0, SGLang 0.5.22). The MoE call per token count against its byte floor under the table the
# process resolves (moe_config_probe.py), then the whole model forward alone (sglang.benchmark.one_batch): a 126-token
# prompt prefilled 1, 2, 4, 8 and 16 at a time and the decode step at those rows. Card 0 only; every step logs any
# other process on card 0 before and after it.
Q=/data/c1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
PIN=0-15,64-79
mkdir -p $Q
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
card0_apps() {
  nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | grep -i "0A:00" | tr '\n' ';'
}
step() {
  local name=$1
  shift
  log "$name start; card 0 apps: [$(card0_apps)]"
  taskset -c $PIN "$@" > $Q/$name.out 2> $Q/$name.err
  local rc=$?
  log "$name rc=$rc; card 0 apps after: [$(card0_apps)]"
  grep -E "^tokens|Fallback|not found|Using MoE|Prefill\. latency|Decode\.  median|Benchmark|batch_size|skipping|Error|error" $Q/$name.out $Q/$name.err | tail -40 | while read -r line; do log "  $line"; done
}
log "check1 start: tools $(git -C /data/tools rev-parse --short HEAD); $(python3 -c 'import sglang, torch, triton; print("sglang", sglang.__version__, "torch", torch.__version__, "triton", triton.__version__)')"
log "card 0: $(nvidia-smi -i 0 --query-gpu=name,pci.bus_id,memory.used,clocks.max.memory --format=csv,noheader)"
step hbm python3 -c '
import torch
x = torch.empty(4 * 1024**3, dtype=torch.bfloat16, device="cuda")
y = torch.empty_like(x)
x.normal_()
for name, fn, nbytes in (("read (sum)", lambda: x.sum(), x.numel() * 2), ("copy", lambda: y.copy_(x), 2 * x.numel() * 2)):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        fn()
    end.record()
    end.synchronize()
    seconds = start.elapsed_time(end) / 1000 / 20
    print(f"tokens hbm {name}: {nbytes / seconds / 1e12:.3f} TB/s")
'
step moe python3 $S/moe_config_probe.py --label h100_triton380 --bandwidth-tbps 3.35
step one_batch python3 -m sglang.benchmark.one_batch --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --batch-size 1 2 4 8 16 --input-len 126 --output-len 32 --run-name h100_q3o_thinker \
  --result-filename $Q/one_batch.jsonl
log "check1 done"
