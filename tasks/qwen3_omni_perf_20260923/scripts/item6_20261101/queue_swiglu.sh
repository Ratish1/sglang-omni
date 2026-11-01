set -u
# 11-01 item 6, design 03 M1's first step (never taken): the thinker decode step alone with SGLang's fused SwiGLU
# epilogue (SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED=1, interleaved W13, no act_and_mul launch) against without, on card 0,
# off / on / off so drift shows. sglang.benchmark.one_batch, a 126-token prompt, decode at 1 to 16 rows.
Q=/data/i6_1101
PIN=0-15,64-79
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
until grep -q "breakdown done" $Q/queue.log 2>/dev/null; do sleep 10; done
for arm in off1 on off2; do
  case $arm in on) flag=1 ;; *) flag=0 ;; esac
  log "swiglu $arm start (flag $flag); card 0 apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | grep -i 0A:00 | tr '\n' ';')]"
  SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED=$flag taskset -c $PIN python3 -m sglang.benchmark.one_batch \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --batch-size 1 2 4 8 16 --input-len 126 --output-len 32 \
    --run-name swiglu_$arm --result-filename $Q/swiglu.jsonl > $Q/swiglu_$arm.out 2> $Q/swiglu_$arm.err
  log "swiglu $arm rc $?; interleaved lines $(grep -ci 'interleav' $Q/swiglu_$arm.err)"
  grep -E "Decode\.  median" $Q/swiglu_$arm.out | tail -5 | while read -r line; do log "  $arm $line"; done
done
log "swiglu done"
