set -u
# 11-01 check 5 (ARCHITECTURE_REVIEW_20261101.md section 7): the colocated H100 BF16 memory budget on upstream main.
# One boot on card 0 after check 2 has left the cards: every stage's pool and load memory from the serve log, every
# process's card memory every 2 s (run_bench_boot.sh apps.csv) under SeedTTS EN, then MMMU talker (images and long
# replies, the thinker's KV pool and the image encoder's activations), 200 samples each at c16.
Q=/data/c5_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
TREE=/data/ab/main
mkdir -p $Q
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
until grep -q "check2 done" /data/c2_1101/queue.log 2>/dev/null; do sleep 10; done
log "check5 start: tools $(git -C /data/tools rev-parse --short HEAD); tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8); apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | tr '\n' ';')]"
SCORE=0 MAX_SAMPLES=200 PIN_CPUS=0-15,64-79 PIN_NODE=0 bash $S/run_bench_boot.sh $TREE $Q/boot 0 33300 bf16 16 "seedtts_en mmmu_talker"
log "boot rc $?; $(tail -3 $Q/boot/progress.txt | tr '\n' ' ')"
python3 $S/check5_20261101/memory_budget.py $Q/boot > $Q/memory_budget.txt 2>&1
log "reader rc $?"
while read -r line; do log "  $line"; done < $Q/memory_budget.txt
log "check5 done"
