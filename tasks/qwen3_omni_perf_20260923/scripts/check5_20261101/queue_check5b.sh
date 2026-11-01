set -u
# 11-01 check 5b, after the item 6 runner study: whether the colocated H100 BF16 thinker's 5,206-token KV pool holds back
# SeedTTS admissions at c64 (q106's thinker queue to prefill p99 921 ms against a 1.3 ms median). One colocated boot of
# upstream main on card 0, SeedTTS EN c64 over the full corpus with request events; thinker_admission.py reads SGLang's
# batch lines for the thinker's pool usage while requests wait.
Q=/data/c5b_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
TREE=/data/ab/main
mkdir -p $Q
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
until grep -q "runners done" /data/i6_1101/queue.log 2>/dev/null; do sleep 10; done
log "check5b start: tools $(git -C /data/tools rev-parse --short HEAD); tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8)"
EVENTS=1 SCORE=0 PIN_CPUS=0-15,64-79 PIN_NODE=0 bash $S/run_bench_boot.sh $TREE $Q/boot 0 33700 bf16 64 "seedtts_en"
log "boot rc $?; $(tail -2 $Q/boot/progress.txt | tr '\n' ' ')"
python3 $S/check5_20261101/thinker_admission.py $Q/boot/serve.log > $Q/admission.txt 2>&1
while read -r line; do log "  $line"; done < $Q/admission.txt
log "check5b done"
