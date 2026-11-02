set -u
# 11-02 the two-card lease's first work once setup_box.py has its files: the full-corpus cells resume on card 1, each
# skipping the samples in its reply_text.json, then the served talker on the image and video stages' requests, server on
# card 0 and HF's references on card 1.
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts/correctness_20261102
SETUP=/data/setup_1102/setup.log
TREE=/data/ab/main
QUEUE=/data/setup_1102/lease_start.log
log() { echo "$(date -u +%T) $*" >> $QUEUE; }

until grep -q '^Qwen/Qwen3-ASR' $SETUP; do sleep 10; done
for pair in "fp32 float32" "bf16 bfloat16"; do
  set -- $pair
  (cd $TREE && exec setsid env PYTHONPATH=$TREE TALKER_CELLS_DEVICE=cuda:1 python3 -u $S/talker_cells.py cell \
    /data/cells_1102/full_$1_argmax/seedtts_en /data/cells_1102/full_rows_$2.pt $2 argmax /data/sample_ids_all.txt 3) \
    >> /data/cells_1102/full_$1_argmax.log 2>&1 < /dev/null &
  log "cell $1 resumed"
done

until grep -q '^setup done' $SETUP; do sleep 10; done
mkdir -p /data/talker_1102
log "media talker start"
CONCURRENCY=16 ACCURACY_DEVICE=cuda:1 bash $S/queue_talker_suite.sh /data/talker_1102/media5 /data/ab/acc 0 media
log "media talker done"
