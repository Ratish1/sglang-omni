set -u
# 11-02 score the cells' first PASSES passes beside any running cell, on card 0 once WAIT_FOR_LOG holds WAIT_FOR_TEXT, the
# job on card 0 having left it, and both cells hold PASSES full passes.
# Usage: queue_two_pass_score.sh OUT PASSES WAIT_FOR_LOG WAIT_FOR_TEXT
OUT=$1
PASSES=$2
WAIT_FOR=$3
WAIT_TEXT=$4
C=/data/tools/tasks/qwen3_omni_perf_20260923/scripts/correctness_20261102
ROWS=$(( PASSES * $(wc -w < /data/sample_ids_all.txt) ))
mkdir -p $OUT
until grep -q "$WAIT_TEXT" $WAIT_FOR; do sleep 10; done
until [ "$(grep -c ^sample /data/cells_1102/full_fp32_argmax.log)" -ge $ROWS ] && [ "$(grep -c ^sample /data/cells_1102/full_bf16_argmax.log)" -ge $ROWS ]; do sleep 10; done
for arm in fp32 bf16; do
  python3 $C/cells_snapshot.py /data/cells_1102/full_${arm}_argmax $OUT/$arm $PASSES >> $OUT/queue.log 2>&1
done
bash $C/queue_score.sh $OUT/queue.log /data/ab/main $OUT/fp32 $OUT/bf16
python3 $C/cells_paired.py $OUT/fp32 $OUT/bf16 $OUT/paired.json >> $OUT/queue.log 2>&1
echo "$(date -u +%T) paired done" >> $OUT/queue.log
