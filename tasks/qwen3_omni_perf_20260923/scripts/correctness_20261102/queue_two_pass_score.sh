set -u
# 11-02 score the cells' first PASSES passes beside the running cells, on card 0 once the talker suite has left it.
# Usage: queue_two_pass_score.sh OUT PASSES WAIT_FOR_LOG
OUT=$1
PASSES=$2
WAIT_FOR=$3
C=/data/tools/tasks/qwen3_omni_perf_20260923/scripts/correctness_20261102
mkdir -p $OUT
until grep -q 'talker suite done' $WAIT_FOR; do sleep 10; done
for arm in fp32 bf16; do
  python3 $C/cells_snapshot.py /data/cells_1102/full_${arm}_argmax $OUT/$arm $PASSES >> $OUT/queue.log 2>&1
done
bash $C/queue_score.sh $OUT/queue.log /data/ab/main $OUT/fp32 $OUT/bf16
python3 $C/cells_paired.py $OUT/fp32 $OUT/bf16 $OUT/paired.json >> $OUT/queue.log 2>&1
echo "$(date -u +%T) paired done" >> $OUT/queue.log
