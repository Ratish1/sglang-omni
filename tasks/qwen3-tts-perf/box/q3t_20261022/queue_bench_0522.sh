#!/usr/bin/env bash
# After the follow-up queue, the PR bodies' micro-benches on the 0.5.22 stack. The predictor step from a CUDA graph,
# main and P6, one process per tree and batch, both trees on each card in opposite orders. Then V3's decoder
# transformer step against the plain step: eager shapes on card 2, compiled on card 3.
set -u
D=/data/q3t_20261022
T=${T:-/workspace/t_marks}
S=$T/tasks/omni_step_profiling_20260918/scripts
M=/workspace/main_057d97629
P6=/workspace/p6pr_afceccbb0
V3=/workspace/v3pr_15849cd2a
L=$D/queue.log
mkdir -p $D/bench
until grep -q queue_followup_0522_done $L; do sleep 30; done
echo "queue bench 0522 start $(date +%T)" >> $L
predictor_card() {
  CARD=$1; shift
  for TREE in "$@"; do
    for BATCH in 1 16 64; do
      (cd $TREE && TRITON_CACHE_DIR=$D/triton_$(basename $TREE) PYTHONPATH=$TREE timeout 900 \
        python3 $S/predictor_layers_bench.py --card $CARD --batch $BATCH) \
        >> $D/bench/predictor_$(basename $TREE)_card$CARD.txt 2>&1
    done
  done
}
predictor_card 2 $M $P6 &
predictor_card 3 $P6 $M &
wait
echo "predictor bench done $(date +%T) $(grep -h 'step replay' $D/bench/predictor_*.txt | wc -l) step lines" >> $L
(cd $V3 && PYTHONPATH=$V3 timeout 1800 python3 $S/vocoder_transformer_bench.py --card 2 --widths 1 4 64 --rows 1 8 16 \
  > $D/bench/transformer_eager.txt 2>&1) &
(cd $V3 && PYTHONPATH=$V3 timeout 1800 python3 $S/vocoder_transformer_bench.py --card 3 --compiled --widths 1 8 --rows 1 8 \
  > $D/bench/transformer_compiled.txt 2>&1) &
wait
echo "queue_bench_0522_done $(date +%T)" >> $L
