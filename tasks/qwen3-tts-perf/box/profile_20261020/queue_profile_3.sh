#!/usr/bin/env bash
# 10-20 evening: BBuf's degenerate cells at the maximum batch (PLAN.md 2, W6 then W5), formal then mapping, from the
# profiling branch (main bcd0d8c68 plus its instrumentation), one boot at a time on card 0 with card 1 idle. Then per
# cell the BBuf pair analysis and the step reads the morning cells got (step_anatomy.py, step_gaps.py).
set -u
D=/data/prof_20261020
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/profile_20261020
B=/data/bbuf/llm-torch-profiler-analysis
L=$D/queue.log
echo "queue 3 start $(date +%T) marks $(git -C $T rev-parse HEAD) tree $(git -C $T rev-parse HEAD:sglang_omni)" >> $L
for mode in formal mapping; do
  python3 $IN/profile_cells.py --tree $T --card 0 --port 31000 --mode $mode --out $D/${mode}_max --cells W6 W5 \
    > $D/${mode}_max.out 2>&1
  echo "${mode}_max rc $? $(date +%T)" >> $L
done
for cell in W6 W5; do
  TRACE=$(ls $D/formal_max/$cell/trace*.trace.json.gz | head -1)
  python3 $B/scripts/analyze_llm_torch_profile.py --framework sglang --mapping-input $D/mapping_max/$cell \
    --formal-input $D/formal_max/$cell > $D/formal_max/$cell/bbuf_pair.md 2>&1
  python3 $IN/step_anatomy.py $TRACE > $D/formal_max/$cell/steps.md 2>&1
  python3 $IN/step_gaps.py $TRACE > $D/formal_max/$cell/step_gaps.md 2>&1
  echo "reads $cell rc $? $(date +%T)" >> $L
done
echo queue_3_done $(date +%T) >> $L
