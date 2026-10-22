#!/usr/bin/env bash
# After queue_profile.sh: recapture the four windowed long cells whose window landed in their run's tail (L1 c16 held
# only 2 streams and no prefill, L2 c64 held none, L2 c16 and L3 c16 the second wave's end), formal then mapping, with
# leads set from the measured run lengths; then their BBuf pair analysis and step reads.
set -u
D=/data/prof_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/profile_20261020
B=/data/bbuf/AI-Infra-Auto-Driven-SKILLS/skills/llm-torch-profiler-analysis
CELLS="L1_c16 L2_c16 L2_c64 L3_c16"
L=$D/queue.log
until grep -q queue_profile_1022_done $L; do sleep 20; done
echo "queue profile fix start $(date +%T) marks $(git -C $T rev-parse HEAD) tree $(git -C $T rev-parse HEAD:sglang_omni)" >> $L
for mode in formal mapping; do
  python3 $IN/profile_cells.py --tree $T --card 2 --port 21000 --mode $mode --out $D/${mode}_fix --cells $CELLS \
    > $D/${mode}_fix.out 2>&1
  echo "${mode}_fix rc $? $(date +%T)" >> $L
done
for cell in $CELLS; do
  TRACE=$(ls $D/formal_fix/$cell/trace*.trace.json.gz | head -1)
  python3 $B/scripts/analyze_llm_torch_profile.py --framework sglang --mapping-input $D/mapping_fix/$cell \
    --formal-input $D/formal_fix/$cell > $D/formal_fix/$cell/bbuf_pair.md 2>&1
  python3 $IN/step_anatomy.py $TRACE > $D/formal_fix/$cell/steps.md 2>&1
  python3 $IN/step_gaps.py $TRACE > $D/formal_fix/$cell/step_gaps.md 2>&1
  echo "reads fix $cell rc $? $(date +%T)" >> $L
done
echo queue_profile_fix_done $(date +%T) >> $L
