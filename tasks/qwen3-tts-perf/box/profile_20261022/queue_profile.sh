#!/usr/bin/env bash
# 10-22, Qwen3-TTS Base profile with long prefills and long decodes (profile_20261022/PLAN.md), from the profiling
# branch (main 272845a9c plus V3 plus its instrumentation), one boot at a time on card 2 (0 and 1 are the CosyVoice
# container's): formal then mapping captures of the long cells and W3, then nsys with the pipeline probe and the
# interpreter-lock trace for L1 c16, L2 c64 and L3 c16, then per cell the BBuf pair analysis and the step reads.
set -u
D=/data/prof_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/profile_20261020
S=$T/tasks/omni_step_profiling_20260918/scripts
B=/data/bbuf/AI-Infra-Auto-Driven-SKILLS/skills/llm-torch-profiler-analysis
CELLS="L1_c1 L1_c16 L2_c1 L2_c16 L2_c64 L3_c16 W3"
L=$D/queue.log
mkdir -p $D
echo "queue profile 1022 start $(date +%T) marks $(git -C $T rev-parse HEAD) tree $(git -C $T rev-parse HEAD:sglang_omni)" >> $L
if [ ! -d /data/bbuf/AI-Infra-Auto-Driven-SKILLS ]; then
  mkdir -p /data/bbuf
  git clone -q https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS /data/bbuf/AI-Infra-Auto-Driven-SKILLS
  git -C /data/bbuf/AI-Infra-Auto-Driven-SKILLS checkout -q 8d62399
else
  :
fi
echo "analyzer $(git -C /data/bbuf/AI-Infra-Auto-Driven-SKILLS rev-parse --short HEAD)" >> $L
for mode in formal mapping; do
  python3 $IN/profile_cells.py --tree $T --card 2 --port 21000 --mode $mode --out $D/$mode --cells $CELLS \
    > $D/$mode.out 2>&1
  echo "$mode rc $? $(date +%T)" >> $L
done
for cell in "L1_c16 16 96" "L2_c64 64 64" "L3_c16 16 32"; do
  set -- $cell
  PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py \
    NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
    bash $S/run_nsys_boot.sh $T $D/nsys_$1 2 23000 $2 $3 $D/formal/meta_w2/long/$1_run.lst > $D/nsys_$1.out 2>&1
  echo "nsys_$1 rc $? $(date +%T)" >> $L
done
for cell in $CELLS; do
  TRACE=$(ls $D/formal/$cell/trace*.trace.json.gz | head -1)
  python3 $B/scripts/analyze_llm_torch_profile.py --framework sglang --mapping-input $D/mapping/$cell \
    --formal-input $D/formal/$cell > $D/formal/$cell/bbuf_pair.md 2>&1
  python3 $IN/step_anatomy.py $TRACE > $D/formal/$cell/steps.md 2>&1
  python3 $IN/step_gaps.py $TRACE > $D/formal/$cell/step_gaps.md 2>&1
  echo "reads $cell rc $? $(date +%T)" >> $L
done
echo queue_profile_1022_done $(date +%T) >> $L
