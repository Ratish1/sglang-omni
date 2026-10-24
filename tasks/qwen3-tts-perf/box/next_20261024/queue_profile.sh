#!/usr/bin/env bash
# After queue_gate.sh: the Qwen3-TTS Base profile again, on the tree every open item lands in (main 67476823a with D8
# and D3's first step, plus the instrumentation), the 10-22 cells and method (profile_20261022/PLAN.md), card 2 alone
# with nothing else on the box: formal then mapping captures of the long cells and W3, the long cells' windows placed
# from each cell's timing list run unprofiled on the same boot (window.json per cell); then nsys with the pipeline
# probe and the interpreter-lock trace for L1 c16, L2 c64 and L3 c16; then per cell the BBuf pair analysis and the
# step reads. The boot's long lists are checked against the lists the pairs use (byte identical by construction).
# Skipped when either branch failed its gate.
set -u
D=/data/prof_20261024
G=/data/next1024
T=${T:?the profile tree}
IN=$T/tasks/qwen3-tts-perf/box/profile_20261020
S=$T/tasks/omni_step_profiling_20260918/scripts
B=/data/bbuf/AI-Infra-Auto-Driven-SKILLS/skills/llm-torch-profiler-analysis
CELLS="L1_c1 L1_c16 L2_c1 L2_c16 L2_c64 L3_c16 W3"
L=$G/queue.log
until grep -q queue_gate_done $L; do sleep 30; done
mkdir -p $D
echo "queue profile start $(date +%T) tree $(git -C $T rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
if ls $G/failed_* > /dev/null 2>&1; then
  echo "stopped: a branch failed its gate" >> $L
  echo "queue_profile_done $(date +%T)" >> $L
  exit 1
else
  :
fi
if [ ! -d /data/bbuf/AI-Infra-Auto-Driven-SKILLS ]; then
  mkdir -p /data/bbuf
  git clone -q https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS /data/bbuf/AI-Infra-Auto-Driven-SKILLS
  git -C /data/bbuf/AI-Infra-Auto-Driven-SKILLS checkout -q 8d62399
else
  :
fi
echo "analyzer $(git -C /data/bbuf/AI-Infra-Auto-Driven-SKILLS rev-parse --short HEAD)" >> $L
for mode in formal mapping; do
  python3 $IN/profile_cells.py --tree $T --card 2 --port 2000 --mode $mode --out $D/$mode --cells $CELLS \
    > $D/$mode.out 2>&1
  echo "$mode rc $? $(date +%T)" >> $L
done
for list in L1_c1_run L1_c16_run L2_c1_run L2_c16_run L2_c64_run L3_c16_run L1_c16_warm L3_c16_warm; do
  cmp -s $D/formal/meta_w2/long/$list.lst /data/d1merge/long/$list.lst && echo "list $list same" >> $L \
    || echo "list $list DIFFERS from the pairs' list" >> $L
done
for cell in "L1_c16 16 96" "L2_c64 64 64" "L3_c16 16 32"; do
  set -- $cell
  PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py \
    NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
    bash $S/run_nsys_boot.sh $T $D/nsys_$1 2 3200 $2 $3 $D/formal/meta_w2/long/$1_run.lst > $D/nsys_$1.out 2>&1
  echo "nsys_$1 rc $? $(date +%T)" >> $L
done
for cell in $CELLS; do
  TRACE=$(ls $D/formal/$cell/trace*.trace.json.gz | head -1)
  python3 $B/scripts/analyze_llm_torch_profile.py --framework sglang --mapping-input $D/mapping/$cell \
    --formal-input $D/formal/$cell > $D/formal/$cell/bbuf_pair.md 2>&1
  python3 $IN/step_anatomy.py $TRACE > $D/formal/$cell/steps.md 2>&1
  python3 $IN/step_gaps.py $TRACE > $D/formal/$cell/step_gaps.md 2>&1
  echo "reads $cell rc $? window $(cat $D/formal/$cell/window.json 2>/dev/null | tr -d '\n ') $(date +%T)" >> $L
done
echo "queue_profile_done $(date +%T)" >> $L
