#!/usr/bin/env bash
# Block 23 of 10-29: the talker on the top priority stream with the final codes ordered by an event
# (perf/qwen3-tts-talker-priority-stream 92b153edc: 3e9fed779 the event, then the switch), and D on it
# (test/qwen3-tts-refenc-on-priority-v2 98b46f83b). Card 0: the qwen3_tts suite on the branch. Then pairs held at
# serve_arm.py's barrier, SeedTTS EN 1.7B Base, both card orders, scored, streaming arms with first chunk and underruns:
# main against the branch (what the priority does for main), then the branch against D on it (D's own cost under the
# priority); non-streaming c16 first (block 17's crash case). Then profile pairs main against the branch (nsys,
# encode_overlap.py): a talker step beside an encode against one without.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
REPO=/workspace/sglang-omni
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
S=$T/tasks/omni_step_profiling_20260918/scripts
OVERLAP=$T/tasks/qwen3-tts-perf/box/next_20261028/encode_overlap.py
W=$D/block23
L=$W/queue.log
export ARM_ROOT=$W/arms
mkdir -p $ARM_ROOT $W/logs $W/barrier $W/profile
git -C $REPO fetch -q origin perf/qwen3-tts-talker-priority-stream test/qwen3-tts-refenc-on-priority-v2
for spec in "prio2 92b153edc" "refprio2 98b46f83b"; do
  set -- $spec
  rm -rf /workspace/n29_$1
  git -C $REPO worktree prune
  git -C $REPO worktree add -q --detach /workspace/n29_$1 $2
done
echo "queue block23 start $(date +%T) kit $(git -C $T rev-parse HEAD) prio2 $(git -C /workspace/n29_prio2 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')refprio2 $(git -C /workspace/n29_refprio2 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
(cd /workspace/n29_prio2 && CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider > $W/units_prio2.txt 2>&1)
echo "== units prio2 rc $?: $(grep -E '[0-9]+ passed|failed|error' $W/units_prio2.txt | tail -n 1) $(date +%T)" >> $L

arm() {
  local barrier=$1 label=$2 tree=$3 card=$4 port=$5
  shift 5
  python3 $ARM --label $label --tree /workspace/n29_$tree --card $card --port $port --barrier $W/barrier/$barrier "$@" >> $W/logs/arm_$label.log 2>&1
  echo "arm $label rc $? $(grep -h -E 'Completed requests|Failed requests|RTF mean|WER \(corpus|Max token hits|Throughput|TTFC mean|TTFC p95|Max underrun p95' $ARM_ROOT/$label/bench.log $ARM_ROOT/$label/score*.log 2> /dev/null | tr -s ' ' | sort -u | tr '\n' ';') $(grep -h -o -E 'speaker similarity: [0-9.]+' $ARM_ROOT/$label/score*.log 2> /dev/null | tail -1) $(grep -c -E 'device-side assert|CUDA error' $ARM_ROOT/$label/serve.log) $(date +%T)" >> $L
}

pair() {
  local name=$1
  arm $name $2 &
  local a=$!
  arm $name $3 &
  local b=$!
  wait $a $b
}

N16="--concurrency 16 --no-stream --score"
S16="--concurrency 16 --score"
pair p_c16n_a "p_c16n_a_main main 0 11000 $N16" "p_c16n_a_prio2 prio2 1 11500 $N16"
pair p_c16n_b "p_c16n_b_prio2 prio2 0 12000 $N16" "p_c16n_b_main main 1 12500 $N16"
pair p_c16s_a "p_c16s_a_main main 0 13000 $S16" "p_c16s_a_prio2 prio2 1 13500 $S16"
pair p_c16s_b "p_c16s_b_prio2 prio2 0 14000 $S16" "p_c16s_b_main main 1 14500 $S16"
echo "main against the branch done $(date +%T)" >> $L
pair d_c16n_a "d_c16n_a_prio2 prio2 0 17000 $N16" "d_c16n_a_refprio2 refprio2 1 17500 $N16"
pair d_c16n_b "d_c16n_b_refprio2 refprio2 0 18000 $N16" "d_c16n_b_prio2 prio2 1 18500 $N16"
pair d_c16s_a "d_c16s_a_prio2 prio2 0 19000 $S16" "d_c16s_a_refprio2 refprio2 1 19500 $S16"
pair d_c16s_b "d_c16s_b_refprio2 refprio2 0 20000 $S16" "d_c16s_b_prio2 prio2 1 20500 $S16"
echo "the branch against D on it done $(date +%T)" >> $L

profile() {
  local tree=$1 out=$2 card=$3 port=$4 stream_flag=$5
  BENCH_STREAM=$stream_flag PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py \
    NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
    bash $S/run_nsys_boot.sh /workspace/n29_$tree $out $card $port 16 640 > $out.out 2>&1
  echo "profile $(basename $out) rc $? $(date +%T)" >> $L
}

for mode in stream nonstream; do
  if [ $mode = stream ]; then flag=--stream; else flag=""; fi
  for x in prio2 refprio2; do
    profile main $W/profile/${mode}_a_main_vs_$x 0 26200 "$flag" &
    profile $x $W/profile/${mode}_a_$x 1 26700 "$flag" &
    wait
  done
done
python3 $OVERLAP $W/profile/*_a_*/ > $W/overlap.txt 2>&1
echo "overlap rc $? $(date +%T)" >> $L
echo "queue block23 done $(date +%T)" >> $L
