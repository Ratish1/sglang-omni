#!/usr/bin/env bash
# Block 11 of 10-29: D's bisect on upstream main 08cecee99 (A merged). Arms, each one server: main; fp32 (main plus D's
# stages.py and engine_builder.py: the float32 encoder load, its codebook tables and the cuBLAS workspace made first);
# exact (main plus D's reference_encoder_cuda_graph.py and request_builders.py: the exact-length tail fill and the
# empty-reference refusal, bf16 encoder); refenc (D, both); main2 (main against itself, the noise band). Cards 0 and 1.
# 1. A warm arm (main, non-streaming c16, scored) fetches the models and the corpus; card 1 runs D's unit suite.
# 2. Served pairs, SeedTTS EN 1.7B Base, every arm scored on its card: two rounds, each arm against main in both card
#    orders, non-streaming c16 then streaming c16, the arm order reversed in round 2.
# 3. Profile pairs under nsys (block 8's flags, 640 requests at c16), each arm against main in both orders, streaming then
#    non-streaming; encode_overlap.py over every run.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
REPO=/workspace/sglang-omni
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
S=$T/tasks/omni_step_profiling_20260918/scripts
OVERLAP=$T/tasks/qwen3-tts-perf/box/next_20261028/encode_overlap.py
L=$D/queue.log
export ARM_ROOT=$D/arms
mkdir -p $ARM_ROOT $D/logs $D/profile
echo "queue block11 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
git -C $REPO fetch -q origin fix/qwen3-tts-reference-encoder-float32-exact-length bisect/qwen3-tts-encoder-fp32-only \
  bisect/qwen3-tts-encoder-exact-only
git -C $REPO fetch -q upstream main
for spec in "main 08cecee99" "fp32 bdd671c4c" "exact beec04c1a" "refenc 19f0f8de5"; do
  set -- $spec
  rm -rf /workspace/n29_$1
  git -C $REPO worktree prune
  git -C $REPO worktree add -q --detach /workspace/n29_$1 $2
  echo "tree $1 $(git -C /workspace/n29_$1 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
done

PORTS=(11000 12000 13000 14000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000)
NEXT=0

arm() {
  local label=$1 tree=$2 card=$3 port=$4
  shift 4
  python3 $ARM --label $label --tree /workspace/n29_$tree --card $card --port $port "$@" >> $D/logs/arm_$label.log 2>&1
  echo "arm $label rc $? $(grep -h -E 'RTF mean|WER \(corpus|Max token hits|Throughput' $ARM_ROOT/$label/bench.log $ARM_ROOT/$label/score*.log 2> /dev/null | tr -s ' ' | sort -u | tr '\n' ';') $(grep -h -o -E 'speaker similarity: [0-9.]+' $ARM_ROOT/$label/score*.log 2> /dev/null | tail -1) $(date +%T)" >> $L
}

pair() {
  local mode_flag=$1 first_label=$2 first_tree=$3 second_label=$4 second_tree=$5
  local port=${PORTS[$((NEXT % ${#PORTS[@]}))]}
  NEXT=$((NEXT + 1))
  arm $first_label $first_tree 0 $port --concurrency 16 $mode_flag --score &
  local a=$!
  arm $second_label $second_tree 1 $((port + 500)) --concurrency 16 $mode_flag --score &
  local b=$!
  wait $a $b
}

arm warm_main main 0 30000 --concurrency 16 --no-stream --score &
warm=$!
(cd /workspace/n29_refenc && CUDA_VISIBLE_DEVICES=1 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider > $D/units_refenc.txt 2>&1)
echo "== units refenc rc $?: $(grep -E '[0-9]+ passed|failed|error' $D/units_refenc.txt | tail -n 1) $(date +%T)" >> $L
wait $warm

for round in 1 2; do
  if [ $round = 1 ]; then order="main2 fp32 exact refenc"; else order="refenc exact fp32 main2"; fi
  for x in $order; do
    if [ $x = main2 ]; then tree=main; else tree=$x; fi
    for mode in n s; do
      if [ $mode = n ]; then flag=--no-stream; else flag=""; fi
      pair "$flag" c16${mode}_r${round}a_main_vs_$x main c16${mode}_r${round}a_$x $tree
      pair "$flag" c16${mode}_r${round}b_$x $tree c16${mode}_r${round}b_main_vs_$x main
    done
  done
  echo "served round $round done $(date +%T)" >> $L
done

profile() {
  local tree=$1 out=$2 card=$3 port=$4 stream_flag=$5
  BENCH_STREAM=$stream_flag PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py \
    NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
    bash $S/run_nsys_boot.sh /workspace/n29_$tree $out $card $port 16 640 > $out.out 2>&1
  echo "profile $(basename $out) rc $? $(date +%T)" >> $L
}

for mode in stream nonstream; do
  if [ $mode = stream ]; then flag=--stream; else flag=""; fi
  for x in fp32 exact refenc main2; do
    if [ $x = main2 ]; then tree=main; else tree=$x; fi
    profile main $D/profile/${mode}_a_main_vs_$x 0 31000 "$flag" &
    profile $tree $D/profile/${mode}_a_$x 1 31500 "$flag" &
    wait
    profile $tree $D/profile/${mode}_b_$x 0 32000 "$flag" &
    profile main $D/profile/${mode}_b_main_vs_$x 1 32500 "$flag" &
    wait
  done
done
python3 $OVERLAP $D/profile/*_a_*/ $D/profile/*_b_*/ > $D/overlap.txt 2>&1
echo "overlap rc $? $(date +%T)" >> $L
echo "queue block11 done $(date +%T)" >> $L
