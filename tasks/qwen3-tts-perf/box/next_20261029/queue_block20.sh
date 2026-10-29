#!/usr/bin/env bash
# Block 20 of 10-29, after block 19 (WAIT_PID).
# 1. The CI case's reference (common_voice_en_10119832-common_voice_en_10119840) encoded as each tree serves it, graphs
#    and eager: D against D on the speedups branch (refcomp lost the speaker in 13 of 300 draws against D's 5), and main
#    against the speedups branch; byte identity names whether the prompt differs or only the draws.
# 2. D on the speedups branch against that branch under nsys (block 11b's flags and reader), both card orders, streaming
#    then non-streaming: block 18's RTF pairs (+2.3, +0.8 % non-streaming, -0.3, -0.4 % streaming) cannot place D's
#    cost on this base; the beside-encode step time can.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box/next_20261029
S=$T/tasks/omni_step_profiling_20260918/scripts
OVERLAP=$T/tasks/qwen3-tts-perf/box/next_20261028/encode_overlap.py
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
CASE=common_voice_en_10119832-common_voice_en_10119840_icl_sentence
W=$D/block20
L=$W/queue.log
mkdir -p $W/codes $W/profile
if [ -n "${WAIT_PID:-}" ]; then
  while [ -n "$(ps -o stat= -p $WAIT_PID 2> /dev/null | grep -v Z)" ]; do sleep 30; done
else
  :
fi
echo "queue block20 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L

codes() {
  local tree=$1 card=$2
  (cd /workspace/n29_$tree && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=/workspace/n29_$tree TORCHINDUCTOR_CACHE_DIR=$W/inductor_$tree \
    python3 $K/reference_codes_one.py encode $D/block14/parity/cases.jsonl --ids $CASE --model $M17 --out $W/codes/$tree.pt \
    > $W/codes/$tree.log 2>&1)
  echo "codes $tree rc $? $(date +%T)" >> $L
}
(codes refenc 0; codes main 0) &
zero=$!
(codes refcomp 1; codes compile 1) &
one=$!
wait $zero $one
python3 $K/reference_codes_one.py compare $W/codes/refenc.pt $W/codes/refcomp.pt > $W/codes_compare.txt 2>&1
python3 $K/reference_codes_one.py compare $W/codes/main.pt $W/codes/compile.pt >> $W/codes_compare.txt 2>&1
echo "codes compare: $(tr '\n' ';' < $W/codes_compare.txt) $(date +%T)" >> $L

profile() {
  local tree=$1 out=$2 card=$3 port=$4 stream_flag=$5
  BENCH_STREAM=$stream_flag PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py \
    NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
    bash $S/run_nsys_boot.sh /workspace/n29_$tree $out $card $port 16 640 > $out.out 2>&1
  echo "profile $(basename $out) rc $? $(date +%T)" >> $L
}

for mode in stream nonstream; do
  if [ $mode = stream ]; then flag=--stream; else flag=""; fi
  profile compile $W/profile/${mode}_a_compile 0 26200 "$flag" &
  profile refcomp $W/profile/${mode}_a_refcomp 1 26700 "$flag" &
  wait
  profile refcomp $W/profile/${mode}_b_refcomp 0 27200 "$flag" &
  profile compile $W/profile/${mode}_b_compile 1 27700 "$flag" &
  wait
done
python3 $OVERLAP $W/profile/*_a_*/ $W/profile/*_b_*/ > $W/overlap.txt 2>&1
echo "overlap rc $? $(date +%T)" >> $L
echo "queue block20 done $(date +%T)" >> $L
