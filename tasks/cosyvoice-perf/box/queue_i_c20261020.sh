#!/usr/bin/env bash
# 10-20, cards 2 and 3. Prerequisite 1 v2: the CUDA speech tokenizer on a top priority torch stream handed to ONNX
# Runtime (user_compute_stream) in place of its default priority unified stream, run lock and warm run kept. Its call
# took 25.5 ms of wall in main's c16 serving against 8.5 ms on an idle card, the lock holding through the wait behind
# the vocoder's kernels (p_q1: c16 first audio p95 +3.5 % over 16 runs). Trees: t_t1 (fix/cosyvoice3-speech-
# tokenizer-startup-arena-v2 1d7e07858, prerequisite 1 alone), t_s4 (perf/cosyvoice3-prefix-pool-sized 6ed7f551d,
# the stack), t_cc4 (prof/cosyvoice3-candidate-prefill-pool a41040811). Units, warms, main and t_t1 profiled at the
# same time at c16 (the tokenizer's calls read by tokenize_ranges_c20261020.py), pairs p_t1 (main against t_t1),
# p_s4 (main against t_s4) and p_cc4 (t_c20 against t_cc4), WER and SIM of p_s4 and p_cc4 c16.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_i.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_i_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
M=t_m20
C=t_c20
T=t_t1
S=t_s4
G=t_cc4
R=$D/prof_i
log() { echo "$* $(date +%T)" >> $D/queue_i.txt; }
units() {
  (
    cd /workspace/$1
    CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=/workspace/$1 taskset -c $3 python3 -m pytest -q -p no:cacheprovider \
      tests/unit_test/fun_cosyvoice3 tests/unit_test/test_stage_device_contract.py \
      tests/unit_test/scheduling/test_engine_factory.py > $D/units/units_$1.log 2>&1
    echo "units $1 rc $? $(grep -E 'passed|failed' $D/units/units_$1.log | tail -n 1) $(date +%T)" >> $D/queue_i.txt
  )
}
score() {
  local dir=$1
  (cd /workspace/$M && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $?"
  (cd /workspace/$M && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $?"
}
pair() {
  (cd /workspace/sglang-omni && PAIR_CELLS=$4 python3 $IN/serve_pair_c20261020.py /workspace/$1 /workspace/$2 $D/$3 \
    > $D/$3.log 2>&1)
  log "pair $3 rc $?"
}

log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
cd /workspace/sglang-omni
for pair in "$T fix/cosyvoice3-speech-tokenizer-startup-arena-v2" "$S perf/cosyvoice3-prefix-pool-sized" \
  "$G prof/cosyvoice3-candidate-prefill-pool"; do
  set -- $pair
  git fetch -q origin $2
  [ -d /workspace/$1 ] || git worktree add -q --detach /workspace/$1 FETCH_HEAD
done
for tree in $M $C $T $S $G; do
  log "tree $tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$tree rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

units $S 2 0-15,64-79 &
(units $T 3 16-31,80-95; units $G 3 16-31,80-95) &
wait

bash $IN/warm_c20261020.sh $T 2 0-15,64-79 28300 $D/warm_$T &
bash $IN/warm_c20261020.sh $S 3 16-31,80-95 28500 $D/warm_$S &
wait
bash $IN/warm_c20261020.sh $G 2 0-15,64-79 28700 $D/warm_$G
for tree in $T $S $G; do
  log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
done

mkdir -p $R
PEER=$R/$T PYSPY=0 CELLS="c16:16:320" bash $IN/cosy_profile_run_c20261020.sh $M 2 0-15,64-79 28900 $R/$M \
  > $R/run_$M.log 2>&1 &
PEER=$R/$M PYSPY=0 CELLS="c16:16:320" bash $IN/cosy_profile_run_c20261020.sh $T 3 16-31,80-95 29100 $R/$T \
  > $R/run_$T.log 2>&1 &
wait
log "profiles $(tail -n 1 $R/$M/progress.txt) / $(tail -n 1 $R/$T/progress.txt)"
python3 $IN/tokenize_ranges_c20261020.py main=$R/$M/c16.sqlite t1=$R/$T/c16.sqlite > $R/tokenize_ranges.txt 2>&1
cat $R/tokenize_ranges.txt >> $D/queue_i.txt

pair $M $T p_t1 c1,c1u,c16,c64,r16
pair $M $S p_s4 c1,c1u,c16,c64,r16,L16
pair $C $G p_cc4 c1,c1u,c16,c64,r16,L16
for dir in p_s4/a/gpu0/c16 p_s4/a/gpu1/c16 p_s4/b/gpu0/c16 p_s4/b/gpu1/c16 \
  p_cc4/a/gpu0/c16 p_cc4/a/gpu1/c16 p_cc4/b/gpu0/c16 p_cc4/b/gpu1/c16; do
  score $D/$dir
done
log "queue done"
