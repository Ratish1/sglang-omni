#!/usr/bin/env bash
# 10-20, cards 2 and 3. The PR stack of DESIGN_PR_STACK_20261020.md, each layer on the one below, on upstream main
# bcd0d8c68: L1 the vocoder candidate (perf/cosyvoice3-stack-l1 8202dcb25, tree t_l1), L2 expandable segments
# (perf/cosyvoice3-expandable-segments-v3 b494c6d29, t_l2), L3 the speech tokenizer from CUDA graphs
# (perf/cosyvoice3-speech-tokenizer-graphs ca4aa4d39, t_l3b), L4 item 3 (perf/cosyvoice3-prefix-pool-sized-v2
# 5ad5d46ab, t_l4), L5 P1 (perf/cosyvoice3-prefill-graphs-v2 3ab6603b9, t_l5); main as t_m21. onnx into the image
# first (the fun-cosyvoice3 extra now declares it). Units of every layer, warms, L2 and L3 profiled at the same time
# at c16 (the tokenizer's calls), the worst step check of L4 at 16 and 32 rows, a pair per layer on the layer below
# (p_l2, p_l3, p_l4, p_l5) and the stack on main (p_all), WER and SIM of p_l3 and p_all c16.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_j.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_j_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
R=$D/prof_j
log() { echo "$* $(date +%T)" >> $D/queue_j.txt; }
units() {
  (
    cd /workspace/$1
    CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=/workspace/$1 taskset -c $3 python3 -m pytest -q -p no:cacheprovider \
      tests/unit_test/fun_cosyvoice3 tests/unit_test/test_stage_device_contract.py \
      tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/scheduling/test_streaming_vocoder.py \
      tests/unit_test/model_runner/test_sglang_execution.py tests/unit_test/qwen3_omni/test_talker_token_readback.py \
      tests/unit_test/qwen3_tts/test_decode_step_staging.py > $D/units/units_$1.log 2>&1
    echo "units $1 rc $? $(grep -E 'passed|failed' $D/units/units_$1.log | tail -n 1) $(date +%T)" >> $D/queue_j.txt
  )
}
score() {
  local dir=$1
  (cd /workspace/t_m21 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_m21 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
pair() {
  (cd /workspace/sglang-omni && PAIR_CELLS=$4 python3 $IN/serve_pair_c20261020.py /workspace/$1 /workspace/$2 $D/$3 \
    > $D/$3.log 2>&1)
  log "pair $3 rc $?"
}

log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
uv pip install -q --python "$(command -v python3)" /tmp/onnxwheel/onnx-1.23.2-*.whl
log "onnx $(python3 -c 'import onnx; print(onnx.__version__)' 2>&1 | tail -n 1)"
mkdir -p $D/units $R
cd /workspace/sglang-omni
git fetch -q upstream main
[ -d /workspace/t_m21 ] || git worktree add -q --detach /workspace/t_m21 FETCH_HEAD
for pair in "t_l1 perf/cosyvoice3-stack-l1" "t_l2 perf/cosyvoice3-expandable-segments-v3" \
  "t_l3b perf/cosyvoice3-speech-tokenizer-graphs" "t_l4 perf/cosyvoice3-prefix-pool-sized-v2" \
  "t_l5 perf/cosyvoice3-prefill-graphs-v2"; do
  set -- $pair
  git fetch -q origin $2
  [ -d /workspace/$1 ] || git worktree add -q --detach /workspace/$1 FETCH_HEAD
done
for tree in t_m21 t_l1 t_l2 t_l3b t_l4 t_l5; do
  log "tree $tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$tree rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

(units t_l1 2 0-15,64-79; units t_l3b 2 0-15,64-79; units t_l5 2 0-15,64-79) &
(units t_l2 3 16-31,80-95; units t_l4 3 16-31,80-95) &
wait

for batch in "t_l1 t_l2" "t_l3b t_l4" "t_l5"; do
  set -- $batch
  bash $IN/warm_c20261020.sh $1 2 0-15,64-79 29300 $D/warm_$1 &
  [ -n "$2" ] && bash $IN/warm_c20261020.sh $2 3 16-31,80-95 29500 $D/warm_$2 &
  wait
done
for tree in t_l1 t_l2 t_l3b t_l4 t_l5; do
  log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
done

PEER=$R/t_l3b PYSPY=0 CELLS="c16:16:320" bash $IN/cosy_profile_run_c20261020.sh t_l2 2 0-15,64-79 29700 $R/t_l2 \
  > $R/run_t_l2.log 2>&1 &
PEER=$R/t_l2 PYSPY=0 CELLS="c16:16:320" bash $IN/cosy_profile_run_c20261020.sh t_l3b 3 16-31,80-95 29900 $R/t_l3b \
  > $R/run_t_l3b.log 2>&1 &
wait
log "profiles $(tail -n 1 $R/t_l2/progress.txt) / $(tail -n 1 $R/t_l3b/progress.txt)"
python3 $IN/tokenize_ranges_c20261020.py l2=$R/t_l2/c16.sqlite l3=$R/t_l3b/c16.sqlite > $R/tokenize_ranges.txt 2>&1
cat $R/tokenize_ranges.txt >> $D/queue_j.txt

python3 $IN/worst_step_c20261020.py /workspace/t_l4 3 16-31,80-95 30100 $D/worst_l4_16 16 $D/worst/meta.lst \
  > $D/worst_l4_16.txt 2>&1 &
python3 $IN/worst_step_c20261020.py /workspace/t_l4 2 0-15,64-79 30300 $D/worst_l4_32 32 $D/worst32/meta.lst \
  > $D/worst_l4_32.txt 2>&1 &
wait
for rows in 16 32; do
  log "worst step $rows rows: $(grep -E 'W[0-9]+:|c64:' $D/worst_l4_$rows.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-300)"
  grep 'alloc retries' $D/worst_l4_$rows.txt | sed 's/^/    /' >> $D/queue_j.txt
done

pair t_l1 t_l2 p_l2 c1,c1u,c16,c64,r16
pair t_l2 t_l3b p_l3 c1,c1u,c16,c64,r16
pair t_l3b t_l4 p_l4 c1,c1u,c16,c64,r16,L16
pair t_l4 t_l5 p_l5 c1,c1u,c16,c64,r16,L16
pair t_m21 t_l5 p_all c1,c1u,c16,c64,r16,L16
for dir in p_l3/a/gpu0/c16 p_l3/a/gpu1/c16 p_l3/b/gpu0/c16 p_l3/b/gpu1/c16 \
  p_all/a/gpu0/c16 p_all/a/gpu1/c16 p_all/b/gpu0/c16 p_all/b/gpu1/c16; do
  score $D/$dir
done
log "queue done"
