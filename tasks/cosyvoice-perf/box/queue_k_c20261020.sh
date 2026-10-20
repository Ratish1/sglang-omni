#!/usr/bin/env bash
# 10-20, cards 2 and 3, after queue j. L6 of DESIGN_PR_STACK_20261020.md, the prefix hop graphs on the prefill frame
# ladder (perf/cosyvoice3-prefix-graph-ladder b72425833, tree t_l6, on L5): units, warm (its boot log holds the 38
# tier capture and the dry capture's reserve), the Euler solve census for L7 (solve_census_c20261020.py), pair p_l6
# (t_l5 against t_l6), WER and SIM of p_l6 c16.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_k.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_k_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
log() { echo "$* $(date +%T)" >> $D/queue_k.txt; }
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

until grep -q "queue done" $D/queue_j.txt 2> /dev/null; do sleep 30; done
log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
cd /workspace/sglang-omni
git fetch -q origin perf/cosyvoice3-prefix-graph-ladder
[ -d /workspace/t_l6 ] || git worktree add -q --detach /workspace/t_l6 FETCH_HEAD
log "tree t_l6 $(git -C /workspace/t_l6 log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/t_l6 rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/t_l6 diff --quiet HEAD -- sglang_omni || echo DIRTY)"

(
  cd /workspace/t_l6
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/t_l6 taskset -c 0-15,64-79 python3 -m pytest -q -p no:cacheprovider \
    tests/unit_test/fun_cosyvoice3 tests/unit_test/test_stage_device_contract.py \
    tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/scheduling/test_streaming_vocoder.py \
    > $D/units/units_t_l6.log 2>&1
  echo "units t_l6 rc $? $(grep -E 'passed|failed' $D/units/units_t_l6.log | tail -n 1) $(date +%T)" >> $D/queue_k.txt
) &
bash $IN/warm_c20261020.sh t_l6 3 16-31,80-95 30500 $D/warm_t_l6 &
wait
log "warm t_l6 $(grep 'bench rc' $D/warm_t_l6/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_t_l6/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_t_l6/serve.log)"
grep -E 'prefix graphs captured|prefix cache reserve|Flow prefix cache: [0-9]+ frames' $D/warm_t_l6/serve.log | cut -c60-300 \
  | sed 's/^/    /' >> $D/queue_k.txt

(cd /workspace/t_l6 && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/t_l6:$COSYVOICE TORCHINDUCTOR_CACHE_DIR=$D/inductor_t_l6 \
  taskset -c 16-31,80-95 python3 $IN/solve_census_c20261020.py $D/solve_census.txt > $D/solve_census.log 2>&1)
log "census rc $? $(grep -c kernels $D/solve_census.txt) steps"

(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16,L16 python3 $IN/serve_pair_c20261020.py /workspace/t_l5 \
  /workspace/t_l6 $D/p_l6 > $D/p_l6.log 2>&1)
log "pair p_l6 rc $?"
for dir in p_l6/a/gpu0/c16 p_l6/a/gpu1/c16 p_l6/b/gpu0/c16 p_l6/b/gpu1/c16; do
  score $D/$dir
done
log "queue done"
