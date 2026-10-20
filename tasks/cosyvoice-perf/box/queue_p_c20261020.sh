#!/usr/bin/env bash
# 10-20, cards 2 and 3. L7's first change on L6 (perf/cosyvoice3-fa3-scheduler-metadata d8b4d68ae, tree t_l7): the
# prefix solve computes FA3's scheduler metadata once and passes it to every layer and Euler step, where each FA3 call
# ran prepare_varlen_num_blocks and its memsets (660 of a prefix hop step's 3,021 kernels, `solve_census.txt`). Waits
# for queue o, then the units and the census of t_l7, its warm, pair p_l7 (t_l6e against t_l7), WER and SIM of its c16.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_p.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_p_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_p.txt; }
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

log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
cd /workspace/sglang-omni
git fetch -q origin perf/cosyvoice3-fa3-scheduler-metadata
[ -d /workspace/t_l7 ] || git worktree add -q --detach /workspace/t_l7 FETCH_HEAD
log "tree t_l7 $(git -C /workspace/t_l7 log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/t_l7 rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/t_l7 diff --quiet HEAD -- sglang_omni || echo DIRTY)"

until grep -q "queue done" $D/queue_o.txt 2> /dev/null; do sleep 30; done
[ -d $D/inductor_t_l7 ] || cp -a $D/inductor_t_l6e $D/inductor_t_l7
(cd /workspace/t_l7 && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/t_l7 taskset -c 16-31,80-95 python3 -m pytest -q \
  -p no:cacheprovider tests/unit_test/fun_cosyvoice3 > $D/units/units_t_l7.log 2>&1)
log "units t_l7 rc $? $(grep -E 'passed|failed' $D/units/units_t_l7.log | tail -n 1)"
(cd /workspace/t_l7 && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/t_l7:$COSYVOICE TORCHINDUCTOR_CACHE_DIR=$D/inductor_t_l7 \
  taskset -c 0-15,64-79 python3 $IN/solve_census_c20261020.py $D/solve_census_l7.txt > $D/solve_census_l7.log 2>&1)
log "census t_l7 rc $? $(grep kernels $D/solve_census_l7.txt | tr '\n' ' ') prepare $(grep -c prepare_varlen $D/solve_census_l7.txt) $(grep prepare_varlen $D/solve_census_l7.txt | head -n 2 | tr -s ' ' | cut -c1-40 | tr '\n' ' ')"

bash $IN/warm_c20261020.sh t_l7 3 16-31,80-95 38400 $D/warm_t_l7
log "warm t_l7 $(grep 'bench rc' $D/warm_t_l7/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_t_l7/bench.log | tr -s ' ' | tr '\n' ' ') token cap hits $(python3 -c "import json; print(json.load(open('$D/warm_t_l7/bench/speed_results.json'))['summary']['max_token_hits'])" 2>&1 | tail -n 1) tracebacks $(grep -c Traceback $D/warm_t_l7/serve.log) recompiles $(grep -ci recompil $D/warm_t_l7/serve.log)"

(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16,L16 python3 $IN/serve_pair_c20261020.py /workspace/t_l6e \
  /workspace/t_l7 $D/p_l7 > $D/p_l7.log 2>&1)
log "pair p_l7 rc $? token cap hits $(grep -E '\| at the token cap \|' $D/p_l7.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
for dir in a/gpu0 a/gpu1 b/gpu0 b/gpu1; do
  score $D/p_l7/$dir/c16
done
log "queue done"
