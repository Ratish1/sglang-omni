#!/usr/bin/env bash
# 10-20, cards 2 and 3. Queue j again on the fixed L3 (the runner keeps the model its graphs read: a2f70194a; queue j's
# L3 freed it, so replays read reused memory and requests ran to the token cap) with L6 added. Trees: t_m21 main
# bcd0d8c68, t_l1 8202dcb25, t_l2 b494c6d29, t_l3c a2f70194a, t_l4c 93d0e252c, t_l5c 362d430b3, t_l6c 9efbb325f
# (perf/cosyvoice3-prefix-graph-ladder). Units, warms on ports 400 apart (each warm logs its token cap hits), L2 and L3
# profiled at the same time at c16, the worst step check of L4, the Euler solve census on L6, a pair per layer on the
# layer below and the stack on main, WER and SIM of p_l3, p_l6 and p_all c16.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_l.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_l_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
R=$D/prof_l
log() { echo "$* $(date +%T)" >> $D/queue_l.txt; }
units() {
  (
    cd /workspace/$1
    CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=/workspace/$1 taskset -c $3 python3 -m pytest -q -p no:cacheprovider \
      tests/unit_test/fun_cosyvoice3 tests/unit_test/test_stage_device_contract.py \
      tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/scheduling/test_streaming_vocoder.py \
      tests/unit_test/model_runner/test_sglang_execution.py > $D/units/units_$1.log 2>&1
    echo "units $1 rc $? $(grep -E 'passed|failed' $D/units/units_$1.log | tail -n 1) $(date +%T)" >> $D/queue_l.txt
  )
}
warm_report() {
  log "warm $1 $(grep 'bench rc' $D/warm_$1/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$1/bench.log | tr -s ' ' | tr '\n' ' ') token cap hits $(python3 -c "import json; print(json.load(open('$D/warm_$1/bench/speed_results.json'))['summary']['max_token_hits'])" 2>&1 | tail -n 1) tracebacks $(grep -c Traceback $D/warm_$1/serve.log)"
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
  log "pair $3 rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$3.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
}

log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
mkdir -p $D/units $R
cd /workspace/sglang-omni
for pair in "t_l3c perf/cosyvoice3-speech-tokenizer-graphs" "t_l4c perf/cosyvoice3-prefix-pool-sized-v2" \
  "t_l5c perf/cosyvoice3-prefill-graphs-v2" "t_l6c perf/cosyvoice3-prefix-graph-ladder"; do
  set -- $pair
  git fetch -q origin $2
  if [ -d /workspace/$1 ]; then
    git -C /workspace/$1 checkout -q --detach FETCH_HEAD
  else
    git worktree add -q --detach /workspace/$1 FETCH_HEAD
  fi
done
for tree in t_m21 t_l1 t_l2 t_l3c t_l4c t_l5c t_l6c; do
  log "tree $tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$tree rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

(units t_l3c 2 0-15,64-79; units t_l5c 2 0-15,64-79) &
(units t_l4c 3 16-31,80-95; units t_l6c 3 16-31,80-95) &
wait

bash $IN/warm_c20261020.sh t_l2 2 0-15,64-79 32000 $D/warm_t_l2 &
bash $IN/warm_c20261020.sh t_l3c 3 16-31,80-95 32400 $D/warm_t_l3c &
wait
bash $IN/warm_c20261020.sh t_l4c 2 0-15,64-79 32800 $D/warm_t_l4c &
bash $IN/warm_c20261020.sh t_l5c 3 16-31,80-95 33200 $D/warm_t_l5c &
wait
bash $IN/warm_c20261020.sh t_l6c 2 0-15,64-79 33600 $D/warm_t_l6c
for tree in t_l2 t_l3c t_l4c t_l5c t_l6c; do
  warm_report $tree
done
grep -E 'prefix graphs captured|prefix cache reserve|Flow prefix cache: [0-9]+ frames' $D/warm_t_l6c/serve.log | cut -c60-300 \
  | sed 's/^/    /' >> $D/queue_l.txt

PEER=$R/t_l3c PYSPY=0 CELLS="c16:16:320" bash $IN/cosy_profile_run_c20261020.sh t_l2 2 0-15,64-79 34000 $R/t_l2 \
  > $R/run_t_l2.log 2>&1 &
PEER=$R/t_l2 PYSPY=0 CELLS="c16:16:320" bash $IN/cosy_profile_run_c20261020.sh t_l3c 3 16-31,80-95 34400 $R/t_l3c \
  > $R/run_t_l3c.log 2>&1 &
wait
log "profiles $(tail -n 1 $R/t_l2/progress.txt) / $(tail -n 1 $R/t_l3c/progress.txt)"
python3 $IN/tokenize_ranges_c20261020.py l2=$R/t_l2/c16.sqlite l3=$R/t_l3c/c16.sqlite > $R/tokenize_ranges.txt 2>&1
cat $R/tokenize_ranges.txt >> $D/queue_l.txt

python3 $IN/worst_step_c20261020.py /workspace/t_l4c 3 16-31,80-95 34800 $D/worst_l4c_16 16 $D/worst/meta.lst \
  > $D/worst_l4c_16.txt 2>&1 &
python3 $IN/worst_step_c20261020.py /workspace/t_l4c 2 0-15,64-79 35200 $D/worst_l4c_32 32 $D/worst32/meta.lst \
  > $D/worst_l4c_32.txt 2>&1 &
wait
for rows in 16 32; do
  log "worst step $rows rows: $(grep -E 'W[0-9]+:|c64:' $D/worst_l4c_$rows.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-330)"
  grep 'alloc retries' $D/worst_l4c_$rows.txt | sed 's/^/    /' >> $D/queue_l.txt
done

(cd /workspace/t_l6c && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/t_l6c:$COSYVOICE TORCHINDUCTOR_CACHE_DIR=$D/inductor_t_l6c \
  taskset -c 16-31,80-95 python3 $IN/solve_census_c20261020.py $D/solve_census.txt > $D/solve_census.log 2>&1)
log "census rc $? $(grep -c kernels $D/solve_census.txt) steps"

pair t_l1 t_l2 p_l2 c1,c1u,c16,c64,r16
pair t_l2 t_l3c p_l3 c1,c1u,c16,c64,r16
pair t_l3c t_l4c p_l4 c1,c1u,c16,c64,r16,L16
pair t_l4c t_l5c p_l5 c1,c1u,c16,c64,r16,L16
pair t_l5c t_l6c p_l6 c1,c1u,c16,c64,r16,L16
pair t_m21 t_l6c p_all c1,c1u,c16,c64,r16,L16
for pair_dir in p_l3 p_l6 p_all; do
  for dir in a/gpu0 a/gpu1 b/gpu0 b/gpu1; do
    score $D/$pair_dir/$dir/c16
  done
done
log "queue done"
