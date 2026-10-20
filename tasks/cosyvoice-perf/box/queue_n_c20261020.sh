#!/usr/bin/env bash
# 10-20, cards 2 and 3. L3 with TF32 matmuls (perf/cosyvoice3-speech-tokenizer-graphs 43b948458, tree t_l3d): in
# p_l3 its fp32 replays took 10.6 ms of device time each in serving (`prof_l`, 205 replays), against about 4.3 ms of
# kernels for ONNX Runtime, and slowed the vocoder (c16 req/s -6 %, r16 inter chunk +15 to +20 %). The stack rebased
# on it: t_l6d 76633f384 (perf/cosyvoice3-prefix-graph-ladder). Waits for queue l's p_l6 (the last layer pair valid
# as a delta), then stops queue l (its p_all measures the fp32 L3) and queue m, runs the census, the units, WER and SIM
# of p_l6, warms, pairs p_l3d (t_l2 against t_l3d) and p_alld (t_m21 against t_l6d), WER and SIM of both c16.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_n.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_n_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
log() { echo "$* $(date +%T)" >> $D/queue_n.txt; }
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

until grep -q "pair p_l6" $D/queue_l.txt 2> /dev/null; do sleep 30; done
for queue in queue_l queue_m; do
  kill -TERM -- -$(cat $D/$queue.pgid) 2> /dev/null
done
sleep 5
for group in $(ps -eo pgid,cmd | grep -E 'sglang_omni.cli serve|serve_pair_c20261020|benchmark_tts_seedtts' | grep -v grep | awk '{print $1}' | sort -u); do
  kill -TERM -- -$group
done
sleep 20
for group in $(ps -eo pgid,cmd | grep -E 'sglang_omni.cli serve|serve_pair_c20261020|benchmark_tts_seedtts' | grep -v grep | awk '{print $1}' | sort -u); do
  kill -KILL -- -$group
done
echo "stopped by queue n after p_l6: its p_all measured the fp32 L3 $(date +%T)" >> $D/queue_l.txt
log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
cd /workspace/sglang-omni
for pair in "t_l3d perf/cosyvoice3-speech-tokenizer-graphs" "t_l6d perf/cosyvoice3-prefix-graph-ladder"; do
  set -- $pair
  git fetch -q origin $2
  [ -d /workspace/$1 ] || git worktree add -q --detach /workspace/$1 FETCH_HEAD
  log "tree $1 $(git -C /workspace/$1 log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$1 rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$1 diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

(cd /workspace/t_l6c && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/t_l6c:$COSYVOICE TORCHINDUCTOR_CACHE_DIR=$D/inductor_t_l6c \
  taskset -c 16-31,80-95 python3 $IN/solve_census_c20261020.py $D/solve_census.txt > $D/solve_census.log 2>&1
  echo "census rc $? $(grep -c kernels $D/solve_census.txt) steps $(date +%T)" >> $D/queue_n.txt) &
for tree in t_l3d t_l6d; do
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$tree taskset -c 0-15,64-79 python3 -m pytest -q \
    -p no:cacheprovider tests/unit_test/fun_cosyvoice3 tests/unit_test/test_stage_device_contract.py \
    tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/scheduling/test_streaming_vocoder.py \
    > $D/units/units_$tree.log 2>&1)
  log "units $tree rc $? $(grep -E 'passed|failed' $D/units/units_$tree.log | tail -n 1)"
done
for dir in a/gpu0 a/gpu1 b/gpu0 b/gpu1; do
  score $D/p_l6/$dir/c16
done
wait

bash $IN/warm_c20261020.sh t_l3d 2 0-15,64-79 36000 $D/warm_t_l3d &
bash $IN/warm_c20261020.sh t_l6d 3 16-31,80-95 36400 $D/warm_t_l6d &
wait
for tree in t_l3d t_l6d; do
  log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') token cap hits $(python3 -c "import json; print(json.load(open('$D/warm_$tree/bench/speed_results.json'))['summary']['max_token_hits'])" 2>&1 | tail -n 1) tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
done

pair t_l2 t_l3d p_l3d c1,c1u,c16,c64,r16
pair t_m21 t_l6d p_alld c1,c1u,c16,c64,r16,L16
for pair_dir in p_l3d p_alld; do
  for dir in a/gpu0 a/gpu1 b/gpu0 b/gpu1; do
    score $D/$pair_dir/$dir/c16
  done
done
log "queue done"
