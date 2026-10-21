#!/usr/bin/env bash
# 10-21, cards 2 and 3. The stack L2 to L6 rebased onto the L1 PR branch (perf/cosyvoice3-overlapped-vocoder-steps,
# tree t_pr), each layer paired alone on the one below, both orders, every reference: p_t2 (t_pr against t_l2),
# p_t3 (t_l2 against t_l3, WER and SIM), p_t4 (t_l3 against t_l4, the worst step check at 16 and 32 rows), p_t5
# (t_l4 against t_l5), p_t6 (t_l5 against t_l6, WER and SIM); L16 where the layer touches the vocoder. First the
# lists, the trees' warms (each tree's Inductor cache, and its smoke) and the touched suites of every layer.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261021/queue_t.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_t_c20261021.sh'
D=/data/c20261021
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
SUITES="tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner tests/unit_test/scheduling
  tests/unit_test/qwen3_omni/test_talker_token_readback.py tests/unit_test/qwen3_tts/test_decode_step_staging.py
  tests/unit_test/moss_tts_local/test_state_pool.py"
log() { echo "$* $(date +%T)" >> $D/queue_t.txt; }
score() {
  local dir=$1
  (cd /workspace/t_m22 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_m22 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
pair() {
  local first=$1 second=$2 name=$3 cells=$4
  (cd /workspace/sglang-omni && PAIR_CELLS=$cells python3 $IN/serve_pair_c20261021.py /workspace/$first \
    /workspace/$second $D/$name > $D/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
}

log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
cd /workspace/sglang-omni
for spec in t_l2:perf/cosyvoice3-expandable-segments-v3 t_l3:perf/cosyvoice3-speech-tokenizer-graphs \
  t_l4:perf/cosyvoice3-prefix-pool-sized-v2 t_l5:perf/cosyvoice3-prefill-graphs-v2 \
  t_l6:perf/cosyvoice3-prefix-graph-ladder; do
  tree=${spec%%:*} branch=${spec#*:}
  git fetch -q origin $branch
  [ -d /workspace/$tree ] || git worktree add -q --detach /workspace/$tree FETCH_HEAD
done
for tree in t_m22 t_pr t_l2 t_l3 t_l4 t_l5 t_l6; do
  log "tree $tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$tree rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

(cd /workspace/t_m22 && PYTHONPATH=/workspace/t_m22 python3 $IN/make_longform_meta.py --out $D/longform > $D/lists.log 2>&1 &&
  PYTHONPATH=/workspace/t_m22 python3 $IN/make_worst_meta_c20261011.py --out $D/worst --requests 16 >> $D/lists.log 2>&1 &&
  PYTHONPATH=/workspace/t_m22 python3 $IN/make_worst_meta_c20261011.py --out $D/worst32 --requests 32 >> $D/lists.log 2>&1)
log "lists rc $? long form $(wc -l < $D/longform/meta.lst) worst $(wc -l < $D/worst/meta.lst) worst32 $(wc -l < $D/worst32/meta.lst)"

port=40000
for trees in "t_pr t_l2" "t_l3 t_l4" "t_l5 t_l6"; do
  set -- $trees
  bash $IN/warm_c20261021.sh $1 2 0-15,64-79 $port $D/warm_$1 &
  bash $IN/warm_c20261021.sh $2 3 16-31,80-95 $((port + 400)) $D/warm_$2 &
  wait
  port=$((port + 800))
  for tree in $trees; do
    log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
  done
done

for tree in t_l2 t_l3 t_l4 t_l5 t_l6; do
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/$tree:$COSYVOICE taskset -c 16-31,80-95 \
    python3 -m pytest -q -p no:cacheprovider $SUITES > $D/units/units_$tree.log 2>&1)
  log "units $tree rc $? $(grep -E 'passed|failed' $D/units/units_$tree.log | tail -n 1)"
done

pair t_pr t_l2 p_t2 c1f,c16,c64,r16,L16
pair t_l2 t_l3 p_t3 c1f,c16,c64,r16
for dir in a/gpu0 a/gpu1 b/gpu0 b/gpu1; do score $D/p_t3/$dir/c16; done
python3 $IN/worst_step_c20261021.py /workspace/t_l4 3 16-31,80-95 46000 $D/worst_l4_16 16 $D/worst/meta.lst \
  > $D/worst_l4_16.txt 2>&1 &
python3 $IN/worst_step_c20261021.py /workspace/t_l4 2 0-15,64-79 46400 $D/worst_l4_32 32 $D/worst32/meta.lst \
  > $D/worst_l4_32.txt 2>&1 &
wait
for rows in 16 32; do
  log "worst step $rows rows: $(grep -E 'W[0-9]+:|c64:' $D/worst_l4_$rows.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-330)"
  grep 'alloc retries' $D/worst_l4_$rows.txt | sed 's/^/    /' >> $D/queue_t.txt
done
pair t_l3 t_l4 p_t4 c1f,c16,c64,r16,L16
pair t_l4 t_l5 p_t5 c1f,c16,c64,r16,L16
pair t_l5 t_l6 p_t6 c1f,c16,c64,r16,L16
for dir in a/gpu0 a/gpu1 b/gpu0 b/gpu1; do score $D/p_t6/$dir/c16; done
log "queue done"
