#!/usr/bin/env bash
# 10-21, cards 0 and 1, after queue t. #2260's buffered route (PR2260_REVIEW.md 5): t_b0 the PR with main merged
# (307288037), t_b1 our branch on it (perf/cosyvoice3-buffered-packed-flow c6e7515bc), t_b1c one Flow call per packed
# batch on t_b1 (perf/cosyvoice3-buffered-packed-one-call 75681715e), t_l1b1 the L1 PR branch with t_b1 merged
# (tmp/l1-buffered-packed b9103ad1d). Warms (t_m22 included: no queue warmed main in this container), the touched
# suites, then p_u1 main against t_b0 over every buffered cell with WER and SIM, p_u2 t_b1 against t_b1c, p_u3 t_pr
# against t_l1b1, and two profiled pairs: main against t_b1 at b16, t_pr against t_l2 at c64 (L2's c64 regression).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261021/queue_u.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_u_c20261021.sh'
D=/data/c20261021
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
SUITES="tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner tests/unit_test/scheduling
  tests/unit_test/qwen3_omni/test_talker_token_readback.py tests/unit_test/qwen3_tts/test_decode_step_staging.py
  tests/unit_test/moss_tts_local/test_state_pool.py"
log() { echo "$* $(date +%T)" >> $D/queue_u.txt; }
score() {
  local dir=$1
  (cd /workspace/t_m22 && CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_m22 && CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
pair() {
  local first=$1 second=$2 name=$3 cells=$4
  (cd /workspace/sglang-omni && PAIR_CELLS=$cells python3 $IN/serve_pair_c20261021.py /workspace/$first \
    /workspace/$second $D/$name > $D/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
}
profiled() {
  local first=$1 second=$2 name=$3 cells=$4
  PEER=$D/$name/$second CELLS=$cells PYSPY=0 bash $IN/cosy_profile_run_c20261021.sh $first 0 0-15,64-79 48600 \
    $D/$name/$first &
  PEER=$D/$name/$first CELLS=$cells PYSPY=0 bash $IN/cosy_profile_run_c20261021.sh $second 1 16-31,80-95 49000 \
    $D/$name/$second &
  wait
  log "profiled $name $(tail -n 1 $D/$name/$first/progress.txt) ; $(tail -n 1 $D/$name/$second/progress.txt)"
}

while pgrep -g "$(cat $D/queue_t.pgid)" > /dev/null; do sleep 60; done
log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
cd /workspace/sglang-omni
git fetch -q origin perf/cosyvoice3-buffered-packed-flow perf/cosyvoice3-buffered-packed-one-call tmp/l1-buffered-packed
for spec in t_b0:307288037ef4ec3b69ab893dcb2a2abbb4ba2147 t_b1:c6e7515bc5d81a88efbe6adc07f9af31ce90e33a \
  t_b1c:75681715ea6576382c35b04ecfd0ba07a6bf9e9b t_l1b1:b9103ad1d8828f86021100f49eb492cdc3792adf; do
  tree=${spec%%:*} commit=${spec#*:}
  [ -d /workspace/$tree ] || git worktree add -q --detach /workspace/$tree $commit
done
for tree in t_m22 t_pr t_l2 t_b0 t_b1 t_b1c t_l1b1; do
  log "tree $tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$tree rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

port=47000
for trees in "t_m22 t_b0" "t_b1 t_b1c" "t_l1b1"; do
  set -- $trees
  bash $IN/warm_c20261021.sh $1 0 0-15,64-79 $port $D/warm_$1 &
  [ -n "$2" ] && bash $IN/warm_c20261021.sh $2 1 16-31,80-95 $((port + 400)) $D/warm_$2 &
  wait
  port=$((port + 800))
  for tree in $trees; do
    log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
  done
done

mkdir -p $D/units
for trees in "t_b0 t_b1" "t_b1c t_l1b1"; do
  set -- $trees
  card=0
  for tree in $trees; do
    cpus=$([ $card = 0 ] && echo 0-15,64-79 || echo 16-31,80-95)
    (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=/workspace/$tree:$COSYVOICE taskset -c $cpus \
      python3 -m pytest -q -p no:cacheprovider $SUITES > $D/units/units_$tree.log 2>&1) &
    card=1
  done
  wait
  for tree in $trees; do
    log "units $tree $(grep -E 'passed|failed' $D/units/units_$tree.log | tail -n 1)"
  done
done

pair t_m22 t_b0 p_u1 b1f,b16,b64,Lb16
for dir in a/gpu0 a/gpu1 b/gpu0 b/gpu1; do score $D/p_u1/$dir/b16; done
pair t_b1 t_b1c p_u2 b16,b64,Lb16
pair t_pr t_l1b1 p_u3 b16,b64
profiled t_m22 t_b1 prof_u1 b16:16:320::::b
profiled t_pr t_l2 prof_u2 c64:64:640
log "queue done"
