#!/usr/bin/env bash
# 10-20, cards 2 and 3, after queue f. Item 3 v4 on the PR candidate with P1: prof/cosyvoice3-candidate-prefill-pool
# 87c965c81 (tree t_cpi3: d27c42294, the candidate with P1, then prerequisites 1 and 2, item 3, and the measured step
# waiting for its HiFT samples as the candidate's serving does). The touched suites, warm, the worst step check at 16
# and 32 rows (the candidate's steps launch before the previous one completes), pairs p_ci (t_cp20b against t_cpi3:
# item 3 alone on the candidate with P1) and p_cc (t_c20 against t_cpi3: P1 and item 3 on the candidate, the arm whose
# c64 inter chunk failed with P1 alone), WER and SIM of p_cc c16.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_g.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_g_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
M=t_m20
C=t_c20
S=t_cp20b
G=t_cpi3
log() { echo "$* $(date +%T)" >> $D/queue_g.txt; }
startup() {
  grep -E 'largest admitted step|prefix cache reserve|Flow prefix cache|graphs captured|KV Cache is allocated' \
    $1 | cut -c1-240 | sed 's/^/    /' >> $D/queue_g.txt
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

until grep -q "queue done" $D/queue_f.txt 2> /dev/null; do sleep 30; done
log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
cd /workspace/sglang-omni
git fetch -q origin prof/cosyvoice3-candidate-prefill-pool
[ -d /workspace/$G ] || git worktree add -q --detach /workspace/$G FETCH_HEAD
for tree in $C $S $G; do
  log "tree $tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$tree rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

(
  cd /workspace/$G
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$G taskset -c 0-15,64-79 python3 -m pytest -q -p no:cacheprovider \
    tests/unit_test/fun_cosyvoice3 tests/unit_test/test_stage_device_contract.py \
    tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/scheduling/test_streaming_vocoder.py \
    tests/unit_test/model_runner/test_sglang_execution.py > $D/units/units_$G.log 2>&1
  echo "units $G rc $? $(grep -E 'passed|failed' $D/units/units_$G.log | tail -n 1) $(date +%T)" >> $D/queue_g.txt
) &
bash $IN/warm_c20261020.sh $G 3 16-31,80-95 27700 $D/warm_$G &
wait
log "warm $G $(grep 'bench rc' $D/warm_$G/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$G/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$G/serve.log)"
startup $D/warm_$G/serve.log

python3 $IN/worst_step_c20261020.py /workspace/$G 3 16-31,80-95 27900 $D/worst_cpi3_16 16 $D/worst/meta.lst \
  > $D/worst_cpi3_16.txt 2>&1 &
python3 $IN/worst_step_c20261020.py /workspace/$G 2 0-15,64-79 28100 $D/worst_cpi3_32 32 $D/worst32/meta.lst \
  > $D/worst_cpi3_32.txt 2>&1 &
wait
log "worst step 16 rows $(grep -c . $D/worst_cpi3_16.txt) lines, 32 rows $(grep -c . $D/worst_cpi3_32.txt) lines"

pair $S $G p_ci c1,c1u,c16,c64,r16,L16
pair $C $G p_cc c1,c1u,c16,c64,r16,L16
for dir in p_cc/a/gpu0/c16 p_cc/a/gpu1/c16 p_cc/b/gpu0/c16 p_cc/b/gpu1/c16; do
  score $D/$dir
done
log "queue done"
