#!/usr/bin/env bash
# 10-20, cards 2 and 3, after queue a. P1, the AR's prefill replayed from full CUDA graphs on CUDA
# (perf/cosyvoice3-prefill-graphs ef2665d60 on main b98a28936, tree t_p20), and the candidate with it
# (prof/cosyvoice3-candidate-prefill 708168120, tree t_cp20). The touched suites, warm (the boot logs read for the
# prefill graph capture), main and P1 profiled at the same time (c16, c64: the EXTEND steps' launches, wall and
# stall), pairs p_p1 (main against P1) and p_cp (the candidate against the candidate with P1), all cells both
# orders, then WER and SIM of p_p1's c16 audio.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_b.pgid; exec bash queue_b_c20261020.sh'
D=/data/c20261020
IN=$D/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
M=t_m20
C=t_c20
P=t_p20
S=t_cp20
R=$D/prof_b
log() { echo "$* $(date +%T)" >> $D/queue_b.txt; }
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

log "queue start"
mkdir -p $R $D/units
cd /workspace/sglang-omni
for pair in "$P perf/cosyvoice3-prefill-graphs" "$S prof/cosyvoice3-candidate-prefill"; do
  set -- $pair
  git fetch -q origin $2
  [ -d /workspace/$1 ] || git worktree add -q --detach /workspace/$1 FETCH_HEAD
  log "tree $1 $(git -C /workspace/$1 log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$1 rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$1 diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

for pair in "$P 2 0-15,64-79" "$S 3 16-31,80-95"; do
  set -- $pair
  (
    cd /workspace/$1
    CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=/workspace/$1 taskset -c $3 python3 -m pytest -q -p no:cacheprovider \
      tests/unit_test/fun_cosyvoice3 tests/unit_test/scheduling/test_streaming_vocoder.py \
      tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/model_runner/test_sglang_execution.py \
      tests/unit_test/qwen3_omni/test_talker_token_readback.py tests/unit_test/qwen3_tts/test_decode_step_staging.py \
      > $D/units/units_$1.log 2>&1
    rc=$?
    echo "units $1 rc $rc $(grep -E 'passed|failed' $D/units/units_$1.log | tail -n 1) $(date +%T)" >> $D/queue_b.txt
  ) &
done
wait

bash $IN/warm_c20261020.sh $P 2 0-15,64-79 26300 $D/warm_$P &
bash $IN/warm_c20261020.sh $S 3 16-31,80-95 26500 $D/warm_$S &
wait
for tree in $P $S; do
  log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
  grep -i -E 'prefill.*(graph|capture)|capture.*prefill|full prefill|breakable|eager' $D/warm_$tree/serve.log | cut -c1-240 \
    > $D/warm_$tree/prefill_graph_lines.txt
  log "prefill graph lines $tree $(wc -l < $D/warm_$tree/prefill_graph_lines.txt)"
done

PEER=$R/$P PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261020.sh $M 2 0-15,64-79 27600 \
  $R/$M > $R/run_$M.log 2>&1 &
PEER=$R/$M PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261020.sh $P 3 16-31,80-95 27800 \
  $R/$P > $R/run_$P.log 2>&1 &
wait
log "profiles $(tail -n 1 $R/$M/progress.txt) / $(tail -n 1 $R/$P/progress.txt)"
for cell in c16 c64; do
  python3 $IN/request_states_c20261019.py main=$R/$M/$cell.sqlite p1=$R/$P/$cell.sqlite > $R/states_$cell.txt 2>&1
done
log "analysis done"

(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261020.py /workspace/$M \
  /workspace/$P $D/p_p1 > $D/p_p1.log 2>&1)
log "pair p_p1 rc $?"
(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261020.py /workspace/$C \
  /workspace/$S $D/p_cp > $D/p_cp.log 2>&1)
log "pair p_cp rc $?"
for dir in p_p1/a/gpu0/c16 p_p1/a/gpu1/c16 p_p1/b/gpu0/c16 p_p1/b/gpu1/c16; do
  score $D/$dir
done
log "queue done"
