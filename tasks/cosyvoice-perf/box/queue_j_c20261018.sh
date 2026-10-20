#!/usr/bin/env bash
# 10-18, cards 2 and 3: A1 and A2 rebased (t_la, perf/cosyvoice3-ar-lookahead-v2 5935c286d: the device repeat window,
# the AR's decode steps launched one ahead on CUDA, the compiled repeat mask) against their base t_ic (d50266ba6).
# Waits for queue_i. Units, the warm smoke test, the pair p_la (c1 seeded, c1u, c16, c64, r16, both orders) with its
# gap breakdown, the kit profile of both at the same time (c16, r16) with each chunk split, then WER and SIM of
# p_la's c16 audio, both arms in both orders, on card 2.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_j.pgid; exec bash queue_j_c20261018.sh'
D=/data/c20261018
IN=$D/in
B=t_ic
L=t_la
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
CELLS="c16:16:640 r16:64:640::16"
log() { echo "$* $(date +%T)" >> $D/queue_j.txt; }
score() {
  local dir=$1
  (cd /workspace/t_main4 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 28100 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $?"
  (cd /workspace/t_main4 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $?"
}

log "queue wait"
until grep -q "queue done" $D/queue_i.txt; do sleep 30; done
log "queue start"
mkdir -p $D/units/$L
(cd /workspace/$L && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py \
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3 > $D/units/$L/units.log 2>&1)
log "units $L rc $? $(grep -E 'passed|failed' $D/units/$L/units.log | tail -n 1) skipped $(grep -cE 'SKIP|skipped' $D/units/$L/units.log)"
bash $IN/warm_c20261018.sh $L 3 16-31,80-95 26900 $D/warm_la
log "warm warm_la $(grep 'bench rc' $D/warm_la/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_la/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_la/serve.log)"

(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261018.py /workspace/$B \
  /workspace/$L $D/p_la > $D/p_la.log 2>&1)
log "pair p_la rc $?"
python3 $IN/inter_chunk_mechanism_c20261018.py $D/p_la d50266ba6 5935c286d c16 c64 r16 > $D/p_la_mechanism.txt 2>&1

mkdir -p $D/prof_la
PEER=$D/prof_la/la PYSPY=1 CELLS="$CELLS" bash $IN/cosy_profile_run_c20261018.sh $B 2 0-15,64-79 27100 \
  $D/prof_la/ic > $D/prof_la/run_ic.log 2>&1 &
PEER=$D/prof_la/ic PYSPY=1 CELLS="$CELLS" bash $IN/cosy_profile_run_c20261018.sh $L 3 16-31,80-95 27500 \
  $D/prof_la/la > $D/prof_la/run_la.log 2>&1 &
wait
log "profiles $(tail -n 1 $D/prof_la/ic/progress.txt) / $(tail -n 1 $D/prof_la/la/progress.txt)"
for cell in c16 r16; do
  python3 $IN/chunk_timeline_c20261018.py ic=$D/prof_la/ic/$cell.sqlite la=$D/prof_la/la/$cell.sqlite \
    > $D/prof_la/chunks_$cell.txt 2>&1
  python3 $IN/first_audio_timeline_c20261017.py ic=$D/prof_la/ic/$cell.sqlite la=$D/prof_la/la/$cell.sqlite \
    > $D/prof_la/timeline_$cell.txt 2>&1
  log "analyses $cell"
done

for order in a b; do
  for gpu in gpu0 gpu1; do
    score $D/p_la/$order/$gpu/c16
  done
done
log "queue done"
