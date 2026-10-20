#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: CosyVoice drawing unseeded requests with FlashInfer (t_fi, the #2846 change for
# Fun-CosyVoice3) against upstream main (t_main). Main warms its Inductor cache on card 2 while card 3 runs both
# trees' touched unit suites, then t_fi's serving smoke test on card 3 (64 c16 requests, no traceback) gates the pair:
# seeded c1 (identity), unseeded c1, c16, c64, both orders; then the seeded c1 identity counts and WER and SIM of the
# c16 audio of both arms in both orders, on card 2.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_a.pgid; exec bash queue_a_c20261017.sh'
D=/data/c20261017
IN=$D/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_a.txt; }
units() {
  local tree=$1
  mkdir -p $D/units/$tree
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
    tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner/test_install_sampling_seeds.py tests/unit_test/sampling \
    > $D/units/$tree/units.log 2>&1)
  log "units $tree rc $? $(tail -1 $D/units/$tree/units.log)"
}
score() {
  local dir=$1
  (cd /workspace/t_main && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $?"
  (cd /workspace/t_main && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $?"
}

log "queue start"
bash $IN/warm_c20261017.sh t_main 2 0-15,64-79 21000 $D/warm_main &
WARM=$!
units t_fi
units t_main
bash $IN/warm_c20261017.sh t_fi 3 16-31,80-95 22000 $D/smoke_fi
log "smoke fi $(grep 'bench rc' $D/smoke_fi/progress.txt) tracebacks $(grep -c Traceback $D/smoke_fi/serve.log)"
wait $WARM
log "warm main $(grep 'bench rc' $D/warm_main/progress.txt) tracebacks $(grep -c Traceback $D/warm_main/serve.log)"

cd /workspace/t_main
if grep -q "bench rc 0" $D/smoke_fi/progress.txt && ! grep -q Traceback $D/smoke_fi/serve.log; then
  PAIR_CELLS=c1,c1u,c16,c64 python3 $IN/serve_pair_c20261017.py /workspace/t_main /workspace/t_fi $D/p_fi > $D/p_fi.log 2>&1
  log "p_fi rc $?"
  python3 $IN/c1_identity_c20261017.py $D/p_fi > $D/p_fi_identity.txt 2>&1
  log "identity $(tr '\n' ' ' < $D/p_fi_identity.txt)"
  for dir in p_fi/a/gpu0 p_fi/a/gpu1 p_fi/b/gpu0 p_fi/b/gpu1; do
    score $D/$dir/c16
  done
else
  log "p_fi skipped"
fi
log "queue done"
