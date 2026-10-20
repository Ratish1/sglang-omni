#!/usr/bin/env bash
# 10-16 second box, replaces queue_f (stopped in its unit phase: it predated the smoke gate, and its Snake and V4
# units fail only the batch invariance test D's matmuls break, which the spectra kernels fix). Units of the spectra
# head on card 2 while the A1 and A2 smoke tests finish on card 3, then the pairs, c1, c16, c64 (A1 c1, c16), both
# orders: A2 and A1 against A3 if both smoke tests served cleanly, V1 with first hops' steps free of held hops against
# A3, Snake on V3 (both with D as matmuls), V4 on Snake, the batch invariant spectra kernels on V4, the bounded wait on
# the spectra head; last, WER and SIM of the A arms' and the spectra pair's c16 audio, order a, on card 2.
# usage (in the container): setsid bash -c 'echo $$ > queue_f2.pgid; exec bash queue_f2_c20261016.sh'
D=/data/c20261016
IN=$D/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_f2.txt; }
pair() {
  local cells=$1 first=$2 second=$3 out=$4
  PAIR_CELLS=$cells python3 $IN/serve_pair_c20261016.py /workspace/$first /workspace/$second $D/$out \
    > $D/$out.log 2>&1
  log "$out rc $?"
}
score() {
  local dir=$1
  (cd /workspace/sglang-omni && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $?"
  (cd /workspace/sglang-omni && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $?"
}

log "queue start"
mkdir -p $D/units_f2/t_spectra
(cd /workspace/t_spectra && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner > $D/units_f2/t_spectra/units.log 2>&1)
log "units t_spectra rc $?"
until [ "$(grep -l "^done\|server died" $D/smoke_a2/progress.txt $D/smoke_a1/progress.txt 2>/dev/null | wc -l)" = 2 ]; do
  sleep 10
done
log "smoke a2 $(tail -1 $D/smoke_a2/progress.txt) tracebacks $(grep -c Traceback $D/smoke_a2/serve.log)"
log "smoke a1 $(tail -1 $D/smoke_a1/progress.txt) tracebacks $(grep -c Traceback $D/smoke_a1/serve.log)"

cd /workspace/sglang-omni
if grep -q "bench rc 0" $D/smoke_a2/progress.txt && ! grep -q Traceback $D/smoke_a2/serve.log \
  && grep -q "bench rc 0" $D/smoke_a1/progress.txt && ! grep -q Traceback $D/smoke_a1/serve.log; then
  pair c1,c16,c64 t_a3 t_a2 p_a2
  pair c1,c16 t_a3 t_a1 p_a1
else
  log "p_a2 p_a1 skipped"
fi
pair c1,c16,c64 t_a3 t_v1b p_v1b
pair c1,c16,c64 t_v3b t_snake p_snake
pair c1,c16,c64 t_snake t_v4 p_v4
pair c1,c16,c64 t_v4 t_spectra p_spectra
pair c1,c16,c64 t_spectra t_v1c p_v1c
for dir in p_a2/a/gpu0 p_a2/a/gpu1 p_a1/a/gpu1 p_a2/b/gpu0 p_a2/b/gpu1 p_spectra/a/gpu0 p_spectra/a/gpu1; do
  if [ -d $D/$dir/c16 ]; then
    score $D/$dir/c16
  else
    log "no audio $dir"
  fi
done
log "queue done"
