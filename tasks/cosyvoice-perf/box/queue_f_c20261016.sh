#!/usr/bin/env bash
# 10-16 second box, after queue_e: first A2 and A1 (t_a3 against t_a2, c1, c16, c64; t_a3 against t_a1, c1, c16) once
# their serving smoke test passed (A1's first build read a Req attribute 0.5.21 keeps under req.kv), then the Snake kernel on V3 on V1 on A3 (t_snake) and V4, the Flow step inputs without a
# host wait, on it (t_v4): their CosyVoice and model runner units on cards 2 and 3 (the Snake bit identity test among
# them), then V1 with first hops' steps free of held hops against A3 (t_a3 against t_v1b, its streaming tests run on
# the spare socket), and, each only if its units pass, t_v3b against t_snake, t_snake against t_v4 and the full
# stack with the batch invariant spectra kernels against it with every ready row waiting one recent step (t_spectra
# against t_v1c), c1, c16, c64, both orders; last, WER and SIM of the A arms' c16 audio and their baseline on card 2.
# usage (in the container): setsid bash -c 'echo $$ > queue_f.pgid; exec bash queue_f_c20261016.sh'
D=/data/c20261016
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_f.txt; }
units() {
  local tree=$1 card=$2 cpus=$3 out=$D/units_f/$1
  mkdir -p $out
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card taskset -c $cpus python3 -m pytest -q -p no:cacheprovider \
    tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner > $out/units.log 2>&1)
  echo $? > $out/rc
  log "units $tree rc $(cat $out/rc)"
}

log "queue start"
(units t_snake 2 0-15,64-79; units t_spectra 2 0-15,64-79) &
units t_v4 3 16-31,80-95
wait

cd /workspace/sglang-omni
until grep -q "^done\|server died" $D/smoke_a2/progress.txt $D/smoke_a1/progress.txt 2>/dev/null \
  && [ "$(grep -l "^done\|server died" $D/smoke_a2/progress.txt $D/smoke_a1/progress.txt | wc -l)" = 2 ]; do
  sleep 10
done
if grep -q "bench rc 0" $D/smoke_a2/progress.txt && ! grep -q Traceback $D/smoke_a2/serve.log \
  && grep -q "bench rc 0" $D/smoke_a1/progress.txt && ! grep -q Traceback $D/smoke_a1/serve.log; then
  PAIR_CELLS=c1,c16,c64 python3 $IN/serve_pair_c20261016.py /workspace/t_a3 /workspace/t_a2 $D/p_a2 > $D/p_a2.log 2>&1
  log "p_a2 rc $?"
  PAIR_CELLS=c1,c16 python3 $IN/serve_pair_c20261016.py /workspace/t_a3 /workspace/t_a1 $D/p_a1 > $D/p_a1.log 2>&1
  log "p_a1 rc $?"
else
  log "p_a2 p_a1 skipped"
fi
PAIR_CELLS=c1,c16,c64 python3 $IN/serve_pair_c20261016.py /workspace/t_a3 /workspace/t_v1b $D/p_v1b > $D/p_v1b.log 2>&1
log "p_v1b rc $?"
for pair in "t_v3b t_snake p_snake" "t_snake t_v4 p_v4"; do
  set -- $pair
  if [ "$(cat $D/units_f/$2/rc)" = 0 ]; then
    PAIR_CELLS=c1,c16,c64 python3 $IN/serve_pair_c20261016.py /workspace/$1 /workspace/$2 $D/$3 > $D/$3.log 2>&1
    log "$3 rc $?"
  else
    log "$3 skipped"
  fi
done
if [ "$(cat $D/units_f/t_spectra/rc)" = 0 ]; then
  PAIR_CELLS=c1,c16,c64 python3 $IN/serve_pair_c20261016.py /workspace/t_spectra /workspace/t_v1c $D/p_v1c \
    > $D/p_v1c.log 2>&1
  log "p_v1c rc $?"
else
  log "p_v1c skipped"
fi
for dir in p_a2/a/gpu0 p_a2/a/gpu1 p_a1/a/gpu1 p_a2/b/gpu0 p_a2/b/gpu1; do
  if [ -d $D/$dir/c16 ]; then
    (cd /workspace/sglang-omni && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
      --transcribe-only --skip-gpu-cleanup --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --host 127.0.0.1 --port 26000 \
      --lang en --output-dir $D/$dir/c16 > $D/$dir/c16/score_wer.log 2>&1)
    log "wer $dir rc $?"
    (cd /workspace/sglang-omni && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
      --similarity-only --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --lang en --output-dir $D/$dir/c16 \
      > $D/$dir/c16/score_sim.log 2>&1)
    log "sim $dir rc $?"
  else
    log "no audio $dir"
  fi
done
log "queue done"
