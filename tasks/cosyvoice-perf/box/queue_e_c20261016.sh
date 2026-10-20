#!/usr/bin/env bash
# 10-16 second box, after queue_d: A2 (t_a2, on A1 on A3) units, the pairs t_a3 against t_a2 (c1, c16, c64) and t_a3
# against t_a1 (c1, c16), both orders, then WER and SIM of each arm's c16 audio for the arms that change numerics (A1
# unseeded redraw draws, A2's min_new_tokens a step late, V3's D) and their baselines, order a, card 2 and card 3 in
# parallel after one scorer alone fills the scoring models' cache.
# usage (in the container): setsid bash -c 'echo $$ > queue_e.pgid; exec bash queue_e_c20261016.sh'
D=/data/c20261016
IN=$D/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_e.txt; }
units() {
  local tree=$1 card=$2 cpus=$3 out=$D/units_e/$1
  mkdir -p $out
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card taskset -c $cpus python3 -m pytest -q -p no:cacheprovider \
    tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner > $out/units.log 2>&1)
  log "units $tree rc $?"
}
score() {
  local dir=$1 card=$2 cpus=$3 port=$4
  (cd /workspace/sglang-omni && CUDA_VISIBLE_DEVICES=$card taskset -c $cpus python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $port --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $?"
  (cd /workspace/sglang-omni && CUDA_VISIBLE_DEVICES=$card taskset -c $cpus python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $?"
}

log "queue start"
units t_a2 2 0-15,64-79 &
units t_spectra 3 16-31,80-95
wait

cd /workspace/sglang-omni
PAIR_CELLS=c1,c16,c64 python3 $IN/serve_pair_c20261016.py /workspace/t_a3 /workspace/t_a2 $D/p_a2 > $D/p_a2.log 2>&1
log "p_a2 rc $?"
PAIR_CELLS=c1,c16 python3 $IN/serve_pair_c20261016.py /workspace/t_a3 /workspace/t_a1 $D/p_a1 > $D/p_a1.log 2>&1
log "p_a1 rc $?"

score $D/p_a2/a/gpu0/c16 2 0-15,64-79 26000
score $D/p_a2/a/gpu1/c16 2 0-15,64-79 26000 &
score $D/p_a1/a/gpu1/c16 3 16-31,80-95 27000
wait
score $D/p_v3b/a/gpu0/c16 2 0-15,64-79 26000 &
score $D/p_v3b/a/gpu1/c16 3 16-31,80-95 27000
wait
score $D/p_v3a/a/gpu1/c16 2 0-15,64-79 26000 &
score $D/p_a2/b/gpu1/c16 3 16-31,80-95 27000
wait
score $D/p_a2/b/gpu0/c16 2 0-15,64-79 26000
log "queue done"
