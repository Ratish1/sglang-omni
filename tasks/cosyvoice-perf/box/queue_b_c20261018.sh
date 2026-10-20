#!/usr/bin/env bash
# 10-18 lease, cards 2 and 3: PR B on the buffered path, which its Snake and spectra kernels reach (the vocoder's
# whole utterance HiFT call, mel2wav_batch, eager) and which no streaming pair covered. Pair p_hgb: upstream main
# (t_main3, 45588d502) against PR B (t_hg3, 45f694edf), buffered b1 (256), b16 and b64, both orders; then WER and SIM
# of both arms' b16 audio in both orders, on card 2.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_b.pgid; exec bash queue_b_c20261018.sh'
D=/data/c20261018
IN=$D/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_b.txt; }
score() {
  local dir=$1
  (cd /workspace/t_main3 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $?"
  (cd /workspace/t_main3 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $?"
}

log "queue start"
cd /workspace/t_main3
PAIR_CELLS=b1,b16,b64 python3 $IN/serve_pair_c20261018.py /workspace/t_main3 /workspace/t_hg3 $D/p_hgb > $D/p_hgb.log 2>&1
log "p_hgb rc $?"
for dir in p_hgb/a/gpu0 p_hgb/a/gpu1 p_hgb/b/gpu0 p_hgb/b/gpu1; do
  score $D/$dir/b16
done
log "queue done"
