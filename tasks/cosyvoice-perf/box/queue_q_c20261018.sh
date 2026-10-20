#!/usr/bin/env bash
# 10-18, card 2: WER and SIM of p_stdl2m's c16 audio (main t_ic against the full stack with the deadline steps v2,
# t_stdl2 c1b50ef9a), both arms in both orders. Waits for queue_p.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_q.pgid; exec bash queue_q_c20261018.sh'
D=/data/c20261018
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_q.txt; }
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
until grep -q "queue done" $D/queue_p.txt; do sleep 30; done
log "queue start"
for order in a b; do
  for gpu in gpu0 gpu1; do
    score $D/p_stdl2m/$order/$gpu/c16
  done
done
log "queue done"
