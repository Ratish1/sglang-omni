#!/usr/bin/env bash
# 10-17 lease, card 2: WER and SIM of `p_pr2`'s c16 audio, both arms in both orders (upstream main against the stack
# 720636a8b, the same behaviour as the PR head 0167dde34), for the PR body. Waits for queue_k; no server.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_l.pgid; exec bash queue_l_c20261017.sh'
D=/data/c20261017
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_l.txt; }
score() {
  local dir=$1
  (cd /workspace/t_main2 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $?"
  (cd /workspace/t_main2 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $?"
}

log "queue wait"
until grep -q "queue done" $D/queue_k.txt; do sleep 30; done
log "queue start"
for dir in p_pr2/a/gpu0 p_pr2/a/gpu1 p_pr2/b/gpu0 p_pr2/b/gpu1; do
  score $D/$dir/c16
done
log "queue done"
