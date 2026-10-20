#!/usr/bin/env bash
# 10-20, cards 2 and 3. L1 (perf/cosyvoice3-stack-l1 8202dcb25, tree t_l1) before its PR: c1 over every reference
# seeded (pair p_c1f, t_m21 main bcd0d8c68 against t_l1; upstream main has the same sglang_omni tree), where p_c20's
# c1 tails are one request of 64; then WER and SIM of p_c20's c16 audio (main b98a28936 against the candidate).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_q.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_q_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_q.txt; }
score() {
  local dir=$1
  (cd /workspace/t_m21 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_m21 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}

log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
for t in t_m21 t_l1; do
  log "tree $t $(git -C /workspace/$t log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$t rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$t diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

(cd /workspace/sglang-omni && PAIR_CELLS=c1f python3 $IN/serve_pair_c20261020.py /workspace/t_m21 /workspace/t_l1 \
  $D/p_c1f > $D/p_c1f.log 2>&1)
log "pair p_c1f rc $? token cap hits $(grep -E '\| at the token cap \|' $D/p_c1f.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
for dir in a/gpu0 a/gpu1 b/gpu0 b/gpu1; do
  score $D/p_c20/$dir/c16
done
log "queue done"
