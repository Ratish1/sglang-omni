#!/usr/bin/env bash
# 10-22, cards 0 and 1, after queue c64q: WER for every c64 cell of p_q1 to p_q3 the rate check could not score (its
# ASR ports met the Qwen3-TTS session's scorer on the shared host network, 26000 to 29000), on ports from 34000, 600
# apart, then the c64 quality summary over p_f1 and p_q1 to p_q3.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_rescore.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_rescore.sh'
D=/data/c20261022b
F=$D/final
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $F/queue_rescore.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
until grep -q "queue c64q done" $F/queue_c64q.txt 2>/dev/null; do sleep 30; done
log "queue rescore start"
round=0
for name in p_q1 p_q2 p_q3; do
  for order in a b; do
    round=$((round + 1))
    for card in 0 1; do
      dir=$F/$name/$order/gpu$card/c64
      if [ ! -s $dir/wer_results.json ]; then
        (cd /workspace/t_m1b && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
          --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $((34000 + 600 * round + 300 * card)) \
          --lang en --output-dir $dir > $dir/score_wer.log 2>&1) &
      fi
    done
    wait
    log "rescored $name $order $(for card in 0 1; do grep -o 'WER mean.*' $F/$name/$order/gpu$card/c64/score_wer.log | head -1; done | tr '\n' ' ')"
  done
done
python3 $IN/quality_summary_c20261022.py $F p_f1:c64 p_q1:c64 p_q2:c64 p_q3:c64 > $F/quality_c64q.txt 2>&1
log "summary rc $?"
log "queue rescore done"
