#!/usr/bin/env bash
# 10-21, card 2. WER and SIM of pair p_l1s's c16 audio (t_m21 main against the L1 PR branch on SGLang's uploads),
# scored from main's tree as every pair's c16 is.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/score_l1s.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/score_l1s_c20261021.sh'
D=/data/c20261020
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
for dir in a/gpu0 a/gpu1 b/gpu0 b/gpu1; do
  out=$D/p_l1s/$dir/c16
  (cd /workspace/t_m21 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $out > $out/score_wer.log 2>&1)
  echo "wer $dir rc $? $(grep 'WER mean' $out/score_wer.log | tr -s ' ') $(date +%T)" >> $D/score_l1s.txt
  (cd /workspace/t_m21 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $out > $out/score_sim.log 2>&1)
  echo "sim $dir rc $? $(grep 'speaker similarity' $out/score_sim.log | cut -c1-40) $(date +%T)" >> $D/score_l1s.txt
done
echo "score done $(date +%T)" >> $D/score_l1s.txt
