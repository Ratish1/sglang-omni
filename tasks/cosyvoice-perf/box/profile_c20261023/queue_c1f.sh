#!/usr/bin/env bash
# 10-23, cards 0 and 1, after queue_l4.sh: WER and SIM of p_h4's and p_h6's seeded c1f, so L4's c16 WER (+0.21 ;
# +0.04 points, unseeded) is read where both arms sample alike; p_h6's arms both carry L4, so its differing clips are
# the processes' own ONNX Runtime variation; then quality_draws.py over the scored pairs.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_c1f.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261023/queue_c1f.sh'
D=/data/c20261022b
P=$D/prs
HERE=/workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
until grep -q 'queue l4 done' $P/queue.txt; do sleep 30; done
log "queue c1f start"
score() {
  local dir=$1 card=$2 port=$3
  (cd /workspace/t_mh && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $port --lang en --output-dir $dir \
    > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_mh && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
round=0
for name in p_h4 p_h6; do
  for order in a b; do
    round=$((round + 1))
    score $P/$name/$order/gpu0/c1f 0 $((28000 + 600 * round)) &
    score $P/$name/$order/gpu1/c1f 1 $((28300 + 600 * round)) &
    wait
  done
done
python3 $HERE/quality_draws.py $P p_h3 p_h4 p_h6 p_h5 > $P/quality_draws.txt 2>&1
log "quality draws rc $?"
log "queue c1f done"
