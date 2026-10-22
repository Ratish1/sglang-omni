#!/usr/bin/env bash
# 10-22, cards 0 and 1, after queue quality: the vocoder route effect (buffered minus streaming audio of the same
# tokens) of L3 alone and of the chain at L5, beside main's and the chain's from the quality pairs. Scores every arm
# of the identity pairs p_id1 (t_l3n, t_l4b) and p_id2 (t_l5b, t_chain) not yet scored, then the draws reader over
# them alone.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_route.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_route.sh'
D=/data/c20261022b
Q=$D/quality
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $Q/queue_route.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
until grep -q "queue quality done" $Q/queue.txt 2>/dev/null; do sleep 30; done
log "queue route start"
score() {
  local dir=$1 card=$2 port=$3
  if [ ! -s $dir/wer_results.json ]; then
    (cd /workspace/t_m1b && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
      --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $port --lang en --output-dir $dir \
      > $dir/score_wer.log 2>&1)
    log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  fi
  if [ ! -s $dir/similarity_results.json ]; then
    (cd /workspace/t_m1b && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
      --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
    log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
  fi
}
slot=0
for name in p_id1 p_id2; do
  for order in a b; do
    for cell in $(cd $Q/$name/$order/gpu0 && ls -d */ | tr -d /); do
      slot=$(((slot + 1) % 13))
      score $Q/$name/$order/gpu0/$cell 0 $((34000 + 600 * slot)) &
      score $Q/$name/$order/gpu1/$cell 1 $((34300 + 600 * slot)) &
      wait
    done
  done
done
python3 $HERE/quality_draws.py $D quality/p_id1 quality/p_id2 > $Q/draws_identity.txt 2>&1
log "draws rc $? $(grep -A 20 'route effect' $Q/draws_identity.txt | grep -E '^\| quality' | tr -s ' ' | tr '\n' ' ')"
log "queue route done"
