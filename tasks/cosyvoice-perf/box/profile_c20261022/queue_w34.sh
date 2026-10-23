#!/usr/bin/env bash
# 10-23, cards 0 and 1, after queue probe: the last two quality pairs of the chain against main that queue_quality.sh
# did not reach on 10-22, p_w3 (seeds 15, 16) and p_w4 (17, 18), each order c1f and b1f under its seed then c16 and
# b16, every arm scored; then the draws reader and the flip split over every scored pair (the 10-22 results, back in
# this home from the Mac). p_w2 order b c1f_s14 stays unscored: its audio went with the 10-22 lease.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_w34.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_w34.sh'
D=/data/c20261022b
Q=$D/quality
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $Q/queue_w34.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
until grep -q "queue probe done" $Q/queue_probe.txt 2>/dev/null; do sleep 30; done
log "queue w34 start t_m1b $(git -C /workspace/t_m1b rev-parse --short HEAD) t_chain $(git -C /workspace/t_chain rev-parse --short HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
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
slot=6
pairs="final/p_f1 final/p_q1 final/p_q2 final/p_q3 quality/p_w1 quality/p_w2"
seed=14
for name in p_w3 p_w4; do
  sa=$((seed + 1))
  sb=$((seed + 2))
  seed=$sb
  (cd /workspace/sglang-omni && PAIR_CELLS=c1f_s$sa,b1f_s$sa,c16,b16 PAIR_CELLS_B=c1f_s$sb,b1f_s$sb,c16,b16 python3 \
    $IN/serve_pair_c20261021.py /workspace/t_m1b /workspace/t_chain $Q/$name > $Q/$name.log 2>&1)
  log "pair $name rc $? $(grep -E '\| (a|b) \| [a-z0-9_]+ \| (failed|at the token cap) \|' $Q/$name.log | awk -F'|' '{printf "%s %s %s:%s/%s ", $2, $3, $4, $5, $6}')"
  for order in a b; do
    for cell in $(cd $Q/$name/$order/gpu0 && ls -d */ | tr -d /); do
      slot=$(((slot + 1) % 13))
      score $Q/$name/$order/gpu0/$cell 0 $((34000 + 600 * slot)) &
      score $Q/$name/$order/gpu1/$cell 1 $((34300 + 600 * slot)) &
      wait
    done
  done
  pairs="$pairs quality/$name"
  python3 $HERE/quality_draws.py $D $pairs > $Q/draws_w34.txt 2>&1
  log "draws rc $? after $name: $(grep -E '^\| (c1f|b1f|c16|b16|c64) \|' $Q/draws_w34.txt | awk -F'|' '{printf "%s n%s WER%s (se%s) SIM%s (se%s); ", $2, $3, $6, $8, $13, $15}')"
done
python3 $HERE/clip_join.py $D $Q/tokenizer_flips_early.json $pairs > $Q/clip_join_w34.txt 2>&1
log "clips rc $?"
log "queue w34 done"
