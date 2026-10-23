#!/usr/bin/env bash
# 10-23, cards 0 and 1: L5 alone at the seeded c1 cells, where the chain against main read b1f +0.066 WER points (5
# draws, se 0.024) and c1f +0.051 (4 draws, se 0.027). Two pairs of t_l4b (L2, L3, L4) against t_l5b (L4 with L5's
# prefill graphs), each order its own seed, every arm scored, then the draws reader and the clips the arms miss in most
# seeds. Started directly on the 10-23 lease (the tokenizer flips are in tokenizer_flips_early.json).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_l5q.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_l5q.sh'
D=/data/c20261022b
Q=$D/quality
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $Q/queue_l5q.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
mkdir -p $Q
log "queue l5q start t_l4b $(git -C /workspace/t_l4b rev-parse --short HEAD) t_l5b $(git -C /workspace/t_l5b rev-parse --short HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
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
pairs=""
seed=20
for name in p_l5q1 p_l5q2; do
  sa=$((seed + 1))
  sb=$((seed + 2))
  seed=$sb
  (cd /workspace/sglang-omni && PAIR_CELLS=c1f_s$sa,b1f_s$sa PAIR_CELLS_B=c1f_s$sb,b1f_s$sb python3 \
    $IN/serve_pair_c20261021.py /workspace/t_l4b /workspace/t_l5b $Q/$name > $Q/$name.log 2>&1)
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
  python3 $HERE/quality_draws.py $D $pairs > $Q/draws_l5q.txt 2>&1
  log "draws rc $? after $name: $(grep -E '^\| (c1f|b1f) \|' $Q/draws_l5q.txt | awk -F'|' '{printf "%s n%s WER%s (se%s) SIM%s (se%s); ", $2, $3, $6, $8, $13, $15}')"
done
python3 $HERE/clip_join.py $D $Q/flips_none.json $pairs > $Q/clip_join_l5q.txt 2>&1
log "clips rc $?"
log "queue l5q done"
