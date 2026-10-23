#!/usr/bin/env bash
# 10-23, cards 0 and 1, after queue w34: L3 alone at the seeded c1 cells. The chain against main read c1f +0.061 WER
# points (6 draws, se 0.019) while L5 alone read c1f -0.002 (4 draws, se 0.028); every other layer is byte identical at
# c1, so the rest is L3's reference tokens. Two pairs of t_m1b (main, ONNX Runtime's tokenizer) against t_l3n
# (perf/cosyvoice3-speech-tokenizer-cuda-graphs b3a463231, main with L3 only), each order its own seed, every arm
# scored, then the draws reader and the flip split.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_l3q.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_l3q.sh'
D=/data/c20261022b
Q=$D/quality
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
L3N=b3a46323181c987d91d0871ae08c13794ad354a2
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $Q/queue_l3q.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
until grep -q "queue w34 done" $Q/queue_w34.txt 2>/dev/null; do sleep 30; done
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-speech-tokenizer-cuda-graphs
[ -d /workspace/t_l3n ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_l3n $L3N
log "queue l3q start t_m1b $(git -C /workspace/t_m1b rev-parse --short HEAD) $(git -C /workspace/t_m1b rev-parse HEAD:sglang_omni | cut -c1-8) t_l3n $(git -C /workspace/t_l3n rev-parse --short HEAD) $(git -C /workspace/t_l3n rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/t_l3n diff --quiet HEAD -- sglang_omni || echo DIRTY)"
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
slot=3
pairs=""
seed=30
for name in p_l3q1 p_l3q2; do
  sa=$((seed + 1))
  sb=$((seed + 2))
  seed=$sb
  (cd /workspace/sglang-omni && PAIR_CELLS=c1f_s$sa,b1f_s$sa PAIR_CELLS_B=c1f_s$sb,b1f_s$sb python3 \
    $IN/serve_pair_c20261021.py /workspace/t_m1b /workspace/t_l3n $Q/$name > $Q/$name.log 2>&1)
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
  python3 $HERE/quality_draws.py $D $pairs > $Q/draws_l3q.txt 2>&1
  log "draws rc $? after $name: $(grep -E '^\| (c1f|b1f) \|' $Q/draws_l3q.txt | awk -F'|' '{printf "%s n%s WER%s (se%s) SIM%s (se%s); ", $2, $3, $6, $8, $13, $15}')"
done
python3 $HERE/clip_join.py $D $Q/tokenizer_flips_early.json $pairs > $Q/clip_join_l3q.txt 2>&1
log "clips rc $? $(grep -E 'flipped:|rest:' $Q/clip_join_l3q.txt | tr -s ' ' | tr '\n' ' ')"
log "queue l3q done"
