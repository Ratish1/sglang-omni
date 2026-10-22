#!/usr/bin/env bash
# 10-22, cards 0 and 1, after queue rescore: the final test's quality rows that read worse in both orders (c1f WER and
# SIM, b1f WER, b16 SIM), powered. First two identity pairs, seeded b1f and c1f under two seeds, each scored only if
# not byte identical: t_l3n (L3 alone, b3a463231) against t_l4b (L2, L3, L4, 4f3ba8bd9), whether L2 and L4 leave the
# audio of both vocoder routes byte identical (the chain's buffered audio read 0.31 SIM nearer its streaming audio of
# the same tokens than main's in the final test); t_l5b (the chain at L5, 4eb6e93e7) against t_chain, the same for L6,
# the shared prefix graph inputs and the tokenizer's event. Then four pairs of t_m1b against t_chain, each order its own seed for c1f and b1f, and
# c16 and b16, every arm scored: nine seeded draws of c1f and b1f and ten of c16 and b16 with the final test's. The
# draws reader runs after every pair.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_quality.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_quality.sh'
D=/data/c20261022b
Q=$D/quality
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
mkdir -p $Q
log() { echo "$* $(date +%T)" >> $Q/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
tree() {
  local name=$1 sha=$2
  if [ -d /workspace/$name ]; then
    git -C /workspace/$name checkout -q --detach $sha
  else
    git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
  fi
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-60) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD -- sglang_omni || echo DIRTY)"
}
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-items-chain perf/cosyvoice3-speech-tokenizer-cuda-graphs
tree t_l3n b3a463231
tree t_l4b 4f3ba8bd9
tree t_l5b 4eb6e93e7
log "queue quality start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD) t_m1b $(git -C /workspace/t_m1b rev-parse --short HEAD) t_chain $(git -C /workspace/t_chain rev-parse --short HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
pair() {
  local first=$1 second=$2 name=$3 cells=$4 cells_b=$5
  (cd /workspace/sglang-omni && PAIR_CELLS=$cells PAIR_CELLS_B=$cells_b python3 $IN/serve_pair_c20261021.py \
    /workspace/$first /workspace/$second $Q/$name > $Q/$name.log 2>&1)
  log "pair $name rc $? $(grep -E '\| (a|b) \| [a-z0-9_]+ \| (failed|at the token cap) \|' $Q/$name.log | awk -F'|' '{printf "%s %s %s:%s/%s ", $2, $3, $4, $5, $6}')"
}
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
score_pair() {
  local name=$1
  for order in a b; do
    for cell in $(cd $Q/$name/$order/gpu0 && ls -d */ | tr -d /); do
      slot=$(((slot + 1) % 13))
      score $Q/$name/$order/gpu0/$cell 0 $((34000 + 600 * slot)) &
      score $Q/$name/$order/gpu1/$cell 1 $((34300 + 600 * slot)) &
      wait
    done
  done
}
identity_pair() {
  local first=$1 second=$2 name=$3
  pair $first $second $name b1f,c1f b1f_s2,c1f_s2
  python3 $HERE/identity.py $Q/$name > $Q/identity_$name.txt 2>&1
  log "identity $(tr '\n' ';' < $Q/identity_$name.txt)"
  if awk '{ if ($5 + 0 != $9 + 0 || $9 + 0 != $12 + 0) bad = 1 } END { exit !bad }' $Q/identity_$name.txt; then
    log "$name not byte identical, scoring it"
    score_pair $name
  fi
}
identity_pair t_l3n t_l4b p_id1
identity_pair t_l5b t_chain p_id2
pairs="final/p_f1 final/p_q1 final/p_q2 final/p_q3"
seed=10
for name in p_w1 p_w2 p_w3 p_w4; do
  sa=$((seed + 1))
  sb=$((seed + 2))
  seed=$sb
  pair t_m1b t_chain $name c1f_s$sa,b1f_s$sa,c16,b16 c1f_s$sb,b1f_s$sb,c16,b16
  score_pair $name
  pairs="$pairs quality/$name"
  python3 $HERE/quality_draws.py $D $pairs > $Q/draws.txt 2>&1
  log "draws rc $? after $name: $(grep -E '^\| (c1f|b1f|c16|b16|c64) \|' $Q/draws.txt | awk -F'|' '{printf "%s n%s WER%s (se%s) SIM%s (se%s); ", $2, $3, $6, $8, $13, $15}')"
done
log "queue quality done"
