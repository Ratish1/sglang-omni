#!/usr/bin/env bash
# 10-22, cards 0 and 1, after queue final: the c64 SIM rate. queue final read the chain's c64 SIM below main's in
# both orders (-0.15, -0.21), and every c64 pair of the day with L4 against without it read -0.11, -0.09, -0.23,
# +0.16, -0.15, -0.21. Three more pairs of t_m1b against t_chain at c64, both orders, every arm scored (WER and SIM),
# then the quality summary over all four pairs: eight readings per arm.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_c64q.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_c64q.sh'
D=/data/c20261022b
F=$D/final
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $F/queue_c64q.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
until grep -q "queue final done" $F/queue.txt; do sleep 30; done
log "queue c64q start t_m1b $(git -C /workspace/t_m1b rev-parse --short HEAD) t_chain $(git -C /workspace/t_chain rev-parse --short HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
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
round=0
for name in p_q1 p_q2 p_q3; do
  (cd /workspace/sglang-omni && PAIR_CELLS=c64 python3 $IN/serve_pair_c20261021.py /workspace/t_m1b /workspace/t_chain \
    $F/$name > $F/$name.log 2>&1)
  log "pair $name rc $? $(grep -E '\| (a|b) \| c64 \| req/s' $F/$name.log | tr -s ' ' | tr '\n' ' ')"
  for order in a b; do
    round=$((round + 1))
    score $F/$name/$order/gpu0/c64 0 $((26000 + 600 * round)) &
    score $F/$name/$order/gpu1/c64 1 $((26300 + 600 * round)) &
    wait
  done
done
python3 $HERE/quality_summary_c20261022.py $F p_f1:c64 p_q1:c64 p_q2:c64 p_q3:c64 > $F/quality_c64q.txt 2>&1
log "summary rc $?"
log "queue c64q done"
