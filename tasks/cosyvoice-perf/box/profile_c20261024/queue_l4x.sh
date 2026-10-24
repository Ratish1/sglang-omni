#!/usr/bin/env bash
# 10-24, cards 0 and 1, after queue_l4w.sh: the rest of #3129's own pair on its head, p_l4d (c1f, b1f, b16, both
# orders, unprofiled, with each second's host load), byte identity of its seeded cells, then WER and SIM of p_l4c's
# c16 and p_l4d's c1f, so the PR body's tables all come from its head against main cd191df23.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261024/l4p/queue_x.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261024/queue_l4x.sh'
D=/data/c20261024
P=$D/l4p
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
until grep -q 'queue l4w done' $P/queue.txt; do sleep 30; done
python3 $IN/host_load_c20261022.py $P/p_l4d_load.jsonl &
sampler=$!
(cd /workspace/sglang-omni && PAIR_CELLS=c1f,b1f,b16 python3 $IN/serve_pair_c20261021.py /workspace/t_mn \
  /workspace/t_l4p $P/p_l4d > $P/p_l4d.log 2>&1)
log "pair p_l4d rc $? token cap hits $(grep -E '\| at the token cap \|' $P/p_l4d.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
kill $sampler
python3 $IN/host_load_c20261022.py --summary $P/p_l4d_load.jsonl $P/p_l4d c1f,b1f,b16 > $P/p_l4d_load.txt 2>&1
python3 $IN/profile_c20261022/identity.py $P/p_l4d > $P/identity_p_l4d.txt 2>&1
log "identity p_l4d $(grep -E 'c1f|b1f' $P/identity_p_l4d.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-300)"
score() {
  local dir=$1 card=$2 port=$3
  (cd /workspace/t_mn && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $port --lang en --output-dir $dir \
    > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_mn && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
round=0
for target in p_l4c:c16 p_l4d:c1f; do
  name=${target%%:*}
  cell=${target##*:}
  for order in a b; do
    round=$((round + 1))
    score $P/$name/$order/gpu0/$cell 0 $((31000 + 600 * round)) &
    score $P/$name/$order/gpu1/$cell 1 $((31300 + 600 * round)) &
    wait
  done
done
log "queue l4x done"
