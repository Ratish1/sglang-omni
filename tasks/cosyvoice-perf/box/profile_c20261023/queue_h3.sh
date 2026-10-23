#!/usr/bin/env bash
# 10-23, cards 0 and 1: L3's own pair while L4's sizing is redesigned (queue_prs.sh stopped after p_h4). t_mh main
# 07367c2ae against t_l3e (L3 with the event fix on it) at c1f, c16, c64, b1f, b16, both orders, every cell warmed at
# its concurrency, with each second's host load; byte identity of the seeded cells; WER and SIM of c16 and c1f.
# Decision rules as in queue_prs.sh.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_h3.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261023/queue_h3.sh'
D=/data/c20261022b
P=$D/prs
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
for name in t_mh t_l3e; do
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-70) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done
log "queue h3 start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
python3 $IN/host_load_c20261022.py $P/p_h3_load.jsonl &
sampler=$!
(cd /workspace/sglang-omni && PAIR_CELLS=c1f,c16,c64,b1f,b16 python3 $IN/serve_pair_c20261021.py /workspace/t_mh \
  /workspace/t_l3e $P/p_h3 > $P/p_h3.log 2>&1)
log "pair p_h3 rc $? token cap hits $(grep -E '\| at the token cap \|' $P/p_h3.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
kill $sampler
python3 $IN/host_load_c20261022.py --summary $P/p_h3_load.jsonl $P/p_h3 c1f,c16,c64,b1f,b16 > $P/p_h3_load.txt 2>&1
python3 $HERE/identity.py $P/p_h3 > $P/identity_p_h3.txt 2>&1
log "identity p_h3 $(grep -E 'c1f|b1f' $P/identity_p_h3.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-300)"
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
for cell in c16 c1f; do
  for order in a b; do
    round=$((round + 1))
    score $P/p_h3/$order/gpu0/$cell 0 $((26000 + 600 * round)) &
    score $P/p_h3/$order/gpu1/$cell 1 $((26300 + 600 * round)) &
    wait
  done
done
log "queue h3 done"
