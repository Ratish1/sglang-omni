#!/usr/bin/env bash
# 10-23, cards 0 and 1: the cost of the fp32 tokenizer graphs. t_chain (fdf92c2e5, TF32 graphs) against t_chain2
# (03f504f0a, the same with the graphs captured in the process's math mode) at c1f seed 1234, c16 and c64, both
# orders. Only c1f is scored: both arms repeat their seeded audio exactly, so its clips differ only where the
# reference's codes do, and identity.py names them.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_tokpair.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_tokpair.sh'
D=/data/c20261022b
Q=$D/quality
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $Q/queue_tokpair.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
log "queue tokpair start t_chain $(git -C /workspace/t_chain rev-parse --short HEAD) t_chain2 $(git -C /workspace/t_chain2 rev-parse --short HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
(cd /workspace/sglang-omni && PAIR_CELLS=c1f,c16,c64 python3 $IN/serve_pair_c20261021.py /workspace/t_chain \
  /workspace/t_chain2 $Q/p_tok > $Q/p_tok.log 2>&1)
log "pair p_tok rc $? $(grep -E '\| (a|b) \| [a-z0-9]+ \| (req/s|first audio mean|inter chunk p99|failed) \|' $Q/p_tok.log | awk -F'|' '{printf "%s %s %s:%s/%s%s; ", $2, $3, $4, $5, $6, $7}')"
score() {
  local dir=$1 card=$2 port=$3
  (cd /workspace/t_m1b && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $port --lang en --output-dir $dir \
    > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_m1b && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
score $Q/p_tok/a/gpu0/c1f 0 36400 &
score $Q/p_tok/a/gpu1/c1f 1 36700 &
wait
python3 $HERE/identity.py $Q/p_tok > $Q/identity_p_tok.txt 2>&1
log "identity $(grep c1f $Q/identity_p_tok.txt | tr '\n' ';')"
log "queue tokpair done"
