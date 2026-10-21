#!/usr/bin/env bash
# 10-22, the 0.5.22 container, cards 0 and 1: queue r's phase b and queue s reordered so each PR's runs come in the
# order the PRs open (L2, L3, L4), on the trees and warms phase b already made. Once the CPU unit run in
# /data/c20261022r/ci has ended: p_r5 t_l1 against t_l2 at c64 with every card's framebuffer sampled each second
# and each order's bench start kept (L2's memory); p_r3 t_l2 against t_l3 (c1f seeded, c16) with c16 scored; p_r4
# t_l3 against t_l4 (c16, c64); every gap pooled.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022r/queue_t.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_t_c20261022.sh'
D=/data/c20261022r
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_r.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
pair() {
  local first=$1 second=$2 name=$3 cells=$4
  (cd /workspace/sglang-omni && PAIR_CELLS=$cells python3 $IN/serve_pair_c20261021.py /workspace/$first \
    /workspace/$second $D/$name > $D/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
}
pooled() {
  local name=$1 cell=$2 first=$3 second=$4
  python3 $IN/pooled_inter_chunk_c20261021.py $D/$name $cell $first $second > $D/pooled_${name}_$cell.txt 2>&1
  log "pooled $name $cell $(grep '^pooled' $D/pooled_${name}_$cell.txt | tr '\n' ' ')"
}
score() {
  local dir=$1 card=$2 port=$3
  (cd /workspace/t_m && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --host 127.0.0.1 --port $port \
    --lang en --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_m && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}

log "queue t start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
while pgrep -f "pytest tests/ -q" > /dev/null; do sleep 30; done
log "cpu unit run ended"
nvidia-smi dmon -i 0,1 -s m -d 1 -o T > $D/p_r5_dmon.log 2>&1 &
dmon=$!
for order in a b; do
  (until [ -e $D/p_r5/$order/bench_gpu0_c64.log ]; do sleep 1; done; echo "$order $(date +%T)" >> $D/p_r5_bench_start.txt) &
done
pair t_l1 t_l2 p_r5 c64
kill $dmon
log "p_r5 bench starts $(tr '\n' ' ' < $D/p_r5_bench_start.txt)"
pair t_l2 t_l3 p_r3 c1f,c16
pooled p_r3 c16 t_l2 t_l3
score $D/p_r3/a/gpu0/c16 0 26000 &
score $D/p_r3/a/gpu1/c16 1 27000 &
wait
score $D/p_r3/b/gpu0/c16 0 28000 &
score $D/p_r3/b/gpu1/c16 1 29000 &
wait
pair t_l3 t_l4 p_r4 c16,c64
pooled p_r4 c16 t_l3 t_l4
pooled p_r4 c64 t_l3 t_l4
log "queue t done"
