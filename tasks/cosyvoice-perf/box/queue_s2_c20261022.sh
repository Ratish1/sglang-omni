#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1: the SeedTTS quality of queue m's pairs, which ran --generate-only. WER
# (Qwen3-ASR) and SIM of every full corpus cell from its saved audio: p_m1 (t_l3 against t_l4) and p_m2 (t_m against
# t_l4) c1f, c16, c64; p_m3 (t_m against t_l2) c16, c64, b16; both orders, both cards; then per clip SIM of each
# pair's arms on the seeded c1f. Scored from t_m's tree, as every pair's audio is; one scorer per card at a time.
# Then the pair audit (pair_audit_c20261022.py) over the run home with the other tenants' windows of 10-22.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_s2.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_s2_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_s2.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
score() {
  local dir=$1 card=$2 port=$3
  (cd /workspace/t_m && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $port --lang en --output-dir $dir \
    > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_m && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
log "queue s2 start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD) scorer t_m $(git -C /workspace/t_m rev-parse --short HEAD)"
python3 $IN/pair_audit_c20261022.py $D /workspace/sglang-omni 07:33:00-08:43:00 09:22:00-11:33:19 11:33:20-11:51:26 \
  > $D/pair_audit.txt 2>&1
log "pair audit rc $?"
for spec in p_m1:c1f,c16,c64 p_m2:c1f,c16,c64 p_m3:c16,c64,b16; do
  name=${spec%%:*}
  for cell in $(echo ${spec#*:} | tr , ' '); do
    for order in a b; do
      score $D/$name/$order/gpu0/$cell 0 51000 &
      score $D/$name/$order/gpu1/$cell 1 53000 &
      wait
    done
  done
done
for name in p_m1 p_m2; do
  python3 $IN/paired_sim_c20261021.py $D/$name/a/gpu0/c1f $D/$name/a/gpu1/c1f > $D/paired_sim_${name}_a.txt 2>&1
  python3 $IN/paired_sim_c20261021.py $D/$name/b/gpu1/c1f $D/$name/b/gpu0/c1f > $D/paired_sim_${name}_b.txt 2>&1
  log "paired sim $name a $(tr '\n' ' ' < $D/paired_sim_${name}_a.txt | cut -c1-300)"
  log "paired sim $name b $(tr '\n' ' ' < $D/paired_sim_${name}_b.txt | cut -c1-300)"
done
log "queue s2 done"
