#!/usr/bin/env bash
# 10-18, cards 2 and 3: async decode alone on the base it would land on. t_evdl (prof/cosyvoice3-events-deadline
# f7e404c4a: main with fix 1 and the deadline steps v2) against t_ship (prof/cosyvoice3-ship-stack fd3dc7b4b: the same
# with async decode), c1 seeded, c1u, c16, c64, r16, both orders, with its gap breakdown, then WER and SIM of its c16
# audio, both arms in both orders, on card 2. Waits for queue_s.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_t.pgid; exec bash queue_t_c20261018.sh'
D=/data/c20261018
IN=$D/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_t.txt; }
score() {
  local dir=$1
  (cd /workspace/t_main4 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 28100 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $?"
  (cd /workspace/t_main4 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $?"
}

log "queue wait"
until grep -q "queue done" $D/queue_s.txt; do sleep 30; done
log "queue start"
(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261018.py /workspace/t_evdl \
  /workspace/t_ship $D/p_ad > $D/p_ad.log 2>&1)
log "pair p_ad rc $?"
python3 $IN/inter_chunk_mechanism_c20261018.py $D/p_ad f7e404c4a fd3dc7b4b c16 c64 r16 > $D/p_ad_mechanism.txt 2>&1
for order in a b; do
  for gpu in gpu0 gpu1; do
    score $D/p_ad/$order/$gpu/c16
  done
done
log "queue done"
