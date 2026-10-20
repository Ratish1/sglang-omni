#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: item 1 (HiFT's decode from CUDA graphs, t_hg_7b073dd66) against upstream main
# 6bdd47967 (t_main2, which holds #2897). Main's tree warms its Inductor cache on card 3 (the smoke test of the HiFT
# graphs tree ran on card 2 already), then the pair: seeded c1 (identity), unseeded c1, c16, c64, both orders; then
# the seeded c1 identity counts and WER and SIM of both arms' c16 audio in both orders, on card 2.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_e.pgid; exec bash queue_e_c20261017.sh'
D=/data/c20261017
IN=$D/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_e.txt; }
score() {
  local dir=$1
  (cd /workspace/t_main2 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $?"
  (cd /workspace/t_main2 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $?"
}

log "queue start"
bash $IN/warm_c20261017.sh t_main2 3 16-31,80-95 22000 $D/warm_main2
log "warm main2 $(grep 'bench rc' $D/warm_main2/progress.txt) tracebacks $(grep -c Traceback $D/warm_main2/serve.log)"
cd /workspace/t_main2
PAIR_CELLS=c1,c1u,c16,c64 python3 $IN/serve_pair_c20261017.py /workspace/t_main2 /workspace/t_hg_7b073dd66 $D/p_hg \
  > $D/p_hg.log 2>&1
log "p_hg rc $?"
python3 $IN/c1_identity_c20261017.py $D/p_hg c1 > $D/p_hg_identity.txt 2>&1
log "identity $(tr '\n' ' ' < $D/p_hg_identity.txt)"
for dir in p_hg/a/gpu0 p_hg/a/gpu1 p_hg/b/gpu0 p_hg/b/gpu1; do
  score $D/$dir/c16
done
log "queue done"
