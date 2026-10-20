#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: item 2 (the step rule on item 1, t_sr_9e7a34164). Its serving smoke test on card 3
# gates two pairs, seeded c1, unseeded c1, c16, c64, both orders: upstream main (t_main2) against items 1 and 2, then
# item 1 (t_hg_7b073dd66) against items 1 and 2, which isolates the step rule; then the seeded c1 identity counts and
# WER and SIM of the first pair's c16 audio in both orders, on card 2.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_f.pgid; exec bash queue_f_c20261017.sh'
D=/data/c20261017
IN=$D/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_f.txt; }
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
pair() {
  local first=$1 second=$2 out=$3
  PAIR_CELLS=c1,c1u,c16,c64 python3 $IN/serve_pair_c20261017.py /workspace/$first /workspace/$second $D/$out \
    > $D/$out.log 2>&1
  log "$out rc $?"
  python3 $IN/c1_identity_c20261017.py $D/$out c1 > $D/${out}_identity.txt 2>&1
  log "$out identity $(tr '\n' ' ' < $D/${out}_identity.txt)"
}

log "queue start"
bash $IN/warm_c20261017.sh t_sr_9e7a34164 3 16-31,80-95 22000 $D/smoke_sr
log "smoke sr $(grep 'bench rc' $D/smoke_sr/progress.txt) tracebacks $(grep -c Traceback $D/smoke_sr/serve.log)"
cd /workspace/t_main2
if grep -q "bench rc 0" $D/smoke_sr/progress.txt && ! grep -q Traceback $D/smoke_sr/serve.log; then
  pair t_main2 t_sr_9e7a34164 p_sr
  pair t_hg_7b073dd66 t_sr_9e7a34164 p_hgsr
  for dir in p_sr/a/gpu0 p_sr/a/gpu1 p_sr/b/gpu0 p_sr/b/gpu1; do
    score $D/$dir/c16
  done
else
  log "pairs skipped"
fi
log "queue done"
