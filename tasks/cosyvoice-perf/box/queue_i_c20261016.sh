#!/usr/bin/env bash
# 10-16 second box, after queue_g (the profile): A2 with the scheduler loop on the device's highest priority stream
# (t_a2p, an experiment): a serving smoke test on card 3, then, if it served cleanly, t_a2 against t_a2p, c1, c16,
# c64, both orders, and WER and SIM of order a's c16 audio of both arms on card 2.
# usage (in the container): setsid bash -c 'echo $$ > queue_i.pgid; exec bash queue_i_c20261016.sh'
D=/data/c20261016
IN=$D/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_i.txt; }

log "queue start"
bash $IN/warm_c20261016.sh t_a2p 3 16-31,80-95 28000 $D/smoke_a2p
log "smoke a2p $(grep 'bench rc' $D/smoke_a2p/progress.txt) tracebacks $(grep -c Traceback $D/smoke_a2p/serve.log)"
cd /workspace/sglang-omni
if grep -q "bench rc 0" $D/smoke_a2p/progress.txt && ! grep -q Traceback $D/smoke_a2p/serve.log; then
  PAIR_CELLS=c1,c16,c64 python3 $IN/serve_pair_c20261016.py /workspace/t_a2 /workspace/t_a2p $D/p_a2p > $D/p_a2p.log 2>&1
  log "p_a2p rc $?"
  for dir in p_a2p/a/gpu0 p_a2p/a/gpu1; do
    (CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts --transcribe-only \
      --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en --output-dir $D/$dir/c16 \
      > $D/$dir/c16/score_wer.log 2>&1)
    log "wer $dir rc $?"
    (CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts --similarity-only \
      --model $MODEL --lang en --output-dir $D/$dir/c16 > $D/$dir/c16/score_sim.log 2>&1)
    log "sim $dir rc $?"
  done
else
  log "p_a2p skipped"
fi
log "queue done"
