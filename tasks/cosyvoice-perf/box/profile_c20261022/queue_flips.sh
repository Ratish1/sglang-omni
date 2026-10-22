#!/usr/bin/env bash
# 10-23, card 0, after queue route: per reference speech tokens of ONNX Runtime (main), L3's served graphs and the
# eager fp32 model, for joining with the quality draws' per clip WER and SIM (the chain's misreads recur on the same
# clips across seeds).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_flips.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_flips.sh'
D=/data/c20261022b
Q=$D/quality
HERE=/workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022
log() { echo "$* $(date +%T)" >> $Q/queue_flips.txt; }
until grep -q "queue route done" $Q/queue_route.txt 2>/dev/null; do sleep 30; done
log "queue flips start"
(cd /workspace/t_l3n && CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/workspace/t_l3n:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS \
  taskset -c 0-15,64-79 python3 -u $HERE/tokenizer_flips.py $Q/tokenizer_flips.json > $Q/tokenizer_flips.log 2>&1)
log "flips rc $? $(tail -3 $Q/tokenizer_flips.log | tr '\n' ' ')"
log "queue flips done"
