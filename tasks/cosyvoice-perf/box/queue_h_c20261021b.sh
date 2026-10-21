#!/usr/bin/env bash
# 10-21, the 0.5.22 container, cards 0 and 1: #3029's review fixes. Phase base: warms of t_main (upstream main
# 057d97629) and t_h0 (#3029 before the fixes, fd2139ac6); the chunk probe on t_h0 at 1 and 16 rows with the spectrum
# aligned to 8 channels (24, the head) and to 4 (20), the alignment bench; t_h0's seeded decode outputs saved as the
# bitwise reference for the fixed head; the probe on t_main (the channel-first decode on cuDNN 9.24).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261021b/queue_h.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_h_c20261021b.sh base'
D=/data/c20261021b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
CKPT=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512'))")
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_h.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
probe() {
  local tree=$1 card=$2 name=$3
  shift 3
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=/workspace/$tree:$COSYVOICE taskset -c $(cpus $card) \
    python3 $IN/probe_hift_chunk_c20261021.py --checkpoint $CKPT "$@" > $D/probe_$name.log 2>&1)
  log "probe $name rc $? $(grep -h gpu_ms_per_chunk $D/probe_$name.log)"
}
tree_log() { log "tree $1 $(git -C /workspace/$1 log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$1 rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$1 diff --quiet HEAD -- sglang_omni || echo DIRTY)"; }

if [ "$1" = base ]; then
  log "queue base start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
  tree_log t_main
  tree_log t_h0
  bash $IN/warm_c20261021.sh t_main 0 $(cpus 0) 50000 $D/warm_t_main &
  bash $IN/warm_c20261021.sh t_h0 1 $(cpus 1) 50400 $D/warm_t_h0 &
  wait
  for tree in t_main t_h0; do
    log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
  done
  probe t_h0 0 h0_r1 --rows 1 --save $D/decode_t_h0.pt &
  probe t_h0 1 h0_r1_a4 --rows 1 --channel-alignment 4 &
  wait
  probe t_h0 0 h0_r16 --rows 16 &
  probe t_h0 1 h0_r16_a4 --rows 16 --channel-alignment 4 &
  wait
  probe t_main 0 main_r1 --rows 1 &
  probe t_main 1 main_r16 --rows 16 &
  wait
  log "queue base done"
else
  log "unknown phase $1"
fi
