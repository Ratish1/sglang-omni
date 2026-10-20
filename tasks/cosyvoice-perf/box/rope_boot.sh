#!/usr/bin/env bash
# #2631 review F1 at serving scale: one boot of a tree with flow_batch_admission_frames 16000,
# whose ladder tops at 15,872 frames, past the 15,000 frame noise buffer the old RoPE table used,
# then a short stream c16 cell when it comes up healthy. Launch each with setsid and record the
# group; stop a boot that failed by its group.
# usage (in the container): bash rope_boot.sh <tree> <card> <cpus> <port> <out dir>
R=/data/p20261009
STAGE4=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4
CELLS="w16:16:64::1" SERVE_ARGS="--vocoder.factory.flow_batch_admission_frames 16000" \
  bash $STAGE4/ab_boot.sh "/workspace/$1" "$5" "$2" "$4" "$3" "$R/inductor_$1"
