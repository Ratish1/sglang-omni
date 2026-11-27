#!/usr/bin/env bash
# 11-27, node-0000, after q1127_c: the causal control that proved 005-3's owner on 11-19 (n5), run on the CI gates of
# both arms. The thinker runs eager (no decode or prefill graphs), without mixed chunks and without the radix cache, and
# thinker_numerics_probe pins FlashAttention's split-KV count at 1 with SGLang's precomputed scheduler metadata dropped
# (OMNI_FA_SPLITS=1; the probe matches no request, OMNI_NUMERICS_TARGET_LENS=1). With the radix cache off every pass
# prefills again, so each pass is its own draw. If every question gives one letter on every pass of both arms, the
# answers #3452 moves are owned by the batch-variant attention path (present on main), not by preprocessing.
# CI speech pipeline, text Video-MME c16 x3 and the talker test x3 a boot, M F F M, FIX 72be1f80e against MAIN 0f70d9c7c.
set -u
O=/data/v1127
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main FIX=/workspace/fix
CPUS=56-111,168-223
CONTROL="--thinker.engine.disable_radix_cache true --thinker.engine.enable_mixed_chunk false \
--thinker.engine.disable_cuda_graph true --thinker.engine.cuda_graph_backend_prefill disabled"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1127_c done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1127_d start"
control() {
  local name=$1 tree=$2 port=$3
  note $name $tree "CI speech pipeline, split control, text Video-MME c16 x3, talker test x3"
  mkdir -p $O/$name
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card 0 --port $port --out $O/$name \
    --serve-extra "$CONTROL" \
    --serve-env "PYTHONPATH=$K/thinker_numerics_probe:$tree OMNI_NUMERICS_PROBE=$O/$name/num OMNI_NUMERICS_TARGET_LENS=1 OMNI_FA_SPLITS=1" \
    --bench videomme_text --loads 16 --passes clean --bench-repeats 3 \
    --after "python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 3" > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(cat $O/$name/num.*.splits 2>/dev/null | sort | uniq -c | tr -s ' ' | head -2 | tr '\n' ';')"
}
control cn1_m $MAIN 22000
control cn2_f $FIX 22200
control cn3_f $FIX 22400
control cn4_m $MAIN 22600
python3 $K/videomme_answer_diff.py $O main=cn1_m/c16_clean,cn4_m/c16_clean fix=cn2_f/c16_clean,cn3_f/c16_clean \
  > $O/gate_text_control.txt 2>&1
python3 $K/videomme_answer_diff.py $O main=cn1_m/talker,cn4_m/talker fix=cn2_f/talker,cn3_f/talker \
  > $O/gate_talker_control.txt 2>&1
log "control read: text $(grep -E '^answers' $O/gate_text_control.txt), talker $(grep -E '^answers' $O/gate_talker_control.txt)"
log "q1127_d done"
