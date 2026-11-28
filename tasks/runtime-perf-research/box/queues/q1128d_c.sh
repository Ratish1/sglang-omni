#!/usr/bin/env bash
# 11-28 second lease, node-0001, after q1128d_b: the batch-invariant thinker (diagnostic only, never serving or CI) on
# the cells check 1 left open, FIX 91c2e2010 against MAIN 0f70d9c7c, same-time pairs on the two cards.
#   1. Radix cache off, CI one card, text c16 x2 and the talker test x2. One text with the radix-on runs dt* means a
#      cache hit equals a full prefill bit for bit, so prefix reuse is exact for multimodal prompts.
#   2. Video-AMME c16 x3 (the CI gate's dataset and concurrency), CI one card, 2 pairs swapped. The encoders run in
#      their own processes without the flag, and on the PR the audio encoder batches several videos' tracks: one text
#      on both arms clears the encoders' batched paths.
#   3. Fused text pipeline, text c16 x2 then c1 x1, one pair. One text on both arms means the fused c16 movement is
#      kernel order, as on the CI layout.
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main F=/workspace/fix
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
DETERMINISTIC="--thinker.engine.enable_deterministic_inference true --thinker.engine.attention_backend fa3"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((20000 + 4000 * $1 + $2)); }
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
until grep -q "q1128d_b done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128d_c start"

uncached() {
  local name=$1 tree=$2 card=$3 offset=$4
  local port=$(port_of $card $offset)
  note $name $tree "CI one card, thinker deterministic, radix cache off, text c16 x2, talker x2, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card $card --port $port \
    --out $O/$name --serve-extra "$DETERMINISTIC --thinker.engine.disable_radix_cache true" --bench videomme_text \
    --loads 16 --passes clean --bench-repeats 2 \
    --after "python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 2" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
both "uncached du1_m $M 0 2700" "uncached du1_f $F 1 2700"
python3 $K/response_identity.py $O radix_on="dt*_?/c16_clean/bench/rep*/videomme_results.json" \
  radix_off="du1_?/c16_clean/bench/rep*/videomme_results.json" > $O/det_radix_text.txt 2>&1
python3 $K/response_identity.py $O radix_on="dt*_?/talker/pass*/*.json" radix_off="du1_?/talker/pass*/*.json" \
  > $O/det_radix_talker.txt 2>&1
log "deterministic radix read: $(grep ' with ' $O/det_radix_text.txt) / $(grep ' with ' $O/det_radix_talker.txt)"

amme() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "CI one card, thinker deterministic, Video-AMME c16 x3, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card $card \
    --port $(port_of $card $offset) --out $O/$name --serve-extra "$DETERMINISTIC" --bench videoamme --loads 16 \
    --passes clean --bench-repeats 3 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
both "amme da1_m $M 0 2900" "amme da1_f $F 1 2900"
both "amme da2_f $F 0 3100" "amme da2_m $M 1 3100"
python3 $K/response_identity.py $O main="da*_m/c16_clean/bench/rep*/videoamme_results.json" \
  fix="da*_f/c16_clean/bench/rep*/videoamme_results.json" > $O/det_amme.txt 2>&1
python3 $K/videomme_answer_diff.py $O main=da1_m/c16_clean,da2_m/c16_clean fix=da1_f/c16_clean,da2_f/c16_clean \
  >> $O/det_amme.txt 2>&1
log "deterministic amme read: $(grep ' with ' $O/det_amme.txt)"

fused() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "fused text pipeline, thinker deterministic, text c16 x2 and c1, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_text_fused --card $card \
    --port $(port_of $card $offset) --out $O/$name --serve-extra "$CONTEXT_32K $DETERMINISTIC" --bench videomme_text \
    --loads 16,1 --passes clean --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
both "fused df1_m $M 0 3300" "fused df1_f $F 1 3300"
python3 $K/response_identity.py $O main="df1_m/c*_clean/bench/rep*/videomme_results.json" \
  fix="df1_f/c*_clean/bench/rep*/videomme_results.json" > $O/det_fused.txt 2>&1
log "deterministic fused read: $(grep ' with ' $O/det_fused.txt)"
log "q1128d_c done"
