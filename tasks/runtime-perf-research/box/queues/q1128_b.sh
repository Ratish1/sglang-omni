#!/usr/bin/env bash
# 11-28, node-0001, after q1128_a: which part of the thinker's attention path moves the Video-MME answers with request
# timing. The 11-27 control (FlashAttention split-KV pinned at 1 with the scheduler metadata dropped, mixed chunk off,
# thinker eager, radix cache off) made all 50 text and 20 talker answers identical on main and on #3452. Here each set
# drops one knob from the control (leave one out), main against the PR head as same-time pairs on the two cards, 2
# rounds a set swapped, text Video-MME c16 x3 and the talker test x3 a boot. A set whose answers stay identical across
# both arms does not need the knob it dropped.
set -u
O=/data/v1128
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main F=/workspace/fix
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
MIXED_OFF="--thinker.engine.enable_mixed_chunk false"
EAGER="--thinker.engine.disable_cuda_graph true --thinker.engine.cuda_graph_backend_prefill disabled"
RADIX_OFF="--thinker.engine.disable_radix_cache true"
until grep -q "q1128_a done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128_b start"
knobs() {
  local name=$1 tree=$2 card=$3 offset=$4 flags=$5 splits=$6
  local port=$(port_of $card $offset)
  local env_args="PYTHONPATH=$tree"
  if [ $splits = 1 ]; then
    env_args="PYTHONPATH=$K/thinker_numerics_probe:$tree OMNI_NUMERICS_PROBE=$O/$name/num OMNI_NUMERICS_TARGET_LENS=1 OMNI_FA_SPLITS=1"
  else
    :
  fi
  mkdir -p $O/$name
  note $name $tree "CI speech pipeline, $flags splits_pinned=$splits, text c16 x3, talker x3, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card $card --port $port \
    --out $O/$name --serve-extra "$flags" --serve-env "$env_args" --bench videomme_text --loads 16 --passes clean \
    --bench-repeats 3 --after "python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 3" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(cat $O/$name/num.*.splits 2>/dev/null | sort -u | head -1)"
}
offset=2000
for spec in "nosplit|$MIXED_OFF $EAGER $RADIX_OFF|0" "mixed|$EAGER $RADIX_OFF|1" "graphs|$MIXED_OFF $RADIX_OFF|1" \
  "radix|$MIXED_OFF $EAGER|1"; do
  set_name=${spec%%|*}
  rest=${spec#*|}
  flags=${rest%|*}
  splits=${rest##*|}
  offset=$((offset + 200))
  both "knobs l${set_name}1_m $M 0 $offset '$flags' $splits" "knobs l${set_name}1_f $F 1 $offset '$flags' $splits"
  offset=$((offset + 200))
  both "knobs l${set_name}2_f $F 0 $offset '$flags' $splits" "knobs l${set_name}2_m $M 1 $offset '$flags' $splits"
  for part in c16_clean talker; do
    python3 $K/videomme_answer_diff.py $O main=l${set_name}1_m/$part,l${set_name}2_m/$part \
      fix=l${set_name}1_f/$part,l${set_name}2_f/$part > $O/leaveout_${set_name}_$part.txt 2>&1
  done
  log "leave out $set_name: text $(grep -E '^answers' $O/leaveout_${set_name}_c16_clean.txt), talker $(grep -E '^answers' $O/leaveout_${set_name}_talker.txt)"
done
log "q1128_b done"
