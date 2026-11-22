#!/usr/bin/env bash
# 11-23, node-0001, both cards, after q1123_b. The defer rule's scope. The no-defer arm read c16 first audio mean +8.8 /
# +12.1 % and p99 +25 / +18 % without it, but inter chunk mean -5.0 / -3.6 % and later-gap p99 215 against 247 ms: the
# rule defers the whole talker while any row waits on text, and at c16 some row nearly always does. Only a stream still
# filling code2wav's first window (10 frames, stream_chunk_size, the talker's prefill frame and 9 decode steps) is
# text-bound for first audio, and a steady stream plays at least one 10-frame window it already has. Arm: F deferring
# only while such a row waits (a5b04474b, tree dda881d378f0), against F, same-time pairs swapped, then traced c16.
# Decision rule: it replaces the rule if c16 first audio stays level against F and inter chunk improves in both pairs.
set -u
O=/data/v1123
K=/workspace/kit/tasks/runtime-perf-research/box
F=/workspace/ocvg7
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
TRACE_PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_TURNS=1"
ARM=defer-first-window
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((28000 + 2000 * $1 + $2)); }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
speech() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "c16,c64 card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads 16,64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio \
    --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
traced() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "traced c16 card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio \
    --passes warm,events --probe --serve-env "$TRACE_PROBES" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
  python3 $K/turn_waits.py $O/$name --cell c16_events > $O/$name.turns.txt 2>&1
  python3 $K/turn_hold_parts.py $O/$name --cell c16_events > $O/$name.holds.txt 2>&1
  python3 $K/first_window_feed.py $O/$name --cell c16_events > $O/$name.feed.txt 2>&1
  { echo "$name"; grep -h "waiter priority" $O/$name.turns.txt; grep -h "thinker holds handed\|holds over" $O/$name.holds.txt
    grep -h "slowest\|all (median)" $O/$name.feed.txt; } >> $O/queue.log
}
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-oc-bisect-$ARM
git worktree add -q --detach /workspace/x-$ARM origin/dev/qwen3-omni-oc-bisect-$ARM 2>/dev/null \
  || git -C /workspace/x-$ARM checkout -q --detach origin/dev/qwen3-omni-oc-bisect-$ARM
log "tree x-$ARM $(git -C /workspace/x-$ARM rev-parse --short HEAD) $(git -C /workspace/x-$ARM rev-parse --short=12 HEAD:sglang_omni)"
until grep -q "q1123_b done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1123_d start"
quiet_host
both "speech ${ARM}_p1_f0 $F 0 710" "speech ${ARM}_p1_x1 /workspace/x-$ARM 1 710"
quiet_host
both "speech ${ARM}_p2_x0 /workspace/x-$ARM 0 730" "speech ${ARM}_p2_f1 $F 1 730"
for cell in c16 c64; do
  python3 $K/body_table.py $O $cell "$ARM against F, $cell, same-time pairs swapped" \
    F=${ARM}_p1_f0,${ARM}_p2_f1 $ARM=${ARM}_p1_x1,${ARM}_p2_x0 >> $O/queue.log 2>&1
done
python3 $K/first_chunk_gap.py --cell c16_events $O F=${ARM}_p1_f0,${ARM}_p2_f1 $ARM=${ARM}_p1_x1,${ARM}_p2_x0 2>&1 \
  | grep gaps >> $O/queue.log
python3 $K/first_audio_states.py --cell c16_events --main $O/${ARM}_p1_f0,$O/${ARM}_p2_f1 \
  --branch $O/${ARM}_p1_x1,$O/${ARM}_p2_x0 > $O/${ARM}_states_c16.txt 2>&1
grep -E "server first audio|thinker prefill|thinker chunks|frames until" $O/${ARM}_states_c16.txt >> $O/queue.log
log "D $ARM read"
quiet_host
traced tr_${ARM} /workspace/x-$ARM 0 750
log "q1123_d done"
