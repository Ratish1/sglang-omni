#!/usr/bin/env bash
# 11-23, node-0001, both cards, after q1123_a. Part 2 bisect: the final combined head F (085064564) against F with one
# mechanism removed (or, for the talker slot state, added), speech alone colocated c16 then c64 in one boot, same-time
# pairs on the two cards, swapped, after an F against F pair for the band. Then one traced c16 boot per arm (turn,
# thinker step and talker batch probes) for the mechanism reads.
# Decision rules, written before the runs: a row moves when both swapped pairs move it the same way past the F/F pair's
# delta for that row. Removal makes a row worse: the mechanism is needed, its traced read names what it buys. Removal
# makes a row better and none worse: the mechanism is dropped from part 2. No row moves and the traced read is level:
# the mechanism is dropped as unneeded. The slot arm opens on part 2 only with every row level or better.
set -u
O=/data/v1123
K=/workspace/kit/tasks/runtime-perf-research/box
F=/workspace/ocvg7
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
TRACE_PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_TURNS=1"
ARMS="no-yields no-step-yields no-defer no-backlog talker-slot equal-turn no-stream-priority lookahead read-in-turn"
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
git fetch -q origin $(for arm in $ARMS; do echo dev/qwen3-omni-oc-bisect-$arm; done)
for arm in $ARMS; do
  git worktree add -q --detach /workspace/x-$arm origin/dev/qwen3-omni-oc-bisect-$arm 2>/dev/null \
    || git -C /workspace/x-$arm checkout -q --detach origin/dev/qwen3-omni-oc-bisect-$arm
  log "tree x-$arm $(git -C /workspace/x-$arm rev-parse --short HEAD) $(git -C /workspace/x-$arm rev-parse --short=12 HEAD:sglang_omni)"
done
until grep -q "q1123_a3 done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1123_b start"
quiet_host
both "speech ff_f0 $F 0 210" "speech ff_f1 $F 1 210"
python3 $K/body_table.py $O c16 "F against F, same time, card 0 / card 1" f0=ff_f0 f1=ff_f1 >> $O/queue.log 2>&1
python3 $K/body_table.py $O c64 "F against F, same time, card 0 / card 1" f0=ff_f0 f1=ff_f1 >> $O/queue.log 2>&1
offset=230
for arm in $ARMS; do
  quiet_host
  both "speech ${arm}_p1_f0 $F 0 $offset" "speech ${arm}_p1_x1 /workspace/x-$arm 1 $offset"
  offset=$((offset + 20))
  quiet_host
  both "speech ${arm}_p2_x0 /workspace/x-$arm 0 $offset" "speech ${arm}_p2_f1 $F 1 $offset"
  offset=$((offset + 20))
  for cell in c16 c64; do
    python3 $K/body_table.py $O $cell "$arm against F, $cell, same-time pairs swapped" \
      F=${arm}_p1_f0,${arm}_p2_f1 $arm=${arm}_p1_x1,${arm}_p2_x0 >> $O/queue.log 2>&1
  done
  python3 $K/first_chunk_gap.py --cell c16_events $O F=${arm}_p1_f0,${arm}_p2_f1 $arm=${arm}_p1_x1,${arm}_p2_x0 2>&1 \
    | grep gaps >> $O/queue.log
  python3 $K/first_audio_states.py --cell c16_events --main $O/${arm}_p1_f0,$O/${arm}_p2_f1 \
    --branch $O/${arm}_p1_x1,$O/${arm}_p2_x0 > $O/${arm}_states_c16.txt 2>&1
  grep -E "server first audio|thinker prefill|thinker chunks|frames until" $O/${arm}_states_c16.txt >> $O/queue.log
  log "B $arm read"
done
set -- F $ARMS
while [ $# -gt 0 ]; do
  first=$1
  second=${2:-}
  tree_of() { if [ $1 = F ]; then echo $F; else echo /workspace/x-$1; fi; }
  quiet_host
  if [ -n "$second" ]; then
    both "traced tr_${first} $(tree_of $first) 0 $offset" "traced tr_${second} $(tree_of $second) 1 $offset"
    shift 2
  else
    traced tr_${first} $(tree_of $first) 0 $offset
    shift 1
  fi
  offset=$((offset + 20))
done
log "q1123_b done"
