#!/usr/bin/env bash
# 11-23, node-0001. Part 2's remaining c16 cost is the serialized host time under the turn: thinker holds over 10 ms
# average 13.4 ms, 10.4 of it the breakable prefill launch (48 segment replays, 48 eager FA3 calls). Prototype arms on F:
#   E1 full prefill graph, request slots = the thinker's 64 running requests, mixed chunk on   (5ee1c55f2)
#   E2 full prefill graph, SGLang's default 16 slots, mixed chunk off                         (6f477b116)
#   E3 breakable prefill graph, mixed chunk off, the control for E2's two changes             (c571ff0d6)
# A full batch past its slots runs eager, and a c16 MIXED step carries 16 running rows plus the new request, so E1 holds
# every running request and E2 removes the running rows from prefill steps. FA3 sizes its split combine over every slot.
# Card 1, beside q1123_e on card 0: one traced c16 boot per arm (captures, serves, the prefill graph's replay counts,
# the thinker's holds and the talker's turn waits). Both cards after q1123_e: same-time pairs against F, swapped.
set -u
O=/data/v1123
K=/workspace/kit/tasks/runtime-perf-research/box
F=/workspace/ocvg7
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
TRACE_PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_TURNS=1"
ARMS="full-prefill-64 full-prefill-unmixed breakable-unmixed"
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
graph_counts() {
  python3 - "$1" <<'PY' >> $O/queue.log 2>&1
import json, sys
info = json.load(open(f"{sys.argv[1]}/model_info.json"))
def find(node):
    if isinstance(node, dict):
        if "replay_count" in node and "backend" in node:
            return node
        for value in node.values():
            found = find(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find(value)
            if found:
                return found
    return None
found = find(info)
keys = ("backend", "replay_count", "standard_eager_count", "custom_eager_count")
print(sys.argv[1].rsplit("/", 1)[-1], "prefill graph", {key: found.get(key) for key in keys} if found else None)
PY
}
speech() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "c16,c64 card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads 16,64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio \
    --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(grep -m1 -o -E 'Traceback|Error[^ ]*' $O/$name/serve.log 2>/dev/null)"
}
traced() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "traced c16 card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio \
    --passes warm,events --probe --serve-env "$TRACE_PROBES" > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(grep -m1 -o -E 'Traceback|Error[^ ]*' $O/$name/serve.log 2>/dev/null)"
  graph_counts $O/$name
  python3 $K/turn_waits.py $O/$name --cell c16_events > $O/$name.turns.txt 2>&1
  python3 $K/turn_hold_parts.py $O/$name --cell c16_events > $O/$name.holds.txt 2>&1
  python3 $K/first_window_feed.py $O/$name --cell c16_events > $O/$name.feed.txt 2>&1
  { echo "$name"; grep -h "waiter priority" $O/$name.turns.txt; grep -h "thinker holds handed\|holds over" $O/$name.holds.txt
    grep -h "slowest\|all (median)" $O/$name.feed.txt; } >> $O/queue.log
  python3 $K/body_table.py $O c16 "$name alone (traced, one boot)" a=$name b=$name 2>&1 | sed -n '3,10p' >> $O/queue.log
}
cd /workspace/sglang-omni
git fetch -q origin $(for arm in $ARMS; do echo dev/qwen3-omni-oc-proto-$arm; done)
for arm in $ARMS; do
  git worktree add -q --detach /workspace/p-$arm origin/dev/qwen3-omni-oc-proto-$arm 2>/dev/null \
    || git -C /workspace/p-$arm checkout -q --detach origin/dev/qwen3-omni-oc-proto-$arm
  log "tree p-$arm $(git -C /workspace/p-$arm rev-parse --short HEAD) $(git -C /workspace/p-$arm rev-parse --short=12 HEAD:sglang_omni)"
done
until grep -q "q1123_c done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1123_f start"
offset=910
for arm in $ARMS; do
  traced tf_$arm /workspace/p-$arm 1 $offset
  offset=$((offset + 20))
done
log "F traced read"
until grep -q "q1123_e done" $O/queue.log 2>/dev/null; do sleep 20; done
for arm in $ARMS; do
  if ! grep -q "boot tf_$arm rc 0" $O/queue.log; then
    log "F $arm skipped: its traced boot failed"
    continue
  else
    :
  fi
  quiet_host
  both "speech pf_${arm}_p1_f0 $F 0 $offset" "speech pf_${arm}_p1_x1 /workspace/p-$arm 1 $offset"
  offset=$((offset + 20))
  quiet_host
  both "speech pf_${arm}_p2_x0 /workspace/p-$arm 0 $offset" "speech pf_${arm}_p2_f1 $F 1 $offset"
  offset=$((offset + 20))
  for cell in c16 c64; do
    python3 $K/body_table.py $O $cell "$arm against F, $cell, same-time pairs swapped" \
      F=pf_${arm}_p1_f0,pf_${arm}_p2_f1 $arm=pf_${arm}_p1_x1,pf_${arm}_p2_x0 >> $O/queue.log 2>&1
  done
  python3 $K/first_chunk_gap.py --cell c16_events $O F=pf_${arm}_p1_f0,pf_${arm}_p2_f1 \
    $arm=pf_${arm}_p1_x1,pf_${arm}_p2_x0 2>&1 | grep gaps >> $O/queue.log
  log "F $arm read"
done
log "q1123_f done"
