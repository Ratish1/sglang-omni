#!/usr/bin/env bash
# 11-22, node-0001, card 1 (free). Feasibility of SGLang's full prefill graph for the thinker, the lever left for part 2's
# c16 tail (talker waits behind thinker holds of about 20 ms at p99, most of a MIXED step's hold being the breakable
# graph's per-layer eager attention). The final head (OCVG7 085064564) with the thinker's prefill backend set to full
# through its server_args_overrides, speech alone c16 with the turn and thinker step probes: does it capture and serve,
# and how long do MIXED steps and the talker's turn waits run, against tb3 (breakable).
set -u
O=/data/v1122
K=/workspace/kit/tasks/runtime-perf-research/box
OCVG7=/workspace/ocvg7
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
FULL_PREFILL="--thinker.factory.server_args_overrides.cuda_graph_backend_prefill full"
CPUS1=56-111,168-223
TRACE_PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_TURNS=1"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until ! pgrep -f "box/baseline.py .*--card 1 " > /dev/null; do sleep 20; done
log "q1122_l start"
log "boot fp1 $(git -C $OCVG7 rev-parse --short HEAD) full prefill traced c16 card 1 start"
taskset -c $CPUS1 python3 $K/baseline.py --tree $OCVG7 --model omni_colocated --card 1 --port 31330 --out $O/fp1 \
  --loads 16 --serve-extra "$CONTEXT_32K $FULL_PREFILL" --omni-chunk-audio --passes warm,events --probe \
  --serve-env "$TRACE_PROBES" > $O/fp1.driver.log 2>&1
log "boot fp1 rc $? $(grep -m1 -o -E \"Traceback|Error[^ ]*\" $O/fp1/serve.log)"
python3 - >> $O/queue.log 2>&1 <<'PY'
import json
info = json.load(open("/data/v1122/fp1/model_info.json"))
def find(o):
    if isinstance(o, dict):
        if "replay_count" in o and "backend" in o:
            return o
        for value in o.values():
            found = find(value)
            if found:
                return found
    elif isinstance(o, list):
        for value in o:
            found = find(value)
            if found:
                return found
print("fp1 prefill graph", {key: find(info).get(key) for key in ("backend", "backend_runner", "replay_count", "standard_eager_count", "custom_eager_count")} if find(info) else None)
PY
python3 $K/turn_waits.py $O/fp1 --cell c16_events > $O/fp1.turns_c16.txt 2>&1
python3 $K/turn_hold_parts.py $O/fp1 --cell c16_events > $O/fp1.holds_c16.txt 2>&1
python3 $K/first_window_feed.py $O/fp1 --cell c16_events > $O/fp1.feed.txt 2>&1
grep -h "waiter priority" $O/fp1.turns_c16.txt >> $O/queue.log
grep -h "thinker holds handed\|holds over" $O/fp1.holds_c16.txt >> $O/queue.log
grep -h "slowest\|all (median)" $O/fp1.feed.txt >> $O/queue.log
grep -h "c16_events" $O/fp1/c16_events/bench/*.log 2>/dev/null | tail -1 >> $O/queue.log
log "L read"
log "q1122_l done"
