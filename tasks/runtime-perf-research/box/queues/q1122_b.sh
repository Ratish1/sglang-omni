#!/usr/bin/env bash
# 11-22, node-0001, card 1 between q1122_a's trace pair and its WER scoring. In one context at c16 the talker waits for
# the turn 174 ms a second, behind thinker holds of 6.4 ms on average (MIXED steps hold 20 to 26 ms before finalize's
# release). py-spy of the combined form (OCVG 7cae415a1) at speech alone c16, 100 Hz for 30 s: the thinker thread is
# sampled only while it runs (its released and turn waits sleep), so its samples are the hold's composition.
set -u
O=/data/v1122
K=/workspace/kit/tasks/runtime-perf-research/box
OCVG=/workspace/ocvg
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS1=56-111,168-223
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "^T read" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py .*--card 1 " > /dev/null; do sleep 20; done
name=pc1_b port=31410
note $name $OCVG "seedtts c16 alone card 1 pyspy port $port"
taskset -c $CPUS1 python3 $K/baseline.py --tree $OCVG --model omni_colocated --card 1 --port $port --out $O/$name \
  --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,pyspy --pyspy-delay 20 --pyspy-seconds 30 \
  > $O/$name.driver.log 2>&1
log "boot $name rc $?"
shared_pid=$(sed -n 's/.*sglang_ar_startup stage=thinker .*pid=\([0-9]*\).*/\1/p' $O/$name/serve.log | head -1)
record=$O/$name/c16_pyspy/pyspy_$shared_pid.txt
for anchor in _run_batch get_next_batch_to_run process_input_requests process_batch_result build_forward_batch prepare_and_forward; do
  python3 $K/anchor_tree.py $record scheduler-thinker $anchor --top 30 > $O/$name.anchor_$anchor.txt 2>&1
done
python3 $K/anchor_tree.py $record scheduler-thinker run_serving_loop --top 40 > $O/$name.anchor_loop.txt 2>&1
python3 $K/anchor_tree.py $record scheduler-talker_ar run_serving_loop --top 30 > $O/$name.anchor_talker_loop.txt 2>&1
log "PC read pid $shared_pid $(head -1 $O/$name.anchor_loop.txt)"
log "q1122_b done"
