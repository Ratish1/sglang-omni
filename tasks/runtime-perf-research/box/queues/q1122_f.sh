#!/usr/bin/env bash
# 11-22, node-0001, card 1 beside q1122_c on card 0. The first window is text-bound (traced: a request's 9 talker steps
# each follow one thinker text row), so talker-first priority slows the stream it serves; the yields slowed the thinker
# 17 % more. Part 2 now ranks the talker below the thinker while a talker row waits on thinker text (230deaf49).
# Combined with it: OCVG4 57389d260 (tree b1a8474a6643) against part 2's base BVG 6aff09f46, card 1, B A A B, speech
# alone c1 and c16; then the same pair traced for the first window's feed.
set -u
O=/data/v1122
K=/workspace/kit/tasks/runtime-perf-research/box
BVG=/workspace/bvg OCVG4=/workspace/ocvg4
OCVG4_SHA=57389d260
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS=56-111,168-223
TRACE_PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_TURNS=1"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
BOOTS=0
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
next_port() {
  BOOTS=$((BOOTS + 1))
  echo $((31000 + 90 + 20 * BOOTS))
}
alone() {
  local name=$1 tree=$2 loads=$3
  local port=$(next_port)
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
  note $name $tree "seedtts loads $loads alone card 1 port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 1 --port $port --out $O/$name \
    --loads $loads --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
traced() {
  local name=$1 tree=$2
  local port=$(next_port)
  note $name $tree "seedtts c16 alone card 1 traced port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 1 --port $port --out $O/$name \
    --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events --probe --serve-env "$TRACE_PROBES" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until ! pgrep -f "box/baseline.py .*--card 1 " > /dev/null; do sleep 20; done
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-one-context-visual-graph-read
git worktree add -q --detach $OCVG4 FETCH_HEAD
if [ "$(git -C $OCVG4 rev-parse --short=9 HEAD)" != "$OCVG4_SHA" ] || ! git -C $OCVG4 diff --quiet HEAD -- sglang_omni; then
  log "q1122_f stopped: ocvg4 is not a clean $OCVG4_SHA"
  exit 1
else :; fi
log "q1122_f start, $(node_free)"
alone db1 $OCVG4 1,16
alone da2 $BVG 1,16
alone da3 $BVG 1,16
alone db4 $OCVG4 1,16
for cell in c1 c16; do
  python3 $K/body_table.py $O $cell "part 2 with the thinker first while a talker row waits on text (57389d260) against its base, card 1 B A A B, speech alone $cell" \
    two=da2,da3 one=db1,db4 >> $O/queue.log 2>&1
done
python3 $K/first_audio_states.py --cell c16_events --main $O/da2,$O/da3 --branch $O/db1,$O/db4 > $O/f_states_c16.txt 2>&1
grep -E "server first audio|frames until|thinker prefill|thinker build" $O/f_states_c16.txt >> $O/queue.log
log "F1 read"
traced tb3 $OCVG4
python3 $K/first_window_feed.py $O/tb3 --cell c16_events > $O/tb3.feed.txt 2>&1
python3 $K/turn_waits.py $O/tb3 --cell c16_events > $O/tb3.turns_c16.txt 2>&1
grep -h "slowest\|all (median)" $O/tb3.feed.txt >> $O/queue.log
grep -h "waiter priority" $O/tb3.turns_c16.txt >> $O/queue.log
log "F2 read"
log "q1122_f done"
