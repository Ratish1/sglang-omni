#!/usr/bin/env bash
# 11-01: the talker's prefill on SGLang's full prefill CUDA graph (one launch a step) against main's breakable graph
# (20 segments with eager attention between them), colocated. Starts when the 3s-on-request-build pairs end and stops
# q1101_next before its two-card boots, which run again at the end of this queue. Pairs: main (breakable) against the
# experiment tree with --talker_ar.engine.cuda_graph_backend_prefill full, same time, both card orders, c64 and c16,
# warm, two clean passes, an events pass (talker prefill step durations).
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 FULL=/workspace/full_b5431862c P3=/workspace/p3_fa091db9e
FULL_ARGS="--talker_ar.engine.cuda_graph_backend_prefill full"
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "boot rs_s2 rc" $O/queue.log 2>/dev/null && grep -q "boot rs_b2 rc" $O/queue.log 2>/dev/null; do sleep 10; done
NEXT=$(pgrep -f q1101_next.sh | head -1)
if [ -n "$NEXT" ]; then
  kill -TERM -$(ps -o pgid= -p $NEXT | tr -d ' ')
  sleep 5
  for d in $O/sd_m1; do [ -f $d/server.pgid ] && kill -TERM -$(cat $d/server.pgid); done
  sleep 20
  log "q1101_next stopped before its two-card boots (they rerun at the end of q1101_full)"
else
  :
fi
log "q1101_full start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
freeport() {
  python3 - "$1" <<'PY'
import socket, sys
p = int(sys.argv[1])
def free(q):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", q)); return True
    except OSError:
        return False
    finally:
        s.close()
while not (free(p) and free(p + 100)):
    p += 200
print(p)
PY
}
boot() {
  local name=$1 tree=$2 card=$3 port=$4 model=$5 extra=$6 passes=$7
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) extra '$extra' start"
  python3 $K/baseline.py --tree $tree --model $model --card $card --port $port --out $O/$name \
    --loads 64,16 --passes $passes --bench-repeats 2 --serve-extra "$extra" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pair() {
  local p0=$(freeport 24000); local p1=$(freeport $((p0 + 400)))
  boot $1 $2 0 $p0 omni_colocated "$3" warm,clean,events &
  boot $4 $5 1 $p1 omni_colocated "$6" warm,clean,events &
  wait
}
pair fp_m1 $MAIN "" fp_f1 $FULL "$FULL_ARGS"
pair fp_f2 $FULL "$FULL_ARGS" fp_m2 $MAIN ""
for arm in sd_m1:$MAIN sd_p1:$P3 sd_p2:$P3 sd_m2:$MAIN; do
  rm -rf $O/${arm%%:*}
  boot ${arm%%:*} ${arm#*:} 0 $(freeport 24000) omni_speech "" warm,clean
done
log "q1101_full done"
