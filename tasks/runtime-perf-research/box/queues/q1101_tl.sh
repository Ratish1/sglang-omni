#!/usr/bin/env bash
# 11-01: the Qwen3-Omni thinker's speech batches on the lookahead (test/rbsm-thinker-lookahead-20261101 314230c54, on
# the mixed stack). Starts when the mixed stack's pairs end and stops q1101_mix before its two-card boots. A smoke boot
# under a 15 minute limit first (10-31 the same gate change hung the talker); only if it serves, same-time pairs
# against main both card orders with events; then 3p against main on the two-card speech default.
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 TL=/workspace/tl_314230c54 P3=/workspace/p3_fa091db9e
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "boot mx_x2 rc" $O/queue.log 2>/dev/null && grep -q "boot mx_m2 rc" $O/queue.log 2>/dev/null; do sleep 10; done
MIX=$(pgrep -f q1101_mix.sh | head -1)
if [ -n "$MIX" ]; then
  kill -TERM -$(ps -o pgid= -p $MIX | tr -d ' ')
  sleep 5
  [ -f $O/sd_m1/server.pgid ] && kill -TERM -$(cat $O/sd_m1/server.pgid)
  sleep 20
  log "q1101_mix stopped before its two-card boots (they rerun at the end of q1101_tl)"
else
  :
fi
log "q1101_tl start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
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
  local name=$1 tree=$2 card=$3 port=$4 model=$5 loads=$6 passes=$7 limit=$8
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  timeout $limit python3 $K/baseline.py --tree $tree --model $model --card $card --port $port --out $O/$name \
    --loads $loads --passes $passes --bench-repeats 2 > $O/$name.driver.log 2>&1
  local rc=$?
  if [ $rc = 124 ] && [ -f $O/$name/server.pgid ]; then
    kill -TERM -$(cat $O/$name/server.pgid); sleep 20; kill -KILL -$(cat $O/$name/server.pgid) 2>/dev/null
    log "boot $name timed out after $limit s, server group stopped"
  else
    :
  fi
  log "boot $name rc $rc"
  return $rc
}
if boot tl_smoke $TL 0 $(freeport 24000) omni_colocated 16 warm 900; then
  pair() {
    local p0=$(freeport 24000); local p1=$(freeport $((p0 + 400)))
    boot $1 $2 0 $p0 omni_colocated 64,16 warm,clean,events 3600 &
    boot $3 $4 1 $p1 omni_colocated 64,16 warm,clean,events 3600 &
    wait
  }
  pair tl_m1 $MAIN tl_t1 $TL
  pair tl_t2 $TL tl_m2 $MAIN
else
  log "tl_smoke failed: no pairs"
fi
for arm in sd_m1:$MAIN sd_p1:$P3 sd_p2:$P3 sd_m2:$MAIN; do
  rm -rf $O/${arm%%:*}
  boot ${arm%%:*} ${arm#*:} 0 $(freeport 24000) omni_speech 64,16 warm,clean 3600
done
log "q1101_tl done"
