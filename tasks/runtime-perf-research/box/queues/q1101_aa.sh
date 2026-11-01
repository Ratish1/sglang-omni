#!/usr/bin/env bash
# 11-01: main's own colocated A/A. Upstream main e354ea41f on card 0 and card 1 at the same time, two rounds, SeedTTS
# EN streamed c64 and c16, warm then two clean passes. Read paired by prompt (latency_paired.py) between the cards
# of a round and across rounds: that band is the bar for every candidate.
# usage: q1101_aa.sh [prefix aa] [rounds 2] [queue.log line to wait for]; launch under taskset on the cards' NUMA node.
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354
PREFIX=${1:-aa} ROUNDS=${2:-2} AFTER=${3:-}
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
if [ -n "$AFTER" ]; then
  until grep -q "$AFTER" $O/queue.log 2>/dev/null; do sleep 30; done
else
  :
fi
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
  local name=$1 tree=$2 card=$3 port=$4
  shift 4
  log "boot $name card $card port $port start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name "$@" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(sed -n 2,3p $O/$name/identity.txt | tr '\n' ' ')"
}
S="--loads 64,16 --passes warm,clean --bench-repeats 2"
log "q1101_aa $PREFIX start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
for round in $(seq $ROUNDS); do
  p0=$(freeport 22000); p1=$(freeport $((p0 + 400)))
  boot ${PREFIX}${round}_c0 $MAIN 0 $p0 $S &
  boot ${PREFIX}${round}_c1 $MAIN 1 $p1 $S &
  wait
done
log "q1101_aa $PREFIX done"
