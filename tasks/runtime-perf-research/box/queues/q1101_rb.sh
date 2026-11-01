#!/usr/bin/env bash
# 11-01, after the single-boot traces (q1101_tr3): the talker request build without a stream sync (rb, 49a25f5f9)
# against main, colocated, same-time pairs on the two cards then swapped, plain trees, SeedTTS EN streamed c64 and
# c16, warm, two clean passes, then an events pass (talker prefill steps per request). Launch under taskset on the
# cards' NUMA node; read paired by prompt and against main's pinned band.
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 RB=/workspace/rb_49a25f5f9
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1101_tr3 done" $O/queue.log 2>/dev/null; do sleep 30; done
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
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64,16 --passes warm,clean,events --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pair() {
  local p0=$(freeport 24000); local p1=$(freeport $((p0 + 400)))
  boot $1 $2 0 $p0 &
  boot $3 $4 1 $p1 &
  wait
}
log "q1101_rb start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
pair rb_m1 $MAIN rb_b1 $RB
pair rb_b2 $RB rb_m2 $MAIN
log "q1101_rb done"
