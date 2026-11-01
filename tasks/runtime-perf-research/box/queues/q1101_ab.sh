#!/usr/bin/env bash
# 11-01, after the pinned A/A (q1101_aa.sh ap): 3s and 3p against main, colocated, same-time pairs on the two cards
# then swapped, plain trees (no marks), SeedTTS EN streamed c64 and c16, warm then two clean passes. Launch under
# taskset on the cards' NUMA node; read paired by prompt and against the ap band.
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 S3=/workspace/s3_76a1fba20 P3=/workspace/p3_fa091db9e
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1101_aa ap done" $O/queue.log 2>/dev/null; do sleep 30; done
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
    --loads 64,16 --passes warm,clean --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pair() {
  local p0=$(freeport 24000); local p1=$(freeport $((p0 + 400)))
  boot $1 $2 0 $p0 &
  boot $3 $4 1 $p1 &
  wait
}
log "q1101_ab start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
pair ab_m1 $MAIN ab_s1 $S3
pair ab_s2 $S3 ab_m2 $MAIN
pair ab_m3 $MAIN ab_p1 $P3
pair ab_p2 $P3 ab_m4 $MAIN
log "q1101_ab done"
