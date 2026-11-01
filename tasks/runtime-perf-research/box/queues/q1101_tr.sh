#!/usr/bin/env bash
# 11-01, after q1101_aa: span traces on the marks trees. Colocated main with marks against 3s and 3p with marks as
# same-time pairs, one per card, then swapped; then the two-card speech default with marks (each model's step without
# the other on its card). Each boot: c16 and c64, warm then an 8 s nsys window 20 s into the pass (NVTX spans, OS
# runtime, graph-level CUDA trace, GPU context switches). Read with span_table.py.
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1101_aa done" $O/queue.log 2>/dev/null; do sleep 30; done
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
MARKS_ENV="SGLANG_ENABLE_NVTX_SCHEDULER=1 SGLANG_ENABLE_NVTX_OPERATIONS=1"
NSYS="--nsys-trace cuda,nvtx,osrt --nsys-graph-trace graph --nsys-gpuctxsw --nsys-delay 20 --nsys-seconds 8"
boot() {
  local name=$1 tree=$2 card=$3 port=$4 model=$5
  shift 5
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model $model --card $card --port $port --out $O/$name \
    --loads 16,64 --passes warm,nsys --nsys-session $name $NSYS --serve-env "$MARKS_ENV" "$@" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1101_tr start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
pair() {
  local a=$1 ta=$2 b=$3 tb=$4
  local p0=$(freeport 23000); local p1=$(freeport $((p0 + 400)))
  boot $a $ta 0 $p0 omni_colocated &
  boot $b $tb 1 $p1 omni_colocated &
  wait
}
pair tr_m1 /workspace/marks tr_s1 /workspace/marks_3s
pair tr_p1 /workspace/marks_3p tr_m2 /workspace/marks
pair tr_s2 /workspace/marks_3s tr_p2 /workspace/marks_3p
boot tr_sd /workspace/marks 0 $(freeport 23000) omni_speech
log "q1101_tr done"
