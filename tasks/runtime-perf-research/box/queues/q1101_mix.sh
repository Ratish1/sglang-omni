#!/usr/bin/env bash
# 11-01: mixed admissions on the request build and 3s (rbsm, test/rbs-mixed-20261101 883b4eb6d) against main,
# colocated: does an admission riding a decode step remove the c16 first-window cost of separate talker prefill
# steps? Suites first (card 0 free), then same-time pairs both card orders, c64 and c16, warm, two clean passes,
# events; then 3p against main on the two-card speech default, sequential boots a b b a.
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 RBSM=/workspace/rbsm_883b4eb6d P3=/workspace/p3_fa091db9e
U=$O/units
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
log "q1101_mix start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
(cd $RBSM && python3 -m pytest -q tests/unit_test/qwen3_omni tests/unit_test/model_runner -p no:cacheprovider > $U/units_rbsm.log 2>&1; echo "rc $?" >> $U/units_rbsm.log)
log "units units_rbsm $(grep -E 'passed|failed' $U/units_rbsm.log | tail -1) $(tail -1 $U/units_rbsm.log)"
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
  local name=$1 tree=$2 card=$3 port=$4 model=$5 passes=$6
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model $model --card $card --port $port --out $O/$name \
    --loads 64,16 --passes $passes --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pair() {
  local p0=$(freeport 24000); local p1=$(freeport $((p0 + 400)))
  boot $1 $2 0 $p0 omni_colocated warm,clean,events &
  boot $3 $4 1 $p1 omni_colocated warm,clean,events &
  wait
}
pair mx_m1 $MAIN mx_x1 $RBSM
pair mx_x2 $RBSM mx_m2 $MAIN
for arm in sd_m1:$MAIN sd_p1:$P3 sd_p2:$P3 sd_m2:$MAIN; do
  rm -rf $O/${arm%%:*}
  boot ${arm%%:*} ${arm#*:} 0 $(freeport 24000) omni_speech warm,clean
done
log "q1101_mix done"
