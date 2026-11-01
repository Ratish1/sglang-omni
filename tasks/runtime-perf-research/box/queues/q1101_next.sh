#!/usr/bin/env bash
# 11-01, after the request build pairs: (1) suites on card 0 while it is free: 3p (Qwen3-Omni and model runner),
# the request build and the stack (Qwen3-Omni), and each new test against main's sources; (2) 3s on the request
# build (rbs) against the request build alone (rb), colocated, same-time pairs both card orders, warm then two clean
# passes; (3) 3p against main on the two-card speech default, sequential boots a b b a (both cards per boot).
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 RB=/workspace/rb_49a25f5f9 RBS=/workspace/rbs_35c464cf8 P3=/workspace/p3_fa091db9e
U=$O/units
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1101_rb done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1101_next start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
mkdir -p $U
suite() {
  local name=$1 tree=$2
  shift 2
  (cd $tree && python3 -m pytest -q "$@" -p no:cacheprovider > $U/$name.log 2>&1; echo "rc $?" >> $U/$name.log)
  log "units $name $(tail -2 $U/$name.log | tr '\n' ' ')"
}
on_main() {
  local name=$1 file=$2 key=$3
  mkdir -p /tmp/onmain_$name && cp $file /tmp/onmain_$name/
  (cd $MAIN && python3 -m pytest -q /tmp/onmain_$name/$(basename $file) -k "$key" -p no:cacheprovider --rootdir=$MAIN > $U/$name.log 2>&1; echo "rc $?" >> $U/$name.log)
  log "units $name $(tail -2 $U/$name.log | tr '\n' ' ')"
}
suite units_3p $P3 tests/unit_test/qwen3_omni tests/unit_test/model_runner
on_main units_3p_test_on_main $P3/tests/unit_test/qwen3_omni/test_logit_shaping.py do_not_synchronize
suite units_rb $RB tests/unit_test/qwen3_omni
on_main units_rb_test_on_main $RB/tests/unit_test/qwen3_omni/test_talker.py prompt_prefill_build_does_not_synchronize
suite units_rbs $RBS tests/unit_test/qwen3_omni
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
  local name=$1 tree=$2 card=$3 port=$4 model=$5
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model $model --card $card --port $port --out $O/$name \
    --loads 64,16 --passes warm,clean --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pair() {
  local p0=$(freeport 24000); local p1=$(freeport $((p0 + 400)))
  boot $1 $2 0 $p0 omni_colocated &
  boot $3 $4 1 $p1 omni_colocated &
  wait
}
pair rs_b1 $RB rs_s1 $RBS
pair rs_s2 $RBS rs_b2 $RB
for arm in sd_m1:$MAIN sd_p1:$P3 sd_p2:$P3 sd_m2:$MAIN; do
  boot ${arm%%:*} ${arm#*:} 0 $(freeport 24000) omni_speech
done
log "q1101_next done"
