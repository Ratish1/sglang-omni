#!/usr/bin/env bash
# 11-01: preprocessing workers bounded (94abfaaa2, on e354ea41f) against main, colocated, pinned, same-time pairs both
# card orders: SeedTTS EN streamed c64 and c16 (warm, two clean passes), then Video-MME CI with audio at c16 (two
# passes per boot: the first misses the encoder cache, the second hits it); then 3p against main on the two-card
# speech default, sequential boots a b b a.
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 WK=/workspace/wk_94abfaaa2 P3=/workspace/p3_fa091db9e
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
log "q1101_wk start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
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
  shift 5
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model $model --card $card --port $port --out $O/$name --bench-repeats 2 "$@" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pair() {
  local p0=$(freeport 24000); local p1=$(freeport $((p0 + 400)))
  local a=$1 ta=$2 b=$3 tb=$4
  shift 4
  boot $a $ta 0 $p0 omni_colocated "$@" &
  boot $b $tb 1 $p1 omni_colocated "$@" &
  wait
}
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
pair wk_m1 $MAIN wk_w1 $WK --loads 64,16 --passes warm,clean
pair wk_w2 $WK wk_m2 $MAIN --loads 64,16 --passes warm,clean
pair wv_m1 $MAIN wv_w1 $WK --bench videomme --loads 16 --passes clean --serve-extra "$CONTEXT_32K"
pair wv_w2 $WK wv_m2 $MAIN --bench videomme --loads 16 --passes clean --serve-extra "$CONTEXT_32K"
for arm in sd_m1:$MAIN sd_p1:$P3 sd_p2:$P3 sd_m2:$MAIN; do
  rm -rf $O/${arm%%:*}
  boot ${arm%%:*} ${arm#*:} 0 $(freeport 24000) omni_speech --loads 64,16 --passes warm,clean
done
log "q1101_wk done"
