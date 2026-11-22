#!/usr/bin/env bash
# 11-23, node-0001, after q1123_b. Two reads outside the turn's lines.
#   1. The host cost part 2 adds to a process holding one SGLang engine (every other model): the no-op turn calls per
#      loop pass, CPU only, one core of card 0's socket.
#   2. Interpreter lock ownership by thread role at speech alone c16, part 2's final head F (one process for thinker,
#      talker and code2wav) on card 0 and its base (two processes) on card 1 at the same time: py-spy --gil 15 s, 20 s
#      into the pass, then the same on swapped cards.
set -u
O=/data/v1123
K=/workspace/kit/tasks/runtime-perf-research/box
F=/workspace/ocvg7 BASE=/workspace/bvg
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((28000 + 2000 * $1 + $2)); }
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
gil() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "pyspygil c16 card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio \
    --passes warm,pyspygil --pyspy-delay 20 --pyspy-seconds 15 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1123_b done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1123_c start"
PYTHONPATH=$F taskset -c 3 python3 $K/process_engines_noop_bench.py >> $O/queue.log 2>&1
until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
both "gil g1_f0 $F 0 810" "gil g1_b1 $BASE 1 810"
both "gil g2_b0 $BASE 0 830" "gil g2_f1 $F 1 830"
for name in g1_f0 g1_b1 g2_b0 g2_f1; do
  cell=$(cd $O/$name && ls -d c16_pyspygil* 2>/dev/null | head -1)
  python3 $K/gil_threads.py $O/$name --cell ${cell:-c16_pyspygil} >> $O/queue.log 2>&1
done
log "q1123_c done"
