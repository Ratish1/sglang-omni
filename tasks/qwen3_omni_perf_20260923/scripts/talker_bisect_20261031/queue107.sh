set -u
# 10-31 queue106's c64 rows leaned against the branch in both card orders (QPS -0.5 / -0.4 %, paired per-request latency
# +0.34 / +0.33 %), inside the node's c64 A/A (queue101 a64a and a64b, +1.15 / +0.03 % paired) but two pairs cannot
# resolve 0.4 %: two more c64 pairs swapped by card and a same-time c64 A/A of today's main, request events per boot.
log() { echo "$(date -u +%T) $*" >> /data/q107/queue.log; }
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
Q=/data/q107
R=/data/q100/q101_boots.py
export SGLANG_OMNI_STRICT_PORT=1 PIN_NODE=0
PIN0=0-15,64-79
PIN1=16-31,80-95
next_port=32500
mkdir -p $Q
log "queue107 start: tools $(git -C /data/tools rev-parse --short HEAD)"
for arm in dts main3cb; do
  log "$arm $(git -C /data/ab/$arm rev-parse --short HEAD) $(git -C /data/ab/$arm rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /data/ab/$arm status --porcelain -- sglang_omni | wc -l) dirty"
done
attempt() {
  local round=$1 tree0=$2 tree1=$3 conc=$4 arms=$5
  local p0=$next_port p1=$((next_port + 200))
  next_port=$((next_port + 400))
  local o0=$Q/$round/card0_$tree0 o1=$Q/$round/card1_$tree1
  log "round $round start ($arms c$conc, card0 $tree0 port $p0, card1 $tree1 port $p1)"
  EVENTS=1 SCORE=0 PIN_CPUS=$PIN0 bash $S/run_bench_boot.sh /data/ab/$tree0 $o0 0 $p0 bf16 $conc "$arms" &
  local b0=$!
  EVENTS=1 SCORE=0 PIN_CPUS=$PIN1 bash $S/run_bench_boot.sh /data/ab/$tree1 $o1 1 $p1 bf16 $conc "$arms" &
  local b1=$!
  while kill -0 $b0 2>/dev/null || kill -0 $b1 2>/dev/null; do
    if [ -f $o0/FAILED ] && [ ! -f $o1/DONE ] && [ -f $o1/serve.pgid ]; then kill -TERM -- -$(cat $o1/serve.pgid) 2>/dev/null; fi
    if [ -f $o1/FAILED ] && [ ! -f $o0/DONE ] && [ -f $o0/serve.pgid ]; then kill -TERM -- -$(cat $o0/serve.pgid) 2>/dev/null; fi
    sleep 5
  done
  wait
  if [ -f $o0/FAILED ] || [ -f $o1/FAILED ]; then
    log "round $round void: $(cat $o0/FAILED $o1/FAILED 2>/dev/null | tr '\n' ' ')"
    return 1
  fi
  python3 $R $Q/$round 2>&1 | while read -r line; do log "  $line"; done
  log "round $round done"
}
pair() {
  attempt "$@" && return 0
  mv $Q/$1 $Q/${1}_void1
  attempt "$@" && return 0
  mv $Q/$1 $Q/${1}_void2
  log "round $1 failed twice, skipped"
}
pair p64c main3cb dts 64 seedtts_en
pair p64d dts main3cb 64 seedtts_en
pair a64a main3cb main3cb 64 seedtts_en
python3 $S/pair_rows.py main3cb /data/q106/p64a /data/q106/p64b $Q/p64c $Q/p64d > $Q/rows_p64_four.txt 2>&1
python3 $S/pair_rows.py main3cb $Q/a64a > $Q/rows_a64.txt 2>&1
log "queue107 done"
