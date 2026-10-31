set -u
# 10-31 the decode payload carries only the text state decode reads (perf/qwen3-omni-decode-payload-text-state d7dc4927d,
# /data/ab/dts) against upstream main 3cb3c6ccb (/data/ab/main3cb, the tools branch's tree, sglang_omni identical).
# 1. The touched component: the branch on card 0 with shm_put_size_probe, one cold SeedTTS EN c64 pass, read against
#    queue105's main pass (the thinker's decode payload pack with device inputs in 680 of 1,089 requests).
# 2. Served: SeedTTS EN H100 BF16 colocated, full corpus, same-time pairs on cards 0 and 1 swapped by card, request events
#    per boot: c64 streaming two pairs, c16 streaming and non-streaming two pairs, and one main A/A at c16 (the node's
#    c64 A/A is queue101's a64a and a64b). No scoring: decode's text and the talker's inputs cannot change.
log() { echo "$(date -u +%T) $*" >> /data/q106/queue.log; }
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
Q=/data/q106
R=/data/q100/q101_boots.py
export SGLANG_OMNI_STRICT_PORT=1 PIN_NODE=0
PIN0=0-15,64-79
PIN1=16-31,80-95
next_port=30100
mkdir -p $Q/put
log "queue106 start: tools $(git -C /data/tools rev-parse --short HEAD)"
for arm in dts main3cb; do
  log "$arm $(git -C /data/ab/$arm rev-parse --short HEAD) $(git -C /data/ab/$arm rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /data/ab/$arm status --porcelain -- sglang_omni | wc -l) dirty"
done

log "component: dts with the put probe on card 0, one c64 pass"
SERVER_PYTHONPATH=$S/shm_put_size_probe OMNI_SHM_PUT_PROBE=$Q/put PIN_CPUS=$PIN0 \
  bash $S/run_bench_boot.sh /data/ab/dts $Q/component 0 $next_port bf16 64 seedtts_en
next_port=$((next_port + 400))
log "component done: $(tr '\n' ';' < $Q/component/progress.txt | cut -c1-300) $(cat $Q/component/FAILED 2>/dev/null)"

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
pair p64a main3cb dts 64 seedtts_en
pair p64b dts main3cb 64 seedtts_en
python3 $S/pair_rows.py main3cb $Q/p64a $Q/p64b > $Q/rows_p64.txt 2>&1
T16="seedtts_en seedtts_en_nostream"
pair p16a dts main3cb 16 "$T16"
pair p16b main3cb dts 16 "$T16"
pair a16a main3cb main3cb 16 "$T16"
python3 $S/pair_rows.py main3cb $Q/p16a $Q/p16b > $Q/rows_p16.txt 2>&1
python3 $S/pair_rows.py main3cb $Q/a16a > $Q/rows_a16.txt 2>&1
log "queue106 done"
