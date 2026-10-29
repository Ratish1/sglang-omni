attempt() {
  local round=$1 tree0=$2 tree1=$3 conc=$4 arms=$5 extra=$6
  local p0 p1 o0=/data/ab/$round/card0_$tree0 o1=/data/ab/$round/card1_$tree1
  p0=$(python3 /data/ab/pick_port.py $USED) && p1=$(python3 /data/ab/pick_port.py $USED) || { log "round $round no free port"; return 1; }
  log "round $round start ($arms c$conc, card0 $tree0 port $p0, card1 $tree1 port $p1)"
  env $extra PIN_CPUS=0-27,112-139 PIN_NODE=0 bash $S/run_bench_boot.sh /data/ab/$tree0 $o0 0 $p0 h200 $conc "$arms" &
  local b0=$!
  env $extra PIN_CPUS=28-55,140-167 PIN_NODE=0 bash $S/run_bench_boot.sh /data/ab/$tree1 $o1 1 $p1 h200 $conc "$arms" &
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
  log "round $round done"
}
pair() {
  local round=$1
  attempt "$@" && return 0
  mv /data/ab/$round /data/ab/${round}_void1
  attempt "$@" && return 0
  mv /data/ab/$round /data/ab/${round}_void2
  log "round $round failed twice, skipped"
}
probe() {
  local round=$1 tree0=$2 tree1=$3 conc=$4 arms=$5 samples=$6
  local p0 p1
  p0=$(python3 /data/ab/pick_port.py $USED) && p1=$(python3 /data/ab/pick_port.py $USED) || { log "probe $round no free port"; return 1; }
  log "probe $round start ($arms c$conc, card0 $tree0 port $p0, card1 $tree1 port $p1)"
  OMNI_SHM_PUT_PROBE=/data/sizes/$round/card0_$tree0/records SERVER_PYTHONPATH=$S/shm_put_size_probe MAX_SAMPLES=$samples PIN_CPUS=0-27,112-139 PIN_NODE=0 bash $S/run_bench_boot.sh /data/ab/$tree0 /data/sizes/$round/card0_$tree0/run 0 $p0 h200 $conc "$arms" &
  OMNI_SHM_PUT_PROBE=/data/sizes/$round/card1_$tree1/records SERVER_PYTHONPATH=$S/shm_put_size_probe MAX_SAMPLES=$samples PIN_CPUS=28-55,140-167 PIN_NODE=0 bash $S/run_bench_boot.sh /data/ab/$tree1 /data/sizes/$round/card1_$tree1/run 1 $p1 h200 $conc "$arms" &
  wait
  for boot in card0_$tree0 card1_$tree1; do
    taskset -c 0-3 python3 $S/shm_put_sizes.py /data/sizes/$round/$boot/records > /data/sizes/$round/$boot/sizes.txt 2>&1
  done
  log "probe $round done"
}
