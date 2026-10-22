#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1, after queue k: L4 and the stack on main 3d7d2c306. t_m (upstream main), t_l2
# (L2), t_l3 (main + L2 + L3), t_l4 (main + L2 + L3 + L4, the L4 PR branch). p_m1 t_l3 against t_l4 (L4 alone) and p_m2
# t_m against t_l4 (the three together), cells c1f seeded, c16, c64, L16; p_m3 t_m against t_l2 (L2 alone) through c16,
# c64, b16, L16 with every card's framebuffer sampled each second and each cell's start kept; then L4's worst step at
# 16 and 32 rows. Both orders, stream gaps pooled per arm.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_m.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_m_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $D/queue_m.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
tree() {
  local name=$1 remote=$2 branch=$3 sha
  git -C /workspace/sglang-omni fetch -q $remote $branch
  sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
  if [ -d /workspace/$name ]; then
    git -C /workspace/$name checkout -q --detach $sha
  else
    git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
  fi
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD -- sglang_omni || echo DIRTY)"
}
warm() {
  local tree=$1 card=$2 port=$3
  bash $IN/warm_c20261021.sh $tree $card $(cpus $card) $port $D/warm_$tree
  log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
}
pair() {
  local first=$1 second=$2 name=$3 cells=$4
  (cd /workspace/sglang-omni && PAIR_CELLS=$cells python3 $IN/serve_pair_c20261021.py /workspace/$first \
    /workspace/$second $D/$name > $D/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
  for cell in ${cells//,/ }; do
    case $cell in
      c16|c64|L16|c1f)
        python3 $IN/pooled_inter_chunk_c20261021.py $D/$name $cell $first $second > $D/pooled_${name}_$cell.txt 2>&1
        log "pooled $name $cell $(grep '^pooled' $D/pooled_${name}_$cell.txt | tr '\n' ' ')"
        ;;
      *)
        :
        ;;
    esac
  done
}
until grep -q "queue k done" $D/queue_k.txt; do sleep 30; done
log "queue m start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
tree t_m upstream main
tree t_l4 origin perf/cosyvoice3-prefix-pool-from-free-memory
for name in t_l2 t_l3; do
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8)"
done
warm t_m 0 44000 &
warm t_l4 1 44400 &
wait
pair t_l3 t_l4 p_m1 c1f,c16,c64,L16
pair t_m t_l4 p_m2 c1f,c16,c64,L16
nvidia-smi dmon -i 0,1 -s m -d 1 -o T > $D/p_m3_dmon.log 2>&1 &
dmon=$!
for order in a b; do
  for cell in c16 c64 b16 L16; do
    (until [ -e $D/p_m3/$order/bench_gpu0_$cell.log ]; do sleep 1; done; echo "$order $cell $(date +%T)" >> $D/p_m3_cell_starts.txt) &
  done
done
pair t_m t_l2 p_m3 c16,c64,b16,L16
kill $dmon
log "p_m3 cell starts $(sort $D/p_m3_cell_starts.txt | tr '\n' ' ')"
python3 $IN/worst_step_c20261021.py /workspace/t_l4 1 16-31,80-95 46000 $D/worst_l4_16 16 $D/worst/meta.lst > $D/worst_l4_16.txt 2>&1 &
python3 $IN/worst_step_c20261021.py /workspace/t_l4 0 0-15,64-79 46400 $D/worst_l4_32 32 $D/worst32/meta.lst > $D/worst_l4_32.txt 2>&1 &
wait
for rows in 16 32; do
  log "worst step $rows rows: $(grep -E 'W[0-9]+:|c64:' $D/worst_l4_$rows.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-330)"
  grep 'alloc retries' $D/worst_l4_$rows.txt | sed 's/^/    /' >> $D/queue_m.txt
done
log "queue m done"
