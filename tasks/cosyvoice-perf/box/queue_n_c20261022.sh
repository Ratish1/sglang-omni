#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1, after queue m: does L4 need L3? L4 sizes the Flow prefix pool from the free
# memory left after startup, less its measured reserve; without L3 the speech tokenizer is ONNX Runtime's CUDA session,
# whose arena grows outside torch while serving (4.0 GiB on 10-08). t_l24 (tmp/cosyvoice3-l2-l4: main + L2 + L4, no
# L3) against t_l4 (main + L2 + L3 + L4): the worst step check at 16 rows on each (W16 then c64, the memory probe:
# peak use, memory outside torch, allocation retries and failures), then a pair at c64 and L16 with every card's
# framebuffer sampled each second.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_n.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_n_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $D/queue_n.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
until grep -q "queue m done" $D/queue_m.txt; do sleep 30; done
log "queue n start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
git -C /workspace/sglang-omni fetch -q origin tmp/cosyvoice3-l2-l4
sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
[ -d /workspace/t_l24 ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_l24 $sha
git -C /workspace/t_l24 checkout -q --detach $sha
[ -d $D/inductor_t_l24 ] || cp -r $D/inductor_t_l4 $D/inductor_t_l24
log "tree t_l24 $(git -C /workspace/t_l24 log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/t_l24 rev-parse HEAD:sglang_omni | cut -c1-8)"
python3 $IN/worst_step_c20261021.py /workspace/t_l24 0 $(cpus 0) 46000 $D/worst_l24_16 16 $D/worst/meta.lst > $D/worst_l24_16.txt 2>&1 &
python3 $IN/worst_step_c20261021.py /workspace/t_l4 1 $(cpus 1) 46400 $D/worst_l4b_16 16 $D/worst/meta.lst > $D/worst_l4b_16.txt 2>&1 &
wait
for arm in l24 l4b; do
  log "worst step $arm 16 rows: $(grep -E 'W[0-9]+:|c64:' $D/worst_${arm}_16.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-330)"
  grep -E 'alloc retries|outside torch|peak' $D/worst_${arm}_16.txt | sed 's/^/    /' >> $D/queue_n.txt
done
nvidia-smi dmon -i 0,1 -s m -d 1 -o T > $D/p_n1_dmon.log 2>&1 &
dmon=$!
(cd /workspace/sglang-omni && PAIR_CELLS=c64,L16 python3 $IN/serve_pair_c20261021.py /workspace/t_l4 /workspace/t_l24 $D/p_n1 \
  > $D/p_n1.log 2>&1)
log "pair p_n1 rc $? token cap hits $(grep -E '\| at the token cap \|' $D/p_n1.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
kill $dmon
for order in a b; do
  for gpu in 0 1; do
    log "p_n1 $order gpu$gpu errors $(grep -c -i -E 'out of memory|Failed to allocate|CUDA error' $D/p_n1/$order/serve_gpu$gpu.log)"
  done
done
log "queue n done"
