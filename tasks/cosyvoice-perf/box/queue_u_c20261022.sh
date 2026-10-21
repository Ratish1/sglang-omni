#!/usr/bin/env bash
# 10-22, the 0.5.22 container, cards 0 and 1, after queue t: p_r6, t_l1 (#3019) against t_l2 (L2) through c16,
# c64, b16 and long form L16, both orders, with every card's framebuffer sampled each second and each cell's bench
# start kept: the card memory a varied run grows on each allocator, and c64 again after c16 (p_r2 read L2's c64 gaps
# +1.7 % pooled, p_r5's cold c64 -1.0 ; -1.3 %).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022r/queue_u.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_u_c20261022.sh'
D=/data/c20261022r
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_r.txt; }
until grep -q "queue t done" $D/queue_r.txt; do sleep 30; done
log "queue u start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
[ -e $D/longform ] || ln -s /data/c20261021/longform $D/longform
nvidia-smi dmon -i 0,1 -s m -d 1 -o T > $D/p_r6_dmon.log 2>&1 &
dmon=$!
for order in a b; do
  for cell in c16 c64 b16 L16; do
    (until [ -e $D/p_r6/$order/bench_gpu0_$cell.log ]; do sleep 1; done; echo "$order $cell $(date +%T)" >> $D/p_r6_cell_starts.txt) &
  done
done
(cd /workspace/sglang-omni && PAIR_CELLS=c16,c64,b16,L16 python3 $IN/serve_pair_c20261021.py /workspace/t_l1 /workspace/t_l2 \
  $D/p_r6 > $D/p_r6.log 2>&1)
log "pair p_r6 rc $? token cap hits $(grep -E '\| at the token cap \|' $D/p_r6.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
kill $dmon
for cell in c16 c64 L16; do
  python3 $IN/pooled_inter_chunk_c20261021.py $D/p_r6 $cell t_l1 t_l2 > $D/pooled_p_r6_$cell.txt 2>&1
  log "pooled p_r6 $cell $(grep '^pooled' $D/pooled_p_r6_$cell.txt | tr '\n' ' ')"
done
log "queue u done cell starts $(sort $D/p_r6_cell_starts.txt | tr '\n' ' ')"
