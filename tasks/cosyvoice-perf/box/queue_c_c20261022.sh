#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1: does L3's small c16 gap cost (p_r3, p_y2, p_y3: mean per order +2.2 ;
# +2.8, -0.4 ; -1.4, +1.2 ; +1.5 %) come from the tokenizer's calls while serving, or from something L3 changes
# statically? Both arms carry a measurement only patch holding every seed-tts reference in the reference cache
# (4,096 items, 1 GiB in place of 256 and 128 MiB): t_l2c (tmp/cosyvoice3-l2-bigcache) and t_l3c
# (tmp/cosyvoice3-l3-bigcache). A c16 pass primes the cache (c16p), then the measured c16 pass serves every reference
# from it, so neither arm tokenizes while measured; both orders, gaps pooled, the cache's hit counts read back.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_c.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_c_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_c.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
tree() {
  local name=$1 branch=$2 sha
  git -C /workspace/sglang-omni fetch -q origin $branch
  sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
  if [ -d /workspace/$name ]; then
    git -C /workspace/$name checkout -q --detach $sha
  else
    git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
  fi
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8)"
}
log "queue c start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
tree t_l2c tmp/cosyvoice3-l2-bigcache
tree t_l3c tmp/cosyvoice3-l3-bigcache
# note (ratish): each tree's own inductor cache, copied from its arm without the patch, which compiles the same code
for pair in t_l2c:t_l2 t_l3c:t_l3; do
  [ -d $D/inductor_${pair%%:*} ] || cp -r $D/inductor_${pair#*:} $D/inductor_${pair%%:*}
done
(cd /workspace/sglang-omni && PAIR_CELLS=c16p,c16 python3 $IN/serve_pair_c20261021.py /workspace/t_l2c /workspace/t_l3c $D/p_c1 \
  > $D/p_c1.log 2>&1)
log "pair p_c1 rc $? token cap hits $(grep -E '\| at the token cap \|' $D/p_c1.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
for cell in c16p c16; do
  python3 $IN/pooled_inter_chunk_c20261021.py $D/p_c1 $cell t_l2c t_l3c > $D/pooled_p_c1_$cell.txt 2>&1
  log "pooled p_c1 $cell $(grep '^pooled' $D/pooled_p_c1_$cell.txt | tr '\n' ' ')"
done
for order in a b; do
  for gpu in 0 1; do
    log "cache $order gpu$gpu $(grep -E 'Fun-CosyVoice3.*(hits|misses)' $D/p_c1/$order/serve_gpu$gpu.log | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  done
done
log "queue c done"
