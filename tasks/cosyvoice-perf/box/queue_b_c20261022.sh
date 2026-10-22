#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1: bisect L3's static c16 gap cost (p_c1, every reference cached: L3c's gaps
# p95 +4.1 ; +4.3 % against L2c with no tokenizer call while measured). Two variants of t_l3c, each measured against
# t_l2c with the cache primed (c16p) then c16, both orders: t_l3f (tmp/cosyvoice3-l3-bigcache-fp32: the graphs
# captured without the TF32 toggle) and t_l3e (tmp/cosyvoice3-l3-bigcache-eager: no graphs, the model eager). Gaps pooled.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_b.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_b_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_b.txt; }
tree() {
  local name=$1 branch=$2 sha
  git -C /workspace/sglang-omni fetch -q origin $branch
  sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
  if [ -d /workspace/$name ]; then
    git -C /workspace/$name checkout -q --detach $sha
  else
    git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
  fi
  [ -d $D/inductor_$name ] || cp -r $D/inductor_t_l3c $D/inductor_$name
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8)"
}
pair() {
  local first=$1 second=$2 name=$3
  (cd /workspace/sglang-omni && PAIR_CELLS=c16p,c16 python3 $IN/serve_pair_c20261021.py /workspace/$first /workspace/$second \
    $D/$name > $D/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
  python3 $IN/pooled_inter_chunk_c20261021.py $D/$name c16 $first $second > $D/pooled_${name}_c16.txt 2>&1
  log "pooled $name c16 $(grep '^pooled' $D/pooled_${name}_c16.txt | tr '\n' ' ')"
}
log "queue b start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
tree t_l3f tmp/cosyvoice3-l3-bigcache-fp32
tree t_l3e tmp/cosyvoice3-l3-bigcache-eager
pair t_l2c t_l3f p_v1
pair t_l2c t_l3e p_v2
log "queue b done"
