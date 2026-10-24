#!/usr/bin/env bash
# 10-24, host-85-234-79-221, cards 0 and 1: L4 (perf/cosyvoice3-prefix-pool-headroom, main merged) against main
# cd191df23 at c16, where its pair p_h4 read inter chunk mean +0.8 ; +1.9 % and p95 +4.3 ; +5.1 %. No L4 code runs per
# step at c16, but two startup changes could leave serving on another compiled entry than main's: the largest step
# measurement (16 rows at the context length through the compiled PackedDiT and HiFT) and the prefix forward compiled
# first on the throwaway pool. Steps: trees; L4's touched units; a warm of each tree under Dynamo's recompile log
# (kept on for every later boot, so serving recompiles show too); the pair p_l4c (c16, c16p, c64) and the A/A pair
# p_aac (main against main, c16, c16p), both orders, unprofiled, with each second's host load; then the nsys probe pass
# of both arms at once, both orders (c16 and c64 traced, py-spy at c16), and the readers.
# Decision rules, written before the runs: a row moves when both orders agree in sign and their mean leaves p_aac's
# band; a state owns the row when its duration moves the same way in both orders of the nsys pass; an L4 only
# recompile at startup or in serving is the mechanism to fix first.
# usage (in the container): mkdir -p /data/c20261024/l4p && setsid bash -c 'echo $$ > /data/c20261024/l4p/queue.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261024/queue_l4p.sh'
D=/data/c20261024
P=$D/l4p
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
MAIN=cd191df237f8c21e631133c6cf349164e6c1fcdf
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
export TORCH_LOGS=recompiles
mkdir -p $P
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
tree() {
  local name=$1 sha=$2
  if [ -d /workspace/$name ]; then
    git -C /workspace/$name checkout -q --detach $sha
  else
    git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
  fi
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-70) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD -- sglang_omni || echo DIRTY)"
}
git -C /workspace/sglang-omni fetch -q upstream main
git -C /workspace/sglang-omni cat-file -e $MAIN^{commit} || { log "main $MAIN missing, stop"; exit 1; }
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-prefix-pool-headroom
L4=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
tree t_mn $MAIN
tree t_mn2 $MAIN
tree t_l4p $L4
log "queue l4p start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"

(cd /workspace/t_l4p && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$P/inductor_units \
  PYTHONPATH=/workspace/t_l4p:$COSYVOICE taskset -c $(cpus 0) \
  timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -rfEs \
  > $P/units.log 2>&1) &
units=$!
bash $IN/warm_c20261021.sh t_l4p 1 $(cpus 1) 29700 $P/warm_t_l4p
wait $units
units_rc=$?
log "units t_l4p rc $units_rc $(grep -E 'passed|failed' $P/units.log | tail -n 1)"
[ $units_rc = 0 ] || { log "units failed, stop"; exit 1; }
bash $IN/warm_c20261021.sh t_mn 0 $(cpus 0) 29400 $P/warm_t_mn
rm -rf $D/inductor_t_mn2
cp -a $D/inductor_t_mn $D/inductor_t_mn2
for name in t_mn t_l4p; do
  log "warm $name $(grep 'bench rc' $P/warm_$name/progress.txt) $(grep -E 'Completed requests|Failed requests' $P/warm_$name/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $P/warm_$name/serve.log) recompiles $(grep -c 'Recompiling function' $P/warm_$name/serve.log)"
  grep -E 'Flow prefix cache|largest admitted step|compile warmup|prefix compile warmup' $P/warm_$name/serve.log | sed 's/^.*\] //' | sed 's/^/    /' >> $P/queue.txt
done

pair() {
  local first=$1 second=$2 name=$3 cells=$4
  python3 $IN/host_load_c20261022.py $P/${name}_load.jsonl &
  local sampler=$!
  (cd /workspace/sglang-omni && PAIR_CELLS=$cells python3 $IN/serve_pair_c20261021.py /workspace/$first \
    /workspace/$second $P/$name > $P/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $P/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
  kill $sampler
  python3 $IN/host_load_c20261022.py --summary $P/${name}_load.jsonl $P/$name $cells > $P/${name}_load.txt 2>&1
  for cell in ${cells//,/ }; do
    python3 $IN/pooled_inter_chunk_c20261021.py $P/$name $cell $first $second > $P/${name}_pooled_$cell.txt 2>&1
    python3 $IN/gap_positions_c20261022.py $P/$name $cell > $P/${name}_positions_$cell.txt 2>&1
  done
  python3 $IN/profile_c20261022/warmup_cohort.py $P/$name $cells > $P/${name}_cohort.txt 2>&1
}
pair t_mn t_l4p p_l4c c16,c16p,c64
pair t_mn t_mn2 p_aac c16,c16p
for name in p_l4c p_aac; do
  for order in a b; do
    log "recompiles in serving $name $order gpu0 $(grep -c 'Recompiling function' $P/$name/$order/serve_gpu0.log) gpu1 $(grep -c 'Recompiling function' $P/$name/$order/serve_gpu1.log)"
  done
done

nsys_order() {
  local order=$1 main_card=$2 l4_card=$3
  CELLS="c16:16:320 c64:64:320" PEER=$P/nsys_${order}_t_l4p \
    bash $IN/cosy_profile_run_c20261021.sh t_mn $main_card $(cpus $main_card) $((47000 + 400 * main_card)) $P/nsys_${order}_t_mn &
  CELLS="c16:16:320 c64:64:320" PEER=$P/nsys_${order}_t_mn \
    bash $IN/cosy_profile_run_c20261021.sh t_l4p $l4_card $(cpus $l4_card) $((47000 + 400 * l4_card)) $P/nsys_${order}_t_l4p &
  wait
  log "nsys $order main card $main_card $(tail -n 1 $P/nsys_${order}_t_mn/progress.txt) l4 card $l4_card $(tail -n 1 $P/nsys_${order}_t_l4p/progress.txt)"
  for cell in c16 c64; do
    python3 $IN/request_states_c20261019.py main=$P/nsys_${order}_t_mn/$cell.sqlite l4=$P/nsys_${order}_t_l4p/$cell.sqlite \
      --skip-head-s 1 --skip-tail-s 1 > $P/states_${order}_$cell.txt 2>&1
    python3 $IN/chunk_timeline_c20261018.py main=$P/nsys_${order}_t_mn/$cell.sqlite l4=$P/nsys_${order}_t_l4p/$cell.sqlite \
      --skip-head-s 1 > $P/chunks_${order}_$cell.txt 2>&1
  done
}
nsys_order a 0 1
nsys_order b 1 0
log "queue l4p done"
