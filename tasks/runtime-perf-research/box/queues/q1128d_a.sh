#!/usr/bin/env bash
# 11-28 second lease, node-0001, 2 H200s: #3452 (FIX 91c2e2010) against main (MAIN 0f70d9c7c) on the Qwen3-Omni modes
# with no reading yet.
#   1. Fused text pipeline (preprocessing in the thinker's process, so the lane's threads share its GIL): text Video-MME
#      c16 x3 and c1 x1, same-time pairs on the two cards, swapped.
#   2. The CI speech pipeline on one card at c1 (a lone video at a time): text Video-MME c1 x2, pairs swapped.
#   3. The CI's own Video-MME server (thinker on GPU 0, talker on GPU 1, both cards per server): fresh boots alternating
#      M F F M M F F M, one text Video-MME c16 pass and one talker test pass each.
# Every pass records the GPU memory of each process after it (gpu_after.txt) for the encoder budget question.
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main F=/workspace/fix
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((20000 + 4000 * $1 + $2)); }
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
until grep -q "setup done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1128d_a start"
fused() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "fused text pipeline, text Video-MME c16 x3 and c1, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_text_fused --card $card \
    --port $(port_of $card $offset) --out $O/$name --serve-extra "--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768" \
    --bench videomme_text --loads 16,1 --passes clean --bench-repeats 3 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
lone() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "CI speech pipeline on one card, text Video-MME c1 x2, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card $card \
    --port $(port_of $card $offset) --out $O/$name --bench videomme_text --loads 1 --passes clean --bench-repeats 2 \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
both "fused tf1_m $M 0 100" "fused tf1_f $F 1 100"
both "fused tf2_f $F 0 300" "fused tf2_m $M 1 300"
for part in c16_clean c1_clean; do
  python3 $K/videomme_answer_diff.py $O main=tf1_m/$part,tf2_m/$part fix=tf1_f/$part,tf2_f/$part > $O/fused_$part.txt 2>&1
done
log "fused read"
both "lone lc1_m $M 0 500" "lone lc1_f $F 1 500"
both "lone lc2_f $F 0 700" "lone lc2_m $M 1 700"
python3 $K/videomme_answer_diff.py $O main=lc1_m/c1_clean,lc2_m/c1_clean fix=lc1_f/c1_clean,lc2_f/c1_clean > $O/lone_c1.txt 2>&1
log "lone read"
disagg() {
  local name=$1 tree=$2 port=$3
  note $name $tree "CI Video-MME server, thinker GPU 0, talker GPU 1, fresh, text c16 once, talker once"
  python3 $K/baseline.py --tree $tree --model omni_disagg_ci --port $port --out $O/$name --bench videomme_text \
    --loads 16 --passes clean --after "python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 1" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(grep -o '\"accuracy\": [0-9.]*' $O/$name/talker/summary.jsonl 2>/dev/null | head -1)"
}
index=0
for arm in m f f m m f f m; do
  index=$((index + 1))
  if [ $arm = m ]; then tree=$M; else tree=$F; fi
  disagg dg${index}_$arm $tree $((22000 + 200 * index))
done
mains=dg1_m/c16_clean,dg4_m/c16_clean,dg5_m/c16_clean,dg8_m/c16_clean
fixes=dg2_f/c16_clean,dg3_f/c16_clean,dg6_f/c16_clean,dg7_f/c16_clean
python3 $K/videomme_answer_diff.py $O main=$mains fix=$fixes > $O/disagg_text.txt 2>&1
python3 $K/videomme_answer_diff.py $O main=${mains//c16_clean/talker} fix=${fixes//c16_clean/talker} > $O/disagg_talker.txt 2>&1
log "disagg read"
log "q1128d_a done"
