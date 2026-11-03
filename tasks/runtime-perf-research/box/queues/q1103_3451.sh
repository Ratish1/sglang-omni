#!/usr/bin/env bash
# 11-03, node-0000, one H200 (one colocated server fits a card, so no same-time pairs): #3451's first audio against its
# base debd337b4, alternating boots m p p m m p p m, which cancels drift and gives main-against-main and 3p-against-3p gaps
# from the same run. Block a is the h200 config, unseeded, as the pairs in the PR body. Block b reproduces the review's
# setup, the h100_bf16 config on the H200 (thinker prefill graph breakable, max bs 2048), seed 3451, warmup 16, where it
# read first audio +7.6 ms and p99 +17.5 ms. Every boot: c16 streamed, warm, two clean passes, events. The kit's boots
# stop their servers themselves; each boot starts only with the card free, other tenants share this node.
set -u
O=/data/rp1103
K=/workspace/kit_q/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 P3=/workspace/p3_1f04aa0e6
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
wait_card() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do
    log "waiting, card used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 30
  done
}
boot() {
  local name=$1 tree=$2 model=$3; shift 3
  wait_card
  log "boot $name $model $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) load $(cut -d' ' -f1 /proc/loadavg) start"
  python3 $K/baseline.py --tree $tree --model $model --card 0 --port 26000 --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 "$@" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
read_block() {
  local tag=$1 main=$2 branch=$3
  cd $O
  {
    echo "== $tag both arms"; taskset -c 0-55 python3 $K/first_audio_states.py --cell c16_events --main $main --branch $branch
    echo "== $tag table"; taskset -c 0-55 python3 $K/pair_table.py . $main $branch c16
    echo "== $tag paired"; taskset -c 0-55 python3 $K/latency_paired.py . c16 $main $branch
  } > $O/read_$tag.txt 2>&1
  log "read $tag"
}
log "q1103_3451 start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot a_m1 $MAIN omni_colocated
boot a_p1 $P3 omni_colocated
boot a_p2 $P3 omni_colocated
boot a_m2 $MAIN omni_colocated
boot a_m3 $MAIN omni_colocated
boot a_p3 $P3 omni_colocated
boot a_p4 $P3 omni_colocated
boot a_m4 $MAIN omni_colocated
read_block a a_m1,a_m2,a_m3,a_m4 a_p1,a_p2,a_p3,a_p4
R="--bench-extra=--seed 3451 --warmup 16"
boot b_m1 $MAIN omni_colocated_h100_config "$R"
boot b_p1 $P3 omni_colocated_h100_config "$R"
boot b_p2 $P3 omni_colocated_h100_config "$R"
boot b_m2 $MAIN omni_colocated_h100_config "$R"
boot b_m3 $MAIN omni_colocated_h100_config "$R"
boot b_p3 $P3 omni_colocated_h100_config "$R"
boot b_p4 $P3 omni_colocated_h100_config "$R"
boot b_m4 $MAIN omni_colocated_h100_config "$R"
read_block b b_m1,b_m2,b_m3,b_m4 b_p1,b_p2,b_p3,b_p4
log "q1103_3451 done"
