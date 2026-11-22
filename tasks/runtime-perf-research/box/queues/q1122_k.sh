#!/usr/bin/env bash
# 11-22, node-0001, card 1 after q1122_i's scoring. A second block for part 2's final turn order at c16, where part 2
# regressed: OCVG7 085064564 against BVG 6aff09f46, speech alone c1 and c16, B A A B on card 1 (q1122_j runs the same on
# card 0).
set -u
O=/data/v1122
K=/workspace/kit/tasks/runtime-perf-research/box
BVG=/workspace/bvg OCVG7=/workspace/ocvg7
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS1=56-111,168-223
PORT=31290
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
card1_used_mib() { nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
alone() {
  local name=$1 tree=$2 loads=$3
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
  note $name $tree "seedtts loads $loads alone card 1 port $PORT"
  taskset -c $CPUS1 python3 $K/baseline.py --tree $tree --model omni_colocated --card 1 --port $PORT --out $O/$name \
    --loads $loads --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "^I5 read" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py .*--card 1 " > /dev/null && [ "$(card1_used_mib)" -lt 1024 ]; do sleep 10; done
until [ -d $OCVG7 ]; do sleep 10; done
log "q1122_k start, $(node_free)"
alone ks1 $OCVG7 1,16
alone ks2 $BVG 1,16
alone ks3 $BVG 1,16
alone ks4 $OCVG7 1,16
for cell in c1 c16; do
  python3 $K/body_table.py $O $cell "part 2 final turn order (085064564) against its base, card 1 B A A B, speech alone $cell" \
    two=ks2,ks3 one=ks1,ks4 >> $O/queue.log 2>&1
done
python3 $K/first_audio_states.py --cell c16_events --main $O/ks2,$O/ks3 --branch $O/ks1,$O/ks4 > $O/k_states_c16.txt 2>&1
grep -E "server first audio|frames until|thinker prefill|thinker build|hop to the thinker|talker build" $O/k_states_c16.txt >> $O/queue.log
log "K read"
log "q1122_k done"
