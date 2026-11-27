#!/usr/bin/env bash
# 11-27, node-0000, after q1127_b: the CI's own condition for the Video-MME gates, one pass on a fresh server. A boot's
# first pass prefills every video prompt, and every later pass of that boot reads those prompts from the prefix cache
# (1 new token over about 14,200 cached), so the 12 passes per arm of q1127_b hold two prefill draws each. Here each boot
# runs text Video-MME c16 once (after the launcher's one warmup request) and the talker test once, 10 boots per arm,
# M F F M M F F M ..., FIX 72be1f80e against MAIN 0f70d9c7c on the CI speech pipeline.
set -u
O=/data/v1127
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main FIX=/workspace/fix
CPUS=56-111,168-223
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
tree_of() { case $1 in m) echo $MAIN ;; f) echo $FIX ;; *) exit 2 ;; esac; }
until grep -q "q1127_b done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1127_c start"
fresh() {
  local name=$1 tree=$2 port=$3
  note $name $tree "CI speech pipeline, fresh server, text Video-MME c16 once, talker test once"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card 0 --port $port --out $O/$name \
    --bench videomme_text --loads 16 --passes clean \
    --after "python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 1" > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(grep -o '\"accuracy\": [0-9.]*' $O/$name/talker/summary.jsonl 2>/dev/null | head -1)"
}
index=0
for arm in m f f m m f f m m f f m m f f m m f f m; do
  index=$((index + 1))
  fresh fr${index}_$arm $(tree_of $arm) $((21000 + 200 * (index % 10)))
done
mains=$(ls -d $O/fr*_m | xargs -n1 basename | sed 's|$|/c16_clean|' | paste -sd,)
fixes=$(ls -d $O/fr*_f | xargs -n1 basename | sed 's|$|/c16_clean|' | paste -sd,)
python3 $K/videomme_answer_diff.py $O main=$mains fix=$fixes > $O/gate_text_fresh.txt 2>&1
mains=$(ls -d $O/fr*_m | xargs -n1 basename | sed 's|$|/talker|' | paste -sd,)
fixes=$(ls -d $O/fr*_f | xargs -n1 basename | sed 's|$|/talker|' | paste -sd,)
python3 $K/videomme_answer_diff.py $O main=$mains fix=$fixes > $O/gate_talker_fresh.txt 2>&1
log "fresh gates read"
log "q1127_c done"
