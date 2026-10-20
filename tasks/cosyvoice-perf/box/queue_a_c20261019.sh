#!/usr/bin/env bash
# 10-19, card 2: S0 of DESIGN_VOCODER_STANDARD_20261019.md on upstream main 0c521f191 (tree t_m19). The long form
# list (48 entries of 10 seed-tts texts each), the warm smoke test, then one profile with the probe (HiFT's F0,
# source, sine and decode ranges added): c16 (320 requests), c64 (640), long form c16 (48), py-spy at c16; per cell
# the ledgers, the vocoder selection analysis and the chunk timeline.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261019/queue_a.pgid; exec bash queue_a_c20261019.sh'
D=/data/c20261019
IN=$D/in
M=t_m19
log() { echo "$* $(date +%T)" >> $D/queue_a.txt; }

log "queue start"
(cd /workspace/sglang-omni && PYTHONPATH=/workspace/sglang-omni python3 $IN/make_longform_meta.py \
  --out $D/longform > $D/longform.log 2>&1)
log "long form list rc $? $(tail -n 1 $D/longform.log)"
bash $IN/warm_c20261019.sh $M 2 0-15,64-79 26100 $D/warm_m19
log "warm $(grep 'bench rc' $D/warm_m19/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_m19/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_m19/serve.log)"

mkdir -p $D/prof_s0
PYSPY=1 CELLS="c16:16:320 c64:64:640 L16:16:48:::$D/longform/meta.lst" \
  bash $IN/cosy_profile_run_c20261019.sh $M 2 0-15,64-79 27100 $D/prof_s0/main > $D/prof_s0/run_main.log 2>&1
log "profile $(tail -n 1 $D/prof_s0/main/progress.txt)"
for cell in c16 c64 L16; do
  python3 $IN/chunk_timeline_c20261018.py main=$D/prof_s0/main/$cell.sqlite > $D/prof_s0/chunks_$cell.txt 2>&1
  log "chunk timeline $cell"
done
log "queue done"
