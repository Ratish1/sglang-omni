#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: why item 1 with the ranking (t_sr_1441d0b60) costs c64 req/s (-2.4 / -4.2 %) and c16
# latency p99 (+5 %) against upstream main (t_main2). Both under the kit at the same time, c16 and c64 of 640
# unseeded requests traced, no py-spy; then each cell's per request first audio timeline and card busy span.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_h.pgid; exec bash queue_h_c20261017.sh'
D=/data/c20261017
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_h.txt; }

log "queue start"
mkdir -p $D/prof_2b
PEER=$D/prof_2b/fi PYSPY=0 CELLS="c16:16:640 c64:64:640" bash $IN/cosy_profile_run_c20261017.sh t_main2 2 0-15,64-79 \
  21000 $D/prof_2b/main > $D/prof_2b/run_main.log 2>&1 &
first=$!
PEER=$D/prof_2b/main PYSPY=0 CELLS="c16:16:640 c64:64:640" bash $IN/cosy_profile_run_c20261017.sh t_sr_1441d0b60 3 \
  16-31,80-95 22000 $D/prof_2b/fi > $D/prof_2b/run_fi.log 2>&1 &
second=$!
wait $first $second
log "profiles $(tail -n 1 $D/prof_2b/main/progress.txt) / $(tail -n 1 $D/prof_2b/fi/progress.txt)"
for cell in c16 c64; do
  python3 $IN/first_audio_timeline_v2_c20261017.py main=$D/prof_2b/main/$cell.sqlite item=$D/prof_2b/fi/$cell.sqlite \
    > $D/prof_2b/timeline_$cell.txt 2>&1
  python3 $IN/card_busy_span_c20261017.py main=$D/prof_2b/main/$cell.sqlite item=$D/prof_2b/fi/$cell.sqlite \
    > $D/prof_2b/card_busy_$cell.txt 2>&1
  log "analysis $cell done"
done
log "queue done"
