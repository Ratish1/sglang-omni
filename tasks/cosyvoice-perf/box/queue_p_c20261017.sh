#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: the kit profile of the priority alone on main (t_mp_ed979c580) for its own PR's body.
# Waits for queue_o; upstream main (t_main2) on card 2 and the priority on card 3 under the kit at the same time, c16
# and c64 of 640 unseeded requests traced, no py-spy; then each cell's per request first audio timeline.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_p.pgid; exec bash queue_p_c20261017.sh'
D=/data/c20261017
IN=$D/in
P=t_mp_ed979c580
log() { echo "$* $(date +%T)" >> $D/queue_p.txt; }

log "queue wait"
until grep -q "queue done" $D/queue_o.txt; do sleep 30; done
log "queue start"
mkdir -p $D/prof_pm
PEER=$D/prof_pm/prio PYSPY=0 CELLS="c16:16:640 c64:64:640" bash $IN/cosy_profile_run_c20261017.sh t_main2 2 0-15,64-79 \
  21000 $D/prof_pm/main > $D/prof_pm/run_main.log 2>&1 &
first=$!
PEER=$D/prof_pm/main PYSPY=0 CELLS="c16:16:640 c64:64:640" bash $IN/cosy_profile_run_c20261017.sh $P 3 16-31,80-95 \
  22000 $D/prof_pm/prio > $D/prof_pm/run_prio.log 2>&1 &
second=$!
wait $first $second
log "profiles $(tail -n 1 $D/prof_pm/main/progress.txt) / $(tail -n 1 $D/prof_pm/prio/progress.txt)"
for cell in c16 c64; do
  python3 $IN/first_audio_timeline_v2_c20261017.py main=$D/prof_pm/main/$cell.sqlite prio=$D/prof_pm/prio/$cell.sqlite \
    > $D/prof_pm/timeline_$cell.txt 2>&1
  log "analysis $cell done"
done
log "queue done"
