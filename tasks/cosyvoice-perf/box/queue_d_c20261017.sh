#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: why #2897 raises the c16 first audio tail. Main (t_main) and F (t_fi) profiled at the
# same time with the kit (nsys with the probe and the OS runtime trace, c16 of 640 unseeded requests, then py-spy at
# c16), once per card order; then each pair's per request first audio timeline (first_audio_timeline_c20261017.py)
# next to the kit's ledgers.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_d.pgid; exec bash queue_d_c20261017.sh'
D=/data/c20261017
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_d.txt; }
profile_pair() {
  local out=$1 tree2=$2 tree3=$3 name2=$4 name3=$5
  mkdir -p $D/$out
  PEER=$D/$out/$name3 CELLS=c16:16:640 bash $IN/cosy_profile_run_c20261017.sh $tree2 2 0-15,64-79 21000 \
    $D/$out/$name2 > $D/$out/run_$name2.log 2>&1 &
  local first=$!
  PEER=$D/$out/$name2 CELLS=c16:16:640 bash $IN/cosy_profile_run_c20261017.sh $tree3 3 16-31,80-95 22000 \
    $D/$out/$name3 > $D/$out/run_$name3.log 2>&1 &
  local second=$!
  wait $first $second
  log "$out profiles $(tail -1 $D/$out/$name2/progress.txt) / $(tail -1 $D/$out/$name3/progress.txt)"
  python3 $IN/first_audio_timeline_c20261017.py main=$D/$out/main/c16.sqlite fi=$D/$out/fi/c16.sqlite \
    > $D/$out/timeline.txt 2>&1
  log "$out timeline rc $?"
}

log "queue start"
profile_pair prof_a t_main t_fi main fi
profile_pair prof_b t_fi t_main fi main
log "queue done"
