#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: is #2897's unseeded tail rise real (c16 first audio p99 +3.1 / +11.6 % and latency p99
# +3.4 / +1.2 % in p_fi, c64 latency p99 -2.0 / +13.6 %)? Two more same time pairs of main against F (t_fi, 7bd786e22)
# interleaved with two of main against main, unseeded c16 and c64 over the full corpus, both orders each: six F order
# samples (with p_fi) against four main main samples on the same cards and the same day.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_c.pgid; exec bash queue_c_c20261017.sh'
D=/data/c20261017
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_c.txt; }
pair() {
  local first=$1 second=$2 out=$3
  PAIR_CELLS=c16,c64 python3 $IN/serve_pair_c20261017.py /workspace/$first /workspace/$second $D/$out > $D/$out.log 2>&1
  log "$out rc $?"
}

log "queue start"
cd /workspace/t_main
pair t_main t_fi p_fi2
pair t_main t_main aa_u2
pair t_main t_fi p_fi3
pair t_main t_main aa_u3
log "queue done"
