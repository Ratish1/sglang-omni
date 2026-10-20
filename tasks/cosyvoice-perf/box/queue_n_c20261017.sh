#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: the rest of the one PR or several question. The ranking alone on main (t_rk_80a6d5857:
# 6bdd47967 with the ranking at two recent steps, cherry picked clean). Waits for queue_m; card 3 runs its unit suite,
# then warms it. Pairs, both orders: main against the ranking alone (p_rk, c16 and c64; c1 has no first hop beside a
# final): does it stand on main; the priority alone on main against the stack (p_mpg, c1u, c16, c64): the graphs PR
# measured on the base it lands on if the priority merges first.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_n.pgid; exec bash queue_n_c20261017.sh'
D=/data/c20261017
IN=$D/in
R=t_rk_80a6d5857
P=t_mp_ed979c580
S=t_pr3_0167dde34
log() { echo "$* $(date +%T)" >> $D/queue_n.txt; }
pair() {
  local first=$1 second=$2 out=$3 cells=$4
  PAIR_CELLS=$cells python3 $IN/serve_pair_c20261017.py /workspace/$first /workspace/$second $D/$out > $D/$out.log 2>&1
  log "$out rc $?"
}

log "queue wait"
until grep -q "queue done" $D/queue_m.txt; do sleep 30; done
log "queue start"
mkdir -p $D/units/$R
(cd /workspace/$R && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/fun_cosyvoice3 > $D/units/$R/units.log 2>&1)
log "units $R rc $? $(grep -E 'passed|failed' $D/units/$R/units.log | tail -n 1)"
bash $IN/warm_c20261017.sh $R 3 16-31,80-95 22300 $D/smoke_rk
log "smoke rk $(grep 'bench rc' $D/smoke_rk/progress.txt) tracebacks $(grep -c Traceback $D/smoke_rk/serve.log)"
cd /workspace/t_main2
if grep -q "bench rc 0" $D/smoke_rk/progress.txt && ! grep -q Traceback $D/smoke_rk/serve.log; then
  pair t_main2 $R p_rk c16,c64
else
  log "p_rk skipped"
fi
pair $P $S p_mpg c1u,c16,c64
log "queue done"
