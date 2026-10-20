#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: whether the AR priority is its own PR or part of the HiFT graphs PR. Trees: the graphs
# PR without the priority (t_sr2_8a986ef5b: item 1 425c67687 and the ranking at two recent steps), the priority alone
# on main (t_mp_ed979c580: 6bdd47967 with the scheduler option and CosyVoice's opt in), the stack (t_pr3_0167dde34,
# warm). Card 3 runs the priority tree's unit suites while card 2 warms the graphs tree, then card 3 warms the priority
# tree. Pairs, both orders: main against the graphs PR alone (p_g2, c16 and c64: does it pass the gate alone), main
# against the priority alone (p_pm, seeded c1 identity, unseeded c1, c16, c64: does the priority stand on main), the
# graphs PR against the stack (p_prio2, c16 and c64: what the priority adds on top with the two step bound).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_m.pgid; exec bash queue_m_c20261017.sh'
D=/data/c20261017
IN=$D/in
G=t_sr2_8a986ef5b
P=t_mp_ed979c580
S=t_pr3_0167dde34
log() { echo "$* $(date +%T)" >> $D/queue_m.txt; }
smoke_ok() { grep -q "bench rc 0" $D/$1/progress.txt && ! grep -q Traceback $D/$1/serve.log; }
pair() {
  local first=$1 second=$2 out=$3 cells=$4
  PAIR_CELLS=$cells python3 $IN/serve_pair_c20261017.py /workspace/$first /workspace/$second $D/$out > $D/$out.log 2>&1
  log "$out rc $?"
}

log "queue start"
bash $IN/warm_c20261017.sh $G 2 0-15,64-79 21200 $D/smoke_sr2b &
WARM=$!
mkdir -p $D/units/$P
(cd /workspace/$P && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py \
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3 > $D/units/$P/units.log 2>&1)
log "units $P rc $? $(grep -E 'passed|failed' $D/units/$P/units.log | tail -n 1)"
bash $IN/warm_c20261017.sh $P 3 16-31,80-95 22200 $D/smoke_mp
log "smoke mp $(grep 'bench rc' $D/smoke_mp/progress.txt) tracebacks $(grep -c Traceback $D/smoke_mp/serve.log)"
wait $WARM
log "smoke sr2b $(grep 'bench rc' $D/smoke_sr2b/progress.txt) tracebacks $(grep -c Traceback $D/smoke_sr2b/serve.log)"
cd /workspace/t_main2
if smoke_ok smoke_sr2b; then
  pair t_main2 $G p_g2 c16,c64
else
  log "p_g2 skipped"
fi
if smoke_ok smoke_mp; then
  pair t_main2 $P p_pm c1,c1u,c16,c64
  python3 $IN/c1_identity_c20261017.py $D/p_pm c1 > $D/p_pm_identity.txt 2>&1
  log "p_pm identity $(tr '\n' ' ' < $D/p_pm_identity.txt)"
else
  log "p_pm skipped"
fi
if smoke_ok smoke_sr2b; then
  pair $G $S p_prio2 c16,c64
else
  log "p_prio2 skipped"
fi
log "queue done"
