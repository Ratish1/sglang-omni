#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: the stack item 1 (HiFT's decode from graphs, shared static inputs) + the ranking + the AR
# loop on the device's top priority stream (t_pr_d520ee951). Card 3 runs its unit suites, then warms the ranking only
# tree (t_sr_dcc7560be, the stack less the priority); card 2 warms the stack (its smoke test). Then the pairs, seeded
# c1, unseeded c1, c16, c64, both orders: upstream main (t_main2) against the stack, the gate; the ranking only tree
# against the stack, the priority alone. No scoring: neither the ranking nor the priority changes what the decode
# computes (item 1's WER and SIM in p_hg stand).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_i.pgid; exec bash queue_i_c20261017.sh'
D=/data/c20261017
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_i.txt; }
pair() {
  local first=$1 second=$2 out=$3
  PAIR_CELLS=c1,c1u,c16,c64 python3 $IN/serve_pair_c20261017.py /workspace/$first /workspace/$second $D/$out \
    > $D/$out.log 2>&1
  log "$out rc $?"
  python3 $IN/c1_identity_c20261017.py $D/$out c1 > $D/${out}_identity.txt 2>&1
  log "$out identity $(tr '\n' ' ' < $D/${out}_identity.txt)"
}

log "queue start"
bash $IN/warm_c20261017.sh t_pr_d520ee951 2 0-15,64-79 21000 $D/smoke_pr &
WARM=$!
mkdir -p $D/units/t_pr_d520ee951
(cd /workspace/t_pr_d520ee951 && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/scheduling/test_engine_factory.py \
  tests/unit_test/fun_cosyvoice3 > $D/units/t_pr_d520ee951/units.log 2>&1)
log "units t_pr rc $? $(tail -n 1 $D/units/t_pr_d520ee951/units.log)"
bash $IN/warm_c20261017.sh t_sr_dcc7560be 3 16-31,80-95 22000 $D/smoke_sr2
log "smoke sr2 $(grep 'bench rc' $D/smoke_sr2/progress.txt) tracebacks $(grep -c Traceback $D/smoke_sr2/serve.log)"
wait $WARM
log "smoke pr $(grep 'bench rc' $D/smoke_pr/progress.txt) tracebacks $(grep -c Traceback $D/smoke_pr/serve.log)"
cd /workspace/t_main2
if grep -q "bench rc 0" $D/smoke_pr/progress.txt && ! grep -q Traceback $D/smoke_pr/serve.log; then
  pair t_main2 t_pr_d520ee951 p_pr
  pair t_sr_dcc7560be t_pr_d520ee951 p_prio_only
else
  log "pairs skipped"
fi
log "queue done"
