#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: A3 (the base runner's seeds uploaded pinned and non blocking, t_a3 at f2e73527e on
# upstream a0366052d) against main (t_main). Card 3 runs t_a3's unit suites and then its serving smoke test while card 2
# profiles the seed install of both trees; the smoke test gates the pair: seeded c1 and seeded c16 over the full corpus,
# both orders; then the seeded audio identity counts of both cells.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_b.pgid; exec bash queue_b_c20261017.sh'
D=/data/c20261017
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_b.txt; }

log "queue start"
mkdir -p $D/units/t_a3
(cd /workspace/t_a3 && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/model_runner tests/unit_test/fun_cosyvoice3 tests/unit_test/sampling \
  > $D/units/t_a3/units.log 2>&1)
log "units t_a3 rc $? $(tail -1 $D/units/t_a3/units.log)"
for tree in t_main t_a3; do
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$tree taskset -c 0-15,64-79 \
    python3 $IN/seed_upload_profile_c20261017.py $tree >> $D/seed_upload_profile.txt 2> $D/seed_upload_profile_$tree.err)
  log "seed profile $tree rc $?"
done
bash $IN/warm_c20261017.sh t_a3 3 16-31,80-95 22000 $D/smoke_a3
log "smoke a3 $(grep 'bench rc' $D/smoke_a3/progress.txt) tracebacks $(grep -c Traceback $D/smoke_a3/serve.log)"

cd /workspace/t_main
if grep -q "bench rc 0" $D/smoke_a3/progress.txt && ! grep -q Traceback $D/smoke_a3/serve.log; then
  PAIR_CELLS=c1s,c16s python3 $IN/serve_pair_c20261017.py /workspace/t_main /workspace/t_a3 $D/p_a3 > $D/p_a3.log 2>&1
  log "p_a3 rc $?"
  for cell in c1s c16s; do
    python3 $IN/c1_identity_c20261017.py $D/p_a3 $cell >> $D/p_a3_identity.txt 2>&1
  done
  log "identity $(tr '\n' ' ' < $D/p_a3_identity.txt)"
else
  log "p_a3 skipped"
fi
log "queue done"
