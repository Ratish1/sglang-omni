#!/usr/bin/env bash
# 10-25, cards 0 and 1: #3129's fixes for its rereview (tasks/pr3129_rereview_20261024): ragged attention decided
# before the vocoder loads, so a CPU vocoder with the default prefix cache never reads the accelerator's memory (R1),
# and the reserve test restoring the allocator fraction it found (R2). The head comes from a bundle at $P/l4t.bundle.
# Steps: units of the touched suite on card 0 beside main's warm boot on card 1; the new CPU test against the reviewed
# head adc65b396 (t_l4o) and the reserve test from an entry fraction of 0.75 on card 0; the head's warm boot on card 0;
# the pair p_l4t (main against the head, c16, both orders).
# Decision rules, written before the run: units pass (288, the new case included); the new case fails on t_l4o with
# the accelerator mismatch; the reserve test leaves the fraction at 0.75; the head's boot serves 64 of 64 with the
# startup lines of 10-24's t_l4r boot (step 7.40 GB streaming, 6.60 GB non-streaming, prefix graphs 0.26 GB, runtime
# headroom 11.79 GB, pool near 56,896 frames), since CUDA sizing is unchanged; p_l4t's c16 rows stay inside the
# reviewed head's c16 pair (req/s +1.1 ; +0.6 %, inter chunk p95 -2.0 ; -0.4 %, latency p99 +1.3 ; -1.6 %) plus the
# A/A spread (req/s -1.4 to +1.7 %, first audio p99 up to 4.1 %), with no row worse in both orders past that spread.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261025/l4t/queue.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261025/queue_l4t.sh'
D=/data/c20261025
P=$D/l4t
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
NEW_TEST=tests/unit_test/fun_cosyvoice3/test_vocoder.py
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
mkdir -p $P
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
for tree in t_mn t_l4t t_l4o; do
  log "tree $tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-70) sglang_omni $(git -C /workspace/$tree rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done
(cd /workspace/t_l4t && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$P/inductor_units \
  PYTHONPATH=/workspace/t_l4t:$COSYVOICE taskset -c $(cpus 0) \
  timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -rfEs \
  > $P/units.log 2>&1) &
units=$!
bash $IN/warm_c20261021.sh t_mn 1 $(cpus 1) 29700 $P/warm_t_mn &
warm_main=$!
wait $units
units_rc=$?
log "units t_l4t rc $units_rc $(grep -E 'passed|failed' $P/units.log | tail -n 1)"
[ $units_rc = 0 ] || { log "units failed, stop"; wait $warm_main; exit 1; }
cp /workspace/t_l4t/$NEW_TEST /workspace/t_l4o/$NEW_TEST
(cd /workspace/t_l4o && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_l4o:$COSYVOICE taskset -c $(cpus 0) \
  timeout 600 python3 -m pytest "$NEW_TEST::test_a_cpu_vocoder_serves_without_the_default_prefix_cache" -q \
  -p no:cacheprovider > $P/prefix_t_l4o.log 2>&1)
log "new case on t_l4o rc $? $(grep -m1 -oE "ValueError: .*" $P/prefix_t_l4o.log) $(grep -E 'passed|failed' $P/prefix_t_l4o.log | tail -n 1)"
git -C /workspace/t_l4o checkout -q -- $NEW_TEST
(cd /workspace/t_l4t && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_l4t:$COSYVOICE taskset -c $(cpus 0) \
  timeout 600 python3 -c "
import torch
from tests.unit_test.fun_cosyvoice3.test_prefix_cache_gpu import test_prefix_pool_leaves_its_reserve_under_the_allocator_cap as case
device = torch.device('cuda', 0)
torch.cuda.set_per_process_memory_fraction(0.75, device)
case()
print('entry 0.75 after', torch.cuda.get_per_process_memory_fraction(device))
" > $P/fraction.log 2>&1)
log "reserve test rc $? $(tail -n 1 $P/fraction.log)"
bash $IN/warm_c20261021.sh t_l4t 0 $(cpus 0) 29800 $P/warm_t_l4t
wait $warm_main
for tree in t_mn t_l4t; do
  log "warm $tree $(grep 'bench rc' $P/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $P/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $P/warm_$tree/serve.log)"
  grep -E 'Flow prefix cache|largest admitted step' $P/warm_$tree/serve.log | sed 's/^.*\] //' | sed 's/^/    /' >> $P/queue.txt
done
(cd /workspace/sglang-omni && PAIR_CELLS=c16 python3 $IN/serve_pair_c20261021.py /workspace/t_mn \
  /workspace/t_l4t $P/p_l4t > $P/p_l4t.log 2>&1)
log "pair p_l4t rc $?"
python3 $IN/pooled_inter_chunk_c20261021.py $P/p_l4t c16 t_mn t_l4t > $P/p_l4t_pooled_c16.txt 2>&1
log "queue l4t done"
