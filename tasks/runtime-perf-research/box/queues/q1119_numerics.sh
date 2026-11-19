#!/usr/bin/env bash
# 11-19, node-0001: where 005-3's thinker computation first differs alone against co-batched (talker test prompt,
# greedy, thinker radix cache off, final tree 828ac76ea, colocated config). n2: serving graphs, the logits of every
# sampled position with full rows saved. n3: every forward eager, plus a fingerprint of the target's rows after every
# layer's attention, router GEMM, experts and layer output.
set -u
O=/data/v1119/num
K=/workspace/kit2/tasks/runtime-perf-research/box
T=/workspace/vfinal2
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> /data/v1119/queue.log; }
EXTRA="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768 --thinker.engine.disable_radix_cache true"
log "q1119_numerics start"
python3 $K/baseline.py --tree $T --model omni_colocated --port 28900 --out $O/n2 --passes none --serve-extra "$EXTRA" \
  --serve-env "PYTHONPATH=$K/thinker_numerics_probe:$T OMNI_NUMERICS_PROBE=$O/n2_probe OMNI_NUMERICS_TARGET_LENS=14226" \
  --after "python3 $K/thinker_cobatch_numerics.py --port 28900 --probe-prefix $O/n2_probe --out $O/n2_result.json" \
  > $O/n2.driver.log 2>&1
log "numerics n2 rc $?"
python3 $K/baseline.py --tree $T --model omni_colocated --port 28900 --out $O/n3 --passes none \
  --serve-extra "$EXTRA --thinker.engine.disable_cuda_graph true --thinker.engine.cuda_graph_backend_prefill disabled" \
  --serve-env "PYTHONPATH=$K/thinker_numerics_probe:$T OMNI_NUMERICS_PROBE=$O/n3_probe OMNI_NUMERICS_TARGET_LENS=14226 OMNI_NUMERICS_LAYERS=1" \
  --after "python3 $K/thinker_cobatch_numerics.py --port 28900 --probe-prefix $O/n3_probe --out $O/n3_result.json" \
  > $O/n3.driver.log 2>&1
log "numerics n3 rc $?"
log "q1119_numerics done"
