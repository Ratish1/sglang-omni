#!/usr/bin/env bash
# 11-05, node-0000, after q1105_fp2. The default thinker path has no float32 mode (its RMSNorm kernel refuses Float), so float16, with
# about 8 times finer rounding than bfloat16, is the numerics reference against fb1 and fb2. Then the stage alone on CPU under
# passive OpenMP: whether more HF calls at once still cost CPU per video, which is the only cost side the 2 had.
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
F=/workspace/wk_final T=/workspace/wk_fix
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
mkdir -p $O/stage
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1105_fp2 done" $O/queue.log 2>/dev/null; do sleep 5; done
video() {
  local name=$1
  log "boot $name two card loads 1,16 float16 start"
  python3 $K/baseline.py --tree $F --model omni_speech --card 0 --port 28400 --out $O/$name \
    --bench videomme_text --loads 1,16 --passes clean,events --bench-repeats 2 \
    --serve-extra "$CONTEXT_32K --thinker.engine.dtype float16" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
cell() {
  local cpus=$1 tag=$2 policy=$3 processors=$4
  local name=${tag}_${policy}_n${processors}
  ( cd $T && OMP_NUM_THREADS=8 OMP_WAIT_POLICY=$policy PYTHONPATH=$T taskset -c $cpus python3 $K/preprocess_stage_bench.py --concurrency 16 \
    --processors $processors --out $O/stage/$name.json > $O/stage/$name.log 2>&1 )
  log "cell $name rc $?"
}
log "q1105_fp3 start"
video fh1
video fh2
C112=56-111,168-223 C16=56-63,168-175
for processors in 1 2 3 4 6 8; do cell $C112 c112 PASSIVE $processors; done
for processors in 2 4; do cell $C112 c112 ACTIVE $processors; done
for processors in 1 2 4; do cell $C16 c16 PASSIVE $processors; done
log "q1105_fp3 done"
