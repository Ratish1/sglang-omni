#!/usr/bin/env bash
# 11-03, after q1103_stagebench, node-0000: do the stage's outputs depend on its concurrency? Video-MME accuracy reads 0.594
# on the single loop against 0.615 on two workers, 11 unstable questions of 50. Every output tensor of the 50 CI videos hashed,
# one pass each: one processor at c1 (serial) as the reference, twice for the A/A; then two processors at c16 and c64 on the
# single loop, and the marks branch's two workers at c16. Equal hashes leave only the thinker's batches as the owner.
set -u
O=/data/rp1103n/identity
K=/workspace/kit_i/tasks/runtime-perf-research/box
T=/workspace/wk_bench
W=/workspace/marks_wk
CPUS=56-111,168-223
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> /data/rp1103n/queue.log; }
until grep -q "q1103_stagebench done" /data/rp1103n/queue.log 2>/dev/null; do sleep 20; done
log "q1103_identity start"
cell() {
  local name=$1 tree=$2 concurrency=$3 processors=$4
  ( cd $tree && OMP_NUM_THREADS=8 PYTHONPATH=$tree taskset -c $CPUS python3 $K/preprocess_stage_bench.py --concurrency $concurrency \
    --processors $processors --passes 1 --hashes $O/$name.hashes.json --out $O/$name.json > $O/$name.log 2>&1 )
  log "identity $name rc $?"
}
cell serial_a $T 1 1
cell serial_b $T 1 1
cell loop_c16_a $T 16 2
cell loop_c16_b $T 16 2
cell loop_c64 $T 64 2
cell workers_c16 $W 16 2
cd $O
python3 -I - > $O/read_identity.txt 2>&1 <<'PY'
import json
reference = json.load(open("serial_a.hashes.json"))
print("samples", len(reference), "tensors a sample", len(next(iter(reference.values()))))
for name in ("serial_b", "loop_c16_a", "loop_c16_b", "loop_c64", "workers_c16"):
    other = json.load(open(f"{name}.hashes.json"))
    differing = {}
    for sample, tensors in reference.items():
        for path, value in tensors.items():
            if other.get(sample, {}).get(path) != value:
                differing.setdefault(path, []).append(sample)
    print(name, "samples compared", len(other), "tensors differing", {path: len(samples) for path, samples in differing.items()})
PY
log "q1103_identity done"
