set -u
# 11-02 the component accuracy suite on one H200, the checks that need no server: inputs first, then each component
# against HF FP32 with HF BF16 as the floor. Usage: queue_suite.sh OUT TREE TEST [TEST ...]
OUT=$1
TREE=$2
shift 2
mkdir -p $OUT
log() { echo "$(date -u +%T) $*" >> $OUT/queue.log; }
log "suite start: tree $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8); tools $(git -C /data/tools rev-parse --short HEAD); gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for test in "$@"; do
  began=$(date +%s)
  (cd $TREE && QWEN3_OMNI_ACCURACY_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct QWEN3_OMNI_ACCURACY_REPORT_DIR=$OUT/report \
    PYTHONPATH=$TREE python3 -m pytest -s -q -x -p no:cacheprovider \
    tests/test_model/qwen3_omni_component_accuracy/$test.py) > $OUT/$test.log 2>&1
  rc=$?
  log "$test rc $rc in $(( $(date +%s) - began )) s: $(grep -E '^\[[a-z_0-9]+\]' $OUT/$test.log | cut -c1-220 | head -3 | tr '\n' ' '); $(tail -1 $OUT/$test.log | cut -c1-120)"
done
log "suite done"
