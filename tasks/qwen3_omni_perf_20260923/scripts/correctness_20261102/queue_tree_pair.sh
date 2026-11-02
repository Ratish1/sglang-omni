set -u
# 11-02 a branch against its base: one server per arm from its own tree, same time on cards 0 and 1, then swapped, on the
# streamed full SeedTTS EN corpus at each load, request events on, each boot scored on its card; then the seeded corpus at
# c1, same time, for byte identity. Usage: queue_tree_pair.sh OUT BASE_TREE ARM_TREE "16 64 1"
OUT=$1
BASE_TREE=$2
ARM_TREE=$3
LOADS=$4
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
PORT=17000
mkdir -p $OUT
log() { echo "$(date -u +%T) $*" >> $OUT/queue.log; }
listening() {
  (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null
}
# note (ratish): the node's network is shared with other tenants, and a server also binds its port plus 100.
free_port() {
  local port=$1
  while listening $port || listening $((port + 100)) || listening $((port + 300)) || listening $((port + 400)); do
    port=$((port + 1000))
  done
  echo $port
}
boot() {
  local tree=$1 dir=$2 card=$3 port=$4 conc=$5 arm=$6 score=$7 cpus
  # note (ratish): both cards sit on NUMA node 1; each arm gets the same number of its CPUs, none shared.
  if [ "$card" = 0 ]; then cpus=56-69,168-181; else cpus=70-83,182-195; fi
  setsid env PIN_CPUS=$cpus PIN_NODE=1 EVENTS=1 SCORE=$score SGLANG_OMNI_STRICT_PORT=1 \
    bash $S/run_bench_boot.sh $tree $dir $card $port h200 $conc $arm > $dir.boot.log 2>&1 &
}
wait_round() {
  until grep -q '^done ' $1/base/progress.txt 2>/dev/null && grep -q '^done ' $1/arm/progress.txt 2>/dev/null; do
    sleep 10
  done
}
log "pair start: base $(git -C $BASE_TREE rev-parse --short HEAD) $(git -C $BASE_TREE rev-parse HEAD:sglang_omni | cut -c1-8); arm $(git -C $ARM_TREE rev-parse --short HEAD) $(git -C $ARM_TREE rev-parse HEAD:sglang_omni | cut -c1-8); tools $(git -C /data/tools rev-parse --short HEAD)"
for conc in $LOADS; do
  for order in 0 1; do
    round=c${conc}_o${order}
    mkdir -p $OUT/$round
    PORT=$(free_port $PORT)
    log "$round start: base card $order port $PORT, arm card $((1 - order)) port $((PORT + 300))"
    boot $BASE_TREE $OUT/$round/base $order $PORT $conc seedtts_en 1
    boot $ARM_TREE $OUT/$round/arm $((1 - order)) $((PORT + 300)) $conc seedtts_en 1
    wait_round $OUT/$round
    for arm in base arm; do
      log "  $round $arm: $(grep -E 'gen seedtts_en rc|FAILED' $OUT/$round/$arm/progress.txt | cut -c1-60); $(python3 $S/state_durations.py $OUT/$round/$arm 2>/dev/null | tail -1 | cut -c1-240)"
    done
    PORT=$((PORT + 1000))
  done
done
round=seeded
mkdir -p $OUT/$round
PORT=$(free_port $PORT)
log "$round start: base card 0 port $PORT, arm card 1 port $((PORT + 300))"
boot $BASE_TREE $OUT/$round/base 0 $PORT 1 seedtts_en_seeded 0
boot $ARM_TREE $OUT/$round/arm 1 $((PORT + 300)) 1 seedtts_en_seeded 0
wait_round $OUT/$round
python3 - "$OUT/$round" >> $OUT/queue.log 2>&1 <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
def digests(arm):
    rows = json.loads((root / arm / "seedtts_en_seeded" / "generated.json").read_text())
    return {row["sample_id"]: hashlib.sha256(Path(row["wav_path"]).read_bytes()).hexdigest() for row in rows if row.get("wav_path")}
base, arm = digests("base"), digests("arm")
shared = sorted(set(base) & set(arm))
print(f"seeded identity: {len(shared)} wavs compared, {sum(base[k] != arm[k] for k in shared)} differ, base only {len(set(base) - set(arm))}, arm only {len(set(arm) - set(base))}")
PY
log "pair done"
