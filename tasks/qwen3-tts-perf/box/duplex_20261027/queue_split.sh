#!/usr/bin/env bash
# After queue_after.sh: which of the encoder branch's two mechanisms moves the 0.6B ICL length stops and the opening.
# Two arms on upstream main plus the two-frame minimum, one a card: the exact-length encode with the bf16 encoder,
# and the float32 encoder without it, each started as the Nemotron profiles start Qwen3-TTS; every icl06 set at 16
# at a time, twice. Prints each arm's KV pool and, per run and set, length stops, errors and the opening.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/duplex_20261027
L=$D/split.log
until grep -q "after done" $D/after.log; do sleep 30; done
cd /workspace/sglang-omni
git fetch -q origin test/qwen3-tts-duplex-exact-bf16 test/qwen3-tts-duplex-fp32-only
declare -A COMMIT=([exact]=3c00fb84e [fp32]=b184b44a9) CARD=([exact]=0 [fp32]=1) PORT=([exact]=4910 [fp32]=4920)
for arm in exact fp32; do
  rm -rf /workspace/tree_$arm; git worktree prune
  git worktree add -q --detach /workspace/tree_$arm ${COMMIT[$arm]}
  echo "tree $arm $(git -C /workspace/tree_$arm rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
done
for arm in exact fp32; do
  (cd /workspace/tree_$arm && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=/workspace/tree_$arm setsid bash -c "echo \$\$ > $D/logs/serve_split_$arm.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port ${PORT[$arm]} --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --config examples/configs/qwen3_tts_0_6b.yaml --mem-fraction-static 0.08 --tts_engine.gpu ${CARD[$arm]} --vocoder.gpu ${CARD[$arm]}" > $D/logs/serve_split_$arm.log 2>&1 < /dev/null &)
done
sleep 2
for arm in exact fp32; do
  waited=0
  until curl -sf http://127.0.0.1:${PORT[$arm]}/health > /dev/null; do
    if [ $waited -ge 900 ]; then echo "$arm not healthy" >> $L; break; else sleep 5; waited=$((waited + 5)); fi
  done
  echo "$arm healthy after $waited s, $(grep -o 'KV pool holds [0-9]* tokens, [0-9.]* GiB' $D/logs/serve_split_$arm.log | head -n 1)" >> $L
done
for draw in r1 r2; do
  for arm in exact fp32; do
    (cd /workspace/tree_main && PYTHONPATH=/workspace/tree_main python3 $HERE/duplex_ab.py serve $D/cases/cases.jsonl \
      --base-url http://127.0.0.1:${PORT[$arm]} --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --label ${arm}_$draw --shape icl06 \
      --out $D/runs/split_${arm}_${draw}_icl06_c16 --concurrency 16 >> $D/logs/client_split_$arm.log 2>&1) &
  done
  wait
done
for arm in exact fp32; do
  pgid=$(cat $D/logs/serve_split_$arm.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
done
python3 - $D >> $L <<'PY'
import collections, json, sys
from pathlib import Path
root = Path(sys.argv[1]) / "runs"
for run in sorted(root.glob("*icl06_c16")):
    records = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines() if line]
    sets = collections.defaultdict(collections.Counter)
    onsets = collections.defaultdict(list)
    for record in records:
        sets[record["set"]]["length"] += record.get("finish_reason") == "length"
        sets[record["set"]]["errors"] += "error" in record
        if record.get("finish_reason") == "stop" and "onset_ms" in record:
            onsets[record["set"]].append(record["onset_ms"])
        else:
            pass
    parts = []
    for name in sorted(sets):
        values = sorted(onsets[name])
        parts.append(f"{name}: length {sets[name]['length']}, errors {sets[name]['errors']}, onset p50 {values[len(values) // 2]:.0f}")
    print(f"{run.name}: " + "; ".join(parts))
PY
echo "split done $(date +%T)" >> $L
