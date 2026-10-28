#!/usr/bin/env bash
# Block 1 of 10-28 (STATE "Now (10-28)", testing items 1 to 3, plus the reference encoder's teacher-forced check).
# Cards 0 and 1: the talker's per-step log-probabilities against official qwen-tts 0.1.1 (float32 reference encoder)
# teacher-forced on the same served frames, main and the encoder branch each with the probe, 0.6B English parity cases.
# Card 2: the final heads' suites, the pre-fix proofs of their new tests, the startup memory per step on main and the
# encoder branch. Card 3: an empty reference against main, the first batch before its fix and the encoder branch.
# Then cards 2 and 3: the split arms (exact-length encode on bf16, float32 encoder alone) on the cascade's 0.6B ICL sets.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
VENV=${VENV:?the official qwen-tts python}
REPO=/workspace/sglang-omni
BISECT=$T/tasks/qwen3-tts-perf/box/bisect_20261027
REVIEW=$T/tasks/qwen3-tts-perf/box/review_20261026
DUPLEX=$T/tasks/qwen3-tts-perf/box/duplex_20261027
M06=Qwen/Qwen3-TTS-12Hz-0.6B-Base
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
L=$D/queue.log
mkdir -p $D/logs
echo "queue block1 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
git -C $REPO fetch -q origin probe/qwen3-tts-tf-main-e9c probe/qwen3-tts-tf-refenc \
  fix/qwen3-tts-reference-encoder-float32-exact-length fix/qwen3-tts-two-frame-minimum-main \
  fix/tts-bench-score-natural-stops fix/qwen3-tts-checkpoint-generation-limit test/qwen3-tts-duplex-first-batch \
  test/qwen3-tts-duplex-exact-bf16 test/qwen3-tts-duplex-fp32-only
for spec in "tf_main b000be444" "tf_refenc 8207cf736" "main e9c058462" "refenc 7021219e5" "refenc_prev d802cf53a" \
  "refenc_prefix 593c67424" "min2 221b71346" "bench 572cd7e30" "bench_prefix ba706ef76" "first 437f81865" \
  "exact 3c00fb84e" "fp32 b184b44a9"; do
  set -- $spec
  rm -rf /workspace/n28_$1
  git -C $REPO worktree prune
  git -C $REPO worktree add -q --detach /workspace/n28_$1 $2
  echo "tree $1 $(git -C /workspace/n28_$1 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
done
(cd /workspace/n28_main && PYTHONPATH=/workspace/n28_main python3 $REVIEW/parity.py cases $D/parity >> $D/logs/cases.log 2>&1)
echo "parity cases $(wc -l < $D/parity/cases.jsonl) $(date +%T)" >> $L
(cd /workspace/n28_main && PYTHONPATH=/workspace/n28_main python3 $DUPLEX/duplex_ab.py cases $D/duplex \
  --profiles examples/streaming_cascade/profiles >> $D/logs/cases.log 2>&1)
echo "duplex cases $(wc -l < $D/duplex/cases.jsonl) $(date +%T)" >> $L

start_server() {
  local label=$1 tree=$2 port=$3
  shift 3
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $D/logs/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $D/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $D/logs/serve_$label.pid) waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label not healthy after $waited s; $(grep -E 'Error|error' $D/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      kill -TERM -$pgid 2> /dev/null
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s, pgid $pgid, $(grep -o 'KV pool holds [0-9]* tokens, [0-9.]* GiB' $D/logs/serve_$label.log | head -n 1) $(date +%T)" >> $L
}
stop_server() {
  local pgid=$(cat $D/logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do
    kill -0 $pgid 2> /dev/null || break
    sleep 1
  done
  kill -KILL -$pgid 2> /dev/null
  echo "$1 stopped $(date +%T)" >> $L
}

teacher_forced() {
  local name=$1 card=$2 port=$3 tree=/workspace/n28_tf_$1 out=$D/tf_$1
  mkdir -p $out/probe $out/served
  export QWEN3_TTS_PROBE_DIR=$out/probe
  start_server tf_$name $tree $port --model-path $M06 --tts_engine.gpu $card --vocoder.gpu $card || return 1
  (cd $tree && PYTHONPATH=$tree python3 $REVIEW/layout.py serve $D/parity/cases.jsonl --base-url http://127.0.0.1:$port \
    --model $M06 --label tf_$name --language English --out $out/served --max-new-tokens 300 >> $D/logs/client_tf_$name.log 2>&1)
  echo "tf_$name served rc $? $(date +%T)" >> $L
  stop_server tf_$name
  echo "tf_$name eos lines $(cat $out/probe/eos_*.txt | wc -l), code lines $(cat $out/probe/codes_*.txt | wc -l)" >> $L
  $VENV $BISECT/teacher_forced_reference.py $out/served $out/probe $D/parity/cases.jsonl $out/official_tf.jsonl \
    --model $M06 --card $card --natural-per-set 50 --encoder-fp32 >> $D/logs/official_tf_$name.log 2>&1
  echo "tf_$name official rc $? $(tail -n 1 $D/logs/official_tf_$name.log) $(date +%T)" >> $L
  python3 $BISECT/teacher_forced_compare.py $out/probe $out/official_tf.jsonl > $out/compare.txt 2>&1
  echo "tf_$name compare rc $? $(date +%T)" >> $L
}

units() {
  echo "== units refenc: $(cd /workspace/n28_refenc && CUDA_VISIBLE_DEVICES=2 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider 2>&1 | tail -n 1) $(date +%T)" >> $L
  echo "== units min2: $(cd /workspace/n28_min2 && CUDA_VISIBLE_DEVICES=2 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider 2>&1 | tail -n 1) $(date +%T)" >> $L
  echo "== units bench: $(cd /workspace/n28_bench && python3 -m pytest tests/unit_test/benchmarks -q -p no:cacheprovider 2>&1 | tail -n 1) $(date +%T)" >> $L
  echo "== prefix empty reference on 593c67424: $(cd /workspace/n28_refenc_prefix && PYTHONPATH=/workspace/n28_refenc_prefix python3 $DUPLEX/prefix_call.py /workspace/n28_refenc/tests/unit_test/qwen3_tts/test_reference_encoder_cuda_graph.py test_reference_code_batcher_refuses_an_empty_reference 2>&1 | grep -v -i -E 'warn|torchada' | tail -n 4 | tr '\n' ' ')" >> $L
  echo "== prefix length-stop gate on ba706ef76: $(cd /workspace/n28_bench_prefix && PYTHONPATH=/workspace/n28_bench_prefix python3 $DUPLEX/prefix_call.py /workspace/n28_bench/tests/unit_test/benchmarks/test_tts_seedtts.py test_wer_scores_natural_stops_and_counts_length_stops 2>&1 | grep -v -i -E 'warn|torchada' | tail -n 4 | tr '\n' ' ')" >> $L
  echo "== prefix base-only float32 loader on d802cf53a: $(cd /workspace/n28_refenc_prev && PYTHONPATH=/workspace/n28_refenc_prev python3 -m pytest /workspace/n28_refenc/tests/unit_test/qwen3_tts/test_reference_encoder_cuda_graph.py -k float32_checkpoint -q -p no:cacheprovider 2>&1 | grep -E 'passed|failed|PASSED|FAILED' | tail -n 4 | tr '\n' ' ')" >> $L
  for tree in main refenc; do
    for model in $M06 $M17; do
      echo "== startup memory $tree $model: $(cd /workspace/n28_$tree && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/n28_$tree python3 $DUPLEX/startup_memory.py --model $model 2>&1 | grep -E '^sglang_omni|resident|Error' | tr '\n' ';')" >> $L
    done
  done
}

empty_references() {
  local ref=$(python3 -c "import json; c = next(json.loads(l) for l in open('$D/parity/cases.jsonl') if json.loads(l)['mode'] == 'icl'); print(c['ref_audio'])")
  local ref_text=$(python3 -c "import json; c = next(json.loads(l) for l in open('$D/parity/cases.jsonl') if json.loads(l)['mode'] == 'icl'); print(c['ref_text'])")
  local port=4730
  for tree in main first refenc; do
    start_server empty_$tree /workspace/n28_$tree $port --model-path $M06 --config examples/configs/qwen3_tts_0_6b.yaml \
      --mem-fraction-static 0.08 --tts_engine.gpu 3 --vocoder.gpu 3 || continue
    python3 $DUPLEX/empty_reference.py --base-url http://127.0.0.1:$port --model $M06 --reference "$ref" \
      --reference-text "$ref_text" --out $D/empty_$tree > $D/empty_$tree.txt 2>&1
    echo "== empty reference $tree: $(tr '\n' ';' < $D/empty_$tree.txt) device errors in the log: $(grep -c -E 'device-side assert|CUDA error' $D/logs/serve_empty_$tree.log)" >> $L
    stop_server empty_$tree
    port=$((port + 10))
  done
}

teacher_forced main 0 4710 &
teacher_forced refenc 1 4720 &
(units; echo "units done $(date +%T)" >> $L) &
(empty_references; echo "empty done $(date +%T)" >> $L) &
until grep -q "units done" $L && grep -q "empty done" $L; do sleep 20; done

declare -A CARD=([exact]=2 [fp32]=3) PORT=([exact]=4910 [fp32]=4920)
pids=()
for arm in exact fp32; do
  start_server split_$arm /workspace/n28_$arm ${PORT[$arm]} --model-path $M06 --config examples/configs/qwen3_tts_0_6b.yaml \
    --mem-fraction-static 0.08 --tts_engine.gpu ${CARD[$arm]} --vocoder.gpu ${CARD[$arm]} &
  pids+=($!)
done
wait ${pids[@]}
for draw in r1 r2; do
  pids=()
  for arm in exact fp32; do
    (cd /workspace/n28_main && PYTHONPATH=/workspace/n28_main python3 $DUPLEX/duplex_ab.py serve $D/duplex/cases.jsonl \
      --base-url http://127.0.0.1:${PORT[$arm]} --model $M06 --label ${arm}_$draw --shape icl06 \
      --out $D/runs/split_${arm}_${draw}_icl06_c16 --concurrency 16 >> $D/logs/client_split_$arm.log 2>&1) &
    pids+=($!)
  done
  wait ${pids[@]}
  echo "split $draw served $(date +%T)" >> $L
done
for arm in exact fp32; do stop_server split_$arm; done
python3 - $D >> $L <<'PY'
import collections, json, sys
from pathlib import Path
root = Path(sys.argv[1]) / "runs"
for run in sorted(root.glob("split_*icl06_c16")):
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
until grep -q "tf_main compare rc" $L && grep -q "tf_refenc compare rc" $L; do sleep 20; done
echo "queue block1 done $(date +%T)" >> $L
