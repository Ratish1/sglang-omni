#!/usr/bin/env bash
# After queue_duplex.sh: the final branches' suites and the pre-fix proofs of their new tests, the engine's startup
# memory per step on main and the encoder branch (0.6B Base), and an empty reference against main, the measured
# first batch (before the fix) and the encoder branch's head, one 0.6B server each.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/duplex_20261027
L=$D/after.log
until grep -q "queue duplex done" $D/queue.log; do sleep 30; done
cd /workspace/sglang-omni
git fetch -q origin fix/qwen3-tts-reference-encoder-float32-exact-length fix/tts-bench-score-natural-stops
for spec in "fix 777e6fb0a" "refenc_prefix 593c67424" "bench 572cd7e30" "bench_prefix ba706ef76"; do
  set -- $spec
  rm -rf /workspace/tree_$1; git worktree prune
  git worktree add -q --detach /workspace/tree_$1 $2
  echo "tree $1 $(git -C /workspace/tree_$1 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
done
echo "== units fix: $(cd /workspace/tree_fix && python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider 2>&1 | tail -n 3 | head -n 1)" >> $L
echo "== units bench: $(cd /workspace/tree_bench && python3 -m pytest tests/unit_test/benchmarks -q -p no:cacheprovider 2>&1 | tail -n 3 | head -n 1)" >> $L
echo "== prefix empty reference on 593c67424:" >> $L
PYTHONPATH=/workspace/tree_refenc_prefix python3 $HERE/prefix_call.py /workspace/tree_fix/tests/unit_test/qwen3_tts/test_reference_encoder_cuda_graph.py test_reference_code_batcher_refuses_an_empty_reference 2>&1 | grep -v -i -E "warn|torchada" | tail -n 6 >> $L
echo "== prefix length-stop gate on ba706ef76:" >> $L
PYTHONPATH=/workspace/tree_bench_prefix python3 $HERE/prefix_call.py /workspace/tree_bench/tests/unit_test/benchmarks/test_tts_seedtts.py test_wer_scores_natural_stops_and_counts_length_stops 2>&1 | grep -v -i -E "warn|torchada" | tail -n 6 >> $L
for tree in main fix; do
  echo "== startup memory $tree:" >> $L
  PYTHONPATH=/workspace/tree_$tree python3 $HERE/startup_memory.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base 2>&1 | grep -E "^sglang_omni|resident|Error" >> $L
done
REF=$(python3 -c "import json; c = next(json.loads(l) for l in open('$D/cases/cases.jsonl') if '\"icl06\"' in l); print(c['ref_audio'])")
REF_TEXT=$(python3 -c "import json; c = next(json.loads(l) for l in open('$D/cases/cases.jsonl') if '\"icl06\"' in l); print(c['ref_text'])")
empty_reference() {
  local tree=$1 card=$2 port=$3 log=$D/logs/serve_empty_$1.log
  (cd /workspace/tree_$tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=/workspace/tree_$tree setsid bash -c "echo \$\$ > $D/logs/serve_empty_$tree.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --config examples/configs/qwen3_tts_0_6b.yaml --mem-fraction-static 0.08 --tts_engine.gpu $card --vocoder.gpu $card" > $log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $D/logs/serve_empty_$tree.pid) waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 900 ]; then echo "empty $tree: server not healthy" >> $L; kill -TERM -$pgid 2> /dev/null; return 1; else sleep 5; waited=$((waited + 5)); fi
  done
  python3 $HERE/empty_reference.py --base-url http://127.0.0.1:$port --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --reference "$REF" --reference-text "$REF_TEXT" --out $D/empty_$tree > $D/empty_$tree.txt 2>&1
  echo "== empty reference $tree: $(tr '\n' ';' < $D/empty_$tree.txt) server log: $(grep -c -E 'device-side assert|CUDA error' $log) device errors" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}
empty_reference main 0 4810 &
empty_reference first 1 4820 &
wait
empty_reference fix 0 4830
echo "after done $(date +%T)" >> $L
