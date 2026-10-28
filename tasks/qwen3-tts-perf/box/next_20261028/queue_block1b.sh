#!/usr/bin/env bash
# The encoder branch with its lasting tensors made before the first encode (d1d7dc79e): its suite, the new table
# assertion against the previous head, the startup memory per step, and the KV pool each profile's server sizes
# (0.6B at static fraction 0.08, 1.7B at 0.12, the profiles' configs) on main and on the branch, one server a card.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
REPO=/workspace/sglang-omni
DUPLEX=$T/tasks/qwen3-tts-perf/box/duplex_20261027
L=$D/queue_b.log
echo "queue block1b start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
git -C $REPO fetch -q origin fix/qwen3-tts-reference-encoder-float32-exact-length
rm -rf /workspace/n28_refenc2
git -C $REPO worktree prune
git -C $REPO worktree add -q --detach /workspace/n28_refenc2 d1d7dc79e
echo "tree refenc2 $(git -C /workspace/n28_refenc2 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
(cd /workspace/n28_refenc2 && CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider > $D/units_refenc2.txt 2>&1)
echo "== units refenc2: $(grep -E 'passed|failed' $D/units_refenc2.txt | tail -n 1); failing: $(grep -E '^(FAILED|ERROR)' $D/units_refenc2.txt | tr '\n' ' ')" >> $L
mkdir -p /tmp/tables_prefix && cp /workspace/n28_refenc2/tests/unit_test/qwen3_tts/test_reference_encoder_cuda_graph.py /tmp/tables_prefix/test_tables_prefix.py
echo "== prefix tables built at load on 7021219e5: $(cd /workspace/n28_refenc && PYTHONPATH=/workspace/n28_refenc python3 -m pytest /tmp/tables_prefix/test_tables_prefix.py -k float32_checkpoint -q -p no:cacheprovider --rootdir /tmp/tables_prefix 2>&1 | grep -E '^FAILED|passed|failed' | tr '\n' ' ')" >> $L
echo "== startup memory refenc2 0.6B: $(cd /workspace/n28_refenc2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/n28_refenc2 python3 $DUPLEX/startup_memory.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base 2>&1 | grep -E 'resident|Error' | tr '\n' ';')" >> $L

pool() {
  local label=$1 tree=$2 card=$3 port=$4 model=$5 config=$6 fraction=$7
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $D/logs/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port --model-path $model --config $config --mem-fraction-static $fraction --tts_engine.gpu $card --vocoder.gpu $card" > $D/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $D/logs/serve_$label.pid) waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1200 ]; then echo "$label not healthy" >> $L; break; else sleep 5; waited=$((waited + 5)); fi
  done
  echo "== pool $label: $(grep -o 'KV pool holds [0-9]* tokens, [0-9.]* GiB' $D/logs/serve_$label.log | head -n 1), silence ids $(grep -o 'masks [0-9]* silence codec ids' $D/logs/serve_$label.log | head -n 1) $(date +%T)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}
pids=()
pool main06 /workspace/n28_main 0 5010 Qwen/Qwen3-TTS-12Hz-0.6B-Base examples/configs/qwen3_tts_0_6b.yaml 0.08 & pids+=($!)
pool refenc2_06 /workspace/n28_refenc2 1 5020 Qwen/Qwen3-TTS-12Hz-0.6B-Base examples/configs/qwen3_tts_0_6b.yaml 0.08 & pids+=($!)
pool main17 /workspace/n28_main 2 5030 Qwen/Qwen3-TTS-12Hz-1.7B-Base examples/configs/qwen3_tts_1_7b.yaml 0.12 & pids+=($!)
pool refenc2_17 /workspace/n28_refenc2 3 5040 Qwen/Qwen3-TTS-12Hz-1.7B-Base examples/configs/qwen3_tts_1_7b.yaml 0.12 & pids+=($!)
wait ${pids[@]}
echo "queue block1b done $(date +%T)" >> $L
