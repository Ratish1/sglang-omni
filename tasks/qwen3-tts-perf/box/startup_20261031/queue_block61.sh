#!/usr/bin/env bash
# Block 61 of 11-01: the reference encoder fix (fix/qwen3-tts-reference-encoder-no-split-reductions: split_reductions
# off in the encoder's compile, so no kernel config is picked by first-launch timing) against main. The Qwen3-TTS suite
# on the fix; the encoder alone (refenc_codes.py) on the fix into three fresh caches over both cards and on main into
# one, codes compared and graph replay device time per bucket; the fix against itself served on both cards
# (seeded_identity.py, the servers' own compiles); the standard SeedTTS A/B against main; then boots, each tree cold
# into its own cache and three warm rounds each. Waits for block 59d.
# usage: D=... T=<kit tree> MAIN=<tree> FIX=<tree> bash queue_block61.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
FIX=${FIX:?the fix tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block61
L=$W/queue.log
mkdir -p $W/logs
until grep -q "queue block59d done" $D/block59d/queue.log 2> /dev/null; do sleep 20; done
for arm in MAIN FIX; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block61 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $FIX && CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts > $W/suite.log 2>&1)
echo "suite: $(tail -n 1 $W/suite.log)" >> $L
probe() {
  local name=$1 tree=$2 gpu=$3 cache=$4
  (cd $tree && HF_HUB_OFFLINE=1 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/$cache/inductor TRITON_CACHE_DIR=$W/$cache/triton \
    timeout 1200 python3 $K/next_20261031/refenc_codes.py --gpu $gpu --out $W/$name > $W/$name.log 2>&1)
  echo "$name exit $? $(date +%T): $(tail -n 1 $W/$name.log)" >> $L
}
probe fix_card0_a $FIX 0 cacheFa
probe fix_card0_b $FIX 0 cacheFb
probe fix_card1_c $FIX 1 cacheFc
probe main_card0_a $MAIN 0 cacheMa
(cd $FIX && HF_HUB_OFFLINE=1 PYTHONPATH=$FIX python3 $K/next_20261031/refenc_codes.py --compare $W/fix_card0_a.pt \
  $W/fix_card0_b.pt $W/fix_card1_c.pt $W/main_card0_a.pt $D/block59e/nosplit_card0_a.pt $D/block59b/card0_eager.pt \
  > $W/compare.txt 2>&1)
grep -vE "INFO|WARN|Using the latest|Found the latest" $W/compare.txt >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $FIX --branch $FIX --root $W/identity_fix_aa --port 20000 > $W/identity_fix_aa.txt 2>&1)
echo "fix aa: $(grep -E 'byte identical|error lines|Error' $W/identity_fix_aa.txt | cut -c1-200 | tr '\n' '#')" >> $L
(cd $K/next_20261031 && python3 ab_runs.py --a main=$MAIN --b fix=$FIX --root $W/ab_fix --port 21000 > $W/ab_fix.txt 2>&1)
echo "ab_fix: $(grep -E '^(n16|s16)_o[12]|paired|Error' $W/ab_fix.txt | cut -c1-400 | tr '\n' '#')" >> $L

boot() {
  local label=$1 arm=$2 port=$3
  local tree=${!arm}
  local began=$(date +%s.%N)
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/boot_${arm,,}/inductor TRITON_CACHE_DIR=$W/boot_${arm,,}/triton \
    setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port \
    --model-path $M17 --tts_engine.gpu 0 --vocoder.gpu 0" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 1
  local pgid=$(cat $W/logs/serve_$label.pid)
  until kill -0 $pgid 2> /dev/null && curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null; then
      echo "$label died; $(grep -E 'Error|error' $W/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      return 1
    else
      sleep 1
    fi
  done
  echo "$label healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s $(date +%T)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}
port=24000
for arm in MAIN FIX; do
  boot cold_${arm,,} $arm $port
  port=$((port + 200))
done
for round in 1 2 3; do
  if [ $round -eq 2 ]; then
    order="FIX MAIN"
  else
    order="MAIN FIX"
  fi
  for arm in $order; do
    boot warm_${arm,,}_$round $arm $port
    port=$((port + 200))
  done
done
python3 $K/startup_20261031/boot_split.py $W > $W/split.txt 2>&1
tail -n 4 $W/split.txt >> $L
for log in $W/logs/serve_*.log; do echo "$(basename $log): $(grep -m1 -oE 'max_total_num_tokens=[0-9]+|#tokens: [0-9]+' $log)" >> $L; done
echo "queue block61 done $(date +%T)" >> $L
