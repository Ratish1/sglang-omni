#!/usr/bin/env bash
# Block 38 of 10-31: the reference encode compiled once (perf/qwen3-tts-reference-encoder-compiled-once, CTREE: integer
# conv padding, encoder graphs only for Base, one compiled program captured into every bucket) against #3368 (DTREE).
# (1) the qwen3_tts suite on CTREE; (2) the served encode of each tree through encoder_speedups_gate.py: device time and
# kernels per encode, codes against official's float32 encode on all 1,088 references, capture time, recompiles
# (TORCH_LOGS); (3) startup of the default 1.7B Base server, cold (fresh inductor cache) then warm, both trees at once;
# (4) a 1.7B CustomVoice server of each tree: startup, and whether it captures encoder graphs; (5) the KV pool at the
# cascade profiles (0.6B static 0.08, 1.7B static 0.12); (6) ab_runs.py, the standard SeedTTS A/B.
# Ports 11000 to 15900, clear of the host's listeners when written (16000, 22100, 22400, 29500 to 29507).
# usage: D=... T=... DTREE=<tree> CTREE=<tree> bash queue_block38.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
DTREE=${DTREE:?the #3368 tree}
CTREE=${CTREE:?the compile tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block38
L=$W/queue.log
mkdir -p $W/logs
echo "queue block38 start $(date +%T) refenc $(git -C $DTREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')compile $(git -C $CTREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L

(cd $CTREE && PYTHONPATH=$CTREE CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider > $W/units.txt 2>&1)
echo "suite: $(grep -E 'passed|failed' $W/units.txt | tail -n 1) $(grep -E '^(FAILED|ERROR) ' $W/units.txt | head -n 5 | tr '\n' ' ') $(date +%T)" >> $L

(cd $DTREE && PYTHONPATH=$DTREE python3 $K/bisect_20261027/stage_seedtts_wavs.py $W/wavs > $W/logs/stage.log 2>&1)
echo "wavs: $(tail -n 1 $W/logs/stage.log) $(date +%T)" >> $L
for spec in "refenc $DTREE 0" "compile $CTREE 1"; do
  set -- $spec
  (cd $2 && PYTHONPATH=$2 TORCHINDUCTOR_CACHE_DIR=$W/ind_gate_$1 TORCH_LOGS=recompiles python3 \
    $K/next_20261029/encoder_speedups_gate.py $W/wavs --model $M17 --card $3 --variants served > $W/gate_$1.txt 2>&1) &
done
wait
for label in refenc compile; do
  echo "gate $label: $(grep -E '^== served' $W/gate_$label.txt | cut -c1-160) | $(grep -E 'official / graphs' $W/gate_$label.txt | tr -s ' ') | recompiles $(grep -c '\[__recompiles\] Recompiling' $W/gate_$label.txt) $(date +%T)" >> $L
done

boot() {
  local label=$1 tree=$2 port=$3 cache=$4
  shift 4
  if (echo > /dev/tcp/127.0.0.1/$port) 2> /dev/null; then
    echo "$label port $port already answers, not started" >> $L
    return 1
  else
    :
  fi
  local began=$(date +%s.%N)
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$cache setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
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
  echo "$label healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s, $(grep -o 'KV pool holds [0-9]* tokens' $W/logs/serve_$label.log | head -n 1), encoder graph lines $(grep -c 'reference encoder graphs captured' $W/logs/serve_$label.log) $(date +%T)" >> $L
}

stop() {
  local pgid=$(cat $W/logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

pair() {
  local name=$1 port=$2
  shift 2
  boot ${name}_refenc $DTREE $port $W/ind_boot_refenc --tts_engine.gpu 0 --vocoder.gpu 0 $* &
  local a=$!
  boot ${name}_compile $CTREE $((port + 200)) $W/ind_boot_compile --tts_engine.gpu 1 --vocoder.gpu 1 $* &
  local b=$!
  wait $a $b
  stop ${name}_refenc
  stop ${name}_compile
}

pair cold 14000 --model-path $M17
pair warm 14400 --model-path $M17
pair customvoice 14800 --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
pair cascade06 15200 --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --config examples/configs/qwen3_tts_0_6b.yaml --mem-fraction-static 0.08
pair cascade17 15600 --model-path $M17 --config examples/configs/qwen3_tts_1_7b.yaml --mem-fraction-static 0.12

(cd $T/tasks/qwen3-tts-perf/box/next_20261031 && python3 ab_runs.py --a refenc=$DTREE --b compile=$CTREE --root $W/ab --port 11000 > $W/ab.txt 2>&1)
echo "ab: $(grep -E '^(n16|s16)_o[12]|paired' $W/ab.txt | cut -c1-400 | tr '\n' '#')" >> $L
echo "queue block38 done $(date +%T)" >> $L
