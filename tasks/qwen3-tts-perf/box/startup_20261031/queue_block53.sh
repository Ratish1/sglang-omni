#!/usr/bin/env bash
# Block 53 of 10-31: (1) three rounds of warm boots of DWARM (one eager warm-up per whole-utterance decoder graph)
# against main, card 0, on block 44's main cache, split by phase. (2) The standard SeedTTS A/B (ab_runs.py) of each
# startup fix against main: LAUNCHER, MING, FREEZE, DWARM. Waits for block 52 first.
# usage: D=... T=<kit tree> MAIN=<tree> LAUNCHER=<tree> MING=<tree> FREEZE=<tree> DWARM=<tree> bash queue_block53.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block53
L=$W/queue.log
mkdir -p $W/logs
until grep -q "queue block52 done" $D/block52/queue.log 2> /dev/null; do sleep 20; done
for arm in MAIN LAUNCHER MING FREEZE DWARM; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block53 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L

boot() {
  local label=$1 tree=$2 port=$3 cache=$4
  if (echo > /dev/tcp/127.0.0.1/$port) 2> /dev/null; then
    echo "$label port $port already answers, not started" >> $L
    return 1
  else
    :
  fi
  local began=$(date +%s.%N)
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton \
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
  echo "$label healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s, $(grep -o 'KV pool holds [0-9]* tokens' $W/logs/serve_$label.log | head -n 1) $(date +%T)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

port=20000
for round in 1 2 3; do
  for arm in MAIN DWARM; do
    boot warm_${arm,,}_$round ${!arm} $port $D/block44/boot_cache_main
    port=$((port + 200))
  done
done
python3 $K/startup_20261031/boot_split.py $W > $W/split.txt 2>&1
tail -n 3 $W/split.txt >> $L

for arm in LAUNCHER MING FREEZE DWARM; do
  (cd $K/next_20261031 && python3 ab_runs.py --a main=$MAIN --b ${arm,,}=${!arm} --root $W/ab_${arm,,} --port 14000 > $W/ab_${arm,,}.txt 2>&1)
  echo "ab ${arm,,}: $(grep -E '^(n16|s16)_o[12]|paired' $W/ab_${arm,,}.txt | cut -c1-300 | tr '\n' '#')" >> $L
done
echo "queue block53 done $(date +%T)" >> $L
