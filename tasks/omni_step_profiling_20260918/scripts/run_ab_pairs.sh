#!/usr/bin/env bash
# A/B on two cards: each round boots tree A and tree B at the same time, one card each,
# runs the same seed-tts stream benchmark (warmup 1) against both, stops each server by its
# own process group, and optionally scores WER and speaker similarity on the boot's WAVs.
# A swapped round puts A on B's card and B on A's, so a card difference cannot pass as
# the change. Rounds run one after another.
# usage: run_ab_pairs.sh <out dir> <tree A> <tree B> <card 1> <card 2> <longform meta.lst>
set -u
OUT=$1 TREE_A=$2 TREE_B=$3 CARD1=$4 CARD2=$5 LONGFORM=$6
PY=/workspace/sglang-omni/.venv/bin/python
MODEL=/data/ratish/models/Qwen3-TTS-12Hz-1.7B-Base
META=zhaochenyang20/seed-tts-eval-arrow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$OUT"
md5sum "$0" > "$OUT/md5.txt"

# boot <label> <tree> <card> <port> <quality yes|no> <meta> <bench args...>
boot() {
  label=$1 tree=$2 card=$3 port=$4 quality=$5 meta=$6; shift 6
  d=$OUT/$label; mkdir -p "$d"
  echo "start $(date +%T) card $card" > "$d/progress.txt"
  git -C "$tree" rev-parse HEAD > "$d/head.txt"
  (cd "$tree" && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=$tree $PY -c \
    "import sglang_omni; print(sglang_omni.__file__)") > "$d/import_path.txt" 2>&1
  nvidia-smi > "$d/gpus_before.txt"
  nvidia-smi dmon -i "$card" -s pucvm -d 1 > "$d/dmon.log" 2>&1 &
  dmon=$!
  began=$(date +%s)
  (cd "$tree" && setsid bash -c "echo \$\$ > $d/server.pgid; exec env CUDA_VISIBLE_DEVICES=$card \
    PYTHONPATH=$tree $PY -u -m sglang_omni.cli serve --model-path $MODEL --port $port" \
    > "$d/serve.log" 2>&1 &)
  healthy=0
  for _ in $(seq 360); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; then healthy=1; break; fi
    sleep 5
  done
  if [ $healthy = 0 ]; then
    echo "server not healthy" > "$d/FAILED"
  else
    echo "healthy $(date +%T) startup_s $(( $(date +%s) - began ))" >> "$d/progress.txt"
    (cd "$TREE_A" && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=$TREE_A timeout 7200 $PY -m benchmarks.eval.benchmark_tts_seedtts \
      --model $MODEL --meta "$meta" --lang en --use-existing-server --host 127.0.0.1 --port $port \
      --warmup 1 --stream --generate-only --output-dir "$d/bench" "$@") > "$d/bench.log" 2>&1
    echo "bench rc $? $(date +%T)" >> "$d/progress.txt"
  fi
  kill -TERM -- -"$(cat "$d/server.pgid")" 2>/dev/null
  sleep 15
  kill -0 -- -"$(cat "$d/server.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$d/server.pgid")"
  kill $dmon 2>/dev/null
  if [ -f "$d/FAILED" ] || [ "$quality" = no ]; then
    echo "done $(date +%T)" >> "$d/progress.txt"
    return
  fi
  for _ in $(seq 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$card")" -lt 100 ] && break
    sleep 2
  done
  (cd "$TREE_A" && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=$TREE_A $PY -m benchmarks.eval.benchmark_tts_seedtts \
    --model $MODEL --meta "$meta" --lang en --port $((port + 100)) --skip-gpu-cleanup \
    --transcribe-only --output-dir "$d/bench") > "$d/wer.log" 2>&1
  echo "wer rc $? $(date +%T)" >> "$d/progress.txt"
  # note(ratish): --skip-gpu-cleanup returns before the ASR server frees the card
  for _ in $(seq 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$card")" -lt 100 ] && break
    sleep 2
  done
  (cd "$TREE_A" && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=$TREE_A $PY -m benchmarks.eval.benchmark_tts_seedtts \
    --model $MODEL --meta "$meta" --lang en --similarity-only --output-dir "$d/bench") > "$d/sim.log" 2>&1
  echo "sim rc $? $(date +%T)" >> "$d/progress.txt"
  echo "done $(date +%T)" >> "$d/progress.txt"
}

# round <name> <card for A> <card for B> <quality> <meta> <bench args...>
round() {
  name=$1 card_a=$2 card_b=$3; shift 3
  echo "round $name start $(date +%T)" >> "$OUT/rounds.txt"
  boot "${name}_a" "$TREE_A" "$card_a" 8201 "$@" &
  boot "${name}_b" "$TREE_B" "$card_b" 8202 "$@" &
  wait
  echo "round $name end $(date +%T)" >> "$OUT/rounds.txt"
}

round c16 "$CARD1" "$CARD2" yes $META --concurrency 16
round c16_swapped "$CARD2" "$CARD1" no $META --concurrency 16
round longform_c16 "$CARD1" "$CARD2" no "$LONGFORM" --concurrency 16
round longform_c16_swapped "$CARD2" "$CARD1" no "$LONGFORM" --concurrency 16
round c1_seeded "$CARD1" "$CARD2" yes $META --concurrency 1 --seed 1234
echo "all done $(date +%T)" > "$OUT/DONE"
