#!/usr/bin/env bash
# PF A/B: upstream main (A) against perf/qwen3-tts-base-prefill-graph (B), six boots at
# once, one card each: stream c16, seeded stream c1, buffered c16, per arm. Each boot:
# provenance and prefill-backend gate, serve, seed-tts full English corpus (warmup 1),
# SGLang prefill log lines, stop its own process group, then WER and speaker similarity
# on the boot's WAVs on the same card. Ends with the seeded c1 byte comparison.
# usage: run_pf_ab.sh <out dir> <tree A> <tree B>
set -u
OUT=$1 TREE_A=$2 TREE_B=$3
PY=/workspace/sglang-omni/.venv/bin/python
MODEL=/data/ratish/models/Qwen3-TTS-12Hz-1.7B-Base
META=zhaochenyang20/seed-tts-eval-arrow
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$OUT"
md5sum "$0" > "$OUT/md5.txt"

boot() {
  label=$1 tree=$2 card=$3 port=$4; shift 4
  d=$OUT/$label; mkdir -p "$d"
  echo "start $(date +%T)" > "$d/progress.txt"
  git -C "$tree" rev-parse HEAD > "$d/head.txt"
  (cd "$tree" && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=$tree $PY -c \
    "import sglang_omni; print(sglang_omni.__file__)") > "$d/import_path.txt" 2>&1
  nvidia-smi > "$d/gpus_before.txt"
  nvidia-smi dmon -i "$card" -s pucvm -d 1 > "$d/dmon.log" 2>&1 &
  dmon=$!
  (cd "$tree" && setsid bash -c "echo \$\$ > $d/server.pgid; exec env CUDA_VISIBLE_DEVICES=$card \
    PYTHONPATH=$tree $PY -u -m sglang_omni.cli serve --model-path $MODEL --port $port" \
    > "$d/serve.log" 2>&1 &)
  healthy=0
  for _ in $(seq 180); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; then healthy=1; break; fi
    sleep 5
  done
  if [ $healthy = 0 ]; then
    echo "server not healthy" > "$d/FAILED"
  else
    echo "healthy $(date +%T)" >> "$d/progress.txt"
    (cd "$TREE_A" && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=$TREE_A timeout 7200 $PY -m benchmarks.eval.benchmark_tts_seedtts \
      --model $MODEL --meta $META --lang en --use-existing-server --host 127.0.0.1 --port $port \
      --warmup 1 --generate-only --output-dir "$d/bench" "$@") > "$d/bench.log" 2>&1
    echo "bench rc $? $(date +%T)" >> "$d/progress.txt"
  fi
  kill -TERM -- -"$(cat "$d/server.pgid")" 2>/dev/null
  sleep 15
  kill -0 -- -"$(cat "$d/server.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$d/server.pgid")"
  kill $dmon 2>/dev/null
  grep -E "Prefill batch|prefill CUDA graph|cuda graph" "$d/serve.log" > "$d/prefill_lines.txt"
  [ -f "$d/FAILED" ] && return
  sleep 5
  asr_port=$((port + 100))
  (cd "$TREE_A" && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=$TREE_A $PY -m benchmarks.eval.benchmark_tts_seedtts \
    --model $MODEL --meta $META --lang en --port $asr_port --skip-gpu-cleanup \
    --transcribe-only --output-dir "$d/bench") > "$d/wer.log" 2>&1
  echo "wer rc $? $(date +%T)" >> "$d/progress.txt"
  (cd "$TREE_A" && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=$TREE_A $PY -m benchmarks.eval.benchmark_tts_seedtts \
    --model $MODEL --meta $META --lang en --similarity-only --output-dir "$d/bench") > "$d/sim.log" 2>&1
  echo "sim rc $? $(date +%T)" >> "$d/progress.txt"
  echo "done $(date +%T)" >> "$d/progress.txt"
}

boot a_c16_stream "$TREE_A" 1 8101 --concurrency 16 --stream &
boot b_c16_stream "$TREE_B" 2 8102 --concurrency 16 --stream &
boot a_c1_stream_seeded "$TREE_A" 3 8103 --concurrency 1 --stream --seed 1234 &
boot b_c1_stream_seeded "$TREE_B" 4 8104 --concurrency 1 --stream --seed 1234 &
boot a_c16_buffered "$TREE_A" 5 8105 --concurrency 16 &
boot b_c16_buffered "$TREE_B" 6 8106 --concurrency 16 &
wait

(cd "$OUT/a_c1_stream_seeded/bench" && find . -name '*.wav' -exec md5sum {} + | sort -k2) > "$OUT/identity_a.txt"
(cd "$OUT/b_c1_stream_seeded/bench" && find . -name '*.wav' -exec md5sum {} + | sort -k2) > "$OUT/identity_b.txt"
join -1 2 -2 2 "$OUT/identity_a.txt" "$OUT/identity_b.txt" | awk '{n++; if ($2 == $3) same++} END {print same+0 " of " n " WAVs byte identical"}' > "$OUT/identity.txt"
echo "all done $(date +%T)" > "$OUT/DONE"
