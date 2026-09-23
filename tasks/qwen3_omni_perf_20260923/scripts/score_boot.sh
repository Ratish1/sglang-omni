#!/usr/bin/env bash
# Score an existing SeedTTS generation dir (OUT/seedtts_en/generated.json and wavs) the
# way run_bench_boot.sh scores its own: WER from a Qwen3-ASR server on the card, then
# speaker similarity after that server stops. The wavs are deleted afterwards.
# usage: score_boot.sh <tree> <out dir> <card> <asr port>
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4
S=$(cd "$(dirname "$0")" && pwd)
cd "$TREE" || exit 1
echo "score start $(date +%T)" >> "$OUT/progress.txt"
setsid bash -c "echo \$\$ > $OUT/asr.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE \
  python3 -u -m sglang_omni.cli serve --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port $PORT" > "$OUT/asr.log" 2>&1 &
healthy=0
for _ in $(seq 180); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then healthy=1; break; fi
  ps -o stat= -g "$(cat "$OUT/asr.pgid" 2>/dev/null)" 2>/dev/null | grep -qv '^Z' || break
  sleep 5
done
echo "asr healthy=$healthy $(date +%T)" >> "$OUT/progress.txt"
if [ $healthy = 1 ]; then
  CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE python3 "$S/run_bench.py" score --arm seedtts_en --asr-port "$PORT" --out "$OUT" > "$OUT/score_seedtts_en.log" 2>&1
  echo "score rc $? $(date +%T)" >> "$OUT/progress.txt"
fi
kill -TERM -- -"$(cat "$OUT/asr.pgid")" 2>/dev/null
sleep 15
kill -0 -- -"$(cat "$OUT/asr.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$OUT/asr.pgid")"
CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE python3 "$S/run_bench.py" sim --arm seedtts_en --out "$OUT" > "$OUT/sim_seedtts_en.log" 2>&1
echo "sim rc $? $(date +%T)" >> "$OUT/progress.txt"
find "$OUT" -name '*.wav' -path '*/audio/*' -delete
echo "done $(date +%T)" >> "$OUT/progress.txt"
