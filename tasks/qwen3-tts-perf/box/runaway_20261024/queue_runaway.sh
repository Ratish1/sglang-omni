#!/usr/bin/env bash
# Is the Qwen3-TTS talker's runaway (a repetition loop that never samples codec EOS, to the 2,048-frame cap) the
# model's or ours? The three SeedTTS EN prompts that ran away on 10-24, default serving: main against #3102,
# non-streaming, c16, unseeded, 200 of each prompt a run, both card orders, no scoring; then the official qwen_tts
# implementation at the checkpoint's own sampling defaults, 100 of each prompt on each card; rates per prompt.
set -u
D=/data/runaway
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
REPEAT=$T/tasks/qwen3-tts-perf/box/chain_20261022/repeat_meta.py
HERE=$T/tasks/qwen3-tts-perf/box/runaway_20261024
M=${M:?the main tree}
P1=${P1:?the 3102 tree}
L=$D/queue.log
mkdir -p $D
BASES="10000 11200"
BUSY=$(python3 - $BASES <<'EOF'
import sys
listening = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    for line in open(table).read().splitlines()[1:]:
        fields = line.split()
        if fields[3] == "0A":
            listening.add(int(fields[1].rsplit(":", 1)[1], 16))
print(" ".join(str(port) for port in sorted(listening) if any(int(base) <= port <= int(base) + 1100 for base in sys.argv[1:])))
EOF
)
echo "queue runaway start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')3102 $(git -C $P1 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) busy ports [$BUSY]" >> $L
if [ -n "$BUSY" ]; then
  echo "stopped: listeners inside the pair port ranges" >> $L
  echo "queue_runaway_done $(date +%T)" >> $L
  exit 1
else
  :
fi
PROMPTS="common_voice_en_19916471-common_voice_en_19916474 common_voice_en_20660224-common_voice_en_20660228 common_voice_en_21319256-common_voice_en_21319257"
(cd $M && PYTHONPATH=$M python3 $REPEAT $D/served $(for p in $PROMPTS; do printf '%s:200 ' $p; done) >> $L 2>&1)
(cd $M && PYTHONPATH=$M python3 $REPEAT $D/reference $(for p in $PROMPTS; do printf '%s:100 ' $p; done) >> $L 2>&1)
run_pair() {
  if [ -f $PAIR_ROOT/$1/a/bench/speed_results.json ] && [ -f $PAIR_ROOT/$1/b/bench/speed_results.json ]; then
    return
  else
    :
  fi
  python3 $IN/pair.py --label $1 --tree $M --b-tree $P1 --concurrency 16 --a-card $2 --b-card $3 --port $4 \
    --meta $D/served/meta.lst --no-stream --skip-score > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T)" >> $L
}
run_pair runaway_ns_c16_a 2 3 10000
run_pair runaway_ns_c16_b 3 2 11200
python3 $HERE/runaway_rates.py --pairs runaway_ns_c16_a runaway_ns_c16_b > $D/runaway_rates_served.txt 2>&1
echo "served rates done $(date +%T)" >> $L
for card in 2 3; do
  (cd $D && ${REFPY:?the reference venv python} $HERE/reference_runaway.py $D/reference/meta.lst $D/reference_card$card.json --card $card \
    > $D/reference_card$card.txt 2>&1; echo "reference card $card rc $? $(date +%T)" >> $L) &
done
wait
python3 $HERE/runaway_rates.py --pairs runaway_ns_c16_a runaway_ns_c16_b \
  --reference $D/reference_card2.json $D/reference_card3.json > $D/runaway_rates.txt 2>&1
echo "queue_runaway_done $(date +%T)" >> $L
