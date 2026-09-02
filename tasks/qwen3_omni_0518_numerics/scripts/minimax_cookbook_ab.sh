#!/usr/bin/env bash
# MiniMax Music 3 before and after, with the cookbook's own requests
# (docs/cookbook/minimax_music3.md, Reference outputs and Concurrency).
#
# Against a server started as the cookbook says, one arm at a time:
#   CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000
# then
#   PORT=8000 OUT=/data/minimax-ab/A bash minimax_cookbook_ab.sh
# and the same with OUT=/data/minimax-ab/B on the other checkout. Compare with
#   diff /data/minimax-ab/A/sequential.sha256 /data/minimax-ab/B/sequential.sha256
#   cat /data/minimax-ab/*/timing.txt
#
# Pass 1 sends the five reference requests one at a time. Every request is a
# batch of one row pair, the same seed and prompt in both arms, so the five
# wavs have to be byte identical across the arms: the scheduler change cannot
# reach a single request's numerics. Pass 2 sends the same five at once, which
# is where admission differs, and reports wall time. Byte identity is not
# expected there, batch composition changes the arithmetic, the earlier c16
# run had 5 of 16 identical.
set -uo pipefail

: "${OUT:?set OUT}"
PORT="${PORT:-8000}"
URL="http://localhost:$PORT/v1/audio/speech"
mkdir -p "$OUT/sequential" "$OUT/parallel"

request() {
  # $1 name, $2 seed, $3 lyrics, $4 caption, $5 out dir
  local name=$1 seed=$2 lyrics=$3 caption=$4 dir=$5 t
  t=$(curl -s -o "$dir/$name.wav" -w '%{time_total}' -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"MiniMaxAI/MiniMax-Music3","input":sys.argv[1],"instructions":sys.argv[2],"seed":int(sys.argv[3]),"max_new_tokens":750,"response_format":"wav","stream":False}))' "$lyrics" "$caption" "$seed")")
  echo "$name $t" >> "$dir/../timing.txt"
}

NAMES=(00_lofi_hiphop 01_jpop_bright 02_synthwave_moody 03_acoustic_folk 04_orchestral_epic)
SEEDS=(1 2 3 4 5)
LYRICS=(
  $'[Verse]\nWalking down the empty street at midnight\nStreetlights flicker like a broken dream\nI\'ve got nothing but the sound of my own heartbeat\nEchoing through the silent concrete stream\n[Chorus]\nAnd I keep on walking\nTill the morning finds me\nLeave the night behind me'
  $'[Verse]\nMorning light is spilling through the curtain\nEvery colour waking up with me\n[Chorus]\nRun into the day and never look back\nEverything we wanted is ahead of us'
  $'[Intro]\n(instrumental)\n[Outro]\n(instrumental)'
  $'[Verse]\nI came up on a dirt road, nothing but a name\nCarried all my summers in a canvas bag\n[Chorus]\nAnd the river keeps on running\nLike it never learned to stay'
  $'[Intro]\n(instrumental)\n[Chorus]\nRise above the ashes of the fallen sky\nWe were never meant to say goodbye'
)
CAPTIONS=(
  "A melancholic lo-fi hip-hop track at 85 BPM in F minor: mellow Rhodes piano riff, soft vinyl crackle, dusty boom-bap drums with a laid-back swing, warm upright bass. Intimate bedroom production, gentle tape saturation, no bright cymbals."
  "A cheerful J-pop song at 128 BPM in C major: bright acoustic piano, chiming electric guitar, punchy four-on-the-floor drums, and a clear female lead vocal. Polished modern pop production, wide stereo, energetic and uplifting."
  "A moody synthwave instrumental at 100 BPM in D minor: pulsing analog bass arpeggio, gated reverb drum machine, wide atmospheric pads, and a soaring lead synth melody. Retro 1980s production, heavy chorus effect, cinematic and nocturnal."
  "A gentle acoustic folk ballad at 76 BPM in G major: fingerpicked steel-string guitar, soft brushed snare, subtle cello underneath, and a warm male vocal close to the microphone. Sparse and organic, natural room sound, very little compression."
  "An epic cinematic orchestral piece at 90 BPM in E minor: sweeping string ostinato, powerful brass swells, timpani and taiko percussion, and a distant choir. Wide concert-hall reverb, dynamic build from restrained to triumphant, no drum kit."
)

: > "$OUT/timing.txt"
echo "pass 1, sequential" >> "$OUT/timing.txt"
start=$(date +%s.%N)
for i in 0 1 2 3 4; do
  request "${NAMES[$i]}" "${SEEDS[$i]}" "${LYRICS[$i]}" "${CAPTIONS[$i]}" "$OUT/sequential"
done
echo "sequential wall $(python3 -c "import sys; print(round(float(sys.argv[1]) - float(sys.argv[2]), 3))" "$(date +%s.%N)" "$start")" >> "$OUT/timing.txt"
(cd "$OUT/sequential" && sha256sum *.wav) > "$OUT/sequential.sha256"

echo "pass 2, the same five at once" >> "$OUT/timing.txt"
start=$(date +%s.%N)
for i in 0 1 2 3 4; do
  request "${NAMES[$i]}" "${SEEDS[$i]}" "${LYRICS[$i]}" "${CAPTIONS[$i]}" "$OUT/parallel" &
done
wait
echo "parallel wall $(python3 -c "import sys; print(round(float(sys.argv[1]) - float(sys.argv[2]), 3))" "$(date +%s.%N)" "$start")" >> "$OUT/timing.txt"
(cd "$OUT/parallel" && sha256sum *.wav) > "$OUT/parallel.sha256"
cat "$OUT/timing.txt"
