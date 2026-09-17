#!/usr/bin/env bash
# One arm of the G1 identity gate: boot a server from an arm worktree and
# generate the seeded c1 streaming corpus slice the arms are compared on.
#
# Run it on the moss box, one arm at a time, from the container:
#
#   ARM=main  REV=27a8293c CARD=4 PORT=8010 bash .../stage2/g1_stream_identity.sh
#   ARM=slice REV=c652aa5e CARD=5 PORT=8011 bash .../stage2/g1_stream_identity.sh
#
# Then compare the two output directories with g1_compare_audio.py. Identity is
# a per sample property and does not depend on timing, so the arms may share the
# box with anyone and may run at the same time on different cards.
#
#   ARM      arm label, also the worktree name under .tmp/wt
#   REV      revision the arm boots from
#   CARD     card index, taken as given
#   PORT     server port
#   SAMPLES  first N of the English split                        16
#   SEED     sampling seed sent with every request               1234
#   MODEL    checkpoint, a local directory                       /data/ms/.../master
#   SERVE    extra serve arguments, the same string on both arms  empty
set -euo pipefail

REPO=${REPO:-/workspace/sglang-omni}
ARM=${ARM:?ARM is the arm label, for example main or slice}
REV=${REV:?REV is the revision this arm boots from}
CARD=${CARD:?CARD is the card index}
PORT=${PORT:-8000}
SAMPLES=${SAMPLES:-16}
SEED=${SEED:-1234}
SERVE=${SERVE:-}
MODEL=${MODEL:-/data/ms/models/FunAudioLLM--Fun-CosyVoice3-0.5B-2512/snapshots/master}
ANALYSIS_BRANCH=${ANALYSIS_BRANCH:-analysis/cosyvoice-utilization-20260912}
# The checkpoint loader imports CosyVoice and its Matcha submodule, which the
# container keeps as a clone rather than a wheel.
COSYVOICE=${COSYVOICE:-/workspace/CosyVoice}

# The checkpoint and the SeedTTS arrow are on disk; offline so a boot cannot
# stall on the hub. The container exports a SOCKS proxy httpx prefers and then
# wants socksio for, which is not installed.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset ALL_PROXY all_proxy

cd "$REPO"
source .venv/bin/activate
mkdir -p .tmp
grep -qx '.tmp/' .git/info/exclude 2>/dev/null || echo '.tmp/' >> .git/info/exclude

worktree() {  # name, revision
  if [ -d ".tmp/wt/$1" ]; then
    git -C ".tmp/wt/$1" checkout --detach "$2"
  else
    git worktree add --detach ".tmp/wt/$1" "$2"
  fi
}

git fetch --no-tags https://github.com/Ratish1/sglang-omni.git "$ANALYSIS_BRANCH"
worktree analysis "$(git rev-parse FETCH_HEAD)"
worktree "$ARM" "$REV"

TREE="$REPO/.tmp/wt/$ARM"
T="$REPO/.tmp/wt/analysis/tasks/cosyvoice_utilization_20260912"
OUT="$REPO/.tmp/out/g1-$ARM-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

# Provenance of the code that will serve, not of the checkout the client runs
# from: the revision, the module the server process imports, and whether the
# hop contract of slice 1.1 is in that tree.
SERVER_PATH="$TREE:$COSYVOICE:$COSYVOICE/third_party/Matcha-TTS"
git -C "$TREE" rev-parse HEAD > "$OUT/head.txt"
(cd "$TREE" && PYTHONPATH="$SERVER_PATH" python -c \
  "import sglang_omni; print(sglang_omni.__file__)") > "$OUT/import_path.txt"
grep -q "^$TREE/" "$OUT/import_path.txt" || {
  echo "server would import $(cat "$OUT/import_path.txt"), not $TREE"; exit 1;
}
grep -c 'the Flow attention chunk' \
  "$TREE/sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py" > "$OUT/marker.txt" || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv > "$OUT/gpus_before.csv"
uptime > "$OUT/host_load.txt"

printf '{"seed": %d}\n' "$SEED" > "$OUT/generation.json"

teardown() {
  [ -f "$OUT/server.pgid" ] || return 0
  kill -- "-$(cat "$OUT/server.pgid")" 2>/dev/null || true
  sleep 5
  kill -9 -- "-$(cat "$OUT/server.pgid")" 2>/dev/null || true
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv \
    > "$OUT/gpus_after.csv" || true
}
trap teardown EXIT

echo "arm $ARM, revision $(cat "$OUT/head.txt"), card $CARD, port $PORT"
echo "out $OUT"
(cd "$TREE" && setsid bash -c "echo \$\$ > '$OUT/server.pgid'; exec env CUDA_VISIBLE_DEVICES=$CARD \
  PYTHONPATH='$SERVER_PATH' python -u -m sglang_omni.cli serve --model-path '$MODEL' --port $PORT $SERVE" \
  > "$OUT/serve.log" 2>&1 &)
echo "$SERVE" > "$OUT/serve_args.txt"

cd "$REPO/.tmp/wt/analysis"
python -u "$T/diagnostics/run_seedtts.py" \
  --mode streaming --lang en --concurrency 1 --warmup 1 --samples "$SAMPLES" \
  --model "$MODEL" --base-url "http://127.0.0.1:$PORT" \
  --generation-json "$OUT/generation.json" --ready-timeout 900 \
  --output "$OUT/seedtts" 2>&1 | tee "$OUT/client.log"

echo "OUT=$OUT"
