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
#   SAMPLES  first N of the English split, empty for all of it     16
#   CONC     client concurrency                                  1
#   MODE     streaming or buffered                              streaming
#   SEED     sampling seed sent with every request, empty for    1234
#            the benchmark's default, which sends none
#   MODEL    checkpoint, a hub id or a local directory     FunAudioLLM/Fun-CosyVoice3-0.5B-2512
#   SERVE    extra serve arguments, the same string on both arms  empty
#   BARRIER  directory the arms of a pair meet in once their         empty
#            servers answer /health; PEERS arms must arrive            2
#   LEDGER   non empty makes it a profiling boot: the stage 0 call   empty
#            ledger wraps the vocoder calls and writes one JSON line
#            per call under the run's ledger directory
set -euo pipefail

REPO=${REPO:-/workspace/sglang-omni}
ARM=${ARM:?ARM is the arm label, for example main or slice}
REV=${REV:?REV is the revision this arm boots from}
CARD=${CARD:?CARD is the card index}
PORT=${PORT:-8000}
SAMPLES=${SAMPLES-16}
CONC=${CONC:-1}
MODE=${MODE:-streaming}
SEED=${SEED-1234}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-}
SERVE=${SERVE:-}
LEDGER=${LEDGER:-}
BARRIER=${BARRIER:-}
PEERS=${PEERS:-2}
# A session name turns the arm into an Nsight capture: the server launches under
# nsys and the runner opens the window around the measured benchmark only. The
# metric device is the nsys ordinal, which is the physical card, not the ordinal
# CUDA_VISIBLE_DEVICES leaves the process.
NSYS=${NSYS-}
NSYS_WARMUP=${NSYS_WARMUP:-32}
NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx,osrt,python-gil}
MODEL=${MODEL:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}
ANALYSIS_BRANCH=${ANALYSIS_BRANCH:-analysis/cosyvoice-utilization-20260912}
# The checkpoint loader imports CosyVoice and its Matcha submodule, which the
# container keeps as a clone rather than a wheel.
COSYVOICE=${COSYVOICE:-/data/src/CosyVoice}

cd "$REPO"
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
# The ledger's sitecustomize has to be found before any other, so its directory
# leads the server's path; it does nothing in a process without the variable.
LEDGER_ENV=""
if [ -n "$LEDGER" ]; then
  SERVER_PATH="$T/stage0/call_ledger:$SERVER_PATH"
  LEDGER_ENV="COSY_CALL_LEDGER_DIR=$OUT/ledger"
fi

# An empty SEED and no token limit is the benchmark's own default request: no
# generation json is written and the flag is left off.
GENERATION_ARG=""
if [[ "$MAX_NEW_TOKENS" == "unset" && -n "$SEED" ]]; then
  printf '{"seed": %d, "max_new_tokens": null}\n' "$SEED" > "$OUT/generation.json"
elif [[ "$MAX_NEW_TOKENS" == "unset" ]]; then
  # The benchmark's own default sends max_new_tokens 2048, which overrides the
  # engine's bound of 20 tokens per text token. A null sends no limit at all,
  # which is what a client that passes only text does.
  printf '{"max_new_tokens": null}\n' > "$OUT/generation.json"
elif [[ -n "$SEED" && -n "$MAX_NEW_TOKENS" ]]; then
  printf '{"seed": %d, "max_new_tokens": %d}\n' "$SEED" "$MAX_NEW_TOKENS" > "$OUT/generation.json"
elif [[ -n "$SEED" ]]; then
  printf '{"seed": %d}\n' "$SEED" > "$OUT/generation.json"
elif [[ -n "$MAX_NEW_TOKENS" ]]; then
  printf '{"max_new_tokens": %d}\n' "$MAX_NEW_TOKENS" > "$OUT/generation.json"
fi
[ -f "$OUT/generation.json" ] && GENERATION_ARG="--generation-json $OUT/generation.json"

teardown() {
  [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null || true
  [ -n "${MEMORY_PID:-}" ] && kill "$MEMORY_PID" 2>/dev/null || true
  [ -f "$OUT/server.pgid" ] || return 0
  kill -- "-$(cat "$OUT/server.pgid")" 2>/dev/null || true
  sleep 5
  kill -9 -- "-$(cat "$OUT/server.pgid")" 2>/dev/null || true
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv \
    > "$OUT/gpus_after.csv" || true
}
trap teardown EXIT

# Whole run utilization of the card under test. The sm column is the share of
# time a kernel was resident, which is GR active, not SM occupancy.
nvidia-smi dmon -i "$CARD" -s u -d 1 > "$OUT/dmon.csv" 2>/dev/null &
DMON_PID=$!
# Memory in use on the card once a second, for the peak of the run.
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -i "$CARD" -l 1 \
  > "$OUT/memory.csv" 2>/dev/null &
MEMORY_PID=$!
echo "arm $ARM, revision $(cat "$OUT/head.txt"), card $CARD, port $PORT"
echo "out $OUT"
# note(ratish): without the strict port the server takes any free port when
# this one is held, and the client then polls the asked for port until it times
# out. Fail loudly on a clash instead.
# The previous point's server may still hold the port while it shuts down.
until ! (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do sleep 2; done
LAUNCH=""
if [ -n "$NSYS" ]; then
  # The traced APIs are an application scope option, so they go on launch; the
  # runner's start only opens the window. python-gil records every GIL wait and
  # hold per thread, which is what tells a thread off the CPU from one blocked.
  LAUNCH="nsys launch --session-new=$NSYS --trace=$NSYS_TRACE \
    --cuda-graph-trace=node --trace-fork-before-exec=true"
fi
(cd "$TREE" && setsid bash -c "echo \$\$ > '$OUT/server.pgid'; exec env CUDA_VISIBLE_DEVICES=$CARD \
  SGLANG_OMNI_STRICT_PORT=1 $LEDGER_ENV PYTHONPATH='$SERVER_PATH' $LAUNCH python -u -m sglang_omni.cli serve --model-path '$MODEL' --host 127.0.0.1 --port $PORT $SERVE" \
  > "$OUT/serve.log" 2>&1 &)
echo "$SERVE" > "$OUT/serve_args.txt"

# With a barrier the client starts only once every arm's server answers /health,
# so an arm that boots slower (a compile) never runs beside the other's benchmark.
if [ -n "$BARRIER" ]; then
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    if ! kill -0 -- "-$(cat "$OUT/server.pgid")" 2>/dev/null; then
      mkdir -p "$BARRIER/failed" && echo "server exited" && exit 1
    else
      sleep 2
    fi
  done
  echo "ready $(date -u +%H:%M:%S)"
  touch "$BARRIER/$ARM"
  until [ "$(ls "$BARRIER" | wc -l)" -ge "$PEERS" ]; do
    if [ -d "$BARRIER/failed" ]; then
      echo "peer failed" && exit 1
    else
      sleep 1
    fi
  done
  echo "barrier passed $(date -u +%H:%M:%S)"
fi

cd "$REPO/.tmp/wt/analysis"
# An empty SAMPLES omits the flag, which is what selects the whole split.
SAMPLE_ARG=""
[ -n "$SAMPLES" ] && SAMPLE_ARG="--samples $SAMPLES"
# With a session the runner sends its own pre capture cohort and runs the
# measured benchmark at internal warmup zero, so the warmup here sizes that
# cohort rather than the benchmark's.
CAPTURE_ARG=""
WARMUP=1
if [ -n "$NSYS" ]; then
  CAPTURE_ARG="--session $NSYS --metrics-devices $CARD"
  WARMUP=$NSYS_WARMUP
fi
python -u "$T/diagnostics/run_seedtts.py" \
  --mode "$MODE" --lang en --concurrency "$CONC" --warmup "$WARMUP" $SAMPLE_ARG $CAPTURE_ARG \
  --model "$MODEL" --base-url "http://127.0.0.1:$PORT" \
  $GENERATION_ARG --ready-timeout 900 \
  --output "$OUT/seedtts" 2>&1 | tee "$OUT/client.log"

echo "OUT=$OUT"
