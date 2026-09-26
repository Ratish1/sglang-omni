#!/usr/bin/env bash
# One census boot of a Qwen3-Omni tree on one card: colocated server, GPU memory log, the
# fixed capture list of omni_captures.py, the step ledger on every trace, teardown.
# MODE formal serves the shipped config; MODE mapping turns every CUDA graph off and
# records python stacks, so each kernel maps to its launch line. MODE speech serves the
# shipped config and captures thinker decode with and without audio output.
# MODE shapes captures the extremes of every stage: the smallest prefill, the largest
# replayable one (2000 tokens, bucket 2048) and the largest chunk (8000, eager), decode at
# batch 1 and at each engine's max_running_requests. EXTRA_SERVE_ARGS is appended to the
# server args (formal on the h200 profile adds the thinker graph flags there); PIN_CPUS and
# PIN_NODE pin the server and the driver like run_bench_boot.sh.
# usage: run_census_boot.sh <tree> <out dir> <card> <port> <bf16|fp8|h200> <formal|mapping|speech|shapes-formal|shapes-mapping>
set -u
TREE=$1 OUT=$2 CARD=$3 PORT=$4 DTYPE=$5 MODE=$6
S=$(cd "$(dirname "$0")" && pwd)
case $DTYPE in
  bf16) MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct CONFIG=examples/configs/qwen3_omni_colocated_h100_bf16.yaml ;;
  fp8) MODEL=marksverdhei/Qwen3-Omni-30B-A3B-FP8 CONFIG=examples/configs/qwen3_omni_colocated_h100_fp8.yaml ;;
  h200) MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct CONFIG=examples/configs/qwen3_omni_colocated_h200.yaml ;;
  *) echo "dtype must be bf16, fp8 or h200"; exit 1 ;;
esac
SERVE_ARGS="--config $CONFIG --colocate --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768 ${EXTRA_SERVE_ARGS:-}"
PIN=${PIN_CPUS:+numactl --physcpubind=$PIN_CPUS --membind=${PIN_NODE:-0}}
STACK=""
case $MODE in *mapping)
  SERVE_ARGS="$SERVE_ARGS --thinker.engine.disable_cuda_graph true --thinker.engine.cuda_graph_backend_prefill disabled --talker_ar.engine.disable_cuda_graph true --talker_ar.engine.cuda_graph_backend_prefill disabled --code2wav.factory.enable_cuda_graph false --audio_encoder.factory.enable_layer_cuda_graph false"
  STACK=--with-stack ;;
esac
URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"
cd "$TREE" || exit 1

git -C "$TREE" rev-parse HEAD > "$OUT/head.txt"
git -C "$TREE" status --short > "$OUT/tree_status.txt"
PYTHONPATH=$TREE python3 -c "import sglang_omni, sglang; print(sglang_omni.__file__, sglang.__version__)" > "$OUT/import_path.txt"
md5sum "$S/omni_captures.py" "$S/step_ledger.py" "$0" > "$OUT/md5.txt"
echo "$SERVE_ARGS" > "$OUT/serve_args.txt"
echo "cpu pin: ${PIN:-none}" > "$OUT/progress.txt"
nvidia-smi > "$OUT/gpus_before.txt"
nvidia-smi -i "$CARD" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "$OUT/mem.csv" 2>&1 &
MEM_PID=$!
uuid=$(nvidia-smi -i "$CARD" --query-gpu=uuid --format=csv,noheader)
(while true; do nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$uuid" | sed "s/^/$(date +%T) /"; sleep 2; done) > "$OUT/apps.csv" 2>&1 &
APPS_PID=$!
(while true; do echo "$(date +%T) $(cat /proc/loadavg)"; sleep 30; done) > "$OUT/loadavg.txt" 2>&1 &
LOAD_PID=$!

began=$(date +%s)
setsid bash -c "echo \$\$ > $OUT/server.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE \
  $PIN python3 -u -m sglang_omni.cli serve --model-path $MODEL $SERVE_ARGS --host 127.0.0.1 --port $PORT" > "$OUT/serve.log" 2>&1 &

teardown() {
  kill -TERM -- -"$(cat "$OUT/server.pgid")" 2>/dev/null
  sleep 15
  kill -0 -- -"$(cat "$OUT/server.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$OUT/server.pgid")"
  kill "$MEM_PID" "$APPS_PID" "$LOAD_PID" 2>/dev/null
  sleep 5
  nvidia-smi > "$OUT/gpus_after.txt"
  echo "pids on the card: $(awk '{print $3}' "$OUT/apps.csv" | sort -u | tr '\n' ' ')" >> "$OUT/progress.txt"
}

healthy=0
for _ in $(seq 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' $URL/health)" = 200 ]; then healthy=1; break; fi
  # the container's pid 1 never reaps, so a dead server group is all zombies
  ps -o stat= -g "$(cat "$OUT/server.pgid" 2>/dev/null)" 2>/dev/null | grep -qv '^Z' || break
  sleep 5
done
if [ $healthy = 0 ]; then
  echo "server not healthy" > "$OUT/FAILED"
  teardown
  exit 1
fi
echo "healthy startup_s $(( $(date +%s) - began )) $(date +%T)" >> "$OUT/progress.txt"

capture() {
  local name=$1; shift
  echo "start $name $(date +%T)" >> "$OUT/progress.txt"
  PYTHONPATH=$TREE $PIN python3 "$S/omni_captures.py" --url $URL --model $MODEL --out "$OUT" "$@" \
    > "$OUT/driver_$name.json" 2> "$OUT/driver_$name.err"
  echo "end $name rc=$? $(date +%T)" >> "$OUT/progress.txt"
}

L=${MODE#shapes-}
if [ "${MODE%-*}" = shapes ]; then
  capture tp_min --stage thinker --kind prefill --batch 1 --prompt-tokens 1 --label ${L}_min $STACK
  capture tp_2k --stage thinker --kind prefill --batch 1 --prompt-tokens 2000 --label ${L}_2k $STACK
  capture tp_8k --stage thinker --kind prefill --batch 1 --prompt-tokens 8000 --label ${L}_8k $STACK
  capture td_b1 --stage thinker --kind decode --batch 1 --label $L $STACK
  capture td_b64 --stage thinker --kind decode --batch 64 --label $L $STACK
  capture kp_min --stage talker_ar --kind prefill --batch 1 --prompt-tokens 1 --max-tokens 8 --label ${L}_min $STACK
  capture kp_2k --stage talker_ar --kind prefill --batch 1 --prompt-tokens 2000 --max-tokens 8 --label ${L}_2k $STACK
  capture kp_8k --stage talker_ar --kind prefill --batch 1 --prompt-tokens 8000 --max-tokens 8 --label ${L}_8k $STACK
  capture kd_b1 --stage talker_ar --kind decode --batch 1 --warmup 1 --max-tokens 512 --label $L $STACK
  capture kd_b32 --stage talker_ar --kind decode --batch 32 --warmup 1 --max-tokens 512 --label $L $STACK
elif [ "$MODE" = speech ]; then
  # thinker decode of requests that also ask for audio, against the same text-only batch
  for pass in 1 2; do
    capture uncaptured_tds_b16_$pass --stage thinker --kind decode --batch 16 --speech --max-tokens 512 --label uncaptured$pass --no-capture
    capture uncaptured_td_b16_$pass --stage thinker --kind decode --batch 16 --max-tokens 512 --label uncaptured$pass --no-capture
  done
  capture tds_b4 --stage thinker --kind decode --batch 4 --speech --max-tokens 512 --label $L
  capture tds_b16 --stage thinker --kind decode --batch 16 --speech --max-tokens 512 --label $L
  capture td_b16 --stage thinker --kind decode --batch 16 --max-tokens 512 --label $L
else
  if [ "$MODE" = formal ]; then
    capture uncaptured_td_b16_1 --stage thinker --kind decode --batch 16 --label uncaptured1 --no-capture
    capture uncaptured_td_b1 --stage thinker --kind decode --batch 1 --label uncaptured --no-capture
  fi
  capture td_b1 --stage thinker --kind decode --batch 1 --label $L $STACK
  capture td_b16 --stage thinker --kind decode --batch 16 --label $L $STACK
  capture tp_b1 --stage thinker --kind prefill --batch 1 --label $L $STACK
  capture kd_b1 --stage talker_ar --kind decode --batch 1 --warmup 1 --label $L $STACK
  capture kd_b16 --stage talker_ar --kind decode --batch 16 --warmup 1 --label $L $STACK
  capture kp_b1 --stage talker_ar --kind prefill --batch 1 --max-tokens 8 --label $L $STACK
  if [ "$MODE" = formal ]; then
    capture uncaptured_td_b16_2 --stage thinker --kind decode --batch 16 --label uncaptured2 --no-capture
  fi
fi

grep -m1 "Torch profiler armed" "$OUT/serve.log" > "$OUT/armed_marker.txt"
teardown

for trace in "$OUT"/"$L"*/*/b*/*.trace.json.gz; do
  [ -e "$trace" ] || continue
  python3 "$S/step_ledger.py" "$trace" --top 30 > "${trace%.trace.json.gz}.ledger.txt" 2>&1
done
echo "done $(date +%T)" >> "$OUT/progress.txt"
