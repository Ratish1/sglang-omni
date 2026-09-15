# Runbook 02: MOSS-TTS Local streaming c16, paired A/B with the request event recorder

Written 2026-09-15. A is `1f6b6843e`, the parent of the streaming scheduler change; B is `442e559b4`,
the change as merged. The trees differ only in `sglang_omni/scheduling/streaming_simple_scheduler.py`,
`sglang_omni/scheduling/streaming_vocoder.py`, `sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py`
and their tests; the benchmark, the event recorder, the stage runtime and the serve code are identical.
Both arms run back to back on the same GPU with the benchmark at its default settings.

## What it answers

Runbook 01 measured B's first audio by hop but no A profile exists, so it cannot show which hop moved.
This runbook records the same census and the same event recorder pass for A and for B in one sitting,
so every hop and every vocoder timing has a paired delta.

## 0. Trees on the box

```bash
cd /sgl-workspace/sglang-omni
git fetch https://github.com/sgl-project/sglang-omni.git main
git fetch https://github.com/Ratish1/sglang-omni.git analysis/cosyvoice-utilization-20260912
ANALYSIS=$(git rev-parse FETCH_HEAD)
git -C /sgl-workspace/wt/cosyvoice-analysis checkout --detach "$ANALYSIS" 2>/dev/null \
  || git worktree add --detach /sgl-workspace/wt/cosyvoice-analysis "$ANALYSIS"
git worktree add --detach /sgl-workspace/wt/stream-a 1f6b6843ed6accd3e8ee48109445b2e9c23f85a7
git worktree add --detach /sgl-workspace/wt/stream-b 442e559b40b5040965ec876650b32da05d31769f
git -C /sgl-workspace/wt/stream-a rev-parse HEAD
git -C /sgl-workspace/wt/stream-b rev-parse HEAD
```

## 1. One arm

Paste this function once into the shell. It runs pass 1 (the census, the benchmark starts and stops its
own server and runs the ASR phase) and pass 2 (the server started by hand with the command the
benchmark launched in pass 1, the recorder on, the benchmark attached), then the two breakdowns.

```bash
T=/sgl-workspace/wt/cosyvoice-analysis/tasks/streaming_scheduler_first_audio_20260915
GPU=0
OUT=/sgl-workspace/wt/cosyvoice-analysis/artifacts/moss_tts_local/pair-stream-en-c16-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"

run_arm() {
  ARM=$1
  WT=$2
  D="$OUT/$ARM"
  mkdir -p "$D"
  cd "$WT"
  git rev-parse HEAD > "$D/head.txt"
  python -c "import sglang_omni; print(sglang_omni.__file__)" > "$D/import_path.txt"
  nvidia-smi > "$D/gpus_before.txt"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5 > "$D/gpus_all.csv" &
  GPU_LOG=$!

  python -m benchmarks.eval.benchmark_tts_seedtts \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --stream \
    --output-dir "$D/pass1" 2>&1 | tee "$D/pass1.log"

  grep -m1 "Starting server:" "$D/pass1.log" | sed 's/.*Starting server: //' > "$D/pass2_serve_cmd.txt"
  setsid bash -c "echo \$\$ > '$D/pass2_server.pgid'; exec $(cat "$D/pass2_serve_cmd.txt")" \
    > "$D/pass2_serve.log" 2>&1 &
  until curl -s http://localhost:8000/health | grep -q healthy; do sleep 5; done
  curl -s -X POST http://localhost:8000/start_request_profile -H 'Content-Type: application/json' \
    -d "{\"run_id\":\"$ARM\",\"event_dir\":\"$D/pass2/events\"}"
  python -m benchmarks.eval.benchmark_tts_seedtts \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --stream \
    --use-existing-server --generate-only \
    --output-dir "$D/pass2" 2>&1 | tee "$D/pass2.log"
  curl -s -X POST http://localhost:8000/stop_request_profile -H 'Content-Type: application/json' \
    -d "{\"run_id\":\"$ARM\"}"
  kill -TERM -- -"$(cat "$D/pass2_server.pgid")"
  while curl -s http://localhost:8000/health > /dev/null; do sleep 2; done
  while [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1024 ]; do
    sleep 5
  done
  kill "$GPU_LOG"

  python "$T/scripts/first_chunk_anatomy.py" "$ARM=$D/pass2/events:$D/pass2/speed_results.json" \
    > "$D/pass2/anatomy.txt"
  python "$T/scripts/vocoder_event_timings.py" "$D/pass2/events" > "$D/pass2/vocoder_event_timings.txt"
  cd /sgl-workspace/wt/cosyvoice-analysis
}
```

## 2. The pair: A, then B

```bash
run_arm a /sgl-workspace/wt/stream-a
run_arm b /sgl-workspace/wt/stream-b
```

Each arm is void unless its `import_path.txt` is under its own worktree
(`/sgl-workspace/wt/stream-a` or `/sgl-workspace/wt/stream-b`) and `head.txt` is its commit.
`ls "$OUT"/*/pass2/events` must show three files per arm: coordinator, pipeline and vocoder.

## 3. Compare

```bash
python "$T/scripts/compare_to_a.py" --model moss_tts_local \
  --a-speed-results "$OUT/a/pass1/speed_results.json" \
  --speed-results "$OUT/b/pass1/speed_results.json" | tee "$OUT/compare_pair.md"
python "$T/scripts/compare_to_a.py" --model moss_tts_local \
  --speed-results "$OUT/a/pass1/speed_results.json" | tee "$OUT/compare_a_to_stored_a.md"
python "$T/scripts/compare_to_a.py" --model moss_tts_local \
  --speed-results "$OUT/b/pass1/speed_results.json" | tee "$OUT/compare_b_to_stored_a.md"
diff "$OUT/a/pass2/vocoder_event_timings.txt" "$OUT/b/pass2/vocoder_event_timings.txt" > "$OUT/vocoder_event_timings.diff"
diff "$OUT/a/pass2/anatomy.txt" "$OUT/b/pass2/anatomy.txt" > "$OUT/anatomy.diff"
```

If pair 1's req/s delta is within 2 percent, run a second pair in the reverse order, B then A, with a
new `OUT` before reading anything.

## 4. Return

```bash
cd /sgl-workspace/wt/cosyvoice-analysis
tar --exclude='*.wav' -czf "$(basename "$OUT").tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
```

Copy the archive into the local checkout's `artifacts/`.

## 5. Readout

`../readouts/02_moss_tts_local_paired_ab_c16_<date>.md`: verdict first (whether the pair reproduces
the gap and, if it does, which hop and which vocoder timing carries it), then `compare_pair.md`, then
the paired hop table and the paired vocoder timings, provenance last.
