# Runbook 01: MOSS-TTS Local streaming c16, B only, first audio by hop

Written 2026-09-15. B is compared with the fixed A in `../baselines/stream_en_c16.json`; A is not
run. Protocol: guide sections 1, 5.1 and 6 (unseeded, warmup 1, full English corpus, one boot).

## What it answers

Where B's first audio time goes, hop by hop, using the request event recorder that already ships in
upstream main. The recorder writes `<event_dir>/events_<stage>_<pid>.jsonl` per process and is a
no-op while stopped (sglang_omni/profiler/event_recorder.py:101-139, :187-189). The coordinator
starts it and broadcasts the start to every stage process, including MOSS-TTS Local's separate
vocoder process (sglang_omni/serve/launcher.py:318-348, sglang_omni/pipeline/stage/runtime.py:1889-1900).

```text
coordinator.request_admission            coordinator.py:467
preprocessing.stage_dispatch             runtime.py:963
preprocessing.stage_complete             runtime.py:1297
tts_engine.stage_input_received          runtime.py:460
tts_engine.scheduler_request_build_*     omni_scheduler.py:962, :968
tts_engine.scheduler_queue_enter         omni_scheduler.py:1245
tts_engine.scheduler_prefill_start/end   omni_scheduler.py:1586, :1617
tts_engine.scheduler_first_emit          omni_scheduler.py:1498
tts_engine.stage_stream_chunk_sent       runtime.py:1576-1685
vocoder.stage_stream_chunk_received      runtime.py:746   (then scheduler inbox put, :952)
vocoder.stage_stream_chunk_sent          runtime.py:1798  (after the outbox drain, :1109)
coordinator.stage_stream_chunk_received  coordinator.py:740
```

The scheduler code that differs from A runs between `vocoder.stage_stream_chunk_received` and
`vocoder.stage_stream_chunk_sent`: inbox wait, `StreamingSimpleScheduler.start`, the chunk
collector, `on_stream_chunk_batch`, the pump and the step, the outbox put and its drain.

Expected from the code, stated before the run: the AR stage sends 1 frame, then 5 frames per message
(sglang_omni/models/moss_tts_local/model_runner.py:57, request_builders.py:98), and the vocoder's first
threshold is 5 frames (stages.py:652, streaming_vocoder.py:689-703). A request's first audio is
therefore emitted after its second code chunk, so "generated code chunks received before first
audio" reads 2 at p50, and the vocoder segment includes one AR cadence of 5 frames.

## 0. Tree on the box

```bash
git fetch https://github.com/Ratish1/sglang-omni.git analysis/cosyvoice-utilization-20260912
git worktree add /sgl-workspace/wt/cosyvoice-analysis FETCH_HEAD
cd /sgl-workspace/wt/cosyvoice-analysis
git diff --quiet 442e559b40b5040965ec876650b32da05d31769f HEAD -- . ':!tasks' \
  && echo "runtime tree equals 442e559b4" || echo "STOP: runtime tree differs"
```

## 1. Boot

```bash
T=tasks/streaming_scheduler_first_audio_20260915
BOOT=artifacts/moss_tts_local/b-stream-en-c16-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BOOT"
git rev-parse HEAD > "$BOOT/head.txt"
git diff --quiet 442e559b40b5040965ec876650b32da05d31769f HEAD -- . ':!tasks'; echo $? > "$BOOT/runtime_tree_diff_exit.txt"
python -c "import sglang_omni; print(sglang_omni.__file__)" > "$BOOT/import_path.txt"
nvidia-smi > "$BOOT/gpus_before.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv >> "$BOOT/gpus_before.txt"
nvidia-smi dmon -i 0 -s pucv -d 1 > "$BOOT/dmon.log" &
CUDA_VISIBLE_DEVICES=0 python -m sglang_omni.cli serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --port 8000 2>&1 | tee "$BOOT/serve.log"
```

The boot is void unless `runtime_tree_diff_exit.txt` is 0 and `import_path.txt` is under
`/sgl-workspace/wt/cosyvoice-analysis`. Use the GPU the A run used; record the other GPUs, never
require them idle.

## 2. Pass 1, the census, recorder off

From the same worktree in a second shell, after the server is ready. The client flags must equal the
A run's (README validation item 1); `--ref-format references --token-count auto` are the MOSS-TTS
Local flags in the benchmark's own usage block (benchmarks/eval/benchmark_tts_seedtts.py:47-53).

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --lang en \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --ref-format references --token-count auto \
  --use-existing-server --host 127.0.0.1 --port 8000 \
  --concurrency 16 --warmup 1 --stream \
  --generate-only --output-dir "$BOOT/pass1"
SPEED1="$BOOT/pass1/speed_results.json"
python "$T/scripts/compare_to_a.py" --model moss_tts_local --speed-results "$SPEED1" \
  | tee "$BOOT/pass1/compare_to_a.md"
```

`--warmup` defaults to the concurrency (benchmark_tts_seedtts.py:854-859), so it is passed explicitly.

## 3. Pass 2, request event recorder on, same server

```bash
EVENTS="$PWD/$BOOT/pass2/events"
curl -s -X POST http://127.0.0.1:8000/start_request_profile -H 'Content-Type: application/json' \
  -d "{\"run_id\":\"pass2\",\"event_dir\":\"$EVENTS\"}"
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --lang en \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --ref-format references --token-count auto \
  --use-existing-server --host 127.0.0.1 --port 8000 \
  --concurrency 16 --warmup 1 --stream \
  --generate-only --output-dir "$BOOT/pass2"
curl -s -X POST http://127.0.0.1:8000/stop_request_profile -H 'Content-Type: application/json' \
  -d '{"run_id":"pass2"}'
ls "$EVENTS"
```

`ls` must show one file per process: coordinator, pipeline (preprocessing and tts_engine) and
vocoder. Pass 2 pays the recorder's JSON writes under a lock (event_recorder.py:191-210), so its
speed numbers are never compared with A; pass 1 is the census.

## 4. The breakdown

```bash
SPEED2="$BOOT/pass2/speed_results.json"
python "$T/scripts/first_chunk_anatomy.py" "B=$EVENTS:$SPEED2" | tee "$BOOT/pass2/anatomy.txt"
```

Read, in order:

1. The segments table (p50 / p95 ms per hop). The two vocoder rows are
   `tts_engine.stage_stream_chunk_sent -> vocoder.stage_stream_chunk_received` (transport from the
   pipeline process to the vocoder process) and
   `vocoder.stage_stream_chunk_received -> vocoder.stage_stream_chunk_sent` (everything the changed
   scheduler code runs).
2. `generated code chunks received before first audio` (expected 2) and the talker code chunk cadence;
   the vocoder segment minus one cadence is the vocoder's own share.
3. The bootstrap and prefill tables were written for Qwen3-TTS's initial worker; for MOSS-TTS Local
   read them only as overlap counts.

## 5. Return

Without WAVs, into the local checkout's `artifacts/moss_tts_local/<boot>/`: `head.txt`,
`runtime_tree_diff_exit.txt`, `import_path.txt`, `gpus_before.txt`, `dmon.log`, `serve.log`,
`pass1/` (speed_results.json, results.csv, compare_to_a.md), `pass2/` (events, speed_results.json,
anatomy.txt).

## 6. Readout

`../readouts/01_moss_tts_local_stream_c16_<head>_<date>.md`: verdict first (which hop carries the
first audio), then the pass 1 table from `compare_to_a.md`, then the segments table, provenance last.
