# Runbook 01: MOSS-TTS Local streaming c16, B only, first audio by hop

Written 2026-09-15. B is compared with the fixed A in `../baselines/stream_en_c16.json`; A is not
run again. The benchmark runs with its default settings, the same way the A runs were taken.

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

The streaming scheduler refactor that landed on main after `442e559b4` only renames these paths for
MOSS-TTS Local (base scheduler loop, base vocoder pump, MOSS-TTS Local vocoder), so this tree still
carries the regression as measured.

## 0. Tree on the box

```bash
git fetch https://github.com/Ratish1/sglang-omni.git analysis/cosyvoice-utilization-20260912
git worktree add /sgl-workspace/wt/cosyvoice-analysis FETCH_HEAD
cd /sgl-workspace/wt/cosyvoice-analysis
git diff --quiet 442e559b40b5040965ec876650b32da05d31769f HEAD -- . ':!tasks' \
  && echo "runtime tree equals 442e559b4" || echo "STOP: runtime tree differs"
```

Run everything below from this worktree, on the GPU you use for your A runs.

## 1. Pass 1, the census, benchmark defaults

The benchmark starts the server itself, generates, stops it, then runs the ASR phase.

```bash
T=tasks/streaming_scheduler_first_audio_20260915
BOOT=artifacts/moss_tts_local/b-stream-en-c16-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BOOT"
git rev-parse HEAD > "$BOOT/head.txt"
python -c "import sglang_omni; print(sglang_omni.__file__)" > "$BOOT/import_path.txt"
nvidia-smi > "$BOOT/gpus_before.txt"
nvidia-smi dmon -s pucv -d 1 > "$BOOT/dmon_pass1.log" &
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --stream \
  --output-dir "$BOOT/pass1" 2>&1 | tee "$BOOT/pass1.log"
kill %1
python "$T/scripts/compare_to_a.py" --model moss_tts_local \
  --speed-results "$BOOT/pass1/speed_results.json" | tee "$BOOT/pass1/compare_to_a.md"
```

`import_path.txt` must be under `/sgl-workspace/wt/cosyvoice-analysis`. Pass 1 is the only pass
compared with A.

## 2. Pass 2, request event recorder on

The recorder is switched on through the server's HTTP endpoints, so the server is started by hand
with the exact command the benchmark launched in pass 1, and the benchmark attaches to it.

```bash
grep -m1 "Starting server:" "$BOOT/pass1.log" | sed 's/.*Starting server: //' > "$BOOT/pass2_serve_cmd.txt"
cat "$BOOT/pass2_serve_cmd.txt"
nvidia-smi dmon -s pucv -d 1 > "$BOOT/dmon_pass2.log" &
bash -c "$(cat "$BOOT/pass2_serve_cmd.txt")" 2>&1 | tee "$BOOT/pass2_serve.log" &
until curl -s http://localhost:8000/health | grep -q healthy; do sleep 5; done

EVENTS="$PWD/$BOOT/pass2/events"
curl -s -X POST http://localhost:8000/start_request_profile -H 'Content-Type: application/json' \
  -d "{\"run_id\":\"pass2\",\"event_dir\":\"$EVENTS\"}"
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --stream \
  --use-existing-server --generate-only \
  --output-dir "$BOOT/pass2" 2>&1 | tee "$BOOT/pass2.log"
curl -s -X POST http://localhost:8000/stop_request_profile -H 'Content-Type: application/json' \
  -d '{"run_id":"pass2"}'
ls "$EVENTS"
```

Stop the server and `dmon` afterwards. `--use-existing-server` requires `--generate-only`
(benchmark_tts_seedtts.py:1078-1082); everything else stays at its default. `ls` must show one file
per process: coordinator, pipeline (preprocessing and tts_engine) and vocoder. Pass 2 pays the
recorder's JSON writes under a lock (event_recorder.py:191-210), so its speed numbers are never
compared with A.

## 3. The breakdown

```bash
python "$T/scripts/first_chunk_anatomy.py" "B=$EVENTS:$BOOT/pass2/speed_results.json" \
  | tee "$BOOT/pass2/anatomy.txt"
```

Read, in order:

1. The segments table (p50 / p95 ms per hop). The two vocoder rows are
   `tts_engine.stage_stream_chunk_sent -> vocoder.stage_stream_chunk_received` (transport from the
   pipeline process to the vocoder process) and
   `vocoder.stage_stream_chunk_received -> vocoder.stage_stream_chunk_sent` (everything the changed
   scheduler code runs).
2. `generated code chunks received before first audio` and the talker code chunk cadence; the
   vocoder segment minus one cadence is the vocoder's own share.
3. The bootstrap and prefill tables were written for Qwen3-TTS's initial worker; for MOSS-TTS Local
   read them only as overlap counts.

## 4. Return

Without WAVs, into the local checkout's `artifacts/moss_tts_local/<boot>/`: `head.txt`,
`import_path.txt`, `gpus_before.txt`, `dmon_pass1.log`, `dmon_pass2.log`, `pass1.log`,
`pass2_serve_cmd.txt`, `pass2_serve.log`, `pass2.log`, `pass1/` (speed_results.json, the WER
results, compare_to_a.md, server_logs), `pass2/` (events, speed_results.json, anatomy.txt).

## 5. Readout

`../readouts/01_moss_tts_local_stream_c16_<head>_<date>.md`: verdict first (which hop carries the
added first audio), then the pass 1 table from `compare_to_a.md`, then the segments table,
provenance last.
