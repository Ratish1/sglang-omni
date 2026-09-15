# Runbook 03: MOSS-TTS Local reference encode, where the time goes

Written 2026-09-15. Tree: upstream main `b29084bfa`; its preprocessing path is identical to
`442e559b4` (the only MOSS changes between them are vocoder renames), so readout 02's event data
describes it.

## What it answers

Readout 02 puts about 80 percent of MOSS-TTS Local first audio variance in the preprocessing stage,
and its duration grows with the number of requests preprocessing at once. The code, read at
`b29084bfa`:

```text
preprocessing thread (one of 16, SimpleScheduler, simple_scheduler.py:277-293)
  preprocess_moss_tts_local_payload        moss_tts_local/request_builders.py:299
    _MossLocalReferenceEncoder.encode       moss_tts_local/stages.py:508
      input_key: torchaudio.info            stages.py:481
                 reference_path_cache_key   preprocessing/cache_key.py:125 (stat, 3 x 8 KiB, full read on memo miss)
      _BatchedReferenceEncoder.encode       stages.py:321
        torchaudio.info (again)             stages.py:324
        queue.put, future.result            stages.py:326-327
single worker thread moss-local-ref-encode  stages.py:281
  _drain_batch: first job, +4 ms per more, up to 8   stages.py:339-351
  _encode_batch                                       stages.py:372
    load_paths: torchaudio.load + resample, one file after another   audio_tokenizer.py:1559-1575
    encode_waveforms: _prepare_waveform per item (mono, loudness, H2D) audio_tokenizer.py:1597-1600
                      batch_encode, zero padded to the longest item   audio_tokenizer.py:1453-1463, :1491
                      codes and lengths to CPU                         audio_tokenizer.py:1612-1613
  future.set_result per job                           stages.py:357-370
```

This runbook measures each line of that path in isolation (part 2) and under the live c16 load with
the AR engine in the same process (part 3).

## 0. Tree and host

```bash
cd /sgl-workspace/sglang-omni
git fetch https://github.com/sgl-project/sglang-omni.git main
git fetch https://github.com/Ratish1/sglang-omni.git analysis/cosyvoice-utilization-20260912
ANALYSIS=$(git rev-parse FETCH_HEAD)
git -C /sgl-workspace/wt/cosyvoice-analysis checkout --detach "$ANALYSIS"
git worktree add --detach /sgl-workspace/wt/stream-main b29084bfa5b80779ea8387f3da951b5b0c506c8c

T=/sgl-workspace/wt/cosyvoice-analysis/tasks/streaming_scheduler_first_audio_20260915
OUT=/sgl-workspace/wt/cosyvoice-analysis/artifacts/moss_tts_local/ref-encode-profile-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
cd /sgl-workspace/wt/stream-main
git rev-parse HEAD > "$OUT/head.txt"
python -c "import sglang_omni; print(sglang_omni.__file__)" > "$OUT/import_path.txt"
{ nproc; lscpu; cat /proc/self/cgroup; cat /sys/fs/cgroup/cpu.max 2>/dev/null;
  python -c "from sglang_omni.utils.cpu import effective_cpu_count; print('effective_cpu_count', effective_cpu_count())";
  py-spy --version; nvidia-smi; } > "$OUT/host.txt" 2>&1
```

`import_path.txt` must be under `/sgl-workspace/wt/stream-main`.

## 1. Nothing else on GPU 0

```bash
nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv
```

must list no process before part 2.

## 2. Offline: every step of one encode, batches, and bursts of 16

```bash
cd /sgl-workspace/wt/stream-main
CUDA_VISIBLE_DEVICES=0 python "$T/scripts/moss_reference_encode_timing.py" \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 2>&1 | tee "$OUT/encode_timing.txt"
```

Section 1 times each line for one reference at a time; section 2 runs `load_paths` and
`encode_waveforms` at batch 1, 2, 4 and 8; section 3 sends 16 simultaneous requests through the
production cache service and batch worker for 16 rounds and reports each request's latency and the
worker's own load and encode time per batch.

## 3. Live: the default c16 benchmark with the recorder and py-spy on the pipeline process

```bash
cd /sgl-workspace/wt/stream-main
setsid bash -c "echo \$\$ > '$OUT/server.pgid'; exec python -m sglang_omni.cli serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --port 8000 --host localhost \
  --tts_engine.engine.max_running_requests 64 --tts_engine.engine.cuda_graph_max_bs 64" \
  > "$OUT/serve.log" 2>&1 &
until curl -s http://localhost:8000/health | grep -q healthy; do sleep 5; done

for p in $(pgrep -g "$(cat "$OUT/server.pgid")"); do
  py-spy dump --pid "$p" 2>/dev/null | grep -q moss-local-ref-encode && echo "$p"
done > "$OUT/pipeline_pid.txt"
PIPE=$(head -1 "$OUT/pipeline_pid.txt")
echo "pipeline pid $PIPE"

mpstat 1 > "$OUT/mpstat.log" 2>&1 &
MPSTAT=$!
pidstat -t -u -p "$PIPE" 1 > "$OUT/pidstat_pipeline.log" 2>&1 &
PIDSTAT=$!
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5 > "$OUT/gpus_all.csv" &
GPU_LOG=$!

curl -s -X POST http://localhost:8000/start_request_profile -H 'Content-Type: application/json' \
  -d "{\"run_id\":\"ref-encode\",\"event_dir\":\"$OUT/events\"}"
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --stream \
  --use-existing-server --generate-only \
  --output-dir "$OUT/bench" > "$OUT/bench.log" 2>&1 &
BENCH=$!
until [ "$(cat "$OUT"/events/events_coordinator_*.jsonl 2>/dev/null | wc -l)" -gt 3000 ]; do sleep 1; done
py-spy record --pid "$PIPE" --threads --idle --native --rate 100 --duration 25 \
  --format raw --output "$OUT/pyspy_wall.txt"
py-spy record --pid "$PIPE" --threads --gil --rate 100 --duration 25 \
  --format raw --output "$OUT/pyspy_gil.txt"
wait "$BENCH"
curl -s -X POST http://localhost:8000/stop_request_profile -H 'Content-Type: application/json' \
  -d '{"run_id":"ref-encode"}'
kill "$MPSTAT" "$PIDSTAT" "$GPU_LOG"
kill -TERM -- -"$(cat "$OUT/server.pgid")"

python "$T/scripts/pyspy_thread_top.py" "$OUT/pyspy_wall.txt" > "$OUT/pyspy_wall_top.txt"
python "$T/scripts/pyspy_thread_top.py" "$OUT/pyspy_gil.txt" > "$OUT/pyspy_gil_top.txt"
```

`pipeline_pid.txt` must hold one pid. The captures start once the coordinator has logged about 150
requests, past the warmup burst, and run 25 s each back to back. `--idle` keeps samples of threads waiting on queues and
locks, so the worker's share split between `_drain_batch`, `load_paths` and `encode_waveforms`
reads as wall time; the `--gil` capture shows which thread holds the GIL.

## 4. Return

```bash
cd /sgl-workspace/wt/cosyvoice-analysis
tar --exclude='*.wav' -czf "$(basename "$OUT").tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
```

Copy the archive into the local checkout's `artifacts/`.

## 5. Readout

`../readouts/03_moss_tts_local_reference_encode_<date>.md`: the line that carries the preprocessing
time at c16 (file read, resample, loudness, H2D, forward, D2H, batch wait, or GIL), with the offline
step table, the burst latency by completion rank, the worker's load and encode split, the py-spy
thread shares, and provenance.
