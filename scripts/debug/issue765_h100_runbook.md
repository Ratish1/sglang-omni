# Issue 765 H100 Profiling Runbook

This runbook profiles Video-AMME Talker failures without changing the test
contract. It collects:

- request-level JSONL events from every coordinator/stage process
- torch profiler Chrome traces with named scheduler/code2wav regions
- GPU memory/utilization samples
- per-process GPU memory samples
- benchmark results and router worker snapshots
- git, CUDA, torch, and environment metadata

## One-Time Setup

```bash
cd /path/to/sglang-omni
export PYTHONPATH=.
python -m benchmarks.dataset.prepare --dataset videoamme-ci-50
```

For small kernel traces, set these before starting the server. Do not use all
of them for a full 20-sample run unless multi-GB traces are acceptable.

```bash
export SGLANG_TORCH_PROFILER_RECORD_SHAPES=1
export SGLANG_TORCH_PROFILER_PROFILE_MEMORY=1
# Optional and much larger:
# export SGLANG_TORCH_PROFILER_WITH_STACK=1
# export SGLANG_TORCH_PROFILER_WITH_FLOPS=1
```

## Stage 10: Router-Backed Talker Path

Create the same shape as the CI managed-router fixture: two complete colocated
speech workers behind the external router.

```bash
cat >/tmp/issue765_stage10_launcher.yaml <<'YAML'
launcher:
  backend: local
  model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct
  model_name: qwen3-omni
  num_workers: 2
  num_gpus_per_worker: 1
  worker_host: 127.0.0.1
  worker_base_port: 8011
  worker_extra_args: "--config examples/configs/qwen3_omni_colocated_h20.yaml --colocate --stages.0.factory-args.thinker-max-seq-len 32768 --stages.4.factory-args.thinker-max-seq-len 32768"
  wait_timeout: 180
YAML

CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_TORCH_PROFILER_DIR=/tmp/issue765_profiles/stage10_server \
python -m sglang_omni_router.serve \
  --host 0.0.0.0 \
  --port 8000 \
  --launcher-config /tmp/issue765_stage10_launcher.yaml \
  --policy least_request \
  --health-success-threshold 1 \
  --health-failure-threshold 2 \
  --health-check-interval-secs 2 \
  --log-level info 2>&1 | tee /tmp/issue765_stage10_router.log
```

Full request-level run, low overhead:

```bash
python scripts/debug/profile_videoamme.py \
  --scenario stage10-router \
  --traffic-base-url http://127.0.0.1:8000 \
  --discover-router-workers \
  --request-events-only \
  --max-samples 20 \
  --max-concurrency 16 \
  --disable-tqdm
```

Small kernel-level run, high detail:

```bash
python scripts/debug/profile_videoamme.py \
  --scenario stage10-router \
  --traffic-base-url http://127.0.0.1:8000 \
  --discover-router-workers \
  --max-samples 4 \
  --max-concurrency 16 \
  --disable-tqdm
```

## Stage 11: FP8 Thinker TP=2 + Talker/Code2Wav

This starts the same topology as the TP2 CI fixture.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_TORCH_PROFILER_DIR=/tmp/issue765_profiles/stage11_server \
python examples/run_qwen3_omni_speech_server.py \
  --model-path marksverdhei/Qwen3-Omni-30B-A3B-FP8 \
  --port 8000 \
  --model-name qwen3-omni \
  --thinker-max-seq-len 32768 \
  --thinker-tp-size 2 \
  --gpu-thinker-tp 0,1 \
  --gpu-talker 1 \
  --gpu-code2wav 1 \
  --thinker-mem-fraction-static 0.55 \
  --talker-mem-fraction-static 0.20 2>&1 | tee /tmp/issue765_stage11_server.log
```

Full request-level run, low overhead:

```bash
python scripts/debug/profile_videoamme.py \
  --scenario stage11-tp2 \
  --traffic-base-url http://127.0.0.1:8000 \
  --profile-base-url http://127.0.0.1:8000 \
  --request-events-only \
  --max-samples 10 \
  --max-concurrency 16 \
  --disable-tqdm
```

Small kernel-level run, high detail:

```bash
python scripts/debug/profile_videoamme.py \
  --scenario stage11-tp2 \
  --traffic-base-url http://127.0.0.1:8000 \
  --profile-base-url http://127.0.0.1:8000 \
  --max-samples 4 \
  --max-concurrency 16 \
  --disable-tqdm
```

## Export

Each harness invocation prints an artifact path like:

```text
/tmp/sglang_omni_issue765_profiles/videoamme_stage11-tp2_YYYYmmdd_HHMMSS
```

Export that directory plus the server/router log:

```bash
tar -czf /tmp/issue765_stage11_profile_bundle.tgz \
  /tmp/sglang_omni_issue765_profiles/videoamme_stage11-tp2_* \
  /tmp/issue765_stage11_server.log
```

For Stage 10, export the router log too:

```bash
tar -czf /tmp/issue765_stage10_profile_bundle.tgz \
  /tmp/sglang_omni_issue765_profiles/videoamme_stage10-router_* \
  /tmp/issue765_stage10_router.log
```

## What To Inspect First

1. `SUMMARY.md`: benchmark success/failure, speed summary, and artifact map.
2. `profiler_report.txt`: stage-level timing and hop timing.
3. `router_workers_after.json`: Stage 10 routing distribution and failures.
4. `nvidia_smi/compute_apps.csv`: which process owns the memory spike.
5. `events/*.jsonl`: first missing terminal/stage event for failed requests.
6. `traces/**/*.trace.json.gz`: open in Perfetto and search for:
   - `omni_scheduler_run_batch:stage=thinker`
   - `omni_scheduler_run_batch:stage=talker_ar`
   - `code2wav_decode`

## Questions The Profile Must Answer

- Does the request reach `scheduler_queue_enter` for thinker and talker?
- Does the request reach `scheduler_prefill_start` for thinker and talker?
- Does talker emit the first code chunk before the failure?
- Does code2wav receive chunks and enter `code2wav_decode_start`?
- Which GPU/process reaches peak memory first?
- Is Stage 10 failure tied to one worker, one GPU, or both workers equally?
- Is Stage 11 failure in thinker prefill, talker prefill/decode, or code2wav?
