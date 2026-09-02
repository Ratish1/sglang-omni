# 15. Step ledger runbook

Branch perf/step-ledger on origin, the ledger of doc 14 section 6.1. One
H100 (two for MiniMax Music 3), the CI image, the checkout installed
editable. No flag: the ledger records whenever a request profile run is
active, and writes one JSON per stage process when the run stops. The
first run on 79cc7c70a is read in doc 16, and the protocol below is the
one corrected by that read: it needs 7b1bee5c4 or later, which charges
cycles to the right step and adds the `forward_ms` and `cached_tokens`
columns. Since cec7b6b11 the branch sits on upstream main 15c4568bb, so
the next run measures the merged predictor, head and sampler fusions
(doc 18 section 1), and every run from now on uses that head or later.

MiniMax Music 3 must be launched in the foreground under a terminal
multiplexer: four detached launches died with exit code -9 in the
dit_dav process during startup and a foreground launch succeeded
(doc 17 section 8.3).

## 0. Before the first server

```
cd $OMNI_ROOT && git fetch origin && git checkout perf/step-ledger
python -m pip install -e "$OMNI_ROOT" --no-deps
cd / && python -c 'import sglang_omni, os; print(os.path.dirname(os.path.dirname(sglang_omni.__file__)))'   # must print $OMNI_ROOT
export OUT=/data/ledger-$(date +%Y%m%d)   # outside the tree
mkdir -p $OUT
```

Every server below runs alone on its GPU. Kill leftovers first with
`bash .github/scripts/delete_gpu_process.sh --kill-orphans`.

## 1. The protocol, the same for every model

One fresh server per model and per concurrency point, one profiled pass
with the benchmark's default warmup. Boot the server, run the point,
stop the server, then the next point. The second run (doc 17 section
1.3) showed why a server cannot be reused across points: the radix cache
and the speaker artifact cache both hold the whole corpus after the
first point, so every later point measures cache hits, and the cache
state also changes what the pool admits. A fresh server per point is
what CI does, so the numbers match CI's cold and warm mix exactly (C
warm of 50 at concurrency C).

```
curl -s -X POST http://127.0.0.1:$PORT/start_request_profile \
  -H 'Content-Type: application/json' \
  -d "{\"run_id\":\"${MODEL}_c${C}\",\"event_dir\":\"$OUT/${MODEL}_c${C}\"}"
<the benchmark at C, warmup left at its default>
curl -s -X POST http://127.0.0.1:$PORT/stop_request_profile \
  -H 'Content-Type: application/json' -d "{\"run_id\":\"${MODEL}_c${C}\"}"
sleep 3
ls $OUT/${MODEL}_c${C}/step_ledger_*.json
grep step_ledger $OUT/logs/serve_${MODEL}.log | tail -20
```

No separate warm pass. The first run did one, on the same prompts, and
the radix cache turned every profiled prefill of Qwen3-TTS, Qwen3-ASR,
MOSS-TD, MOSS-TTS-Local, MOSS-TTS Delay and Higgs into a one token extend
on a full prefix hit (doc 16 section 1). The benchmark's own warmup sends
C requests first (`benchmarks/benchmarker/runner.py:23`), the same
protocol CI runs, so at most C of the profiled prefills are hits and the
`cached_tokens` column shows which rows they are. Backbone CUDA graphs are
captured at server start. Model owned graphs are not all captured at
start (the Qwen3-TTS predictor graph is captured per batch bucket on
first use, doc 17 section 4), so the first steps of a window carry first
use costs in the ledger's max and in the request view's mean. Read the
p50 columns of both, the view prints one since fa5ee9631.

Each stage process writes `step_ledger_<stage>_<pid>.json` and logs one line
per batch shape. The request view of the same window comes from
`python -m sglang_omni.profiler $OUT/${MODEL}_c${C} --format table`.
Before stopping the server, also keep its `/model_info`: since 5a0a6a6e2
it lists, under `prefill_cuda_graph`, the replay buckets and the token
count and batch size of every prefill that ran eager, which is what
attributes an extend row whose `graph_share` is below one.

```
curl -s http://127.0.0.1:$PORT/model_info > $OUT/${MODEL}_c${C}/model_info.json
```

Keep per model and concurrency: the JSON files, the server log, and the
benchmark's own results directory.

## 2. Why c1 and the CI concurrency are two different rows

At c1 every step carries one row, so the ledger reads the step's fixed
cost: host per step with nothing to amortize, the wait when the device is
slower than the host, the prefill's fixed host cost, and the idle sleeps
between arrivals. That is the latency floor of the loop.

At the CI concurrency the same columns say how host time grows with rows
(the per request Python of doc 14 section 5.1), whether the graph still
replays at the realized batch sizes, whether the pool or the row limit caps
the rows, and how much of each cycle the device idles. That is the
throughput picture. A stage can be device bound at c1 and host bound at
c16, so both rows go in the fleet table and neither replaces the other.

## 3. Per model

Ports and logs: one port per model, `$OUT/logs/serve_<model>.log`. Launch
with `setsid ... > $OUT/logs/serve_<model>.log 2>&1 &` and wait for
`curl -sf http://127.0.0.1:$PORT/health`.

### 3.1 Qwen3-Omni, bf16 colocated H100 (the CI TTS topology on one worker)

Concurrencies: 1, 16 (CI), 32 (the talker's row limit).

```
CUDA_VISIBLE_DEVICES=0 sgl-omni serve --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model-name qwen3-omni --config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate \
  --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768 \
  --host 127.0.0.1 --port 31000

python -m benchmarks.eval.benchmark_omni_seedtts --generate-only --voice-clone \
  --meta zhaochenyang20/seed-tts-eval-50-arrow --model qwen3-omni --port 31000 \
  --max-concurrency $C --max-samples 50 --warmup 0 --output-dir $OUT/omni_c${C}_bench
```

Expected ledger files per window: thinker and talker_ar. code2wav runs its
own vocoder scheduler (components/code2wav_scheduler.py), not an
OmniScheduler, so it has no ledger, and the same holds for every vocoder
stage of the TTS models below. The encoders and preprocessing carry no AR
scheduler either.

### 3.2 Qwen3-TTS 1.7B (the CI preset)

Concurrencies: 1, 16 (CI), 32. At 64 the first run lost 32 of 50 requests
to `The request queue is full`: the engine default `max_queued_requests`
is 16 (`qwen3_tts/engine_builder.py:46`) and the benchmark keeps 64 in
flight. The launch below raises it. Read 64 as a capacity point, not a CI
point.

```
CUDA_VISIBLE_DEVICES=0 sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --tts_engine.engine.max_running_requests 64 --tts_engine.engine.max_queued_requests 64 \
  --tts_engine.engine.cuda_graph_max_bs 64 \
  --tts_engine.engine.torch_compile_max_bs 64 --vocoder.process vocoder \
  --tts_engine.gpu_memory_fraction 0.85 --vocoder.gpu_memory_fraction 0.10 \
  --port 31001

python -m benchmarks.eval.benchmark_tts_seedtts --generate-only --use-existing-server \
  --meta zhaochenyang20/seed-tts-eval-50-arrow --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --ref-format references --port 31001 --max-concurrency $C --warmup 0 \
  --output-dir $OUT/qwen3tts_c${C}_bench
```

### 3.3 Qwen3-ASR 1.7B

Concurrencies: 1, 8, 32 (CI). The benchmark talks to the running server.

```
CUDA_VISIBLE_DEVICES=0 sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B \
  --model-name Qwen/Qwen3-ASR-1.7B --port 31002

python -m benchmarks.eval.benchmark_asr_seedtts --port 31002 --lang en \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --concurrencies $C --repeats 1 --output $OUT/qwen3asr_c${C}.json
```

The warm pass is the same command with `--warmup`. The full corpus is 1088
clips, about 65 s at c1 and about 6 s at c32.

### 3.4 MOSS-TTS-Local (the CI preset)

Concurrencies: 1, 16 (CI).

```
CUDA_VISIBLE_DEVICES=0 sgl-omni serve --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --port 31003

python -m benchmarks.eval.benchmark_tts_seedtts --generate-only --use-existing-server \
  --meta zhaochenyang20/seed-tts-eval-50-arrow --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --ref-format references --token-count auto --port 31003 --max-concurrency $C --warmup 0 \
  --output-dir $OUT/mosslocal_c${C}_bench
```

### 3.5 MOSS-TTS (Delay)

Concurrencies: 1, 8 (the cookbook's serving point).

```
CUDA_VISIBLE_DEVICES=0 sgl-omni serve --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --config examples/configs/moss_tts.yaml --port 31004

python -m benchmarks.eval.benchmark_tts_seedtts --generate-only --use-existing-server \
  --meta zhaochenyang20/seed-tts-eval-50-arrow --model OpenMOSS-Team/MOSS-TTS-v1.5 \
  --ref-format references --token-count auto --port 31004 --max-concurrency $C --warmup 0 \
  --output-dir $OUT/mossdelay_c${C}_bench
```

### 3.6 MOSS-Transcribe-Diarize

Concurrencies: 1, 16 (the cookbook's serving point and row limit).

```
CUDA_VISIBLE_DEVICES=0 sgl-omni serve --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --asr.engine.max_running_requests 16 --asr.engine.cuda_graph_max_bs 16 \
  --mem-fraction-static 0.80 --port 31005

python -m benchmarks.eval.benchmark_asr_transcribe_diarize --use-existing-server --port 31005 \
  --dataset movies800times --concurrency $C --max-samples 100 --warmup 0 \
  --output-dir $OUT/mosstd_c${C}_bench
```

### 3.7 Higgs TTS (the CI preset)

Concurrencies: 1, 16 (CI).

```
CUDA_VISIBLE_DEVICES=0 sgl-omni serve --model-path bosonai/higgs-tts-3-4b --port 31006

python -m benchmarks.eval.benchmark_tts_seedtts --generate-only --use-existing-server \
  --meta zhaochenyang20/seed-tts-eval-50-arrow --model bosonai/higgs-tts-3-4b \
  --port 31006 --max-concurrency $C --warmup 0 --output-dir $OUT/higgs_c${C}_bench
```

### 3.8 MiniMax Music 3

Concurrencies: 1 (the five cookbook requests one at a time) and 5 (the same
five at once). Guidance doubles the rows, so the ledger shows rows 2 and
rows 10. One profile window covers both passes of the script, the rows
column separates them. The radix cache is off on this engine
(`minimax_music3/engine_builder.py:56`), so a warm pass does not touch its
prefills. The script computes its wall times with python, the CI image
has no `bc`.

```
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 31007

# warm pass, unprofiled
PORT=31007 OUT=$OUT/minimax_warm bash tasks/qwen3_omni_0518_numerics/scripts/minimax_cookbook_ab.sh
# profile window
curl ... start_request_profile run_id minimax event_dir $OUT/minimax
PORT=31007 OUT=$OUT/minimax_bench bash tasks/qwen3_omni_0518_numerics/scripts/minimax_cookbook_ab.sh
curl ... stop_request_profile
```

## 4. Reading a window

For each stage file, the rows to read first, per (mode, rows). Cycle,
idle sleeps and allocations belong to the step whose launch opened the
interval, so a prefill row's cycle is the prefill's own.

- `cycle_ms` p50 against `gpu_span_ms` p50: the gap is the device idle
  floor after the step.
- `wait_ms` p50: near zero means the host never waits for the device,
  the stage is host bound, on runners with no synchronize inside the step
  (table below). Large means device bound.
- `forward_ms` against `gpu_span_ms`: the model forward's own device time
  against the whole step's. For a graph replay `forward_ms` is exact. The
  rest of the span is hooks, sampling, publish and any host starvation.
  `host_ms` minus `forward_ms` bounds the exposed host on a runner whose
  wait is zero by construction.
- `host_ms` at rows 1 against rows 16: the growth is the per row host work.
- `graph_share`: below 1.0 at a steady batch size means eager steps. On
  extend rows it is the prefill graph, which sglang refuses when the
  bucket would pad past twice the token count.
- extend rows: `extend_tokens` p50 equal to rows means cache hits, read
  `cached_tokens`. On real prefills `host_ms` minus `forward_ms` is the
  fixed host cost that coalescing amortizes and `extend_tokens` p50 the
  batch it was paid for.
- `idle_sleeps_per_step` at c1: the loop waiting for input, for the talker
  that is the thinker's cadence.
- `allocations_per_step` above a handful in steady decode: tensor churn
  inside the step.

The request view's `scheduler_queue_enter->scheduler_prefill_start` is
the admission wait. On the first run it was the largest single interval
of Qwen3-Omni at c16 and c32 (doc 16 section 5.1).

Per runner, what the columns cannot see, from the second review of the
instrument (verified against the runner code):

| runner | read with care |
|---|---|
| Qwen3-Omni talker | the cleanest: wait is the staged token event, span brackets the code predictor. Cycle and idle sleeps at c1 follow the thinker's cadence, so read host and span for the talker's own step |
| Qwen3-Omni thinker | sync path for audio output: wait is the end event wait before the token copy. Prefill with images or video reports graph false by construction (custom forward) |
| Qwen3-TTS | prefill reports graph false by construction, the code predictor graph is not represented |
| Qwen3-ASR, MOSS-Transcribe-Diarize | the lookahead path is the one configuration where every column means exactly what the table says. Allocations include the encoder thread in the same process |
| Higgs | sync path below the lookahead threshold: its blocking collect copy is inside host, wait shows only the end event part |
| MOSS-TTS-Local | frame decode may run eager while the backbone reports graph true |
| MOSS-TTS Delay | about ten host synchronizations inside its sampler each step, so span includes host stalls and wait understates. The sync contract test names them |
| MiniMax Music 3 | one explicit stream synchronize and one token list copy inside post decode each step, same effect as MOSS-TTS Delay |

What to send back per model: the ledger JSON files, the benchmark summary
line per concurrency, and the server log. The fleet table (doc 14 section
6.4) is filled from the JSON files directly.
