# H100 capture and evaluation runbook

These tools are an initial diagnostic implementation with static checks only. Run them in the H100 serving environment. No local pytest or profiling unit tests are required. Use branch `analysis/cosyvoice-utilization-20260912` and the same Python executable/package paths as the server.

**Scope: English only (`en`), at c1 and c16, for streaming and buffered output.** The current task is baseline collection and small NSYS captures. Full performance A/B begins after an optimization candidate exists; a small annotation-off/on check only qualifies diagnostic overhead.

## 1. Establish identity and a profiler-off baseline

From the Omni checkout:

```bash
DIAG=tasks/cosyvoice_utilization_20260912/diagnostics
mkdir -p artifacts/cosyvoice
python "$DIAG/collect_env.py" --output artifacts/cosyvoice/environment-eager.json
python -m sglang_omni.cli config resolve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --vocoder.factory.enable_dit_torch_compile false \
  --vocoder.factory.enable_flow_estimator_trt false \
  --show config > artifacts/cosyvoice/resolved-eager.yaml
SGLANG_OMNI_PIPELINE_NVTX=0 SGLANG_OMNI_STRICT_PORT=1 \
python -m sglang_omni.cli serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000 \
  --vocoder.factory.enable_dit_torch_compile false \
  --vocoder.factory.enable_flow_estimator_trt false
```

Save the complete launch command and startup log separately. `config resolve` describes pipeline resolution, not every SGLang late-resolved runtime value. Preserve startup attention/sampling backend, graph bucket, placement and KV logs too. Match local checkout SHAs and imported module paths; attach installed checkpoint revision/weight hashes. The collector does not load model weights and cannot prove wheel build identity from a parent Git repository.

The commands above describe the inspected eager defaults; first reproduce the team's actual launch when it differs. GPU “0” may be a remapped CUDA ordinal. Match it to Nsight metric device ID and physical GPU UUID. Save CPU cgroup quota/affinity, GPU clocks/power, other GPU processes, MIG/MPS status, Torch/CUDA/driver/NSYS/ONNX/TRT versions. Use a dedicated measurement interval.

## 2. Full English SeedTTS baseline, profiling disabled

With the server ready, in another terminal:

```bash
DIAG=tasks/cosyvoice_utilization_20260912/diagnostics
COSY_VARIANT=eager
COSY_REPEAT=r1
for COSY_CONCURRENCY in 1 16; do
  for COSY_MODE in streaming buffered; do
    python "$DIAG/run_seedtts.py" --mode "$COSY_MODE" --lang en \
      --concurrency "$COSY_CONCURRENCY" \
      --output "artifacts/cosyvoice/${COSY_VARIANT}-${COSY_MODE}-en-c${COSY_CONCURRENCY}-${COSY_REPEAT}" \
      || exit 1
  done
done
```

**Omitting `--samples` and using offset zero selects the full English split.** Default dataset revision is `27f4c1adee83b5b29b7c4b375f6b976324bda308`; source order is preserved. `inputs.json` hashes each selected target/reference text and reference-audio bytes; `experiment.json` records the ordered identity, complete client configuration and run status. Native outputs are in `measured/`: `speed_results.json`, `generated.json`, `results.csv`, and all WAVs. Fresh output paths are mandatory. The model requires reference audio; `--reference audio` removes only its transcript, while the default uses audio and text.

Default warmup is 32 copies of the first sample. It is excluded from native measured wall time. Preserve this cache policy across A/B; a warm-cache repeat and a fresh-server run answer different questions. Both concurrency 1 and 16 use the full English split for the baseline and each later PR A/B. Keep their results separate; c16 additionally carries the SM target. Repeat the same run order and cache policy for each variant, or restart consistently between matrix cells.

Record any optional `--generation-json params.json` used for the baseline and reuse it unchanged for a future candidate. For example `{"seed":1234,"max_new_tokens":2048}`. An explicit seed selects a particular upstream deterministic sampling path; do not add one to only one side or assume it reproduces the team's unseeded baseline. The harness preserves benchmark max-new-token default 2048, which differs from an API request that omits it and lets Cosy derive its length bound. Record this distinction.

### Later: compare an optimization candidate

Skip this subsection during initial baseline collection. Once an optimization candidate is implemented, restart with that one change and fresh matching state, set `COSY_VARIANT=candidate`, and repeat the complete c1/c16 matrix. Compare each matching mode/concurrency separately, for example:

```bash
python "$DIAG/compare_ab.py" \
  artifacts/cosyvoice/eager-streaming-en-c16-r1 artifacts/cosyvoice/candidate-streaming-en-c16-r1 \
  --output artifacts/cosyvoice/comparison-streaming-en-c16-r1.json
```

The comparator rejects profiled/incomplete runs and mismatched ordered inputs/client contracts. It reports ratios without declaring a winner. Verify identical model/weight and server contracts separately; repeat paired A/B runs to quantify variance. English quality for each mode and tail-latency gates are in [the PR acceptance contract](../plans/00_stack.md).

## 3. Small complete-pipeline NSYS capture

Stop the ordinary server through its normal lifecycle before launching the traced instance. Check the installed tool interfaces:

```bash
nsys --version
nsys launch --help
nsys start --help
nsys start --gpu-metrics-devices=help
```

The Python `nvtx` package must exist in the server environment for opt-in annotations. Check its installed version in the collector. The annotations use documented `nvtx.annotate` and `nvtx.mark`; no Torch profiling context is started. Use a **fresh named session** and launch the server before its child workers exist:

```bash
SGLANG_OMNI_PIPELINE_NVTX=1 SGLANG_OMNI_STRICT_PORT=1 \
nsys launch --session-new=cosyStream01 \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=process-tree \
  --cuda-graph-trace=node --trace-fork-before-exec=true \
  python -m sglang_omni.cli serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000 \
  --vocoder.factory.enable_dit_torch_compile false \
  --vocoder.factory.enable_flow_estimator_trt false
```

`nsys launch` prepares injection; collection starts separately. Kernel-node graph tracing is necessary because graph-level summaries can hide individual AR kernel intervals. The full process tree includes the API/coordinator and the pipeline worker. CPU context-switch collection availability depends on the container; record any unavailable collection rather than inferring CPU scheduling from absent records.

After checking which NSYS metric device corresponds to the H100, for example device 0:

```bash
python "$DIAG/run_seedtts.py" --mode streaming --lang en \
  --concurrency 16 --samples 32 --warmup 32 \
  --session cosyStream01 --metrics-devices 0 \
  --output artifacts/cosyvoice/profile-stream-en-c16
```

The runner loads/stages/hashes the selected samples, checks health, sends a separate pre-capture warmup cohort of 32 copies of sample zero, starts only the named NSYS session, runs the native benchmark with internal warmup zero, then stops that same session in `finally`. `nsys start` receives the absolute output prefix and optional GPU metric device list; those options belong to `start`, not `launch`. Exact control commands/stdout/stderr are saved. The measured client opens a new HTTP session after pre-warm. Collection includes a small amount of client launch/result-writing slack; choose the actual workload interval below. An externally killed harness cannot guarantee cleanup—use the named `nsys stop --session=cosyStream01` if necessary.

Use another fresh output/session for buffered c16 and c1. Select an offset covering longer prompts/outputs when the first cohort is unrepresentative, and add a cold unique-reference cohort. Thirty-two requests at c16 has ramp-up/drain; label it a **finite cohort**, not automatically steady state. A longer small cohort can supply an interior interval after verifying active request count. Full corpus profiling is intentionally not the acceptance benchmark.

## 4. Analyze on the exported clock

```bash
nsys export --type sqlite \
  --output artifacts/cosyvoice/profile-stream-en-c16/trace.sqlite \
  artifacts/cosyvoice/profile-stream-en-c16/trace.nsys-rep
python "$DIAG/analyze_pipeline_nsys.py" \
  artifacts/cosyvoice/profile-stream-en-c16/trace.sqlite --catalog \
  --output artifacts/cosyvoice/profile-stream-en-c16/catalog.json
```

Use the actual report name printed by NSYS if the tool adds a suffix. The catalog lists GPU identity, kernel extents, metric IDs/names and point events. Choose an explicit interval from HTTP `received` through the last `stream_body_complete` / `buffered_body_ready`, or the same predeclared interior interval used by the team's 3% metric. Inspect edge requests and active concurrency. Never choose only a busy kernel interval to inflate utilization.

```bash
python "$DIAG/analyze_pipeline_nsys.py" \
  artifacts/cosyvoice/profile-stream-en-c16/trace.sqlite \
  --device 0 --start-ns START_INTEGER --end-ns END_INTEGER \
  --kernel-details --output artifacts/cosyvoice/profile-stream-en-c16/analysis.json
```

Replace both interval placeholders with integers from that export. Windows are half-open; include the final endpoint marker by setting the end to its timestamp plus one nanosecond. Optionally add `--metric-type TYPE_ID --metric-id METRIC_ID` from the catalog for the exact SM metric. Its raw samples and units remain separate from kernel coverage. The program does not invent a mapping between CUDA ordinals and NSYS metric device types, nor rename a counter to “SM utilization.” Preserve the exact Nsight summary/aggregation alongside the JSON when assessing 3%→30%.

The analyzer provides per-device compute union, summed kernel time, compute/copy union, stage unions, exclusive stage time, pairwise overlap, largest gaps with overlapping host scopes/APIs, realized batch distributions, request lifecycle timings and optional kernel-to-host-scope detail. Attribution uses `(global process ID, CUDA correlation ID)` and the innermost containing host launch range on the launching thread. Device kernels may outlive a host range; this is expected. Duplicate/unknown correlations stay unattributed. Pairwise overlaps are not additive when three stages overlap.

Request markers share the Nsight clock. `input_identity` hashes target text to help match `inputs.json`; duplicate target texts remain ambiguous and must not be assigned by arrival order. Batched Flow time belongs to a group, not independently to each member. `request_ids` and nested scopes preserve group membership; buffered grouping indices and lengths must be inspected when splitting groups. Server first PCM yield is before client reception; the native benchmark's `audio_ttfp_s` follows HTTP chunk framing. Do not subtract client `perf_counter` or JSONL wall timestamps from Nsight ns.

## 5. Qualify the diagnostic slice before trusting its numbers

On H100, verify one c1 request and a small c16 cohort in each mode:

- Every accepted request has the expected stage handoffs, terminal event and HTTP endpoint marker; missing/error endpoints remain visible as null timings. No unexpected unrelated GPU process contributes.
- AR graph node kernels are present. Compare a few eager and graph CUDA launch→kernel links against the GUI. Record dropped-event warnings and unattributed-kernel fraction; incomplete attribution blocks conclusions about stage shares.
- Packed Flow, native Flow, HiFT and D2H are distinguished. Native Flow is an enclosing stage range, not a per-Euler trace. HiFT internal F0/FFT kernels are visible through CUDA but not individually annotated by this patch.
- Realized AR/Flow/HiFT batch counts agree with launch shapes/logs; streaming ready-candidate/peer-wait/step markers explain cohort selection. Do not interpret an entered peer-wait function's entire host time as pure sleep.
- Compare the same small cohort with annotations off, annotations on/no collection, and NSYS collection. Record overhead and reject profiler-on timing as A/B performance evidence. NVTX registers dynamic strings, so use fresh short capture runs, not an indefinitely annotated production service.
- Same model inputs yield the same output contract with annotations toggled. No additional synchronization, tensor materialization or model state mutation is introduced by annotations; verify actual runtime behavior rather than treating static inspection as certification.

Schema incompatibility or missing GPU metrics requires adapting/qualifying against the installed NSYS version. The analyzer fails on unsupported required kernel fields, requires an explicit window, refuses output overwrite and opens SQLite read-only. No synthetic local tests have been added or run.

## 6. Quality after generation, before returning results

Stop TTS before GPU-heavy quality scoring so quality work cannot contaminate serving measurements. Reuse `measured/generated.json` and WAVs. With an existing Qwen3-ASR server on port 8001:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --transcribe-only --use-existing-server --port 8001 --lang en \
  --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --output-dir artifacts/cosyvoice/eager-streaming-en-c16-r1/measured
python -m benchmarks.eval.benchmark_tts_seedtts \
  --similarity-only --lang en \
  --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --output-dir artifacts/cosyvoice/eager-streaming-en-c16-r1/measured
python -m benchmarks.eval.benchmark_tts_seedtts \
  --utmos-only --lang en \
  --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --output-dir artifacts/cosyvoice/eager-streaming-en-c16-r1/measured
```

Repeat for both c1/c16 and buffered output, then for a future candidate when available; preserve identical scorer/checkpoint identities. Score English only. SIM is cosine×100; the scorer provides no acceptance threshold. Inspect skipped/evaluated counts: all-failure WER summaries can contain zeros. Use paired per-sample differences and listening around streaming joins, not mean quality alone.

## 7. Return plain artifacts without WAVs

Keep the directory layout under `artifacts/cosyvoice/` when bringing results back to the local Omni checkout. Complete remote quality scoring before excluding WAVs. There is no archive or packaging step. For example, from the local checkout, replace the host and checkout path:

```bash
rsync -a --exclude='*.wav' --exclude='*.WAV' \
  H100:/path/to/sglang-omni/artifacts/cosyvoice/ artifacts/cosyvoice/
```

Retain each run's `experiment.json`, `inputs.json`, `measured/speed_results.json`, `results.csv`, `generated.json`, and all quality summaries/per-sample results and failure counts. Also return launch/config/startup logs, `environment*.json`, generation parameter files, full commit SHAs, and any candidate diff if the server tree was dirty. For captures retain `.nsys-rep`, exported `.sqlite`, `nsys_start.log`, `nsys_stop.log`, catalog/analysis JSON and the exact metric/window definition. Empty audio directories are harmless. Do not delete WAVs remotely until scoring and any needed listening checks are complete.

The supplied comparator and SQLite analyzer read these artifacts without opening WAVs. `generated.json` may refer to remote WAV paths; those references preserve provenance but cannot support local rescoring after audio is omitted. Per-request speed/quality rows, input hashes and trace request markers support selecting a failing cohort and replaying its source offset on H100. Inspect duplicate target hashes before matching server requests; never infer their identity from arrival order.

Use `analyze_pipeline_nsys.py` for each explicit baseline trace window. Once a candidate exists, use `compare_ab.py` for each matching English c1/c16 pair. The raw SQLite and per-request JSON also remain available for perfkit or another later analysis tool; this branch does not depend on an uninspected perfkit interface. Promote a component PR only after the [open-PR overlap audit](../reports/17_open_prs.md), [existing optimization assessment](../plans/09_existing_optimizations.md), and the measured trigger in its plan agree.

References: [Nsight Systems command/lifecycle documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [export schema](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html), [NVTX API and message caching](https://nvidia.github.io/NVTX/python/reference.html). Upstream SGLang's complete profiling code and examples are covered by [report 08](../reports/08_profiling.md).
