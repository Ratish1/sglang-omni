# H100 capture and evaluation runbook

These tools are an initial diagnostic implementation with static checks only. Run them in the H100 serving environment. No local pytest or profiling unit tests are required. These scripts live on branch `analysis/cosyvoice-utilization-20260912` and run with the same Python executable and package paths as the server; the servers themselves boot from the arm worktrees of section 0.

**Scope: English only (`en`), at c1 and c16, for streaming and buffered output.** The current task is baseline collection, the B stream c16 pair of section 2, and small NSYS captures. A small annotation-off/on check only qualifies diagnostic overhead.

## 0. Session protocol for an A/B pair

Arms of the current pair: A is upstream `main` at `51e2f7ec2`, which is `f58228dfb` plus the Flow graph fix of #2147 and nothing else (`git log f58228dfb..51e2f7ec2` is that one commit); B is the PR branch head `a8700d833`. Nsight runs only on `622bcd198`, the same tree plus the pipeline NVTX instrumentation, and never on a measured boot.

One server per arm, from that arm's worktree, started with `python -m sglang_omni.cli serve` and nothing but `CUDA_VISIBLE_DEVICES` in front of it. Never the `sgl-omni` console script: it imports the venv's editable install, which is the main checkout, from any directory. Every other environment variable on a measured server command, `SGLANG_OMNI_STRICT_PORT` in section 1 included, is a protocol change and is recorded in the readout; `SGLANG_OMNI_PIPELINE_NVTX` stays off, which is its default, on every measured boot. Before each boot, from the arm's worktree:

```bash
BOOT=artifacts/cosyvoice/b-stream-en-c16
mkdir -p "$BOOT"
git rev-parse HEAD > "$BOOT/head.txt"
python -c "import sglang_omni; print(sglang_omni.__file__)" > "$BOOT/import_path.txt"
nvidia-smi > "$BOOT/gpus_before.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv >> "$BOOT/gpus_before.txt"
nvidia-smi dmon -i 0 -s pucv -d 1 > "$BOOT/dmon.log" &
```

The path in `import_path.txt` must be under that worktree or the boot is void. The other GPUs are recorded, not required to be idle: a pair is valid when both arms ran under the same load, so the readout quotes the paired delta next to the census.

Gate the B boot on a log line only `a8700d833` prints. Its one new logger call is `Fun-CosyVoice3 causal Flow batch size=%d hop=%d token_offset=%d` (`sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py:398` at `a8700d833`); at `51e2f7ec2` the same site prints `first-hop Flow batch size=` or `follow-up Flow batch size=` (`streaming_vocoder.py:663` and `:670`) and never the causal line. After the first traffic reaches the vocoder:

```bash
grep -c "Fun-CosyVoice3 causal Flow batch size=" "$BOOT/serve.log"     # B nonzero, A zero
grep -c "Fun-CosyVoice3 follow-up Flow batch size=" "$BOOT/serve.log"  # A nonzero, B zero
```

Two passes per arm at every point, pass 1 unprofiled and pass 2 with the event recorder. Both arms of a pair use the same warmup and no seed. Expected values are written into this runbook before the run, each with the file and line it comes from; the readout states the verdict first and the provenance second.

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

The runner's default warmup is 1 copy of the first sample and no seed is set, the standing protocol since 2026-09-09 and the playbook's client read. The warmup is excluded from native measured wall time. Both arms of a pair must use the same warmup: arm A's recorded corpus run used warmup 1 ([findings](../FINDINGS_20260913.md), item 4), no file in this task directory records a run at warmup 32, and the runner defaulted to 32 until this change, so any earlier run that omitted `--warmup` is not comparable with these. A warm-cache repeat and a fresh-server run answer different questions; preserve the cache policy across A/B. Both concurrency 1 and 16 use the full English split for the baseline and each later PR A/B. Keep their results separate; c16 additionally carries the SM target. Repeat the same run order and cache policy for each variant, or restart consistently between matrix cells.

Record any optional `--generation-json params.json` used for the baseline and reuse it unchanged for a future candidate. For example `{"seed":1234,"max_new_tokens":2048}`. An explicit seed selects a particular upstream deterministic sampling path; do not add one to only one side or assume it reproduces the team's unseeded baseline. The harness preserves benchmark max-new-token default 2048, which differs from an API request that omits it and lets Cosy derive its length bound. Record this distinction.

### The B stream c16 rerun

The candidate exists, so this is the pair that is owed: A `51e2f7ec2`, B `a8700d833`, streaming, English, full split, c16, warmup 1, no seed, two passes per arm, both arms in one session on the same GPU. Pass 1 is the unprofiled read:

```bash
python "$DIAG/run_seedtts.py" --mode streaming --lang en --concurrency 16 \
  --output "$BOOT/pass1" || exit 1
```

Pass 2 repeats it with the request event recorder on. Start the recorder once the pass is running and stop it after about 200 completions, so the window is one steady-state cohort rather than ramp-up and drain:

```bash
curl -s -X POST http://127.0.0.1:8000/start_profile -H 'Content-Type: application/json' \
  -d "{\"run_id\":\"pass2\",\"event_dir\":\"$BOOT/pass2/events\",\"enable_torch\":false}"
# after about 200 completions
curl -s -X POST http://127.0.0.1:8000/stop_profile -H 'Content-Type: application/json' \
  -d '{"run_id":"pass2"}'
```

`enable_torch` false records the JSONL milestones without paying for a kernel trace; the HTTP surface and the file layout `<event_dir>/events_<stage>_<pid>.jsonl` are in `docs/developer_reference/profiler.md`. Read the events with the shipped views and with the anatomy script from the Qwen3-TTS backlog, fetched from its commit rather than copied here:

```bash
python -m sglang_omni.profiler "$BOOT/pass2/events" --format table
git show fb590a2c3:tasks/perf_backlog/scripts/first_chunk_anatomy.py > /tmp/first_chunk_anatomy.py
python /tmp/first_chunk_anatomy.py "B=$BOOT/pass2/events:$BOOT/pass2/measured/speed_results.json"
```

The anatomy script's `CHAIN` is a list of stage and event-name pairs, and the recorder is generic. CosyVoice's stage names are the same three the script expects, `preprocessing`, `tts_engine` and `vocoder` (`sglang_omni/models/fun_cosyvoice3/config.py:107-132` at `622bcd198`), and first audio is the vocoder's `stage_stream_chunk_sent`, which is what the script already reads. CosyVoice emits no recorder events of its own at `622bcd198` (no `emit`, `emit_model_path` or `get_recorder` call under `sglang_omni/models/fun_cosyvoice3`), so only the generic stage and `OmniScheduler` events appear; on the first event directory check which `CHAIN` segments come back empty and read the rest.

Reads per arm and pass, from `measured/speed_results.json` and the dmon log: failures, requests per second, audio seconds per second, first audio (`audio_ttfp_s`) mean and p95, inter-chunk p99, C50, WER, and the dmon utilization mean during traffic. Expected values for the B stream c16 rerun, written before the run, every number with its source:

| arm | failures | WER | first audio | C50 | req/s | audio s/s |
|---|---|---|---|---|---|---|
| A, `51e2f7ec2` | 23 timeouts (findings item 11) | 6.82 percent on 1065 scored (item 11) | 0.91 s mean, 2.17 s p95 (item 11) | 79.8 (item 11) | 1.62 (item 11) | 7.40 (item 11) |
| previous B, `691c18371` | 0 (slice 02 line 14) | 1.59 percent (item 11) | 3.35 s mean, 6.60 s p95 (item 11) | 9.3 (item 11) | 1.95 (item 11) | 9.47 (item 11) |
| B, `a8700d833` | 0 (item 11) | 1.38 percent (item 11) | 3.32 s mean, 6.35 s p95 (item 11) | 13.2 (item 11) | 2.05 (item 11) | 9.72 (item 11) |
| PR #2086 | 0 (item 11) | 1.27 percent (item 11) | 11.58 s mean, 15.39 s p95 (item 11) | 98.6 (item 11) | 1.32 (item 11) | 6.39 (item 11) |

Item 11 is in [the findings](../FINDINGS_20260913.md); the slice 02 line is in [the vocoder step cost slice](../slices/02_vocoder_step_cost.md). All four rows are one session on GPU 0, archived under `artifacts/full-20260913T171454Z/cosyvoice-ab-gpu0/<arm>/stream-c16/` in the local checkout's `artifacts/` directory, so the rows are a real pair and safe to read across.

The buffered cells and c1 are collected the same way when the matrix is repeated. Compare each matching mode and concurrency separately, for example:

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

The Python `nvtx` package must exist in the server environment for opt-in annotations. Check its installed version in the collector. The annotations use documented `nvtx.annotate` and `nvtx.mark`; no Torch profiling context is started. The capture runs from the `622bcd198` worktree, which is `a8700d833` plus those annotations, and never during a measured pass. Use a **fresh named session** and launch the server before its child workers exist:

```bash
SGLANG_OMNI_PIPELINE_NVTX=1 SGLANG_OMNI_STRICT_PORT=1 \
nsys launch --session-new=cosyStream01 \
  --trace=cuda,nvtx,osrt,python-gil --sample=none --cpuctxsw=process-tree \
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

The runner loads/stages/hashes the selected samples, checks health, sends a separate pre-capture warmup cohort of `--warmup` copies of sample zero, starts only the named NSYS session, runs the native benchmark with internal warmup zero, then stops that same session in `finally`. The 32 above is deliberate and is not the benchmark warmup of section 2: it is the preconditioning pass the capture window needs, and with `--session` the measured benchmark runs at internal warmup zero regardless. `nsys start` receives the absolute output prefix and optional GPU metric device list; those options belong to `start`, not `launch`. The runner passes `--gpu-metrics-devices` only (`run_seedtts.py:56-62`), while the Qwen3-TTS layer-4 protocol also sets `--gpu-metrics-frequency=20000`; adding that flag to the runner is an open item, not done here. Exact control commands/stdout/stderr are saved. The measured client opens a new HTTP session after pre-warm. Collection includes a small amount of client launch/result-writing slack; choose the actual workload interval below. An externally killed harness cannot guarantee cleanup, so use the named `nsys stop --session=cosyStream01` if necessary.

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

The same export feeds the two perfkit tools, which is where the thread and hop answers come from:

```bash
python tasks/cosyvoice_utilization_20260912/perfkit/slice_trace.py \
  artifacts/cosyvoice/profile-stream-en-c16/trace.sqlite \
  --md artifacts/cosyvoice/profile-stream-en-c16/slice.md \
  --replay-json artifacts/cosyvoice/profile-stream-en-c16/replay.json
python tasks/cosyvoice_utilization_20260912/perfkit/nsys_threads.py \
  artifacts/cosyvoice/profile-stream-en-c16/trace.sqlite \
  artifacts/cosyvoice/profile-stream-en-c16/threads.json
```

`nsys_threads.py` is the Qwen3-TTS script at `fb590a2c3`, copied unchanged. It reads `StringIds` (line 28), `ThreadNames` (31), `CUPTI_ACTIVITY_KIND_RUNTIME` (51), `CUPTI_ACTIVITY_KIND_KERNEL` (68) and `NVTX_EVENTS` (81). All but `ThreadNames` are tables `slice_trace.py` already reads from these exports, and `graphNodeId` in its kernel query needs `--cuda-graph-trace=node`, which the launch command above already passes. Its GIL wait and hold totals (lines 84 and 87) come from the `Waiting for GIL` and `Holding GIL` NVTX ranges, which need `python-gil` in the trace list; that is why section 3 adds it. `ThreadNames` is the one table nothing else here reads and no flag above demonstrably fills, so on the first export confirm it is populated before quoting a per-thread name.

`perfkit.py` (`tasks/qwen3_omni_0518_numerics/scripts/perfkit.py` at `fb590a2c3`) is the layer-3 tool for chrome traces, not for these exports, and it needs one per-model change before it says anything about CosyVoice: its `--predictor-marker` defaults to `gather_codec_embedding|seeded_top_k_top_p|seeded_gumbel` (lines 1238 and 1254), the Qwen3-TTS code predictor kernels, which no CosyVoice kernel matches, and `steps` keeps only steps that matched a predictor, so `steps` and `census` come back empty until the marker names CosyVoice's second graph, the Flow replay.

## 5. Qualify the diagnostic slice before trusting its numbers

On H100, verify one c1 request and a small c16 cohort in each mode:

- Every accepted request has the expected stage handoffs, terminal event and HTTP endpoint marker; missing/error endpoints remain visible as null timings. No unexpected unrelated GPU process contributes.
- AR graph node kernels are present. Compare a few eager and graph CUDA launch→kernel links against the GUI. Record dropped-event warnings and unattributed-kernel fraction; incomplete attribution blocks conclusions about stage shares.
- Packed Flow, native Flow, HiFT and D2H are distinguished. Native Flow is an enclosing stage range, not a per-Euler trace. HiFT internal F0/FFT kernels are visible through CUDA but not individually annotated by this patch.
- Realized AR/Flow/HiFT batch counts agree with launch shapes/logs; the `scheduler/ready_candidates` mark, one per step selection at `622bcd198`, explains cohort selection and is what the hop ledger charges admission waits from. There is no peer wait at this head.
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
