# Sector 09: benchmarks and evaluation

## Scope and ownership

This report describes the checked-out Omni baseline identified by the research contract. It is source inspection only; no benchmark, model, profiler, or test was run.

The fixed-corpus SeedTTS path is separate from `benchmarks/tts_serving`. The former owns generation speed plus optional ASR WER, WavLM speaker similarity, and UTMOS quality evaluation. The latter owns a deterministic API serving-contract and stress matrix and does not launch the service or calculate SeedTTS WER/SIM (`benchmarks/tts_serving/README.md:1-18`).

The reachable SeedTTS call graph is:

```text
benchmark_tts_seedtts.main
  -> load_seedtts_samples
  -> managed_omni_server (unless --use-existing-server)
  -> run_tts_seedtts_benchmark
       -> make_tts_send_fn -> POST /v1/audio/speech
       -> BenchmarkRunner.run (warmup, then measured dispatch)
       -> compute_speed_metrics
       -> speed_results.json/results.csv/generated.json/audio/*.wav
  -> managed ASR server + run_seedtts_transcribe (unless generate-only)
       -> run_asr_transcription -> POST /v1/audio/transcriptions
       -> apply_wer/calculate_wer_metrics/calculate_asr_speed_metrics
       -> wer_results.json/wer_results.csv/asr_speed_results.json

independent reuse of generated.json:
  --similarity-only -> WavLMSpeakerSimilarity -> similarity_results.json
  --utmos-only      -> UTMOSScorer -> utmos_results.json
```

`benchmarks/eval/benchmark_tts_seedtts.py:1085-1162` owns mode selection and server lifecycle. Generation against a managed server forwards `max_running_requests`, `max_queued_requests`, `cuda_graph_max_bs`, and `quantization` to the single engine stage, except the special `auk`/`auk-flash` profile. Full-pipeline mode stops TTS, performs optional GPU cleanup, starts the ASR checkpoint on the same host/port, then transcribes. `--use-existing-server` is accepted only with generate-only or transcribe-only. In transcribe-only plus existing-server mode, the configured TTS port is treated as the already-running ASR router port (`benchmark_tts_seedtts.py:1046-1082,1085-1162`; `benchmarker/utils.py:260-359`).

## CLI contract

The parser occupies `benchmark_tts_seedtts.py:736-1043`; validation is at lines 1046-1082. Flags and source defaults are:

| Flag | Default / behavior |
|---|---|
| `--base-url` | `None`; overrides host/port for generation requests. |
| `--host`, `--port` | `localhost`, `8000`. |
| `--model` | `fishaudio/s2-pro`; request model and managed TTS model. |
| `--voice` | `None`; optional server-side preset. |
| `--task-type`, `--instructions` | `None`; forwarded only when present. |
| `--meta`, `--testset` | Alias to `meta`; default `zhaochenyang20/seed-tts-eval-arrow`. |
| `--no-ref-audio` | Plain TTS: omits all reference audio/text. |
| `--no-ref-text` | Retains reference audio but omits transcript; irrelevant when reference audio is omitted. |
| `--ref-format` | `flat`; choices `flat` (`ref_audio`, `ref_text`) and `references` (one `{audio_path,text?}` object). |
| `--response-format` | `wav`; `_config_from_args` forces `pcm` whenever `--stream` is set. |
| `--output-dir` | `results/tts_seedtts`. |
| `--max-samples` | `None`, meaning the whole selected split/source. |
| `--sample-offset` | `0`; nonnegative, applied before the retained shard. |
| `--max-new-tokens` | `2048`; forwarded unless programmatic config sets `None`. |
| `--token-count` | `None`; positive integer or literal `auto`. Auto estimates each sample independently as `max(32, int(character_count * factor))`, factor `3.098411951313033` when Chinese ideographs are present and at least as numerous as ASCII letters, otherwise `0.8673376262755219` (`tasks/tts.py:1016-1044`). |
| `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty` | `None`; omitted from payload when absent. |
| `--seed` | `None`; if supplied, the same integer is placed in every generation request. It does not seed client arrival timing. |
| `--subtalker-dosample-ratio` | `None`; omits the setting. A supplied value must be in `[0,1]`; request index `i` is true when `floor((i+1)r)>floor(ir)`, keyed by sample ID (`benchmark_tts_seedtts.py:252-262,316-348`). |
| `--warmup` | `None`, resolved to concurrency; explicit `0` disables. |
| `--concurrency`, `--max-concurrency` | Alias; environment `TTS_BENCHMARK_CONCURRENCY`, default `16`. |
| `--concurrencies` | `None`; comma-separated positive integers, generate-only only. |
| `--sustained-overshoot` | False; generate-only, mutually exclusive with sweep, requires queued limit. |
| `--overshoot-duration-s` | `10.0`, positive. |
| `--request-rate` | Infinity (burst dispatch); finite values sleep exponential intervals before task creation. |
| `--stream` | False; selects raw PCM streaming. |
| `--initial-codec-chunk-frames` | `None`, nonnegative; only sent for streaming. |
| `--save-audio` | Legacy no-op; generation always saves WAV files in normal/sweep modes. |
| `--disable-tqdm` | False. |
| `--lang` | `en`; choices `en`, `zh`; selects dataset split and ASR normalization/language. |
| `--device` | `cuda:0`; used by similarity/UTMOS, not router ASR requests. |
| `--asr-model-path` | `Qwen/Qwen3-ASR-1.7B`; Whisper paths are selected by substring `whisper`. |
| `--asr-concurrency` | `32`. |
| `--similarity-checkpoint` | `None`; official fine-tuned WavLM asset otherwise. |
| `--server-timeout` | `1200` seconds readiness timeout. |
| `--max-running-requests` | `64`, positive. |
| `--max-queued-requests` | `None`, or integer at least one. |
| `--cuda-graph-max-bs` | `64`, positive. |
| `--server-config` | `None`; managed TTS only. A config pin for the same bare repo ID can replace the bare model with its `@revision` (`benchmarker/utils.py:321-331`). |
| `--quantization` | `None`; managed TTS engine only, never managed ASR. |
| `--skip-gpu-cleanup` | False; suppresses post-server `ensure_gpus_idle`. |
| `--use-existing-server` | False; no process start/stop. |
| mutually exclusive `--generate-only`, `--transcribe-only`, `--similarity-only`, `--utmos-only` | All false gives generation followed by router ASR WER. |

Model checkpoint basename `auk` or `auk-flash` causes the parser to reparse with defaults `output_dir=results/auk_seedtts`, concurrency 1, warmup 1, seed 1234, and prevents forwarding SGLang engine overrides. Explicit user values still override parser defaults (`benchmark_tts_seedtts.py:132-172`).

Concurrency sweep writes each point below `<output-dir>/c<N>` and a top-level `concurrency_sweep.json`. Sustained overshoot defines admission capacity as running plus queued, defaults rate to twice that capacity, requires rate strictly above capacity, calculates `ceil(rate*duration)` samples by cyclic replication with `#index` IDs, sets client concurrency to zero/unbounded, and writes below `<output-dir>/overshoot` (`benchmark_tts_seedtts.py:532-725`).

## Dataset, order, staging, references, and randomness

`load_seedtts_samples` dispatches to local `meta.lst` when the source is an existing file or merely ends in `.lst`; all other strings are Hugging Face dataset IDs (`dataset/seedtts.py:80-99`).

Local order is file order. Blank lines and records with fewer than four pipe-separated fields are skipped; fields 0-3 become sample ID, reference text, reference path relative to the meta file directory, and target text. The loader stops when `max_samples` valid records have accumulated. A nonpositive maximum returns an empty list (`dataset/seedtts.py:102-127`).

Remote order is the selected dataset split's iteration order. For the canonical dataset ID from `benchmarks.dataset.prepare`, pinned revision `27f4c1adee83b5b29b7c4b375f6b976324bda308` is substituted; other IDs use the provider default unless a programmatic revision is passed. Required columns are `sample_id`, `ref_text`, `ref_audio_path`, `target_text`, and `ref_audio`. Audio decoding is disabled, and raw bytes are staged under a process temporary directory. Absolute/anchored paths, drive paths, parent traversal, and resolved escapes are rejected. Repeated relative audio paths are written once. Staging is registered for process-exit removal (`dataset/seedtts.py:43-77,130-214`; `dataset/prepare.py:30-31`).

The in-process staging cache key is `(repo_id, split, revision, max_samples)`. A fully cached `(…, None)` list satisfies later prefixes by slicing. Returns are new list objects containing the same immutable-value sample records. `sample_offset=N,max_samples=M` asks the loader for the first `N+M` records and slices off the first N; offset with no maximum loads the full split then slices. Negative offset raises. Therefore source order is preserved and there is no shuffle (`benchmark_tts_seedtts.py:301-313`).

Flat and structured reference payloads carry paths/URIs as strings. Every ordinary measured sample uses its dataset-specific reference unless `--no-ref-audio`. Warmup deliberately repeats only `samples[0]`; it does not warm distinct references. A task-local diagnostic note recognizes that repeated versus unique references can produce different cache traffic, but the benchmark does not report server cache hits itself.

Generation randomness is server-owned. `--seed N` sends identical `seed=N` on every request. With no flag, no seed field is sent. Subtalker sampling selection is deterministic by post-offset sample index, but any sampling inside the selected mode remains server behavior. Finite-rate dispatch uses `numpy.random.exponential` without setting NumPy's RNG in this entry point, so arrival intervals are not controlled by `--seed` (`benchmarker/runner.py:128-153`). The separate `tts_serving` matrix uses `random.Random(f"{spec.seed}:{stage.id}")`, assigns baseline payload seeds as `spec.seed + index`, hashes the scenario list, and, for the optional pinned `seedtts-en` corpus, shuffles corpus texts with a stage-derived local RNG before assigning globally non-overlapping texts to supported scheduled workloads (`tts_serving/scenarios.py:267-281,360-383,535-645,921-936`).

## Warmup, dispatch, timing, and denominators

`resolve_warmup(None,c)` is `c` when `c>0`, otherwise one; an explicit integer is returned unchanged (`benchmarker/runner.py:23-29`). Warmup occurs in the same `aiohttp.ClientSession`, sends `effective_warmup` concurrent copies of the first sample under the same semaphore, logs each result, and aborts on any failure. Empty sample input produces zero actual warmup calls. Warmup outputs do not enter saved results, and measured `wall_clock_s` begins only after warmup immediately before measured dispatch (`benchmarker/runner.py:64-125`).

Measured closed-loop dispatch creates tasks in sample order; the semaphore bounds in-flight send calls. Infinite request rate creates all tasks without inter-arrival sleeps. Finite rate samples one exponential delay and sleeps before creating each task, including before the first task. `asyncio.gather` returns results in task/sample order. Concurrency zero removes both the semaphore and aiohttp's normal connection limit; this is used by overshoot (`benchmarker/runner.py:64-153`). Per-session total timeout defaults to 300 seconds for TTS.

Per-request latency uses client `perf_counter` from immediately before `session.post` until the response context finishes or throws. Nonstream audio duration uses a minimal fixed-offset WAV calculation: byte count after 44 bytes divided by header sample rate, channels, and bytes/sample. It does not walk arbitrary RIFF chunks. Streaming duration is exact byte count divided by response format rate/channels/sample width; missing PCM headers default to 24 kHz, mono, 16 bit. PCM must have positive format values, bit depth divisible by eight, nonempty body, and final byte length aligned to a sample frame. Streaming is saved by wrapping concatenated PCM in a WAV container (`tasks/tts.py:1047-1190,1243-1315`; `benchmarker/utils.py:373-384`).

Speed summary filtering and denominators (`metrics/performance.py:80-225`):

- request counts include every measured output; latency aggregates, throughput numerator, audio durations, RTF, TTFA/TTFT/chunks, tokens, and playback continuity use successful outputs only, with the additional positive/non-null filters in code;
- `throughput_qps = successful requests / measured wall clock`; only when no positive wall clock is supplied does it fall back to successes divided by summed successful per-request latencies;
- `audio_throughput_s_per_s = sum(positive generated audio seconds) / measured wall clock`;
- per-request `rtf = elapsed request seconds / output audio seconds`; RTF aggregates exclude nonpositive and infinite values;
- `output_throughput = total positive completion tokens / measured wall clock`;
- `output_tok_per_req_s = total positive completion tokens / sum(positive engine times)`. This is an aggregate over summed per-request engine/request times, not wall-clock token throughput;
- percentiles are NumPy percentiles. Summary rounding is three decimals for latency/audio throughput/QPS, four for RTF and streaming latencies, one for token rates, and zero for mean token counts.

ASR transcription creates samples only for successful generation rows, but reconstructs a `SampleOutput` in original `generated.json` order for every row. Generation failures are retained as skipped WER rows. ASR warmup is `asr_concurrency * 2` copies of its first successful generated sample, outside the ASR measurement window. `asr_concurrency` is clamped to at least one. ASR throughput is successful WER evaluations with positive ASR latency divided by the ASR runner wall clock; if no positive wall clock is supplied, it uses summed ASR latencies (`tasks/tts.py:511-632`; `metrics/wer.py:248-307`).

## Streaming TTFA, HTTP fragments, and playback continuity

The SeedTTS generation metric is named `audio_ttfp_s` in data and JSON and printed as TTFC. It is client time from request-send timestamp to the timestamp attached to the first reconstructed HTTP body chunk. `_iter_response_http_chunks` uses aiohttp `iter_chunks()`: it can receive several transport data pieces, accumulates them until `end_of_http_chunk`, and assigns that yielded chunk the timestamp of its first nonempty piece. Any final pending bytes at EOF become another chunk. Thus `audio_chunk_count`, first payload bytes, inter-chunk gaps, and TTFA describe HTTP chunk framing as exposed by aiohttp, not model codec chunks, CUDA work, or arbitrary `read()` frames (`tasks/tts.py:1086-1167`).

The first SeedTTS streaming chunk is not independently subjected to the `tts_serving` first-chunk validator. The full concatenated response is required to be nonempty and frame-aligned; success then follows. `first_audio_payload_bytes` is raw first HTTP body-chunk size. `chunk_audio_duration_s[i]` is each body-chunk bytes divided by bytes/second. Playback underrun starts the first chunk's deadline at its arrival plus its duration, then for each next chunk measures `max(0,arrival-deadline)` and advances from the later of deadline/arrival plus duration. It is undefined (`None`) for fewer than two chunks. C50/C100/C200 are percentages of multi-chunk successful streams with maximum underrun at most 50/100/200 ms; single-chunk requests are counted as N/A (`metrics/playback_continuity.py:11-82`).

Nonstreaming requests buffer the entire response and never populate `audio_ttfp_s`. Headers `X-Prompt-Tokens`, `X-Completion-Tokens`, and `X-Engine-Time` are parsed only on the nonstream path. Their integer/float conversion exceptions are not locally caught by `_handle_non_streaming_response`, and `make_tts_send_fn` catches only aiohttp client errors and asyncio timeouts, so malformed usage headers can escape the send coroutine (`tasks/tts.py:1047-1058,1170-1190,1257-1315`).

The separate serving-stress `ScenarioResult.ttfa_s` has the same reconstructed HTTP-chunk timing rule (`tts_serving/http_client.py:439-518,596-613`). Its first-chunk disconnect probe additionally validates nonempty, 16-bit alignment and non-placeholder-like bytes, but does not require minimum duration or nonzero signal. Full raw PCM validation requires at least 0.05 seconds at the fixed mono/16-bit constants, alignment, non-placeholder bytes, and nonzero samples. WAV requires RIFF/WAVE, PCM format code 1, valid fmt/data, exactly 16-bit, nonzero signal, and the same 2,400-byte minimum; the minimum byte threshold is fixed from 24 kHz mono even if the WAV declares another rate/channel count. MP3/FLAC/AAC/Opus require matching content type/container prefixes and successful ffmpeg decode to 24 kHz mono s16le, subject to a 10-second decode timeout and 64 MiB decoded cap (`tts_serving/audio_validation.py:17-49,58-350`).

## Output schemas

Normal SeedTTS generation always writes:

- `audio/<sample_id>.wav` for each successful saved response;
- `generated.json`, an ordered JSON array with `{sample_id,target_text,wav_path,is_success,latency_s,audio_duration_s,error?}`;
- `speed_results.json` with `{summary,config,per_request}`. `config` contains model/base URL/meta/reference controls/response format/voice/task/instructions/stream/sample bounds/max tokens/seed/subtalker ratio/token count/resolved warmup/concurrency/request rate/first codec frames/server capacity/config/quantization. `per_request` contains `{id,text,is_success,latency_s,audio_duration_s,rtf,prompt_tokens,completion_tokens,output_token_rate,wav_path,error,audio_ttfp_s,text_ttft_s,inter_chunk_s,chunk_audio_duration_s,max_playback_underrun_s,audio_chunk_count,first_audio_payload_bytes}`; empty optional values become null. `text` is only the first 60 target characters;
- `results.csv` columns `id,text,latency_s,audio_duration_s,rtf,prompt_tokens,completion_tokens,output_token_rate,audio_ttfp_s,audio_chunk_count,first_audio_payload_bytes,is_success,error`. It omits inter-chunk durations, chunk audio durations, underrun, TTFT, and WAV path (`tasks/tts.py:1318-1398`; `metrics/performance.py:379-422`).

Transcription writes `wer_results.json={summary,config,per_sample}` and a CSV. JSON per sample is `{id,target_text,whisper_text,ref_norm,hyp_norm,wer,substitutions,deletions,insertions,hits,audio_duration_s,latency_s,is_success,error}`. The CSV omits normalized texts. `asr_speed_results.json` is the ASR summary dictionary directly, including ASR model/concurrency (`tasks/tts.py:79-142,570-632`). Similarity writes `{summary:{speaker_similarity_mean,total_samples,evaluated,skipped},config:{model,meta,device,max_samples,similarity_checkpoint},per_sample:[{id,ref_audio,wav_path,speaker_similarity,is_success,error}]}`. UTMOS writes the analogous summary/config/per-sample document with mean, median, p5, p95 and each `utmos_score` (`tasks/tts.py:162-446`).

Concurrency sweep `rows[]` embeds each point's entire summary plus concurrency, resolved warmup, directory, success/failed, latency p95, and audio TTFA p95. Sustained overshoot writes `{plan,outcomes,summary,config}` with total/success/queue-full/other-failed and nearest-rank p95 success TTFA/rejection latency (`benchmark_tts_seedtts.py:584-725`).

The serving-stress writer atomically creates `results.json`, `manifest.json`, `raw/requests.jsonl`, `raw/events.jsonl`, and `logs/harness.log`. Manifest schema version 1 records creation time, labels, redacted platform metadata, spec/workload/scenario hashes, scenario schema version 4, serialized load stages, and relative artifact paths. Request JSONL is the full scenario record with audio reference strings replaced by length, 16-hex SHA-256 prefix, scheme, and a sanitized URL or kind. Event JSONL serializes every `ScenarioResult` field; infinities become strings. JSON serialization rejects NaN (`tts_serving/artifacts.py:39-217`; `tts_serving/metrics.py:24-81`). `results.json` aggregation is owned by `benchmarks/tts_serving/report.py`, outside the assigned full-read set; its README contract identifies overall pass/harness/coverage/load-generation/mixed-arrival status, status and endpoint counts, per-stage/workload metrics, unsupported contracts, and coverage failures (`tts_serving/README.md:277-309`).

## WER, CER-like Chinese scoring, SIM, and MOS numerical contracts

English uses `whisper.normalizers.EnglishTextNormalizer`. Chinese removes zhon plus ASCII punctuation except apostrophe, removes ASCII/full-width spaces, strips, and inserts spaces between every remaining Unicode character. Both then call `jiwer.process_words`, so the emitted field names remain `wer`; for Chinese, the tokenization makes this character-level in effect, but this SeedTTS path exposes no separate `cer` key (`tasks/asr.py:61-95,284-299`).

Per-sample WER is JiWER `(substitutions + deletions + insertions)/(substitutions + deletions + hits)`. Empty normalized references are failures/skipped. Corpus WER micro-aggregates edit counts and reference words across successes. It also reports mean/median/std/p95/max of per-sample WER. The `below_50` corpus and mean include samples with WER at most 0.5; `n_above_50` uses strict `>0.5`. WER is not clamped, so insertions can produce values above one. An all-failure corpus returns numeric zeros plus evaluated zero, rather than null metrics (`metrics/wer.py:40-115`).

Qwen3 ASR sends model/language/JSON response and uses its server greedy default; Whisper additionally sends temperature zero. Whisper audio longer than 30 seconds is loaded/resampled to mono 16 kHz, divided into 30-second chunks at 25-second strides (5-second overlap), transcribed, and concatenated without overlap reconciliation. Qwen3 sends the saved WAV directly (`tasks/asr.py:98-241`).

SIM loads reference and generated audio as float32, averages multichannel audio, resamples to 16 kHz using polyphase resampling, embeds reference and generated batches together with frozen WavLM feature extraction plus an ECAPA-TDNN head, and computes PyTorch cosine similarity along embedding dimension multiplied by 100. The code neither clips scores nor declares a pass threshold; the mathematical cosine range implies `[-100,100]`, subject to runtime numeric behavior. Batches are eight. Only rows with successful generation and readable generated/reference files are scored; others are exhaustive skipped rows. No scoreable rows raises before writing a result (`metrics/speaker_similarity.py:262-344`; `tasks/tts.py:162-314`).

Default SIM assets are `popsoda2002/seedtts-wavlm-sim/wavlm_large_finetune.pth` and `s3prl/converted_ckpts/wavlm_large.pt`. Cache priority is explicit arg, `SEEDTTS_SIM_CACHE_DIR`, then `~/.cache/sglang-omni/speaker_sim`. Marker schema 2 records exact expected file set, repo ID, and observed size; files must also exceed 100 MiB. A custom fine-tune checkpoint is used as-is after existence checking, while the base remains cache-managed (`metrics/speaker_similarity_assets.py:75-354`).

UTMOS states scores are in `[1,5]`, loads each WAV, averages channels, resamples to 16 kHz, converts/clamps to int16, invokes a TorchScript predictor per file, and takes its output mean. The wrapper does not independently clamp or reject predictor values outside that documented range. Task-level batches are eight, but `score_batch` loops sequentially. Cache priority is explicit, `UTMOS_CACHE_DIR`, then user cache; it validates repo/filename/size marker and a 10 MiB download floor (`metrics/utmos.py:29-203`; `tasks/tts.py:325-446`).

## Separate `tts_serving` scenario and numerical contract

`BenchmarkSpec` requires an absolute HTTP(S) base URL, nonempty model, integer seed, and test type `engine|e2e|external`. Only profile `stress` exists. Default implicit load is 100 requests, concurrency 8, infinite request rate, timeout 120, and all five endpoint families. Explicit load modes are closed/open/ramp/burst/soak/scheduled; open/ramp require finite rates, ramp requires a finite start rate, soak derives rate as request count/duration, and scheduled arrivals come exclusively from strictly increasing offsets with six named workload contracts and exact collision cohorts (`tts_serving/spec.py:102-539,693-928`).

Scenario construction first creates required coverage cases, round-robin interleaves endpoint groups, then fills to at least the requested count with a seeded weighted mix; required coverage can therefore exceed `request_count`. It covers speech languages, six formats, three task types, speed 0.25/1/4, codec first-chunk override, SDK success/error, six length extremes including 4096 valid and 5120/12544 invalid, long decode, reference shapes/failures, disconnect, streaming, batch 1/2/8/32/33 and item overrides, stateful voice operations, and WebSocket flows. Full symbol mechanics and constructors are in `tts_serving/scenarios.py:235-2549`; the exact constants and generated text corpora are at lines 14-232.

HTTP nonstream validates the entire body and parses optional usage headers as finite, nonnegative numbers. Streaming accumulates all reconstructed HTTP chunks, records TTFA/inter-chunk/count/first bytes, caps body size through the HTTP contract dependency, then validates the complete audio. A scenario's final `finish_timing` is idempotent by `completed_s`; RTF is set when duration is positive and RTF is not already positive. Network/timeout errors are classified transport errors; other client exceptions are failed/client-error records (`tts_serving/http_client.py:54-135,374-620`; `tts_serving/metrics.py:84-143`).

## Existing profiling and c16 hooks

The SeedTTS benchmark has no NVTX, Nsight-control, PyTorch-profiler, GPU-counter, or stage-batch hook. Its built-in observability is client timing, raw HTTP chunk timing, optional SGLang usage headers on buffered responses, managed server logs, saved per-request records, and server admission parameters. The managed launcher records the exact constructed command in logs but `--use-existing-server` records no resolved server configuration (`benchmark_tts_seedtts.py:351-411,1085-1162`; `benchmarker/utils.py:260-318`).

The tracked task-local diagnostics bundle adds three relevant short tools:

- `run_sweep.sh:1-34` never launches/stops a server. Defaults are CosyVoice3 checkpoint, port 8000, concurrency `1 2 4 8 16`, three repeats, 256 samples, warmup 32, streaming, and `results/cosyvoice_diagnostics`. It calls the existing benchmark generate-only against an existing server, creates one fresh directory per label/mode/run/concurrency, refuses reuse, and tees `console.log`. `buffered` simply omits `--stream`.
- `collect_env.py:1-115` is read-only with respect to server/model state. It records an allowlist of CUDA/thread/compiler environment variables, executables, package versions, Git identities/status/submodules, GPU/CPU/cgroup information, and SHA/factory signatures from six CosyVoice source files. It does not load the model and labels factory signatures as source defaults.
- `analyze_nsys_sqlite.py:1-171` opens exported SQLite read-only, chooses one supported populated CUDA kernel table unless explicitly selected, requires start/end/device for analysis, clips ordered intervals to the explicit window, unions overlaps using integer nanoseconds, and reports kernel count, invalid intervals, raw and union duration, overlap factor, uncovered duration, gap count/max/largest intervals. Its percentage is explicitly captured-kernel timeline coverage, not SM activity, occupancy, FLOP efficiency, or physical-device utilization.

The bundle's `experiment_template.json` reserves workload and evidence fields for concurrency, arrival model, request count, dataset/order hash, reference mix, streaming, generation parameters, warmup/measurement boundaries, first-audio definition, audio/TTFA/completion/goodput, actual stage batch distributions and tensor shapes, compiler/TRT fallbacks, counters, memory/queue depth, profiler-off results, Nsight report, trace window, device counters, and quality results. Those are empty record fields, not implemented benchmark outputs.

The existing explicit A/B server commands are eager (`enable_dit_torch_compile false`, `enable_flow_estimator_trt false`) and compiled (`enable_dit_torch_compile true`, TensorRT false). The documented c16 screening command uses 256 samples, warmup 32, streaming, generate-only, existing server. A full-dataset variant is the same command with `--max-samples` omitted. The diagnostic scripts do not calculate input order hashes or automatically coordinate profiler windows.

## Symbol inventory

The complete top-level symbol spans in the primary path are: `benchmark_tts_seedtts.py`: `_ModelBenchmarkProfile` 132-136; `_model_checkpoint_name` 154-155; `_profile_for_model` 158-161; `_parse_args` 164-172; `TtsSeedttsBenchmarkConfig` 176-226; generation/sample/send helpers 229-348; benchmark/transcribe/config parsers 351-528; `SustainedOvershootPlan` and overshoot/sweep functions 532-725; `benchmark` 728-733; parser/validation/main 736-1162. `benchmarker/data.py`: `RequestResult` 10-29. `benchmarker/runner.py`: `resolve_warmup` 23-29, `RunConfig` 33-42, `BenchmarkRunner` 45-153. `dataset/seedtts.py`: `SampleInput` 33-37 and loader/path helpers 43-214. `metrics/performance.py`: token/speed calculation and persistence formatting helpers 80-422. `metrics/playback_continuity.py`: its three functions 11-82. `metrics/wer.py`: `SampleOutput` 22-37 and all calculation/printing functions 40-367.

`tasks/tts.py`: WER persistence 79-142; similarity protocol/run 145-314; UTMOS protocol/run 317-446; endpoint/transcription protocols and pipeline 454-632; legacy voice clients 640-959 (not called by this entry point's generation); active payload/HTTP/audio/persistence helpers 967-1398. `tasks/asr.py`: normalization/local WAV/router helpers 61-281; WER application 284-316; active async ASR send/stream/runner 319-454; reusable evaluation/consistency assembly 457-662. `metrics/speaker_similarity.py`: model blocks 31-259, scorer 262-327, audio/model loaders 330-344. `speaker_similarity_assets.py`: asset dataclass/cache/download/validation functions 75-404. `metrics/utmos.py`: cache functions 29-161 and scorer 164-203.

For the assigned serving files: `tts_serving/http_client.py` contains `run_http_scenario` 54-135, disconnect paths 138-341, response handlers 344-518, body/header/chunk/format helpers 521-620. `metrics.py` contains `ScenarioResult` 24-81 and duration/timing/status functions 84-143. `audio_validation.py` contains result types and all validators 52-350. `artifacts.py` contains error/output writers/serialization/redaction 39-217. `spec.py` contains spec dataclasses 102-539 and parsing/validation/hash/redaction helpers 542-928. `scenarios.py` contains `Scenario` 235-264, deterministic construction and corpus assignment 267-918, speech constructors 921-1605, batch constructors 1608-1757, voice constructors 1760-2289, and WebSocket/reference constructors 2292-2549. Imports, constants, module documentation, and comments occupy the remaining lines in each fully read file.

## Unresolved runtime and external boundaries

Source inspection does not establish dataset cardinality/content at runtime, Hugging Face provider state for unpinned alternate dataset IDs, server-side generation defaults when payload fields are omitted, effective existing-server configuration, codec/model chunk boundaries behind HTTP framing, cache hit rates, actual sample audio duration, GPU work/utilization, ASR/model numeric results, or whether external model/checkpoint assets match their expected behavior. `datasets`, aiohttp, JiWER, Whisper normalizer, zhon, s3prl/WavLM, torchaudio, scipy, ffmpeg, Hugging Face Hub, the target TTS/ASR servers, and `tts_serving` client/report dependencies are external or separately owned boundaries. The report makes no byte-for-byte numerical claim from source inspection.
