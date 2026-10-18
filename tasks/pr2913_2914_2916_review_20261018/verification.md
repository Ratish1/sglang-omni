# Verification and test admission

## Environment and source boundaries

All pytest and CUDA work ran on the assigned H200, node-radixark-16-0001, in the separate `sglang-omni-review2913-2914-2916` Podman container. No local pytest was run. Both owned servers and all probe processes exited; the final check showed 0 MiB GPU usage. The existing GPU lease and containers are retained.

The image used Torch 2.13.0+cu130, Transformers 5.12.1 and the project's pinned SGLang 0.5.21. Dependencies were installed with the Radix skill's system-Python editable-install command. No HF credentials, HF environment override, GPU-visibility override or virtualenv was used.

Unit suites ran on each exact author head. Independent probes are committed on the separate `review/pr2914-20261018` branch based on #2914; that branch changes review artifacts only. The codec-table probe is later executed against #2916 production code. Source branches are unchanged.

## Completed targeted suites

| Head | Cases | Outcome |
| --- | --- | --- |
| 2913 | `test_retract_prefill.py` and graph batch validator suite | 43 passed |
| 2913 | Decode-table/extend inputs, prepared payload handoff and prompt paths selected from `test_pipeline.py` | 8 passed |
| 2916 | Predictor CUDA graph and predictor kernel suites | 103 passed, 1 Ascend-only skip |
| 2916 | Codec-table lookup/loader, fresh ICL and frontend checkpoint tests | 5 passed |
| 2914 | ASR model suite and real HF config helper | 10 passed |
| 2914 | Text-RoPE/default config, full-prefill policy and sidecar cases | 4 passed |

Total: **173 passed, 1 unrelated accelerator skip**. These are targeted suites, not a claim that the entire repository test suite passed. Warnings from the image's optional torchada import and upstream qwen-tts docstring/deprecation checks were not mistaken for PR failures.

## Independent CUDA checks

`rope_probe.py` checks the equal-axis derivation at FP32, then compares real BF16 CUDA RoPE to a separately written rotation equation using the FP32 angle cache. Positions cover zero, neighboring values, cache/page-adjacent values, the serving-context end and the checkpoint position-table end. Both int32 and int64 indices are exercised.

The fused K/V operation is captured and replayed three times per index type while positions and nonconsecutive output slots change. Rotated Q/K satisfy an operand-scaled BF16 rounding bound, stored K equals returned K, stored V equals input V, and untouched cache rows retain their sentinel. Maximum observed absolute errors against the FP32 equation were 0.00779/0.00775 for Q/K with int32 indices and 0.00781/0.00781 with int64 indices. These are output-rounding observations, not a model-quality tolerance.

`native_backend_probe.py` runs the actual shared attention method, real Q/K normalization, real CUDA RoPE and a real BF16 MHA KV pool. Only the surrounding attention/batch/backend wiring is reduced. It confirms old MRoPE/native and new text/CUDA success, new text/native failure, and success after disabling only the unsupported fused-store argument. The exception is asserted by its exact native-RoPE contract, so an unrelated setup error cannot satisfy the finding.

`serving_probe.py` launched Qwen/Qwen3-TTS-12Hz-1.7B-Base with full and breakable prefill graphs, sequentially on the same H200. Each launch completed the built-in pipeline warmup, then served one buffered request and a concurrent group of three unequal-length requests using both PCM streaming and buffered WAV. The synthetic tone reference intentionally exercises ICL preparation; each explicit request has a 32-frame cap. Both launches returned valid nonempty audio for all four requests and shut down normally.

The model-info responses attest the actual graph backends: FullCudaGraphBackend and BreakableCudaGraphBackend. Each reported four prefill replay calls, zero standard/custom eager prefills, live embedding slots, and zero predictor capture failures/eager predictor calls. Replays reached multiple buckets, including small cached-prefix extends. Counters include the launch's warmup and are not latency measurements. Startup used a four-request/512-token-chunk policy to keep this a targeted lifecycle check, not a claim to have measured production concurrency throughput.

`codec_tables_probe.py` then ran against the exact #2916 head. It loaded the real Base checkpoint into the standalone frontend and checked all 15 residual matrices against their original safetensors names. The actual shape was `(15, 2048, 2048)` in BF16. Vocabulary-edge lookups matched per-codebook lookups; lookups also followed a Parameter.data storage rebind. The checkpoint weights were restored before comparing prompts.

The prompt reference is the actual pre-PR `generate_icl_prompt` method extracted from merge base `76de3ff8f2ee98cb96c989e7d89527dc3d7d263f`, with the old embedding-module layout backed by the same checkpoint rows. Three cases cover text longer than the reference, reference longer than text, and non-streaming ICL concatenation. Prompt and trailing-text outputs were bit-identical in all three. This checks reduction order and alignment at checkpoint width, not just a small synthetic table.

Exact summaries, hashes, graph counters and request results are in [evidence.json](evidence.json). Raw logs and full model-info responses remain under `/data/review2913-2914-2916/` on the node. No raw audio or profiler trace was copied locally.

## Reproduction

Use the Radix skill's per-task container setup and editable-install command. The standalone probes require CUDA. On the artifact branch based on #2914:

```bash
python3 -m tasks.pr2913_2914_2916_review_20261018.rope_probe
python3 -m tasks.pr2913_2914_2916_review_20261018.native_backend_probe
python3 -m tasks.pr2913_2914_2916_review_20261018.serving_probe \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --run-directory /data/review2913-2914-2916/pr2914-full \
  --prefill-backend full
```

Use `--prefill-backend breakable` and another run directory for the second serving check. Preserve `codec_tables_probe.py` under the run directory, check out #2916 and reinstall the editable project before running that script with the same model ID. It intentionally reads the pre-PR method from the pinned merge base; it is a review oracle, not a proposed permanent test that depends on Git history. The native-backend probe succeeds when it reproduces the precise known regression and its controls; its exit code alone must not be read as the affected path being fixed.

## Test admission assessment

| Test concept | Why it is admissible; future regression it catches |
| --- | --- |
| Missing decode feedback | Critical negative contract: restoring bare-codec fallback would produce audio from an invalid model input. |
| Text and pad staging/history | Queue/history bookkeeping: dropping the pad contribution, consuming the wrong row or retaining a reused buffer corrupts later decode/re-prefill. |
| Capacity probe | Registry/bookkeeping: the model buffer layout changes and the graph validator can no longer read the actual staging capacity. |
| Indexed checkpoint tables and missing table | External serialization contract: off-by-one mapping, dropped codebook or one-table-counts-as-all-loaded errors. |
| Vocabulary-edge embedding lookup | Derived addressing property: wrong codebook stride, offset or flatten axis crosses into another table. |
| Fresh ICL prompt comparison | Derived sequence/alignment property: changed row placement or reduction grouping breaks the actual prompt. |
| Predictor graph/eager and fused/unfused comparison | Numerical/capture contract: graph buffer reuse, gather layout or stale storage silently changes generated codes/embeddings. |
| Real config normalization | Critical external-config contract: legacy aliasing or root initialization leaves SGLang constructing MRoPE. |
| Suggested native-backend regression | Reproduced before/after failure: class-only fusion eligibility sends unsupported arguments to the selected native operation. |

No test request is made for a private constant, wall-clock speed threshold, a hardware we cannot exercise, or an impossible request type. No fake-only happy-path test is proposed. CI's normal unit environment includes qwen-tts in its image and inherits system site packages; missing qwen-tts was considered and rejected as a finding about these new config cases.

## Limits

The PRs' full SeedTTS corpus, WER, speaker-similarity and profiler tables are author-reported evidence. This review read and assessed them but did not rerun those corpora. A synthetic-reference serving probe cannot establish WER/speaker similarity. The numerical proof for #2916 is stronger than an unseeded audio sample comparison; #2914 changes precision and cannot claim byte identity to old serving from equal-axis algebra alone.

No latency/throughput gain is independently claimed. Small changes in the PR tables need noise controls before being described as established gains. No MPS, XPU or NPU execution claim is made; their changed paths were checked by source inspection only. Qwen3-TTS IPC remains audit-only, so no supported multi-replica IPC claim is made.
