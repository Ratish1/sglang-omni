# Mechanical decomposition and source audit

## Revision boundaries

- GitHub head: be255e7ea058af8c83a0ca0a36adefa919388121.
- PR parent and merge base: d07df0b4339cf6efcb6cec1d0da7e7ecd4b80ddf.
- Target main snapshot: 369776998166e4114b4c6868f1b1069fd9a06282.
- Local SGLang checkout: baa9befa381511cd4d56f83980951e5a45fffc6d; the inspected sampler and model runner files are clean.
- All four changed files were read completely, including the full 1488-line stages module and 442-line thinker wrapper. The diff has 53 additions and 9 deletions. Unchanged code in those files was checked for interactions, not treated as new review findings.

## Change-by-change audit

| Mechanical part | Exact change | Review result |
|---|---|---|
| Stage backend selection | CUDA selects FlashInfer, other platforms select PyTorch | Correct placement; incoming engine overrides still win; exposes F1 |
| Load-hook dependency | Imports the shared compilation types and resolved runtime execution context | Concrete dependencies; no model-to-model import |
| Load-hook ordering | Calls the sampler helper before the existing MoE backend guard | Correct for both Triton and other MoE backends |
| Load-hook inputs | LM-head device, text vocabulary size, resolved backend predicate | Matches sampler device/vocabulary; no new dtype override required |
| Existing CPU MoE test | Publishes the PyTorch backend explicitly | Avoids relying on ambient backend configuration |
| Accelerator cache test | Clears the real sampling module cache, invokes the real hook, checks selected and unselected backends | Valid startup bookkeeping; does not replace the compilation helper with a fake |
| Budget/factory tests | Expected backend follows the selected platform | Preserves the existing memory-budget assertions; existing platform-gate parametrization also exercises CUDA and non-CUDA defaults |

## High-level request flow

1. `serve/protocol.py` accepts optional temperature, top-k, top-p, min-p and seed fields. `serve/openai_api.py:1339` forwards them into the client sampling object.
2. `client/types.py:95` owns optional sampling values; `SamplingParams.to_dict` omits unset values. `client/client.py:1107` packages stage parameters.
3. `models/qwen3_omni/request_builders.py:686` builds the real thinker request. It defaults temperature to zero, top-p to one, top-k to unrestricted, and min-p to zero. It normalizes and validates SGLang SamplingParams. Min-p is neither discarded nor forbidden for unseeded requests. The cookbook documents it as a thinker parameter.
4. The stage factory builds server arguments, respecting operator overrides. `bootstrap.create_thinker_scheduler` constructs the SGLang worker, shared request/cache infrastructure, thinker runner and output adapters.
5. Model loading invokes `Qwen3OmniThinkerForCausalLM.precompile_kernels_after_loading`. SGLang synchronizes around this hook. The new call loads the selected FlashInfer sampling module; the old MoE variant compilation remains unchanged.
6. `ModelWorker` reseeds the worker RNG after model-runner initialization. Load-time sampling does not determine a request's explicit seed. Each TP rank executes this lifecycle, using the broadcast worker seed.
7. Prefill and decode produce thinker logits. The shared LogitsProcessor trims the padded vocabulary and returns float32 logits. It does not leave the thinker's BF16 LM-head output as BF16 probabilities.
8. The synchronous runner applies token suppression and installed request seeds before SGLang ModelRunner.sample. The latter applies penalties/bias and invokes its registered sampler. Text-only lookahead invokes the same underlying sampler at launch; requests with explicit seeds, rollout logprobs or history-dependent sampling are gated to synchronous execution. Speech requests also use the synchronous thinker path.
9. IDs are copied to the existing pinned staging buffer and published to the next decode step. Completion events cover the sample and copy. The PR changes no transport, ownership, event, cache or output accounting code.
10. Thinker text streams to the detokenizer and, for speech requests, the talker/code2wav pipeline. The thinker draw happens after the model forward/graph replay, rather than inside a captured talker sampling loop. Existing talker comments about captured RNG are not evidence of that problem in this thinker change.

## Low-level dispatch matrix

| Batch condition | Selected draw after this PR | Effect |
|---|---|---|
| Every row greedy | SGLang argmax | Unchanged |
| Temperature only, no filters, no request seed | SGLang plain PyTorch draw | Unchanged algorithm |
| Filters and no request seed, min-p absent | FlashInfer joint top-k/top-p rejection kernel | Intended optimization |
| Filtered batch contains an explicit seed | Omni FlashInferSeededSampler's PyTorch fallback | Same per-row seeded Gumbel draw as before |
| Mixed seeded/unseeded rows | Rank-shared request-derived seeds for unseeded rows; whole batch takes seeded fallback | Existing shared behavior retained |
| Global deterministic-inference mode | Resolved backend becomes PyTorch | Existing SGLang policy retained |
| Any row has min-p | Sequential top-k renormalization, top-p renormalization, min-p draw | F1: changes joint support for the entire batch |

SGLang's SamplingBatchInfo computes need_min_p_sampling using any row, and ORs it when batches merge. The effect on a zero-min-p neighboring row follows the actual batching contract.

## Kernel and numerical checks

The installed FlashInfer 0.6.18 joint kernel uses a CTA per output row, vectorized probability reads, reductions over the number and mass of probabilities greater than its pivot, and rejection until both constraints hold. It neither requires a full sort nor hard-codes the thinker's vocabulary or an H200-specific threshold in this PR. Its probability and mass arithmetic is float32.

The pinned kernel reads seed_arr[0] and uses the block index as a Philox subsequence. Passing request seeds directly to it would not preserve Omni's per-request seed/position draw. The existing wrapper correctly retains SGLang's compiled float64 Gumbel path, whose noise comes from the request seed, token position and vocabulary column via MurmurHash. That wrapper is registered before SGLang constructs the sampler.

The inspected CUDA sgl_kernel top-k/top-p renormalization wrappers delegate to FlashInfer. The min-p kernel samples proportional to surviving weights above max_probability × min_p. Consequently, it cannot restore a token already removed by applying top-p to the top-k-renormalized distribution. This establishes the F1 mechanism independently of random output samples or the support-capture implementation.

FlashInfer's threshold keeps equal probabilities at a cutoff. The existing shared test deliberately checks this. A tied maximum can therefore choose a different equal maximum in a mixed batch; this review records that distinction without calling it a lower-probability or quality failure.

The official [FlashInfer joint-sampling documentation](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_probs.html) describes the rejection algorithm and distinguishes joint filtering from top-k-first filtering. That online page is newer than the installed package; kernel/seed conclusions here come from the installed 0.6.18 code, not its newer seed API documentation.

## Sources read

Complete model-local production files: stages.py, components/sglang_thinker.py, thinker_model_runner.py, shared model_runner/thinker_model_runner.py and model_runner/base.py. Complete changed test files: test_sglang_thinker.py and test_sglang_ar_budget.py. Complete shared seed/backend test file: test_install_sampling_seeds.py. Complete shared sampler_compile.py and SGLang layers/sampler.py.

Boundary/caller reads: request builder and its SamplingParams normalization/verification, API sampling forwarding, client serialization, config topology/warmup/defaults, thinker bootstrap, SGLModelRunner registration/construction, ModelWorker initialization/reseed, generation override merging, SGLang ModelRunner.sample, LogitsProcessor FP32 publication, SamplingBatchInfo construction/merge, deterministic backend policy and load-hook invocation. These are targeted sections of large unchanged infrastructure files; this review does not claim to have read all of SGLang or every unrelated kernel.

Complete relevant sampling wrappers and kernels read: local SGLang sampling exports/softmax dispatch, temperature_softmax.py, murmur_hash.py, aot/sgl_kernel/sampling.py; installed FlashInfer joint wrapper, get_sampling_module joint operator wrapper, seed/offset helper, full TopKTopPSamplingFromProbKernel and MinPSamplingFromProbKernel; top-p renormalization cutoff implementation. Unchanged MoE, vision and audio neural-network kernels are outside this PR's changed mechanics.

The local SGLang sampler includes a max_top_k optimization that is absent from the installed 0.5.21 wheel. The deployed-wheel measurements are therefore explicitly reported as such, not as performance against every newer SGLang source change. Both implementations retain the filtering-order distinction in F1. Source hashes are retained in the evidence files.
