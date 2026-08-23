# Qwen3-TTS mechanics and decisions

## Current sampling path

`Qwen3TTSModelRunner` shapes codec logits, then calls SGLang's sampler. Standard
repetition penalty is currently applied in both places. For a previously emitted
token, `1.05` therefore behaves as `1.05² = 1.1025`; `1.1` behaves as `1.21`.
Codec-vocabulary suppression is Qwen-specific and must remain in Omni.

The intended ownership is:

- SGLang sampler: repetition, frequency, presence, min-new-token, top-k/top-p,
  temperature, RNG, batch filter/merge, and emitted-token history.
- SGLang-Omni Qwen adapter: valid codec-vocabulary suppression and Qwen model
  execution only.

## Retraction blocker

The pinned SGLang 0.5.16 `SamplingBatchInfo.from_schedule_batch` constructs a
fresh penalizer orchestrator after re-prefill. Repetition starts as `[B,V]` ones,
frequency/presence as `[B,V]` zeros, and min-new-token length as `[B,1]` zeros.
Normal decode then cumulates only the latest token. Qwen3-TTS retains
`req.output_ids` across retraction because it does not set SGLang's
`Req.input_embeds`; Omni separately replays its projected embedding history.

Consequently, deleting Omni's duplicate repetition code first would forget the
pre-retraction repetition set. The current duplicate accidentally hides a
generic SGLang output-history restoration gap.

The dependency order is:

1. Add a public SGLang batch-construction contract that initializes all
   output-history penalizers from each retained `req.output_ids`. This must cover
   repetition membership, frequency counts, presence bits, and generated length.
2. Prove first prefill is unchanged, while re-prefill/merge/filter restore the
   same state as uninterrupted decode.
3. In Omni, split codec suppression from the coupled repetition mask and remove
   Qwen's repetition hook/state.
4. Prove exactly one sign-aware repetition transform, codec suppression parity,
   fixed-seed token parity, and retract/re-prefill parity before benchmarking.

Do not mutate SGLang private penalizer tensors from the Qwen runner and do not
neutralize public repetition parameters before calling SGLang. Both approaches
put standard sampling ownership in the wrong layer.

## Verified synchronization work

The local 0.6B detector and traces proved these candidate paths execute without
the selected blocking-copy mechanism after their rewrites:

- sampling metadata;
- speaker mel and cached speaker embedding;
- prompt token rows and reference code;
- codec-suppression mask rebuild mechanics; and
- text-tokenizer IDs.

Only the production mechanism for one row is moved to each implementation
branch. Profiler ranges, detector controls, trace analyzers, reports, and this
task workspace stay local.

## Next unresolved owners

1. Vocoder tokenizer decode: scalar D2H, code H2D, and waveform D2H.
2. Reference tokenizer encode: scalar length reads, pageable H2D, and cache
   publication wait.
3. Cache-key and cached-speaker publication D2H.
4. Final engine-code stage handoff.
5. Other model families, using the same inventory/ownership/rewrite gates.
