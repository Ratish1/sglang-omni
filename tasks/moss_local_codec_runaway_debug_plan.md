# MOSS-TTS Local codec ownership runaway debug plan

## Problem

The `perf/moss-local-codec-ownership` branch produced four successful SeedTTS outputs with `completion_tokens=2048` and `audio_duration_s=163.84`. This is not a vocoder length bug: `2048 / 12.5 fps = 163.84s`, so the AR local transformer failed to emit/recognize `audio_end` and stopped only at `max_new_tokens`.

## Current-state evidence

- PR review TODO was about codec ownership in `streaming_vocoder.py`: stop getting the codec from `processor.audio_tokenizer` and stop monkey-patching `codec.decoder` at decode time.
- Current PR changes more than non-streaming decode ownership:
  - `sglang_omni/models/moss_tts_local/audio_tokenizer.py` implements a separate reference-audio encode wrapper.
  - `sglang_omni/models/moss_tts_local/stages.py` uses that wrapper for preprocessing reference codes and a separate codec model for vocoder decode.
  - `sglang_omni/models/moss_tts_local/request_builders.py` feeds the new reference encoder output into `processor.build_user_message`.
- Upstream MOSS v1.5 processor code in `moss_tts_local_v1.5/processing_moss_tts.py` encodes references by:
  - loading with `torchaudio.load`
  - resampling to `model_config.sampling_rate`
  - expanding mono to stereo, truncating >2 channels
  - loudness-normalizing to `-20 dBFS` with gain clamp `[-3, 3]`
  - calling `audio_tokenizer.batch_encode(prepared, num_quantizers=n_vq)`
  - returning CPU long `[T, NQ]` codes
- The current local wrapper mostly mirrors this, but the failure is on AR stop behavior. Therefore the first invariant to prove is reference-code parity, not vocoder waveform parity.

## Boundary map

```text
Request text + reference audio
        |
        v
preprocessing stage
  processor.build_user_message(reference=[reference_codes])
        |
        v
AR local transformer / model_runner
  emits codec rows until audio_end or max_new_tokens
        |
        v
vocoder stage
  decodes rows to waveform
```

Changed owner in this PR:

```text
Before: processor.audio_tokenizer owns both reference encode codec and vocoder decode codec.
After:  stages.py loads separate codec instances:
        - preprocessing reference encoder codec
        - vocoder decode codec
```

Failure location:

```text
AR local transformer generated 2048 frames.
Vocoder decoded exactly what AR emitted.
```

## Violated invariant to test first

For every reference audio used by a request:

```text
processor.encode_audios_from_path([path], n_vq=n_vq)[0]
== or acceptably matches
MossTTSLocalAudioTokenizer.encode_paths([path], num_quantizers=n_vq)[0]
```

If this fails for the four runaway sample references, the branch changed AR conditioning. That is the likely root cause.

## Execution plan

### Phase 1: decisive parity probe on runaway references

Add a temporary debug script on this debug branch only:

`tmp/moss_local_codec_debug/compare_reference_codes.py`

Inputs:
- model path: `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`
- list of sample IDs or direct reference audio paths
- device
- optional batch grouping mode: `single`, `batch-all`

It should load two paths:

1. Upstream-style reference processor:
   - use the HF processor with codec loaded, or construct processor + audio tokenizer exactly as upstream does.
   - call `processor.encode_audios_from_path(paths, n_vq=n_vq)`.

2. PR separate-codec path:
   - use `load_moss_tts_local_audio_tokenizer(...)`.
   - call `encode_paths(paths, num_quantizers=n_vq)`.

Report per path:
- code shape
- token equality
- mismatch count and mismatch ratio
- max absolute token delta is not meaningful for IDs, but include unique mismatch positions by codebook
- first 20 mismatches `(frame, codebook, upstream_id, pr_id)`
- whether shape/length differs
- three comparison classes:
  - candidate single encode vs upstream single encode
  - candidate batched encode vs upstream single encode (the old serving behavior)
  - upstream batched encode vs upstream single encode (detects codec batch-sensitivity)

Run on:
- the four 163.84s sample references
- 10 normal samples from the same generated run
- the same paths in single-item batches and in the batch grouping used by the preprocessing coalescer

Exit criteria:
- If codes differ materially, root cause is reference encode parity. Move to Phase 2A.
- If codes match exactly, root cause is not reference encode tokens. Move to Phase 2B.

### Phase 2A: fix reference encode parity

Candidate fixes, in order:

1. Make `MossTTSLocalAudioTokenizer` call the upstream processor’s encoding methods through a codec-bearing lightweight processor instance for preprocessing only.
   - This preserves codec ownership separation from vocoder and avoids `processor.audio_tokenizer` in vocoder.
   - It removes reimplementation risk in audio loading/prep/trimming.
   - The preprocessing stage can own a codec-bearing encode helper while the request-building processor remains codec-free.

2. If we keep the wrapper, make it bit-equivalent to upstream:
   - exact `n_vq = processor._assert_fixed_nq(n_vq)` behavior if needed
   - exact channel handling and `to(device)` ordering
   - exact codec weight dtype / compute dtype / attention implementation settings from upstream processor load
   - exact batch grouping behavior if batch composition affects tokens

3. If batch composition is the source:
   - disable cross-request batching for reference encode by default, or
   - batch only duplicate paths, not unrelated references, or
   - add a config flag for reference encode batching with conservative default.

Review criterion:
- It is better to lose a little preprocessing throughput than create AR stop runaways.
- This TODO was about codec ownership, not changing AR conditioning.

### Phase 2B: if reference codes match, trace stop decision

Instrument only on debug branch:
- request ID
- prompt row count
- reference code length
- first generated stop-choice distribution if accessible
- final `completion_tokens`
- whether `audio_end_token_id` appears in text channel

Compare one runaway request against the last known good commit and current head.

Exit criteria:
- identify whether stop token generation differs before vocoder.

### Phase 3: production patch selection

Preferred production design if Phase 2A confirms reference parity failure:

```text
preprocessing:
  load reference encode codec independently, but reuse upstream processor encode contract
  no `processor.audio_tokenizer` dependency in request-building processor

vocoder:
  load decode codec independently
  keep packed SGLang non-streaming decoder path
  no monkey-patching of processor-owned codec
```

This satisfies the TODO without reimplementing the reference encode semantics incorrectly.

### Phase 4: validation gate

Run on H100:

1. Reference-code parity:
   - four runaway references
   - normal sample set
   - single and batched modes

2. Full c8 SeedTTS EN:
   - completed/failed = `1088/0`
   - no `completion_tokens=2048`
   - duration max back in normal envelope around dataset target, not `163.84s`
   - WER/UTMOS/speaker-similarity in previous envelope

3. Log proof:
   - processor loaded without codec for request building
   - preprocessing codec loaded independently
   - vocoder codec loaded independently
   - packed SGLang vocoder path active

## Immediate commands for H100 after script exists

```bash
git checkout debug/moss-local-codec-runaway
python tmp/moss_local_codec_debug/compare_reference_codes.py \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --sample-ids \
    common_voice_en_628642-common_voice_en_628644 \
    common_voice_en_25748530-common_voice_en_25748532 \
    common_voice_en_20005340-common_voice_en_20005341 \
    common_voice_en_19599129-common_voice_en_19599130 \
  --device cuda:0 \
  --out /data/moss_local_codec_runaway_reference_parity
```

## Risk register

- Reference codec batching can be faster but may alter discrete codec IDs under BF16. Detection: code parity probe and max-token runaway count.
- Explicit codec metadata resolution is semantically correct if it follows upstream `from_pretrained`; upstream code confirms it does.
- Full benchmark speed alone is not sufficient. Acceptance requires no max-token runaway and reference-code parity or a deliberate quality/speed tradeoff approved by maintainers.
