# Vendored dots TTS

This directory contains the dots TTS inference components used by the native
sglang-omni dots pipeline.

## Source

- Upstream project: <https://github.com/rednote-hilab/dots.tts>
- Public checkpoint family: `rednote-hilab/dots.tts-base`
- Model card license: Apache-2.0

## Serving Path

```text
Omni request/state
  -> DotsTTSNativeAdapter.prepare_inputs
  -> DotsTtsSideRuntime._prepare_inputs
  -> DotsTtsSideModel.prepare_request
  -> DotsTTSSGLangModel.forward/decode_audio_batch
  -> DotsTtsSideModel.decode_audio_batch_step
  -> DotsTtsNativeVocoder.decode
```

`DotsTtsSideModel` is the boundary between sglang-omni integration code and
vendored dots internals.

## File Boundary

Mostly upstream inference components:

- `models/dots_tts/core.py`
- `models/dots_tts/config.py`
- `modules/backbone/*`
- `modules/speaker/*`
- `modules/vocoder/*`
- `data/pipelines/*`
- `utils/*`

sglang-omni integration layers:

- `side_runtime.py`
- `vocoder_runtime.py`

## Removed Upstream Runtime

`models/dots_tts/model.py` does not expose the upstream full-runtime request loop.
The native Omni pipeline owns scheduling, latent streaming, and vocoder staging.
The following upstream surfaces are intentionally removed:

- `generate_audio`
- `generate_audio_stream`
- `_generate_latents_stream`
- upstream top-level prefill/decode loops

The vendored model still keeps checkpoint/model assembly helpers and the prompt
conditioning, FM state, DiT/flow, and patch-encoder primitives used by
`DotsTtsSideModel`.
