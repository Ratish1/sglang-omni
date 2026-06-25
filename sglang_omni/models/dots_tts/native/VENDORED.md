# Vendored dots TTS Code

This directory contains the dots TTS inference components needed by the
sglang-omni native dots pipeline.

## Source

- Upstream project: <https://github.com/rednote-hilab/dots.tts>
- Public checkpoint family: `rednote-hilab/dots.tts-base`
- Model card license: Apache-2.0
- Source snapshot: the copied files do not encode an upstream git commit. Treat this
  directory as a vendored inference snapshot of the public dots TTS implementation used
  with the `dots.tts-base` checkpoint.

## Serving Path

The sglang-omni serving path is:

```text
Omni request/state
  -> DotsTTSNativeAdapter.prepare_inputs
  -> DotsTtsSideRuntime._prepare_inputs
  -> DotsTtsSideModel.prepare_request
  -> DotsTTSSGLangModel.forward/decode_audio_batch
  -> DotsTtsSideModel.decode_audio_step
  -> DotsTtsNativeVocoder.decode
```

`DotsTtsSideModel` is the boundary between sglang-omni integration code and vendored
dots internals. Integration files should call its public serving methods instead of
calling upstream private methods directly.

## Mostly Upstream Math And Runtime

These files keep the upstream inference math and model components close to their source:

- `models/dots_tts/core.py`
- `models/dots_tts/config.py`
- `modules/backbone/*`
- `modules/speaker/*`
- `modules/vocoder/*`
- `data/pipelines/*`
- `utils/*`

## Integration Files

These files are sglang-omni integration layers around the vendored components:

- `side_runtime.py`: owns request preparation, checkpoint loading for side modules, and
  the public side-model serving API.
- `vocoder_runtime.py`: loads only the native AudioVAE vocoder and decodes latent
  patches emitted by the Omni pipeline.

## Carried Upstream Surfaces

`models/dots_tts/model.py` still carries upstream full-runtime methods for provenance
and parity. They are not the sglang-omni serving path:

- `generate_audio`
- `generate_audio_stream`
- `_generate_latents_stream`
- upstream top-level prefill/decode loops
- `save_pretrained` / `from_pretrained`
- artifact state-dict helpers
- upstream warmup orchestration

The side runtime has its own warmup path and does not call the upstream full generation
API.

## Preprocessing Boundary Follow-Up

Generic audio loading/resampling, language detection, and text normalization overlap
with Omni preprocessing. They remain in `side_runtime.py` in this PR to preserve parity
with dots TTS inputs. Move them only after adding parity tests for prompt audio,
language tags, and text normalization.

## Known Native Limitations

- `tp_size > 1` is not supported.
- DiT/flow is not cross-request batched yet.
- async decode / overlap schedule remains disabled for dots because latent feedback is
  sequential.
- dots text/audio interleave double streaming is not supported.
