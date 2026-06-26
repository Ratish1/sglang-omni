# Vendored dots TTS

This directory contains the vendored dots TTS model components used by the
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
- `utils/*` (audio resample, tokenizer ids, profiling helpers)

sglang-omni integration layers:

- `side_runtime.py`
- `vocoder_runtime.py`

Generic preprocessing that is **not** vendored dots model math lives outside
this tree, under the package root:

- `../text_preprocessing.py` — text normalization (WeTextProcessing), language
  detection (lingua), and language-code resolution (langcodes). Previously sat
  at `utils/text.py`; moved out so this tree only holds dots model internals.

## Removed Upstream Runtime

`models/dots_tts/model.py` does not expose the upstream full-runtime request loop.
The Omni pipeline owns scheduling, latent streaming, and vocoder staging. The
following upstream surfaces are intentionally removed:

- `generate_audio`
- `generate_audio_stream`
- `_generate_latents_stream`
- upstream top-level prefill/decode loops
- upstream checkpoint authoring/IO (`from_pretrained`, `save_pretrained`,
  `load_pretrained_weights`, and the artifact state-dict helpers); the serving
  load path lives in `DotsTtsSideModel.from_pretrained`

The vendored model keeps the prompt conditioning, FM state, DiT/flow, and
patch-encoder primitives used by `DotsTtsSideModel`.
