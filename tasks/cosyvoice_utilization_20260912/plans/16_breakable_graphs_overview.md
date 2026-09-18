# Plan 16, overview: breakable CUDA graphs across Fun-CosyVoice3

Written 2026-09-18 as the map for two whole file research passes. It holds only what is already
verified; the design is written after the research comes back. Trees: pinned SGLang `v0.5.19`
(`/Users/ratish/sglang`), omni `upstream/main` 27b5b0d4f.

## 1. The model, part by part

| part | per request | today | launches, busy over wall (stage 1 ledger, H100) |
|---|---|---|---|
| preprocessing | one reference encode, ONNX and CPU | not a CUDA graph subject | |
| AR prefill | one forward | eager: the runner owns it through `custom_prefill_forward` and returns `can_run_cuda_graph=False` (`fun_cosyvoice3/model_runner.py:53-61, 406-409`) | 350 to 440 launches, 10 to 18 ms, 0.14 to 0.42 |
| AR decode | one forward per token | SGLang's full decode graph, batch 1 to 32 | 51 to 55 launches, 2.2 to 2.8 ms |
| Flow DiT, hops and finals | ten Euler steps, 22 blocks each | eager on the packed path; the one Flow graph is the whole padded solver per exact (batch, frames), 55 shapes, reached by buffered traffic only (`stages.py:342-517, 656`) | about 19,000 launches per call (15,000 after the weight cast), 0.41 at 1 row, 0.87 to 0.99 at 16 rows |
| HiFT | once per request inside each vocoder step | eager, whole history recomputed | 1,196 launches, 0.34 on the typical call |

Serving weights at c16 streaming (readout 03): hop Flow 58.5, final Flow 20.5, HiFT 20.7 percent
of vocoder step time; the AR is about 8 percent of GPU time per request.

## 2. What SGLang ships

Two breakable graph stacks, one primitive under both (`eager_on_graph`, segments sharing a pool):

| stack | files | subject | key, padding |
|---|---|---|---|
| SRT | `srt/model_executor/runner_backend_utils/breakable_cuda_graph/{breakable_cuda_graph,context,cuda_utils}.py`, `runner_backend/breakable_cuda_graph_backend.py`, `runner/prefill_cuda_graph_runner.py` (1,921 lines), `runner/{base_cuda_graph_runner,shape_key}.py` | LLM prefill; attention is the break, metadata rebuilt eagerly per replay | `num_tokens`, round up to a bucket, 2x waste cap, one request slot |
| multimodal_gen | `multimodal_gen/runtime/breakable_cuda_graph/{runner,replay_token,prompt_padding}.py`, `model_padders/{qwen_image,zimage,minimax_h3,ideogram,longcat_image,sana_video}.py`, `runtime/layers/attention/layer.py:1859-1897`, driven from `runtime/pipelines_core/stages/denoising.py` (`_maybe_get_bcg_runner`, `_bcg_run`, `_bcg_pad_prompt_kwargs`, `:2367-2430`) | a DiT forward inside a denoising loop; every attention module is a break | exact input signature; variable prompt lengths are padded to buckets by per model padders before the call (`DEFAULT_BCG_TEXT_BUCKETS = (64, 128, 256, 512, 1024)`) |

## 3. What omni already does with them

| model | use |
|---|---|
| Higgs, Fun-ASR, ArkASR | SRT breakable prefill: `supports_breakable_prefill_cuda_graph = True`, `cuda_graph_backend_prefill = BREAKABLE`, embeddings attached through `attach_omni_prefill_inputs` (`engine_factory.py:165-174`, `model_runner/prefill_inputs.py`, `higgs_tts/model_runner.py:84-95`) |
| MiniMax Music3 | the multimodal_gen runner over its DiT, one fixed `mel_len`, no padding, off by default, free memory guard (`minimax_music3/dit.py:372-407`) |
| about twenty others | full (unbroken) graphs over encoders, codecs, vocoders and flow tails, through `torch.cuda.graph` or `platforms/device_graph.py` |

## 4. The two research cuts

R1, the AR side: the SRT stack whole, how omni wires a model into breakable prefill, Higgs as the
worked example, and everything Fun-CosyVoice3's AR path does that the standard path does not
(why it owns prefill, what its embeddings and streaming outbox need).

R2, the vocoder side: the multimodal_gen stack whole including the padders and the denoising
stage that drives it, a DiT that uses it, MiniMax in omni, the vocoder graph precedents in omni,
and Fun-CosyVoice3's Flow and HiFT paths whole, with every host side operation in a Flow step and
a HiFT call listed.

Both report mechanisms with lines, no design.
