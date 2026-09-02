# 11. Where each model samples, and where the Higgs fusion (#816) carries over (2026-09-02)

Read at main 332f4fc67. #816 (higgs_tts/sampler.py:253-273) replaced a
full vocabulary torch.sort plus cumsum masking with sgl_kernel
top_k_renorm_prob and top_p_renorm_prob, then a multinomial (seeded rows
through multinomial_with_seed on log probs). It is the only use of the
sgl_kernel sampling kernels in sglang_omni. Its measured gain on Higgs was
64 to 32 us per step, 1.7 percent of a 4.1 ms step.

## 1. Who owns the draw

| model | draw | ops per step | in a graph | vocab x codebooks per step | seeded |
|---|---|---|---|---|---|
| higgs_tts | own, sgl_kernel renorm plus multinomial | softmax, top_k_renorm_prob, top_p_renorm_prob, multinomial | yes, SGLang decode graph | 1026 x N codebooks | yes |
| qwen3_omni talker layer 0 | SGLang Sampler called from model code (talker.py:1327-1363), torch backend | rep penalty and suppress mask in the model, then top_k_top_p_min_p_sampling_from_probs_torch | yes | text_config.vocab_size x 1 | yes, seeds always passed |
| qwen3_omni code predictor | own | argmax | yes, own graph | cp vocab x (groups minus 1) | greedy |
| qwen3_tts semantic | SGLang Sampler, torch backend | SGLang torch path | yes | vocab x 1 | yes |
| qwen3_tts sub talker | own Triton kernel under capture, torch topk or sort fallback eager | fused topk, bitonic sort, cumsum, murmur3 Gumbel | yes, own graph | 2048 x (groups minus 1) | yes |
| moss_tts graph path | own torch plus own Triton Gumbel | topk, sort, softmax, cumsum, scatter, seeded_gumbel_argmax | yes, sampling graph per bucket | audio vocab x n_vq plus control | yes |
| moss_tts_local | own Triton fused (vocab up to 2048), torch branchless above | tl.sort, tl.cumsum, murmur3 Gumbel | yes, 13 passes per frame captured | 1025 x 12 plus a text channel | yes |
| zonos2 | own, pure torch | scatter_add rep penalty, topk, sort, softmax, cumsum, scatter, min p, multinomial | eager by default, opt in tail graph | 1026 x 9 | no |
| fishaudio_s2_pro | own, pure torch | sort for RAS, gather and scatter rep penalty, topk 30, softmax, cumsum, multinomial | yes | LM vocab topk 30, then argmax heads | yes |
| minimax_music3 | own, pure torch | nan_to_num, topk 50, masked_fill, multinomial_with_seed | depth pass yes, c0 eager | 16385 for c0, audio vocab x 7 depth | yes |
| voxtral_tts | own | argmax plus flow matching | eager | greedy | no |
| ming_tts, ming_omni talker, dots_tts | own, continuous | flow matching, no categorical draw | tail graphs | none | n/a |
| fun_cosyvoice3, thinkers, llada2_uni, every ASR | SGLang Sampler | torch backend where the stage pins it, else the engine default | yes | LM vocab x 1 | yes |
| audar_tts | llama.cpp | inside llama.cpp | no | n/a | yes |

## 2. What the map says

Three models already wrote their own fused sampler in Triton (qwen3_tts
sub talker, moss_tts, moss_tts_local) and did not use sgl_kernel. The
reason is visible in the code: the sgl_kernel renorm kernels take no
seed, and these samplers fuse the seeded draw itself (murmur3 Gumbel keyed
by request seed and position). Higgs kept the seeded draw separate
(multinomial_with_seed after the renorm), which is why sgl_kernel fit
there. So the repository holds three near duplicate seeded top-k/top-p
Gumbel kernels plus the Higgs recipe.

Where the #816 recipe (renorm kernels, then the model's own draw) applies
without a new kernel:

- zonos2: the same shape as Higgs (1026 wide, 9 codebooks, sort plus
  cumsum plus scatter per step), unseeded, so a plain multinomial after
  the renorm. Its rep penalty and min p stay in torch. Eager today, so
  the sort is also launch bound.
- qwen3_omni talker: the draw is SGLang's torch path over the talker
  vocabulary, but the call site is model code (talker.py:1363), so the
  model can renorm with sgl_kernel and draw with multinomial_with_seed
  exactly as Higgs does, no upstream change. The penalties and the
  suppress mask already sit before the draw.
- fishaudio_s2_pro and minimax_music3 already narrow to a fixed top k of
  30 and 50 before the softmax, so a renorm kernel would replace one topk
  call, not a sort. Small.
- The moss and qwen3_tts Triton paths are already fused. Their torch
  fallbacks (moss text channel above 2048, qwen3_tts sub talker outside
  capture) are where the recipe would apply.

Where the lever is SGLang's backend flag, not model code: the thinkers,
qwen3_tts semantic token, fun_cosyvoice3 and the ASR stages sample
through SGLang's Sampler. The Qwen3-Omni stages pin sampling_backend to
pytorch (stages.py:1035, 1158) because the flashinfer path asserts that
no seed is set (sampler.py:270-273 in 0.5.18). For unseeded traffic the
flashinfer path works. Flipping the flag needs a policy for seeded
requests on that stage (reject, or serve them through the torch path in
a separate batch), which is omni's decision and not an sglang patch.

## 3. Before any kernel

The Higgs gain was 1.7 percent of its step. Nothing here should be built
before the sampler's share of the decode step is measured per model with
the torch profiler at the CI concurrency: one number per model, the
sampler kernels' GPU time over the step time. The candidates with a share
worth the work are the ones whose sort runs over a wide vocabulary or
many codebooks per step: the thinker (152k vocabulary times the batch,
through the torch backend), zonos2 (9 sorts of 1026 per step, eager), and
the Qwen3-Omni talker if its vocabulary is wide. Task: that profile, then
the order.
