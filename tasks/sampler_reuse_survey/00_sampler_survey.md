# Sampler reuse survey (2026-09-01)

Question: should #816 (Higgs, sgl_kernel top-k and top-p renorm in place of
a sort based nucleus chain, merged 2026-06-17) be repeated for the other
models, and can the per model samplers be refactored onto sgl_kernel or
sglang's Sampler. Read on sglang-omni main 94fd2f272 and sglang release
v0.5.18 (71de97b264), with the pinned sgl-kernel 0.4.6.post1 and flashinfer
0.6.17.

## 1. What the pinned kernels expose

On CUDA, sgl_kernel/sampling.py defines two sampling functions,
`top_k_renorm_probs` and `top_p_renorm_probs` (aliases `*_renorm_prob`),
both thin wrappers that delegate to flashinfer when it imports
(sampling.py:28-62, 79-115, exported at __init__.py:107-110). Inputs are
forced to contiguous fp32 by the wrapper, per row thresholds come as a
tensor, the in tree fallback has no host read. There is no
`sampling_from_probs`, no min_p renorm, no penalty kernel on CUDA. The
`top_k_top_p_sampling_from_probs` and `min_p_sampling_from_probs` names are
imported from sgl_kernel only under the MUSA build (__init__.py:139-154),
sglang itself takes them from flashinfer.sampling (sampler.py:30-38).
Renorm keeps ties at the pivot (documented in the ROCm triton port,
renorm_triton.py:122-123, 163-165), so top-k can keep more than k tokens.
No test in the pinned tree asserts distributional equality with
torch.multinomial, only value closeness at 1e-3 and support membership
(aot/tests/test_sampling.py:104-184). #816's own equivalence claim
(4.8e-7 max probability difference, identical support, 225 configs) lives
in its commit message.

## 2. Where each model samples

Every package that reaches sglang's Sampler pins `sampling_backend` to
pytorch (qwen3_omni stages.py:1035,1158, qwen3_tts engine_builder.py:63,
moss_tts:31, moss_tts_local:56, zonos2:165, fun_cosyvoice3:59,
ming_tts:112, ming_omni, llada2_uni, voxtral, the five ASR builders), because
Sampler.forward does not pass a seed to flashinfer (base.py:845-850,
stages.py:1150-1153 on #408). The pytorch backend is the only device
agnostic sampling path.

| model | implementation | in a graph | per row knobs | launches per call (static op count) |
|---|---|---|---|---|
| qwen3_omni talker layer 0 | sglang Sampler from static buffers inside the model forward (talker.py:1284-1288, 1327-1391) | yes, the only graph captured Sampler call in either tree | temp, top_p, top_k, seed, sparse repetition penalty | about 7 omni plus about 10 sglang |
| qwen3_omni code predictor | argmax, 15 sequential sub steps (talker.py:1020-1027, 1607-1610) | yes | none | 15 argmax plus embeds |
| higgs_tts | #816 chain, fp32, B x N flattened (sampler.py:221-287) | yes | temp, top_p, top_k, seed | about 13 unseeded |
| qwen3_tts layer 0 | sglang Sampler via the base runner | no | sglang | |
| qwen3_tts sub talker | repo Triton kernel when captured (sampling_kernels.py:508), sort chain eager (sglang_model.py:1603-1646) | yes, own graph | temp, top_p, top_k, seed | 1 captured, about 16 eager, times codebooks |
| moss_tts | graph sampler with checkpoint default scalars (sampler.py:145-269), fused Triton for vocab <= 2048 (sampling_kernels.py:163-326), branchless torch (:329-375) | yes, own graph, exact default profile only | eager path per row | about 14 to 22 plus FSM ops |
| moss_tts_local | fused Triton per channel, 13 draws per frame in one graph per bucket (sglang_model.py:408-500) | yes | text and audio temp, top_p, top_k, seed, no penalty on the graph path | 13 launches captured |
| zonos2 | pure torch chain on B x C (zonos2/sampler.py:100-134), optional frame graph | opt in | temp, top_k, top_p, min_p, repetition penalty, no seed (reverted, sampler.py:126-129) | about 30 |
| fishaudio_s2_pro | custom in forward chain (sglang_model.py:389-481), top-p before temperature (:436-444) | yes | temp, top_p, top_k clamped to graph width 30, penalty, seed | about 45 plus 9 argmax |
| minimax_music3 | topk 50 plus seeded draw (rvq_decoder.py:168-188) | c1 to c7 in a graph | constant top_k, per row seed | about 6 times 8 |
| dots, voxtral, ming_tts, ming_omni talker | argmax or flow matching, no categorical sampling | | | |
| llada2_uni, ASR family, fun_cosyvoice3 | delegated to sglang | | | |
| audar_tts | llama.cpp | | | |

## 3. Reuse map

- #816's substitution transfers structurally to zonos2 (already fp32, flattened, torch.multinomial, graph capturable) and to the eager sub talker branch of qwen3_tts. zonos2 is blocked on min_p (no CUDA kernel) and on its penalty before temperature order (zonos2/sampler.py:103). qwen3_tts eager needs `log(renorm probs)` into `multinomial_with_seed`.
- moss_tts_local, moss_tts graph path and qwen3_tts under capture already run one fused Triton launch per draw, renorm plus multinomial would add launches.
- moss_tts eager carries two host syncs per step (model_runner.py:623, :688) that matter more than the sort, and its tie order is a bit exact contract (cub stable sort reproduced in sampling_kernels.py:191-198) which the renorm rule (ties kept at the pivot) breaks.
- fishaudio applies top-p to unscaled logits, a renorm port is a behavior change.
- minimax has a constant k of 50 and needs logits for the seeded draw.
- The talker's sampler is not where its step time is (about 17 of 1890 launches), the code predictor sub steps are.

Cross cutting: nine packages depend on `multinomial_with_seed`, Gumbel-max
over logits (sampler.py:687-729), while renorm returns probs
(moss_tts/model_runner.py:631-633 records the probs trap). sglang applies
repetition penalty as a dense [B, vocab] multiply before temperature,
omni applies it sparsely to seen tokens, cheaper at codec vocab sizes.
higgs_tts/sampler.py:16-17 imports sgl_kernel at module scope, a shared
helper must guard the import (NPU uses sgl-kernel-npu, XPU a different op
set, both without the renorm entry points as far as this tree shows).

Conclusion: no broad sampler refactor. Two targeted candidates (zonos2,
qwen3_tts eager sub talker), each behind an equivalence sweep like #816's.

## 4. #642

`[codex] Support current SGLang chunk counter` (open, conflicting, last
touched 2026-07-04) adds a shim reading `inflight_middle_chunks` with a
fallback to `is_chunked`. sglang 0.5.18's Req has only
`inflight_middle_chunks` (schedule_batch.py:1021, 1691) and main reads that
name at every site, with no `is_chunked` reader left. Superseded by the
0.5.18 upgrade, to be closed.

## Not verified

flashinfer 0.6.17's kernel source is not in the tree (contiguity
assertion, host reads, top_k above vocab handling). Launch counts are
static op counts, not profiler counts.
