# Qwen3-Omni numerics and RTF after the sglang 0.5.18 bump

Date: 2026-08-28. Written from the CI artifacts of 16 Omni CI runs, the two
sglang checkouts, and the omni tree at `upstream/main`. No GPU was used.
Every claim below names its source. Items that need a GPU are written as
measurements in section 4.

## 0. Trees, images, runs

| Item | Value |
| --- | --- |
| omni analysis worktree | `.worktrees/qwen3-omni-0518-numerics`, branch `analysis/qwen3-omni-0518-numerics` at `d5eac2627` (upstream/main) |
| bump commit | `470965eb2` (#1719, merged 2026-08-27 12:07 UTC), parent `8f8b73d3c` |
| omni commits after the bump on the Qwen3-Omni path | none (`d5eac2627` ARK-ASR, `afb5910ef` XPU, `40bf88604` ROCm) |
| omni commits on the Qwen3-Omni path before the bump | #1494 (2026-08-25 08:09), #1574 (2026-08-24), both inside the 08-25 and 08-26 pre-bump CI runs |
| sglang v0.5.16 | `fdebc938f7`, `/Users/ratish/sglang-worktrees/v0.5.16` |
| sglang v0.5.18 | `71de97b264`, `/Users/ratish/sglang` |
| old CI image | `hongccc/sglang-omni@sha256:374d0b1c...` (sglang 0.5.16, torch 2.11.0, sgl-kernel 0.4.5, flashinfer 0.6.14, Triton 3.6.0, sgl-deep-gemm 0.1.4.post1, CUDA base 13.0.1, Python 3.12 per the log paths) |
| new CI image | `hongccc/sglang-omni@sha256:02a85f00...` (sglang 0.5.18, torch 2.13.0, sgl-kernel 0.4.6.post1, flashinfer 0.6.17, Triton 3.7.1, sgl-deep-gemm 0.1.5.post3, CUDA base 13.0.3, Python 3.12) |
| pre-bump runs (main, 0.5.16) | 32443156739 (08-21), 32547843426 (08-22), 32599828664 (08-22), 32867559656 (08-25), 32981649732 (08-26) |
| post-bump runs (PR merge refs on main after 470965eb2) | 33071447456, 33082648186, 33085872333, 33088462837, 33093059035, 33126429888, 33129336761, 33142984717, 33144941370, 33147701232 (08-27 12:21 to 08-28 06:22) |
| run 33158948928 (PR #1785, the linked run) | still in progress when read, Qwen3-Omni stages not started, TTS stage 1 passed on attempt 3 |

Artifacts: `scripts/ci_artifacts.py download` fetches the stage 5, 8, 9, 10
result JSONs (one per attempt). `scripts/ci_artifacts.py compare` prints
every table in section 2.

## 1. What each CI stage runs

| Stage | Server fixture | Checkpoint | Thinker dtype | Input | Output | Gates |
| --- | --- | --- | --- | --- | --- | --- |
| 5 MMSU | `qwen3_omni_bf16_colocated_thinker_server`: `examples/configs/qwen3_omni_mmmu_h100.yaml`, `Qwen3OmniPipelineConfig` (thinker only), 2 DP workers behind the router | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | bf16 | text only, 2000 prompts of about 178 tokens, `max_tokens 32`, temperature 0 | 2 tokens | accuracy >= 0.707, qps >= 48.0, latency <= 0.3 s |
| 8 Video-MME talker | `qwen3_omni_bf16_disagg_server`: `examples/run_qwen3_omni_speech_server.py`, thinker on GPU 0, talker and code2wav on GPU 1 | Instruct | bf16 | 20 videos at 2 fps (14,210 prompt tokens each) with their audio track, temperature 0 | text plus speech | thinker accuracy >= 0.6, `rtf_mean` <= 1.12, latency <= 12.2 s, qps >= 0.86, WER <= 2.875 percent |
| 9 Video-AMME | `qwen3_omni_fp8_colocated_server`: `examples/configs/qwen3_omni_colocated_h100_fp8.yaml`, `--colocate`, 2 DP workers | `marksverdhei/Qwen3-Omni-30B-A3B-FP8` | FP8 block [128,128], dynamic activations (`quantization_config` in the checkpoint) | 50 videos with audio, 14,345 prompt tokens mean, temperature 0 | text | accuracy >= 0.64, qps >= 1.39, latency <= 9.8 s |
| 10 Video-AMME talker TP2 | `qwen3_omni_fp8_tp2_server`: thinker TP=2 on GPUs 0,1, talker on GPU 1 | FP8 | FP8 | 10 videos with audio | text plus speech | accuracy >= 0.5, WER <= 1.65 percent, `rtf_mean` <= 5.04 |

The FP8 checkpoint leaves `thinker.visual`, `thinker.audio_tower`,
`code2wav`, `lm_head`, `mlp.gate`, the norms and the embeddings in bf16.
Its talker layers are FP8 as well.

## 2. CI evidence

### 2.1 Gate values per attempt

Stage 5, accuracy (gate 0.707), throughput in req/s:

| Group | Attempts | Accuracy values | qps range |
| --- | --- | --- | --- |
| pre | 12 | 0.705, 0.7055, 0.706, 0.7065 (never 0.707) | 79.4 to 90.5 |
| post | 19 | 0.7035 to 0.709, final attempt at or above 0.707 in every run | 74.6 to 82.5 |

Stage 9, accuracy (gate 0.64), mean output tokens, throughput:

| Group | Attempts | At 0.64 | At 0.62 | Output tokens mean | qps range |
| --- | --- | --- | --- | --- | --- |
| pre | 7 | 3 | 4 | 53 to 56 | 1.58 to 1.75 |
| post | 30 | 3 | 27 | 47 to 49 | 1.53 to 1.70 |

Stage 8, `rtf_mean` (gate 1.12), thinker accuracy (gate 0.6), latency:

| Group | Attempts | rtf_mean values | Accuracy | latency_mean_s |
| --- | --- | --- | --- | --- |
| pre | 10 | 0.855, 0.914, 1.151, 1.175, 1.181, 1.182, 1.187, 1.201, 1.228, 1.253 | 0.65 in 10 | 9.7 to 10.6 |
| post | 23 | 1.086 to 1.275, 4 attempts at or below 1.12 | 0.65 in 17, 0.60 in 6 | 10.4 to 11.2 |

Stage 10, accuracy (gate 0.5), `rtf_mean` (gate 5.04), WER (gate 1.65 percent):

| Group | Attempts | Accuracy | rtf_mean | WER |
| --- | --- | --- | --- | --- |
| pre | 5 | 0.5 in 5 | 2.91 to 3.42 | 0.0 to 1.70 |
| post | 11 | 0.6 in 11 | 3.26 to 3.81 | 0.0 to 2.87 (one failed attempt, retry 1.24) |

### 2.2 Per-sample changes (temperature 0 everywhere)

Stage 9 (FP8 thinker), 7 pre attempts against 28 post attempts:

- Prediction majority differs on 2 of 50 samples: 007-1 D (correct in 6 of 7)
  to A (wrong in 28 of 28), 017-1 A to D (wrong on both stacks, 229 tokens to
  32 tokens). 011-3 was A in 4 of 7 pre attempts and B in 3, post it is A in
  28 of 28. 008-2 is D on both stacks, with B (correct) in 3 of 28 post
  attempts.
- Completion token majority differs on 38 of 50 samples (011-1 135 to 57,
  011-2 50 to 13, 018-2 83 to 46, 002-2 55 to 107, 006-1 42 to 59).
- Within a stack the predictions are deterministic: 2 unstable samples pre,
  1 post.

Stage 8 (bf16 thinker), 10 pre attempts against 23 post attempts:

- 003-1: C with 25 tokens in 10 of 10 pre attempts. Post: C with 25 tokens in
  10 of 23, A with 61 tokens in 13 of 23. This is the 0.65 to 0.60 flip.
- 001-3: 24 tokens in 10 of 10 pre. Post: 24 tokens in 10, 18 tokens in 13.
- 005-3: C in 10 of 10 pre. Post: C in 17 of 23, D, A or B in 6.
- 001-1 (the RTF sample): 8 tokens in 8 of 10 pre attempts and in 23 of 23
  post attempts. The two pre attempts with 38 tokens are the two pre attempts
  with rtf_mean 0.855 and 0.914.
- Run-to-run variance of completion tokens rose: 003-3, 005-1, 005-2, 002-2,
  006-1, 006-2 take 3 or more distinct token counts across post attempts.

Stage 5 (bf16 thinker, text only), 12 pre attempts against 19 post:

- 12 of 2000 questions flip systematically (correct in at least 90 percent of
  one group's attempts and at most 10 percent of the other's): 4 toward
  wrong, 8 toward right. Mean accuracy 0.7058 pre, 0.7065 post.
- Questions whose correctness varies across attempts: 37 pre, 48 post.

Stage 10 (FP8 thinker TP=2): 003-2 D (wrong) to C (correct) in every post
attempt, which is the 0.5 to 0.6 move. Token majority differs on 7 of 10.

### 2.3 Stage 8 RTF mechanism

`rtf_mean` is the mean over 20 samples of latency divided by audio
duration. Sample 001-1 answers in 8 tokens, the talker turns those into
1.8 to 2.6 s of audio, and the 14,217-token video prefill plus talker
prefill takes 13.5 to 15.9 s, so its rtf is 5.7 to 8.4 and contributes 0.28
to 0.42 of the mean. The mean without the two largest samples is 0.77 to
0.88 pre and 0.76 to 0.89 post. The gate value is decided by the thinker's
answer length on one sample and the talker's sampled audio length, on both
stacks. The two pre attempts that passed with margin (0.855, 0.914) are the
two where 001-1 answered with 38 tokens.

### 2.4 Speed

| Stage | Metric | pre | post |
| --- | --- | --- | --- |
| 5 | qps at c16, 2 workers | 79.4 to 90.5 | 74.6 to 82.5 |
| 8 | latency_mean_s (14k-token prefill dominated) | 9.7 to 10.6 | 10.4 to 11.2 |
| 9 | qps (output tokens 54 pre, 48 post) | 1.58 to 1.75 | 1.53 to 1.70 |
| 10 | rtf_mean | 2.9 to 3.4 | 3.3 to 3.8 |

Stage 9 produces 11 percent fewer tokens post and is 3 percent slower in
qps, so its per-token cost rose by about 10 percent. Doc
`tasks/sglang_0518_upgrade/19_h200_regressions_root_cause.md` section 11
attributes a 3 percent c1 cost to prefill and single-request decode on the
new stack. The stage 8 and stage 10 latency moves are the same size.

### 2.5 What the evidence establishes

1. The thinker's greedy decode path changed for both bf16 and FP8. The
   changes are deterministic within a stack and systematic across stacks
   (007-1, 017-1, 003-1, the 12 MMSU questions, 003-2), so they come from
   kernel arithmetic, not from a race or a scheduler difference.
2. The bf16 thinker is more sensitive to batch composition on the new stack
   (stage 8 run-to-run variance, stage 5 unstable count).
3. Net accuracy is flat to slightly better (stage 5 +0.0007, stage 10 +0.1,
   stage 9 net -1 sample on the majority, MMMU +1 in doc 19). The gate
   failures are one-sample margins against gates calibrated on 2026-08-01
   (#1260) that main already failed before the bump (stage 5 in 12 of 12
   attempts, stage 8 in 8 of 10, stage 9 in 4 of 7).
4. Stage 8 RTF is a metric-definition problem plus a 4 percent latency move,
   not a talker or code2wav slowdown.

## 3. The thinker's kernel path and what changed

Dimensions (HF config): hidden 2048, 48 layers, 32 query heads, 4 KV heads,
head_dim 128, 128 experts, top-8 with `norm_topk_prob`, expert intermediate
768, vocab 152064. Audio tower 32 layers, d_model 1280, 20 heads (head_dim
64). Vision tower 27 layers, hidden 1152, 16 heads (head_dim 72). Talker 20
layers, hidden 1024, 128 experts top-6, intermediate 384.

Effective thinker ServerArgs (both tags, `stages.py`
`create_sglang_thinker_executor_from_config` and
`server_args_builder.py`): `attention_backend` unset (resolves to fa3 on
SM90), `moe_runner_backend auto`, `fp8_gemm_runner_backend auto`,
`kv_cache_dtype auto`, `sampling_backend pytorch`, `random_seed 123`,
`chunked_prefill_size 8192`, `enable_mixed_chunk`, `max_running_requests 64`,
decode CUDA graphs bs 1 to 32 (log), prefill graphs off, `page_size` default,
`enable_fused_qk_norm_rope False` (default at both tags). Omni's platform
policy (`sglang_omni/platforms/cuda.py:130-143`) overrides
`moe_runner_backend` to `cutlass` for a native block-FP8 Qwen3-Omni
checkpoint, and the talker's `fp8_gemm_runner_backend` to `triton`. The
stage 8 logs of both stacks confirm `moe_runner_backend=auto` for the bf16
thinker and `flashinfer_cutlass` for the bf16 talker.

### 3.1 Kernels on the bf16 thinker path (stages 5, 8, and the bf16 parts of 9 and 10)

| Component | Kernel at v0.5.16 | Kernel at v0.5.18 | Source between tags | What still differs | Switch on the H100 |
| --- | --- | --- | --- | --- | --- |
| gate GEMM (router logits) | `torch.matmul` bf16 (cuBLAS) | same | same | cuBLAS in torch 2.11 vs 2.13 on CUDA 13.0.1 vs 13.0.3 | none, `kernel_ab.py gate_gemm` |
| router top-k | `moe_fused_gate` Triton `_router_triton_kernel` (JIT default on at 0.5.16, `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=True`) | same function, no flag | identical except `num_warps` for N > 512 (N is 128) | Triton 3.6.0 vs 3.7.1 codegen | none, `kernel_ab.py router` |
| MoE experts | Triton `fused_moe_kernel` with tile config `triton_3_2_0/E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json` for up, reused for down (both stage 8 logs print the same fallback lines) | same kernel, `FUSE_SWIGLU` epilogue added but off (`SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED=False`) | same launched code | Triton recompile | `--stages.thinker.engine.moe_runner_backend flashinfer_cutlass` on both stacks (the bf16 talker path) |
| SiLU inside MoE | JIT `act_and_mul_kernel` (`silu_and_mul`) | same, two template flags added with defaults off | same math | nvcc rebuild | none, `kernel_ab.py silu_and_mul` |
| top-k weighted sum | `sgl_kernel.moe_sum_reduce` | same | same | rebuild | none |
| attention | FA3 from `sgl_kernel.flash_attn` (sgl-attn `f89bc2306632`) | same pin | identical | sgl-kernel 0.4.5 to 0.4.6.post1 rebuilt with the newer nvcc, `num_splits 0` heuristic on both | `--stages.thinker.engine.attention_backend triton` on both stacks |
| input, post-attention, final RMSNorm | `sgl_kernel.fused_add_rmsnorm` and `rmsnorm` (flashinfer `norm.cu` at `bc29697b`) | same pin, python dispatch unchanged for the bf16 path | identical | rebuild | none, `kernel_ab.py rmsnorm` |
| q_norm, k_norm | `sgl_kernel.rmsnorm` through `apply_qk_norm` (fused qk-norm-rope off, MRoPE incompatible) | same | identical | rebuild | none |
| MRoPE | `triton_mrope_fused` (`kernels/ops/attention/rotary_triton.py`) | same file, no diff | identical | Triton recompile | none, `kernel_ab.py mrope` |
| lm_head | `torch.matmul` bf16 (cuBLAS), bf16 logits, argmax (`enable_fp32_lm_head` unset by omni) | same | same | cuBLAS | `--stages.thinker.engine.enable_fp32_lm_head true` exists on both stacks |
| sampler | argmax at temperature 0 | same | same | none | none |
| MRoPE positions | omni `request_builders._compute_mrope_positions` | same | same | none | none |

### 3.2 Additional kernels on the FP8 thinker path (stages 9, 10)

| Component | Kernel at v0.5.16 | Kernel at v0.5.18 | Source between tags | What still differs | Switch |
| --- | --- | --- | --- | --- | --- |
| q, k, v, o projections | `Fp8LinearMethod` block [128,128] dynamic, `dispatch_w8a8_block_fp8_linear` auto to `deepgemm_w8a8_block_fp8_linear_with_fallback` (SM90, `SGLANG_ENABLE_JIT_DEEPGEMM=True`) | same dispatch | identical dispatch | sgl-deep-gemm 0.1.4.post1 to 0.1.5.post3, JIT compiled with the newer nvcc | `SGLANG_ENABLE_JIT_DEEPGEMM=0` routes to `triton_w8a8_block_fp8_linear` on both stacks |
| activation quant before DeepGEMM and inside the MoE | `per_token_group_quant_8bit_v2.cuh` (JIT v2, default since `SGLANG_OPT_USE_JIT_PER_TOKEN_GROUP_QUANT=False`) | `per_token_group_quant.cuh` (new JIT kernel) | rewritten. Same arithmetic by reading: amax floored at 1e-10, `scale_inv = amax / 448`, `q = clamp(x * (448 / amax))`, fp32 scales | kernel binary | none, `kernel_ab.py fp8_group_quant` |
| MoE experts | omni policy sets `moe_runner_backend=cutlass`: `cutlass_fused_experts_fp8` (group quant, `shuffle_rows`, `sgl_kernel.fp8_blockwise_scaled_grouped_mm`, JIT `silu_and_mul`, group quant, grouped GEMM, `apply_shuffle_mul_sum`) | same wrapper (2-line import diff), `fp8_blockwise_moe_kernel.cu` +3 lines for sm107 only, CUTLASS pin `57e3cfb47a2d` at both tags | identical launched code | sgl-kernel rebuild, quant kernel above | `--stages.thinker.engine.moe_runner_backend triton` on both stacks (Triton block-FP8 `fused_moe_kernel`, default tile config since no H100 `E=128,N=768,fp8_w8a8` file exists at either tag) |
| weight scales | omni `convert_fp8_weight_scale_inv` reciprocal (omni-owned) | same | same | none | none |
| gate, lm_head, norms, embeddings | bf16 as in 3.1 | | | | |

### 3.3 Encoders (stages 8, 9, 10 inputs)

| Component | v0.5.16 stack | v0.5.18 stack | Switch |
| --- | --- | --- | --- |
| audio tower, captured layer stack (`audio_layer_graph.py`) | transformers `ALL_ATTENTION_FUNCTIONS["flash_attention_2"]` inside the CUDA graph (log: "audio layer CUDA graphs captured for buckets [128 ... 4096]") | `VisionFlash3Attention` (sgl-kernel FA3 varlen) inside the graph (log: "... with fa3 attention"), rewritten by #1719 | `--stages.audio_encoder.factory-args.enable-layer-cuda-graph false` on both stacks (eager transformers path, `config._attn_implementation`) |
| vision tower | transformers 5.12.1 `Qwen3OmniMoeVisionEncoder` (omni compat subclass), attention through transformers' default SDPA | same transformers, torch SDPA backend selection under torch 2.13 and its cuDNN | none in omni, `kernel_ab.py vision_sdpa` per backend |
| video frames and mel features | HF processor with the torchvision video backend on torchcodec (`preprocessor.py:621`), torchcodec 0.11.1, torchvision 0.26 | torchcodec 0.15.0, torchvision 0.28 | none, `kernel_ab.py hf_processor` hashes `pixel_values_videos`, `input_features`, `input_ids` for one CI clip |

### 3.4 Talker and code2wav (speech output only, stages 8 and 10)

bf16 talker: `moe_runner_backend flashinfer_cutlass` (flashinfer 0.6.14 to
0.6.17, autotuner reports no tuned config for `trtllm::fused_moe` gemm1 and
gemm2 at the 14,208-token prefill on both stacks). FP8 talker: cutlass MoE
plus Triton dense FP8 by omni policy. Code2wav: 4 exact CUDA graphs on both
stacks. These decide WER and audio duration (hence `rtf`), not the thinker
text.

### 3.5 Ruled out by reading

| Candidate | Why it is out |
| --- | --- |
| FA3 kernel source | same sgl-attn pin at both tags |
| CUTLASS, flashinfer norm sources in sgl-kernel | same pins |
| MoE Triton tile config | both stage 8 logs load `triton_3_2_0/E=128,N=768` for up and reuse it for down |
| `SGLANG_ENABLE_MOE_DEFERRED_FINALIZE` (False to True) | only for `flashinfer_trtllm` with NvFp4 (`fused_moe_triton/layer.py:386-390`) |
| top-k renormalization epsilon (`4d5917e744`) | lands in `fused_topk_torch_native`, `grouped_topk_gpu` and the biased paths, not in the softmax `fused_topk` path the thinker takes |
| SwiGLU epilogue fusion, `SGLANG_OPT_MOE_QUANT_ONCE`, `SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE` | off by default, or channelwise-only |
| RMSNorm `quant_linear` fusion | static per-tensor FP8 only, omni passes no `quant_linear` |
| JIT `rope.cuh` refactor | not on the thinker path (MRoPE uses the Triton kernel), arithmetic unchanged anyway |
| `moe_align_small_numel` (bs 1 decode) | changes token ordering only |
| omni #1719 changes on the thinker (`bootstrap.py`, `talker.py` kwargs, `thinker_model.py` tp accessors) | no arithmetic |
| omni #1494 and #1574 | inside the 08-25 and 08-26 pre-bump runs, whose values match the 08-21 and 08-22 runs |
| transformers pin | 5.12.1 at both |
| scheduler and batching | same decode graph ladder, chunk size and mixed chunk on both logs |

Everything that remains is a binary or compiler difference under an
unchanged source (cuBLAS, Triton, nvcc rebuilds, DeepGEMM), the rewritten
FP8 group quant kernel, the audio tower attention swap, and the input-side
video and audio decoders. None of these can be classified further by
reading. Section 4 measures them.

## 4. H100 attribution protocol

One variable per run. Both stacks on the same host and GPU, same HF cache,
same clips. The commands are in `scripts/h100_runs.sh`.

### 4.0 Containers and checkouts

| Arm | Image | omni checkout |
| --- | --- | --- |
| old | `hongccc/sglang-omni@sha256:374d0b1c...` | `8f8b73d3c` (main before the bump) |
| new | `hongccc/sglang-omni@sha256:02a85f00...` | `d5eac2627` (upstream/main) |

Start the containers with `SGLANG_OMNI_AUTO_CLONE=0` and mount the two
checkouts (the image entrypoint clones main otherwise, doc 19 section 7).
Record the versions of torch, triton, sglang, sgl_kernel, flashinfer and
deep_gemm (`scripts/kernel_ab.py run` prints them) and `nvidia-smi` in each. Servers run one worker on one GPU
(the router and DP=2 change batch composition only).

### 4.1 Reproduce with single workers

Per stack: FP8 colocated server (stage 9 config) and Video-AMME 50 at c16
three times and at c1 once (`run_bench.py videoamme`). bf16 thinker-only server (stage 5 config) and
MMSU text at c16 once and at c1 once. Readout with
`scripts/ci_artifacts.py compare-local`. The c1 runs give one deterministic
answer per sample per stack and are the reference for every arm below. The
c16 runs must show the same flips as CI (007-1, 017-1, 003-1, the MMSU
twelve) for the reproduction to count.

### 4.2 Kernel-level A/B

`scripts/kernel_ab.py run --seed 0 --out /shared/ab/<arm>.pt` in each
container, then `scripts/kernel_ab.py compare old.pt new.pt`. Inputs are
generated on the CPU from a seeded generator and hashed, and compare refuses
to run when the input hashes differ. Each case records bitwise equality,
mismatch count, max absolute and max relative difference:

`gate_gemm`, `lm_head` (with argmax agreement), `qkv_gemm_bf16`, `rmsnorm`,
`fused_add_rmsnorm`, `qk_norm`, `mrope`, `router`, `silu_and_mul`,
`moe_sum_reduce`, `moe_bf16_triton`, `fa3_prefill`, `fa3_decode`,
`fp8_group_quant`, `fp8_dense_deepgemm`, `fp8_dense_triton`,
`moe_fp8_cutlass`, `moe_fp8_triton`, `vision_sdpa` (default, and each SDPA
backend forced), `audio_attention` (SDPA against FA3 varlen at head_dim 64),
and `hf_processor` on one CI clip (hashes of `pixel_values_videos`,
`input_features`, `input_ids`).

Readout: cases that are bitwise identical across stacks are excluded. The
remaining cases are ranked by max relative difference at M = 4096 and at
M = 1. A case that differs across stacks but is bitwise identical across
two runs on the same stack is a deterministic kernel change. A case that
differs between two runs on the same stack is nondeterministic and is
reported as such.

### 4.3 Server-level ablations

Each arm on both stacks, Video-AMME at c1 (deterministic) and at c16 three
times, MMSU text at c1 for the bf16 arms. Per-sample readout against the
4.1 reference of the same stack and of the other stack.

| Arm | Flag or env | Removes | Stages |
| --- | --- | --- | --- |
| dense FP8 GEMM | `SGLANG_ENABLE_JIT_DEEPGEMM=0` | DeepGEMM (Triton block GEMM instead, same family on both stacks) | 9, 10 |
| FP8 MoE runner | `--stages.thinker.engine.moe_runner_backend triton` | cutlass grouped GEMM path (Triton block-FP8 `fused_moe_kernel` instead) | 9, 10 |
| attention | `--stages.thinker.engine.attention_backend triton` | FA3 | 5, 8, 9 |
| audio graph | `--stages.audio_encoder.factory-args.enable-layer-cuda-graph false` | FA2 (old) or FA3 (new) inside the captured audio stack, eager transformers attention instead | 8, 9, 10 |
| bf16 MoE runner | `--stages.thinker.engine.moe_runner_backend flashinfer_cutlass` | Triton `fused_moe_kernel` (bf16 only, the policy rejects it for FP8) | 5, 8 |
| fp32 logits | `--stages.thinker.engine.enable_fp32_lm_head true` | bf16 logits ties at argmax | 5, 8, 9 |

Decision rules:

- An arm on the new stack that reproduces the old stack's c1 predictions and
  token counts on the flipped samples names the kernel family. The same arm
  on the old stack then shows whether that family is where the old stack's
  answers came from (its outputs move too) or not.
- An arm that moves the flipped samples on both stacks without matching
  either reference shows a near-tie sample, not a defect.
- If no arm reproduces, the residual is in the components without a switch
  (cuBLAS GEMMs for gate and lm_head, rebuilt norm binaries, the Triton
  recompiles) and 4.2 ranks them.

### 4.4 Logprob margins (stage 5)

The omni chat endpoint the benchmarks use has no logprob field. `/generate`
returns `[logprob, token_id]` per generated token when `return_logprob` is
set (`serve/protocol.py:248`, `serve/openai_api.py:1178`,
`model_runner/base.py:888`) but carries no video or audio inputs, so the
measurement covers the text-only MMSU questions. `scripts/logprob_probe.py`
sends the mmsu-ci-2000 questions (all 2000, or the 12 flipped ids) with the
benchmark's prompt at temperature 0 and records the answer token's
probability p and `log(p / (1 - p))`, a lower bound of the margin over the
runner-up. Run it on both stacks (`run_logprob_probe` in `h100_runs.sh`).
Margins under 0.1 nat on both stacks classify a question as a near-tie. A
margin that is large on one stack and small or reversed on the other is a
systematic shift and the 4.3 arm that removes it is the cause. The whole-set
run also gives each stack's count of near-tie questions, which bounds the
attempt-to-attempt variance seen in 2.2.

The video samples (007-1, 017-1, 003-1) have no logprob path today. Adding
`return_logprob` to the chat request (`serve/protocol.py:71-80` already
carries `videos`, `audios`, `video_fps`) and forwarding it in
`extra_params` is the one omni change that would extend the probe to
stages 8 and 9. It is not part of this plan.

### 4.5 Speed and RTF

- Stage 8: send the 001-1 prompt alone ten times per stack, record thinker
  latency, talker audio duration and rtf. This separates the answer-length
  effect (numerics) from pipeline speed. Record the 14k-token prefill time
  of the thinker and the talker from the server logs ("Prefill batch ...
  input throughput") on both stacks.
- Stage 5 and 9 throughput: the per-stage traces of
  `tasks/sglang_0518_upgrade/20_profiling_protocol.md` apply unchanged.
  The stage 9 per-token cost (+10 percent) is measured with the FP8 arms of
  4.3, since DeepGEMM and the cutlass MoE are the only FP8-specific kernels.

### 4.6 Order

1. 4.0 and 4.1 (reproduction, about 1 hour of GPU time).
2. 4.2 (kernel A/B, minutes per container).
3. 4.3 FP8 arms (dense GEMM, MoE runner), then attention, then audio graph.
4. 4.4 on the samples still unexplained.
5. 4.5.

## 5. Follow-up work after attribution

- Gates: the stage 5, 8, 9 gates have no margin over the new stack's
  distribution (stage 9 passes in 3 of 30 attempts). Recalibrate from the
  post-bump per-sample data once 4.3 has named the kernel families, and
  replace stage 8 `rtf_mean` with the ratio of total latency to total audio
  (0.74 to 0.81 on both stacks) or exclude answers shorter than a fixed
  token count from the per-sample rtf.
- MoE Triton configs: no tuned `triton_3_7_1` file for `E=128,N=768` on
  H100 exists, bf16 or fp8_w8a8. Tune with sglang's `tuning_fused_moe_triton`
  benchmark on the new stack for TP=1 and TP=2 and check accuracy and speed
  through 4.3.
- FlashInfer autotune cache for the talker's `trtllm::fused_moe` at
  (14208, 1024) is missing on both stacks.
- Stage 10 WER gate (1.65 percent on ten clips) fails on sampling noise on
  both stacks (doc 19 section 13).

## 6. Files

- `00_plan.md`: this document.
- `scripts/ci_artifacts.py`: artifact download and comparison, also reads
  local benchmark result files (`compare-local`).
- `scripts/runs_postmerge_20260828.tsv`: the Omni CI runs created after the
  bump, with head SHA, branch and time.
- `scripts/run_bench.py`: runs one stage's benchmark with the CI settings
  against a running server, without the inline WER pass.
- `scripts/logprob_probe.py`: MMSU answer-token logprobs through `/generate`.
- `scripts/kernel_ab.py`: kernel-level A/B dump and compare.
- `scripts/h100_runs.sh`: server launch, benchmark, probe and log-grep
  functions for every arm of section 4.

## 7. Verification status of the scripts

Checked on this machine (no GPU):

- `ci_artifacts.py compare` reproduced every table in section 2 from the
  downloaded artifacts. `compare-local` was run on artifact files renamed as
  local results.
- `kernel_ab.py`: every kernel signature it calls was read at both tags
  (`sgl_kernel.rmsnorm`, `fused_add_rmsnorm`, `silu_and_mul`,
  `flash_attn_varlen_func`, `flash_attn_with_kvcache`, `moe_sum_reduce`,
  `moe_fused_gate`, `sglang_per_token_group_quant_fp8`,
  `deepgemm_w8a8_block_fp8_linear_with_fallback`,
  `triton_w8a8_block_fp8_linear`, `invoke_fused_moe_kernel`,
  `moe_align_block_size`, `cutlass_fused_experts_fp8`, `MRotaryEmbedding`)
  and the runtime-context recipe is the one
  `tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py:82` uses. The file
  compiles. It has not executed, since it needs CUDA and sgl_kernel. Every
  case is isolated and a failing case records its traceback.
- `run_bench.py` mirrors the three CI test files' configurations
  (`VideoEvalConfig` fields, the stage 8 short-answer prompt, the MMSU
  `argparse.Namespace`). It compiles and has not executed.
- `logprob_probe.py` follows the `/generate` request and response shapes read
  in `serve/openai_api.py:1002-1183`. It compiles and has not executed.
- `h100_runs.sh` passes `bash -n`. Flags were read from
  `sglang_omni_router/launcher/local.py:build_worker_command`, the CI
  conftest, `examples/launchers/qwen3_omni.py` and the benchmark parsers.

The first execution on the H100 is the remaining verification. Read the
server log with `backend_lines` after each launch to confirm the arm took
effect (backend policy line, MoE config lines, audio graph line).
