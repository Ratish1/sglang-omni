# 13. The talker step, kernel by kernel, and what sgl_kernel can still take (2026-09-02)

Source: the talker process trace of the E0 bundle
(e0-prep4-20260831/traces_tts/talker_ar_pid1500047_rank0.trace.json, voice
clone at c16, bf16 colocated H100), read with scripts/trace_kernels.py
summary. 592 decode graph launches, 1131622 kernels, 3246 ms of kernel
time: 1912 kernels and 5.48 ms per step, which is the 5.1 to 5.8 ms GPU
time of doc 06 section 3.1.

## 1. Per step, by family

| family | kernels per step | ms per step | share |
|---|---|---|---|
| gemm (cuBLAS nvjet small M, split K and its reduce, cutlass grouped) | about 440 | 2.67 | 49% |
| copy (direct_copy 169, memcpy32 42, fill 21, index_select 16, DtoD 14) | about 260 | 0.86 | 16% |
| attention (cudnn sdpa 75, FA3 19 plus combine 19, varlen 5) | about 120 | 0.79 | 14% |
| moe (fused_moe, routing, sort, expand, finalize, for the 20 talker layers) | about 100 | 0.39 | 7% |
| elementwise adds (bf16 add 199, sigmoid 20, binary 21) | about 240 | 0.24 | 4% |
| reduce (argmax 15, gemv reduce 15, dot 15) | about 45 | 0.16 | 3% |
| norm (rmsnorm 177, fused_add_rmsnorm 40) | about 217 | 0.15 | 3% |
| rope (fused_rope 80, fused_rope_store 20) | 100 | 0.15 | 3% |
| sampling (the talker's in graph sampler) | 1 | 0.08 | 1% |

Average kernel: 2.9 us. A talker with 1024 hidden and at most 16 rows is
launch and latency bound, not compute bound.

## 2. Where the 1912 kernels come from

The code predictor runs 16 sub steps per talker step (the qk norm kernel
count is 100 per step: 20 talker layers plus 5 predictor layers times 16),
each through its 5 dense layers, so 80 predictor layer sub steps against
20 talker layers. Read from components/talker.py:1640-1720, one predictor
layer sub step is:

| op | kernels | source |
|---|---|---|
| qkv GEMM (M rows, K 1024, N 4096) | 1 to 2 (split K adds a reduce) | attn.qkv_proj |
| qk norm | 1 | apply_qk_norm, sglang fused_qknorm_warp |
| rope | 1 | attn.rotary_emb with fused_set_kv_buffer_arg None |
| K and V into the private cache | 2 | layer_k_cache[...].copy_(k), same for v (:1704-1705) |
| attention | 1 | torch scaled_dot_product_attention, cudnn flash |
| o GEMM (K 2048, N 1024) | 1 to 2 | attn.o_proj |
| residual add, rmsnorm | 2 | residual + attn_out then input_layernorm without the residual argument |
| gate up GEMM (K 1024, N 6144), act_and_mul, down GEMM (K 3072, N 1024) | 3 to 4 | layer.mlp |
| residual add, rmsnorm | 2 | residual + mlp_out then post_attention_layernorm |
| reshapes that materialize | 1 to 2 | transpose(1, 2) into sdpa, reshape after |

About 16 kernels times 80 is about 1300 of the 1912. The talker's own 20
layers are already on the fused path: fused_add_rmsnorm (40 per step),
fused_rope_store into the paged pool (20), FA3, the cutlass MoE. The
predictor is where the plain torch ops sit.

## 3. What sgl_kernel and sglang's layers can take, in model code

All in components/talker.py, no upstream change.

| change | kernels per step removed | ms per step | how |
|---|---|---|---|
| fused residual add and rmsnorm in the predictor loop | 160 | about 0.2 | pass residual into RMSNorm.forward, the fused_add_rmsnorm path the talker layers already use (sglang layers/layernorm.py:194-224) |
| fused qk norm and rope | 80 | about 0.1 | current_platform.get_fused_qk_norm_rope (platforms/interface.py:34), one kernel for the norm and the rope |
| the two cache copies | 160 | 0.34 | only by moving the predictor's attention onto a pool layout with sglang's fused rope store and a decode attention kernel, the way the talker layers do. SDPA wants the private [rows, heads, len, dim] layout, so this is a layout change, not a kernel swap |
| split K GEMM reduces | 80 | 0.13 | cuBLAS picks split K for the K 3072 down projection at small M. No sgl_kernel counterpart for bf16 small M GEMM. Would need a GEMV style kernel, omni code |

The first two are a day of model code and take the step from 1912 to
about 1670 kernels (minus 13 percent) and 5.5 to about 5.2 ms of GPU,
with the graph launch cost shrinking in proportion. Proof: this census
before and after (kernels per step by family), codes on the numerics
harness (the fused add rounds the residual once in fp32, so expect a
last bit difference on some steps, then WER and UTMOS), voice clone at
c16 and c32.

The GEMMs are the ceiling. 440 GEMMs at 5.4 us read 2 to 6 MB of weights
each, which is 0.6 to 1.8 us at HBM bandwidth, so cuBLAS's small M kernels
sit three to five times above the roofline and cost 2.7 ms of the 5.5.
Nothing in sgl_kernel serves bf16 GEMM at M of 16. Closing that gap means
a fused per layer sub step kernel (norm, qkv, rope, attention over a
short cache, o, norm, mlp in one launch), which is a kernel project with
its own measurement, not reuse.

## 4. Against the host side

Doc 06: 3.3 to 4.4 ms of every 8.6 ms cycle is host time the GPU waits
on (the event wait, the two stream syncs, the pageable copies, the
rebuilds). Removing those (E2, doc 05 section 7.2) is worth more than
everything in section 3 together and is a prerequisite for any overlap,
and it is also model code in talker.py. Order: E2, then the two fused
predictor items, then the layout change if the copies still matter, with
the GEMM ceiling measured before anyone writes a kernel.
