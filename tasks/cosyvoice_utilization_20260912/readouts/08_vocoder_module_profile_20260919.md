# Readout 08: the vocoder per module and per kernel (2026-09-19)

Moss box, RTX 4090 D, card 6 alone, one process, no AR beside it. Branch
`slice/cosyvoice-4-2-hop-step-graph` 66e6ce77f (hop cache plus captured steps) built by its own
factory, under Nsight (`--trace=cuda,nvtx,osrt --cuda-graph-trace=node`, one capture range).
Forward hooks name every module of the Flow and HiFT with an NVTX range, so a kernel's innermost
range is its module; code outside any module (the packed forward's arithmetic, RoPE, the FA3 call,
the solver, packing) is `(glue)`. Scripts: `stage2/s4_vocoder_profile.py`,
`stage2/nsys_module_ledger.py`. Raw: `artifacts/cosyvoice-4090-20260918/s4-r1`. The hooks add host
time, so wall times here are upper bounds; kernel counts and kernel times are exact. Runtime call
counts include one `cuKernelGetName` per launch, a CUPTI artifact: launches are about half.

Schedule of the S1 gate, three steps: step 0 (2 rows, 400 frames, all first hops), step 3 (8 rows,
window 3,850, new 1,600), step 5 (8 rows, window 6,850, new 1,600).

## 1. Per call

| call | step 0: wall / kernels / kernel ms | step 3 | step 5 |
|---|---|---|---|
| plain hop (main's path) | 372 ms / 16,910 / 92 | 469 / 17,050 / 473 | 981 / 17,044 / 983 |
| cached hop, eager | 372 / 16,914 / 94 | 404 / 17,054 / 246 | 394 / 17,048 / 265 |
| cached hop, replayed | 94 / 16,914 / 89 | 240 / 17,054 / 251 | 255 / 17,048 / 263 |
| final | 363 / 16,885 / 86 | 472 / 16,879 / 476 | 955 / 16,879 / 960 |
| HiFT, all rows of the step | 55 / 2,408 / 9 (2 rows) | 205 / 9,784 / 91 (8 rows) | 253 / 9,708 / 196 (8 rows) |

A Flow call is about 17,000 kernels whatever it computes. At step 0 the kernels total 92 ms and
the call takes four times that: launch bound. At step 5 the plain hop and the final are device
bound (983 and 960 ms of kernels). The cache takes the hop's kernel time from 983 to 265 ms and
the replay takes its wall to the kernel time.

## 2. Where the kernel time goes (plain hop, step 5, 983 ms)

| kernel kind | ms | share | kernels |
|---|---|---|---|
| matmul (cuBLASLt `Kernel2`, `ampere_bf16 gemm`, `Kernel`) | 423 | 43 % | 1,659 |
| pointwise (`vectorized_elementwise`, `elementwise`, `unrolled_elementwise`) | 376 | 38 % | 12,658 |
| FA3 attention (`device_kernel`) | 124 | 13 % | 220 |
| layer norm | 47 | 5 % | 450 |


By module the conv position embed is 59 ms (6 %) in 1,360 kernels; its kernels are counted by
kind in the rows above.

Pointwise kernels are three quarters of the launches and 38 % of the device time of a large
call. Two sources, by module:

- Every Linear issues three pointwise kernels beside its matmul (`attn.to_q`: 220 matmuls, 660
  `vectorized_elementwise`, 17.6 of its 64 ms). Under autocast the weight, the bias and the
  float32 input are cast to bfloat16 on every call. About 160 Linear calls per Euler step.
- `(glue)` holds 7,637 kernels and 297 ms: FA3 124 ms, and 168 ms in 6,663 pointwise kernels, which
  is the adaptive layer norm arithmetic, the gates and residual adds, RoPE (`apply_rotary_pos_emb`)
  and the CFG and Euler updates, all on float32 tensors of the whole packed sequence.

The final has the same profile (960 ms: matmul 424, pointwise 381, FA3 96).

## 3. What is left in a replayed cached hop (step 5: 255 ms wall, 263 ms of kernels)

Runtime calls 5,945 (about 3,000 launches): the conv position embed break 3,260 (55 %), glue
2,304 (250 `cudaGraphLaunch`, 789 kernel launches, 379 `cudaMemcpyAsync`, 48
`cudaStreamSynchronize`), the conditioning (`pre_lookahead_layer`, embeddings) about 370. Kernel
time: matmul 130 ms, FA3 49 ms, pointwise 73 ms (the captured casts and arithmetic replay too),
the conv 35 ms.

- The conv position embed is two grouped convs with Mish; it issues 69 kernels per conv per step
  (cuDNN layout conversions around each) and is more than half of the host work left.
- 48 stream synchronizations per Flow call, in every call kind: the conditioning's host to device
  copies.

## 4. HiFT (step 5, 8 rows: 253 ms wall, 196 ms of kernels, 9,708 kernels)

| part | kernel ms | kernels |
|---|---|---|
| `resblocks` convs | 71 | 2,632 |
| `f0_predictor.condnet` (float64: `implicit_convolve_dgemm`) | 37 | 192 |
| `source_resblocks` convs | 32 | 916 |
| Snake activations (`activations1/2`) | 30 | 4,608 |
| upsampling `ups` | 13 | 160 |

About 1,200 kernels per row per call, the rows one after another, each over the row's whole mel
history (150 to 750 frames here). The f0 predictor's float64 convs are 19 % of the device time in
2 % of the kernels; the Snake activations are 47 % of the launches for 15 % of the time.

## 5. What this points at, by level

High: nothing new; the cache and the captured step do what section 1 shows.

Mid (PyTorch code): the casts and the unfused float32 arithmetic are 38 % of a large call's device
time and 75 % of every call's launches, on main and on every branch. The DiT weight precast
(plan 14, branch `slice/cosyvoice-3-2-dit-weight-precast`) removes two of the three casts per
Linear; the input cast goes only if the activations stay bfloat16 between modules, which is a
numerics question (the norms and the residual stream are float32 today). The conv position embed
break is the next host cost of a replayed hop. HiFT: rows batched per step, the Snake
activation fused, the f0 predictor's precision are three separate questions.

Low (kernels): FA3 is 13 % and the matmuls are already bfloat16 tensor core kernels; nothing to
win there before the pointwise share is gone.

Each of these is a validation task, not a plan: exactness against the float32 truth first (the
forensics harness of readout 07 section 3), then the probe, then serving.
