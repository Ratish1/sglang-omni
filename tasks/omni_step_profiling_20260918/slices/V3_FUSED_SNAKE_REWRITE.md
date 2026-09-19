# Slice V3: the fused snake activation, rebuilt from first principles

Replaces the compile-based V2 (READOUT_11_14: about 1 percent of c16 throughput for 3.6
minutes of startup). Inputs: our reading of `vocoder_kernels.py` and its call sites, V-e7
(run 14), and five research reports under
`artifacts/qwen3_tts_step_profiling_20260918/kernel_parts/` (A launch design, B numerics,
C integration, D generality, E vehicles). A line below is a fact only if it carries a
file:line that was re-read, or a measurement; everything else is a validation task.

## 1. What the activation is and where it runs

`y = x + r_c * sin(a_c * x)^2`, per channel c. Qwen3-TTS derives `a = exp(alpha)`,
`r = 1 / (exp(beta) + 1e-9)`. 29 calls per decode (`incremental_codec.py:656`, `:356`,
`:360`, `:670`: one per decoder block, two per residual unit, one before the final conv),
on the largest tensors of the vocoder: [B, 1536, few] early, [B, 96, up to 122,880] late.
Eager runs it as separate elementwise kernels, each a full pass over the tensor.

```
_decode_tensors (incremental_codec.py:610)
  4 decoder blocks: block[0] SnakeBeta :656 -> transposed conv :657
                    3 residual units :349: act1 :356 -> conv1 (cat with history)
                                           act2 :360 -> conv2 -> + skip
  decoder[-2] SnakeBeta :670 -> final conv
fuse_vocoder_decoder (vocoder_kernels.py:237)      called at streaming_vocoder.py:647,
  after the weights load (stages.py:319), before the graph runners are built (:773);
  capture happens later in warmup_now() (stages.py:365)
```

## 2. Measured (V-e7, RTX 4090 D, plain CUDA graphs, no torch.compile)

Today's kernel against eager, all 44 captured keys: waveform bitwise identical at 44 of
44; replay -10 to -23.5 percent at 40 keys; w1 b1 +3 percent; 64-frame windows -5 to -12
percent; 261 kernels fewer per decode; 0.01 s to fuse. Left: 383 elementwise and 156 to
190 copy kernels.

## 3. Defects, each with its evidence

| # | defect | evidence |
| --- | --- | --- |
| D1 | one Triton program per (batch x channel) row: [1,1536,1] launches 1,536 programs, 196,608 threads for 1,536 elements (1.6 percent of the thread slots do work); [1,1536,8] launches the same | `vocoder_kernels.py:110` grid, `num_warps=4`; report A arithmetic; V-e7 w1 b1 +3 percent |
| D2 | every program recomputes 2 exp, an add and a reciprocal that depend only on the weights | `:77-82` |
| D3 | channel allowlist, batch cap and `_MAX_T = 65536` are not used by the kernel; the length cap sends tensors up to 122,880 samples to the eager chain | `:54-56`, `:162`; V-e7 64-frame rows keep 453 elementwise kernels against 383 |
| D4 | the wrapper returns None from ten exits and one blanket `except Exception`, with no log and no counter | `:138-171`; SGLang's rule: "The kernel raises on an unsupported input. It does not return None" (`kernels/ops/diffusion/README.md:81`) |
| D5 | pointers are specialized on 16-byte alignment (Triton default); a differently aligned input compiles a second binary at run time | `:62` protects only C and T; hazard documented at sglang `kernels/ops/memory/allocator.py:9-14` |
| D6 | with the flag on, the launcher sits inside main's `fullgraph=True` compile of the 8-frame chunk; Dynamo strips only `num_warps`, `num_stages`, `num_ctas`, `num_consumer_groups`, `num_buffers_warp_spec`, `num_cpu_threads` from a Triton launch, so `enable_fp_fusion` and `enable_reflect_ftz`, which the bit-identity claim rests on, are not handled as launch options; untested | torch 2.13 `_higher_order_ops/triton_kernel_wrap.py:2053-2060`; `stages.py:317-318` compile default on |
| D7 | modules are matched by the class name string; `models/auk/vae.py:175` has a class of the same name whose exp is conditional | `:200-204` |
| D8 | the docstring names a `div`; eager runs `reciprocal()` then `* 1.0` (`torch/_tensor.py:1112`); bits agree, the documented reference does not | report B, re-read |
| D9 | no check on the serving machine that fused equals eager; the proof is offline only. SGLang verifies at first sight and disables permanently on a mismatch | sglang `kernels/ops/diffusion/sites/bitexact_gate.py:63-88` |
| D10 | epsilon has two sources: the module copies `no_div_by_zero`, the kernel hardcodes 1e-9 | `:81`, `:183` |

## 4. Generality

Ten of eleven snake activations in omni and SGLang are this exact form with the same five
x-sized ops in the same order (mul, sin, square, mul, add); they differ only in how (a, r)
come from the weights (report D table; MiniCPM-o on current main,
`minicpm_o/components/token2wav/hift_layers.py:45-50`, is one more). SGLang v0.5.19 has no
fused snake (0 hits in its kernel directories); all its snakes are eager or scripted. So
the kernel takes precomputed per-channel `a` and `r`; the derivation stays with the model.
Other consumers run fp32 (MiniMax, LTX) or the checkpoint dtype (AUK), some inside an
anti-alias wrapper; each dtype needs its own rounding points and its own proof (in bf16 the
per-op proof is exhaustive over 65,536 inputs; in fp32 it cannot be). This slice ships
bf16 for Qwen3-TTS with the general signature; a second consumer is a later PR with its
own validation.

## 5. Vehicle

Triton stays. jiterator cannot turn FMA contraction off (fixed NVRTC arguments);
TorchScript's fuser keeps fp32 intermediates, so it is not bit identical when it fuses;
a regional torch.compile needs process-global Inductor switches and brings the startup
cost back; a CUDA source through `sglang.kernels.jit.utils.load_jit` is possible without
patching SGLang but needs nvcc at run time and keys its cache on the absolute source
path. Triton's cache is already redirected under `SGLANG_CACHE_DIR` when sglang is
imported (`sglang/__init__.py:14`, `srt/environ.py:1808`).

## 6. Design

1. `snake(x, a, r)`: one flat 1-D launch over `x.numel()`, channel from the element
   offset, five rounded steps exactly where eager materializes bf16. Sizes and pointers
   unspecialized, one binary per block size.
2. `a` and `r` are built once per module with the model's own eager expression, on the
   module's device and dtype, when the decoder is fused (weights are final there).
3. Block size and `num_warps` come from V-e8 over the decoder's real shapes, not from a
   table of guesses; if one setting is within noise everywhere, one binary.
4. Dispatch by a named predicate (bf16, CUDA, contiguous 3-D, int32-indexable); a module
   outside it keeps the original class. No try/except, no None, no allowlists.
5. Modules matched by class identity (the qwen_tts class), not by name.
6. At fuse time, outside capture, every replaced module is run fused and eager on a
   random tensor of its channel count and compared with `torch.equal`; a mismatch keeps
   the eager module and logs it.
7. Under torch.compile the op is an opaque registered custom op with a fake
   implementation (SGLang's pattern, `silu_mul_bitexact.py:77-84`), so main's compiled
   8-frame chunk never traces the launcher. End state, once fused eager matches compiled
   at the 8-frame chunk: the vocoder's torch.compile goes and startup drops below today's.

## 7. Validation tasks (nothing below is claimed yet)

- V-e8: eager, today's kernel, new kernel on the 29 real activation shapes of every
  captured key and as full decodes; bitwise everywhere; block size and num_warps sweep;
  the cost of unspecialized pointers.
- V-e8b: what `enable_fp_fusion` and `enable_reflect_ftz` do in the container's Triton
  (one compile with each, IR diff); whether the cast pairs alone already block the FMA.
- V-e8c: `a`, `r` built once against eager's per-call values, bitwise, on the 4090; the
  same for `1.0 / t` against `t.reciprocal()`.
- V-e9: boot with the flag on and compile on, before and after the custom-op registration.
- V-e10: server A/B on two cards with swapped rounds and the barrier, stream c16,
  long-form c16, seeded c1 byte identity, WER and similarity.
- Later experiment, not this slice: the activation writing straight into the conv's
  history concatenation (156 to 190 copy kernels per decode).
