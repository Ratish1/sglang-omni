# Targeted kernel reuse

Open PR #2110 changes Euler timestep input from one row to `2B` rows. Its reported quality effects require independent reproduction; the shape change alone does not prove scalar broadcasting mathematically incorrect. Establish which numerical baseline is accepted before graph/kernel parity comparisons. See [report 17](../reports/17_open_prs.md).

Use Nsight Systems to identify dominating operations first; only then use a short Nsight Compute pass on representative isolated kernels for memory throughput, tensor/FP32/FP64 pipelines, occupancy, launch geometry and stalls. Nsight Compute replay/instrumentation changes overlap and cannot replace end-to-end timing. Do not run it alongside the acceptance benchmark.

| Candidate | Existing implementation and contract | Decision boundary |
|---|---|---|
| Qwen attention | Conditional H100/CUDA default FA3, page size 1; wrapper normally calls installed `sgl_kernel.flash_attn`. | Confirm backend and installed artifact before benchmarking FlashInfer/Triton alternatives. Decode graph eligibility and custom eager prefill have different paths. |
| Qwen RMSNorm / residual norm | SGLang AOT/JIT or FlashInfer dispatch already present; Omni also has a dtype fallback patch. | Verify actual dtype/dispatch and dimensions. Another wrapper offers no automatic gain. |
| Qwen SiLU×gate | In-tree vectorized activation preferred when eligible, AOT fallback otherwise. | Inspect actual fallback/shape; avoid duplicating the existing operation. |
| AR top-k/top-p sampling | Cosy defaults to PyTorch; filtered seeded sampling has a different deterministic path. | A fused sampler must preserve filtering, ties, RNG/seed+position, penalties and stops. FlashInfer's seeded-filter limitation prevents a blind switch. |
| DiT adaptive LayerNorm + modulation + gate | LayerNorm with timestep-dependent shifts/scales and residual gates, not Qwen RMSNorm. | Fuse only the exact operation sequence; preserve accumulation, broadcast and dtype casts. |
| DiT Q/K/V projections | Three projections followed by partial-channel RoPE then head reshape. | A packed linear can concatenate weights/biases without changing layout, but requires checkpoint-load mapping and numerical verification. Separate PR. |
| DiT attention | Full valid-key or 50-frame chunk visibility; RoPE only first 64 projected channels in inspected source; output masking retained. | A backend must express these exact masks. AR paged KV APIs are not a drop-in substitute; no cross-Euler KV reuse. |
| Euler CFG/update | `2B` rows, conditional block then unconditional block; timestep-dependent guidance update. | A fused update could remove small kernels if they matter. Preserve floating evaluation order or explicitly qualify numerical drift. |
| HiFT Snake/convolution/FFT | Different nonlinearities, phase/noise and temporal boundaries. | Only optimize identified kernels with a complete waveform boundary contract; do not transplant SiLU/RMSNorm. |

Source paths and kernel geometry are documented in reports [11](../reports/11_upstream_execution.md), [12](../reports/12_cosyvoice_internals.md), [14](../reports/14_attention_backends.md), [15](../reports/15_upstream_config.md), [16](../reports/16_fa3_backend.md). Relevant FlashInfer AOT sources were inspected at SGLang's pinned build revision; the installed FlashInfer wheel may differ and must be captured.

Each kernel PR has one owner, input/output shape/stride/dtype/device contract, accumulation/rounding and alias rules, supported shape predicate, fallback, graph/compile compatibility, weight-loading implications and memory accounting. Use existing SGLang operations directly only when their contracts match. Otherwise keep a model-local adapter or a separately reusable upstream primitive with a small API. Remote validation compares frozen tensors and end-to-end quality, includes awkward widths/lengths and padding, and verifies that setup/copies do not erase the kernel gain. No speculative fusion stack is scheduled before a dominant-kernel trace exists.
