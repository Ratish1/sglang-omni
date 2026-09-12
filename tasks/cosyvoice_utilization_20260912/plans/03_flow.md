# Flow preparation, compilation and graphs

Owners: `FunCosyVoice3Flow`, `_pack_flow_inputs`, `_generate_flow`, `_solve_flow_euler`, `_forward_flow_estimator`, compile/TRT loaders; pinned native Flow/DiT/mask source. Evidence: reports 02, 03, 11, 12.

Open PR #2110 changes the timestep tensor to one row per CFG input, and includes a FIFO change also covered differently by #2086. Its body reports quality/timing changes; these are not measurements from this investigation. Freeze and qualify the chosen timestep/ordering baseline before applying F1/F2 or comparing an existing accelerator. See [report 17](../reports/17_open_prs.md).

## Existing-path experiments first

Run eager, estimator `torch.compile`, and TensorRT as separate server launches with identical dtype and corpus. The two accelerators are mutually exclusive. Log actual estimator class, compilation/fallback counts, shapes, graph breaks and startup memory. Compiled warmup shape is not evidence that real prefix lengths avoid recompilation. TRT currently splits packed `2B` into request CFG pairs; a gain on singleton timing does not prove c16 packed throughput. Streaming mask parity is a mandatory gate for TRT.

## F1: immutable preparation and explicit workspace ownership

Trigger: repeated CPU→GPU noise transfers, mask construction or small setup kernels materially contribute. Cache only immutable device/dtype-specific values first: the exact fixed noise source, cosine time schedule, and masks/rotary inputs whose cache key includes mode, shape, valid lengths and model revision. Do not change noise realization, timestep dtype, CFG row order, mask truth convention or RoPE layout. Limit cache bytes and evict only when no active reference uses an entry.

Workspaces for x/CFG/condition buffers need a single explicit owner per in-flight solve. Current vocoder scheduling serializes its calls, but assert/encapsulate that contract rather than assuming a future second worker is safe. Return tensors must not alias storage overwritten by the next solve while HiFT or retained streaming history still reads them; clone or transfer ownership at the output boundary. Separate immutable caches from mutable workspaces so each can be reviewed independently.

Remote proof: B1 and B>1; unequal prompt/target lengths; causal and finalized masks; control padding; fixed noise; repeated calls that previously returned tensors still alive; dtype/device changes; upper noise length and empty/invalid inputs; exceptions/fallback cleanup. Compare each Euler result, final mel and HiFT input. Measure preparation cost and memory as well as end-to-end throughput. Roll back via the uncached eager path.

## F2: bounded graph execution

Open [PR #1861](https://github.com/sgl-project/sglang-omni/pull/1861) already implements whole-solver Flow CUDA graphs. This design is an acceptance checklist for reconciling that head, not authorization to duplicate its implementation. Consult [the dated overlap report](../reports/17_open_prs.md), recheck current state, and narrow any follow-up to a demonstrated remaining gap.

Trigger: launch gaps persist after existing compile/F1, and observed shapes justify bounded buckets. A graph key must include batch bucket, mel bucket, streaming/finalize mode, dtype/device and accelerator path. Fixed buffers hold `2B` CFG layout, lengths, masks, conditioning, timestep and output. Invalid requests/frames remain masked through attention and final cropping. Capture setup must move host-reading chunk-mask decisions outside the graph. Preserve all ten solver evaluations and timestep-dependent activations; caching a DiT hidden state across Euler steps is invalid.

Choose estimator-only versus full-solver capture from measured opportunity and memory cost. Full-solver capture adds tensor-pointer/loop/output ownership obligations; SGLang's graph buckets, stable-buffer registry, padding gates and pool ownership are useful designs, but its AR `ForwardBatch`/KV cache is not a direct Flow adapter. Keep graph pools independent unless serialization and allocator lifetime are proven. Bound the total graph memory and retain eager fallback for oversized/unsupported shapes. No capture while another stage's stream enters an incompatible global capture mode.

Before implementation, produce the exact bucket list from trace histograms, peak-memory calculation, capture-stream handoff/event rules, and output lifetime proof. Compare valid rows against eager at identical padded shapes, then evaluate actual mixed batches. Capture hit/fallback counters and profiler-off paired full-corpus results decide rollout. F2 remains conditional until these shape/ownership inputs exist.
