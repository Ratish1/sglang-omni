# HiFT batching, precision and incremental state

The open-PR audit found no CUDA streaming HiFT batching or incremental-state implementation. PRs #2086/#2110 still affect the scheduler that would own H1 participants and output ordering; reconcile those heads first. See [report 17](../reports/17_open_prs.md).

Owners: Omni `_hift_delta`, `_mel2wav_batch`, streaming per-request state; pinned Cosy `CausalHiFTGenerator`, F0/source modules, causal convolutions and FFT. Evidence: reports 02, 03, 12.

## H1: compatible streaming batch execution

Trigger: sequential per-request HiFT consumes a material fraction after batched Flow. Begin with **equal accumulated mel lengths and identical finalize mode**, without padding different histories. Concatenate complete histories in stable participant order, invoke HiFT once, then crop each row at its own prior `speech_offset`. Preserve retained mel per request and update offsets only after successful output creation. Return each delta to the same request; abort suppression and resource release stay in the scheduler.

The native causal source has fixed batch-one phase/noise that broadcasts, which makes this path plausible; it is not proof of exact batched equivalence. F0 changes module dtype to float64; convolution/FFT kernels may choose different algorithms with a batch. Establish raw waveform prefix/suffix parity before implementing the integration. A request that produces no new samples must not advance incorrectly or cause a duplicate final.

Do not initially copy buffered right-zero-padding semantics into streaming: padded suffixes can affect non-final lookahead, F0/source and FFT boundaries. Mixed-history batching is its own later proof/PR if equal-length batching demonstrates value. Measure batch fill, cumulative frames and padding separately. Keep per-request fallback; rollback is one policy switch, not a model checkpoint change.

## Precision experiment

F0's float64 conversion is explicit in the reference. It may be costly on the actual shapes, or a small fraction. Changing it to float32/BF16 is a numerical change and needs frozen-mel F0/waveform comparisons, long voiced spans, unvoiced boundaries and final tails, plus full quality/listening evaluation. No dtype change belongs in the initial profiling or batching PR. Reuse existing autocast controls only for the operations they actually cover; default buffered FP32 HiFT does not remove the internal FP64 F0 branch.

## H2: incremental HiFT is blocked on a complete state derivation

Retaining only the last N mel frames and concatenating emitted waveforms is not yet a correct design. Prove and document state for: right lookahead in F0 and input convolution; every causal convolution/dilation at every upsample scale; harmonic phase integration and wrap behavior; fixed noise prefix indexing by absolute sample offset; source STFT centered framing; magnitude/phase generation; ISTFT overlap/window normalization; non-final cropped tail; final zero context and final output length. State tensors must be per request, device/dtype-owned, bounded and released on finish/abort.

For a prefix ending at t, identify the largest sample index whose value cannot change when more mel arrives. Emit only through that index and carry the rest. Include the exact receptive-field calculation and units at each scale; the 480 samples/mel ratio alone is insufficient. Separate a proven continuation implementation from scheduler adoption. Frozen full-history reference outputs at every existing hop are the oracle, including initial/final boundary cases and long sequences. Until that proof exists, H2 is an investigation, not an execution-ready cache patch.
