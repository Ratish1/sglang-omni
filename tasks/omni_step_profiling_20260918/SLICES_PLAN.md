# Qwen3-TTS slices from the step profiles: plan

Base: upstream 144bd6399 on moss (RTX 4090 D, sm89), Qwen3-TTS-12Hz-1.7B-Base, streaming.
Evidence: READOUT_01.md and the ledger3 outputs. Every number below is a measured read
from those traces or arithmetic on the checkpoint config; every open question is an
experiment with a name.

## 1. Measurement contract

One step = the scheduler loop from one `omni.step` span start to the next. Per steady
step (same kind and batch size), from `scripts/step_ledger.py`:

| read | definition |
| --- | --- |
| wall | next span start minus this span start |
| scheduler stream busy | union of the durations of GPU activity owned by the scheduler thread, clipped to the step |
| scheduler stream empty | wall minus scheduler stream busy, split into head (step start to first kernel), gaps inside graph replays (between nodes of one replay), gaps between scheduler activities, tail (last kernel end to next step) |
| replay span / in-graph gaps | first kernel start to last kernel end of one graph launch; span minus the union of its kernels |
| other threads concurrent | overlap of other-thread GPU activity with scheduler activity (contention) |
| other threads in scheduler-empty time | other-thread activity while the scheduler stream is empty (fill) |
| device empty | wall minus the union of all GPU activity (true bubbles) |
| host loop outside run_batch | span end to next span start |
| alone / shared split | steps with zero other-thread activity concurrent, and the rest |

Ownership: cpu op thread through External id, else the launch thread; 0 mismatches and
0 unknown are required in every trace (C3). Timing comes from formal traces only (no
stack); mapping traces name kernels. A/B: one boot per arm on the same card, the same
driver command, the ledger on both, deltas quoted on p50 and mean of the same reads.

## 2. Where decode bs 16 goes today (38 steady steps)

| read | p50 | mean | p95 |
| --- | ---: | ---: | ---: |
| wall | 11.23 | 13.29 | 25.37 |
| talker replay kernel sum (alone / shared) | 4.28 / 5.04 | | |
| predictor replay kernel sum (alone / shared) | 6.58 / 6.59 | | |
| vocoder threads GPU busy (sum of the 2 threads' means; p95 9.2 and 9.7 per thread) | 0 | 3.00 | |
| scheduler stream empty | 0.13 | 1.73 | 10.20 |
| in-graph gaps, talker replay | 0.02 | 1.02 | 8.30 |
| device empty | 0.12 | 0.37 | 1.21 |

- Alone steps (27): 11.14 ms p50; the device is the step (0.13 ms empty).
- Shared steps (11): 13.75 ms p50. Vocoder kernels interleave with the talker graph's
  nodes; the talker replay's in-graph gaps reach 8.3 ms and its kernels slow 18 percent.
- The device is 97 percent busy, so every ms of vocoder GPU time is a ms of step time:
  vocoder 3.0, talker 4.3 to 5.0, predictor 6.6 ms of a 13.3 ms mean step.

First audio path (prefill bs 1, per request): talker prefill + predictor 10.45 ms device
inside a 25.9 ms host-bound span; vocoder first decode 24.5 ms device; reference encode
3.8 ms; preprocessing 1.0 ms.

## 3. Slice V: vocoder conv algorithms and layout

Code path: `streaming_vocoder.py` workers call the graph runner
(`incremental_codec_cuda_graph.py`), which precompiles `Qwen3TTSIncrementalDecoder
._decode_tensors` with `torch.compile(dynamic=False, fullgraph=True)`
(`incremental_codec.py:688`) and captures it. Convs are the tokenizer's `nn.Conv1d`
(`incremental_causal_conv1d`, `:122`, called on `torch.cat((history, hidden))`) and
`F.conv_transpose1d` (`:144`). No Qwen3-TTS code sets `torch.backends.cudnn.benchmark`,
so cuDNN's heuristic picks each algorithm once, at warmup and capture, and the graph
freezes it.

Measured: `implicit_convolve_sgemm` up to 480 us per call, `precomputed_convolve_sgemm`
390 us, cudnn `nchwToNhwc` / `nhwcToNchw` transposes 780 calls in the b16 window
(11.6 ms), dgrad kernels for the transposed convs.

Unknowns, as experiments:

- V1 attribution: one eager incremental decode per captured shape (the runner's
  `(fresh_frames, batch_bucket)` set) under the torch profiler with `record_shapes`,
  mapping every conv call (module key, input shape, groups, kernel size) to its kernels,
  time and transposes. Output: a table per conv key.
- V2 algorithm headroom: for each conv key, the same input through
  `torch.backends.cudnn.benchmark = True` (cuDNN times its candidates) against the
  heuristic pick, eager and inside a captured graph, output compared with
  `torch.equal` and max abs diff.
- V3 layout: whether the transposes are cuDNN converting NCHW to its NHWC kernels per
  call, and whether presenting the conv in the layout the chosen kernel wants removes
  them.

Gate: the vocoder's own outputs against the current path on the same codes (exact or
bounded diff, stated), then the seed-tts census (WER, similarity) on moss, then H100.
Decision rule: a choice is taken only if it is made per shape by measurement on the
device it runs on (not a constant tuned to one card).

## 4. Slice P1: one two-token predictor forward instead of two one-token forwards

Code path: `_code_predictor_forward_incremental` (`sglang_model.py:1518`). Per decode
position it runs `_predictor_forward_one_token` on the projected talker hidden
(`:1568`), then again on the projected layer-0 embedding (`:1574`), then 14 more times,
once per sub-step (`:1615`). That is 16 passes over the 5-layer predictor, each pass
reading all its weights. Count check: 16 x 20 layer GEMMs + 15 heads + 17
`small_to_mtp_projection` calls = 352 GEMMs per replay, the traced count.

The first two passes have no dependency between them other than causal attention (the
second token attends to the first). One pass over both tokens reads the weights once
and launches the layer kernels once.

Bytes per replay, bf16: 16 x 157.3 MB layers + 62.9 MB heads + 17 x 4.2 MB
projections = 2.65 GB, 2.63 ms at 1008 GB/s. Measured GEMM time 4.92 ms (bs 16).
Folding the first two passes removes 1/16 of the layer reads and 1 projection read, and
one pass of launches (20 GEMMs plus the norm, rope, attention and residual kernels of 5
layers).

Unknowns, as experiments:

- P1a cost of a two-token pass at bs 16 (M = 32 GEMMs, causal attention over 2) against
  two one-token passes, micro-benched on the replay's own modules.
- P1b numerics: codes from the folded chain against the current chain on recorded
  talker hiddens, same seeds; M = 32 GEMMs may round differently from M = 16.

Gate: P1b stated, then census. Applies to both cards: fewer bytes and fewer launches.

## 5. Slice P2: predictor sampling kernel latency

Code path: `_seeded_top_k_top_p_sample_kernel` (`sampling_kernels.py:304`), grid
`(batch_size,)`, `num_warps=8` (`:687`): one program per row sorts 2048 packed keys with
`tl.topk(k=block_k)`, `block_k = 64` for the checkpoint's subtalker top-k 50
(`_fused_raw_logit_block_k`, `:622`).

Measured: 40.4 us per call at bs 1 and at bs 16, 15 calls per step, 0.61 ms per step.
The time does not move with batch size: it is one program's selection latency, and 16
programs leave most SMs idle.

Unknowns, as experiments:

- P2a where the 40 us goes: the same kernel with the vocab split across programs (each
  program selects its top 64 of a slice, a second pass merges), bs 1 and 16, against
  the current kernel.
- P2b bit identity: the kernel's contract is bit-exact seeded sampling against the
  reference path (tie order and the float bit order are documented in the kernel); the
  existing `tests/unit_test/qwen3_tts` sampling tests plus a sweep of recorded logits
  must pass unchanged.

Gate: P2b exact. Applies to both cards (latency, not bandwidth).

## 6. Not a slice yet: GEMM kernel choice at M = 16

cuBLAS picks `cutlass_80_wmma_tensorop_bf16_s161616gemm` for the bs 16 decode GEMMs on
sm89: talker 35.4 us per GEMM (1.20 ms over its 2.80 ms floor), predictor 14.0 us (2.29
ms over its 2.63 ms floor), while bs 1 uses gemv at 86 and 68 percent of bandwidth. The
kernel cuBLAS picks depends on the card; SGLang's `--bf16-gemm-backend gemv` engages at
M = 1 on SM90 only. Recorded for the H100 pass; no change is designed on sm89 numbers
alone.

## 7. Order and runbooks

1. V1 and P2a micro-benches, P1a micro-bench: one runbook, one card, no server.
2. Design the slice whose measured headroom is largest; code on its own branch from
   upstream main; unit tests on the box; A/B with the driver (decode b1, b16, prefill
   b1) and the ledger; census on moss.
3. H100 confirmation before any PR claims time.
