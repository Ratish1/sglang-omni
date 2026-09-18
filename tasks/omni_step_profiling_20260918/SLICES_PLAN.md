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

## 7. Bench 01 results (moss GPU 6, `scripts/predictor_bench.py`, `scripts/vocoder_conv_bench.py`)

P1a, GEMM chain in a CUDA graph, distinct weights per layer, median ms per replay:

| bs | current (16 passes, 352 GEMMs) | folded (2-token pass + 14, 331 GEMMs) | delta |
| ---: | ---: | ---: | ---: |
| 1 | 3.626 | 3.367 | -0.259 |
| 16 | 5.252 | 4.699 | -0.553 |

The bench's current chain at bs 16 (5.25 ms, GEMMs plus residual adds) sits near the
traced 4.92 ms of GEMM time per replay. The two-token pass runs its GEMMs at M = 32 on
`wmma_..._32x32_64x1` and `16x16_64x1` kernels, no slower per GEMM than M = 16. The fold
also drops one pass of the non-GEMM layer kernels (norm, rope, attention, adds), not in
this bench.

P2a: the fused sampler costs 40.8 us per call at bs 1 and bs 16 (the trace: 40.4).
`tl.topk(k=64)` alone costs 2.75 / 3.28 / 5.20 / 10.37 us over 256 / 512 / 1024 / 2048
keys, flat in batch size. Selection is a quarter of the kernel; the other 30 us is
elsewhere in it (next read: the kernel's own stages, timed apart).

V2: `torch.backends.cudnn.benchmark = True` against the heuristic on 22 decode shapes
(bs 1 and 8, fresh frames 1 to 8, 16, 32, 64): waveform bit-identical on all 22, decode
time within 1 percent. cuDNN's measured pick equals its heuristic pick; the algorithm is
not the lever.

V1 (bs 1, 1 fresh frame, 37 conv calls, 2.88 ms profiled GPU): every tensor-core conv
call runs two `nchwToNhwc` kernels before and one `nhwcToNchw` after its compute kernel;
the second input-side conversion is the constant weight, re-laid out on every call. The
1-frame `ConvTranspose [1024,1024,k2]` without conversions costs 31.9 us, the 2-frame one
with them 80.6 us; `ConvTranspose [1536,768,k16]` 160 us and `Conv [1024->1536,k7]`
100 us at one frame. Convs at 96 and 192 channels over 600 to 2000 samples take the
legacy `implicit_convolve_sgemm` without conversions, 36 to 68 us each. Next:
`scripts/vocoder_layout_bench.py` times each call with the weight laid out channels-last
once, and with input and weight both resident in that layout, with output identity.

## 8. Bench 02 results (`scripts/vocoder_layout_bench.py`, run01 prefill trace)

Every conv call of one incremental decode, device time per call from CUDA graph replays,
summed over the 37 calls (us):

| batch x fresh frames | current | weight channels-last once | weight and input resident | resident delta | bit-exact calls |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 x 1 | 1077.6 | 567.4 | 506.5 | -53% | 26/37 |
| 1 x 4 | 1378.4 | 815.9 | 710.0 | -48% | 25/37 |
| 1 x 64 | 7757.3 | 5429.4 | 3673.2 | -53% | 25/37 |
| 8 x 1 | 1325.3 | 1303.1 | 1112.1 | -16% | 22/37 |
| 8 x 4 | 3284.6 | 3240.4 | 2418.7 | -26% | 25/37 |
| 8 x 8 | 6505.3 | 5328.6 | 3656.1 | -44% | 28/37 |

- Bit-exact where cuDNN keeps the same tensor-core kernel and only the per-call layout
  conversions go: the transposed convs and the wide k7 convs.
- Not bit-exact: the depthwise ConvNeXt conv (groups 1024; the resident path picks a
  different kernel, and it is slower there), some 1x1 convs, and the 96 / 192 channel k7
  convs that today run `implicit_convolve_sgemm` (resident picks a tensor-core kernel, 3x
  faster at long lengths, different rounding). Max abs differences are per-call activation
  deltas; waveform-level error is not measured yet.
- A few calls are slower resident (the 1-channel final conv at bs 8, one k7 conv at bs 8 x
  86 samples), so a per-call layout choice has to come from measurement, not a blanket
  switch.

Server trace (run01 mapping prefill b1): per request the vocoder thread replays 3 to 5
decode graphs of 1,120 to 1,146 kernels, 11.4, 7.0, 3.2 to 3.4, 2.8 ms, 21 to 26 ms per
request in all. Eager vocoder kernels per request: 52 to 56 kernels, 0.06 ms. The first
audio of a voice clone request pays the reference-prefix bootstrap (window graphs, #2151)
and the first chunk through these graphs.

Consequence for the order: slice V is the largest measured lever (vocoder device time is
about 3.0 ms of a 13.3 ms decode step and 21 to 26 ms before the first audio, and its conv
time falls 16 to 53 percent resident). Next for V: an end-to-end prototype decoder that
keeps the conv chain channels-last, compared on full decode time and waveform error
(SNR against the current decoder on the same codes and state) per captured shape.

## 9. Order and runbooks

1. V1 and P2a micro-benches, P1a micro-bench: one runbook, one card, no server.
2. Design the slice whose measured headroom is largest; code on its own branch from
   upstream main; unit tests on the box; A/B with the driver (decode b1, b16, prefill
   b1) and the ledger; census on moss.
3. H100 confirmation before any PR claims time.
