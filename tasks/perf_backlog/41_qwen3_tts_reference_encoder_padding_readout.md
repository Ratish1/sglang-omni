# 41. Reference encoder padding check readout, 2026-09-14

Archive `ref_encoder_padding_check-handoff.zip`, unpacked under
`artifacts/ref_encoder_padding_check-handoff`. Upstream main in `tmp/main`, GPU 1, 64
distinct references, two tokenizer instances (bfloat16 as the vocoder loads it, float32
as the checkpoint stores the encoder), host side padding buffers and 16 quantizers on
both. Every variant is compared to the unpadded encode of the same instance; a
difference is split into the last two frames and the frames before them.

| variant, against the unpadded encode | bfloat16 | float32 |
| --- | --- | --- |
| again (same shape, second run) | 64 of 64 exact | 64 of 64 exact |
| aligned (zero padded to the next 1,920 multiple) | 60 differ, median 110 codes, 6 tail only | 60 differ, median 16 codes, 33 tail only |
| bucket (zero padded to 48, 64, 96, 128 or 192 frames) | 63 differ, median 119 codes, 3 tail only | 62 differ, median 24 codes, 18 tail only |
| float32 against bfloat16, unpadded | 64 differ, median 510 of about 880 codes | |
| float32 graph against float32 eager at the bucket | | 64 of 64 exact |

Eager one reference host ms: bfloat16 7.5, float32 6.5. Float32 replay device ms per
bucket: 3.1, 3.3, 3.8, 4.0, 5.0 (bfloat16 in doc 40: 2.4 to 3.7).

## 1. The two mechanisms, located

The encoder is deterministic at a fixed shape and a captured graph reproduces its own
eager shape bit for bit. The differences come from two places in the model, neither a
bug and neither in omni code.

**The last frame.** `MimiConv1d.forward` (transformers 5.12.1 `modeling_mimi.py`)
pads each layer's input on the right with that layer's own `extra_padding` zeros before
the conv, so the last window of a stride 8, 6, 5 or 4 layer reads zeros beyond the
signal. With the waveform zero padded up front, that window reads the conv of the zero
tail instead (bias, ELU of bias, the residual block's output on zeros). The last output
frame changes at every downsampling level, and its 16 codes differ: the tail rows
above are 13 to 16 codes in nearly every reference, in both dtypes, aligned or bucket.
Only the last frame is touched, which is why 33 float32 references differ in the tail
alone.

**Shape dependent numerics through the quantizer.** Everything before the quantizer is
causal, so frame t of a longer input is the same arithmetic as frame t of the shorter
one, but not the same kernel: cuDNN picks its conv algorithm and cuBLAS its tiling by
the length, and the reduction order changes the rounding of the embeddings by a few
ulps. `MimiEuclideanCodebook.quantize` turns that into a discrete flip wherever two of
the 2,048 centroids are near equidistant, and `MimiResidualVectorQuantizer.encode`
subtracts the chosen centroid before the next stage, so one flip at stage k changes
the frame's later stages too. The float32 earlier frame differences are sparse (a few
frames, 2 to 52 codes) because float32 rounding rarely reaches a near tie; bfloat16
rounds at 3 significant digits and reaches one in most frames (86 to 268 codes). The
same amplifier is why the float32 and bfloat16 encodes of the same unpadded waveform
disagree on 58 percent of their codes: the codes the server feeds the talker today are
already one rounding of a code set that has no single exact answer.

A non causal layer is excluded: it would move every frame in float32 the way bfloat16
does, and the float32 earlier frame differences are a handful of flips.

## 2. What this means for the design

Bit identity with today's codes is not a property any batched or bucketed encode can
have, in either dtype, and today's own batching (pad to the longest member) already
lacks it. The correctness gate for the encoder slice is the one the census already
runs, WER and speaker similarity inside the bands on the full corpus, with the codes
compared as a distribution rather than as bits. Two choices follow from the numbers:

1. Dtype stays bfloat16, the dtype the stage loads the tokenizer with today. It keeps
   the smallest distance from the codes the talker sees now (a median of 119 codes
   moved by the bucket against 510 for a dtype change), the cheapest replay, and no
   memory change. Whether the tokenizer should run in float32 at all is a real
   question (58 percent of codes move), but it is a separate slice with its own
   quality census, not part of removing launches.
2. Every reference is zero padded to its bucket before the encode in both the eager
   fallback and the graph path, the frame count stays `ceil(samples / 1920)`, so a
   reference's codes depend on its bucket only: deterministic across boots and
   batches, which today's codes are not.

Buckets 48, 64, 96, 128 frames at batch 1 cover the corpus (p90 75, max 98) at 32 MiB
per key; batch 2 keys wait for a measured need, since the batcher's 2 ms window
almost never holds two waveforms.

## 3. Next

The runner: an omni owned graph runner for the reference encoder in the batcher's
thread and stream, shaped like the codec runners, capturing `encoder.encode` at the
bucket keys after the padding buffers move to the host at load, with the eager path
(same padding, same 16 quantizers) as the miss path. One PR, measured on the doc 33
protocol with the quality census as its gate, then #2123 and #2126 remeasured on the
new main.
