# Slice V1: keep the vocoder conv chain channels-last

## 1. Mechanism

cuDNN's tensor-core convolution kernels read and write NHWC (channels fastest). The
incremental decoder hands every conv an NCL tensor (time fastest) and a NCL weight, so
each tensor-core call runs `nchwToNhwc` on the input, `nchwToNhwc` on the constant
weight, the compute kernel, and `nhwcToNchw` on the output (bench 01 V1, on the real
decoder). Keeping activations and weights in the layout the kernel consumes removes
those three kernels per call. This holds on every NVIDIA card with tensor cores; the
size of the saving is what varies by card and shape, and it is measured, not assumed.

Measured (bench 02, conv calls only, CUDA graph replay per call): total conv time falls
16 to 53 percent across bs 1 to 8 and 1 to 64 fresh frames with weight and input
resident; 22 to 28 of 37 calls bit-exact.

## 2. The decoder, from the checkpoint config and qwen-tts source

`speech_tokenizer/config.json` decoder_config: codebook_dim 512, latent_dim 1024,
decoder_dim 1536, upsampling_ratios (2, 2), upsample_rates (8, 5, 4, 3), 8 transformer
layers (16 heads, head_dim 64, window 72), 16 quantizers of 2048 codes. Total upsample
1920 samples per frame.

```
codes (B,16,T)
 quantizer.decode: 16 embedding lookups summed, 2 output_proj Conv1d k1 (512)  -> (B,512,T)
 pre_conv Conv1d 512->1024 k3 (history 2)                                         incremental_codec.py:622
 pre_transformer 8 layers on (B,T,1024) (already time-major)                      :628
 2 x [ ConvTranspose1d 1024->1024 k2 s2 ; ConvNeXt(1024): dwconv k7 groups 1024,
       LayerNorm, Linear 1024->4096, GELU, Linear 4096->1024 (time-major) ]       :632-646
 decoder.0 Conv1d 1024->1536 k7                                                    :648
 4 blocks, rates 8,5,4,3: SnakeBeta, ConvTranspose1d C->C/2 k=2r s=r,
       3 residual units (SnakeBeta, Conv1d k7 dil 1|3|9, SnakeBeta, Conv1d k1)     :651-669
 SnakeBeta(96), Conv1d 96->1 k7, clamp                                             :670-673
```

Conv calls per decode: 2 (quantizer) + 1 + 2 transposed + 2 depthwise + 1 + 4
transposed + 24 + 1 = 37, the count bench 02 recorded. State per stream: 29 conv
histories, 6 transposed-conv overlaps, 8 x 2 K/V buffers.

## 3. Design

Carry activations as (B, T, C) contiguous tensors through the conv chain. Present each
conv with `x.transpose(1, 2)` (a (B, C, T) view whose channels are fastest) and a weight
copy laid out channels fastest; cuDNN then sees channels-last input and weight and
returns channels-last output, whose `.transpose(1, 2)` is again (B, T, C) contiguous.

```
graph replay                                      incremental_codec_cuda_graph.py:300
 arena.gather_by_index   histories (slots, L, C), overlaps (slots, right_pad, C)   codec_state_arena.py:187
 _decode_tensors                                  incremental_codec.py:610
   quantizer.decode (qwen-tts, returns a time-major view already) -> (B,T,512)
   causal conv:      cat((history, x), dim=1) on (B,T,C); conv on views; history = last L rows
   transformer:      unchanged, takes and returns (B,T,C)
   transposed conv:  output (B,T',C); add overlap on the first rows; bias on the last dim
   ConvNeXt:         dwconv (layout per V1-e3), LayerNorm and Linears on (B,T,C) as today
   SnakeBeta:        the tokenizer module on the (B,C,T) view; elementwise ops keep strides
   final conv:       (B,T,1) -> (B,1,T), clamp
 arena.scatter_by_index
```

Changes outside `_decode_tensors`:

- `init_state` (`:524`) allocates histories and overlaps as (B, L, C); `state_spec`
  triples keep their meaning. Arena gather and scatter work per row, so they are
  unchanged.
- The two `.contiguous()` calls that force NCL (`:122`, `:173`) go.
- Weight copies: made once in `Qwen3TTSIncrementalDecoder.__init__` for the 37 convs
  (decision D1, section 5).
- Tests that build NCL states by hand: `tests/unit_test/qwen3_tts/
  test_incremental_codec_cuda_graph.py:77`, `test_pipeline.py:2327`.
- The eager single-stream path (`streaming_vocoder.py:1349`) builds its state lazily from
  `Qwen3TTSIncrementalCodecState()`; `incremental_causal_conv1d` must allocate a missing
  history as (B, L, C).

Not changed: the non-streaming `tokenizer.decode` path (qwen-tts modules), the legacy
left-context graphs (`streaming_vocoder.py:412`), the fused SnakeBeta (off by default,
audit A1).

## 4. Experiments (script `scripts/vocoder_resident_bench.py`, TESTING.md section 2)

- V1-e0 dispatch. For one decode at bs 1 and 8, every conv call of the resident chain:
  kernels launched (by name), and the output strides. Pass: no `nchwToNhwc` /
  `nhwcToNchw` kernels, every output (B, T, C) contiguous after its transpose. Also
  prints the state_spec counts and the bytes of the channels-last weight copies. If the
  3D view does not reach the NHWC kernels, the chain uses explicit 4D conv2d calls
  (bench 02's form) instead.
- V1-e1 timing, every captured key: widths 1 to 8, 16, 32, 64 at batch 1, 2, 4, 8 (the
  cold, warm and window sets of `streaming_vocoder.py:789-802,988-995` with defaults).
  Per key, decode captured in a CUDA graph, median device ms over replays, and kernel
  count and transpose count from one profiled replay. Arms: current, resident with the
  depthwise conv resident, resident with the depthwise conv in NCL. Width 8 also
  compiled (`torch.compile(dynamic=False, fullgraph=True)` as in `:688`) for current and
  resident.
- V1-e2 numerics on real activations. Codes from `tokenizer.encode` of seed-tts
  reference audio (real speech, 40 to 120 frames). Each utterance decoded incrementally
  with the default chunk schedule (1, 2, 4, then 8), at bs 1 and as a bs 4 cohort. Arms:
  current eager, current compiled (width 8 chunks), resident eager, resident compiled,
  and the truth: the same decoder in fp32 (weights upcast, fp32 state). Per arm and
  stream: SNR in dB of the concatenated waveform against the truth, max abs. The
  existing spread (current eager vs current compiled) is today's tolerance; pass when
  resident's SNR to the truth is not below current's at the same compile setting.
- V1-e3 depthwise layout: read from V1-e1's two resident arms. The depthwise conv stays
  NCL (one transpose in and out per ConvNeXt block) only if resident depthwise is
  slower at every key. The same read on the H100 decides it for good.
- V1-e4 non-streaming path: `tokenizer.decode` of a 10 s and a 30 s utterance at bs 1
  and 8, with the decoder's own conv weights NCL (today) and re-laid channels-last in
  place. Decides D1.

## 5. Decisions after the experiments

- D1 weights. A separate channels-last copy costs the conv weight bytes again (V1-e0
  prints them). Re-laying the module weights in place costs nothing, but the
  non-streaming decode then sees channels-last weights with NCL inputs. In place if
  V1-e4 shows non-streaming no slower; otherwise the copy.
- D2 compiled width. If the resident chain compiles with no transposes and no slower
  than today's compiled width 8, keep compiling it; otherwise report the inductor
  output (kernels, copies) before designing further.

## 6. Gates and A/B

1. Unit tests on the box (`tests/unit_test/qwen3_tts/test_incremental_codec*.py`,
   `test_pipeline.py`), updated for the (B, L, C) state.
2. V1-e2 pass.
3. Matrix A/B (P1 as A, P1 + V1 as B). Predicted: vocoder GPU per decode step down by
   V1-e1's delta at the observed key mix; TTFC down by the bootstrap windows' delta.
4. Census on B.
