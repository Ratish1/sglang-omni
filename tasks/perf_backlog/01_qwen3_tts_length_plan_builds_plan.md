# 01. Qwen3-TTS: the per new length cost in the speech tokenizer and the vocoder (T40)

Status: Conditional. The design is chosen, two facts it rests on are
verified only against library defaults and one measurement has to run
before the slices are sized (section 9). Task class: cross boundary
(preprocessing thread, vocoder stage, torch's cuDNN attention cache,
qwen-tts model code that omni does not patch).

Revisions read: omni `perf/qwen3-tts-predictor-startup-capture` at
`d7e34a16c`, torch `v2.13.0` (`/Users/ratish/pytorch`, `git describe
--tags` checked), sglang `v0.5.18`, qwen-tts `0.1.1` (the wheel, unpacked
in the scratchpad), transformers `5.12.1` (the omni pin, from the uv
cache). Two research passes, the second a line by line verification of
the first, both read the files named below in full.

## 1. Requirement

A request whose reference length or output length the process has not
seen before pays a host stall on the thread that runs the model. Remove
that stall for the shapes serving actually produces, with no change to
the audio of any request and no steady state cost past the noise of the
c1 A/B (doc 24 section 4, about 2 ms per request between runs).

Non goals: the talker and its code predictor (docs 21 to 24), streaming
mode's initial chunk graphs (already captured at startup), other models.

## 2. What it costs today

Measured on the run 7 `A_c1` and `B_c1` artifacts
(`scripts/t40_firstseen_residual.py`): latency fitted as
`a + b * completion_tokens + c * prompt_tokens` over the 951 requests
whose reference length and output length had both been seen earlier in
the run, then the residual of the rest. `prompt_tokens` is the reference
length in codec frames (request_builders.py, `apply_sglang_qwen3_tts_result`,
reports `data.ref_code_len`).

| Requests at c1 | Count of 1087 | Excess over the fit, A | B |
| --- | ---: | ---: | ---: |
| reference length new, output length seen | 28 | +81 ms | +77 ms |
| output length new, reference length seen | 76 | +68 ms | +69 ms |
| both new | 32 | +134 ms | +135 ms |
| both seen | 951 | 0 | 0 |

Fit: 65 ms + 7.07 ms per output token, and -0.02 ms per reference frame,
so a known reference length costs nothing at steady state. 136 of 1087
requests pay the excess, 42 of the first 50 and 70 of the first 100.
Total 11.7 s over the run, 2.4% of c1 latency, about +16% on the first
50 requests. Doc 22 section 3 attributed 38 ms (tokenizer) and 34 ms
(vocoder) of this to cuDNN attention plan builds by switching cuDNN off.
The remaining 40 ms and 34 ms per side are per new shape work that is not
attributed yet (section 9, M1).

## 3. Repository and library facts

Every line below was read at the revisions above. R marks a repository
fact, X an external one, M a measurement.

Encoder path:

- R1 `stages.py:43-67` loads `Qwen3TTSTokenizer.from_pretrained` with no
  `attn_implementation`, and no caller sets one for Qwen3-TTS
  (`stages.py:128,153`, `engine_builder.py:32-33,88`). transformers
  resolves it to `sdpa` (`modeling_utils.py:2018`).
- R2 Every `speech_tokenizer.encode` in the process runs on one persistent
  daemon thread, `qwen3-tts-ref-code` (`request_builders.py:641-646`, the
  only two call sites at `:740` and `:752`). The batcher drains up to 8
  waveforms within 2 ms and encodes them as one batch. The preprocessing
  pool threads (8, `stages.py:105-119`) run the speaker encoder and block
  on the batcher's future (`request_builders.py:854-856`).
- R3 `Qwen3TTSTokenizer.encode` (`qwen3_tts_tokenizer.py:241-257`) calls
  the feature extractor with no padding argument. `EncodecFeatureExtractor`
  pads to the longest waveform in the batch (`feature_extraction_encodec.py:137-139,182-190`)
  and returns the true lengths as `padding_mask`.
- R4 `Qwen3TTSTokenizerV2Model.encode` runs the whole padded batch through
  the Mimi encoder once and trims each row's codes by its mask after
  attention (`modeling_qwen3_tts_tokenizer_v2.py:961-991`, trim at `:984`).
- R5 The encoder is transformers Mimi (`:899`). One transformer pass, no
  chunking, `use_streaming` False (`modeling_mimi.py:1455-1488,1522-1611`,
  `configuration_mimi.py:117`). The only attention call is
  `MimiSdpaAttention.forward`, `F.scaled_dot_product_attention` with
  `attn_mask=causal_mask`, `is_causal = causal_mask is None and q_len > 1`
  (`modeling_mimi.py:900-906`).
- R6 The mask is skipped (None) when q_len equals kv_len, no padding mask
  is passed and kv_len is below `sliding_window`
  (`masking_utils.py:292-301,521-522`). `_encode_frame` passes no
  attention mask (`modeling_mimi.py:1473-1478`).
- R7 Mimi convolutions are causal by default (`use_causal_conv` True,
  `configuration_mimi.py:95`, left padding at `modeling_mimi.py:341-343`,
  transposed conv trims the right at `:383-386`).
- R8 Mimi defaults: 24 kHz, hidden 512, 8 layers, 8 heads, head dim 64,
  sliding window 250, upsampling ratios [8, 6, 5, 4]
  (`configuration_mimi.py:86-127`), plus a stride 2 downsample
  (`modeling_mimi.py:1422-1431`). qwen-tts sets `encode_downsample_rate`
  1920 and `decode_upsample_rate` 1920 (`configuration_qwen3_tts_tokenizer_v2.py:150-151`).
  So the codes run at 12.5 Hz and the encoder transformer at 25 Hz.

Vocoder path:

- R9 Non streaming requests never touch the vocoder's CUDA graphs.
  `_vocode_payloads` (`streaming_vocoder.py:1705-1735`) calls
  `tokenizer.decode` on the batch, which pads the code rows to the longest
  with value -1 (`qwen3_tts_tokenizer.py:329`) and runs
  `decoder.chunked_decode` with chunk 300 and left context 25
  (`modeling_qwen3_tts_tokenizer_v2.py:886-896,1015`). It runs on the
  stage's scheduler thread `scheduler-vocoder` (`pipeline/stage/runtime.py:241-246`,
  `streaming_simple_scheduler.py:124-145,381`). Batch up to 8 within 2 ms
  (`stages.py:154-155`).
- R10 The decoder is causal end to end: causal conv and transposed conv
  primitives (`:181,190-191,199-205`), a pre transformer whose every layer
  is sliding causal attention with window 72 (`configuration_qwen3_tts_tokenizer_v2.py:75-87,116-121`,
  `modeling_qwen3_tts_tokenizer_v2.py:293,419,550-551`), 8 layers, 16
  heads, head dim 64. `chunked_decode` trims the left context (`:894`).
  The attention goes through transformers `sdpa_attention_forward`
  (`integrations/sdpa_attention.py:77,92-101`).
- R11 The vocoder decodes reference codes plus generated codes: omni
  concatenates them (`request_builders.py:1313-1323`) and trims the
  reference share of the waveform (`streaming_vocoder.py:1746-1749`). So
  the decode length is `prompt_tokens + completion_tokens` frames.
- R12 Streaming mode captures 40 CUDA graphs (10 frame counts, 4 batch
  sizes) at `warmup_now` on the stage construction thread and falls back
  to eager `chunked_decode` on a miss (`streaming_vocoder.py:42-77,294-383,611-615,1074-1076`).

torch:

- X1 On sm90 and sm100 with cuDNN above 9.15, the SDPA priority order is
  set to cuDNN first, once, process wide, on the first dispatch
  (`sdp_utils.cpp:76-121`), unless `TORCH_CUDNN_SDPA_DEPRIORITIZED` is
  set. Elsewhere flash is first and none of this applies.
- X2 The cuDNN attention plan cache is `thread_local` and unbounded
  (`MHA.cpp:343-400`). The key holds the q, k, v dims and strides, the
  bias dims and strides, b, h, s_q, s_kv, d_qk, d_v, dropout, is_causal,
  return_softmaxstats, has_attn_bias and use_ragged (`:198-222`). A miss
  runs validate, build_operation_graph, create_execution_plans (heuristic
  mode A only), check_support and build_plans (`:640-645,1384-1399`). No
  API prewarms, bounds, clears or serialises it. The only way to fill a
  thread's cache is to run the shape on that thread.
- X3 `TORCH_CUDNN_SDPA_AVOID_RECOMPILE=1` rounds s_q, s_kv and the
  sequence dims up to powers of two and drops the batch strides from the
  key, for calls with no attention bias whose q, k and v satisfy
  `x.transpose(1, 2).is_contiguous()` (`MHA.cpp:150-181,280-294`).
- X4 `torch.nn.attention.sdpa_kernel` and
  `torch.backends.cuda.enable_cudnn_sdp` flip flags on the process global
  context (`Context.cpp:122-125`, `Module.cpp:900-908,1043-1053`). They
  are not thread scoped. That is why doc 22's verdict on the predictor
  forbids using them around the tokenizer and vocoder in the same process.

Measurements:

- M1 Section 2. M2 In the run 7 corpus at c1 the encoder sees 61 distinct
  reference lengths (36 to 111 frames, 72 to 222 encoder positions) and the
  vocoder 108 distinct decode lengths (57 to 199 frames). At c16 the shape
  is the maximum over each batch of up to 8, so the realised key set is
  larger and depends on arrival timing. M3 Reference audios repeat (666
  distinct over 1088 requests) and the ad hoc reference cache
  (`request_builders.py:928-934`) serves repeats, so encode calls are
  fewer than requests.

## 4. Current mechanics

```text
pipeline process (preprocessing, tts_engine, vocoder stages)

8 preprocessing threads      ->  speaker encoder (convs, no attention)
   |  submit(waveform)            block on future
   v
1 thread qwen3-tts-ref-code  ->  encode(batch <= 8, padded to longest)          [cuDNN cache A]
      feature extractor pad  ->  Mimi convs (causal) -> 8 x SDPA (B, 8, T_enc, 64), is_causal
      -> stride 2 downsample -> quantizer -> trim by mask
      new (B, T_enc)  =>  plan build on this thread, ~38 ms, plus other per shape work

talker (sglang)              ->  codes, ref codes + generated codes to the vocoder

1 thread scheduler-vocoder   ->  decode(batch <= 8, padded to longest with -1)    [cuDNN cache B]
      chunked_decode(300, 25) -> quantizer -> causal pre conv
      -> 8 x SDPA (B, 16, T, 64), mask materialised at T >= 72
      -> causal upsample and decoder convs -> trim left context, trim ref share
      new (B, T)  =>  plan build on this thread, ~34 ms, plus other per shape work

construction thread          ->  40 streaming graphs (warmup_now)                [cuDNN cache C]
```

Two persistent threads, one cache each, no rotation. A startup warmup
must run on those two threads to be seen by them.

## 5. Supported state space

| Dimension | Variants | Consequence |
| --- | --- | --- |
| Hardware | sm90 and sm100 with cuDNN above 9.15: cuDNN first (X1). A100 and others: flash first, no plan cost | the fix must be a no op where cuDNN is not first |
| Mode | non streaming: eager chunked decode on the scheduler thread. Streaming: graphs for the 40 initial shapes, eager fallback for the rest on two more threads | slices cover non streaming first, streaming eager fallback is the same code path |
| Batch | encoder 1 to 8, vocoder 1 to 8, padded to the longest row | the key set is (B, T max) per thread |
| Lengths | references 2.9 to 8.9 s here, encoder positions below the 250 window so no mask. Outputs up to 2048 tokens, vocoder chunks of at most 325 frames, mask from 72 frames | a reference above 10 s flips the encoder to a materialised mask, a different key family |
| Deterministic mode | vocoder decodes one row at a time (`streaming_vocoder.py:1717-1722`) | out of scope, not served |

## 6. Candidates

C1 Length ladder plus a warmup on the owning thread. Pad every encode
batch to the next ladder length in samples and every decode batch to the
next ladder length in frames, at the omni seams (the batcher before
`encode`, `_vocode_payloads` before `decode`), trim the codes and the
waveform to the true lengths in omni (the models' own trims only know
the lengths they were given), and run the ladder times the batch sizes
once at stage start on `qwen3-tts-ref-code` and on `scheduler-vocoder`.
Removes every per new shape cost on both sides, attention and
convolutions alike, because every shape becomes one of a fixed set built
before the first request. Costs padded compute on every call and a
bounded startup sweep. Numerics: R7 and R10 say appended zeros cannot
change earlier outputs of any conv or of causal attention. What can
change is the kernel cuDNN picks for the padded shape and its reduction
order in bf16, the same variation batching to the longest row already
introduces today (R3, R9). Gate G1 measures both.

C2 Warmup of exact shapes only. Without padding, only exact (B, T) hits
help. At B equal 1 the encoder has about 180 possible positions and the
vocoder about 270 frame counts, several seconds of startup per thread,
and batches of 2 to 8 cannot be covered. Rejected alone, it is the
warmup half of C1.

C3 `TORCH_CUDNN_SDPA_AVOID_RECOMPILE=1`. Zero code. Applies only to bias
free calls with BSHD contiguous inputs (X3): the encoder qualifies (no
mask below 250 positions, transformers builds q, k, v by view and
transpose), the vocoder does not above 72 frames, and the predictor's
BHSD cache tensors do not, so the talker is untouched. Covers at most the
attention share of the encoder side, 38 of about 80 ms, nothing of the
vocoder side and nothing of the conv share. Kept as an experiment (T2),
not as the design.

C4 Flash for these modules through a registered attention function. Not
thread scoped flags (X4) but a per module implementation. Mimi selects
its attention by class name (`MIMI_ATTENTION_CLASSES`), so the encoder
needs a class patch, the vocoder can use `AttentionInterface.register`.
Removes the attention share only, adds whatever flash costs over cuDNN
per call at these shapes for the life of the process, and patches model
code omni does not own. Rejected.

Decision: C1, gated by G1 to G4.

## 7. Target design

Encoder, in `_Qwen3TTSRefCodeBatcher._run` before `encode` (R2):

- ladder in codec frames, `ENCODER_LENGTH_LADDER`, from the corpus range
  (frames 36 to 111 today) with a step chosen by G2, converted to samples
  as `frames * 1920` (R8).
- pad each waveform of the batch with zeros to the ladder length of the
  batch's longest waveform. The feature extractor then pads nothing.
- after `encode`, trim each row's codes to `ceil(samples / 1920)` of its
  true length. The quantiser's frame completion pad stays inside the last
  true frame, which is the same rule the model applies today.
- the warmup: at thread start, for `B in (1, 2, 4, 8)` and every ladder
  length, one `encode` of zero waveforms. Plans and conv algorithms then
  exist in cache A before the first request.

Vocoder, in `_vocode_payloads` before `decode` (R9):

- ladder in frames, `VOCODER_LENGTH_LADDER`, covering the range up to the
  chunk cap of 325 frames, step by G2.
- pad every code row with the model's own pad value -1 to the ladder
  length of the batch's longest row, then trim the waveform to
  `true_frames * 1920` samples before the existing reference trim.
- the warmup: at scheduler start, on `scheduler-vocoder` (the loop's
  first iteration, or a start hook that runs on that thread), for
  `B in (1, 2, 4, 8)` and every ladder length, one `decode` of -1 rows.

Both warmups are no ops where cuDNN is not first (X1): they cost the
forward passes and nothing else, so no hardware gate is needed, but the
startup budget G4 applies everywhere.

Unchanged: qwen-tts model code, the talker, the streaming graphs, the
reference cache, the request and result contracts (the trims restore the
exact lengths the caller sees today).

Observability: one log line per warmup with the shape count and the wall
time, and the existing per request latency. Rollback: revert, no state.

## 8. Proof

- G1 Numerics. For 64 references and 64 outputs of the corpus, codes and
  waveforms at the natural length against the ladder padded length, on
  the box. The reference band is the variation the current pad to longest
  already produces: the same reference encoded alone and in a batch with
  a longer one. Pass: padded equals natural bit for bit, or differs
  within that band on the same rows.
- G2 Padded compute. Encode and decode time at natural against ladder
  length, p50 over the corpus lengths, for candidate steps of 4, 8 and 16
  frames. Pass: the step whose p50 cost is under 1 ms on both sides.
- G3 The A/B of doc 15 on the full corpus, c1 and c16, arms alternated,
  `--seed 1234`. Pass: the first seen residual of section 2 goes to zero,
  c1 mean latency down by about 2%, the first 50 requests down by about
  15%, steady state within the 2 ms band of doc 24 section 8, c16 tail
  latency not worse, WER and similarity within the band of doc 24 section
  3.2.
- G4 Startup. The two warmups together under 3 s on H100, logged.

## 9. Before the slices are sized

- M1 Attribute the non attention share of the per new shape cost. One
  c1 window of first seen requests under the torch profiler with host
  ranges on the two threads (doc 15 runbook), split into cuDNN attention
  plan builds, cuDNN convolution plan builds, allocator growth and the
  rest. If the conv share is small, C3 for the encoder becomes a real
  alternative for that side.
- M2 Count the realised keys per thread in one c16 run with
  `TORCH_CUDNN_SDPA_CACHE_DEBUG=1` (X2 prints counts and hits), to size
  the ladder and the batch axis.
- V1 Read the checkpoint's `speech_tokenizer/config.json` for the Mimi
  fields (heads, head dim, sliding window, upsampling ratios) and the v2
  decoder fields. R8 and R10 are library defaults until then.
- V2 Confirm the decoder's treatment of pad value -1 rows before the
  quantiser (R9 relies on it today for batches, so it is safe, but the
  exact clamp should be cited).

## 10. Slices

- S0 M1, M2, V1, V2 above. Box time, no code.
- S1 Encoder ladder, trim and warmup in `request_builders.py`, with a
  unit test that pins the trim to the true frame count for lengths on and
  off the ladder and a startup test that the warmup runs on the batcher
  thread. G1 and G2 on the encoder side.
- S2 Vocoder ladder, trim and warmup in `streaming_vocoder.py`, same
  tests, G1 and G2 on the vocoder side.
- S3 G3 and G4, then the PR.

Open questions the code could not answer: the checkpoint configs (V1),
the realised batch composition at c16 (M2), where the other half of the
per new shape cost goes (M1).
