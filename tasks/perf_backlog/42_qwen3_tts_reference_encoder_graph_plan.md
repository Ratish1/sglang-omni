# 42. Plan: replay the reference encoder through captured graphs

## What we noticed

A streaming request with a reference clip encodes it on the `qwen3-tts-ref-code`
thread before the talker can start. On the H100 the encode is 10 ms alone and 38 to
53 ms inside the server at c16, because it is about 1,000 eager launches and every
launch releases and reacquires the interpreter lock among 18 threads. Three things in
the encoder make it that many:

- `MimiConv1d` keeps `stride`, `kernel_size` and `padding_total` as device buffers, so
  the padding arithmetic before each of the 15 conv layers is six tiny kernels and
  `F.pad` reads the result back: one device to host sync per layer.
- `MimiModel.encode` runs all 32 quantizer stages; the tokenizer keeps the first 16.
- Nothing is captured. The codec decoder next to it replays graphs; the encoder does
  not.

Measured on 64 references: moving the buffers to the host and asking for 16 stages
changes no code and takes the eager encode from 9.9 to 6.7 ms; a captured graph at a
bucket length replays in 2.4 to 3.7 ms of device time and 50 us of host time, 32 MiB
per key. Codes at a padded length are not bit identical to the unpadded encode (the
last frame's conv windows see the padded tail, and bfloat16 rounding flips near tie
codebook picks), which today's batching already does when it pads to the longest
member; the quality census is the gate, not bit identity.

## How the code does it today

```
request_builders.py
  _Qwen3TTSAdhocReferenceHook.encode_one            (pool thread)
    _ref_code_batcher.submit(waveform, sr) ----------------------------.
    extract_speaker_embedding(...)                                      |
    ref_code_future.result()  <-------------------------------------.  |
                                                                    |  |
  _Qwen3TTSRefCodeBatcher._run                        (ref-code thread)|
    _drain(): up to 8 waveforms within 2 ms  <--------------------------'
    with torch.inference_mode(), torch.cuda.stream(encode_stream):  |
      speech_tokenizer.encode(waveforms, sr)                        |
        qwen_tts wrapper: feature_extractor pads to the longest     |
        Qwen3TTSTokenizerV2Model.encode                             |
          MimiModel.encode(values)            15 convs, 8 layers,   |
                                              32 RVQ stages         |
          codes[:, :16], sliced per waveform to ceil(n / 1920)      |
    _synchronize_outcomes(): event on encode_stream, wait           |
    future.set_result(codes) -------------------------------------'

stages.py
  _load_qwen3_tts_tokenizer(): Qwen3TTSTokenizer.from_pretrained(..., dtype)
  set_qwen3_tts_preprocessing_context(): builds the service and the batcher
```

The batcher owns one thread and one CUDA stream, resolves futures after an event on
that stream, and the hook records the consumer stream on the returned codes. All of
that stays.

## The change

1. `stages.py`, `_load_qwen3_tts_tokenizer`: after `from_pretrained`, for every
   `MimiConv1d` under `tokenizer.model.encoder`, replace `stride`, `kernel_size`,
   `padding_total` with their CPU copies and recompute `padding_left` and
   `padding_right` from them. The padding arithmetic runs on host integers, the sync
   per layer is gone, and the encoder can be captured.

2. New module `sglang_omni/models/qwen3_tts/reference_encoder_cuda_graph.py`,
   `Qwen3TTSReferenceEncoderGraphRunner(encoder, *, device, dtype, hop, bucket_frames,
   stream)`:
   - `capture()`: for each bucket, a static input `(1, 1, bucket * hop)` in the
     encoder's dtype, two warmup calls of
     `encoder.encode(static_in, num_quantizers=16, return_dict=True)` on the stream,
     then the capture on the same stream into one shared pool. Keys are captured
     largest first so the pool is sized once. A capture failure disables the runner
     with the reason, as the codec runner does.
   - `bucket_for(frames)`: the smallest captured bucket at or above `frames`, or None.
   - `encode(waveform)`: zero the static input, copy the waveform in, replay, return
     `static_out[0, :, :frames].clone()` where `frames = ceil(samples / hop)`. The
     clone is what the next replay must not overwrite.
   - `stats()`: captured keys, footprint, replays, misses, disable reason, for the
     codec state log line.

3. `request_builders.py`, `_Qwen3TTSRefCodeBatcher`:
   - takes the runner (or None) at construction; the hook builds it from
     `model.speech_tokenizer.model.encoder` on the batcher's encode stream and
     captures it inside `set_qwen3_tts_preprocessing_context`, so the graphs exist
     before the first request and before the engine sizes its KV pool.
   - `_run`: per waveform, `bucket_for(frames)`; if captured, `runner.encode`;
     otherwise the eager path, `encoder.encode(values, num_quantizers=16)` on the
     waveform zero padded to the next multiple of `hop`, sliced to `frames`. Both
     paths go through the same `_encode_waveform(waveform)` so the frame count and
     the padding rule are defined once. The wrapper's `encode` is no longer called;
     its normalization already happened in `encode_one`, and the batcher only ever
     receives waveforms at the tokenizer's sample rate (the sample rate groups stay
     as the guard they are today).
   - `_synchronize_outcomes` and the future resolution stay as they are.

4. Knob: `reference_encoder_cuda_graph_bucket_frames`, default
   `(32, 48, 64, 96, 128, 192, 256)` (2.6 to 20.5 s of reference audio, 7 keys, about
   224 MiB); an empty tuple disables the runner and keeps the eager path of step 3.
   Forwarded by the factory in `stages.py` like the window frames knob.

5. Tests, `tests/unit_test/qwen3_tts/`:
   - the loader leaves every conv's three buffers on the CPU and the encoder's
     parameters where they were;
   - `bucket_for` picks the smallest fitting key and None above the largest;
   - the batcher's eager path pads to the hop multiple and returns `ceil(n / hop)`
     frames per waveform, on a fake encoder that records the shapes it saw;
   - a miss (frames above the largest bucket) takes the eager path and the runner
     counts it;
   - CUDA: the runner's replay equals the eager encode of the same padded length,
     bit for bit, for two references at different buckets, and a second replay of
     the same key leaves the first result untouched.

6. Gate: the doc 33 protocol, main against the branch, default launch, streaming c16
   and seeded c1, full corpus. Reads: req/s, TTFC, the preprocessing segment p50 and
   p95 from the anatomy, the runner's replay and miss counts from the log line, WER
   and speaker similarity inside the bands. Then the same pair with `early_ids.patch`
   on both arms, which is the #2123 gate.

## After

```
  _Qwen3TTSRefCodeBatcher._run                        (ref-code thread)
    for waveform in batch:
      frames = ceil(n / hop); bucket = runner.bucket_for(frames)
      captured: static_in <- waveform, replay, clone codes[:, :frames]   ~50 us host
      miss:     encoder.encode(pad to hop multiple, num_quantizers=16)  eager, 16 stages
    _synchronize_outcomes(), futures resolved as today
```

One thread, one stream, one launch per reference instead of a thousand, and the same
codes for a reference on every boot.
