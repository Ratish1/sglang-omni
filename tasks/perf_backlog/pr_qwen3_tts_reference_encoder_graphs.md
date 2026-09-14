### Mechanism

A streaming request with a reference clip encodes the clip before the talker can start. The encode is the transformers Mimi encoder inside the speech tokenizer, run eagerly on the `qwen3-tts-ref-code` thread: about 1,000 kernel launches, each a release and reacquire of the interpreter lock among the process's 18 threads, plus one device to host sync per conv layer, because `MimiConv1d` keeps its padding integers as device buffers and `F.pad` reads them back. It also runs all 32 quantizer stages when the tokenizer keeps 16. Alone the encode takes 10 ms; inside the server at c16 it takes 38 to 53 ms, and under early ids it was the whole of the remaining first chunk regression.

This PR moves the three padding integers of every conv to the host at load (the arithmetic becomes Python integers, the syncs disappear, the encoder becomes capturable), calls the encoder with the 16 quantizers the tokenizer keeps, and captures one graph per reference length bucket (32 to 256 frames, 2.6 to 20.5 s of audio, 32 MiB each) on the batcher's stream. A reference is zero padded to its bucket, replayed, and its codes sliced to `ceil(samples / hop)` frames; longer references take the same padded eager path. Codes at a padded length are not bit identical to the unpadded encode (the last frame's conv windows see the tail, and bfloat16 rounding flips near tie codebook picks), which the wrapper's own batching already does when it pads a batch to its longest member; the quality census is the gate and it holds.

### Changes

- `reference_encoder_cuda_graph.py`: `move_conv_padding_to_host`, `smallest_bucket`, `Qwen3TTSReferenceEncoderCudaGraphRunner` (capture per bucket into one pool, replay, stats).
- `request_builders.py`: the batcher resolves the encoder, hop and quantizer count from the tokenizer, takes its stream from the encoder's device, captures the runner before its thread starts, and encodes each waveform through one path: runner if the length fits, else eager on the waveform padded to whole frames with 16 quantizers. The knob travels from `set_qwen3_tts_preprocessing_context` to the hook to the batcher.
- `stages.py`: the loader moves the padding buffers after `from_pretrained`; `create_sglang_tts_engine_executor` takes `reference_encoder_cuda_graph_bucket_frames`. `engine_builder.py` carries it into the context, before the KV pool is sized.
- Tests: the conv move, the loader, bucket choice, the batcher's padding, and two CUDA tests on a small real Mimi model (replays equal eager bit for bit, a repeat does not disturb an earlier result, a miss takes the eager path). The pipeline fakes model the encoder's real shape.

Non streaming is untouched: the reference encode is the streaming prompt path only.

### Census, default launch, streaming c16, full seed-tts corpus, H100, one boot per arm

| read | main 69ddc6baa | this PR | delta |
| --- | ---: | ---: | ---: |
| req/s | 15.88 | 15.89 | 0 |
| audio s/s | 66.05 | 66.07 | 0 |
| RTF mean | 0.2434 | 0.2427 | |
| TTFC mean ms | 116.5 | 99.5 | -17.0 |
| TTFC p50 ms | 107.6 | 94.3 | -13.3 |
| TTFC p95 ms | 198.1 | 141.8 | -56.3 |
| TTFC p99 ms | 279.2 | 284.4 | +5.2 |
| inter chunk mean ms | 109.8 | 111.8 | +2.0 |
| preprocessing segment p50 ms | 35.4 | 23.5 | -11.9 |
| preprocessing segment p95 ms | 106.9 | 46.8 | -60.1 |
| admission to first audio p50 ms | 106.3 | 92.8 | -13.5 |
| seeded c1 TTFC mean ms | 55.6 | 54.9 | -0.7 |

Quality on this PR: WER 1.05 percent, speaker similarity 71.35, inside the bands. Throughput is flat because closed loop c16 is bound by the talker; the first chunk moves.

Graph footprint 7 keys x 32 MiB. Knob to disable: `reference_encoder_cuda_graph_bucket_frames=[]`.
