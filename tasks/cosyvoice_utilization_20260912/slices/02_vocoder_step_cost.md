# Slice 02: vocoder step cost (batched HiFT, no in-call syncs, streaming Flow graphs)

Base: branch `perf/cosyvoice3-stream-scheduler-liveness` at c0767cd47 (upstream main 51e2f7ec2 plus
slice 01 and the first-hop priority). Owners: `sglang_omni/models/fun_cosyvoice3/stages.py`,
`sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py`, `sglang_omni/models/fun_cosyvoice3/config.py`.
Reference implementation read for numerics: CosyVoice `cosyvoice/cli/model.py` (CosyVoice3Model,
lines 397-450), `cosyvoice/hifigan/generator.py` (causal HiFT, lines 596-726),
`cosyvoice/flow/DiT/dit.py` (lines 76-166), `cosyvoice/utils/mask.py` (lines 127-238),
`cosyvoice/flow/flow_matching.py` (lines 71-125, 196-227), `cosyvoice/hifigan/f0_predictor.py`.

## 0. Trigger

The slice 01 A/B on the H100 (English corpus, 1088 requests, c16 streaming): B fixes WER (6.8 to
1.6 percent), failures (23 to 0), the tail (p99 43.6 to 14.5 s) and throughput (7.4 to 9.5 audio
s/s), and loses first audio (0.9 to 3.35 s) and continuity (C50 80 to 9). The vocoder is
overloaded: capacity 9.5 audio seconds per second against 16 real-time streams. c1 has 100 percent
continuity on both arms because at RTF 0.18 every hop finishes long before the previous chunk has
played; c16 loses it because demand exceeds capacity and every policy then only decides who waits.
The first-hop priority commit restores first audio; continuity at c16 returns only when a step
stops costing one to two seconds. This slice cuts the step cost with three changes ordered by the
measured cost they remove.

## 1. Measured step cost at c16 (perfkit results, sections 4 to 6, and the B serving log)

A batched step for n participants at hop h is one packed causal Flow call plus n serial HiFT calls
plus the scheduler glue. Measured on the H100:

| item | host ms | GPU ms | notes |
|---|---:|---:|---|
| packed causal Flow, batch 14, 400 frames | 491 | 408 | GPU bound; 170 ms of the host time is 71 `cudaStreamSynchronize` calls |
| packed causal Flow, batch 2 | 692 to 757 | 75 to 80 | launch bound, 18.2k launches, host cost per launch doubles while the AR thread runs |
| native Flow, batch 1 (singleton hop, every final) | 273 to 350 | 64 to 89 | launch bound, 18.1k launches |
| HiFT, batch 1, 0 to 400 frames | 27 to 72 | 5 to 9 | 1150 launches and about 85 syncs per call |
| HiFT, batch 12, 400 frames | 148 | 83 | one call |

B's serving log: a 13 to 16 row step at hop 25 is followed by the next batched step after 1.0 s
p50, at hop 50 after 1.84 s p50. The 14 serial HiFT calls in a 14-row step are about 380 ms; the
Flow call about 490 ms; the rest is singleton hops and finals that run between batched steps
(1088 finals per corpus, each a native Flow call plus a full-history HiFT call).

Per-row structure of the cost: HiFT is called once per participant on that participant's whole
mel history (`stages.py:1372-1390`, `hift_delta` concatenates `hift_mel` and runs `hift.inference`
on all of it). The reference does the same (`model.py:425-450` keeps the whole mel in
`hift_cache_dict[uuid]['mel']` and slices the speech by `speech_offset`), so this is faithful, not
a defect; it is simply serial.

```
FunCosyVoice3StreamingVocoderScheduler.run_step (streaming_vocoder.py:331)
  _run_causal_hop_batch (:364)
    items = [FlowBatchInput(token=tensor(state.tokens[:end]).unsqueeze(0), prompt..)]   CPU tensors
    self._vocoder.first_hop_batch(items)                     stages.py:1357
      FunCosyVoice3Flow.inference_causal                      stages.py:724
        pack_flow_inputs   .to(device) per row               stages.py:165-185  (4 syncs per row)
        generate_flow(streaming=True, finalize=False)         stages.py:523
          rand_noise prefix .to(device)                       stages.py:616     (sync per call)
          solve_flow_euler(streaming=True)  eager, 10 steps   stages.py:244     (18k launches)
        split_generated_mels                                  stages.py:661
    for each participant:                                    streaming_vocoder.py:386-398
      self._vocoder.hift_delta(mel[:, :, offset:], hift_mel, speech_offset, finalize=False)
        torch.cat([hift_mel, tts_mel]); hift.inference(full history)   stages.py:1380-1383
          f0_predictor.to(float64); stft window .to(device); source prefixes .to(device)   generator.py:713-726, 491-505
```

## 2. Change 1: one HiFT call per step (V4a)

Mechanism. `CosyVoice3Vocoder.hift_delta_batch(new_mels, hift_mels, speech_offsets, *, finalize=False)`:
concatenate each row's history and new frames, right-pad rows to the longest row, run
`hift.inference` once on the (n, 80, T_max) batch, and slice each row's emitted samples by that
row's own valid length. `_run_causal_hop_batch` calls it once for all participants and writes back
per-row `hift_mel` and `speech_offset`. `hift_delta` stays for the singleton and final paths.

Why it is exact per row. The causal HiFT (`generator.py:596-726`) is built from causal
convolutions (`CausalConv1d` left type, `CausalConv1dUpsample`, causal `ResBlock`), one right-context
convolution at the input (`conv_pre`, `conv_pre_look_right` frames) and a causal f0 predictor
(`f0_predictor.py:62-99`, one right-context layer). The sine source integrates phase along time
per row (`generator.py` SineGen2, `cumsum` over dim 1). So a row's output at frame t depends on
that row's frames up to t plus the right context, never on other rows and never on frames beyond
the right context. For `finalize=False` the single-row call trims the right context and one frame
of samples at the end (`generator.py:709`, `prod(upsample_rates) * hop_len` = 480 samples), so
the emitted samples of a row end one frame before its right context starts. In the padded batch a
shorter row's padded frames lie beyond that boundary; the only cross-boundary effect is the
ISTFT overlap-add, and an ISTFT frame starting at sample p only writes samples at or after p, so
samples before the row's boundary receive nothing from padded frames. Per row the valid sample
count is `(frames_i - condnet_causal_padding - conv_pre_look_right) * 480 - 480`, computed from the
module's own attributes, never from constants in our code.

Experiment E1 (bit exactness, box): for 32 corpus requests, run the current per-row path and the
batched path at rows 2, 8 and 16 with the same histories; assert `torch.equal` on the emitted
samples and on the returned history. If not equal, report the max absolute difference per row
and the WER gate decides. Kernel selection can differ between batch sizes (cuDNN), which is the
same class of difference as CUDA graphs and the packed Flow already accept.

Also in this change: make the HiFT constants device resident once at load (the STFT window used by
`_stft` and `_istft`, the sine and noise prefixes of the source module, the float64 f0 predictor),
so every `.to(device)` in the reference code becomes a no-op. This is state initialization on the
omni side, not a patch of the package. Experiment E2: perfkit sync count per HiFT call before and
after (today about 85).

Expected: a 14-row step drops the 14 serial calls (about 380 ms) to one call (about 150 ms) and
loses most of the 14 x 85 syncs.

## 3. Change 2: no in-call syncs in the packed Flow call (V1)

Mechanism. The per-row `.to(device)` copies in `pack_flow_inputs` (`stages.py:165-185`) move the
token, prompt token, prompt feature and embedding tensors on every hop. The prompt tensors are
latched once per request (`streaming_vocoder.py:204-233`); latch them on the vocoder device and
keep them there in `_CosyVoice3StreamState`, so a hop moves nothing but the token window. Build the
token window tensor on the device once per step. Make the noise prefix (`stages.py:616`) resident
by moving `decoder.rand_noise` to the device at load. Build the length tensors of `token2wav_chunk`
(`stages.py:1330-1338`) on the device.

Measured: 4 syncs per row plus 15 per call; 170 ms of a 491 ms batch-14 call. Experiment E3:
perfkit sync count per packed call before and after; target is the per-call count only.

## 4. Change 3: streaming Flow graphs captured from the workload (V3)

Why the current runner does not apply. `FlowCudaGraphRunner` (`stages.py:344-520`) captures
`solve_flow_euler` without the streaming mask and is used only when `not streaming and finalize`
(`stages.py:630`). Its capture set is a table in `config.py:19-40` (batch 1: 304 to 640 frames in
16-frame steps, batch 2: 384 to 544). Any shape outside the table runs eager, so the table encodes
one benchmark's length distribution. The table is deleted by this change.

Mechanism.

- Key: `(streaming, rows, frame_bucket)`. Capture on first use, on the vocoder thread, with
  `capture_error_mode="thread_local"` as today (the AR thread issues CUDA work concurrently).
  One eager warm solve then one capture per new key; the captured result is used for that step.
- Rows: buckets are powers of two up to `max_batch_size`; a batch of n rows runs as the binary
  decomposition of n (for example 13 = 8 + 4 + 1), each part a replay. No dummy rows, so no wasted
  GPU work; a replay is one launch, so the extra launches are free against 18k eager launches.
- Frames: rows in one packed call share `T_max`; the bucket is the smallest value at or above
  `T_max` on a geometric ladder whose step is the existing `flow_merge_pad_budget_percent` knob
  (25 percent today, already the accepted padding budget of the non-streaming merge). The ladder is
  anchored at the DiT chunk size (50 frames, `dit.py:119`) so buckets stay chunk aligned. This
  bounds padding waste by the budget and the number of buckets by the logarithm of the frame range
  (`decoder.rand_noise` is the ceiling, 15000 frames), with no table.
- The non-streaming finalize path uses the same runner and the same policy; the current 16-frame
  granule and its constant go away.
- Memory: static inputs and output per key are small; the graph pool is shared across keys as
  today. The count is bounded by row buckets times frame buckets. Log every capture with its key
  and duration so the census sees the set the corpus produced.

Why right padding is exact under the streaming mask. `DiT.forward` builds the attention mask as the
key padding mask ANDed with the 50-frame chunk mask (`dit.py:163-166`, `mask.py:161-238` with
`static_chunk_size`), so padded frames are never keys for real frames. The positional convolution is
causal (`dit.py:82`, `CausalConvPositionEmbedding`), so padded frames never feed real frames. Rows
are independent everywhere. Padded query rows with an all-false mask are forced to attend to
themselves (`mask.py:236` and the omni `_chunk_mask` patch, `stages.py:1872-1893`) and are sliced
away by `split_generated_mels`. Experiment E4: graph versus eager bit exactness for streaming
windows at each hop of the ladder and rows 1, 2, 8, 16. Experiment E5: capture time per key and
the number of keys after the full corpus at c1 and c16.

Expected: the singleton native hop and every final drop from 273 to 350 ms host to about the GPU
time (64 to 89 ms); batch 2 calls from about 700 ms to about 80 ms; batch 14 calls lose the launch
share (about 80 ms) on top of the sync share removed by change 2. At c1 this also takes about
0.25 s off first audio (the first hop is a singleton native call).

## 5. Order, gates and measurement

Order: change 1, then change 2, then change 3; each is its own commit series and its own A/B with
the previous B as A (protocol memory: stacked slices). Only B c16 streaming is rerun between
changes; the full matrix (c1 and c16, streaming and non-streaming, WER on every cell) runs once at
the end of the slice.

Gates per change: WER within 0.2 points of A per cell; c16 streaming TTFP, inter-chunk p99 and C50
not worse than A; req/s and audio s/s not worse; zero failures; the exactness experiment of the
change passed on the box before the A/B boot.

Perfkit on one 16-request c16 capture after the slice: step host time by rows and hop, syncs per
Flow and HiFT call, SM Active while Flow and HiFT kernels are present (today 15 to 25 percent for
native batch 1 Flow, 51 percent for HiFT batch 10 to 12).

## 6. Out of this slice, recorded as the next levers

- Incremental DiT with a per-Euler-step key and value cache (V3b). Under the 50-frame chunk mask an
  earlier chunk never attends a later one, so its trajectory through the 10 Euler steps is the
  same on every hop and could be cached instead of recomputed; a hop would then cost its own
  frames, not the whole prefix. Exact in principle; a custom forward over the DiT blocks.
- Windowed HiFT with carried source phase (V4b). The reference recomputes the whole history; a
  window needs the sine source's cumulative phase carried across calls and a left context covering
  the causal receptive field. Exact only with the phase state; measured against the full-history
  output before adoption.
- Admission at the vocoder (how many streams to start when demand exceeds capacity) is a product
  decision that the numbers after this slice will inform.
