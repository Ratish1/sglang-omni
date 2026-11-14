# Qwen3-TTS synthetic profile of main `13364eb45` (11-14, host-62, H100, card 0)

Tree hash `488dc114` on the box and the Mac. Kit: this folder on `prof/qwen3-tts-synthetic-20261114`. Raw reports: `/data/qwen3-tts-synthetic-20261114/main13/` on the box.

## Method

The method follows SGLang's `benchmark/one_batch.py` (random ids, a static batch, the prefill and the decode profiled separately) and InferenceX's fixed-sequence benchmark (random sequences so nothing hits a cache, fixed input and output lengths, EOS ignored).

- Prefill regime: each request clones a 20.48 s reference (256 codec frames, the largest captured reference encoder bucket) in ICL mode, so the talker prefill holds about 259 rows; `max_new_tokens` 1.
- Decode regime: each request clones a 3 s reference by x-vector and speaks a one-token text; the codec EOS is masked, so `max_new_tokens` 2048 sets 2048 frames.
- References are fresh noise, one per request: no reference cache hits.
- Static batches of 1, 16 and 64. A prefill cell is three rounds, and a decode cell's window opens once every stream has 256 frames and lasts 3 s.
- One server boot per collection mode, one report per cell (`nsys launch`, then `nsys start/stop`).

| Mode | Collection | Used for |
|---|---|---|
| none | no profiler | baseline numbers |
| light | CUDA, NVTX, function marks, graph-level graph trace | the prefill time slice |
| timing | light plus OS runtime, CPU context switches, GPU metrics at 10 kHz, CUDA event trace | the decode time slice, hardware counters |
| attribution | node-level graph trace with NVTX precapture, autograd NVTX, GIL trace, capture marks | kernels inside graphs, by graph and launch site |

Marks: `marks.json` (53 functions across preprocessing, talker and vocoder) is passed to `nsys --python-functions-trace`, with no source change, and `check_marks.py` reports 0 missing. The probe labels each talker step `step[MODE bs toks]` and each vocoder batch with its stream count.

Every transfer is listed by copy kind and memory kind (pageable, pinned, device), and every host wait by CUDA API. Each is attributed to the innermost marked function on its thread (`kernel_digest.py`).

## Profiler overhead

| | bs 1 | bs 16 | bs 64 |
|---|---|---|---|
| none: prefill first audio | 31.7 ms | 149.9 ms | 389.8 ms |
| light, captured: prefill first audio | 33.2 ms | 146.2 ms | 377.7 ms |
| timing, captured: prefill first audio | 40.8 ms | 166.1 ms | 432.1 ms |
| none: decode ms per frame | 4.458 | 5.699 | 10.072 |
| timing, captured: decode ms per frame | 4.481 | 5.785 | 10.247 |

So prefill is read from light captures and decode from timing captures. Timing collection inflates the CPU-heavy preprocessing by about 9 ms a request at batch 1.

## Decode (timing captures, steady window)

| | bs 1 | bs 16 | bs 64 |
|---|---|---|---|
| step period (window / steps) | 4.48 ms | 5.61 ms | 9.28 ms |
| talker decode graph span | 1.99 ms | 2.78 ms | 3.86 ms |
| code predictor graph span (15 codebooks) | 2.41 ms | 2.51 ms | 4.22 ms |
| device busy, any stream | 98.3 % | 98.0 % | 96.6 % |
| SM issue / SMs active | 4.8 / 54 % | 8.3 / 67 % | 13.9 / 75 % |
| DRAM read / tensor active | 36 / 2.4 % | 37 / 5.7 % | 35 / 13.2 % |
| host waits a step | 1 event sync | 1 | 1 |

- **Syncs.** The step's only host wait is `ModelRunner.finalize`'s event sync (2.28 ms at bs 1, 3.47 ms at bs 64). It wakes on the layer 0 sample, while the predictor graph is still running.
- **Copies.** Every decode copy is pinned or device to device.
- **Host on the critical path.** At bs 1 the host hides under the predictor. At bs 64 the host's per-step Python takes about 4.8 ms and outlasts the 4.2 ms predictor, so the talker stream waits about 1.2 ms a step. The largest pieces are `write_feedback_buffers` (1.3 ms mean, 3.8 ms p90) and `get_next_batch_to_run` (1.2 ms), plus 64 per-request messages to the vocoder each step.

Kernels inside the graphs (attribution, node level):

| Graph | Kernels a replay | bs 1 kernel time | bs 64 kernel time | Main kernels |
|---|---|---|---|---|
| talker decode | 340 | 1.81 ms | 3.52 ms | bs 1: GEMMs 68 %; bs 64: attention (`device_kernel`, 56 a replay, 28.6 us) 45 % |
| code predictor | 443 to 469 | 2.75 ms (overlapping, span 2.40) | 4.74 ms (span 4.22) | 75 layer steps of `norm_qkv_rope_store`, `decode_attention`, 2 x `gemv_add`, `norm_gate_up_silu`, plus 15 samplers |
| vocoder follow-up (2 workers) | 205 to 222 | 1.25 ms | 2.5 ms | snake activation 21 %, convolutions |

Floors from the checkpoint config, at 3.35 TB/s:
- **Talker:** 1.41 B parameters, 2.83 GB, 0.85 ms a step at bs 1. Measured 1.81 ms, 47 %.
- **Predictor:** 5 layers of 1,024 hidden, 161 MB a codebook step, 0.72 ms a frame at bs 1. Measured 2.40 ms, 30 %. At bs 64 the floor is about 0.75 ms against 4.22 ms measured, 18 %. One predictor layer takes 58 us there against a 9.4 us weight read.

## Prefill and first audio (light captures)

| | bs 1 | bs 16 | bs 64 |
|---|---|---|---|
| preprocessing a request | 18.6 ms | 61.9 ms | 68.3 ms |
| of it, speaker mel on the CPU | 12.0 ms | not split | 18.1 ms |
| of it, speaker embedding in all | 12.5 ms | 26.4 ms | 25.7 ms |
| reference encode graph, fp32, one clip at a time on one batcher thread | 4.6 ms | 6.0 ms | 6.7 ms span |
| talker prefill steps | 4.6 ms, 259 rows | 11 to 30 ms, 518 to 1,813 rows | 16 to 33 ms, up to about 2,000 rows |
| vocoder first chunk batch | 2.7 ms | 7 to 25 ms | 18 to 30 ms |

- **Batch 1, about 33 ms.** First audio is preprocessing 18.6 ms, then the talker prefill graph 2.7 ms, the predictor 2.4 ms and the vocoder first chunk 2.7 ms. The vocoder's first chunk decodes the reference codes as left context.
- **Batch 64.**
  - *Reference encode.* The batcher encodes 64 clips of 20 s one by one at about 6.7 ms each, about 430 ms in series, and every request waits behind it. The encoder runs in fp32 kernels (`sm80_xmma_gemm_f32f32`, `fmha_cutlassF_f32`, `implicit_convolve_sgemm`).
  - *Preprocessing waits.* The reference encoder's stream sync takes 4.0 ms mean a request. The speaker mel's pageable copy waits 1.2 ms mean and up to 16.7 ms. The prompt build makes 6 stream syncs and 6 pageable copies a request, and the waveform copy is 1.97 MB pageable.

## Ranked by measured size (main, this profile)

Decode step:
1. **Code predictor graph:** 2.41 ms of 4.48 at bs 1 and 4.22 of 9.28 at bs 64. It runs at 18 to 30 % of its weight-read floor, as 75 small layer steps a frame.
2. **Talker decode graph:** 1.99 ms at bs 1, at 47 % of floor. At bs 64 it is 3.86 ms, with attention 45 %.
3. **Host per-step Python at bs 64:** about 1.2 ms a step exposed on the talker stream.
4. **Vocoder follow-up graphs at bs 64:** about 1.1 s of 3.1 s overlap the talker and share its SMs.

First audio:
1. **Reference encode:** fp32, serial, one clip at a time.
2. **Speaker mel on the CPU:** 12 to 18 ms a request.
3. **Pageable copies and stream syncs** on the shared preprocessing stream.
4. **Talker prefill and vocoder first chunk** that include the reference context.

Nsight Compute per-kernel results: `ncu2_*` (see below).
