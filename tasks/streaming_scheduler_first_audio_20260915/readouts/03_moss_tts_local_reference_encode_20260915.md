# Readout 03: MOSS-TTS Local reference encode, where the time goes (2026-09-15)

Archive `artifacts/ref-encode-profile-20260915T090649Z.tar.gz`, runbook 03. Tree `b29084bfa`,
import path `/sgl-workspace/wt/stream-main/sglang_omni/__init__.py`, H100 80GB HBM3 on GPU 0 empty
at start, 2 x Xeon Platinum 8462Y+ (128 logical CPUs, no cgroup quota, effective_cpu_count 128),
torchaudio 2.11.0+cu130, py-spy 0.4.2. References: 666 unique of 1,088 samples, all 24 kHz MP3
(libavcodec mpegaudio frames in the stacks), resampled to 48 kHz, 4.6 s mean.

## Verdict

- Bursts: the single encode worker decodes and resamples every file of a batch one after another
  before the GPU encode. At a burst of 16 that is 67 ms of file work against 44 ms of encode per
  batch of 8, and the second batch of 8 waits for both. This is the line that pays in bursts, and
  the fix is small (below).
- Steady state: with 1 to 2 requests preprocessing at once, preprocessing is about 100 ms p50 for
  a 30 ms offline encode. The worker's live time is in the encoder's eager host launch path (packed
  RoPE Triton launches, FA3 varlen calls, layer norms) competing for the interpreter with the
  tts_engine scheduler. That is the batch 1 graph replay shape of the Qwen3-TTS preprocessing
  change, not a small fix.
- Decision: parked. Neither item is a #2169 regression; CosyVoice work takes priority.

## 1. Offline, GPU 0 alone (`encode_timing.txt`)

One reference at a time, ms:

| step | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| torchaudio.info x2 | 0.17 | 0.16 | 0.21 | 0.27 |
| path cache key (first read) | 0.19 | 0.18 | 0.24 | 0.27 |
| torchaudio.load | 6.35 | 6.24 | 7.13 | 9.53 |
| resample 24 to 48 kHz | 1.82 | 1.57 | 3.14 | 4.58 |
| prepare waveform + H2D | 0.44 | 0.43 | 0.56 | 0.70 |
| encoder forward | 29.68 | 27.72 | 41.49 | 52.68 |
| codes D2H | 0.04 | 0.04 | 0.05 | 0.06 |
| total | 38.69 | 36.42 | 51.65 | 63.87 |

Batches as the worker runs them, mean ms:

| batch | load_paths | encode_waveforms | padded / real samples |
|---|---|---|---|
| 1 | 7.63 | 26.48 | 1.00 |
| 2 | 16.32 | 33.16 | 1.12 |
| 4 | 30.90 | 34.83 | 1.24 |
| 8 | 61.70 | 46.73 | 1.35 |

The GPU encode batches well (8 references in 47 ms against 26 ms for one); the file work is linear
in the batch because it is sequential.

Bursts of 16 through `_MossLocalReferenceEncoder` and `_BatchedReferenceEncoder`, 16 rounds: every
worker batch is 8; worker `load_paths` 67.0 ms mean, `encode_waveforms` 44.2 ms mean; request
latency by completion rank 1 and 8: 118 and 120 ms mean, rank 9 and 16: 227 and 229 ms mean.

## 2. The code that pays, at `b29084bfa`

```text
preprocessing thread (16)  _MossLocalReferenceEncoder.encode       stages.py:508
                           _BatchedReferenceEncoder.encode          stages.py:321-327  queue.put, future.result
                                    |
worker moss-local-ref-encode        _drain_batch                    stages.py:342      17.4% wall (waiting for jobs)
                                    _encode_batch
                                      load_paths (file + resample)  stages.py:391      15.1% wall  <- 67 ms of 111 ms per burst batch
                                      encode_waveforms              stages.py:393      65.8% wall
                                        batch_encode stage loop     audio_tokenizer.py:1511
                                          transformer layers        audio_tokenizer.py:605, 449, 308
                                            packed RoPE (Triton)    attention.py:1445, 546; vocoder_kernels.py:138   29.9% of worker GIL
                                            FA3 varlen              attention.py:787                                 13.7% of worker GIL
```

Wall shares are of the worker's 2,497 `--idle` samples; GIL shares are of its 314 GIL samples.

## 3. Live run: the capture distorted the benchmark

The benchmark read 6.70 req/s, first audio mean 0.564 s, C50 72.06, against A's 12.95 req/s and
0.260 s. py-spy `record` suspends the process for every sample unless `--nonblocking` is given,
here at 100 Hz with native unwinding. Requests by submit time, ms:

| window | n | first audio p50 / p95 | preprocessing p50 / p95 | after preprocessing p50 / p95 |
|---|---|---|---|---|
| during the captures (12 to 64 s) | 203 | 1,196 / 1,936 | 756 / 1,294 | 428 / 717 |
| everything else | 885 | 278 / 1,510 | 117 / 995 | 158 / 498 |
| 100 to 150 s, after recovery | 569 | 229 to 283 p50 | 90 to 126 p50, 144 to 214 p95 | 141 to 157 p50 |

The degradation lasts until about 95 s, past the nominal 50 s of captures. From 100 s on the run
matches readout 02 (preprocessing 102.8 / 204.7 ms p50 / p95). The benchmark numbers of this run
are therefore not a measurement of the tree, and the live py-spy shares are shares of a slowed
process: they locate the lines, they do not size them.

GIL ownership over the 25 s capture: 896 samples of about 2,500, so the interpreter was held about
36 percent of the time; tts_engine scheduler 399, encode worker 314, main thread 151. py-spy
ownership samples do not give lock wait time.

## 4. The small fix, and what it would and would not move

`_BatchedReferenceEncoder.encode` can load and resample in the calling preprocessing thread and
queue a waveform job through the existing `encode_wav` path (stages.py:329-332); the worker then
runs only `encode_waveforms`. The waveforms entering `batch_encode` are the same tensors, so codes
are unchanged for a given batch composition.

Derived from section 1, not measured: if the 16 loads run in parallel, a burst's first 8 complete
after about 8 + 44 ms instead of 118 ms and the second 8 after about 96 ms instead of 227 ms.
Whether `torchaudio.load` releases the interpreter during MP3 decode is unverified; section 3 of the
timing script with the change measures it. At steady state the worker runs batches of 1 to 2, so
the gain there is at most the 7.6 ms load per request.

## 5. Corrections to runbook 03

- Live captures need `py-spy record --nonblocking`, or Nsight with the recorder off, so the run is
  not slowed by the sampler.
- `scripts/pyspy_thread_top.py` treated any frame starting with "process" as a thread label, which
  split the tts_engine thread by `process_batch_result` frames in `pyspy_gil_top.txt`; fixed to
  match only py-spy's `process <pid>:` and `thread (` labels.
