# 38. The preprocessing segment under early ids, readout, 2026-09-14

Archive `pp-session-results.zip` (runbook 37), unpacked under `artifacts/pp-session`.
Both boots are dbef8d539 from `tmp/bw` through `python -m sglang_omni.cli serve`, import
path archived under the worktree, window gate passed on both (`shapes_ok=True
window_ok=True`), GPU 1 clear before each boot, no recompile line, zero failed requests
in any pass. P is the PR alone, PE is the PR plus `early_ids.patch` (13 lines in
`model_runner.py`: the layer 0 ids are staged before the predictor and the lookahead is
off). The PE lock record comes from a separate redo boot of the same arm after the
first py-spy run hung; its client pass ran at c16 like PE pass 3.

Nsight numbers below come from the session 2 archive (doc 36): `s2a-nsys` is main plus
early ids, `s2b-nsys` is the branch plus early ids, 20 s windows at c16, the OS runtime
table of each `window.sqlite`. Those two databases are now unpacked under
`artifacts/bw-session-nsys`.

## 1. Verdict

The regression is one mechanism, the per launch interpreter lock handoff, seen from the
preprocessing side. A reference encode is about 1,000 eager launches on one thread
(`qwen3-tts-ref-code`), 15 of them ending in a device to host sync, and every launch
releases and reacquires the lock. Early ids makes the talker thread run more steps per
second (cadence 12.7 to 10.7 ms p50), so each reacquire finds the lock busy more often,
and the encode's own cost rises from 38 to 53 ms p50 with nothing else in the
preprocessing pool. Load only adds pool queueing on top of that. The fix is the same
shape as the vocoder fix: take the encoder's launches out of the thread with captured
graphs. Doc 32 item 3 is the next slice; section 6 has the design and doc 39 the
measurement to run before it is built.

## 2. The three passes, streaming c16, full corpus

| read | P pass 2, closed loop c16 | PE pass 2, closed loop c16 | PE pass 4, open loop 15.8 req/s |
| --- | ---: | ---: | ---: |
| req/s | 15.61 | 18.10 | 15.39 |
| mean requests in flight (Little) | 15.8 | 15.8 | 12.0 |
| TTFC mean ms | 118.4 | 145.9 | 115.1 |
| TTFC p50 ms | 109.7 | 135.7 | 110.8 |
| TTFC p99 ms | 266.6 | 345.0 | 222.6 |
| preprocessing p50 / p95 ms | 36.7 / 104.5 | 59.4 / 158.9 | 44.3 / 89.8 |
| preprocessing, encode mode, nothing else in the pool, p50 ms | 37.9 | 52.9 | 45.6 |
| preprocessing, cache or follower mode, nothing else in the pool, p50 ms | 5.1 | 6.8 | 6.0 |
| prefill p50 ms | 18.8 | 18.0 | 15.0 |
| talker cadence p50 ms | 12.7 | 10.7 | 9.6 |
| first frame to first audio p50 ms | 28.1 | 31.8 | 28.6 |

The preprocessing segment is bimodal. The corpus repeats a reference in 422 of 1,088
requests (the pairs are adjacent), and the ad-hoc reference service caches 256 entries:
over each boot 14 to 19 percent of requests hit the cache, 20 to 25 percent follow a
leader already encoding the same file, and the rest encode. The encode mode with no other
preprocessing request in flight is the clean per request cost, and it moves with the
talker's step rate, not with the pool: 37.9, 45.6, 52.9 ms as the cadence goes 12.7,
9.6, 10.7 ms and the in flight count 16, 12, 16. The cache mode moves the same way on
a smaller base (5.1 to 6.8 ms): tokenization, prompt embedding and a handful of small
launches, no encoder at all.

Pool queueing is the second term. With one other preprocessing request in flight the
segment is 41, 51 and 43 ms p50 on the three passes; with two, 61, 75 and 65 ms. The
open loop pass has fewer in flight, so its tail is shorter than P's (p95 89.8 against
104.5) while its median is longer.

The first pass after each boot is 28 to 31 ms slower on TTFC mean than the second on
both boots (P 146.2 to 118.4, PE 176.6 to 145.9); pass 1 is not quoted anywhere.

## 3. What the lock record says

py-spy with `--gil` writes one trace per sampling interval, the thread holding the
lock; it writes nothing for an interval with no holder or a torn read. Its `Samples: N`
on stderr counts traces written, not intervals, so the fraction of time the lock is
held is not in this record. The report's 24.4 and 27.3 percent lines were
`traces / (rate x duration)` and are void; `scripts/gil_share.py` no longer prints
them. What the record does give is the ownership share per thread:

| thread | P share | PE share |
| --- | ---: | ---: |
| scheduler-tts_engine (talker) | 48.8 | 46.7 |
| qwen3-tts-ref-code (reference encoder) | 17.3 | 18.1 |
| ThreadPoolExecutor-2_0 to 7 (preprocessing pool, 8 threads) | 14.1 | 15.2 |
| vocoder follow-up workers (2) | 9.2 | 9.3 |
| vocoder initial worker | 2.8 | 2.8 |
| MainThread (stage runtime, zmq) | 2.6 | 2.7 |
| scheduler-vocoder | 1.9 | 1.5 |

The talker thread owns the lock about half the time it is held, the reference encoder
a fifth, the pool a seventh. The stacks name the work: on the encoder thread the top
leaf frames are `torch.nn.functional.pad` (14 to 21 percent), `torch.cdist` (10 to 12)
and the Mimi quantizer and conv padding lines; on the pool threads
`extract_speaker_embedding`, the librosa mel filterbank and the prompt builders; on the
initial worker `_zero_slot` in `codec_state_arena.py` (33 to 39 percent of its lock
time, one `zero_` launch per state buffer at every slot acquisition); on the vocoder
scheduler `validate_chunk`'s dtype cast (53 to 59 percent of its small share).

## 4. The lock wait, measured: Nsight OS runtime, early ids arms, 20 s at c16

CPython 3.12 takes a contended interpreter lock through `pthread_cond_timedwait` on
the lock's condition variable and never calls it when the lock is free; Python level
locks and queues are semaphores (`sem_wait`, `sem_clockwait`). The OS runtime rows per
thread therefore separate lock waiting from idle waiting. Window requests: about 325 on
A, 348 on B.

| thread | A: lock waits, ms (percent of window) | B: lock waits, ms | A idle, s | B idle, s |
| --- | ---: | ---: | ---: | ---: |
| scheduler-tts_engine | 355,938 / 6,216 (31.1) | 261,645 / 4,797 (24.0) | 0 | 0 |
| qwen3-tts-ref-code | 225,797 / 4,586 (22.9) | 193,554 / 3,900 (19.5) | 9.4 | 9.9 |
| vocoder initial worker | 324,654 / 6,293 (31.5) | 27,423 / 465 (2.3) | 6.7 | 13.3 |
| vocoder follow-up worker | 56,314 / 1,364 (6.8) | 43,386 / 1,042 (5.2) | 17.0 | 17.1 |
| one preprocessing pool thread | 34,414 / 837 (4.2) | 31,765 / 740 (3.7) | 18.8 | 18.8 |

Each row is count / summed wait. The waits are about one per launch (the encoder thread
has 210 thousand launches and 194 thousand waits on B) at 17 to 24 us each. Two things
this table settles:

- The talker thread waits for the lock 31 percent of its time on main and 24 percent
  with the PR: PR #2151 removed 250 thousand handoffs per 20 s from the initial worker
  and the talker's wait fell 1.4 s per 20 s with it. This is the lock side of the origin
  doc 36 could not measure.
- The reference encoder thread is idle half the window (its queue) and busy 9.8 s; of
  the busy time on B, 3.9 s is lock waiting (40 percent), 1.4 s is launch API time (15
  percent), 0.2 s is the 6,179 stream syncs, 0.6 s a read lock inside the CUDA runtime,
  and the rest is Python dispatch. Its kernels take 0.95 s of GPU time in the window,
  about 4.6 ms per encode. An encode costs the thread 45 to 50 ms and the GPU 5.

## 5. The encoder, per request

Per request in the B window the encoder thread issues 602 launches, 17.8 stream syncs
and 21 copies; the eight pool threads issue about 220 launches between them. The
encoder is the transformers `MimiModel` inside `Qwen3TTSTokenizerV2Encoder`
(transformers 5.12.1, qwen-tts 0.1.1), and its cost is structural:

- 15 causal conv layers (14 in the SEANet stack, ratios 8, 6, 5, 4 with one residual
  block each, plus the downsample conv). `MimiConv1d` keeps `stride`, `kernel_size` and
  `padding_total` as device buffers, so `_get_extra_padding_for_conv1d` runs its integer
  arithmetic as six tiny kernels per layer and `F.pad` then reads the result back: one
  device to host sync per conv layer, the `pad` leaf frame in the lock record.
- 8 transformer layers with a sliding causal window of 250 frames.
- 32 residual vector quantizer stages, each a `cdist`, `argmin`, embedding lookup and
  subtraction; the tokenizer then keeps the first 16 (`encoder_valid_num_quantizers`).
  Half the quantizer launches produce codes that are discarded. `MimiModel.encode` takes
  `num_quantizers`, and the first 16 codes of a residual chain do not depend on the
  stages after them.
- Batching in `_Qwen3TTSRefCodeBatcher` waits 2 ms for up to 8 waveforms; admissions
  arrive 35 to 47 ms apart, so nearly every batch is one waveform.

Everything in the encoder is causal: left padded convs, causal windowed attention, per
frame quantization. A waveform padded with zeros at the tail gives the same codes for
its own frames as the unpadded waveform, which is what the wrapper's batching already
relies on when it pads a batch to its longest member and slices each result to
`ceil(samples / 1920)`. Length buckets are therefore exact by construction, up to the
kernel selection differences that batching already exposes.

## 6. Design for doc 32 item 3

Three parts, in the order they unlock each other, all inside omni's batcher and the
tokenizer load in `stages.py`:

1. Move the three integer buffers of every `MimiConv1d` in the encoder to the host
   after loading. The padding arithmetic becomes Python integers, the 15 syncs and about
   90 launches per encode disappear, and the encoder becomes capturable (a sync inside a
   capture fails). Numerics unchanged.
2. Call `tokenizer.model.encoder.encode(input_values, num_quantizers=16)` from the
   batcher with the wrapper's own normalization and feature extraction, instead of the
   wrapper's `encode` that runs all 32 stages. About 190 launches per encode fewer,
   codes identical.
3. Capture the encoder into CUDA graphs keyed by (batch, length bucket) on the batcher's
   encode stream, the runner shaped like the codec runners: static input buffer per key,
   the waveform copied in, the codes sliced to the waveform's frames. Bucket lengths are
   multiples of 1,920 samples chosen from the reference length distribution; batch keys
   1 and 2 cover what the 2 ms batching window produces.

Expected from the census: the encode mode preprocessing cost falls from 38 ms to near
the GPU time plus one replay, about 6 to 10 ms, in both launches; 200 thousand lock
handoffs per 20 s leave the process, and the talker's lock wait falls again as it did
when the initial worker's 250 thousand left. The speaker encoder on the pool threads
(about 220 launches per request, a per call librosa filterbank on the CPU under the
lock) is the same shape of item and follows once the reference encoder is measured.

Two smaller items from the lock record, for their own slices: `_zero_slot` issues one
launch per arena buffer at every slot acquisition on the initial worker (about 35 of
its remaining 47 launches per request; one `torch._foreach_zero_` call is a handful),
and `validate_chunk` casts every chunk to long on the vocoder scheduler thread.

## 7. Next

Doc 39 is the box run: a micro bench on the tokenizer alone that checks parts 1 and 2
are bit exact against the wrapper on real references, captures graphs at candidate
buckets, checks the sliced codes against eager on the same references, and times eager
against replay per key with the capture footprint. Its numbers pick the buckets and
decide whether batch 2 keys earn their memory before the runner is written.
