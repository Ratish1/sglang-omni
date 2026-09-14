# 40. Reference encoder bench readout, and the padding check to run next

Archive `runbook39-handoff.zip`, unpacked under `artifacts/runbook39-handoff`. Code under
test upstream main after the #2151 squash (868a94d08), import path
`tmp/main/sglang_omni/__init__.py`, GPU 1, 64 distinct references, encoder in bfloat16
as the vocoder loads it. Reference lengths in frames: min 40, p50 55, p90 75, max 98
(3.2 to 7.8 s at 12.5 Hz).

## 1. Parts 1 and 2 of the design hold

| encode of one reference, idle GPU | host ms | device ms | codes against the wrapper |
| --- | ---: | ---: | --- |
| wrapper, today's path | 9.93 | 9.90 | |
| host side padding buffers (15 conv layers moved) | 8.99 | 8.98 | exact, 64 of 64 |
| plus `num_quantizers=16` | 6.71 | 6.70 | exact, 64 of 64 |

On an idle GPU with a free lock the encode is 10 ms; in the server it is 38 to 53 ms.
That gap is the contention doc 38 measured, not the encoder's own work.

## 2. Graphs are cheap

| key (frames x batch) | replay device ms | replay host ms | footprint MiB |
| --- | ---: | ---: | ---: |
| 48x1 | 2.40 | 0.04 | 64 (pool init) |
| 64x1 | 2.55 | 0.05 | 32 |
| 96x1 | 2.80 | 0.05 | 32 |
| 128x1 | 3.10 | 0.05 | 32 |
| 192x1 | 3.67 | 0.05 | 32 |
| 48x2 to 192x2 | 2.76 to 5.28 | 0.05 to 0.07 | 32 each |

A replay is 2.4 to 3.7 ms of device time against 6.7 eager for the same work, the host
side is 50 us against 6,700, and a key costs 32 MiB. Buckets 48, 64, 96, 128 cover the
corpus with room; memory is not a constraint on the key set.

## 3. The bucket padded codes are not the eager codes

Batch 1: 1 of 64 references exact, 63 differ, 70 to 293 differing codes each out of
about 16 x 55, in 40 of the 64 from quantizer 0 on. Batch 2: 0 of 64. That is far more
than the last frame, so the doc 38 argument (causal encoder, tail padding exact) is not
what the numbers show, and a graph cannot ship on it. Two mechanisms can produce this
and the bench does not separate them:

1. Shape dependent bfloat16 numerics. The encoder runs in bfloat16 at a different
   length, cuDNN and cuBLAS choose different kernels and reduction orders, the
   embeddings round differently, and the codebook argmin flips where two entries are
   close. A flip at stage k of the residual chain changes every later stage of that
   frame. Today's wrapper has the same exposure when a batch pads a reference to a
   longer member, but batches are singletons in practice.
2. The last partial frame. When the length is not a multiple of a layer's stride, the
   layer appends its own zeros before the conv; with the input padded up front, that
   window covers the conv of a zero tail instead. Only the last frame at each layer is
   affected, so this explains at most the last one or two output frames, not 14 to 30
   percent of the codes; it is real all the same and the runner has to define the last
   frame one way.

## 4. The check to run

`scripts/ref_encoder_padding_check.py`, no graphs except a float32 control, a few
minutes on GPU 1. For each reference in bfloat16 and in float32 (a second tokenizer
instance loaded in the checkpoint's own dtype) it encodes the waveform as is, again,
zero padded to the next multiple of 1,920 samples, and zero padded to its bucket, and
splits every difference against the first run into the last two frames and the frames
before them. It also compares float32 against bfloat16 on the unpadded waveform, and
captures float32 graphs per bucket to check the replay against the eager padded run.

```bash
cd /sgl-workspace/sglang-omni/tmp/an && git pull -q
cd /sgl-workspace/sglang-omni/tmp/main
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python ../an/tasks/perf_backlog/scripts/ref_encoder_padding_check.py \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base --meta zhaochenyang20/seed-tts-eval-arrow --lang en \
  --samples 64 --bucket-frames 48,64,96,128,192 \
  --out /sgl-workspace/sglang-omni/tmp/ref_encoder_padding_check.json 2>&1 \
  | tee /sgl-workspace/sglang-omni/tmp/ref_encoder_padding_check.log
```

Reads, stated before the run:

- `again` must be exact in both dtypes; if not, the encoder is nondeterministic at a
  fixed shape and nothing downstream can be bit gated.
- If bfloat16 `aligned` and `bucket` differ from `plain` in the earlier frames while
  float32 `aligned` and `bucket` differ only in the tail (`tail_only` equal to
  `mismatched`), mechanism 1 is the cause, and the runner encodes references in
  float32: the eager fallback and every graph then give the same codes for a waveform
  at any bucket, and the codes are the checkpoint's own numerics rather than a
  bfloat16 rounding of them. `float32_vs_bfloat16_plain` says how far today's codes
  sit from those.
- If float32 also differs in the earlier frames, the encoder is not causal somewhere
  and the bucket design is wrong; the finding is then which layer.
- `float32_graph.against_eager_bucket` must be exact: a captured graph reproduces its
  own eager shape.
- The tail: the runner pads every reference to a multiple of 1,920 samples before
  encoding, in the eager path and the graph path alike, so the last frame is defined
  once. That last frame differs from today's wrapper by construction; the quality
  bands of the census carry it.

## 5. The default launch lock read, from the s3 Nsight boots

`nsys_lock_waits.py` on the doc 36 default launch pair, 21.8 and 21.7 s windows at c16,
main against the branch, `pthread_cond_timedwait` count / ms (percent of window):

| thread | main 3060470a8 | branch f73586369 |
| --- | ---: | ---: |
| talker (scheduler-tts_engine) | 217,780 / 3,477 (16.0) | 168,055 / 2,827 (13.1) |
| vocoder initial worker | 164,685 / 2,731 (12.5) | 18,197 / 284 (1.3) |
| reference encoder thread | 127,974 / 2,324 (10.7) | 113,688 / 2,039 (9.4) |

The talker's lock wait on the default launch is half of what it is under early ids (doc
38: 31 and 24 percent), which is why the regression only shows with #2123 applied.

## 6. Where the open PRs stand

#2123 (early ids) is held on its doc 31 gate, first chunk within 10 ms of main at early
ids throughput; with the encoder at 38 to 53 ms per request under it the gate misses by
about 23 ms of TTFC mean. #2126 (non blocking copies, draft) was measured on top of it.
Order: item 3 lands on main, #2123 is rebased and remeasured as main against main plus
#2123 on the doc 33 protocol, then #2126 the same way on the new main.
