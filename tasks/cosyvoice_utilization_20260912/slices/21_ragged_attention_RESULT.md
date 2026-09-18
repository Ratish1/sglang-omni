# Slice 2.1: the packed sequence read ragged

Plan row: `../plans/12_flow_graph_redesign.md` section 4.1. Branch
`slice/cosyvoice-2-1-ragged-attention`, head `a81317fc`, on upstream main
`27a8293c`. Runs on the moss box, RTX 4090 D, card 4, 2026-09-17.

## What changed

`RowAttention` scattered the packed rows back to a `(rows, width, channels)`
layout, ran one SDPA under a `(rows, 1, width, width)` mask and gathered the rows
back (`packed_dit.py:104-136`). Every row therefore paid for the widest row in the
batch, twice over: once in the scatter and once in the attention itself.

`RaggedRowAttention` reads the packed sequence where it lies. `chunk_segments`
cuts it into one query segment per (row, chunk), and one FA3 call reads
[0, chunk end) of each segment's own row through a page table of single frame
pages. Nothing is padded to the widest row.

The padded read stays for every platform that cannot run the ragged one:
`ragged_attention_supported` requires CUDA, FA3, and a solve running under an
autocast in the half precision FA3 reads. A `dtype="float32"` vocoder disables
autocast and keeps the padded read, which is the case review caught: an earlier
draft cast unconditionally to bfloat16 and would have silently downcast a float32
deployment's attention.

## Gate: the shipped hop against the same float32 truth

The G0 harness on both trees, 4 streams, 4 steps, 10 emitted hops each. The
comparison is the `production` column, the shipped path, against the float32
padded truth that both trees compute identically outside autocast.

| shipped hop | mel min | mel median | spectrum min | spectrum median |
|---|---|---|---|---|
| main, padded SDPA | 23.793 | 36.068 | 5.583 | 14.433 |
| 2.1, ragged FA3 | **24.380** | **36.772** | **7.317** | **14.699** |

All four move up, all finite, every length matching. The gate asked for no fall
beyond the G0 margin; the measured direction is favourable. It should not be read
as an improvement worth claiming: the 4090 work established that two kernels of
equal accuracy differ by a few dB of per sample rounding either way over ten Euler
steps and 22 blocks (`../stage2/README.md`). What this shows is that the ragged
read is not a precision compromise, which is what E6 predicted.

## Proof: the isolated hop call, same shapes, same card

From the same two G0 runs, the wall of one production hop call:

| step | rows | window frames | main | ragged | |
|---|---|---|---|---|---|
| 0 | 1 | 150 | 225.4 ms | 187.7 ms | -16.7% |
| 1 | 2 | 450 | 229.4 ms | 189.4 ms | -17.4% |
| 2 | 3 | 950 | 228.9 ms | 194.8 ms | -14.9% |
| 3 | 4 | 1,700 | 279.1 ms | 230.2 ms | -17.5% |

Consistent at every shape, and the absolute gap widens with the batch, 37.7 ms at
one row to 48.9 ms at four: the `rows x width squared` term is what leaves.

## c16 streaming, the whole English split, one boot per arm

1,088 requests, the full English SeedTTS split, streaming, seed 1234, warmup 1,
card 4, KV pool declared at 2 GiB on both arms so this is not a memory
comparison. The arms ran back to back on the same card.

| | main | ragged | |
|---|---|---|---|
| completed / failed | 1088 / 0 | 1088 / 0 | |
| RTF mean | 1.7150 | 1.0838 | -36.8% |
| RTF p99 | 6.8663 | 2.3578 | -65.7% |
| first audio mean | 3.480 s | 2.175 s | -37.5% |
| first audio p95 | 9.563 s | 3.307 s | -65.4% |
| inter chunk mean | 2.089 s | 1.339 s | -35.9% |
| req/s | 2.074 | 3.318 | +60.0% |
| audio s/s | 10.332 | 15.780 | +52.7% |

The tail moves most, which is the shape of the defect: the padded read charged
every row for the widest row in its batch, so a batch holding one long request
paid for it 16 times over. The stage 1 ledger measured that case directly, a
16 row hop holding one row of 2,051 tokens spending 67 percent of a 3,061 ms
call in padded attention and 10 percent in the mask.

An earlier pass of this same pair over the first 32 samples reported +15.0
percent req/s. That subset is the head of the corpus and holds none of the long
requests, so it understated the gain roughly fourfold. The full split is the
number; the subset is recorded here only because it is why the protocol asks for
the whole corpus.

RTF p99 is still above 1, so the roadmap target is not met on this card, but it
is 2.9x closer than main.

## Buffered c16: unchanged, by design

Buffered traffic never reaches the packed path. `decode_batch` goes through
`inference` to `generate_flow` (`stages.py:778`), the padded and graphed path;
this slice only changes `solve_flow_euler_packed`, which serves streaming hops
and stream finals. So the expectation is no change, and the measurement agrees:

| buffered c16 | main | ragged |
|---|---|---|
| completed / failed | 32 / 0 | 32 / 0 |
| RTF mean | 0.6538 | 0.6247 |
| RTF p99 | 1.0768 | 1.1023 |
| req/s | 5.278 | 5.099 |
| audio s/s | 24.919 | 23.060 |

Mixed in both directions, which is what an unchanged code path on a shared box
looks like. It is useful as a noise floor: about 5 to 7 percent on c16 buffered
throughput here, which puts the streaming gain at roughly three times the noise.

## Utilization, Nsight GPU metrics

`nsys launch --session-new` around the server and the runner opening the window
on the measured benchmark only, the protocol in `../diagnostics/README.md`
section 3, `--gpu-metrics-devices=4` resolving to "General Metrics for NVIDIA
AD10x". 32 requests at c16 streaming, a finite cohort under profiling overhead,
not the acceptance benchmark, which is the full split above.

Raw means over the capture window:

| metric | main | ragged |
|---|---|---|
| GR Active | 51.04% | 48.40% |
| SMs Active | 42.78% | 39.01% |
| SM Issue | 8.62% | 6.57% |
| Compute Warps in Flight | 20.19% | 15.59% |
| Unallocated Warps in Active SMs | 22.63% | 23.47% |

**Every number falls, and that is the result, not a regression.** Utilization as
a percentage is the wrong headline when throughput moved 60 percent: the same
audio is produced with less of the card. Normalised by the audio each arm
produced:

| per second of audio produced | main | ragged | |
|---|---|---|---|
| GR Active | 0.0548 s | 0.0457 s | -16.6% |
| SMs Active | 0.0459 s | 0.0369 s | -19.6% |
| SM Issue | 0.0092 s | 0.0062 s | -32.6% |

19.6 percent less SM active time per second of audio, which is the same size as
the 15 to 17 percent the isolated hop call moved, measured a different way.

The standing story is in the absolute numbers rather than the delta. SM Issue is
8.62 percent on main and 6.57 percent here, against SMs Active of 42.78 and
39.01: the SMs are resident and barely issuing, and Unallocated Warps in Active
SMs sits above 22 percent on both arms. The card is occupied, not working. That
is a launch and occupancy bound, not a compute bound, so the next lever is the
graph work of `../plans/12_flow_graph_redesign.md`, not another kernel.

## What this unblocks

The cost of a Flow call is now total frames in every module, so the graph shape
key can be one number. That is the prerequisite plan 12 section 4.1 named, and
slice 2.2, the derived bucket table, is what it was for.

## Review notes

- 201 unit tests pass, 2 skipped, full `tests/unit_test/fun_cosyvoice3/`.
- The ragged read is tested against dense per row SDPA for both the chunked and
  the full mask, on CUDA, and `chunk_segments` is tested on its own as host
  arithmetic.
- `chunk_segments` duplicates what `flow_hop_cache.hop_layout` does on the parked
  slice 1.2 branch. When 1.2 is unparked it should import this one rather than
  keep its own: the segment layout is a property of the packed sequence, not of
  the cache.

## FA3 varlen for finals: measured, not worth a second path (2026-09-18)

A final's keys do not overlap between rows, so `flash_attn_varlen_func` with
`cu_seqlens_q = cu_seqlens_k = row starts` can express it, as SGLang's packed
bidirectional attention does (`srt/layers/attention/vision.py:943-952`). A hop
cannot: its (row, chunk) segments read overlapping prefixes of one row, which
varlen's consecutive key ranges cannot describe; SGLang's own prefix read is the
paged call too.

`stage2/row_attention_kernels.py`, RTX 4090 D, bfloat16, 16 heads of 64, rows
doubled for CFG, one attention call, clean run:

| shape | paged ms | varlen ms | padded SDPA ms | page table build ms |
|---|---|---|---|---|
| 1 row, 400 | 0.049 | 0.046 | 0.137 | 0.145 |
| 1 row, 1,200 | 0.113 | 0.110 | 0.242 | 0.100 |
| 4 rows, 300 to 900 | 0.152 | 0.149 | 0.566 | 0.105 |
| 8 rows, 300 to 1,500 | 0.513 | 0.504 | 2.524 | 0.111 |
| 16 rows, 300 to 1,500 | 0.991 | 0.982 | 5.135 | 0.120 |
| 16 rows, one of 4,000 | 1.344 | 1.313 | 24.993 | 0.147 |

The two FA3 outputs are bit identical on every shape. varlen is 1 to 3 percent
of one attention call faster; over the 220 attention calls of a Flow call that is
about 2 ms at 16 rows plus 0.1 ms of page table build, under half a percent of
the call. One kernel path for hops and finals stays. A second run shared the
card with a server boot and is discarded.
