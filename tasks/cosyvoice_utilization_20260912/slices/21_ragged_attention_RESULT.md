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

## c16 streaming, one boot per arm

32 English SeedTTS samples, streaming, seed 1234, warmup 1, card 4, KV pool
declared at 2 GiB on both arms so this is not a memory comparison. The other
cards carried another tenant at about 24 GB throughout, equally for both arms,
which ran back to back.

| | main | ragged | |
|---|---|---|---|
| completed / failed | 32 / 0 | 32 / 0 | |
| RTF mean | 1.3567 | 1.1840 | -12.7% |
| RTF p99 | 3.2589 | 2.9608 | -9.1% |
| first audio mean | 2.562 s | 2.391 s | -6.7% |
| first audio p95 | 3.277 s | 2.852 s | -13.0% |
| inter chunk mean | 1.667 s | 1.324 s | -20.6% |
| req/s | 2.589 | 2.978 | +15.0% |
| audio s/s | 11.833 | 13.493 | +14.0% |

One boot per arm is the standing protocol, and a 13 to 15 percent delta is far
outside the roughly 2 percent band that would call for a repeat.

Two things this table is not. It is not an H100 number: RTF p99 stays near 3 and
the roadmap's target of p99 below 1 is not reachable on this card. And c16 only
boots here at all because the KV pool is declared: without it the pool takes 9.78
GiB and the run fails on ONNX cuBLAS handles (`../MEMORY_TRACE_20260917.md`).

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
