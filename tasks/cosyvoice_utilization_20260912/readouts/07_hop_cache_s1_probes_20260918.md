# Hop prefix cache, slice S1: probe results (2026-09-18)

RTX 4090 D, card 0 alone on every run, torch 2.13.0+cu130, sglang 0.5.19. Branch
`slice/cosyvoice-4-1-hop-prefix-cache` at 439c62518, built by its own factory. Probe
`stage2/s1_hop_cache_gate.py`, kernel probe `stage2/fa3_append_probe.py`. Raw outputs on the Mac
under `artifacts/cosyvoice-4090-20260918/s1-r2` to `s1-r9`. Unit suite on the box: 213 passed,
2 skipped. No serving run yet.

## 1. Correctness

- A step whose rows are all first hops is bit identical to main's hop (3 of 3 rows): conv tails,
  RoPE positions, slicing, CFG lanes, pool write and FA3 read reproduce main exactly.
- Every output finite; the pool returns all 14,650 slots and 293 rows on release.
- Later hops sit 27.5 dB (minimum) and 35 to 37 dB (median) from main's hop. They cannot be bit
  identical: the rows packed beside them differ.

## 2. Speed and memory, median of 3 synchronized calls

| schedule | main's hop | cached hop |
|---|---|---|
| 8 streams x 6 steps, total | 2,794 ms | 1,407 ms |
| last step, 8 rows, 6,850 frame window | 965 ms | 251 ms |
| 2 streams x 16 steps, total | 7,539 ms | 3,606 ms |
| last step, 2 rows, 5,950 frame window | 996 ms | 222 ms |
| first hop, 2 rows, 400 frames | 198 ms | 221 ms |

The cached hop is flat at 220 to 250 ms, which is the launch floor. Peak activation memory of
the largest hop: 367 MiB on main, 92 MiB cached (8 streams); 319 against 30 MiB (2 streams).

**The small hop is 20 to 26 ms slower than main's**, in every run, until the window passes about
2,000 frames. That breaks the no regression rule at low concurrency and blocks the serving runs.

Where it goes (step 0, 2 rows): kernel launches 16,936 to 17,240 (+220 store calls, +84 conv
tail operations). Under cProfile the write chain (`set_kv_buffer`, `_set_kv_buffer_impl`, the
`store_cache` wrapper and op, its views, an environment lookup per call) is about 18 ms of the
34 ms difference; the rest is spread over about 9,000 more Python calls, mostly the two buffer
getters and their views per layer. A tight loop understates it: 220 writes cost 5.3 ms shipped,
2.4 ms calling `store_cache` on views made once, 3.8 ms with two `index_copy_`.

## 3. FA3 can append the new K and V itself

`flash_attn_with_kvcache(k=, v=, cu_seqlens_k_new=)` appends at `cache_seqlens` and attends in
one call. Against store then attend, on 1 row, 8 mixed follow up rows and 32 lane rows with 128
chunk segments sharing page table rows: attention output, K cache and V cache **bit identical**
on all three, so a later segment of a row does read what an earlier one appended in the same
call. 220 calls: host 7.2 ms against 11.0 ms; wall at 32 rows 65.6 against 70.9 ms.

Proposed for the branch, not committed: one FA3 call per layer with `cache_seqlens` holding the
lengths before the hop, the page shaped pool views made once at construction, `slots` dropped.
Pool, allocator and request table stay SGLang's. Expected to remove most of the gap in section 2;
to be measured.

## 4. The cached path is about 1 dB further from the float32 truth, and why

Reference: main's own hop on each row alone, which is +0.04 dB from main batched (batch size does
not move accuracy).

| hops | n | cached minus main alone | standard error |
|---|---|---|---|
| first hops in mixed steps | 6 | -0.28 dB | 0.14 |
| later hops | 28 | -1.31 dB | 0.26 |
| later hops of 100 tokens | 20 | -1.73 dB | 0.27 |

What it is not:

- Not the boundary state. Along the new span the deficit is -0.9 to -1.5 dB in the first 30
  frames and -1.7 to -2.7 dB at frames 130 to 200; a wrong tail or position would pile up at the
  start.
- Not accumulated over the Euler steps. The DiT output is already -1.4 dB at Euler step 0,
  where every path starts from the same noise, and stays between -1.1 and -2.1 dB to step 9.
- Not the conv. The hidden state entering block 0 (projection plus conv position embedding with
  the cached tails) is 48.7 dB on both paths, a gap of 0.0. A float32 conv lifts block 0 by
  2.2 dB, is gone by block 2, and leaves the step 0 output gap at -1.5 dB.

What it is: the gap opens gradually with depth at Euler step 0, with no step at any block.

| block | 0 | 4 | 8 | 12 | 16 | 18 | 21 |
|---|---|---|---|---|---|---|---|
| cached minus main alone, dB | +0.0 | -0.5 | -0.5 | -0.6 | -1.0 | -1.4 | -1.1 |

Each block's attention reads prefix K and V that another pass rounded (different packed shapes,
so different kernels), while main's prefix carries this pass's rounding, consistent with its
queries. The cache is exact in float64 (E5); in bfloat16 this is the price of reusing K and V
across passes, the same property an LLM prefix cache has. For scale: main's own hidden state
falls from 48.7 dB at block 0 to 15 dB at block 21 in the same forward, and main's emitted mel
ranges from 23 to 40 dB across rows.

Consequence: no code fix exists or is needed; the serving quality metrics (WER, similarity,
continuity) decide, and the breakable graph PR inherits nothing new from this.

## 5. Owed before S1 can be a PR

1. The append change of section 3, then the first hop against main again.
2. Serving runs on this branch only, against the main boots on disk (b5c3b44aa, memory fraction
   0.3): seeded c1 for continuity and first audio, unseeded c8 and c16 with the fallback count,
   one starved budget run that must complete on the fallback.
