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

## 4a. The append change, applied (branch at 7b3e3c41e)

`CachedHop.__call__` is one FA3 call with `k`, `v`, `cu_seqlens_k_new` and `cache_seqlens` as
the lengths before the hop; the page shaped pool views are made once; `slots` is gone. Both
SGLang wrappers pass the arguments straight to the kernel and ask only for contiguous inputs
(`kernels/ops/attention/flash_attention.py`, `flash_attention_v3.py`). Raw: `s1-r10`.

- Unit suite 213 passed, 2 skipped.
- Append against store then attend: bit identical on the three layouts, and 0 mismatches in 200
  repeats on fresh values per layout (output, K cache, V cache).
- The small hop regression is gone: 201.0 ms cached against 201.2 ms main at step 0, 208.5
  against 208.8 at step 1. The 8 x 6 schedule is 2,812 ms on main and 1,335 ms cached; pure
  first hops still bit identical; later hops still -1.29 dB against main alone (section 4).

## 4b. Serving, c16 only, whole English split, unseeded, memory fraction 0.3

Branch with `--vocoder.factory.flow_kv_cache_bytes 6442450944` (6 GiB, 6,950 slots, 3,475
frames) against the two main boots on disk (b5c3b44aa). Runaways are requests of 80 s or more.

| run | runaways | req/s | audio s/s | latency mean | latency p95 | TTFP mean | TTFP p95 | inter chunk mean | underrun mean | c50 | c100 | c200 | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main boot 1 | 1 | 3.301 | 15.65 | 4.84 | 7.09 | 2.25 | 3.45 | 1.32 | 0.477 | 22.6 | 27.3 | 41.2 | 0 |
| main boot 2 | 3 | 2.939 | 14.62 | 5.43 | 8.51 | 2.53 | 4.14 | 1.43 | 0.610 | 10.7 | 21.3 | 39.5 | 0 |
| hop cache | 3 | 2.997 | 15.03 | 5.23 | 7.45 | 2.52 | 3.77 | 1.33 | 0.489 | 40.3 | 42.7 | 47.9 | 0 |

- Against the main boot with the same runaway count: +2.0 % req/s, +2.8 % audio throughput,
  latency mean -3.7 %, p95 -12 %, inter chunk mean -6.9 %, TTFP mean equal, TTFP p99 5.35
  against 5.02 s. Against the one runaway boot it is lower on throughput, which is the draw
  (readout 06: 12 % between boots of one tree).
- Continuity is the clear gain: c50 40.3 against 22.6 and 10.7, c100 42.7 against 27.3 and
  21.3.
- The pool was far too small for c16 on this card, as expected: 59.4 % of hop rows ran cached
  (1,313 of 2,212), 494 fallbacks, 167 of 228 hop steps were mixed (two Flow calls), 46 had no
  cached row. So this run measures the fallback regime more than the cache.
- Memory: peak on the card 23,766 of 24,564 MiB. With 6 GiB taken before the engine, SGLang's
  AR pool came out at 27,537 tokens (0.32 GB); token usage peaked at 0.19, no retraction, a
  queue on 5 of 405 decode log lines. No out of memory, no traceback.
- Not available from this pair: per step hop times. Neither tree logs them, so the throughput
  comparison stays exposed to the runaway draw.

## 4c. Second c16 boot and c8 (raw: `s1-r11`, `s1-r12`)

Same branch, same 6 GiB cache, graphs on. The second c16 boot ran at memory fraction 0.28, the
lowest that still boots beside the cache (AR pool 14,335 tokens, no retraction); c8 at 0.3.

| run | runaways | req/s | audio s/s | latency mean | TTFP mean | inter chunk mean | underrun mean | c50 | c100 | c200 | cached rows | fallbacks | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c16 main boot 1 | 1 | 3.301 | 15.65 | 4.84 | 2.25 | 1.32 | 0.477 | 22.6 | 27.3 | 41.2 | | | |
| c16 main boot 2 | 3 | 2.939 | 14.62 | 5.43 | 2.53 | 1.43 | 0.610 | 10.7 | 21.3 | 39.5 | | | |
| c16 cache boot 1 | 3 | 2.997 | 15.03 | 5.23 | 2.52 | 1.33 | 0.489 | 40.3 | 42.7 | 47.9 | 59.4 % | 494 | 23,766 |
| c16 cache boot 2 | 2 | 3.055 | 14.85 | 5.22 | 2.65 | 1.29 | 0.422 | 47.7 | 51.2 | 54.2 | 60.0 % | 480 | 24,082 |
| c8 main boot 1 | 2 | 2.758 | 13.31 | 2.90 | 1.495 | 0.705 | 0.090 | 74.7 | 76.1 | 83.2 | | | |
| c8 main boot 2 | 4 | 2.611 | 12.94 | 3.06 | 1.501 | 0.771 | 0.130 | 70.5 | 72.4 | 79.2 | | | |
| c8 cache | 1 | 3.023 | 14.22 | 2.64 | 1.404 | 0.637 | 0.051 | 78.9 | 83.0 | 89.8 | 98.2 % | 22 | 23,992 |

- c8, where the pool holds the traffic: +9.6 % req/s and +6.8 % audio throughput against main's
  better boot, every latency and continuity metric better, 0 failed. The draw favours the cache
  boot by one runaway.
- c16 means of two boots each: req/s 3.026 against 3.120 (-3.0 %), audio throughput 14.94
  against 15.13 (-1.3 %), TTFP mean 2.58 against 2.39 s, all inside the spread of main's own
  two boots; continuity c50 44.0 against 16.6, c100 46.9 against 24.3. Only 60 % of rows fit
  the 6 GiB pool and about 70 % of hop steps are mixed, which pays the launch floor twice.
- Lowering the memory fraction frees nothing for the cache on this card: the AR pool was already
  0.32 GB at 0.3 and the card peaks within 0.5 GB of full.

## 4d. Third c16 boot, and what the five boots say together (raw: `s1-r13`)

Cache boot 3, memory fraction 0.28, same cache: 4 runaways, 2.871 req/s, 14.42 audio s/s,
latency mean 5.55 s, TTFP mean 2.84 s, c50 48.6, c100 49.5, c200 52.4, 0 failed, 59.5 % of rows
cached, 479 fallbacks, peak 24,056 MiB.

c16 req/s ordered by runaway count, all five boots:

| runaways | 1 | 2 | 3 | 3 | 4 |
|---|---|---|---|---|---|
| tree | main | cache | main | cache | cache |
| req/s | 3.301 | 3.055 | 2.939 | 2.997 | 2.871 |
| c50 | 22.6 | 47.7 | 10.7 | 40.3 | 48.6 |

Throughput follows the runaway count on one line for both trees, and at the one count both
trees drew (3) the cache is +2.0 %. So at c16 on this card the cache is throughput neutral,
with 60 % of rows cached, and continuity is better on every boot (c50 40 to 49 against 11 to
23). A runaway is the AR model never sampling its stop token; the benchmark's default
`max_new_tokens=2048` (`benchmarks/eval/benchmark_tts_seedtts.py:195`) overrides the engine's
20 x text bound (`request_builders.py:785-798`), so it runs to 82 s. The runner sets nothing
beyond model, dataset, concurrency and streaming. The PR body carries the c8 pair only.

## 5. Owed before S1 can be a PR

1. The append change of section 3, then the first hop against main again.
2. Serving runs on this branch only, against the main boots on disk (b5c3b44aa, memory fraction
   0.3): seeded c1 for continuity and first audio, unseeded c8 and c16 with the fallback count,
   one starved budget run that must complete on the fallback.
