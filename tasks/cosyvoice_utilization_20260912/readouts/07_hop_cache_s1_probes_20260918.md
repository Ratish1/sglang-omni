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

## 4e. c16 with `max_new_tokens` left unset (raw: `s1-r14`)

Branch only, memory fraction 0.28, same cache, the runbook's `MAX_NEW_TOKENS=unset` (the client
sends no limit, the engine applies 20 tokens per text token).

| | cap 2048, three boots | unset |
|---|---|---|
| requests of 80 s or more | 3, 2, 4 | 0 (longest audio 10.0 s) |
| req/s | 2.997, 3.055, 2.871 | 3.749 |
| audio s/s | 15.03, 14.85, 14.42 | 17.41 |
| RTF mean | 1.152, 1.169, 1.212 | 0.967 |
| latency mean / p95 | 5.23 / 7.45, 5.22 / 7.85, 5.55 / 8.77 s | 4.25 / 5.80 s |
| TTFP mean / p95 | 2.52 / 3.77, 2.65 / 3.86, 2.84 / 4.95 s | 1.99 / 2.80 s |
| underrun mean | 0.489, 0.422, 0.548 s | 0.269 s |
| c50 / c100 / c200 | 40 to 49 / 43 to 51 / 48 to 54 | 55.2 / 58.7 / 61.1 |
| cached rows, fallbacks | 59 to 60 %, 479 to 494 | 60.7 %, 485 |
| peak card memory | 23,766 to 24,082 MiB | 22,066 MiB |
| failed | 0 | 0 |

The two to four runaways of a boot cost this tree about a fifth of its c16 throughput and set
its memory peak (their finals). No main boot exists under the same rule, so this row compares
with nothing on main. The pool still covers only 61 % of rows, so c16 on this card stays a
memory question with or without runaways.

## 4f. The clean c16 pair, `max_new_tokens` unset on both sides (raw: `s1-r14`, `s1-r15`)

Main at 27b5b0d4f (the branch's base), memory fraction 0.28 on both, no request above 13.6 s
on either side, one boot each.

| metric | main | hop cache | delta |
|---|---|---|---|
| req/s | 3.624 | 3.749 | +3.4 % |
| audio s/s | 16.92 | 17.41 | +2.8 % |
| RTF mean | 1.000 | 0.967 | -3.4 % |
| latency mean / p95 / p99 | 4.40 / 5.82 / 6.40 s | 4.25 / 5.80 / 7.19 s | -3.3 % / -0.3 % / +12.3 % |
| TTFP mean / p95 / p99 | 2.06 / 2.69 / 3.09 s | 1.99 / 2.80 / 3.09 s | -3.0 % / +4.1 % / +0.1 % |
| inter chunk mean | 1.206 s | 1.168 s | -3.2 % |
| underrun mean | 0.339 s | 0.269 s | -20.8 % |
| c50 / c100 / c200 | 20.4 / 33.2 / 48.3 | 55.2 / 58.7 / 61.1 | +34.8 / +25.5 / +12.9 points |
| peak card memory | 18,008 MiB | 22,066 MiB | +4.1 GB (the 6 GiB pool less smaller hops) |
| failed | 0 | 0 | |

With the runaways gone the c16 result has the sign of the c8 one, at the size a pool that covers
61 % of rows allows. The latency p99 is the one metric that moved the wrong way; with 70 % of
steps mixed, a row in the second Flow call of a step waits for the first.

## 4g. Why the serving gain is small: the call ledger at c16 (2026-09-19, raw: `s1-r16`, `s1-r17`)

Two profiling boots of the branch (7b3e3c41e, the settings of 4f) under the stage 0 call ledger,
which now also wraps `hop_batch_cached` and records the calling thread's CPU time per call
(`time.thread_time`). Profiling boots, 3.678 and 3.470 req/s, never compared with the census.
Scripts: `stage2/step_gaps_from_log.py`, `stage2/ledger_step_split.py`.

**The request shape bounds the cache.** Both c16 logs give 1.94 hops per request: hop 1 (prompt
plus 25 tokens), hop 2 (50 tokens), then the final. Hop 1 has nothing to reuse. The final is
bidirectional over the whole history, and upstream CosyVoice runs it the same way
(`cosyvoice/cli/model.py:367-373` calls `token2wav` with `finalize=True` and no `stream`), so it
is the model's behaviour. The cache can serve one Flow pass in three on this corpus.

**The vocoder is the saturated stage.** Its thread is inside a step for 283.6 of 296.8 s (first
boot). The AR decodes 2,300 to 3,100 tok/s when its batch is full and half of its decode steps
hold 4 requests or fewer. Step time by call (first boot): cached hops 85.1 s, finals 73.2 s,
HiFT 63.7 s (3,208 per row calls), plain hops 60.2 s, everything else 1.4 s.

**Every Flow call has a launch floor, and AR activity stretches it.** Per call, p50, second boot;
"AR active" means the AR logged a prefill or decode batch during the call:

| call | computed frames | AR quiet: wall / CPU ms | AR active: wall / CPU / off CPU ms |
|---|---|---|---|
| cached hop | up to 2,500 | 230 / 226 | 659 / 356 / 290 |
| cached hop | 2,500 to 4,000 (first hop cohorts) | 353 / 347 | 664 / 338 / 321 |
| plain hop | up to 2,500 | 248 / 242 | 526 / 308 / 220 |
| plain hop | 2,500 to 4,000 | 343 / 338 | 431 / 335 / 104 |
| final | up to 2,500 | 229 / 225 | 463 / 289 / 169 |
| final | above 4,000 | 881 / 875 | none in this boot |
| HiFT | | 15.3 / 14.9 | 53.4 / 28.9 / 25.0 |

- AR quiet, a call of up to 2,500 computed frames costs 229 to 248 ms whatever it computes: the
  eager launches of the DiT. The typical cached hop computes 800 frames of a 3,050 frame window
  (first boot, p50 of 55 calls) and lands on the same floor main's hop over the window does.
- The finals above 4,000 frames wait for the GPU for most of their 881 ms and their CPU time is
  875 ms, so a wait for the GPU is on the CPU (the sync spins). The 170 to 320 ms a call spends
  off the CPU while the AR is active is therefore not the GPU. The three stages are threads of
  one process; the signature (CPU over wall 0.52 to 0.65 with one other busy Python thread) is a
  wait for the GIL. Certainty needs the `pthread_cond_timedwait` rows of the vocoder thread in an
  Nsight OSRT capture.
- Over the second boot the vocoder's steps are 302.5 s of wall and 247.0 s of CPU. Priced at
  their AR quiet p50, the calls that ran while the AR was active would be 56 s instead of 119 s:
  about a fifth of the vocoder's time is this stretch. First hops always run while their cohort's
  AR is decoding, so they always pay it.
- At a device bound rate of 0.123 ms per computed frame (the AR quiet finals above 4,000 frames),
  the first boot's Flow calls hold 137 s of device work in 219 s of wall; 51 of the 81 s
  difference sit in the cached hop calls.

**Mixed steps.** Hop step p50 in the 4f cache boot: all rows cached 725 ms, mixed 1,039 ms (143
of 203 hop steps); at 5 to 9 rows 926 against 1,188 ms, at 1 to 4 rows 684 against 1,089 ms. A
mixed step makes two launch bound calls and both stretch under AR activity; a mixed first hop
cohort of 10 to 16 rows is 1,168 ms p50 in the ledger against 1,028 ms for main's hop steps of
that size (log gaps, 4f main boot). That is the first audio p95 and the latency p99 of 4f.

**What this says.** The cache removes device work from a call whose cost is its launches, so
little of it reaches the clock: all of it on long requests (section 2: 971 to 258 ms at a 6,850
frame window), a third of the passes on this corpus. The cost that is left, on main and on the
branch alike, is about 17,000 eager launches per Flow call, each holding the GIL the AR
scheduler also needs. The breakable graph over the cached step (S3) removes exactly that, and the
cached step is what makes it capturable, since its shapes are bounded by the new frames.

## 5. Owed before S1 can be a PR

1. The append change of section 3, then the first hop against main again.
2. Serving runs on this branch only, against the main boots on disk (b5c3b44aa, memory fraction
   0.3): seeded c1 for continuity and first audio, unseeded c8 and c16 with the fallback count,
   one starved budget run that must complete on the fallback.
