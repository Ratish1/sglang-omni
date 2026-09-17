# Readout 04: plan 13, the AR KV pool ceiling, on the 4090 (2026-09-17)

Branch `slice/cosyvoice-b1-kv-pool-ceiling` at d3d68de4b (the head differs by a cast only fixup,
no behaviour change). A is upstream main 3f4226937, the branch base. Moss box, RTX 4090 D 24 GB,
card 4, one arm at a time, default launch, English SeedTTS, streaming, warmup 1. The identity
runner always seeds, so every run here carried seed 1234. Run directories under
`/workspace/sglang-omni/.tmp/out/g1-*-20260917T18*` and `T19*`.

## Verdict

No regression from the ceiling. With the runaway variable removed (every request capped at 600
tokens, 24 s), the branch is at or above main under a 2 GiB byte budget on every read, and the
differences sit inside main's own boot to boot spread. The pool never acted on any arm: zero
retract lines, zero rejections, cached token sums equal. On 24 GB the ceiling turns main's c8
failure into none and leaves 10.2 GB free after the pool instead of 1.9.

The uncapped c16 pair that preceded it read the branch 7 percent behind in the mean and 25 percent
in the tail. That pair carried four runaways on the branch's arm against three on main's, and a
runaway widens every step for the 50 to 60 s it lives; the seed does not pair lengths at c16
(91 of 1,088 identical). The capped pair is the measurement; the uncapped one is recorded as the
reason the protocol now caps when the question is not about runaways.

## Boot

| arm | pool tokens | pool | free after pool | startup line |
|---|---|---|---|---|
| main | 852,866 | 9.76 GB | 1.93 GB | none |
| branch | 131,072 | 1.50 GiB | 10.19 GB | "Fun-CosyVoice3 KV pool holds 131072 tokens, 1.50 GiB, against a configured maximum demand of 131072 (32 running x 4096 context), mem_fraction_static 0.846" |
| main + kv_cache_bytes 2 GiB | 174,762 | 2.00 GB | 9.72 GB | byte budget line |

SGLang prints the pool as its "KV Cache is allocated" line; there is no `max_total_num_tokens=`
field in the serve log.

## c1 identity, 16 samples, seed 1234

G1 gate, control boot on main: pass, 12 gated, 12 identical, 0 differing; 4 excluded as unstable
by the control, the usual four.

## c8, 32 samples

| | main | branch |
|---|---|---|
| completed / failed | 31 / 1 (cuBLAS handle in the ONNX tokenizer) | 32 / 0 |
| req/s | 2.502 | 2.610 |
| audio s/s | 11.45 | 12.00 |
| first audio mean / p95 s | 1.472 / 2.682 | 1.424 / 2.630 |
| RTF mean / p99 | 0.689 / 1.419 | 0.684 / 1.366 |
| C50 / C100 | 77.4 / 77.4 | 78.1 / 78.1 |

## c16, whole split, capped at 600 tokens, alternating boots

| boot | arm | req/s | audio s/s | RTF mean | RTF p99 | first audio mean / p95 s | inter chunk s | C50 / C100 | latency mean / p95 s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | main + 2 GiB | 3.520 | 16.96 | 1.010 | 1.955 | 1.948 / 2.492 | 1.296 | 13.9 / 18.8 | 4.53 / 5.58 |
| 2 | branch | 3.645 | 17.25 | 0.985 | 1.885 | 1.951 / 2.655 | 1.232 | 18.8 / 21.3 | 4.37 / 5.73 |
| 3 | main + 2 GiB | 3.487 | 16.60 | 1.032 | 2.115 | 2.060 / 2.727 | 1.265 | 16.0 / 22.2 | 4.56 / 5.97 |

Main's own spread, boot 1 against boot 3: req/s 0.9 percent, RTF p99 8 percent, first audio p95
9 percent. Every branch read is inside or beyond that band. Samples at the cap: 9, 5, 5. A fourth
boot (branch) was aborted at 1,067 of 1,088 by request; the branch's own spread is therefore not
measured.

Continuity at c16 is 14 to 22 percent on both arms: the 4090 is over capacity at c16 for this
model on either tree, which is the vocoder's step time, not this change.

## c16, whole split, uncapped, one pair (recorded, superseded by the capped pair)

| | main + 2 GiB | branch |
|---|---|---|
| req/s | 3.070 | 2.863 |
| RTF mean / p99 | 1.137 / 2.478 | 1.216 / 3.101 |
| first audio mean / p95 s | 2.346 / 4.298 | 2.527 / 4.712 |
| runaways above 60 s | 3 | 4 |
| total audio s | 5,395.8 | 5,468.3 |
| identical durations | 91 of 1,088 | |

Runaway excluded subsets moved the same way, because the runaways tax the requests served beside
them; that is why the capped pair was run.

## Logs, every arm and boot

"KV cache pool is full": 0. "Request requires more tokens" and input length rejections: 0.
"retract": 0. Prefill cached token sums 12.4k to 12.7k against 155k new tokens on every c16 boot.

## Open

- The H100 or H200 pair, main against the branch, throughput within 2 percent: predicted identical
  by the code (no decision reads the pool beyond 131,072 for this model), not measured.
- A second branch boot at c16 capped, for the branch's own spread.
- The benchmark protocol for A/B pairs whose question is not runaways: cap `max_new_tokens` on
  both arms (`MAX_NEW_TOKENS` in the G1 runner), and always report per sample pairing.
