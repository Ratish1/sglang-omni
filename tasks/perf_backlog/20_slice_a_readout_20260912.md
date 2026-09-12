# 20. Slice A readout, the token ids staged before the predictor, 2026-09-12

Archive `artifacts/qwen3-tts-stage-ids-early-51c5bb064-20260911-compact.tar.gz`, H100, GPU 0,
sglang 0.5.19. A is upstream main `b2cc93b0a`, B is `51c5bb064`, two commits on it. S3 (#2108)
was not in either arm: it merged as `96c727af6` after the run, and the branch now carries it
through merge `96d0d9cbb`, which touches no line of the slice (S3 is in sglang_model.py, the
slice in model_runner.py). The pair measures the slice alone on its own base, which is what
the protocol asks. Head after the run: `b26971164`, the merge plus the removal of the base
gate test below.

Tool note: `perfkit.py` read the row count of a step from the ids copy after the predictor
replay, so every B step was labelled rows 0 and the timeline read the queue delay of the
backbone launch as idle. Fixed on the analysis branch (`0c9940304`): the copy is taken from
the before span first, and the idle before the backbone is read against the previous step's
device end. Every figure below is from the fixed tool on the archived pickles.

## 1. Mechanism gate, passed

Per decode step, p50, fixed tool, the census boots of each arm and the E2 boot of B:

| | A c1 | B c1 | B c1 (E2) | A c16, 16 rows | B c16, 16 rows | B c16, 16 rows (E2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step wall ms | 8.039 | 6.251 | 6.232 | 8.573 | 6.778 | 6.740 |
| device idle in the step ms | 2.749 | 0.982 | 0.970 | 2.836 | 1.041 | 0.995 |
| predictor wall (busy) ms | 4.021 (3.350) | 4.022 (3.351) | 4.015 (3.342) | 4.266 (3.589) | 4.265 (3.587) | 4.267 (3.588) |
| kernels per replay | 1062 | 1062 | 1062 | 1062 | 1062 | 1062 |

The E2 timeline of the median B c1 step: the previous step's predictor is still running for
1.95 ms when the backbone launch is issued, the backbone runs on the device at 1.95 to 3.99,
the predictor launch is issued at 2.19 and the replay runs at 3.99 to 8.03, and the next
backbone launch is issued at 6.23, 1.8 ms before the replay ends. The host waits in
`event.synchronize` 1.26 ms per step at c1 and 0.81 ms at 16 rows, so the host has slack and
the step is device bound: backbone 1.8, eager sampling kernels 0.12, predictor 4.02 with its
0.67 ms of in graph gaps, small kernels and gaps 0.3. The gap before the predictor is 0.13 ms
now, not 0.52: the launch is submitted while the device is still on the backbone. Slice D's
value at c1 is that 0.13 ms; its value at c16 is read below.

Bit identity: seeded c1, 1088 of 1088 WAV hashes equal between A and B. Census equal in every
family at 1 and 16 rows.

## 2. Full corpus, unseeded, warmup 1, one boot per arm and point, c16 twice

| arm, boot | c | req/s | audio s/s | median s | p95 s | p99 s | RTF mean | WER | errors (of 11943 words) | similarity | GPU 0 peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A r1 | 1 | 2.269 | 9.432 | 0.429 | 0.632 | 0.735 | 0.1078 | 1.013% | 121 | 71.166 | 71269 |
| B r1 | 1 | 2.589 | 10.722 | 0.377 | 0.550 | 0.657 | 0.0950 | 1.038% | 124 | 71.200 | 71197 |
| A r1 | 16 | 15.297 | 63.527 | 1.023 | 1.478 | 1.776 | 0.2550 | 0.963% | 115 | 71.325 | 76007 |
| B r1 | 16 | 15.451 | 66.334 | 0.987 | 1.421 | 1.632 | 0.2485 | 1.482% | 177, 56 in one 164 s output | 71.193 | 78747 |
| A r2 | 16 | 15.216 | 65.654 | 0.999 | 1.445 | 1.643 | 0.2488 | 1.264% | 151, one long output | 71.480 | 77583 |
| B r2 | 16 | 15.927 | 65.759 | 0.979 | 1.393 | 1.587 | 0.2489 | 1.089% | 130 | 71.249 | 78197 |

The seeded c1 pass gives the same speed as the unseeded one (A 2.283 against 2.269 req/s,
B 2.579 against 2.589, medians within 4 ms), so for a bit exact slice the seeded c1 boot can
carry both the identity gate and the c1 speed point.

c1: req/s +14.1 percent, median −12.1, p95 −13.0, p99 −10.6, RTF −11.9. c16: req/s +1.0 and
+4.7 percent in the two boots, median −3.5 and −2.0, p95 −3.9 and −3.6, p99 −8.1 and −3.4.

### Quality is the draw, not the code

The change is bit exact, so a quality difference has no mechanism; the unseeded numbers are
different random draws. Per sample similarity, 1088 samples, standard deviation 7.6 to 8.0,
standard error of the corpus mean 0.24:

| pair | similarity delta | 95 percent interval | error delta | 95 percent interval |
| --- | ---: | ---: | ---: | ---: |
| A r1 to A r2, same code | +0.155 | −0.21 to +0.52 | +36 | +2 to +90 |
| B r1 to B r2, same code | +0.055 | −0.29 to +0.41 | −47 | −168 to +20 |
| A r1 to B r1 | −0.132 | −0.48 to +0.21 | +62 | −3 to +187 |
| A r2 to B r2 | −0.231 | −0.57 to +0.10 | −21 | −74 to +13 |
| A c1 to B c1 | +0.034 | −0.32 to +0.36 | +3 | −14 to +20 |

The same code moves as much between two boots as A and B differ. Each of the 164 s and long
outputs is a single draw, one on B r1 and one on A r2. Nothing here is a regression.

### Memory

The level after readiness is 69327 MiB on every boot. The peak above it is the vocoder's
dynamic part, 6.7 and 8.3 GB on A's two c16 boots and 9.4 and 8.9 GB on B's. The one boot of
each arm with a long output is not the higher one on A, so the composition of the vocoder's
batches, which the talker's faster finishes change, is the likelier cause than the outlier. A
per process reading on the box (`nvidia-smi --query-compute-apps`) during a c16 run settles
it; the talker's pool is fixed at readiness either way.

## 3. What c16 is made of now

At c16 the batch is rarely full: 28 of 1077 decode steps in B's census window ran 16 rows (61
of 992 on A). The rest are churn steps, a request finished or joined, and their p50 wall on B
is 8.4 to 12.8 ms with 2.7 to 7.1 ms idle (A: 9.2 to 13.2 with 3.8 to 7.3). That is why the
full step's −21 percent turned into +1 to +5 percent of c16 throughput.

The median rows 8 step on B: 14.6 ms wall, 8.9 ms with no scheduler launched kernel on the
device, of which 1.7 ms is the reference audio encoder's convolutions from the request build
threads on the same stream, and 7.2 ms is idle. The predictor launch is issued 6.2 ms after the
backbone launch (1.8 ms on a full step). The host frames of these steps at p90 name the two
synchronizations plan 18 assigned to slices B and C, both now blocking behind the queued
predictor instead of an empty device:

| frame | p50 us | p90 us | on main (readout 17, p90) |
| --- | ---: | ---: | ---: |
| `sglang_model.py: prepare_decode_buffers` (the restage, six pageable H2D copies) | 19 | 3149 | 293 |
| `request_builders.py: apply_sglang_qwen3_tts_result` (the finish copy `.cpu()`) | 0 | 2989 | 100 |

Slice C (the restage into pinned buffers, non blocking) and slice B (the finish copy with an
event) are therefore next, in that order, and they are worth up to 3 ms on the steps that
have them, which at c16 is most steps. The eager sampling frame doubles on churn steps (616 us
against 277 on full steps) and the graph launch submission grows (2.0 against 1.7 ms), host
contention from the request build threads is the suspect; slice D and the sampler path come
after B and C.

## 4. Suites

The focus file 4 passed, the Qwen3-TTS directory 457 passed and 1 failed:
`test_qwen3_tts_public_penalty_disables_async_lookahead` asserted the base gate's answer on
this runner, which is now sync only by declaration and covered by
`test_lookahead_is_never_eligible`. Removed in `b26971164`. The broad suites were stopped on
the box; CI runs them on the PR.

## 5. Streaming

Not run: the CI layout needs GPU 1 and it was held by another user's process for the whole
session (76 GB, 81 percent). Owed when GPU 1 is free, or as a one worker pair if it stays
busy, since the delta is the measurement.

## 7. E3, the seeded c16 pair, 2026-09-12

Archive `qwen3-tts-stage-ids-early-b26971164-e3-upstreamA-20260912.tar.gz`. A is upstream main
`9147eb5b3`, B is `b26971164`; the one upstream commit between B's merge base and A is the
Fun-CosyVoice3 flow coalescing (#1899), not on this path. Seed 1234, warmup 1, c16, one boot
per arm, 1088 of 1088 completed on both.

| | A | B | paired delta, 95 percent interval |
| --- | ---: | ---: | ---: |
| WAV hashes equal | | | 81 of 1088 |
| similarity | 71.2888 | 71.2892 | +0.0004, −0.28 to +0.28 |
| errors | 114 | 125 | +11, −1 to +23, 30 samples changed |
| req/s | 16.662 | 16.863 | |
| median, p95 | 0.944, 1.361 s | 0.940, 1.315 s | |

Every equal WAV has a similarity delta of exactly zero; over the differing ones the per sample
deltas have a standard deviation of 4.8 and a mean of +0.0004. So at c16 the seeded draws
still diverge between arms, the sampled tokens depend on the batch composition (the seeded c1
pair is bit identical, so the kernels are), and once one token differs a sample's similarity
moves by about 5 points either way. The corpus mean does not move. The streak of six negative
unseeded pairs is settled as draws. Protocol consequence: the hash identity gate stays at c1,
and a bit exact slice runs no c16 quality comparison.

Streaming: the archive carries three passes of A only, two workers on GPUs 0 and 1: 19.5 to
20.4 req/s, TTFC mean 0.126 to 0.182 s, inter chunk mean 0.078 to 0.081 s, continuity 100
percent, 3264 of 3264. B's three passes are owed for the delta table.

Per process memory in the streaming layout: the engines 71.0 and 69.0 GB, the vocoders 3.6
and 3.8 GB. In the single process layout the one process peaks at 77.4 GB, so the 8 GB above
readiness in section 2 is the vocoder and preprocessing stages living in the same process,
not the talker.

## 6. Protocol, cheaper from here

- A stacked slice reuses the previous slice's B boots as its A when the base is the same; here
  it was not (S3's B sat on `80b5aaed7`), so both arms ran.
- One seeded c1 boot per arm, warmup 1, gives identity and the c1 speed; the unseeded c1 boot
  is dropped.
- For a bit exact slice the c16 quality numbers are the draw; the c16 boot is for speed and
  the band check comes free from its WAVs.
- The census boot carries `SGLANG_TORCH_PROFILER_WITH_STACK=1` so census and the E2 windows
  are one boot.
- A bit exact stacked slice is then four boots on B: census with stacks, seeded c1, c16,
  streaming. The broad suites run in CI.
