# Qwen3-TTS slices: principles, workload matrix, audit of the existing code

Base: upstream 144bd6399 (main moved only in OmniTyper since). Card: moss RTX 4090 D
(sm89). Checkpoint: Qwen3-TTS-12Hz-1.7B-Base. Measurement contract and step ledger:
`../SLICES_PLAN.md` section 1, `../scripts/step_ledger.py`. Slice files in this folder:

| file | slice | state |
| --- | --- | --- |
| `P1_PREDICTOR_PAIR_PASS.md` | one two-token predictor pass instead of two one-token passes | design ready, numerics gate open |
| `V1_VOCODER_CHANNELS_LAST.md` | vocoder conv chain kept channels-last | experiments V-e0 to V-e4 open |
| `P2_SAMPLER_LATENCY.md` | seeded top-k sampler latency | attribution P2-e1 open |
| `V2_VOCODER_KERNEL_COUNT.md` | vocoder launches that are not convs | research only |
| `TESTING.md` | every box step, exact commands | |

## 1. Principles every slice follows

1. A mechanism comes from the model's structure or the hardware's documented behavior,
   never from one measured shape. P1 is the same math in fewer passes. V1 presents
   tensor-core convs in the layout their kernels consume. P2 changes only how the work
   maps onto threads, never the values.
2. No constant tuned to one card or one workload. Where a choice has to depend on the
   device, it is measured on the device at startup, or it is justified on both the 4090
   and the H100.
3. A slice is measured over every shape it touches (all captured keys, all batch
   buckets), then over the workload matrix (section 2). The micro numbers predict a
   delta for each matrix cell before the A/B runs; the A/B checks the prediction.
4. Numerics are judged against a common truth (an fp32 run of the same computation), not
   against today's bf16 bits, unless the contract is bit identity (P2). The pass rule is
   "no farther from the truth than today's path", plus the census.
5. No regression in any matrix cell beyond the A/A band (about 2 percent, from the three
   unprofiled decode b16 runs of run01).

## 2. Workload matrix (one server boot per arm, all cells in that boot)

| cell | driver | concurrency / batch | what it exercises |
| --- | --- | --- | --- |
| S1 seed-tts voice clone, streaming | `benchmark_tts_seedtts.py --stream` | 1, 4, 16, 32 | TTFC, inter-chunk, RTF; 32 queues past the default `max_running_requests=16` |
| S2 long form, streaming | same, `--meta` a joined-text list (TESTING.md) | 1, 16 | steady decode, warm widths, long codec state |
| S3 non-streaming | same, no `--stream` | 1, 16 | full `tokenizer.decode` path (`streaming_vocoder.py:2931`) |
| L1 ledger captures | `scripts/profile_workloads.py` | prefill b1, b8 (1 token); decode b1, b4, b16 (max tokens) | step wall, busy, bubbles, contention per step class |
| Q1 census | `benchmark_tts_seedtts.py` full corpus, WER and SIM | 1, 16 | quality |

Client reads: TTFC p50/p95, inter-chunk p50/p95, RTF, audio s/s, req/s. Ledger reads:
section 1 of SLICES_PLAN.md, plus vocoder graph replays per step grouped by node count
(each captured key has its own node count, so the histogram names the keys).

## 3. Audit of the existing code

Read in full for this audit: `sglang_model.py` (predictor, graphs, prompt builder),
`model_runner.py`, `predictor_kernels.py`, `sampling_kernels.py`, `vocoder_kernels.py`,
`incremental_codec.py`, `incremental_codec_cuda_graph.py`, `codec_state_arena.py`,
`streaming_vocoder.py`, `stages.py`, `config.py`, and qwen-tts 0.1.1
`modeling_qwen3_tts_tokenizer_v2.py` (the decoder modules). Each row: what the code
does, why it is shape, workload or hardware specific, and what happens to it.

| # | where | what | specific to | disposition |
| --- | --- | --- | --- | --- |
| A1 | `vocoder_kernels.py:54-56,157-163`, `stages.py:298` | fused SnakeBeta exists, is off by default, and its envelope is `C in {1536,768,384,192,96}`, `B <= 8`, NCL contiguous | checkpoint channel list, a batch cap with no kernel reason (grid is `B*C`), one layout | V2: measure it on; if kept, the envelope becomes layout- and size-generic. V1 must not depend on it |
| A2 | `streaming_vocoder.py:976,1021` | only the steady stride (8) is torch.compiled; every other captured width replays eager kernels | the default chunk schedule | V2: measure compile coverage (all widths, or dynamic width) |
| A3 | `streaming_vocoder.py:2593` | follow-up cohorts group by exact decode width, so jittered widths split into separate replays | arrival timing | V2: measure replays per step and rows per replay (ledger histogram) before any design |
| A4 | `streaming_vocoder.py:54,902`, `stages.py:292-294` | warm graph buckets (1,2,4,8), follow-up cohort cap 8, 2 workers | a concurrency range | record; the matrix shows where c16 and c32 split cohorts |
| A5 | `codec_state_arena.py:187-238`, `incremental_codec.py:127-131` | gather and scatter run one `index_select` / `index_copy_` per state buffer (one per conv and transconv, plus 2 per transformer layer; V1-e0 prints the count from `state_spec`), and each conv clones its history | none, a launch count that scales with depth | V2: count them in the replay attribution |
| A6 | `streaming_vocoder.py:667-672` | bootstrap suppression switches off above 24 live streams, from a win to 10 RPS and a regression at 20 RPS | one setup's measured load | record, not a perf slice here; the cause of the 20 RPS regression is unmeasured |
| A7 | `streaming_vocoder.py:50-52` | window widths capped at 64, "the measured cap" | the card it was measured on | V1-e1 times every window width on sm89; recheck on H100 |
| A8 | `stages.py:283,291,293`, `streaming_vocoder.py:521-526` | batch collection waits 2 ms (initial), 4 ms (follow-up, factory) vs 1 ms (class default) | time constants | record |
| A9 | `sampling_kernels.py:318,699` | the sampler hard-codes vocab 2048 and `num_warps=8`; its Gumbel is fp64 by contract (parity with SGLang `multinomial_with_seed`) | the checkpoint vocab; fp64 rate of the card (1/64 on sm89, 1/2 on H100) | P2 |
| A10 | `streaming_vocoder.py:55` | codebook size 2048 literal instead of the decoder config | the checkpoint | record |
| A11 | `incremental_codec_cuda_graph.py:94`, `stages.py:305` | capture keeps 3.0 GiB free, absolute | card size | record |
| A12 | `sglang_model.py:1567-1579` | the first two predictor passes run serially, one token each | none | P1 |
| A13 | `incremental_codec.py:122,144` | convs run on NCL tensors; cuDNN converts input and weight to NHWC and the output back on every tensor-core call (bench 02) | none, a layout mismatch | V1 |

Consequences for the slices:

- V1 keeps the existing SnakeBeta modules working in either layout (their elementwise
  ops keep the input's strides), so it does not inherit A1's envelope.
- No slice changes A4, A6, A8, A10, A11. They are listed so the matrix reads them: a
  cell that moves only because a cap was crossed is attributed to the cap, not the slice.
- The eager vs compiled split of A2 means today's vocoder output already depends on
  which width a chunk lands on. V1-e2 measures that spread; it is the tolerance V1's
  numerics are judged against.

## 4. Order

1. P1 code (no dependency on the V or P2 experiments) while one box run executes V1-e0
   to V1-e4 and P2-e1 (TESTING.md sections 2 and 3).
2. P1 A/B over the matrix, then the census.
3. V1 code from the V1-e results; A/B with P1's B as A.
4. P2 design from P2-e1; A/B.
5. V2 research from V1-e1's attribution.
6. H100 confirmation of each before a PR claims time. One PR per slice from upstream main.
