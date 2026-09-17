# Slices: what ships, in what order, gated on what

One row per shippable change. A slice gets its own document when it is next and
its design is settled by a read of the tree; the rest carry their plan row and
the measurement that has to come first. Roadmap: `../ROADMAP_20260915.md`.

## Landed

| slice | document | head | gate | result |
|---|---|---|---|---|
| 1.0 | `../stage2/README.md` | analysis branch, no runtime change | G0 numerics | passed 2026-09-16 (H100), and its correctness half re-passed on the 4090 |
| 1.1 | `11_1_state_and_constants.md` | `slice/cosyvoice-1-1-state-constants` `c652aa5e` | G1 byte identity | passed 2026-09-17, 14 of 14 gated samples, 196 unit tests |

## Next

| slice | document | depends on | gate | blocked by |
|---|---|---|---|---|
| 1.2 cached hop call | `11_2_cached_hop_call.md` | 1.1 | G2 | P2 (quality threshold) and P3 (memory budget); P3 needs an H100 census |
| 2.1 hop CUDA graphs | `21_hop_cuda_graphs.md` | 1.2 | replay bit identical, then census | 1.2 landing; its own trigger is already measured |
| D1 prompt padding | `11_D1_prompt_padding.md` | none technically, the plan sequences it after 1.2 | its own A/B on WER, SIM, first audio | P4 |

G0 moved 2.1 up. The cached hop call is flat at 138 to 143 ms whether it computes
400 or 1,600 new frames, which is the eager launch floor of 19,072 launches per
Flow call. The cache removes a hop's device work and not its launches, so the
hop is launch bound the moment 1.2 lands and graphs are the next lever rather
than a deferred one.

## Held, with what would release them

| row | item | what has to happen first |
|---|---|---|
| 1.3 | remove the 2 syncs per Flow call | roadmap 0.2: the per call sync ledger on the stack head still has to name them. The origin the earlier ledger gave does not apply to this checkpoint |
| 2.2 | graphs for the bidirectional call, stream finals and buffered | 2.1 first; the two share the capture and replay seam |
| 2.3 | batched and graphed HiFT | re-measure HiFT's share on the stack head after 1.2. Today's number, 16 to 30 percent of the vocoder thread, predates the stack. #1883 reports HiFT output depends on batch shape, so batching it is an output change and needs its own gate |
| 2.4 | one slack order for unstarted streams | bites only above 16 runnable streams, so it needs a c32 A/B that 24 GB cannot boot |
| 2.5 | AR: one host snapshot per step | independent of the Flow work; gate is seeded c1 identical codes |
| 2.6 | preprocessing: one decoded reference | independent; gate is exact reference tokens, features and speaker embedding. The G1 control found two of sixteen requests that do not reproduce across boots, both on one reference clip, which is a question for this row |
| 3.x | fusions and kernels | only kernel time counts once the hop is graphed, so every 3.x is sized from a graphed trace, after 2.1 |

## Rules these slices follow

- One measured change per PR, on current upstream main, with the run that
  validates it in the same PR.
- Every slice states its gate before it runs, and the gate names what a 4090
  settles and what it does not. Byte identity and launch counts transfer between
  cards; wall times, SNR thresholds and memory budgets do not.
- No constant that no measurement pins. A budget or a bucket table is either
  derived from a census in this directory or it is not in the diff.
