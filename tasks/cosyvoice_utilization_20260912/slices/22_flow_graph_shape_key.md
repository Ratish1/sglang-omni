# Flow CUDA graphs: one shape knob, generated, padded up

Replaces roadmap rows 2.1 and 2.2 with one slice, and reorders 3.3 in front of
both. Trees read: upstream main `27a8293c`, pinned SGLang `v0.5.19`,
`sgl-project/sglang-omni#1861` (the PR that added the table).

## What is wrong, stated as the rule it breaks

SGLang captures graphs by a rule, and the rule is written down in its own source:
"For tc_piecewise prefill, bs carries the captured token count (**one shape knob
per phase**)" (`arg_groups/cuda_graph_hook.py:495-496`). Three properties follow
from it.

| SGLang | where | Fun-CosyVoice3 Flow |
|---|---|---|
| one shape dimension per table; everything else variable lives in static buffers | decode keys on batch size, prefill on token count (`cuda_graph_hook.py:512-544`, `:492-509`) | two, `(batch_size, mel_frames)` (`config.py:19-75`) |
| the table is generated from a max by a monotone rule, fine at the bottom and coarse at the top, always containing the max | `[1,2,4,8,12] + range(16,257,8) + range(272,512,16) + range(512,max_bs+1,32)` (`cuda_graph_hook.py:523-542`) | 54 pairs enumerated by hand, no max, no rule |
| a size not captured rounds **up** to the next captured one and replays | `bisect_left(capture_bs, raw_bs)` (`cpu_graph_runner.py:911`); `disable_cuda_graph_padding` is the opt-out (`cuda_graph_hook.py:519`) | exact match only, and PR #1861 says "Batch size is never padded"; a miss silently runs eager (`stages.py:475-496`) |

Where the 54 numbers came from is not a matter of inference. PR #1861: "Capture a
fixed, **trace-selected** K32 table of exact `(batch_size, q16_mel_frames)`
solver shapes at startup." It also shipped "off by default"; `config.py:160` now
has `enable_flow_cuda_graph=True`.

## What it costs and what it buys, measured 2026-09-17

4090 D 23.52 GiB, card 4 or 5, head `84e745ea`, 32 English SeedTTS samples
buffered and 16 streaming, one arm at a time, KV pool pinned at 2 GiB on every
arm so the comparison is not a memory comparison.

| | graphs on | graphs off | |
|---|---|---|---|
| buffered c8, req/s | 4.058 | 2.437 | **1.67x** |
| buffered c8, RTF p99 | 0.9999 | 1.4955 | |
| buffered c1, RTF mean | 0.1558 | 0.1639 | 1.05x |
| streaming c1, RTF mean | 0.2441 | 0.2350 | 0.96x, no effect |
| memory before the AR engine loads | 13.41 GB free | 21.07 GB free | **7.66 GB** |

Three readings, and all three come from the same defect.

- **Graphing this solver is the largest win measured anywhere in this task.**
  1.67x on buffered c8 dwarfs the 3 percent the hop cache returned. The solver is
  launch bound and a graph is what fixes a launch bound solver.
- **Streaming gets none of it.** `generate_flow` consults the runner only when
  the call is neither streaming nor a chunk: `if streaming or not finalize or
  flow.cuda_graph_runner is None` takes the eager solve (`stages.py:656`), and
  hops and stream finals both leave through `generate_flow_packed`
  (`:797`, `:814-816`). The streaming path never had a graph.
- **The gain depends on whether the corpus matches the table.** 1.67x at c8 and
  1.05x at c1. A launch bound solver should gain *more* at batch 1, where there
  is less device work to hide the launches behind, so 1.05x is not the solver
  running out of headroom; it is most c1 requests missing a table that holds
  eight frame values for batch 1 and running eager. The number a feature returns
  should not be a function of which corpus was traced to build it.

## The shape distribution, measured rather than assumed

`log_flow_solve` (branch `analysis/cosyvoice-flow-shapes`, one line per solve
naming the path, the rows, the width, the total and the row lengths) and
`../stage2/flow_shapes.py` aggregate it. Streaming c8, 32 English samples,
4090 D card 6, KV pool pinned at 1 GiB:

```
21 Flow solves
  packed_hop        13   61.9%
  packed_final       8   38.1%

  path           calls   rows  width p50  width max  total p50  total max   keys
  packed_hop        13   1-8         400        650       2050       3500     11
  packed_final       8   1-8         532        784       2900       4518      8
```

- **No streaming call is graph eligible.** Zero `padded` and zero `graph` solves
  in the whole run, which is the code trace confirmed at runtime.
- **The 2-D key cannot cover streaming.** 13 hop calls carry 11 distinct
  `(rows, width)` keys, and 8 final calls carry 8. A table of exact pairs would
  need one entry per call, which is why the streaming path was never given one.
- **The derived buckets cover every call.** The 17 buckets that follow from the
  chunk quantum, the admission ceiling and a 25 percent padding bound hold 100
  percent of these calls, at 12.7 percent padding on hops and 19.6 percent on
  finals, with none above the ceiling.

The run also corrected an architectural claim made earlier in this document.
Hop lengths are chunk quantised, as derived: the run's hops are 100, 250, 250
frames. **Final lengths are not**: 106, 316, 312, 352. A stream final covers the
whole token history and the AR stops where it stops, so nothing rounds it to the
chunk. That is precisely why the key needs round-up rather than exact match, and
it is the one thing a derivation from the hop schedule alone would have missed.

## The first principles fix

### Why the key has two dimensions today, and what removes one

The cost of a Flow call, by layout:

| layout | per token modules | attention | shape knobs |
|---|---|---|---|
| padded, `solve_flow_euler` | rows x width | rows x width^2 | two, irreducibly: the layout materialises `rows x width` frames whatever the real lengths are |
| packed, `solve_flow_euler_packed` | **total frames** | rows x width^2, because `RowAttention` scatters back to the padded layout (`packed_dit.py:103-109`) | two |
| packed with ragged attention | total frames | total frames | **one** |

So the second knob is not intrinsic to the model. It is the padded attention. The
same is true upstream: SGLang can key decode graphs on batch size alone precisely
because its attention is varlen over a page table, so sequence length left the key
and moved into static buffers.

The ragged kernel is not hypothetical here. The hop cache path already attends
this way, FA3 with `cu_seqlens_q`, `cache_seqlens` at the chunk end and
`causal=False`, and the chunk causal semantics are already expressed as segments
by `hop_layout` (`flow_hop_cache.py:73-98`). E6 qualified that kernel on real
activations at 53.3 to 54.0 dB against a float32 SDPA truth, and G0 measured the
two kernels as equally accurate, differing only in per sample rounding.

### The slice

1. **Ragged attention in the packed path.** Replace `RowAttention`'s
   scatter, SDPA under a `(rows, width, width)` mask, gather with one varlen FA3
   call over the packed sequence, the chunk causal mask expressed as the segment
   layout that `hop_layout` already produces. This deletes the padded batch cost
   the task has been circling since `slices/02_vocoder_step_cost.md`, and it is
   what makes step 2 possible.
2. **One shape knob, generated.** The capture table becomes a 1-D list of total
   frame buckets produced by a rule from one maximum, in SGLang's shape:
   fine at the bottom, coarse at the top, always containing the max. A call pads
   its packed sequence up to the next bucket with a sentinel row and replays;
   nothing misses, nothing is hand-picked, and `(batch_size, mel_frames)`
   disappears along with `FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES`.
3. **One table for all three callers.** Buffered `inference`, stream finals
   `inference_leftover` and streaming hops `inference_causal` all reach
   `solve_flow_euler_packed`, so all three replay from the same table. Streaming
   gets the win buffered already has.

The maximum is a real bound, not a tuned constant: a Flow call can never exceed
`flow_batch_admission_frames` (`config.py:153`, the scheduler's own admission
budget), and the noise ceiling is 15,000 frames (`flow_matching.py:199-200`,
checked at `stages.py:589-593`).

### What this does to memory

The 7.66 GB is 54 captures whose static buffers are sized `rows x width` and
whose private pools hold the activations of a padded batch. A 1-D table pays for
`total frames` per bucket, with no padding to the widest row, and holds roughly a
dozen buckets rather than 54 points. The exact figure is a measurement this slice
reports, not a number to promise here.

## Gates

1. Ragged attention against padded SDPA on the same real activations, per call,
   the E6 protocol: minimum SNR against a float32 truth, no call below the
   threshold E6 set. Self comparison on one card, so a 4090 settles it.
2. Replay bit identical to the eager packed solve, per bucket, over a schedule
   that reaches every bucket.
3. Buffered and streaming, c1 and c8, against the current default: req/s, RTF
   mean and p99, first audio, plus the memory census at boot. The buffered c8
   arm must not regress against the 1.67x the trace-selected table gets on the
   corpus it was traced from, which is the hardest number in this table to beat
   and the honest bar.

## Ordering

This lands before any further hop cache work. The hop cache removes device work
from a call that is launch bound, which is why it returned 3 percent at c1; the
graph removes the launches. Once the hop is graphed, the cache's device saving
becomes visible, and the two compose: the cached hop computes fewer frames and
replays a smaller bucket.
