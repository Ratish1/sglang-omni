# Slice 1.2: the cached hop call

Plan row: `../plans/11_hop_prefix_cache.md` section 7, slice 1.2. Stacks on slice
1.1 (`c652aa5e`). Trees read: upstream main `27a8293c`, slice 1.1, the G0 script
`../stage2/g0_hop_cache_numerics.py` at the revision that passed, and the pinned
SGLang `v0.5.19`.

## What the read changed about the plan

Four things. Three narrow the slice, one is a correction to section 5.3.

- **Ship the conditioning G0 validated, not the incremental one.** Section 5.3
  says the cached call builds conditioning for new frames only, from tokens
  `[T0-2, T1+3)`. The run that passed does not do that: it packs the whole
  window (`g0_hop_cache_numerics.py:475`), runs `prepare_flow_conditioning` over
  the whole window (:476) and slices rows `[M, M')` out of the padded result
  (:489-494). So the measured 2.1x over the schedule, and the flat 138 to 143 ms
  cached call, already carry full window conditioning. Conditioning is 80 channel
  work over tokens: an embedding lookup, the two pre-lookahead convs, a repeat,
  one noise slice and clone (`stages.py:541-611`). The DiT it feeds is 1,024
  channels, 22 blocks, 10 steps. The part of the call the cache does not remove
  is the cheap part, and it is inside the number we have. 1.2 keeps
  `prepare_flow_conditioning` unchanged and slices its output. Incremental
  conditioning becomes its own slice, with its own gate, if the launch floor
  makes it worth the second receptive field to get wrong.
- **torch.compile never reaches this path.** `compile_dit_backbone` replaces
  `estimator.forward` (`stages.py:1144`). `PackedDiT` never calls `forward`; it
  walks `dit.time_embed`, `dit.input_embed`, `dit.transformer_blocks` directly
  (`packed_dit.py:143-159`). So `enable_dit_torch_compile` changes nothing for
  hops today and nothing for the cache, and 1.2 needs no interaction with it.
- **TensorRT already switches the packed path off, so it switches the cache off
  with it.** `attach_flow_estimator_trt` sets `flow.packed_estimator = None`
  (`stages.py:864`) and `inference_causal` then takes the padded `generate_flow`
  (`stages.py:812-814`). The cache hangs off `packed_estimator`, so "TRT on"
  means "no cache" with no new branch and no new flag.
- **A hop's rows are ragged by construction.** `select_step_participants` groups
  by `next_decode()` and takes up to `max_batch_size` rows
  (`streaming_vocoder.py:317-349`); each row carries its own `token_offset` and
  its own `hop_len` (:421-426). One step therefore mixes rows at different
  absolute frames computing different numbers of new frames. `begin_call` takes
  `(stream, start, end)` per row for exactly this reason.

## The state, and what it costs

Per stream, per CFG lane: a list of pool slots, the frame count they cover, and
the causal conv tails. Per (lane, Euler step): the last 30 rows of each conv's
input (`g0_hop_cache_numerics.py:95-146`).

| item | size | note |
|---|---|---|
| K and V, one frame, one lane | 901,120 B | 220 layers x 16 heads x 64 dims x 2 x 2 B |
| K and V, one mel frame | 1,802,240 B | with the CFG twin |
| conv tails, per stream | 2.46 MB | 2 convs x 30 frames x 1,024 x 2 lanes x 10 steps, fixed |
| G0's pool | 11.5 GiB | 13,700 slots, all used, the plan's arithmetic exactly |

The c16 ledger's hop calls hold 8,350 window frames at p95 and 10,200 at most
(readout 03 section 6), so a cache that holds every live stream's history at c16
is 14.0 GiB at p95 and 17.1 GiB at the peak, derived.

## Design

### The budget and what happens when it runs out

`flow_kv_cache_bytes` on the vocoder factory args, next to
`flow_batch_admission_frames`. Slots are `bytes // 901,120`. The pool is built in
`create_vocoder_executor` (`stages.py:2083-2096`), which runs before the AR
engine sizes its own pool from free memory: the stage list builds the vocoder
first and says so (`config.py:144-146`). So the bytes named here come out of the
AR pool, visibly, rather than out of a later allocation failure.

A stream that cannot get slots for its next hop releases whatever it holds and
runs today's `hop_batch` for the rest of its life, which returns today's output.
A counter records it. No stream is ever refused and no request changes shape
because the pool is full.

The default is decision P3 and is not settled here. It needs the V2 census (live
cached frames at c16 against the AR pool's actual need) and that census needs a
card that can hold both: a 24 GB 4090 boots this model with about 1.5 GB free
after graph capture, so it can settle c1 and c4 and cannot settle c16.

### The cached hop

Everything below is the G0 configuration, unchanged, moved behind the serving
seams. It is the configuration E6 qualified on real activations and the one G0
passed on: FA3, page size 1, one query segment per (row, chunk), `cache_seqlens`
at the chunk end, `causal=False`.

```text
run_step, causal_window                     streaming_vocoder.py:392-433
  split participants
    row holds a cache, or its slots allocate  -> cached
    otherwise                                 -> today's hop_batch
  cached rows, one call:
    spans: (stream, M, M') per row, M' = 2(P' + offset + hop), first hop M = 0
    pack_flow_inputs over the whole window    stages.py:151        unchanged
    prepare_flow_conditioning, finalize=False stages.py:530        unchanged
      assert mel_lengths[row] == M'
    begin_call(spans)                         flow_hop_cache.py    NEW
      per row per lane: alloc M' - M slots, append to the lane's slot list
      segments: one per (lane row, chunk) of [M, M')
        cache_seqlens = chunk end, cu_seqlens_q the query offsets
      page_table = each segment's lane slot list, padded to M'
    slice conditioning rows to [M, M')        noise, cond, mu
    solve_flow_euler_packed over the new frames only     packed_dit.py:191
      CachedDiT(PackedDiT)                    flow_hop_cache.py    NEW
        _rope        absolute positions M + j
        _conv_pos_embed  concat(tails, new), valid conv, tails updated
        row_attention    store_cache then flash_attn_with_kvcache through
                         the page table
      every other module, the Euler update and the CFG combine unchanged
    emit frames [max(M, 2P'), M')             follow-up hops: all of them
  per row: hift_delta over the whole mel history  stages.py:1563   unchanged
  offset += hop; hop = min(100, 2 hop)        streaming_vocoder.py:421-426

leftover                                      streaming_vocoder.py:368-391
  release the stream's cache first; the final call is bidirectional over the
  whole history and cannot read it (plan H9)

release_stream_resources                      streaming_vocoder.py:503-510
  release the cache handle; reached from completion, abort, step failure and
  scheduler stop (scheduling/streaming_vocoder.py:147-164, 259-263, 509-527)
```

The emitted slice needs no arithmetic change downstream. Today
`split_generated_mels` strips the prompt (`stages.py:718-737`) and `run_step`
then drops `token_offset * mel_ratio` frames (`streaming_vocoder.py:412`), so the
first emitted frame sits at absolute frame `2P' + 2 * offset`. For a follow-up
hop that is exactly `M`, so the cached call's output is already the delta; only
the first hop, where `M = 0`, drops its prompt frames.

### Files

| file | change |
|---|---|
| `models/fun_cosyvoice3/flow_hop_cache.py` | new: pool and allocator, per stream handle, segment metadata, `CachedDiT`, the attention |
| `models/fun_cosyvoice3/stages.py` | cached call on `FunCosyVoice3Flow`, `hop_batch_cached` on the vocoder, pool built in the factory |
| `models/fun_cosyvoice3/streaming_vocoder.py` | state field, the cached/fallback split in `run_step`, release in two places |
| `models/fun_cosyvoice3/config.py` | `flow_kv_cache_bytes` on the vocoder factory args |

`packed_dit.py` is not edited: `CachedDiT` subclasses `PackedDiT` and overrides
`row_attention`, `_rope` and `_conv_pos_embed`, which is what G0 did.

## Tests

The segment metadata is host arithmetic and is the part that silently corrupts
audio when it is wrong, so it is a pure function of `(spans, chunk)` and is
tested without a GPU: ragged rows, a hop that starts mid chunk, a first hop with
`M = 0`, and the `cu_seqlens_q` and `cache_seqlens` a known layout must produce.
The pool tests need CUDA and are skipped without it: slots allocated and freed
balance over a stream's life, an exhausted pool returns no slots rather than
raising, and a released stream returns every slot it held.

## Gate

G2 as declared in the plan, section 6: c1 and c16, full English corpus, one boot
per arm against main, memory census, stage 1 hop points rerun. WER within 0.3
absolute and SIM within 0.005; fallback counter 0 at c16 under the chosen budget;
hop wall follows new frames.

What this box can settle: the hop wall against new frames, the fallback counter
at c1 and c4, the numerics (already settled by G0), and WER and SIM at c1. What
it cannot: the c16 census, the memory budget P3 and the c16 RTF that the roadmap
target is written in. Those are H100 work and the slice is not merged without
them.
