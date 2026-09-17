# Slice 2.1: CUDA graphs for the hop step

Roadmap row 2.1. Depends on slice 1.2. Trees read: upstream main `27a8293c`,
slice 1.1, the G0 script, pinned SGLang `v0.5.19`.

## Why this is next after 1.2, and not before it

G0 measured it. The cached hop call is flat at 138 to 143 ms whether it computes
400 new frames or 1,600, while the full window call it replaces runs from 1,774
ms down to 843 ms over the same schedule. Both paths pay one eager launch floor
of about 140 ms, which is stage 1's 19,072 launches per Flow call
(`flow_hop_first_rows1`, wall 152.6 ms at busy over wall 0.41). 1.2 removes the
device work above that floor. What is left is the floor, and a graph is what
removes a launch floor.

Before 1.2 the same graph would have to cover calls whose frame count grows with
the stream, which is the shape problem the padded runner already has. After 1.2 a
hop computes `2 x hop` new frames per row, from a hop schedule of 25, 50 and 100
tokens, so the capture table is small and known.

## What already exists, and what it does not cover

`FlowCudaGraphRunner` (`stages.py:342-518`) captures `solve_flow_euler`, the
padded non streaming solve, on a side stream into a shared pool
(`:394-430`), keyed by `(batch_size, mel_frame)` with frames bucketed to 16
(`:78`, `:469-473`). `run` right pads the real batch into the static inputs,
replays, and returns a clone of the static output sliced back to the real frame
count (`:447-517`). It is attached only for `finalize=True` non streaming calls:
`generate_flow` takes the eager solve whenever `streaming or not finalize`
(`stages.py:659`), and `generate_flow_packed`, the path every hop takes, never
consults it at all (`stages.py:690-716`).

So the mechanism is present and proven in this file, and the hop path uses none
of it. 2.1 is a second runner over the packed cached solve, not a change to this
one.

## What has to be static for a hop graph

Everything `solve_flow_euler_packed` reads, plus the cached call's metadata. From
1.2's design the per call inputs are:

| input | varies with | in a graph |
|---|---|---|
| packed noise, token condition, prompt mel | total new frames | static buffer at the bucket's total, right padded |
| speaker embedding | row count | static buffer at the bucket's rows |
| `time_span`, `flow_time` | nothing, and the step index | already static; `flow_time` is written in place today (`packed_dit.py:212-216`) |
| `PackedRows` for the twin rows | row count and each row's new frames | derived host side from the bucket, so it is fixed per capture |
| slot indices for `set_kv_buffer` | which slots the allocator handed out | static buffer, filled per call, exactly as SGLang fills its decode metadata |
| `page_table` | each lane's whole history, so it grows with the stream | static buffer of `(segments, max frames)`; SGLang keeps the same thing static in its decode and prefill runners (`flashattention_backend.py:601-703, 2187-2211`) |
| `cache_seqlens`, `cu_seqlens_q`, `max_seqlen_q` | chunk layout of `[M, M')` | the first two are static buffers; `max_seqlen_q` is a python int baked into the capture, so it belongs in the bucket key |

The one input that is not obviously bounded is the page table's width: it is the
stream's whole cached history, which grows to the noise ceiling of 15,000 frames
(`flow_matching.py:199-200`, checked at `stages.py:589-593`). A static table of
`segments x 15,000` int32 is 60 KB per segment, so the whole table is a few MB
and the bound is the model's, not a tuned constant.

## The capture table

Not chosen here. It is a measurement: the c16 ledger's distribution of
`(rows, new frames, segments, max_seqlen_q)` per hop call after 1.2 lands. The
padded runner's own table is 54 entries derived the same way
(`config.py:19-75`), and the hop table should be smaller because new frames per
row take three values under the shipped schedule. A row that misses the table
replays nothing and runs eager, which is what `run` already does when it finds
no capture (`stages.py:475-496`).

## Gate

Two parts, in order.

1. Replay is bit identical to the eager cached path on the same inputs, per hop,
   over a schedule that hits every entry of the capture table. This is a
   self comparison on one card, so a 4090 settles it.
2. Census: c1 and c16 against the 1.2 head, first audio mean and p95, RTF p99,
   C50, req/s, plus the launch count per Flow call from the ledger, which is
   what this slice claims to move. The c16 half is H100 work.

## What could stop it

The cached attention call is `flash_attn_with_kvcache` with a page table. It is
capturable in SGLang's own graph runners, which is the precedent, but this is
the first time this repo captures it, and the FA3 build here is a wheel, so the
first thing 2.1 does is capture and replay one cached hop and compare it to
eager. If that fails, the slice stops there and the finding is the deliverable.
