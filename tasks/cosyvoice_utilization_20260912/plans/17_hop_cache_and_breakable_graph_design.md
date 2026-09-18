# Plan 17: the hop prefix cache and the breakable graph, one design

Written 2026-09-18 for review before any code. It rests on whole file reads of pinned SGLang
`v0.5.19` and omni `upstream/main` 27b5b0d4f (research cuts R1 and R2 of plan 16, their load
bearing claims re-opened by me in the source), on plan 11 (E5, G0) and on today's measurements.
Supersedes the code plan of `11_hop_prefix_cache.md` section 5 and of `12_flow_graph_redesign.md`
section 4; their measurements stand.

## 1. Why these two, in this order

Vocoder step time at c16 streaming is hop Flow 58.5, final Flow 20.5, HiFT 20.7 percent; the AR is
about 8 percent of GPU time. A hop today recomputes the whole prefix (CosyVoice's own algorithm,
`cosyvoice/cli/model.py:425-436`), so a request costs on the order of its length squared and one
runaway costs about 6 percent of c16 throughput (readout 06). The cache makes a hop cost its new
frames. G0 measured what is left: a cached hop is flat at 138 to 143 ms from 400 to 1,600 new
frames, which is pure launch time, and that is what the breakable graph removes. The graph is
designed against the cached step because the cache decides what a step looks like.

## 2. The hop prefix cache, on SGLang's classes

Exactness is already settled: in float64 a hop computed from cached K and V of finished chunks
equals the full recompute, and earlier frames never change (E5); in bfloat16 the cached hop is
53 to 54 dB from a float32 truth, the same as the uncached one (G0).

| piece | SGLang class, used unchanged | how |
|---|---|---|
| storage | `MHATokenToKVPool(size, page_size=1, dtype=bfloat16, head_num=16, head_dim=64, layer_num=220, device, enable_memory_saver=False)` (`srt/mem_cache/memory_pool.py:1809-1831`) | layer id = Euler step x 22 + block. `set_kv_buffer(None, slots, k, v, layer_id_override=layer)` is supported with no model layer (`:2380-2401`); Whisper offsets layer ids into one pool the same way (`srt/models/whisper.py:205-207`). The class reads no server args, model config or TP group |
| slots | `TokenToKVPoolAllocator(size, dtype, device, kvcache, need_sort=False)` (`allocator/token.py:31-75`) | `alloc(n)` returns None when short; `free` on stream end; `free_group_begin/end` batches the frees of a step |
| per stream rows | `ReqToTokenPool(size, max_context_len, device, enable_memory_saver=False)`, `alloc_rows` and `free_rows` take no `Req` (`memory_pool.py:256-331`) | two rows per stream, one per CFG lane; `req_to_token[row, :frames]` holds the stream's slots. Replaces the parked branch's Python lists of tensors |
| attention | `flash_attn_with_kvcache` with `page_table = req_to_token[rows, :max_frames]`, `cache_seqlens`, `cu_seqlens_q`, `max_seqlen_q`, K and V passed as `get_key_buffer(layer).view(-1, 1, 16, 64)` | the form of SGLang's extend path with a prefix (`flashattention_backend.py:1001-1054, 1565-1583`) and of `RaggedRowAttention` today. Segments are (lane, new chunk); `cache_seqlens` is the chunk's absolute end, so the chunk causal rule needs no mask |
| radix tree | not used | it shares identical token prefixes across requests; a stream's history is its own and append only, which is what `ChunkCache` is: a `req_to_token` row and the allocator (`chunk_cache.py:60-92`) |

Checked against the other candidate, SGLang's diffusion side, which ships a causal K/V cache for
chunked video DiTs (`multimodal_gen/runtime/layers/kvcache/causal_attention_cache.py`, driven by
`pipelines_core/stages/causal_denoising.py`). It does not fit, for two reasons in its source:

- it is one request's contiguous `(batch, cache_size, heads, dim)` buffer per block with a sliding
  window, eviction and sink tokens (`causal_attention_cache.py:19-37, 136-335`): no allocator, no
  slots shared between requests, so it cannot back a step that packs sixteen streams;
- it holds one K/V per block, filled by an extra forward on the clean latent at `context_noise`
  after the chunk is denoised (`causal_denoising.py:833-873, 949-964`). That is the contract those
  models were trained with. CosyVoice3's DiT sees its prefix at each Euler step's own noise
  level, so the K and V differ per step and the cache is per (step, block): the 220 layers above.

So the cache comes from SRT and the breakable graph from the diffusion side, as two PRs: the
cache first (S1), the graph over the cached step after it (S3).

Per hop, for the rows that hold a cache: allocate slots for the new frames of both lanes, write
them into the rows, run the ten steps over the new frames only (rope and noise at absolute
positions, the conv position embed fed each row's new frames after the last 30 inputs of each of
its two convs kept from the previous hop, per step and lane, about 2.4 MB per stream), attend
through the page table, keep the frames past `token_offset` as today. Conditioning (token
embedding, lookahead layer, prompt mel) stays whole prefix: it is a few small layers over tokens,
and slicing its output is simpler than caching its state.

Finals stay uncached and whole sequence: the last chunk is bidirectional by the model's design
(`leftover_batch`, the note that a causal last chunk dropped the tail cosine to 0.29), and the
final releases the stream's rows and slots.

Memory: 1,802,240 bytes per frame (10 x 22 x 2 lanes x 2,048 elements, bfloat16), 14.0 GiB at
the c16 p95 of 8,350 held frames. One vocoder factory argument in bytes, no default constant: 0
is off. When `alloc` returns None the row runs today's uncached hop and its stream drops its
cache, which is the parked branch's fallback; a stream that missed its first hop stays uncached.
Where the bytes come from on a given card is the operator's memory fraction, as for every other
consumer; this plan adds no sizing logic.

As built (branch `slice/cosyvoice-4-1-hop-prefix-cache`, 0dbcb01a3, new file
`flow_hop_cache.py`):

```
scheduler.run_step (streaming_vocoder.py)
  split_hop_participants ── cache.open_stream()      ReqToTokenPool.alloc_rows(2)
        │                   cache.reserve(stream,end) allocator.alloc -> req_to_token[lanes, reserved:end]
        │                   None / False -> release, fallback_hops += 1, row runs hop_batch
        ├─ cached rows -> vocoder.hop_batch_cached -> flow.inference_cached_hop (stages.py)
        │       prepare_flow_conditioning (whole prefix, unchanged)
        │       cache.begin_hop(streams) -> CachedHop: slots, absolute positions, page_table =
        │                                   req_to_token[lanes[segment rows], :max_end], cache_seqlens
        │       solve_flow_euler_packed(CachedDiT, new frames only)
        │           _conv_pos_embed: conv_tails[step][:, lanes] ++ new frames, tails written back
        │           _rope: absolute positions
        │           CachedHop(q, k, v): pool.set_kv_buffer(layer) + flash_attn_with_kvcache
        └─ plain rows  -> vocoder.hop_batch (main's path)
  leftover step: release_flow_cache first, then main's bidirectional final
```

The conv tails live in one tensor indexed by request table row, `(steps, 2, rows + 1, 30, 1024)`,
read and written with one index each per step, and the byte budget covers slots, table rows and
tails together: `slots = budget * chunk // (bytes_per_slot * chunk + bytes_per_row)`,
`rows = slots // chunk`, because a lane that holds a row holds at least one chunk of slots. The
scheduler refuses the cache when `token_hop_len` or `token_max_hop_len` does not cover whole
chunks: a cached frame is final only once its chunk is complete. Since main now reads ragged
FA3 too, a stream's first cached hop runs the same rows through the same kernels as main's hop,
so the probe checks it for bit identity (`stage2/s1_hop_cache_gate.py`).

What is reused from the parked branch `slice/cosyvoice-1-2-hop-cache`: the pool and allocator
construction, the lane major layout, the conv tails, the fallback split in the scheduler. What
is replaced: its slot lists by `ReqToTokenPool` rows, its `hop_layout` by `chunk_segments`
(already on main), its `CachedDiT(PackedDiT)` constructor and `row_attention` signature, which no
longer match main.

## 3. The breakable graph over the cached step

SGLang's diffusion runner, unchanged: `DiffusionBreakableCudaGraphRunner(transformer, device)`
(`multimodal_gen/runtime/breakable_cuda_graph/runner.py:201-533`). Mechanics that decide the
design, each re-opened in the source:

| mechanic | where | consequence |
|---|---|---|
| tensor kwargs become static buffers, leaves matched by position; `_map_tensors` recurses tuple, list, dict only | `runner.py:78-109, 356-369, 455-472` | everything the step reads must arrive as plain tensor kwargs; an object holding tensors is frozen at its capture address |
| the key is every tensor's shape and dtype plus constants; any other object keys by `id` | `runner.py:112-130` | a metadata object rebuilt per call would miss every time; metadata travels as tensors of fixed capacity |
| a break's tensor arguments are frozen addresses, but the break re-executes real Python on replay, so attribute reads and context variables are live | `breakable_cuda_graph.py:236-261`; shipped use: `DynamicVarlenMaskMeta.resolve` rebuilds varlen metadata once per replay from a static mask (`layers/attention/layer.py:262-285`, `replay_token.py`) | the ragged attention and the conv can stay eager and still see this call's metadata, provided its tensors are static leaves |
| serving never captures; capture happens at warmup per signature; 32 entries and 512 segments by default; an over limit capture disables the runner | `runner.py:289-318, 427-448, 507-513` | the bucket set is enumerated and captured at startup, largest first |
| variable lengths are padded to buckets by the caller before the runner, zeros plus a mask that keeps padded positions inert | `prompt_padding.py:89-113, 221-249`, `pipelines_core/stages/denoising.py:2387-2430` | the same shape for us: pad the packed step to a bucket of total frames |
| torch.compile and BCG are exclusive; the denoising loop keeps scheduler math in Python and replays one DiT forward per step | `denoising.py:500-503, 2348-2372` | one captured callable = one DiT step; the Euler loop stays in Python, ten replays per call |

In a packed step every module's shape depends on total frames alone except two (R2, table D1):
ragged attention and the conv position embed. Those are the breaks. 22 blocks plus the conv give
23 breaks and 24 segments per step, under the 512 limit.

The captured callable is one `PackedDiT` step taking tensors only: `x, mu, cond, spks` of
`(1, T, channels)`, `t`, rope `(1, T, 64)`, and the attention and conv metadata as tensors of a
capacity fixed by the bucket (`cache_seqlens`, `cu_seqlens_q`, `page_table`, the conv gather
index). `streaming` is a constant kwarg, so it selects the graph, as a variant.

Buckets: with the cache a hop's new frames are 50, 100 or 200 per stream (25, 50, 100 tokens
after the prompt hop), doubled for CFG, so T is a multiple of 100 bounded by the step's batch.
The ladder is derived from that, not measured on a card; its spacing is a waste bound, V3.
Padded frames belong to no query segment and to no page table, so attention never reads or
writes them, the conv break computes real rows only, and every other module is per frame; the
output is cropped to T. That padded frames stay inert is V2, not an assumption.

Prerequisites, all exact and already listed in plans 12 and 14: the rope built once per call
instead of per step (`packed_dit.py:261-266`), the `apply_rotary_pos_emb` import out of the step
(`:275`), the weight cast of plan 14 (or each segment captures 320 casts).

Not covered by the graph: finals (uncached, whole sequence, large and device bound at 16 rows)
first run eager; whether they get buckets is decided by V4.

## 4. The AR prefill, a port onto omni's existing contract

Small and standard (R1). Today `custom_prefill_forward` returns a result, so SGLang's prefill
dispatch and its graph path are never reached (`fun_cosyvoice3/model_runner.py:53-61, 406-409`).
The port: gather the per request embedding slices in `before_prefill` and attach them with
`attach_omni_prefill_inputs` (the sidecar exists because upstream refuses a batch carrying
`forward_batch.input_embeds`, `prefill_cuda_graph_runner.py:1159-1160`); let the model forward
accept `input_embeds` and `omni_prefill_rids`; keep the last token gather and the `llm_decoder`
head outside the captured body; set `supports_breakable_prefill_cuda_graph` and the breakable
prefill backend with Higgs's bucket policy. The hash ids used as `input_ids` for the radix cache
are untouched: the captured body reads `input_embeds` only. Worth first audio at low concurrency
(prefill is 10 to 18 ms eager, 350 to 440 launches), not throughput. V5 checks that the model's
body and head split the way the runner's body patching expects.

## 5. HiFT, after the above

20.7 percent of vocoder time, per request, 1,196 launches, the whole mel history recomputed each
hop, one host built tensor per call (`generator.py:297`), the f0 predictor in float64. Omni's
precedents are full graphs with exact lengths because padding advances causal state
(`moss_tts_local/vocoder_cuda_graph.py:36-38`) or a finite enumerated frame domain
(`qwen3_tts/streaming_vocoder.py:66-102`, `higgs_tts/audio_codec.py:201-280`). No design here;
its own plan once the Flow work lands, because the hop cache changes nothing in HiFT and the
history recompute is its real cost.

## 6. Slices and gates

Test rig for every serving gate: the default launch plus `--tts_engine.engine.mem_fraction_static
0.3` on both arms (24 GB card), the benchmark's default request, the whole English split, c1, c8,
c16, alternating boots, all outputs copied to the Mac and read there. Because runaway counts
swing c16 by 12 percent between boots of one tree (readout 06), every serving pair also reports
the per step times from `serve.log` by batch size, which do not depend on the draw.

| slice | content | gates |
|---|---|---|
| S1 | hop prefix cache on the SGLang classes, off unless a byte budget is given | unit suite; G0 harness: cached hop against the float32 truth within the G0 margin of the uncached hop; one process probe: hop wall time and peak memory, cached against uncached, at 1, 4, 16 rows and with a runaway; serving pairs with continuity; zero fallbacks at the chosen budget, and a run at a starved budget that must complete on the fallback |
| S2 | exact hoists (rope, import) | one process probe: bit identical mel, launches per step |
| S3 | breakable graph over the cached step | V1 to V4 first; then replay bit identical to eager per bucket; launches and wall time per step; memory per bucket; serving pairs |
| S4 | AR prefill port | c1 byte identity on the stable samples; first audio at c1 and c8 |

## 7. Validation tasks, before the design of S3 is frozen

| # | unknown | how |
|---|---|---|
| V1 | launches and wall time of one cached step as 24 segments plus eager breaks, against eager, at 1, 4 and 16 rows | a probe that wraps today's step in the runner with attention and conv marked `eager_on_graph` |
| V2 | padded frames stay inert: outputs of real frames bit identical with and without padding | same probe, pad to the next bucket |
| V3 | the waste bound of the ladder | padding per bucket over the frame totals a c16 run produces (from `serve.log`) |
| V4 | memory per captured bucket at one step granularity (the whole solver was 130 MiB) | `stage2/vocoder_memory.py` with the runner |
| V5 | the AR model's body and head split as `prefill_cuda_graph_runner.py:165-185, 1754-1811` expects | read `sglang_model.py` against `_resolve_transformer_layer_model` |
| V6 | FA3 paged attention inside a break reads a page table that is a static leaf refreshed per replay | part of V1 |
