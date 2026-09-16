# Plan 1: hop prefix K/V cache for Fun-CosyVoice3 streaming

Roadmap rows 1.1 and 1.2. Tree: upstream main `cc85ddaa9`. Every line reference below was read on
that tree, the pinned SGLang `v0.5.19` (sglang-kernel `0.4.6.post1`) or the vendored CosyVoice at
`074ca6dc` (`inputs/external_sources/CosyVoice`). Numbers marked derived are arithmetic, not
measurements; unknowns are named validation tasks in section 8.

## 1. Why this first

- Hop Flow is 58.5 percent of vocoder step time at c16 (readout 03 section 6).
- At 16 rows a hop is device bound, busy over wall 0.87 to 0.99 (stage 1 profile, roadmap section
  "Stage 1 component profile"), so frames removed are time removed.
- Every hop recomputes the prompt and all earlier frames, then drops them
  (`streaming_vocoder.py:376-395`, reference `model.py:436`). On the c16 ledger a cache runs 49.1
  percent of hop frames (E5, `MECHANICS.md` row 39).
- Derived: if hop Flow time follows frames, removing 50.9 percent of hop frames removes about 30
  percent of vocoder step time (0.585 x 0.509). Attention is superlinear in frames and conditioning is
  not, so this is an estimate to be replaced by the G2 census.
- Measured 2026-09-16 (G0, 8 rows, staggered growth schedule, eager both sides): the hop call falls
  from 1,774 ms to 843 ms over the schedule, 2.1x, at a new over window frame ratio of 0.348. Per step
  the ratio runs 1.06x at 400 window frames to 3.70x at 6,850. This schedule caches a larger share of
  frames than the c16 ledger (0.348 against 0.491), so G2 will show less.
- Also measured: the cached call is flat at 138 to 143 ms whether it computes 400 or 1,600 new frames.
  Both paths pay the same eager launch floor of about 140 ms, which is stage 1's 19,072 launches per
  Flow call (`flow_hop_first_rows1`, wall 152.6 ms at busy over wall 0.41). The cache removes device
  work, not launches, so after 1.2 the hop is launch bound and roadmap 2.1 (hop CUDA graphs) becomes
  the next lever rather than a deferred one.
- The cached call also attends through paged FA3 without padding, so the runaway hop (attention plus
  mask 77 percent of a 3,061 ms call in the stage 1 profile) stops paying rows x widest squared.

Prior art (tracker and upstream, checked): no CosyVoice3 server caches Flow K/V; upstream, the
Triton runtime, FastCosyVoice, vllm-omni and Omni all recompute. Upstream CosyVoice2 had a windowed
decoder cache on a separate one-left-chunk checkpoint and removed it (`68100c2`, "remove flow_cache").
Omni #1652 lists no flow cache; #1883 reports HiFT output depends on batch shape; #2110 (open)
changes the padded solve's timestep layout and edits `streaming_vocoder.py`.

## 2. The hop today

```text
AR token -> _queue_or_emit_code_chunk          model_runner.py:292-312  first flush 28, then every 25
vocoder loop: drain inbox, else run_ready_step streaming_simple_scheduler.py:135-165
  ingest: tokens.extend, latch padded prompt   streaming_vocoder.py:202-233, 263-274
  _pump_one_step                                scheduling/streaming_vocoder.py:365-381
    select_step_participants: up to 16 rows    streaming_vocoder.py:300-332
    run_step, causal_window                     streaming_vocoder.py:376-416
      window = tokens[: offset + hop + 3]       :379-381
      hop_batch (bf16 autocast)                 stages.py:1531-1541
        inference_causal                        stages.py:803-828
          pack_flow_inputs: prompt + window     stages.py:151-239
          prepare_flow_conditioning             stages.py:530-626
            pre-lookahead with 3 token context  :547-578
            noise = rand_noise[:, :, :frames]   :606-611
          generate_flow_packed                  stages.py:690-715
            solve_flow_euler_packed: 10 steps,  packed_dit.py:191-232
              CFG twins, flow_time in spks dtype :206-212
              PackedDiT.forward, 22 blocks      packed_dit.py:131-159
                conv pos embed on padded rows   :161-163
                RowAttention: padded SDPA with  :97-109
                  key mask and 50 frame chunk mask
          split: frames [2P, 2(P+offset+hop))   stages.py:718-737, 815-828
      per row: hift_delta over whole mel history stages.py:1559-1572
      offset += hop; hop = min(100, 2 hop)      streaming_vocoder.py:402-409
    run_step, leftover: full window, streaming=False, finalize=True   :351-374
release_stream_resources                        streaming_vocoder.py:486-493
  reached from completion, abort, step failure and shutdown
  (scheduling/streaming_vocoder.py:147-164, 259-263, 509-527)
```

## 3. What the model fixes, and therefore what a cache must keep

Read in the vendored source; each row decides a piece of state.

| module | receptive field | source | consequence |
|---|---|---|---|
| token embedding, speaker projection, input projection, AdaLN, FF, output | pointwise | `packed_dit.py:145-159` | none |
| pre-lookahead | tokens i-2 to i+3: conv1 kernel 4 over body plus 3 context tokens, conv2 kernel 3 left padded 2, residual | `upsample_encoder.py:66-102` | recompute from token ids over [T0-2, T1+3); no state |
| upsampling | frame f from token f // 2 | `stages.py:580-584` | none |
| noise | `randn([1, 80, 15000])` after seed 0, indexed by absolute frame | `flow_matching.py:199-200` | slice [M, M'); positions capped at 15,000 |
| conv position embedding | 2 convs, kernel 31, groups 16, left pad 30 each, Mish: 60 frames left, 0 right | `modules.py:115-144` | per (step, CFG row): last 30 frames of conv1 input and of conv2 input |
| RoPE | first 64 of 1,024 channels rotated before the head split, absolute row position | `x_transformers.py:763-780`, `packed_dit.py:165-170` | post-RoPE keys are final |
| attention, streaming | a frame sees frames 0 to the end of its 50 frame chunk | `packed_dit.py:67-74`, `mask.py:155-158`, `dit.py:163-164` (all left chunks) | K and V of every earlier frame, prompt included, per (Euler step, block, CFG row) |
| time embedding | cast to the timestep dtype | `modules.py:606-616` | the cached path must use the production timestep dtype |

Output frame f is final once its chunk is complete and its tokens plus 3 lookahead tokens exist.
Hops end on chunk boundaries only because the prompt is padded to a 25 token multiple and hops are
25, 50 and 100 tokens (`streaming.py:29-36, 56-71`).

State per request, derived:

- K and V: 10 steps x 22 blocks x 2 CFG rows x (1,024 + 1,024) elements per frame, 1,802,240 bytes in
  bf16 (matches E5's measured 1,802,240).
- Conv position embedding tails: 2 x 30 x 1,024 x 2 rows x 10 steps, 2.46 MB, fixed per request.
- Nothing for tokens or noise; the speaker projection is recomputed or kept (80 elements).

## 4. Heuristics and names, audited

| # | item | today | reference | verdict | action |
|---|---|---|---|---|---|
| H1 | prompt padding | pads the prompt to a 25 token multiple by repeating the last token and last mel frame, up to 48 fabricated frames (`streaming.py:100-144`, `streaming_vocoder.py:217`); buffered and fallback paths do not pad (`stages.py:1574-1601`) | the first hop waits for hop + pad generated tokens; the prompt stays real (`model.py:345-350`) | differs: conditions on fabricated frames for first audio latency, and streaming and buffered prompts differ | decision D1, its own A/B (section 7); the cache works with either |
| H2 | hop growth 25, 50, 100 | per request (`streaming_vocoder.py:405-409`) | "increase token_hop_len incrementally to avoid duplicate inference" (`model.py:411`), and shared across requests (`model.py:360`) | the Flow reason disappears with the cache; HiFT still reruns history per hop; E5: emitted mel bit identical under growth and fixed 25 in float64 and bf16 | keep in 1.2; the schedule becomes a HiFT plan decision |
| H3 | chunk alignment | `token_hop_len` and `token_max_hop_len` only checked positive and ordered (`streaming_vocoder.py:96-104`); prompt padded to the hop, not to the chunk | chunk is `static_chunk_size` 50 frames (`cosyvoice3.yaml:17,74`) | a hop that is not a chunk multiple silently changes frames already emitted and breaks a cache | 1.1: derive chunk tokens from `static_chunk_size / token_mel_ratio`, require hop and max hop multiples of it |
| H4 | `PRE_LOOKAHEAD_LEN`, `TOKEN_MEL_RATIO` | module constants in the scheduler (`streaming.py:13-14`) | `flow.pre_lookahead_len`, `flow.token_mel_ratio`; stages already reads these (`stages.py:546, 727, 815`) | equal for this checkpoint, divergent by construction | 1.1: read from the flow |
| H5 | dead code | `stream_hop_len`, `tokens_needed_for_causal_chunk` have no callers; `first_ar_flush_tokens` deletes its `prompt_len` argument (`streaming.py:39-97`) | | dead | 1.1: remove |
| H6 | streaming step size | only `max_batch_size` 16 bounds a step; `flow_batch_admission_frames` and the merge knobs apply to buffered requests only (`streaming_simple_scheduler.py:265-328`) | | with the cache a hop call costs new frames, so the missing frame bound matters less | none |
| H7 | AR follow-up flush | fixed 25 tokens (`model_runner.py:45, 327-329`) while vocoder hops grow | reference polls every 0.1 s (`model.py:347`) | flushes at 53, 103, 128, 153 wake the vocoder with no work | none here |
| H8 | E5 bf16 cached result (17.6 to 43.1 dB) | E5 built `flow_time` from `time_span` (`e5_hop_prefix_exactness.py:195`), float32 under autocast | production uses the speaker embedding dtype, bf16 (`packed_dit.py:212`) | the comparison mixed a timestep dtype change with caching | G0 reruns it with the production dtype and against float32 |
| H9 | final call | full bidirectional recompute of the whole history (`stages.py:784-800`) | same (`model.py:366-373`); a causal final dropped the tail 0.5 s cosine to about 0.29 (`stages.py:1548-1550`) | reference semantics; no causal cache can serve it | out of scope; the cache is released before the final |
| H10 | noise length | 15,000 frames (`flow_matching.py:199-200`), checked per call (`stages.py:589-593`) | same | caps absolute positions | the cached path checks M' against it |

## 5. Design

### 5.1 One owner for per-request Flow state (1.1)

`CosyVoice3StreamState` (`streaming_vocoder.py:48-72`) gains one field, the request's hop cache handle,
created on the first hop and released in `release_stream_resources`, the single hook for completion,
abort, step failure and shutdown. A stream that reaches `leftover` releases its cache before the final
call. Chunk tokens, lookahead and mel ratio come from the flow (H3, H4).

### 5.2 Memory, reused from SGLang

| piece | SGLang source | use |
|---|---|---|
| `MHATokenToKVPool(size, page_size=1, dtype=bfloat16, head_num=16, head_dim=64, layer_num=220, device, enable_memory_saver=False, enable_alt_stream=False)` | `srt/mem_cache/memory_pool.py:1808-1831`; buffers allocated per layer at construction (2150-2161) | K and V; layer id `step * 22 + block` |
| `TokenToKVPoolAllocator(size, dtype, device, kvcache, need_sort=False)` | `srt/mem_cache/allocator/token.py:31-75` | frame slots; device side, no sync on alloc or free (the paged allocator's `free` calls `torch.unique`) |
| `ReqToTokenPool(size, max_context_len=15000, device, enable_memory_saver=False)` with `alloc_rows` and `free_rows` | `srt/mem_cache/memory_pool.py:256-331` | one slot row per (request, CFG row); row 0 reserved |
| `store_cache` | `kernels/ops/kvcache/kvcache.py:55-68` | write new K and V before attention |
| `flash_attn_with_kvcache` | `kernels/aot/python/sgl_kernel/flash_attn.py:36-73` | read path with `k=v=None`, as SGLang's backend (`flashattention_backend.py:1330, 1565`) |

Not reused: `FlashAttentionBackend`, which needs a ModelRunner and the global runtime getters;
SGLang's `CausalSelfAttentionKVCache` (`multimodal_gen/runtime/layers/kvcache/causal_attention_cache.py`),
which is dense with one cursor for the whole batch.

Budget. The pool is built in the vocoder factory, which runs before the AR engine sizes its pool
(`config.py:144-172`); stage 0 measured that pool at 60.3 GB for 5.27 M tokens. A cached frame costs
1,802,240 bytes; the c16 ledger's hop calls hold 8,350 frames at p95 and 10,200 at most (readout 03
section 6), 14.0 and 17.1 GiB derived. The budget is a vocoder factory argument in bytes,
`flow_kv_cache_bytes`, next to `flow_batch_admission_frames`; its default is set by V2, not by a
constant. When the allocator cannot serve a stream's next hop, that stream releases its cache and
runs the existing full window path for its remaining hops, which returns today's output; a counter
records it.

### 5.3 The cached hop call (1.2)

```text
run_step causal_window                                    streaming_vocoder.py:376
  rows with a cache, or whose slots allocate -> cached call
  rows whose slots do not allocate            -> today's hop_batch (fallback)
  cached call, per row r:
    new frames [M_r, M'_r), M'_r = 2 (P' + offset + hop); first hop M_r = 0 (prompt computed and cached)
    slots: 2 (M'_r - M_r) from the allocator, written into the row's req_to_token entries on device
    conditioning for new frames: pre-lookahead over tokens [T0 - 2, T1 + 3), repeat 2, cond from the
      prompt feat below 2P', noise rand_noise[M_r : M'_r]
    metadata, host built once per call, one pinned copy: per (row, CFG, chunk) segment
      cu_seqlens_q, cache_seqlens = chunk end, page_table = the row's slot list; shared by all 220 layers
  10 Euler steps over the packed new frames and their CFG twins:
    time embedding in the production timestep dtype
    input projection
    conv position embedding: per row concat(tail, new), valid conv, tails updated
    RoPE at absolute positions M_r + j
    22 blocks:
      q, k, v
      store_cache(k, v -> layer step * 22 + block, new slots)
      flash_attn_with_kvcache(q, k_cache[layer].view(-1, 1, 16, 64), v_cache[layer]..., page_table,
        cache_seqlens, cu_seqlens_q, max_seqlen_q = 50, causal = False)
      AdaLN, FF (unchanged modules)
    Euler update, CFG combine (unchanged arithmetic)
  output frames [max(M_r, 2P'), M'_r) -> hift_delta as today
```

This is the configuration E6 qualified on real activations: FA3 with page size 1, one query segment
per (row, chunk), `cache_seqlens` at the chunk end, `causal=False`; minimum 53.3 dB against float32
SDPA with 0 calls under 40 dB, production 54.0 (readout 03 section 4). The buffered call, the final
call and the TensorRT estimator (`packed_estimator` None, `stages.py:864`) keep their paths.

Files: a new `models/fun_cosyvoice3/flow_hop_cache.py` (pool, per-request handle, segment
metadata), `packed_dit.py` (cached forward over paged attention), `stages.py` (cached conditioning and
call, factory builds the pool), `streaming_vocoder.py` (state field, split into cached and fallback
rows, release), `config.py` (the budget argument).

### 5.4 Rejected alternatives

| alternative | why not |
|---|---|
| cache block inputs instead of K and V | half the memory, but to_k, to_v and RoPE rerun over the whole history every call; linear layers are 16 to 18 percent of hop device time, so the history cost returns |
| windowed cache (CosyVoice2 `flow.cache.pt`, Step-Audio2 prompt plus 100 frames) | approximate: CosyVoice3 attends all left chunks; upstream removed its windowed cache |
| dense per-request tensors grown by concatenation (E5) | a reallocation per hop and block, no paged kernel, fragmentation at c16 |
| a causal final to reuse the cache | changes output (tail cosine about 0.29 in the stages note); reference is bidirectional |

## 6. Gates, declared before any run

Thresholds below were proposals agreed before the first run, not after it. G0 ran on 2026-09-16 at
the proposed 1 dB margin and passed; G1 and G2 are still proposals.

The raw HiFT waveform was in the G0 proposal and is no longer gated. HiFT is a deterministic function
of the mel, but its excitation phase is a cumulative sum of the predicted F0, so a bf16 level mel
difference drifts the phase and the waveform decorrelates in L2 for the shipped path too: production
against the float32 truth scores -1.6 dB on the waveform while its mel scores 36.4 dB, and waveform
SNR correlates with mel SNR at 0.11 over the 36 hops. The magnitude spectrum of the delta replaces it
and the raw waveform stays a reported diagnostic.

| gate | what | pass |
|---|---|---|
| G0 numerics (no runtime change) | box script: 8 real references with mixed prompt lengths, growth schedule, one packed multi-row call per hop; paths (a) production packed bf16 full window, (b) cached bf16 through the SGLang pool and FA3 with the production timestep dtype, (c) float32 full window truth; per emitted hop mel and the magnitude spectrum of its HiFT delta: SNR against (c), max abs, NaN and Inf, lengths | (b) min SNR at least (a) min minus 1 dB and median at least (a) median minus 1 dB; no NaN or Inf; equal lengths. If it fails, trace per block before any runtime code. **Passed 2026-09-16** on run `g0-20260916T061846Z`: mel -0.21 dB at the minimum, +0.15 dB at the median over 36 hops, unbiased (cached is closer in 18 of 36, mean +0.03 dB, spread 0.90 dB) |
| G1 refactor (1.1) | seeded c1 stream, same boot shape as main | emitted audio byte identical to main; c16 census no regression beyond noise |
| G2 cache (1.2) | stream c1 and c16, full English corpus, one boot per arm against main; memory census; stage 1 hop points rerun | req/s, audio s/s, first audio mean and p95, RTF p99, C50 reported; WER within 0.3 absolute and SIM within 0.005 of main; fallback counter 0 at c16 with the chosen budget; hop wall follows new frames |

## 7. Slices

| slice | content | runtime change | gate |
|---|---|---|---|
| 1.0 | `stage2/g0_hop_cache_numerics.py`, steps and results in `stage2/README.md`; ran 2026-09-16, G0 passed | no | G0, done |
| 1.1 | state owner, chunk validation (H3), flow constants (H4), dead code (H5) | yes, no numeric change | G1 |
| 1.2 | SGLang pool, cached hop call, fallback, budget argument | yes | G2 |
| D1 | prompt padding to reference semantics (H1): first hop waits hop + pad tokens, prompt real; `first_ar_flush_tokens` becomes prompt aware | yes, output change | own A/B: WER, SIM, first audio (derived cost at most 24 decode steps, about 60 ms at 2.53 ms per step) |

Deferred: the hop schedule (H2) with the HiFT plan; the final call (H9); hop CUDA graphs (roadmap
2.1), which SGLang shows are possible with this kernel (decode and prefill graph runners keep
`cache_seqlens` and `page_table` in static buffers, `flashattention_backend.py:601-703, 2187-2211`).

## 8. Validation tasks

| # | unknown | how it is settled |
|---|---|---|
| V1 | the bf16 cached hop against float32 (H8) | G0. Settled 2026-09-16: cached is within -0.21 dB of production at the minimum and +0.15 dB at the median, unbiased over 36 hops |
| V2 | the budget: live cached frames at c16 on the stack head, and the AR pool's actual need | memory census plus the ledger's per call frames on the G2 boots |
| V3 | FA3 `causal=False` with `cache_seqlens` at chunk end reads exactly [0, chunk end) at C++ level (the FA3 C++ source is fetched at build time, not on disk) | G0 compares against float32 SDPA math, as E6 did. Settled 2026-09-16: a wrong key range would collapse the SNR, and it did not on any of 36 hops with chunk alignment held throughout |
| V4 | ragged conv position embedding with tails equals the full padded conv | G0 per block trace on the first failure; exact by construction in float64 (E5 table 3). Settled 2026-09-16: no trace needed, G0 passed with the tails as the only conv state carried |
| V5 | #2110 merge order | if it merges first, rebase and rerun G0 on its timestep layout. G0 as run is on `cc85ddaa9`, before it |
| V6 | HiFT output depends on batch shape (#1883) | HiFT stays per request in this plan, so G1 identity holds |

## 9. Decisions pending

Status 2026-09-16: slice 1.0 ran at the proposed P1 margin of 1 dB and G0 passed, by a wide enough
margin that a tighter threshold would also have passed (cached is -0.21 dB at the minimum and +0.15 dB
at the median, and closer than production on 18 of 36 hops). P1 is therefore settled unless the owner
wants it restated. No runtime code yet; P2, P3 and P4 are still open and block 1.2 and D1, not 1.1.

| # | decision | proposal | options | blocks | takes effect in |
|---|---|---|---|---|---|
| P1 | G0 pass threshold | cached bf16 SNR against float32 at least today's bf16 minus 1 dB, at the minimum and at the median; no NaN or Inf; equal lengths | tighter (0.5 dB), looser, or a waveform level criterion instead of mel | slice 1.0 run | section 6, G0 |
| P2 | G2 quality threshold | WER within 0.3 absolute and SIM within 0.005 of main, full English corpus | other bounds, or add UTMOS | slice 1.2 merge | section 6, G2 |
| P3 | Flow cache memory budget | `flow_kv_cache_bytes` on the vocoder factory args; default from the V2 memory census; c16 ledger bound 17.1 GiB at 10,200 frames; the AR pool (60.3 GB at stage 0) shrinks to make room; shortage falls back to today's path | a fixed default, an explicit engine `kv_cache_bytes` for the AR pool alongside it, or opt in only | slice 1.2 implementation | section 5.2 |
| P4 | D1, prompt padding to reference semantics | first hop waits hop + pad generated tokens with the real prompt; `first_ar_flush_tokens` prompt aware; own A/B on WER, SIM and first audio (derived cost at most 24 decode steps, about 60 ms) | proceed after 1.2, proceed before 1.2, or park | slice D1 | section 4 H1, section 7 |

Deferred items that also need a later decision: the hop schedule (H2) with the HiFT plan, the final
call (H9), hop CUDA graphs (roadmap 2.1).
