# Qwen3-TTS memory provisioning plan

Draft for discussion. Every mechanism below was read at the cited line in the chain worktree
(`0a88253c6` and later) or in the sglang v0.5.18 blob (`git show v0.5.18:<path>` in
`/Users/ratish/sglang`, whose working tree now sits at v0.5.19, see section 8). The mechanics
reports behind it are `research/qwen3_tts_stage_memory_mechanics.md`,
`research/torch_cuda_memory_mechanics.md` and their verification.

## 1. The requirement, stated once

The engine process on the card holds, in construction order, the talker weights, the KV pool,
the attention workspaces, the per row decode buffers, one copy of the speech tokenizer, the
sglang decode graphs, the predictor graphs, a second copy of the speech tokenizer for the
vocoder and 192 vocoder decode graphs. At request time it adds the reference encode
activations, the prefill and decode activations, the history clones, the vocoder windows, and
the allocations that happen outside the torch allocator: cuDNN plans and workspaces at first
use of each shape, and lazily loaded CUDA modules.

Only one of those is sized by policy rather than by what the code needs: the KV pool. Its
size follows from admission. The scheduler refuses any request longer than
`min(context_length - 1, max_total_num_tokens - 1)` (omni_scheduler.py:329-332, 1292-1319),
and runs at most `max_running_requests` of them. So the pool the running batch can ever use
is

    need = max_running_requests * context_length tokens

For this profile that is 16 times 8192, 131072 tokens, at the 114688 bytes per token the log
reports (K plus V, 62.92 GiB over 589142 tokens), 14.0 GiB. Anything above that is capacity no
request can occupy. Today the pool holds 589142 tokens, 4.5 times the need, and the 48.9 GiB
between the two is what starves everything else.

The requirement is therefore: the pool covers the admission bound and no more, and the rest
of the card stays free for the allocations the process makes as it serves. No fraction, no
tuned constant: two settings the deployment already owns, multiplied.

## 2. What sizes the pool today, and why the card fills

The Base profile declares no memory setting (`examples/configs/qwen3_tts_1_7b.yaml` carries
two keys), so the pool comes from sglang's profile with the builder's
`mem_fraction_static: 0.85` (engine_builder.py:91):

    pool_bytes = free_after_weights - pre_model_load_free * (1 - 0.85) - multimodal_reserve
                                                                (v0.5.18 kv_cache_configurator.py:1764-1811)

which leaves `pre_model_load_free * 0.15`, 11.79 GB in the log, for everything in section 1
that comes after the pool. What actually comes after it, from the memory samples of the
validation archives:

| Point | GPU total used | Free of 81079 MiB |
| --- | ---: | ---: |
| ready, c1 and c16 alike | 75167 to 75183 MiB | about 5.9 GB |
| c16 peak, either arm | 80343 to 81055 MiB | 24 MiB to 736 MiB |
| c16 peak with 39 predictor graphs and lazy captures | card full | cuDNN conv1d fails at 6 MiB free |

So the process needs about 14 GB beyond weights and pool at c16, the 0.85 rule reserved 11.8,
and the difference is exactly the margin that vanished. The symptoms in the archives follow
from that alone: the caching allocator's retry warnings on both arms at c16 (it frees cached
blocks and retries, torch CUDACachingAllocator.cpp:1778-1792 and 3933), the rope store c16
failing inside cuDNN's SDPA plan build in the reference encoder, and the 128 running run
failing inside cuDNN's conv1d in the speaker encoder. cuDNN allocates its plans and workspaces
at first use of a shape, outside the torch pool, so it is the first thing to fail when the card
is full, whatever the operator.

The other five knobs on this path and what they do, so the plan touches only the one that
matters:

| Knob | Where | Effect |
| --- | --- | --- |
| `engine.kv_cache_bytes` | schema.py:128-142, sglang_model_runner.py:93-120 | operator declares the pool in bytes, authoritative, drops the builder fraction (engine_factory.py:138-145) |
| `engine.max_total_tokens` | schema.py:120, v0.5.18 kv_cache_configurator.py:1844-1871 | a cap on the profiled token count, the pool is then allocated at the capped size (config_from_budget, 1946-1968) |
| `engine.mem_fraction_static` | schema.py, engine_builder.py:91 | the 0.85 above, a user value is refused, the builder's is the one in force |
| `gpu_memory_fraction` per stage | schema.py:331-341, runtime.py:152-183 | never reaches this factory, it declares no such parameter (stages.py:187-197) |
| `total_reserve_bytes` per stage | schema.py:342-355, stage_workers.py:810-845 | a per process torch allocator cap, unset here |

The schema already refuses `kv_cache_bytes` together with `max_total_tokens` (schema.py:154-160)
because the lower token cap would silently shrink the byte pool. That rule shapes the seam.

## 3. Design

One change at the builder: derive the token cap from the admission bound after the overrides
are merged, when no byte budget is declared.

```
Qwen3TtsEngineBuilder.adjust_overrides(overrides)           engine_builder.py:146
    existing: refuse enable_torch_compile
    new:      if peek_stage_kv_cache_bytes() is None:
                  overrides.setdefault(
                      "max_total_tokens",
                      overrides["max_running_requests"] * self.context_length,
                  )
```

Why there and not in `generation_defaults`: the defaults are merged before a deployment's
`max_running_requests` is known (`build_generation_batch_overrides`, generation_batch_policy.py:99-217,
`{**stage_defaults, **incoming}`), and `build` resolves `self.context_length` before it calls
`adjust_overrides` (engine_factory.py:100-127). At that point both factors are the resolved
ones. `setdefault` keeps a deployment's own `max_total_tokens`, and the byte budget check keeps
the schema's rule: a stage that declares `kv_cache_bytes` gets no token cap from the builder,
the same way it gets no fraction.

What the cap does downstream, all upstream semantics: `_apply_token_constraints` takes the
minimum of the profiled capacity and the cap, `config_from_budget` recomputes the pool at the
capped count, and the allocation happens at that size. Everything past the pool then sees
about 60 GB free instead of 11.8.

The fraction. With the cap binding, `mem_fraction_static: 0.85` decides nothing on this
profile. It would decide only when need exceeds what the card can hold (a running cap of 128
needs 112 GiB), where sglang's warning applies, the profiled value wins, and the pool is
smaller than admission. The 0.85 is a constant no measurement pins, so the plan drops it and
lets sglang derive its own reserve (v0.5.18 server_args.py:4955-4980: 512 MB plus 1.5 MB per
activation token plus the graph reserve, floor 10 GB above 60 GB of VRAM), which on this card
lands near 0.84. Decision 2 in section 7 is whether to drop it in this slice or keep it and
retire it separately.

Coverage log. When the cap does not fit, the deployment should read it at startup rather
than discover retraction under load. One omni owned line after engine construction, on the
existing `Memory pool end` information: the pool's token count against the need, so a
deployment sees `131072 of 131072` or `589142 of 1048576`.

### 3.1 Memory map, before and after, at c16 on an 80 GB H100

```
before (0.85 fraction)                          after (cap = 16 x 8192 tokens)
+------------------------------+ 81.1 GB        +------------------------------+ 81.1 GB
| free at c16 peak: 24-736 MiB |                | free at c16 peak: ~48 GB     |
+------------------------------+                |                              |
| activations, cuDNN plans,    |  ~14 GB        |  cuDNN plans and workspaces, |
| vocoder windows, histories,  |  (needs 14,    |  lazy captures, profiler     |
| graphs, tokenizer copy 2     |   got 11.8)    |  windows, 39 graph ladders,  |
+------------------------------+                |  all land here               |
|                              |                +------------------------------+
|  KV pool 589142 tokens       |  62.9 GB       | activations, graphs, copy 2  |  ~14 GB
|  (need: 131072)              |                +------------------------------+
|                              |                | KV pool 131072 tokens        |  14.0 GiB
+------------------------------+                +------------------------------+
| talker weights               |  3.7 GB        | talker weights               |  3.7 GB
+------------------------------+                +------------------------------+
```

### 3.2 Alternatives read and not taken

- Deriving the bytes inside `_OmniKVCacheConfigurator._profile_available_bytes`
  (sglang_model_runner.py:68-147) for every engine. That changes the pool policy of every
  model in the repo from one place. The requirement is per builder, and this slice is
  Qwen3-TTS.
- A stage `engine.kv_cache_bytes` default in the YAML. It states bytes, which move with the
  checkpoint's head geometry and dtype, while the admission bound states tokens the deployment
  already reasons in, and the byte path refuses the fraction and the cap by schema.
- Turning off cuDNN attention in the stage processes (`perf/qwen3-tts-cudnn-attention`, held).
  The failing call at 128 running was a convolution, so it is not an attention problem, and the
  branch holds on a measured regression.

## 4. What the freed memory unblocks

- The larger graph ladder: 39 predictor graphs at 128 running plus sglang's 20, and the lazy
  captures a client's sampling values trigger, without the card reaching cuDNN's failure.
- The rope store slice's c16 point, which failed in the reference encoder with the card two
  MiB from full on both arms.
- Profiler windows at c16 without allocator retries, and the retraction memory measurement.
- Colocating a second stage process on the card, which the placement check today skips only
  because this profile is one process (topology.py:456-458).

## 5. Validation

Unit, on the builder: the cap is derived from the merged running cap and the resolved context
length, a deployment's own `max_total_tokens` wins, a declared `kv_cache_bytes` yields no cap,
and the values are the resolved ones for an off grid running cap such as 89.

Box, with the paired protocol of plan 07 (interleaved boots, all GPU sample kept):

1. Startup log: `KV Cache is allocated ... #tokens: 131072` and `Memory pool end` with about
   59 GB available, ready memory about 27 GB, in place of 75.2.
2. c1 full corpus against the chain head: byte identical WAVs, the pool size does not touch a
   kernel. Latency and qps within the paired spread.
3. c16 full corpus, two boots per arm: no allocator retry warning in either B boot, quality
   inside the identical kernel band, peak memory about 32 GB.
4. The 128 running server with the request level subtalker top k 64: 64 of 64 complete, six
   lazy captures, no cuDNN error.
5. The retraction run at c16 with the profiler memory flag, after confirming the flag reached
   the stage process (`/proc/<pid>/environ`): allocator peak recorded.
6. The rope store branch rebased on the chain head, its c16 pair on top of this slice.

## 6. Slices

1. This slice, one PR: `[Qwen3-TTS] Size the KV pool from the admission bound`. Diff:
   `adjust_overrides` in engine_builder.py, the coverage log line, the tests, and the fraction
   line removed if decision 2 says so. No new constant, no new configuration key.
2. The rope store, rebased, with its own A/B.
3. Later, separately measured, none required by this plan: the vocoder's second tokenizer copy
   (stages.py:245-250), the 192 vocoder graph keys (streaming_vocoder.py:54-90, 473-479), the
   reference cache device policy. Each is a memory reduction inside the 14 GB, not a
   provisioning question.

## 7. Decisions

1. The seam: the builder derived token cap in `adjust_overrides` as above. Recommended.
2. Drop `mem_fraction_static: 0.85` in this slice, or keep it and retire it separately.
   Recommended: drop, the profile never depends on it once the cap binds.
3. The coverage log line: keep as one info line, or leave it to sglang's warning alone.

## 8. Validation tasks and open facts

- `/Users/ratish/sglang` now sits at tag v0.5.19 while the chain branch pins 0.5.18. The
  functions this plan cites were read from the v0.5.18 blob. Before implementation, confirm
  which pin main carries and re-read `_apply_token_constraints`, `config_from_budget` and the
  automatic fraction at that tag.
- The 114688 bytes per token is arithmetic on the log, its factorization is not read from the
  checkpoint. The unit test uses the resolved cell size, not the number.
- The 14 GB process need at c16 is inferred from two memory samples (ready and peak). The
  first box run of this slice reads it directly from the new free memory at ready and at peak.
- Whether `max_total_tokens` also caps the prefill graph ladder in a way that matters here:
  generation_batch_policy.py:82-84 and v0.5.18 server_args.py:4928-4931 take the minimum with
  the chunk size, 131072 is above every prefill bucket, so no.
