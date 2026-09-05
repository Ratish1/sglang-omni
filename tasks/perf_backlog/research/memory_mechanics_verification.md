# Verification of the two memory mechanics reports

Verified on 2026-09-06 by re-reading every cited line myself. Sources: torch files fetched raw
from the v2.13.0 tag (version.txt reads 2.13.0a0), the cudnn-frontend headers at the submodule
commit c4a97621eca52fa0c3a1862a411a16be580b25c6 recorded by that tag, the NVIDIA runtime API
page fetched raw, the lazy loading page served as v13.3, the cuDNN backend API page on the
latest channel, the pinned sglang checkout at v0.5.18, and the chain worktree at 0a88253c6.
The S2 results archive supplied the environment record (env/pip_freeze.txt) and the run script.

## torch_cuda_memory_mechanics.md

| Claim | Verdict | Evidence |
| --- | --- | --- |
| allocator is a process local static | CONFIRMED | CUDACachingAllocator.cpp:5303 |
| free_block never calls cudaFree | CONFIRMED | cudaFree appears at 4141 (release_block) and 4520 (uncached delete) only |
| warning site, sticky error cleared, returns false | CONFIRMED | 3927-3954, LOG at 3933, cudaGetLastError at 3947 |
| retry chain releases cached blocks then retries | CONFIRMED | 1778-1792 |
| release_available_cached_blocks inert at default max_split | CONFIRMED | 4005-4007, AllocatorConfig.h:341 |
| release_cached_blocks always true, frees unsplit blocks | CONFIRMED | 4059-4073, 4102, gate at 4265 |
| second failure raises OutOfMemoryError with its own text | CONFIRMED | 1928-1932, bookkeeping at 1799, 1840, 1858, 1903 |
| allocate reaches malloc, no null path | CONFIRMED | 4975 |
| gc threshold inert without a memory fraction | CONFIRMED | 1758-1763, 2550-2561, AllocatorConfig.h:350 |
| expandable segments off by default, failure path logs no warning | CONFIRMED | 3896-3911, AllocatorConfig.h:352 |
| empty_cache is release_cached_blocks | CONFIRMED | 2580-2584 |
| allocation rounding constants | CONFIRMED | 3700-3708, AllocatorConfig.h:16-24, 339 |
| four bare TORCH_CHECK sites around execute, workspace from the caching allocator | CONFIRMED | MHA.cpp:1450-1454, 1575-1576, 1735-1737, 1879-1881 |
| build steps use AT_CUDNN_FRONTEND_CHECK, execute does not | CONFIRMED | MHA.cpp:640-645 |
| bare TORCH_CHECK text matches the logged message | CONFIRMED | Exception.h:524-527 |
| cuDNN first on sm90 when cuDNN > 9.15.0 and CUDA 13 | CONFIRMED | sdp_utils.cpp:76-122, Context.h:480-490, 275 |
| error codes, is_good, message discarded | CONFIRMED | graph_helpers.h:36-74, 124-145 |
| execute paths that return non OK | CONFIRMED | graph_interface.h:1288-1293, 1361-1463, 244-268 |
| OSS engine compiles and loads a module in build, not execute | CONFIRMED | oss_engine_interface.h:164, 243, sm90_sdpa_prefill_engine.h:186-203, 232 |
| cudaMemGetInfo free is device wide and not guaranteed allocatable | CONFIRMED | runtime API page, description paragraph, verbatim including multi-tenet |
| lazy loading allocates module memory at first use, documented failure mode | CONFIRMED | programming guide 4.7, v13.3 page, verbatim |
| cuDNN documents internal device allocation failure | CONFIRMED | backend API page, CUDNN_STATUS_INTERNAL_ERROR sub codes, verbatim |

Open items of that report resolved here: item 6, the runtime cuDNN is 9.20.0.48
(nvidia-cudnn-cu13 in env/pip_freeze.txt), so the predicate at sdp_utils.cpp:97 holds on the
H100 and cuDNN was first in the order unless TORCH_CUDNN_SDPA_DEPRIORITIZED was set, which the
serve log does not show. Item 13, one OS process held the card, see below. Items 1 to 5 and 7 to
12 stay open, the log carries no evidence for them.

## qwen3_tts_stage_memory_mechanics.md

| Claim | Verdict | Evidence |
| --- | --- | --- |
| all three stages share one OS process under the shipped profile | CONFIRMED | config.py:48-75, examples/configs/qwen3_tts_1_7b.yaml (two keys), stage_workers.py:269-299, run.sh passes that yaml |
| colocation footprint check skipped for one process group | CONFIRMED | topology.py:456-458 |
| KV pool is free memory minus pre load memory times 0.15, divided by cell size | CONFIRMED | kv_cache_configurator.py:1764-1811, pool_configurator.py:327-333, 415-424 |
| free memory read device wide after an empty_cache | CONFIRMED | utils/common.py:414-443, distributed/bootstrap.py:132-137 |
| max_total_tokens only caps | CONFIRMED | kv_cache_configurator.py:1844-1871 |
| multimodal reserve is zero unless the flag is set, omni flips it only after sizing | CORRECTED | the log shows 0.10 GB reserved at sizing, so the flag was already set. sglang sets it from the nested text config test at model_config.py:445-458, has_multimodal_subconfig, which is true for this checkpoint. The omni flip at bootstrap.py:56-65 is unrelated to sizing |
| kv_cache_bytes replaces the fraction, fraction cannot reach this factory | CONFIRMED | engine_factory.py:134-145, runtime.py:152-153, 179-182, stages.py:187-197, sglang_model_runner.py:93-120, schema.py:128-160, 331-355 |
| total_reserve_bytes caps the torch allocator per process | CONFIRMED | stage_workers.py:810-845 |
| speech tokenizer loaded twice in the process | CONFIRMED | engine_builder.py:123-129, stages.py:245-250 |
| 16 frame counts times 4 batch sizes times 3 holders | CONFIRMED | streaming_vocoder.py:54-90 recomputed by hand, 473-479, 549-594, matches the serve log's captured list |
| decode graphs captured before readiness | CONFIRMED | stages.py:276-279, streaming_vocoder.py:699-704 |
| streaming window at most left context plus stride | CONFIRMED | streaming_vocoder.py:1047-1057 |
| non streaming decodes whole utterances in one call, batch 8 within 2 ms | CONFIRMED | streaming_vocoder.py:1865-1895, stages.py:223-224 |
| reference codes batched 8 within 2 ms on a private stream, service cache 256 entries 64 MiB host side | CONFIRMED | request_builders.py:716-750, 774-795, 1028-1034, reference_encoder.py:57-58, 141, 172 |
| predictor graph holds three static buffers, one shared pool | CONFIRMED | sglang_model.py:134-141, 1241-1246 |
| history rows are views of one clone per step, compacted only at retraction | CONFIRMED | model_runner.py:342-346, omni_scheduler.py:87-94, 2197-2200 |

Open items of that report resolved here: item 7, the run used the shipped profile, the yaml
sets only the config class and the model path. Item 3 stays open, the 114688 bytes per token
factorization is arithmetic on the log. Items 1, 2, 4, 5, 6, 8 and 9 stay open.

## Correction to my own earlier statement

I told the user the three stages were three processes with three caching allocators. They are
one process with one caching allocator. The consequence is stronger, not weaker: every torch
allocation of the three stages shares one reserved pool, and a library allocating outside that
pool, cuDNN's internal buffers or a lazily loaded module, sees only what the pool has not taken.
