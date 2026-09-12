# Sector 10 — upstream SGLang scheduler contracts used by Omni

## Scope and baselines

This report is a source trace of SGLang `v0.5.19` at `0bcd822377da7b5718e674eaf9c870d349424dd1` against the Omni worktree at `645b472cdb2d7b93a1825a5bfa6a603b62b03936`. It covers the complete upstream scheduler, policy, and batch files; every `scheduler_components` implementation directly instantiated by `OmniScheduler`; the prefix-cache interface and radix implementation; the request-row and token-slot allocation contracts; and the Omni composition/event-loop code that invokes them.

The complete-file evidence is in `10_upstream_scheduler.coverage.json`. It includes all 5,071 lines of `memory_pool.py`: request-row mappings, recurrent-state pools, physical MHA/MLA storage, quantized variants, and composite sparse/hybrid owners. Runtime selection and construction of a concrete physical KV storage family remain model-runner/configurator boundaries.

## Composition and ownership

`OmniScheduler` is a composition shell, not a subclass of upstream `Scheduler` (`omni_scheduler.py:167-2735`). Its `__getattr__` looks up an attribute on the upstream class and binds upstream callables to the Omni instance with `types.MethodType` (`732-756`). Consequently, every delegated upstream method executes against Omni-owned fields. Omni initializes the matching scheduler state itself (`182-717`), including queues, batch pointers, cache/pools, scheduling limits, policy state, overlap state, metrics state, and the component objects required by delegated methods.

`_init_upstream_scheduler_components` (`599-717`) constructs and owns these upstream objects:

- `SchedulerDPAttnAdapter`, `SchedulerPoolStatsObserver`, `SchedulerLoadInquirer`, `SchedulerBatchResultProcessor`, `LogprobProcessor`, `BeamCoordinator`, `SchedulerMetricsReporter`, and `SchedulerLoadPublisher`.
- A small `output_streamer` namespace whose `_stream_output_generation` callback points to Omni `stream_output`; Omni does not instantiate upstream `SchedulerOutputStreamer` for normal output delivery.
- An adapter callback from the result processor's abort path to `self.abort(request.rid)`.

Omni directly implements `recv_requests` (`823-863`), `process_input_requests` (`865-938`), its request-building/backlog pipeline (`940-1364`), `get_next_batch_to_run` wrapper (`1366-1378`), `get_new_batch_prefill` override (`1380-1414`), `run_batch`/`_run_batch` (`1416-1454`), `process_batch_result` wrapper (`1423-1428`), output emission (`1467-1737`), and all three event loops (`2348-2735`). Other scheduler mechanics are delegated through `__getattr__`.

The central reachable call graph is:

```text
Omni event loop
  -> Omni recv_requests / process_input_requests
  -> Omni get_next_batch_to_run
       -> upstream Scheduler.get_next_batch_to_run
          -> process_pending_chunked_abort / timeout handling
          -> get_new_batch_prefill (dynamic dispatch back to Omni override)
             -> optional Omni coalescing hold
             -> upstream Scheduler.get_new_batch_prefill
                -> upstream _get_new_batch_prefill_raw
                -> SchedulePolicy.calc_priority
                -> PrefillAdder admission
                -> ScheduleBatch.init_new / prepare_for_extend
          -> upstream update_running_batch
             -> ScheduleBatch.check_decode_mem / retract_decode / prepare_for_decode
          -> SchedulerDPAttnAdapter and ngram hooks
  -> Omni run_batch -> Omni _run_batch -> model_runner.execute
  -> Omni process_batch_result
       -> upstream Scheduler.process_batch_result
          -> SchedulerBatchResultProcessor
             -> prefill/decode result commit, finish, cache/release, output callback
       -> Omni prompt-only radix insert flag
  -> Omni stream/output callback -> stage result channel
```

The model runner owns tensor execution and physical K/V writes. The scheduler owns admission, the request-to-slot mapping, allocation timing, batch metadata, request lifecycle state, cache insertion/locking, and release timing. The stage transport owns incoming `ScheduleRequest` objects and outgoing stage results. The upstream detokenizer transport is replaced by the Omni callback namespace.

## Scheduling state machine

### Request admission and priority

`Scheduler.get_next_batch_to_run(running_batch, last_batch)` (`scheduler.py:3335-3482`) is the upstream selection entry point Omni calls. It first processes a pending chunk abort and queued/running timeouts. If the previous batch was extend, it excludes the active `chunked_req`, caches the already-computed portion when that request grew, filters finished rows, and merges surviving rows into the running decode batch. Prefill-only rows are filtered from the running batch.

The method then tries diffusion/PD-specific paths and a new prefill batch. When no prefill work is admitted, `update_running_batch` (`3869-3951`) filters the decode batch, checks token capacity, retracts rows if required, and calls `ScheduleBatch.prepare_for_decode`. The DP-attention adapter may synthesize an idle/synchronization batch or convert decode to a one-token extend representation. The ngram manager receives every selected batch before return. The return contract is exactly `NextBatchPlan(batch_to_run, running_batch)` (`schedule_batch.py:3661-3663`). Omni's wrapper writes `plan.running_batch` back to `self.running_batch` and returns only `plan.batch_to_run` to its batch-owning loop.

`SchedulePolicy.calc_priority` (`schedule_policy.py:240-291`) validates/selects the active policy and mutates the waiting queue order. FCFS is cache-agnostic. LPM and DFS-weight first call `match_prefix_for_req`, storing `prefix_indices` and `last_node`, then sort by prefix length or tree traversal. LPM falls back to FCFS when the queue exceeds 128 requests. Priority scheduling adds priority and arrival-time ordering. Routing-key policy groups by routing key and delegates ordering inside groups.

`Scheduler._get_new_batch_prefill_raw` (`scheduler.py:3531-3839`) computes allocatable request rows, refreshes priorities/cache matches, builds `PrefillAdder`, attempts the active chunk first, scans the waiting queue, removes admitted rows, restores preempted rows, updates `chunked_req`, and creates/prepares an extend batch. The request-row budget is `get_num_allocatable_reqs` (`3484-3502`): PP microbatch limits and the free `ReqToTokenPool` rows bound the result, with pending beam member rows subtracted.

`PrefillAdder` (`schedule_policy.py:478-1476`) is the admission ledger. Its initial token budget is allocator availability plus cache-evictable slots, minus the running batch's estimated future decode need. The estimate uses each running request's remaining generation budget, clipped for estimation at 4096 tokens, and accounts for page granularity and model-specific SWA/Mamba budgets. Separate counters enforce `max_prefill_tokens`, chunk size, request-row count, and optional tile limits. Cache node locks are acquired during candidate admission because moving a node from evictable to protected changes the available budget. `add_one_req` returns `CONTINUE`, `NO_TOKEN`, or `OTHER`; first-request exceptions permit forward progress under selected budget conditions.

Omni's `get_new_batch_prefill` performs its coalescing decision before delegation. If coalescing is disabled, bypassed, satisfied, or its oldest-request deadline has expired, it calls `_Upstream.get_new_batch_prefill(self, running_batch)`; that upstream public method constructs any prefill-delayer pass, calls `_get_new_batch_prefill_raw`, finalizes prefill-delay observations, and returns `NextBatchPlan`. Only the hold case is produced directly by Omni as `NextBatchPlan(batch_to_run=None, running_batch=running_batch)`. Request construction, deferred model preparation, build futures, backlog limits, stage-priority normalization, prompt-cache epoch handling, and early ingress buffering are separately Omni-owned ingress mechanics.

### Chunked prefill

The active partial request is held in `Scheduler.chunked_req`. Before scheduling its next piece, upstream `stash_chunked_request`/`get_next_batch_to_run` commits the already-computed portion with `tree_cache.cache_unfinished_req`. `PrefillAdder.add_chunked_req` (`schedule_policy.py:973-1022`) calls `Req.init_next_round_input`, chooses the next extend range from the current token budget, updates all ledgers, and returns the same request only when another prompt portion remains.

Scheduling a middle chunk increments `inflight_middle_chunks`. `ScheduleBatch.prepare_for_extend` (`schedule_batch.py:2504-2761`) creates host/device input tensors, allocates request rows and KV slots, writes cached prefix indices followed by new slot indices into `ReqToTokenPool.req_to_token`, and advances `kv_allocated_len` and `kv_committed_len` to the scheduled sequence length before the forward.

`SchedulerBatchResultProcessor.process_batch_result_prefill` (`batch_result_processor.py:240-864`) waits for the staged result copy when present. A middle chunk decrements `inflight_middle_chunks`, does not append a sampled output token, and does not expose that token to output streaming. The final chunk appends/commits the sampled token, updates finish state, and transitions the request into decode unless it finishes. An unfinished request is cached immediately except where the mixed-batch/decode or explicit skip-insert contract owns it. A mixed extend can include running decode rows when configured; those rows retain decode semantics inside result processing.

### Decode, capacity, and retraction

`ScheduleBatch.prepare_for_decode` (`schedule_batch.py:3286-3374`) allocates the next token slot for every active request, writes it at the old sequence length in the row mapping, and increments sequence, allocated, and committed lengths before forward execution. Speculative and hybrid-cache modes use specialized size/allocation branches; the ordinary path requires one slot per row.

`ScheduleBatch.check_decode_mem` (`3019-3028`) asks the allocator whether the next step fits and can evict prefix-cache entries first. `retract_decode` (`3030-3114`) orders candidates by output-length/cache criteria, repeatedly releases request KV and request rows until the remaining batch fits, and returns retracted and aborted lists plus a new-token ratio. A retracted request is reset and placed back into the waiting queue by upstream `update_running_batch`. Priority preemption uses the same release/reset contract through `PrefillAdder.preempt_to_schedule`.

`filter_batch` (`3376-3461`) filters `Req` objects and every request-aligned tensor/list. `merge_batch` (`3463-3529`) concatenates those structures. `copy` (`3531-3576`) snapshots only the fields needed by delayed result processing while sharing the underlying `Req` objects; this shared-object fact is part of the overlap contract.

## Forward/result ordering and stream ownership

Upstream normal scheduling (`Scheduler.event_loop_normal`, `scheduler.py:1808-1840`) receives input, selects a batch, runs it, and processes that result before selecting the next batch. Upstream overlap (`1843-1915`) queues `(batch.copy(), result)`, launches the current forward, and processes the preceding result one iteration later. `FutureMap` owns next-token relay between the scheduler and model streams, and `_apply_war_barrier` protects write-after-read ordering. Upstream overlap state assumes `last_batch`, `cur_batch`, and delayed result processing advance together.

Omni's `_event_loop_overlap` (`omni_scheduler.py:2381-2397`) raises `NotImplementedError`: its source states that chunk accounting would lag one iteration. The FunCosyVoice3 builder sets `disable_overlap_schedule=True` (`sglang_omni/models/fun_cosyvoice3/engine_builder.py:65-87`), so its default path uses `_event_loop_normal` (`2348-2379`).

Omni also has a separate one-step `_event_loop_async_decode` (`2560-2647`). It launches an eligible decode step, stores a `ScheduleBatch.copy` plus model-runner pending state, and resolves the preceding step before consuming its result. Prefill and ineligible decode paths flush pending work first. Because the copied batch shares `Req` objects, the resolution code filters stale/finished/retracted rows before result processing. `enable_overlap` and `enable_async_decode` are mutually exclusive at initialization (`295-300`). This async path is Omni-owned and does not call upstream `event_loop_overlap`.

`OmniScheduler.run_batch` calls the custom runner and adapts its return into upstream `GenerationBatchResult`. It supplies next-token IDs and graph status but no `logits_output`. Thus the reachable FunCosyVoice path supplies the fields used by ordinary token commit, while upstream logprob, hidden-state, beam-selection, routed-expert, and sampling-mask result paths require model-runner fields that this adapter does not populate. Those optional feature contracts remain model/stage configuration boundaries.

`OmniScheduler.process_batch_result` delegates to upstream `Scheduler.process_batch_result` (`scheduler.py:4357-4398`), which runs trace/load publication, dispatches to `SchedulerBatchResultProcessor`, records counters/metrics, clears multimodal inputs, emits health state, and updates the device timer. After that delegation, the Omni wrapper only sets `skip_radix_cache_insert=True` on requests that have output and the `_omni_prompt_only_radix` marker. It does not emit a stage stream event. The component's normal prefill/decode paths update the `Req`, decide finish, cache or release KV, and invoke the configured stream callback while grouped frees are active.

Omni `_run_batch` calls `_emit_stream_output` after synchronous model execution and before it returns the adapted result to the event loop (`1438-1454`). `_emit_stream_output` (`1467-1484`) acts only when a `stream_output_builder` exists; FunCosyVoice3 supplies none, so that construction's generic path is inactive. Omni `stream_output` (`1628-1737`) is the upstream component callback and owns terminal stage-result delivery plus request cleanup. When a builder is configured, per-forward emission occurs in `_run_batch` before result commit; terminal cleanup occurs through `stream_output` after finish state and cache/release decisions. The upstream `SchedulerOutputStreamer` duplicate-finish guard and detokenizer payload accumulator are not the active Omni egress implementation.

## KV cache, request rows, and release contracts

`ReqKvInfo` (`schedule_batch.py:849-901`) is the scheduler's per-request allocation record: `req_pool_idx`, `cache_protected_len`, `kv_committed_len`, `kv_allocated_len`, hybrid-cache state, and release flags. `kv_allocated_len` describes owned row-mapping slots; `kv_committed_len` bounds content eligible for cache insertion. Preparation advances both before the forward. Result processing and abort/retraction paths use the effective committed length when deciding what can enter the prefix cache.

`ReqToTokenPool` (`memory_pool.py:256-368`) owns the GPU `int32` request-row matrix `(size + 1, max_context_len)`. Row zero is padding. Host-side free indices start at one; per-row generation counters distinguish recycled rows. `alloc(reqs)` binds `Req.kv.req_pool_idx`, `alloc_rows` supports bare beam rows, and `free/free_rows` return ownership and clear request binding.

`BaseTokenToKVPoolAllocator` (`allocator/base.py:23-189`) defines slot availability, ordinary/extend/decode allocation, free operations, grouped frees, eviction-to-fit, and decode-capacity checks. `TokenToKVPoolAllocator` (`allocator/token.py:28-83`) owns a flat GPU free-slot vector with slot zero reserved for padding. `PagedTokenToKVPoolAllocator` (`allocator/paged.py:105-346`) owns free page IDs, page-aware extend/decode allocation, deduplicated page freeing, and grouped free buffering. Runtime `page_size` selects the concrete allocator; the scheduler consumes only this interface.

`BasePrefixCache` (`base_prefix_cache.py:284-509`) defines matching, finished/unfinished insertion, locking, eviction accounting, optional host-load, event, session, SWA, and Mamba traits. `RadixCache` (`radix_cache.py:303-846`) owns `TreeNode` (`238-300`) and the token-key to KV-slot tree. Matching is page-aligned and may split a node. Lock increments move node weight from evictable to protected accounting; decrements reverse it. Eviction walks unlocked leaves, calls the supplied allocator free callback, deletes leaves, and records events.

`RadixCache.cache_unfinished_req` (`516-584`) inserts the committed fill prefix, frees duplicate newly allocated slots, rematches the canonical tree node, rewrites the request row to canonical cache indices, transfers the node lock, and updates `prefix_indices`, `last_node`, and `cache_protected_len`. `cache_finished_req` (`459-514`) inserts the committed sequence, frees duplicate/uninserted and overallocated tails, and drops the request's lock. `release_req` (`schedule_batch.py:2036-2070`) selects cache-finished versus direct-free behavior, frees `[committed, allocated)` over-allocation, releases optional Mamba state, returns the request row, and marks the record released.

Omni's cache factory (`sglang_backend/cache.py:1-43`) returns upstream `ChunkCache` when radix caching is disabled, Omni `EvictHeapRadixCache` for LRU, and upstream `RadixCache` for other policies. `EvictHeapRadixCache` (`evict_heap_radix_cache.py:1-76`) changes LRU leaf selection to a persistent heap with stale-entry validation; its release/event semantics remain those inherited from `RadixCache`.

The physical storage base is `KVCache` (`memory_pool.py:1670-1805`). It owns common capacity, page size, dtype/storage dtype, layer range, memory-saver/custom-pool handles, transfer synchronization, CPU-copy doors, and the abstract per-layer read/write contract. `MHATokenToKVPool` (`1808-2919`) is the ordinary Qwen2-compatible physical owner: construction allocates per-layer K/V buffers or post-capture virtual-memory owners, records pointer/stride tables, optionally creates an alternate copy stream, and reports capacity. The default NHD layout stores `(size + page_size, heads, dimension)`; HND and ROCm vectorized layouts alter the physical shape while preserving token/page locations. Slot zero's padded page absorbs dummy writes.

Attention writes enter through `set_kv_buffer` (`2374-2454`). The pool validates the kernel-facing location, maps global to local layer ID, applies quantization or dtype conversion, handles DCP masks and HND page/offset addressing, then stores into the selected physical rows. `KVWriteLoc` (`1590-1625`) carries the generic location plus optional SWA/full-subpool locations; the location has already been translated to the kernel-facing ID space before the pool receives it. Prefix-valid stores write only each row's committed prefix. KV relocation copies every layer's K/V rows, and quantized pools move their scale rows with the data. CPU offload/load synchronizes around chunked row copies. PD registration exposes buffer pointers, final backed spans, and page-sized item lengths.

Physical variants are explicit owners rather than scheduler data structures: `NoOpMHATokenToKVPool 2922-3033` keeps logical capacity with placeholders for prefill-only execution; `MHATokenToKVPoolFP4 3036-3187` and `MHATokenToKVPoolMXFP8 3260-3618` own packed/scaled data; the static `PageMajorMHATokenToKVPool 3190-3257` is non-constructible; `MLATokenToKVPool 3968-4253` owns combined latent/rope buffers; FP4 MLA and DSA add their scale/index state; and `MiniMaxSparseKVPool 4708-5071` composes main and index sub-pools. `HybridLinearKVPool 3621-3965` dispatches full-attention layers to MHA/MLA storage and linear layers to `MambaPool`; `HybridReqToTokenPool 1192-1586` couples request rows with Mamba slot allocation and frees both the primary and optional ping-pong slots.

`MambaPool` (`370-1189`) owns convolution and temporal state, optional speculative intermediate buffers, ReplaySSM rings/cursors, copy/clear, CPU movement, and RDMA descriptors. Its slot IDs are allocated by the hybrid request-pool owner. A newly assigned Mamba slot resets ring cursors; release returns the Mamba slot and any nonretained ping-pong slots. These physical pools are constructed by the model runner/cache configurator and written by attention backends. The scheduler consumes their capacity and slot indices through the allocator and request-row mapping contracts.

## Abort, finish, and cleanup

Upstream `Scheduler.abort_request` (`scheduler.py:4966-5113`) searches queued/disaggregated/session states. A queued request can be removed and output immediately. A running request receives `to_finish=FINISH_ABORT`; normal processing observes it after the forward, sets the terminal reason, releases resources, and streams the terminal result. A chunked request is placed in `pending_abort_chunked_req`; `process_pending_chunked_abort` (`3248-3289`) releases it at a scheduling boundary without inserting its partial suffix and clears chunk state. Beam groups add member-row and shared-slot teardown through `BeamCoordinator` (`beam_search/coordinator.py:101-534`).

Omni's public abort path is stage-owned: the component callback maps to `self.abort(rid)`, and Omni maintains tombstones/callbacks plus build/backlog cancellation. For a live running request it marks `to_finish`; for immediately removable work it updates finish state, releases resources, and emits cleanup itself. Normal terminal cleanup remains in Omni `stream_output`, which removes scheduler maps, flushes output-builder state, and releases stage-side resources. This is an explicit composition boundary rather than delegation to upstream detokenizer/session routing.

## Components and observability

`SchedulerBatchResultProcessor` (`batch_result_processor.py:79-1376`) owns result-to-request state transitions, finish checks, KV cache insertion/release, grouped frees, auxiliary/logprob consumption, beam commits, speculative counters, and the output callback. `LogprobProcessor` (`logprob_result_processor.py:73-377`) owns prompt/output logprob slicing and accumulation when requested. `BeamCoordinator` owns beam admission, member rows, device-side selection/relay, orphan-slot reclamation, group finish, and group abort.

`SchedulerDPAttnAdapter` (`dp_attn.py:509-596`) mediates DP-attention batch preparation and load synchronization. `SchedulerLoadInquirer` (`load_inquirer.py:34-237`) derives `LoadSnapshot` from running/waiting queues and pool usage. `SchedulerPoolStatsObserver` (`pool_stats_observer.py:146-335`) samples allocator/cache/session pool counts. `NewTokenRatioTracker` (`new_token_ratio_tracker.py:10-51`) updates the decode reservation ratio after prefill, retraction, and decode progress.

`SchedulerMetricsReporter` (`metrics_reporter.py:123-1242`) records prefill/decode throughput, queue counts, cache-hit windows, pool use, retractions, speculative acceptance, optional device occupancy/FPM estimates, LoRA/HiCache state, and idle transitions. `SchedulerLoadPublisher` (`load_publisher.py:114-267`) publishes a throttled ZMQ `LoadStat` gauge from `SchedulerLoadInquirer`; it is best-effort and disabled without a bound socket. `SchedulerKvEventsPublisher` (`kv_events_publisher.py:42-103`) publishes cache events from `tree_cache.take_events()` and derives KV metrics from reporter stats.

Omni instantiates these objects and delegated upstream code calls them. Omni additionally records stage queue/build/prefill/first-emission/model-path events and exposes stage queue/build/running counts through its admin path. Upstream metrics observe scheduler-visible queue, batch, and allocator state; stage backlog/build-future state is visible only through Omni instrumentation.

## Exact API inventory

The following ranges cover all top-level classes/functions in the three required upstream files. Imports, module constants, aliases, type-checking blocks, and comments outside these symbols were read as complete file ranges and are grouped here as module scaffolding.

- `scheduler.py`: `_prewarm_hccl_group 357-360`; `_MultimodalInputBroadcast 378-380`; `_MultimodalInputProcessingError 383-384`; `_accumulate_decode_moment 387-403`; `Scheduler 410-5432`; `dispatch_event_loop 5435-5462`; `configure_scheduler_process 5465-5530`; `run_scheduler_process 5533-5621`; `_make_abort_req 5624-5635`. `Scheduler` method spans are `__init__ 424-703`, initialization/configuration `705-1713`, timeout/query helpers `1715-1754`, loops `1756-1963`, input/component initialization `1966-2370`, request construction/admission `2372-3243`, chunk/scheduling `3245-4010`, execution/results `4013-4430`, idle/cache/admin state `4432-4955`, abort/pause/control `4957-5432`.
- `schedule_policy.py`: `_ceil_div 111-112`; `estimate_prefill_extend_tile_metrics 115-138`; `match_prefix_for_req 141-200`; `CacheAwarePolicy 203-207`; `CacheAgnosticPolicy 210-216`; `SchedulePolicy 219-469` (`__init__ 222-238`, priority/matching/sorts `240-469`); `AddReqResult 472-475`; `PrefillAdder 478-1476` (construction/budgets `479-924`, lock/chunk paths `926-1039`, admission `1041-1404`, preemption `1406-1476`).
- `schedule_batch.py`: helper/finish/modality symbols `168-327`; `MultimodalDataItem 335-536`; `MultimodalProcessorOutput 539-621`; `MultimodalInputs 625-817`; `ReqLogprob 822-845`; `ReqKvInfo 849-901`; `Req 904-1975`; Mamba helpers `1978-2033`; `release_req 2036-2070`; `retract_all 2073-2091`; extend helpers `2094-2127`; `ScheduleBatch 2131-3658`; `NextBatchPlan 3661-3663`. Within `ScheduleBatch`: construction/introspection `2325-2376`, extend preparation `2378-2879`, split/mixed/converted modes `2881-2988`, decode capacity/retraction/release `2990-3175`, idle/Mamba/penalty preparation `3177-3284`, decode preparation `3286-3374`, filter/merge/copy `3376-3576`, SWA eviction `3578-3652`, formatting `3654-3658`.
- Prefix-cache interfaces: `PrefixCacheTrait 42-46`, parameter/result records `50-224`, `zero_match_result 227-245`, `_dfs_weight_order 248-281`, `BasePrefixCache 284-509`; concrete `RadixKey 59-235`, `TreeNode 238-300`, `RadixCache 303-846`.
- Physical pool file: helpers/kernels `117-253`; `ReqToTokenPool 256-367`; `MambaPool 370-1189`; `HybridReqToTokenPool 1192-1586`; `KVWriteLoc 1590-1625`; `unwrap_write_loc 1628-1632`; `KvBufferDesc 1635-1667`; `KVCache 1670-1805`; `MHATokenToKVPool 1808-2919`; `NoOpMHATokenToKVPool 2922-3033`; `MHATokenToKVPoolFP4 3036-3187`; `PageMajorMHATokenToKVPool 3190-3257`; `MHATokenToKVPoolMXFP8 3260-3618`; `HybridLinearKVPool 3621-3965`; `MLATokenToKVPool 3968-4253`; `MLATokenToKVPoolFP4 4256-4393`; `DSATokenToKVPool 4396-4542`; move kernel/helper `4545-4606`; `MHATokenToKOnlyPool 4609-4705`; `MiniMaxSparseKVPool 4708-5071`.
- Directly instantiated component classes: `SchedulerBatchResultProcessor 79-1376`, `SchedulerDPAttnAdapter 509-596`, `SchedulerLoadInquirer 34-237`, `SchedulerPoolStatsObserver 146-335`, `NewTokenRatioTracker 10-51`, `LogprobProcessor 73-377`, `SchedulerMetricsReporter 123-1242`, `SchedulerLoadPublisher 114-267`, `SchedulerKvEventsPublisher 42-103`, `BeamCoordinator 101-534`. The upstream `SchedulerOutputStreamer 40-300` and `_GenerationStreamAccumulator 303-774` were read to establish the boundary replaced by Omni.

## Facts and unresolved external boundaries

Established facts:

- Omni already calls upstream batch selection, the public prefill scheduler and its internal raw admission path, decode update/retraction, batch preparation, result processing, cache operations, metrics, load reporting, DP-attention adaptation, and beam coordination through method binding or component instances.
- Omni owns ingress/build queues, stage-aware admission metadata, its event loops, custom model-runner adaptation, stage emission, final callbacks, and stage-resource cleanup.
- The scheduler commits row/slot ownership before forward execution and releases or canonicalizes it only in result, abort, retraction, or cache-eviction paths.
- Chunked prefill has a single active `chunked_req` and an `inflight_middle_chunks` counter whose scheduling and result-processing sides must remain paired.
- Upstream overlap assumes delayed result processing plus token relay; Omni's default FunCosyVoice configuration disables it. Omni async decode is a distinct one-step pipeline.

Unresolved externally owned boundaries:

- Runtime server arguments determine radix disablement, cache policy, page size, allocator selection, and physical KV layout. The FunCosyVoice builder establishes Qwen2, context length 4096, `max_running_requests=32`, `max_prefill_tokens=4096`, and disabled upstream overlap, but it does not fix every cache argument.
- The custom model runner owns actual forward stream execution, physical K/V tensor writes, staged next-token copy completion, and async pending-state correctness.
- Stage request/output schemas, builder futures, prompt-cache epochs, and downstream cleanup callbacks are owned outside upstream SGLang.
- Optional upstream result features that require logits or auxiliary tensors depend on stage/model configuration; the Omni result adapter shown here supplies no `logits_output`.
