# Qwen3-TTS memory provisioning plan

Revision 3, Sep 7 2026, after the design review in
`tasks/qwen3_tts_memory_provisioning_review_20260907/` (review.md and mechanics.md). Every
finding of that review was re-read against the sources it cites and holds, and each is folded
in below with its number. One PR carries the whole thing, the rope store lands after it on its
own branch. Sources: the branch `perf/qwen3-tts-kv-pool-admission-bound` at `52ee606fa`, merged
with main `53e94dfa5`, sglang v0.5.18 at `/Users/ratish/sglang-worktrees/v0.5.18`, torch
v2.13.0 at `/Users/ratish/pytorch`, qwen-tts 0.1.1 from the review's evidence tarball, which is
the version the box runs (`env/pip_freeze.txt` of the kv pool archive).

## 1. The origin of the problem

The process is one OS process with three stages built in config order, preprocessing,
tts_engine, vocoder (stage_workers.py:493-500). Memory is provisioned once, when sglang sizes
the KV pool inside the engine stage's factory (bootstrap.py:180). sglang sizes it from device
free memory after `empty_cache` (v0.5.18 utils/common.py:433-443), minus a slack it derives
for an LLM's own activations and graphs (server_args.py:4956-4983, kv_cache_configurator.py:
1764-1811). That rule sees what is resident at that moment and nothing else.

Two things are wrong with the moment, and they are the origin. Everything else in the archives
is a symptom of them.

1. Resources of the same process are created after the pool is sized: the engine's tokenizer
   copy in `setup_model` (engine_builder.py:126-132), the vocoder's tokenizer copy and its 192
   decode graphs in the vocoder factory (stages.py:245-279). The graph pools alone are 4.5 GiB,
   three private pools of 1544 MiB, one per capture stream, proven by pool id in the kv pool
   snapshot. sglang never sees them, so its slack is spent on them before a request arrives.
2. No transient of the non LLM components is accounted anywhere. The reference encoder, the
   speaker encoder and the whole utterance decode allocate at request time out of whatever is
   left, and sglang's slack is a formula for LLM activations, not for them.

At 16 running the admission cap on this branch hides both, the pool is 14 GiB and 48 GB stays
free. At 128 running the cap does not bind, the pool takes the card again, the vocoder's graph
pools and transients fill the rest, and the run failed with `CUDNN_STATUS_INTERNAL_ERROR` at
6 MiB free (revalidation LIMITATIONS.md:9). Which allocation failed is not proven, cuDNN's
convolution workspace goes through torch's allocator (torch Conv_v8.cpp, `run_conv_plan`),
so the fix rests on headroom, not on that attribution (F6).

The rule this plan implements:

    pool = min(what admission can commit, budget − resident need − transient allowances)

The first term is on the branch and measured. The second term is what this revision adds:
make the resident need visible to sglang's own rule by building it first, and subtract the
transient allowances measured at the bounds the deployment already declares.

## 2. What already exists and is reused, not rebuilt

The review's instruction was to fix the origin and not re-implement bounds the code has.
These are the bounds the plan relies on, each read at its line.

| Bound | Where it lives today | How the plan uses it |
| --- | --- | --- |
| Prompt plus generation must fit the context | `enforce_request_limits` is on for Qwen3-TTS (request_builders.py:127), the engine rejects prompt plus `max_new_tokens` above `context_length − 1` (omni_scheduler.py:1171-1180, 1291-1299, 319-322) | It bounds the reference too: the prompt ids carry one id per reference frame (request_builders.py:1239, 1427-1428), so a reference can never exceed `context_length` frames, 8191 frames or 655 s at 12.5 frames per second. The plan applies this same bound before the encoder runs, section 3.3 |
| Whole utterance decode is chunked and padded | qwen-tts 0.1.1: the wrapper pads the batch to its longest item (`qwen3_tts_tokenizer.py:329`), `model.decode` calls `chunked_decode` with 300 frame chunks and 25 frames of left context, chunks are kept on the GPU and concatenated (`modeling_qwen3_tts_tokenizer_v2.py:886-896, 1015`) | The decoder's peak is bounded by `max_batch_size` × 325 frames of chunk activation plus batch × `context_length` × 1920 samples × 4 bytes of retained output. A fixed shape, probed once. No split, no window, no schedule change (F1) |
| One whole utterance decode at a time | the non streaming loop calls `_vocode_payloads` synchronously on the scheduler thread (streaming_simple_scheduler.py:383-430) | One decode allowance, not three (F2) |
| Reference encode concurrency | one batcher thread, batches of up to 8 clips (request_builders.py:743-756, 834-843), up to 8 preprocessing workers each running the speaker encoder (stages.py:109, request_builders.py:923-949) | The encode and speaker allowances are taken at those widths (F2) |
| Streaming decode shapes | windows of at most 24 frames, initial batches up to 32, follow up batches up to 8, 2 follow up workers (streaming_vocoder.py:54-90, 519-521, stages.py:227-235) | Graph pools become resident before the pool by the reorder, the eager fallback for groups above 8 is one more allowance |
| The engine's own reserve | sglang's derived `mem_fraction_static`, reachable as `pre_capture_activation_reserve_mb` and `reserve_for_graph_mb` (server_args.py:5057-5118) | Kept as is for the LLM, never re-derived here |
| Byte budget and stage fraction paths | `engine.kv_cache_bytes` and `gpu_memory_fraction` (sglang_model_runner.py:93-147) | Untouched, a deployment that declares either keeps its own rule |

## 3. Design

### 3.1 Build the vocoder before the engine, one tokenizer per process

Config order becomes preprocessing, vocoder, tts_engine (config.py:54-83). Stages build in
list order, the dispatcher registers all of them after the last factory returns, readiness
follows the last start, and routing follows the named `next` and `stream_to` edges, so the
reorder changes construction and not data flow (stage_workers.py:450-500, F5 agrees).

The vocoder factory loads the tokenizer and captures its graphs (stages.py:245-279), then
publishes the tokenizer through a Qwen owned, process local registry keyed by checkpoint
revision, resolved device, dtype and attention implementation. The engine's `setup_model`
asks the registry with its own key and loads its own copy on a miss, which is what happens in
the split preprocessing layout where the vocoder lives in another process (config.py:85-94,
F5). The two users touch disjoint halves of the object: the encoder on the ref code thread
(request_builders.py:840-843, 884-893), the decoder in the vocoder (streaming_vocoder.py:503,
1884-1886, 1927). The graph holders own their static buffers per worker, not the module
(streaming_vocoder.py:671-675). The fused SnakeBeta swap mutates the decoder in place and is
off by default (streaming_vocoder.py:504-512), a registry entry records whether it was applied
so a consumer never receives a module it did not expect.

Effect: the tokenizer weights and the 4.5 GiB of graph pools are resident when sglang takes
both of its free memory readings, so its own rule charges them. One tokenizer copy fewer.

### 3.2 Probe at the loaded model, before the pool

`pre_infra_setup` runs before the model exists, so it cannot touch the speaker encoder, which
is a sub module of the talker (sglang_model.py:878-884, F4). The probe runs at the one
boundary where the real weights exist and the pool does not: between `ModelWorker`
construction and `alloc_memory_pool` (bootstrap.py:159-180). `create_sglang_infrastructure`
gains one optional keyword, a callback that receives the model worker and the resolved server
args, carried through `infra_kwargs` like `total_gpu_memory_fraction` is today
(engine_factory.py:175, bootstrap.py:98-110). The default is none, so no other model changes.

Qwen3-TTS's callback runs each non LLM component once at its derived maximum and records its
peak the way vLLM's profiling context does (F6): synchronize, reset the peak counter, run,
synchronize, read `max_memory_allocated` minus the allocation live before the run, free the
outputs, `empty_cache`. Each run is on the thread that owns it in serving where that thread
exists at startup, and the plan records that cuDNN's plan caches are per thread (torch
Conv_v8.cpp:357, 366), so plans for other serving threads build at first use inside the
headroom of 3.6.

| Probe | Shape, all derived | Bounds it from |
| --- | --- | --- |
| tokenizer encoder | 8 clips of `context_length` frames of audio at the tokenizer's input rate | batcher width, section 2 |
| speaker encoder | 8 clips of the same length, one per preprocessing worker | preprocessing width |
| whole utterance decode | 8 items of 325 frames, one chunk with its left context, plus retained output computed as 8 × `context_length` × `decode_upsample_rate` × 4 bytes | vocoder batch, decoder chunking, engine context |
| streaming eager fallback | one initial batch of 32 windows of 24 frames through `chunked_decode` | initial batch width |

Every input is a configuration value or a value read from the loaded checkpoint. Nothing is a
constant chosen here. If the encoder probe at 655 s of audio is too large for a card, the knob
that shrinks it is `context_length`, which the deployment owns, and the startup line says so.

### 3.3 Apply the existing context bound before the encoder runs

Preprocessing knows a clip's duration after normalization and before it submits the encode
(request_builders.py:923-935). It rejects a reference whose frame count, duration times the
tokenizer's frame rate, exceeds `context_length − 1`, with the same error the engine would
raise later. This is the engine's rule moved to the point where the GPU work starts, not a
new limit, and it does not touch the 30 s uploaded voice rule or the byte limits of the
serving layer (speech_voices.py:33-34, speech_limits.py:10). The seed-tts corpus sits between
36 and 111 frames, so no request of any archive changes. The 30 s limit of revision 2 is
withdrawn (F3).

### 3.4 The reserve, static allowances, one accounting

The reserve is the sum of the probe peaks of 3.2, one allowance per component that can
allocate at the same time as the others (F2): encode, speaker, one whole utterance decode,
one eager streaming fallback. Summed, never shared. The double spend of revision 2, the
decoder allowed to consume the whole reserve, is gone. Shared credits are not in this plan
and only become a question if the sum does not fit a card, section 7.

The reserve reaches the pool sizing through the omni configurator, which is the existing
integration point. `ModelWorkerConfig` and `_OmniKVCacheConfigurator` gain
`kv_cache_reserve_bytes` (model_worker.py:28-35, sglang_model_runner.py:600-618), carried by
`infra_kwargs` next to the callback above. On the upstream path only, the reserve is subtracted
from what sglang's own rule returns:

    bytes = upstream(free after load, pre load free, slack, mm reservation) − reserve

sglang keeps its own slack for the LLM and its multimodal and hybrid handling, the reserve is
subtracted from its result and never recomputed here (F7 second half). The admission cap then
applies as the minimum (kv_cache_configurator.py:1844-1859):

    tokens = min(running × context, (upstream − reserve) // cell size)

Both `resident need` terms of section 1 are inside `upstream` because 3.1 made them resident
before the readings, so nothing is charged twice (F6).

The post capture resize path (kv_pool_runtime.py:41-101) recomputes the budget from live free
memory and would not carry the reserve (F7). It is off by default
(`SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`, environ.py:513). The omni override that already wraps
it (sglang_model_runner.py:442-497) refuses the flag when a reserve is present, with a message
naming both, until that path is integrated.

### 3.5 The startup line

`post_scheduler_setup` (engine_builder.py:206-218) reports: the accounting mode (cap, upstream
with reserve, byte budget, fraction), the resolved envelope (running, context, decoder batch,
encode width, preprocessing width), each probe's peak, the reserve, the pool in tokens and
bytes, the admission bound, and the free memory sampled at that point, named as free at
startup end rather than pool end (F8).

### 3.6 Headroom

Two things stay outside the reserve by their nature: cuDNN plan caches built at first use on
threads the probe did not run on, and any growth outside the torch allocator. The
qualification run of section 5 measures them as whole device used minus torch reserved at the
steady peak, on both configured points. That measured number is recorded per card in the
readout and becomes the free floor the run must hold, a declared value with a measurement
behind it, never an inferred guarantee (F6, F8).

### 3.7 Memory map and startup flow, 80 GB H100

```
main, fraction 0.85            branch today, cap             this plan
+----------------------+ 81 GB +----------------------+ 81 GB +----------------------+ 81 GB
| free at peak: <1 GB  |       | free: ~48 GB         |       | free: >= headroom    |
+----------------------+       |                      |       +----------------------+
| decode transients,   | ~7    |                      |       | allowances: encode,  | reserve,
| cache, cuDNN plans   |       +----------------------+       | speaker, decode,     | measured
+----------------------+       | transients, graphs   | ~7    | eager fallback       |
| graphs 4.5, tok x2   | ~6    +----------------------+       +----------------------+
+----------------------+       | graphs 4.5, tok x2   | ~6    | graphs 4.5, tok x1   | resident,
|                      |       +----------------------+       | seen by sglang       | charged once
|  KV pool 589142      | 62.9  | KV pool 131072       | 14.0  +----------------------+
|  (bound 131072)      |       |                      |       | KV pool = min(bound, | 14.0 at 16
+----------------------+       +----------------------+       |  upstream − reserve) | fits at 128
| weights              | 4.6   | weights              | 4.6   | weights              | 4.6
+----------------------+       +----------------------+       +----------------------+
```

```
preprocessing factory                                           stages.py:106
vocoder factory: tokenizer, publish, 192 graphs                 stages.py:216-279      3.1
tts_engine factory -> build()
  adjust_overrides: max_total_tokens = running x context        engine_builder.py:152-158
  infra_kwargs: before_pool callback, reserve holder            engine_factory.py:175  3.2, 3.4
  create_sglang_infrastructure
    consume byte budget if declared                             bootstrap.py:131
    ModelWorker: weights load, both free readings               bootstrap.py:159
    before_pool callback: probes, peaks -> reserve              new, bootstrap.py:159-180  3.2
    alloc_memory_pool: upstream − reserve, min with the cap     bootstrap.py:180, kvcc 1844
  setup_model: registry hit or local load                       engine_builder.py:126  3.1
  graphs, predictor graphs                                      engine_factory.py:223, 235
  post_scheduler_setup: the line of 3.5                         engine_builder.py:206
serving: preprocessing applies the context bound before encode  request_builders.py:923  3.3
```

### 3.8 Alternatives read and not taken

- Splitting the decode batch to a byte budget and windowing oversize items, revision 2. The
  decoder is already chunked and padded, the transient is bounded by configuration, and the
  windowed path would change the audio of every non streaming response (F1).
- A 30 s limit for ad hoc references. Not a model rule, the card and README state none, and
  the context already bounds the reference (F3, section 3.3).
- Probing in `pre_infra_setup`. The speaker encoder does not exist yet (F4).
- A stage `gpu_memory_fraction` for tts_engine. A constant no measurement pins, and it still
  measures no transient.
- Fencing torch with `total_reserve_bytes` alone. It caps this process's allocator and reserves
  nothing for what fails outside it.
- Generalizing the probes to every model. Only the callback and the reserve field are
  generic. Each model's envelope is its own contract, with default reserve zero (review).

## 4. What the freed memory unblocks

Measured on the branch: 48 GB free at c16 on an 80 GB card. Expected from this revision: the
128 running run holding the headroom instead of 2 GB, the rope store's c16 pair with the card
free, a second replica or a colocated stage under declared budgets, which remain the
deployment's to declare (review).

## 5. Validation

Unit, per commit: the reorder keeps routing and readiness order, the registry hits on an equal
key and loads locally on a different device, dtype, revision or attention implementation, the
callback runs between weights and pool and its failure aborts startup with the component
named, the reserve reaches the configurator and is subtracted only on the upstream path, the
post capture flag with a reserve is refused, the context bound rejects a clip before the
encode is submitted and passes one at the bound, the startup line carries every field.

Box, paired protocol of plan 07, three arms on one frozen base (F8): A upstream main, C the
cap alone, B the whole branch. Same packages, same checkpoint revision recorded.

| Gate | Setup | Evidence |
| --- | --- | --- |
| Startup accounting | default and 128 running, cold boots | allocator snapshots before resources, after weights, after probes, after the pool, after each capture, at ready. Allocated, reserved, private pools, device free. No term counted twice. Startup time |
| Decoder shape contract | fixed codes at 1, 24, 25, 299, 300, 301, 325, 326 frames and at the context bound, batch 1, 2, 8, ragged batches | padded shape, chunk shape, peak, retained output. Sample counts, trimming, order and audio identical to A on the same codes, deterministic mode included |
| Reference envelope | cold distinct references at the context bound, above it, cache hits, x vector only | encoder and speaker peaks, rejection before GPU work above the bound, cache hit behaviour unchanged |
| Concurrent serving | 128 requests in flight with long completions, reference misses and streaming mixed in | no OOM, no cuDNN failure, no allocator retry, free memory above the headroom throughout |
| Alternate decode route | streaming transport with codec streaming off, eager fallbacks forced | every full decode route completes and stays inside its allowance |
| c1 and c16 corpus | two boots per arm | c1 byte identical to A, c16 inside the archived range, throughput inside the paired spread, cached tokens 13576 of 74096 at c1 |
| Lifecycle | cancellation while queued, during encode, during decode, forced retraction | no leak, no missing or duplicate result, memory returns to its steady envelope |
| Layouts | shared process, split preprocessing, explicit byte budget, stage fraction | local fallback, unchanged override authority, reserve absent where a budget is declared |
| Cache capacity | a prefix working set above the cap | hit rate and prefill work documented, the opt out through explicit sizing works |

## 6. The commit series inside the one PR

Each reviewed by you before it lands.

1. On the branch: the cap, the fraction removal, the startup line.
2. The context bound before the encode, with its test.
3. Stage order, the tokenizer registry with its key and fallback, the engine and vocoder
   consumers.
4. The before pool callback in bootstrap and the engine factory, default none.
5. Qwen3-TTS's probes and the measurement helper.
6. `kv_cache_reserve_bytes` through the worker config and the configurator, the post capture
   refusal.
7. The startup line.

## 7. Decisions

1. Static allowances, no shared credits. Shared credits become a question only if the summed
   reserve does not fit a card at its deployment's context, and the first answer then is the
   context, not a scheduler.
2. Scope is the default CUDA layout. Split and budgeted layouts keep today's behaviour with
   the local fallback.
3. Post capture sizing with a reserve is refused until integrated.
4. Headroom is the measured non torch growth of the qualification run, recorded per card.

## 8. Validation tasks and open facts

1. Tokenizer resident bytes and the graph pool footprint at startup, from the snapshots of
   gate one.
2. The four probe peaks on the H100 at `context_length` 8192, and whether the encoder probe at
   that bound is acceptable at startup.
3. Non torch growth at the steady peak, the headroom of 3.6.
4. Whether `Qwen3TTSTokenizer.decode` output padding changes any sample of a shorter item in a
   ragged batch, from the decoder shape gate. The plan makes no batch change, so this only
   documents the existing behaviour.
5. The cell size and the 114688 bytes per token, from the resolved checkpoint rather than the
   log.
