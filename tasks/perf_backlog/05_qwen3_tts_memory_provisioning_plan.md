# Qwen3-TTS memory provisioning plan

Revision 4, Sep 7 2026, after `review_revision3.md` in
`tasks/qwen3_tts_memory_provisioning_review_20260907/`. Every finding R3-1 to R3-6 was re-read
against its sources and holds, and each is folded in below by number. One PR carries the
whole thing, the rope store lands after it. Sources: the branch
`perf/qwen3-tts-kv-pool-admission-bound` at `52ee606fa` on main `53e94dfa5`, sglang v0.5.18,
torch v2.13.0, qwen-tts 0.1.1 as installed on the box, transformers v5.12.1 Mimi source from
the review's evidence, and the speech tokenizer config of `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.

## 1. The origin of the problem

The process is one OS process with three stages built in config order, preprocessing,
tts_engine, vocoder (stage_workers.py:493-500). Memory is provisioned once, when sglang sizes
the KV pool inside the engine's factory (bootstrap.py:180), from device free memory after
`empty_cache` (v0.5.18 utils/common.py:433-443) minus a slack derived for an LLM's activations
and graphs (server_args.py:4956-4983, kv_cache_configurator.py:1764-1811). The rule sees what
is resident at that moment and nothing else.

Two things are wrong with that moment. Resources of the same process are created after it:
the engine's tokenizer copy in `setup_model` (engine_builder.py:126-132), the vocoder's copy
and its 192 decode graphs in the vocoder factory (stages.py:245-279), whose three private
pools hold 1544 MiB each, proven by pool id in the kv pool snapshot. And no transient of the
non LLM components is accounted anywhere, they allocate at request time out of what the pool
left. At 16 running the admission cap hides both, at 128 running the card fills and the run
failed with `CUDNN_STATUS_INTERNAL_ERROR` at 6 MiB free, with the failing allocation not
attributed (R3, F6).

The rule:

    pool = min(admission bound, upstream sizing over the resident set − transient allowances)

The first term is on the branch and measured. This revision adds the second by building the
resident set before the readings and by subtracting allowances measured for a declared
envelope. Conservation is over what is simultaneously live on the device, so every allowance
below names its execution owner and its lifetime (R3-3).

## 2. Every allocator in the process

The inventory the plan has to conserve, with who creates it, when, and what bounds it.

| Allocation | Owner and creation point | Lifetime | Bound today |
| --- | --- | --- | --- |
| talker, code predictor, speaker encoder weights | ModelWorker in create_sglang_infrastructure (bootstrap.py:159). The speaker encoder exists for Base only (sglang_model.py:878-884) | resident | checkpoint |
| per row decode buffers, predictor K and V cache | talker init (sglang_model.py:255-272, 900-954) | resident | `max_running_requests` |
| KV pool, req_to_token, allocator free list | alloc_memory_pool (bootstrap.py:180) | resident | this plan |
| sglang decode and prefill graphs | init_sglang_cuda_graphs (engine_factory.py:223) | resident, private pools | sglang's graph reserve |
| predictor graphs | setup_model_resources (engine_builder.py:172-179) | resident | bucket ladder |
| speech tokenizer weights, encoder and decoder halves | engine `setup_model` and vocoder factory, two copies today | resident | checkpoint |
| vocoder decode graph pools, 3 holders x 64 graphs | `warmup_now` in the vocoder factory (streaming_vocoder.py:336-372, 699-704) | resident, private pools (0,3) (0,4) (0,5) | shape table x graph flags |
| per holder static input and output buffers | same | resident, 32 MiB per stream | shape table |
| LLM prefill and decode activations | sglang forward | transient | sglang's slack |
| reference encode activations | two callers: the ref code batcher thread, batches of up to 8 clips (request_builders.py:743-756, 834-843), and the uploaded voice miss path calling `wrapper.create_voice_clone_prompt` on a preprocessing worker (request_builders.py:1105-1112, wrapper `qwen3_tts_model.py:418-419`) | transient, until the batcher's stream event or the worker's call returns. A caller that times out at 130 s (request_builders.py:951) stops waiting, the GPU work still completes | nothing. Mimi's first convolution writes `samples x 64 channels` at the input rate (encoder config `num_filters` 64, MimiEncoder `layers.0`) |
| speaker encoder mel and forward | each preprocessing worker, singleton calls, up to 8 workers (stages.py:109, request_builders.py:946-949) | transient | nothing, same clip |
| whole utterance decode | one call at a time on the vocoder scheduler thread (streaming_simple_scheduler.py:383-430), also reachable through `fallback_full_decode` (streaming_vocoder.py:1836-1843, 1923-1927) | transient | batch <= `max_batch_size`, padded to the longest item, forwards of at most 325 frames (qwen-tts `chunked_decode`, 300 plus 25), retained chunk views, concatenation, float32 conversion |
| streaming decode, graph replay | initial worker plus 2 follow up workers, each its own stream and holder (streaming_vocoder.py:519-521, 626-648, 713-726) | transient per worker: the replay clones its output (streaming_vocoder.py:396), the deltas are converted to float32 and kept alive through the D2H copy (streaming_vocoder.py:1193-1215) | window <= 24 frames, batches 32 initial and 8 follow up |
| streaming decode, eager fallback | any worker when a shape has no graph: capture continues past a failed shape (streaming_vocoder.py:355-368), replay returns None on a miss (379-393), the worker then runs `chunked_decode` eagerly (1191-1192) | transient per worker, can overlap across all three | same batches, no graph pool |
| pinned host staging, speaker cache | per worker and module level | host memory | not on the device |
| CUDA context, cuDNN plans, graph executables | driver and libraries, plan caches are per thread (torch Conv_v8.cpp:357, 366), the conv workspace goes through torch's allocator | resident and growing at first use per thread and shape | nothing |
| allocator cache and fragmentation | torch | reserved but not live, released only by `empty_cache` and never for surviving graph pools (torch CUDACachingAllocator.cpp:111-120) | nothing |

## 3. What already exists and is reused

| Bound | Where it lives | How the plan uses it |
| --- | --- | --- |
| prompt length | Qwen3-TTS sets `enforce_request_limits` (request_builders.py:127), the engine rejects an input at or above `context_length − 2` (omni_scheduler.py:1247-1258, sglang utils.py:193-223) | final authority for the ICL prompt, which carries one id per reference frame plus text and control positions (sglang_model.py:621-660, 773-800) |
| generation length | sglang clamps `max_new_tokens` to `max_req_len − input − 1` and to the pool (scheduler.py:2176-2210), it does not reject | the decoder's input from the engine is bounded by `context_length` frames in every mode (R3-4) |
| whole utterance decode shape | qwen-tts 0.1.1 pads and chunks | the decoder scratch is a fixed shape, no algorithm change (F1) |
| explicit budgets | `engine.kv_cache_bytes`, `gpu_memory_fraction`, `total_reserve_bytes` | untouched, and legitimate operator contracts, not heuristics (R3-2) |
| the 30 s uploaded voice bound | speech_voices.py:33-34 | the shipped default of the reference envelope, section 3.3 |

Two relations the previous revision claimed and the review disproved (R3-1). The context does
not bound a reference in x vector only mode: `ref_code` is None there
(request_builders.py:956-960), `build_voice_clone_inputs` takes the plain text route
(sglang_model.py:626-643), and the clip's length never reaches the prompt. The flag is a
request field (protocol.py:354). And for ICL the reference frames fitting the context is a
necessary precheck, not the engine's check, because the prompt adds text and control
positions. So the context bounds the decoder's input, never the raw reference.

## 4. Design

### 4.1 Build the vocoder before the engine, one tokenizer per process

Config order becomes preprocessing, vocoder, tts_engine (config.py:54-83). Construction is
list order, registration and start follow the last factory, routing follows the named edges
(stage_workers.py:450-500). The vocoder factory loads the tokenizer, captures its graphs and
publishes the object in a Qwen owned, process local registry keyed by checkpoint revision,
device, dtype, attention implementation and whether the fused SnakeBeta swap was applied
(streaming_vocoder.py:504-512). The engine acquires it inside the before pool callback of 4.4
with its own key, and loads its own copy on a miss before any probe runs (R3-6). A split
preprocessing layout moves preprocessing only (config.py:85-88), the vocoder and the engine
still share a process and the registry works there.

Effect: the tokenizer weights and the graph pools are resident when sglang takes both of its
free memory readings. sglang's own rule charges them, nothing here charges them again.

### 4.2 The envelope, declared by the pipeline config

The engine builder does not know the sibling stages' settings. The pipeline config does, and
it already injects cross stage facts per stage (config.py:88-110). It passes the engine
factory one `memory_envelope` argument built from its own stage list and factory settings:

- whether preprocessing runs in this process, and its worker width (8)
- whether the vocoder runs in this process, its decode batch (8), initial and follow up
  batches (32, 8), follow up worker count (2), graph flags
- the model variant, Base or not
- the accepted reference duration, section 4.3
- an optional `headroom_bytes`, section 4.6

Probes run only for components this process executes for this variant (R3-2, R3-6): no
reference probes for CustomVoice or VoiceDesign, none when preprocessing is in another
process, no decoder probes when the vocoder is, none at all under a byte budget or a stage
fraction, and nothing when the envelope is absent, so every other model is unchanged.

### 4.3 The reference envelope is a serving contract

An encoder whose memory is linear in samples cannot be reserved for an input with no bound,
and the model provides none. At the context bound the first Mimi activation alone is
8 clips x 15,728,640 samples x 64 channels x 2 bytes, about 16 GB, before the residual and
transformer layers (R3-2). So the accepted reference duration is a declared value, applied at
input normalization for every encoder caller, the ad hoc hook, the uploaded voice miss path
and x vector only mode alike, rejected before any GPU work with a named error. The shipped
default is the uploaded voice bound, 30 s, so ad hoc and uploaded references follow one rule.
A deployment that serves longer references raises the value and pays the measured allowance.
Frame counts use the tokenizer's rounding, ceiling of samples over the downsample rate
(qwen-tts `modeling_qwen3_tts_tokenizer_v2.py:983`), never a float multiplication. The ICL
precheck against the context stays, and the engine keeps final authority. The context derived
bound of revision 3 is withdrawn.

### 4.4 The before pool callback

`create_sglang_infrastructure` gains one optional keyword, a callback invoked between
`ModelWorker` construction and `alloc_memory_pool` (bootstrap.py:159-180), carried through
`infra_kwargs`. It receives the model worker and the resolved server args and returns a
result: the allowances by component, the retained bytes the probes left behind, and the
envelope they used. Bootstrap consumes the result once and assigns it on the runner before
`alloc_memory_pool` creates the configurator (v0.5.18 model_runner.py:799-807), the way the
omni configurator swap already reads runner attributes (sglang_model_runner.py:600-618).
`ModelWorkerConfig` is not the carrier, it is built and copied before the callback runs
(R3-6). Failure of a probe aborts startup naming the component.

Measurement, per probe, on the thread that owns it where that thread exists at startup:
synchronize, record allocated, reset the peak, run, synchronize, read the peak, free the
outputs, record allocated again.

    transient = peak − allocated after the run
    retained  = allocated after − allocated before

Only the transient enters an allowance. The retained bytes are reported and not subtracted,
sglang's reading after the callback sees them (R3-5, vLLM `mem_utils.py:314-328`). Peak
allocated does not include cross stream deferred frees or fragmentation, which section 4.6
addresses.

### 4.5 The allowances, one per execution owner

| Owner | Probe | Geometry, all declared |
| --- | --- | --- |
| reference encode | the larger of one batched encode of 8 clips at the envelope and 8 singleton wrapper encodes of one clip, since under 8 workers at most 8 clips are in preparation and either caller geometry can hold them (R3-3) | batcher width, worker width, envelope |
| speaker encoder | 8 singleton forwards at the envelope, mel included | worker width, envelope |
| whole utterance decode | scratch: `decoder.forward` on `(8, num_quantizers, 325)`, the real maximum forward, which a `tokenizer.decode` of 325 frames never runs (R3-4). Retained storage: the maximum over the loop's three phases, retained chunk views including their discarded context plus the current forward, all retained chunks plus the concatenation, the final output plus the float32 conversion of one item, computed from batch, `context_length`, `decode_upsample_rate` 1920 and the resident dtype | vocoder batch, engine context, decoder config |
| streaming workers | three eager `chunked_decode` calls at once, 32 windows on the initial worker and 8 on each follow up, plus per worker the replay clone and the float32 deltas at the largest captured shape | worker policy, batches, shape table |

Summed, never shared (decision 2). No allowance for a component this process does not run.

### 4.6 Sizing, the post capture path, and headroom

`_OmniKVCacheConfigurator` gains `kv_cache_reserve_bytes`, set from the callback result. On
the upstream path only:

    bytes  = upstream(free after load, pre load free, slack, mm reservation) − reserve
    tokens = min(running × context, bytes // cell size)

sglang keeps its slack, its multimodal and hybrid handling, and the cap applies as the
minimum (kv_cache_configurator.py:1844-1859). The byte budget and fraction paths are
untouched.

The post capture resize (kv_pool_runtime.py:41-101) would recompute without the reserve.
With a reserve present the flag is refused when the server args are validated, before any
weight loads (engine_factory.py:173), not after startup work (R3, F7).

Headroom has an owner and a definition (R3-5). Two quantities are measured on the
qualification run and reported separately: external growth, device used minus torch reserved
at time t minus the same at ready, its maximum over the run, and torch retention above live,
reserved minus allocated at the peak. The readout states whether sglang's slack covered both.
If it did not, the deployment declares `headroom_bytes` in the envelope and it is added to
the reserve. No number is written into code for any card, and a card the run did not
qualify is architecture compatible, not qualified (section 5).

### 4.7 The startup line

`post_scheduler_setup` reports the accounting mode, the envelope, each allowance and the
retained bytes, the reserve, the pool in tokens and bytes, the admission bound, and free
memory at startup end, named as such.

### 4.8 Memory map and flow, 80 GB H100

```
main, fraction 0.85            branch today, cap             this plan
+----------------------+ 81 GB +----------------------+ 81 GB +----------------------+ 81 GB
| free at peak: <1 GB  |       | free: ~48 GB         |       | free: slack, headroom|
+----------------------+       |                      |       +----------------------+
| transients, cache,   | ~7    |                      |       | allowances: encode,  | measured
| cuDNN plans          |       +----------------------+       | speaker, decode,     | at the
+----------------------+       | transients, graphs   | ~7    | streaming            | envelope
| graphs 4.5, tok x2   | ~6    +----------------------+       +----------------------+
+----------------------+       | graphs 4.5, tok x2   | ~6    | graphs 4.5, tok x1   | resident,
|                      |       +----------------------+       | charged by sglang    | once
|  KV pool 589142      | 62.9  | KV pool 131072       | 14.0  +----------------------+
|  (bound 131072)      |       |                      |       | KV pool = min(bound, | 14.0 at 16
+----------------------+       +----------------------+       |  upstream − reserve) | fits at 128
| weights              | 4.6   | weights              | 4.6   | weights              | 4.6
+----------------------+       +----------------------+       +----------------------+
```

```
preprocessing factory                                           stages.py:106
vocoder factory: tokenizer, publish, 192 graphs                 stages.py:216-279      4.1
tts_engine factory -> build()
  envelope from the pipeline config                             config.py:88           4.2
  adjust_overrides: max_total_tokens = running x context        engine_builder.py:152
  validate: refuse post capture sizing with a reserve           engine_factory.py:173  4.6
  infra_kwargs: before pool callback                            engine_factory.py:175  4.4
  create_sglang_infrastructure
    consume byte budget if declared                             bootstrap.py:131
    ModelWorker: weights, both free readings                    bootstrap.py:159
    callback: acquire tokenizer, probe, return result           new                    4.4, 4.5
    result assigned on the runner                               new
    alloc_memory_pool: upstream − reserve, min with the cap     bootstrap.py:180
  setup_model: registry object already attached                 engine_builder.py:126
  graphs, predictor graphs                                      engine_factory.py:223, 235
  post_scheduler_setup: the line of 4.7                         engine_builder.py:206
serving: normalization applies the reference envelope           request_builders.py:931 4.3
```

### 4.9 Alternatives read and not taken

- A context derived reference bound. It bounds nothing in x vector mode and only prechecks
  ICL (R3-1).
- Probing the encoder at the context bound. 16 GB for one activation, and it shrinks the
  text and generation capacity if used as the remedy (R3-2).
- A decoder split or window, revision 2 (F1).
- Probing in `pre_infra_setup`, the model does not exist there (F4).
- Carrying the reserve through `ModelWorkerConfig`, it is built before the callback (R3-6).
- A per card headroom constant in code (R3-5).

## 5. Portability

The algorithm is the same on every card: the resident set built first, sglang's own reading
of the local device, allowances measured locally at the declared envelope. The numbers are
not: an H200 sizes a larger pool from its larger reading when the cap does not bind, and its
workspaces, graph residency and headroom are its own measurements. A run on one card
qualifies that card. The plan promises the accounting, not equal pool sizes or throughput.

## 6. Validation

Remote unit checks, per commit: the reference envelope rejects above the bound on all three
caller paths and before any device work, an x vector reference longer than the context is
accepted below the envelope and rejected above it, the ICL precheck passes what the engine
would accept and the engine keeps final authority, the registry hits on an equal key and
loads locally otherwise, the callback runs between weights and pool, returns a consumed
result, and aborts startup naming a failed probe, the reserve reaches the configurator on the
upstream path only, the post capture flag is refused before weights load, the allowances
follow the envelope predicates for every layout and variant, the startup line carries every
field.

Box, three frozen arms on one base: A upstream main, C the cap alone, B the whole branch.

| Gate | Setup | Evidence |
| --- | --- | --- |
| Probe cost | default and 128 running, cold boots | each probe's time and transient, retained bytes, startup time against A |
| Startup accounting | same | snapshots before resources, after weights, after probes, after the pool, after each capture, at ready. Each category mapped to the sizing input exactly once |
| Reference contract | ICL and x vector, ad hoc batched and uploaded misses, cache hits, clips at and above the envelope | encoder and speaker transients per caller geometry, rejection before device work above the envelope |
| Decoder contract | fixed codes at 1, 24, 25, 299, 300, 301, 325, 326, 625 frames and the context bound, batch 1, 2, 8, ragged | forward shapes, scratch and retained storage against the formula, audio identical to A on the same codes, deterministic mode |
| Execution classes | full decode, replay, eager fallback with a shape removed from the capture, singleton and overlapping workers, timeout and cancellation | every path inside its allowance, memory back to the steady envelope |
| Concurrent serving | 128 in flight with long completions, reference misses and streaming mixed | no OOM, no cuDNN failure, no allocator retry, external growth and torch retention recorded for 4.6 |
| c1 and c16 corpus | two boots per arm | c1 byte identical to A, c16 inside the archived range, throughput inside the paired spread, cached tokens 13576 of 74096 |
| Layouts | shared process, split preprocessing, byte budget, stage fraction, CustomVoice | predicates hold, override authority unchanged, no probe where the process does not run the component |
| Cache capacity | a prefix working set above the cap | hit rate and prefill work documented, the explicit sizing opt out works |

## 7. The commit series inside the one PR

1. On the branch: the cap, the fraction removal, the startup line.
2. The reference envelope at normalization, all three callers, with tests.
3. Stage order, the keyed tokenizer registry, the vocoder publish and the engine acquire.
4. The envelope argument from the pipeline config.
5. The before pool callback in bootstrap and the engine factory, default none.
6. Qwen3-TTS's probes and the measurement helper.
7. `kv_cache_reserve_bytes` on the runner and the configurator, the early post capture
   refusal, `headroom_bytes`.
8. The startup line.

## 8. Decisions

1. The reference envelope default. Recommended: the uploaded voice bound, 30 s, so both
   reference paths follow one rule. It rejects ad hoc references above 30 s that succeed
   today when memory allows, which is a serving contract change and yours to make.
2. Static allowances, no shared credits.
3. Scope: the default CUDA layout for Base. Other variants and layouts run only the probes
   for what they execute.
4. Post capture sizing with a reserve refused at validation.
5. Headroom is deployment declared, measured by the qualification run, never a constant in
   code.

## 9. Validation tasks and open facts

1. The resident bytes of one tokenizer copy and of the graph pools at startup, from the
   snapshots of the accounting gate.
2. The four allowances on the H100 at the 30 s envelope and the probes' startup cost.
3. External growth and torch retention at the steady peak, and whether sglang's slack covers
   them.
4. Whether `tokenizer.decode` output padding changes any sample of a shorter item in a ragged
   batch, documented from the decoder gate, no change planned.
5. The cell size from the resolved checkpoint.
