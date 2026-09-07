# Qwen3-TTS memory provisioning plan

Revision 5, Sep 7 2026, after `review_revision3.md` and `reference_envelope_clarification.md`
in `tasks/qwen3_tts_memory_provisioning_review_20260907/`. Every finding of both was re-read
against its sources and holds. One PR carries the whole thing, the rope store lands after it.
Sources: the branch `perf/qwen3-tts-kv-pool-admission-bound` at `52ee606fa` on main
`53e94dfa5`, sglang v0.5.18, torch v2.13.0, qwen-tts 0.1.1 as installed on the box,
transformers v5.12.1 Mimi source from the review's evidence, and the checkpoint configs of
`Qwen/Qwen3-TTS-12Hz-1.7B-Base`.

## 1. The origin of the problem

The process is one OS process with three stages built in config order, preprocessing,
tts_engine, vocoder (stage_workers.py:493-500). Memory is provisioned once, when sglang sizes
the KV pool inside the engine's factory (bootstrap.py:180), from device free memory after
`empty_cache` (v0.5.18 utils/common.py:433-443) minus a slack derived for an LLM's activations
and graphs (server_args.py:4956-4983, kv_cache_configurator.py:1764-1811). The rule sees what
is resident at that moment and nothing else.

Three things are wrong, and they are the origin.

1. Resources of the same process are created after that moment: the engine's tokenizer copy
   in `setup_model` (engine_builder.py:126-132), the vocoder's copy and its 192 decode graphs
   in the vocoder factory (stages.py:245-279), whose three private pools hold 1544 MiB each,
   proven by pool id in the kv pool snapshot.
2. No transient of the non LLM components is accounted anywhere. They allocate at request
   time out of what the pool left.
3. The reference path runs work it discards. In x vector only mode both callers encode the
   clip through the Mimi tokenizer and then drop the codes: the ad hoc hook submits the encode
   unconditionally and sets `ref_code` to None afterwards (request_builders.py:936, 954), and
   the wrapper path encodes every clip before deciding the same (qwen-tts
   `qwen3_tts_model.py:426-432, 449`). That encode is the only allocation in the process whose
   input has no bound at all, and it runs in the one mode where the context does not bound
   the clip.

At 16 running the admission cap hides the first two, at 128 running the card fills and the
run failed with `CUDNN_STATUS_INTERNAL_ERROR` at 6 MiB free, the failing allocation not
attributed (F6).

The rule:

    pool = min(admission bound, upstream sizing over the resident set − transient allowances)

The first term is on the branch and measured. This revision adds the second by building the
resident set before the readings and by subtracting allowances measured at declared bounds,
and it removes the third by not running the discarded encode. Conservation is over what is
simultaneously live on the device, so every allowance names its execution owner and lifetime.

## 2. Every allocator in the process

| Allocation | Owner and creation point | Lifetime | Bound today |
| --- | --- | --- | --- |
| talker, code predictor, speaker encoder weights | ModelWorker in create_sglang_infrastructure (bootstrap.py:159). The speaker encoder exists for Base only (sglang_model.py:878-884) | resident | checkpoint |
| per row decode buffers, predictor K and V cache | talker init (sglang_model.py:255-272, 900-954) | resident | `max_running_requests` |
| KV pool, req_to_token, allocator free list | alloc_memory_pool (bootstrap.py:180) | resident | this plan |
| sglang decode and prefill graphs | init_sglang_cuda_graphs (engine_factory.py:223) | resident, private pools | sglang's graph reserve |
| predictor graphs | setup_model_resources (engine_builder.py:172-179) | resident | bucket ladder |
| speech tokenizer weights, encoder and decoder halves | engine `setup_model` and vocoder factory, two copies today | resident | checkpoint |
| vocoder decode graph pools, 3 holders x 64 graphs | `warmup_now` in the vocoder factory (streaming_vocoder.py:336-372, 699-704) | resident, private pools (0,3) (0,4) (0,5) | shape table x graph flags |
| per holder static buffers | same | resident, 32 MiB per stream | shape table |
| LLM prefill and decode activations | sglang forward | transient | sglang's slack |
| Mimi encode of a reference | two callers today: the ref code batcher thread, batches of up to 8 clips padded to the longest by the feature extractor (request_builders.py:743-756, 834-843, qwen-tts `qwen3_tts_tokenizer.py:241-252`), and the uploaded voice miss path through the wrapper on a preprocessing worker (request_builders.py:1105-1112) | transient until the batcher's stream event, a caller that times out at 130 s stops waiting while the kernels finish | nothing. The first convolution writes samples x 64 channels at 24 kHz (encoder config `num_filters` 64), the rest is causal convolutions and a sliding window transformer (`sliding_window` 250), memory linear in samples |
| speaker encoder forward | each preprocessing worker, up to 8, on a CPU computed mel (sglang_model.py:404-425) | transient | nothing. ECAPA TDNN, Res2Net, SE and attentive statistics pooling (qwen-tts `modeling_qwen3_tts.py:311-373`), memory linear in mel frames, pooling needs the whole clip |
| whole utterance decode | one call at a time on the vocoder scheduler thread (streaming_simple_scheduler.py:383-430), also through `fallback_full_decode` (streaming_vocoder.py:1836-1843, 1923-1927) | transient | batch <= `max_batch_size`, padded to the longest item, forwards of at most 325 frames (qwen-tts `chunked_decode`, 300 plus 25), retained chunk views, concatenation, float32 conversion |
| streaming decode, graph replay | initial worker plus 2 follow up workers, own stream and holder each (streaming_vocoder.py:519-521, 626-648, 713-726) | transient per worker: the replay clones its output (396), deltas become float32 and live through the D2H copy (1193-1215) | window <= 24 frames, batches 32 and 8 |
| streaming decode, eager fallback | any worker whose shape has no graph: capture continues past a failed shape (355-368), replay returns None on a miss (379-393), the worker runs `chunked_decode` (1191-1192) | transient per worker, can overlap across all three | same batches |
| pinned staging, speaker cache | per worker, module level | host memory | not on the device |
| CUDA context, cuDNN plans, graph executables | driver and libraries, plan caches per thread (torch Conv_v8.cpp:357, 366), conv workspace through torch's allocator | resident and growing at first use | nothing |
| allocator cache and fragmentation | torch | reserved, not live, kept for surviving graph pools (CUDACachingAllocator.cpp:111-120) | nothing |

## 3. What already exists and is reused

| Bound | Where it lives | How the plan uses it |
| --- | --- | --- |
| prompt length | Qwen3-TTS sets `enforce_request_limits` (request_builders.py:127), the engine rejects an input at or above `context_length − 2` (omni_scheduler.py:1247-1258, sglang utils.py:193-223) | final authority for the ICL prompt, which carries one id per reference frame plus text and control positions (sglang_model.py:621-660, 773-800). So an ICL reference is bounded by `context_length` frames, 8191 frames or 655 s at 12.5 frames per second, and a longer one is rejected by the engine today after the encode ran |
| generation length | sglang clamps `max_new_tokens` to the context and the pool (scheduler.py:2176-2210) | the decoder's input is bounded by `context_length` frames in every mode |
| whole utterance decode shape | qwen-tts 0.1.1 pads and chunks | the decoder scratch is a fixed shape, no algorithm change |
| the reference batcher | one thread, already serializes every batched encode, groups by sample rate (request_builders.py:774-832) | it becomes the single caller of the encoder and groups by bytes, section 4.3 |
| explicit budgets | `engine.kv_cache_bytes`, `gpu_memory_fraction`, `total_reserve_bytes` | untouched, legitimate operator contracts |
| the 30 s uploaded voice bound | speech_voices.py:33-34 | stays where it is, it is not extended to ad hoc references (clarification) |

In x vector only mode the codes are never consumed: `build_voice_clone_inputs` takes the text
route when `ref_code` is None (sglang_model.py:626-643), the cache artifact skips a None code
(request_builders.py:681-684), and the vocoder trims nothing when `ref_code_len` is 0. So not
running the encode in that mode changes no output.

## 4. Design

### 4.1 Build the vocoder before the engine, one tokenizer per process

Config order becomes preprocessing, vocoder, tts_engine (config.py:54-83). Construction is
list order, registration and start follow the last factory, routing follows the named edges
(stage_workers.py:450-500). The vocoder factory loads the tokenizer, captures its graphs and
publishes the object in a Qwen owned, process local registry keyed by checkpoint revision,
device, dtype, attention implementation and whether the fused SnakeBeta swap was applied
(streaming_vocoder.py:504-512). The engine acquires it inside the before pool callback of 4.4
with its own key and loads its own copy on a miss before any probe runs. A split
preprocessing layout moves preprocessing only (config.py:85-88), the vocoder and the engine
still share a process.

Effect: the tokenizer weights and the graph pools are resident when sglang takes its free
memory readings. sglang's own rule charges them, nothing here charges them again.

### 4.2 The envelope, declared by the pipeline config

The pipeline config already injects cross stage facts per stage (config.py:88-110). It passes
the engine factory one `memory_envelope` built from its own stage list and factory settings:
whether preprocessing runs in this process and its worker width, whether the vocoder does and
its decode batch, initial and follow up batches, worker count and graph flags, the model
variant, the reference bound of 4.3, and the optional `headroom_bytes` of 4.6. Probes run only
for components this process executes for this variant: no reference probes for CustomVoice
or VoiceDesign, none when preprocessing is in another process, no decoder probes when the
vocoder is, none under a byte budget or a stage fraction, nothing when the envelope is absent.

### 4.3 The reference path: one caller, no discarded work, bytes aware batching

Three changes, all inside the component that owns the work.

1. The uploaded voice miss path stops calling the wrapper and goes through the omni
   reference hook like the ad hoc path (request_builders.py:1105-1112 today). The hook is then
   the only caller of the tokenizer encoder and of the speaker encoder in the process, and the
   batcher is the only thread that runs the Mimi encoder. One geometry, one lifetime.
2. The hook submits the Mimi encode only in ICL mode. In x vector only mode it runs the
   speaker encoder alone. This is the origin fix of section 1 item 3: the one allocation with
   no bound at all no longer runs in the one mode the context does not bound.
3. The batcher groups by bytes as well as by sample rate. The reference bound is
   `context_length` frames, derived, since that is the longest ICL reference the engine can
   admit. The allowance is the measured cost of one clip at that bound, section 4.5. A batch
   is filled while `B x padded samples` stays within the samples of that probe, so the cost of
   any batch is at most the measured cost of the probe, linear by structure and validated on
   three lengths at startup. A single clip above the bound fails before any device work with
   a capacity error naming the bound. For ICL that clip is one the engine rejects today after
   the encode ran, so no admissible request changes.

The ICL precheck, frames from the tokenizer's own rounding, ceiling of samples over the
downsample rate (qwen-tts `modeling_qwen3_tts_tokenizer_v2.py:983`), runs before the submit and
the engine keeps final authority.

The speaker encoder keeps its 8 parallel callers. Its bound is the same `context_length`
frames in both modes, so a clip above it is refused before device work. This is the one
contract change of the plan: an x vector only reference longer than 655 s, which today runs
when memory happens to allow it, is refused with a capacity error. Below that nothing
changes. A deployment that wants a smaller reference bound declares `max_reference_seconds`
in the envelope, an opt in policy per the clarification, and the allowances shrink with it.
No such policy ships by default.

### 4.4 The before pool callback

`create_sglang_infrastructure` gains one optional keyword, a callback invoked between
`ModelWorker` construction and `alloc_memory_pool` (bootstrap.py:159-180), carried through
`infra_kwargs`. It receives the model worker and the resolved server args and returns a
result: the allowances by component, the retained bytes the probes left behind, and the
envelope they used. Bootstrap consumes the result once and assigns it on the runner before
`alloc_memory_pool` creates the configurator (v0.5.18 model_runner.py:799-807), the way the
omni configurator swap reads runner attributes (sglang_model_runner.py:600-618).
`ModelWorkerConfig` is not the carrier, it is built and copied before the callback runs. A
failed probe aborts startup naming the component.

Measurement, per probe, on the thread that owns it where it exists at startup: synchronize,
record allocated, reset the peak, run, synchronize, read the peak, free the outputs, record
allocated again.

    transient = peak − allocated after the run
    retained  = allocated after − allocated before

Only the transient enters an allowance. The retained bytes are reported and not subtracted,
sglang's reading after the callback sees them (vLLM `mem_utils.py:314-328`). Peak allocated
does not include cross stream deferred frees or fragmentation, section 4.6 covers them.

### 4.5 The allowances, one per execution owner

| Owner | Probe | Geometry, all derived |
| --- | --- | --- |
| Mimi encode, the batcher thread | one clip at the reference bound, batch 1, plus two shorter lengths to validate linearity on the card | the batcher's byte grouping keeps every batch within this |
| speaker encoder, 8 workers | 8 singleton forwards on the mel of a clip at the reference bound | worker width, reference bound |
| whole utterance decode, the vocoder scheduler thread | scratch: `decoder.forward` on `(8, num_quantizers, 325)`, the real maximum forward. Retained storage: the maximum over the loop's three phases, retained chunk views with their discarded context plus the current forward, all retained chunks plus the concatenation, the final output plus the float32 conversion of one item, computed from batch, `context_length`, `decode_upsample_rate` 1920 and the resident dtype | vocoder batch, engine context |
| streaming workers | three eager `chunked_decode` calls at once, 32 windows on the initial worker and 8 on each follow up, plus per worker the replay clone and the float32 deltas at the largest captured shape | worker policy, batches, shape table |

Summed, never shared. No allowance for a component this process does not run. The encoder
probe at the bound is one clip of 655 s: its first activation is about 2 GB in bfloat16 and
the full peak is measured, which is feasible at startup, unlike the eight clip probe of
revision 3.

### 4.6 Sizing, the post capture path, and headroom

`_OmniKVCacheConfigurator` gains `kv_cache_reserve_bytes`, set from the callback result. On
the upstream path only:

    bytes  = upstream(free after load, pre load free, slack, mm reservation) − reserve
    tokens = min(running × context, bytes // cell size)

sglang keeps its slack and its multimodal and hybrid handling, the cap applies as the minimum
(kv_cache_configurator.py:1844-1859), the byte budget and fraction paths are untouched.

The post capture resize (kv_pool_runtime.py:41-101) would recompute without the reserve. With
a reserve present the flag is refused when the server args are validated, before weights
load (engine_factory.py:173).

Headroom has an owner and a definition. Two quantities are measured on the qualification run
and reported separately: external growth, device used minus torch reserved at time t minus
the same at ready, its maximum over the run, and torch retention above live, reserved minus
allocated at the peak. The readout states whether sglang's slack covered both. If it did not,
the deployment declares `headroom_bytes` in the envelope and it is added to the reserve. No
number is written into code for any card.

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
+----------------------+       | transients, graphs   | ~7    | streaming            | bounds
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
serving: the hook prechecks, skips the encode in x vector mode,
  the batcher groups by bytes                                   request_builders.py:923 4.3
```

### 4.9 Alternatives read and not taken

- A 30 s limit on ad hoc references, revision 4. Not a model rule, and an input policy where
  the fix is in the work (clarification).
- A context derived bound used as an input rule for x vector mode, revision 3. The context
  bounds the ICL prompt, not the clip. In this revision the x vector clip is bounded by the
  allowance the batcher and the workers hold, with the encode itself no longer running.
- Probing eight clips at the context bound. 16 GB for one activation (R3-2).
- Chunking the speaker encoder. Its pooling needs the whole clip (clarification).
- Mimi's streaming encode for long clips. Exact by design for causal convolutions and a
  sliding window, but a numerical qualification of discrete codes, and unnecessary once the
  batcher bounds bytes and the discarded encode is gone.
- A decoder split or window, revision 2 (F1).
- Probing in `pre_infra_setup` (F4), carrying the reserve through `ModelWorkerConfig` (R3-6),
  a per card headroom constant (R3-5).

## 5. Portability

The algorithm is the same on every card: the resident set built first, sglang's own reading
of the local device, allowances measured locally at the derived bounds. The numbers are not.
A run on one card qualifies that card. The plan promises the accounting, not equal pool sizes
or throughput.

## 6. Validation

Remote unit checks, per commit: the hook is the only encoder caller, x vector mode submits no
encode and produces the same prompt, cache artifact and vocoder trim as today, the ICL
precheck passes what the engine accepts and refuses the rest before the submit, the batcher
never exceeds the probe's samples in one call and refuses a single clip above the bound
before device work, the registry hits on an equal key and loads locally otherwise, the
callback runs between weights and pool, returns a consumed result and aborts startup naming a
failed probe, the reserve reaches the configurator on the upstream path only, the post capture
flag is refused before weights load, the allowances follow the envelope predicates for every
layout and variant, the startup line carries every field.

Box, three frozen arms on one base: A upstream main, C the cap alone, B the whole branch.

| Gate | Setup | Evidence |
| --- | --- | --- |
| Probe cost | default and 128 running, cold boots | each probe's time and transient, retained bytes, startup time against A |
| Startup accounting | same | snapshots before resources, after weights, after probes, after the pool, after each capture, at ready. Each category mapped to the sizing input once |
| Reference contract | ICL and x vector, ad hoc and uploaded misses, cache hits, clips at and above the bound, mixed lengths in one batch window | encoder and speaker transients, batches within the allowance, refusal before device work above the bound, codes and speaker embeddings identical to A for the same clips |
| Decoder contract | fixed codes at 1, 24, 25, 299, 300, 301, 325, 326, 625 frames and the context bound, batch 1, 2, 8, ragged | forward shapes, scratch and retained storage against the formula, audio identical to A on the same codes, deterministic mode |
| Execution classes | full decode, replay, eager fallback with a shape removed from capture, singleton and overlapping workers, timeout and cancellation | every path inside its allowance, memory back to the steady envelope |
| Concurrent serving | 128 in flight with long completions, reference misses and streaming mixed | no OOM, no cuDNN failure, no allocator retry, external growth and torch retention recorded |
| c1 and c16 corpus | two boots per arm | c1 byte identical to A, c16 inside the archived range, throughput inside the paired spread, cached tokens 13576 of 74096 |
| Layouts | shared process, split preprocessing, byte budget, stage fraction, CustomVoice | predicates hold, override authority unchanged, no probe where the process does not run the component |
| Cache capacity | a prefix working set above the cap | hit rate and prefill work documented, the explicit sizing opt out works |

## 7. The commit series inside the one PR

1. On the branch: the cap, the fraction removal, the startup line.
2. The reference hook as the single encoder caller, no encode in x vector mode, the ICL
   precheck, with tests.
3. The batcher's byte grouping and the capacity refusal, with tests.
4. Stage order, the keyed tokenizer registry, the vocoder publish and the engine acquire.
5. The envelope argument from the pipeline config.
6. The before pool callback in bootstrap and the engine factory, default none.
7. Qwen3-TTS's probes and the measurement helper.
8. `kv_cache_reserve_bytes` on the runner and the configurator, the early post capture
   refusal, `headroom_bytes`.
9. The startup line.

## 8. Decisions taken

1. No input policy ships by default. The reference bound is the context, derived, and
   `max_reference_seconds` exists as an opt in deployment policy only.
2. Static allowances, no shared credits. The batcher's byte grouping is the only admission
   mechanism, inside the component that already serializes the work.
3. Scope: the default CUDA layout for Base. Other variants and layouts run only the probes
   for what they execute.
4. Post capture sizing with a reserve refused at validation.
5. Headroom is deployment declared, measured by the qualification run, never a constant in
   code.

## 9. Validation tasks and open facts

1. The resident bytes of one tokenizer copy and of the graph pools at startup.
2. The Mimi encode transient at the bound, one clip of 655 s, and the speaker encoder's at
   the same clip, on the H100, with the probe time.
3. Linearity of the Mimi encode cost on the card across the three probe lengths.
4. External growth and torch retention at the steady peak, and whether sglang's slack covers
   them.
5. Whether `tokenizer.decode` output padding changes any sample of a shorter item in a ragged
   batch, documented from the decoder gate, no change planned.
6. The cell size from the resolved checkpoint.
