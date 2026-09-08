# Qwen3-TTS memory provisioning plan

Revision 6, Sep 8 2026. Replaces revision 5 after the sglang bump readout of Sep 8 reproduced
the failure on main and attributed it. The design is reduced to ordering and the cap, no
measurement machinery. Sources: upstream main `53239c285`, sglang v0.5.18, the Sep 6 archive
readout (`11_kv_pool_readout_20260906.md`), the bump readout quoted in section 1, the review
documents in `tasks/qwen3_tts_memory_provisioning_review_20260907/`, qwen-tts 0.1.1 as
installed on the box. Code branch: `perf/qwen3-tts-memory-provisioning` from `53239c285`.

## 1. The origin of the problem

The process is one OS process with three stages built in config order, preprocessing,
tts_engine, vocoder (stage_workers.py:493). Memory is provisioned once, when sglang sizes the
KV pool inside the engine's factory. What the process does, in order, on `53239c285`:

```
ModelWorker: talker weights, sglang's two free memory readings   bootstrap.py:159
alloc_memory_pool: the pool is sized and allocated               bootstrap.py:180
setup_model: the engine's speech tokenizer copy                  engine_factory.py:210, engine_builder.py:111
init_sglang_cuda_graphs: decode and prefill graphs               engine_factory.py:223
setup_model_resources: predictor graphs, #1947                   engine_factory.py:235, engine_builder.py:168
vocoder factory: second tokenizer copy, codec state arena,       stages.py:259-302
  codec graphs, #1912 #1930 #1997
```

sglang's rule: pool = free after load − pre load free × (1 − `mem_fraction_static`) − mm
reservation (kv_cache_configurator.py:1764-1811), with the fraction derived as 512 MB plus
1.5 MB per activation token plus a graph reserve, floor 10 GB above 60 GB
(server_args.py:4956-4983). The graph reserve counts sglang's own decode and prefill graphs
only (server_args.py:5078-5118). Nothing below the pool line is in the reading or in the
reserve. It is paid from the slack sglang left for its own transients.

The bump readout, main against the sglang bump, Sep 8: both arms captured the same prefill
buckets through 512 at the same 0.31 GB, so the prefill graphs are not the difference.
Disabling either the predictor startup capture or the vocoder graphs restores 16 of 16 on
the upgrade arm. At the failure the engine's allocator held 75.6 GiB with 4.5 GiB in graph
pools, the device was at 78.7 of 79.2 GiB, and an 844 MiB prefill transient found 502 MiB.
Since #1900 makes prefill graphs the CustomVoice default, main has all three consumers on.

So the failure is residency created after the reading, not a transient the process could
not afford. The Sep 6 archive shows the same mechanism at 128 running on the older main,
where the card filled and the run failed with `CUDNN_STATUS_INTERNAL_ERROR` at 6 MiB free.

Two more facts from revision 5 stay true and are handled as follows.

- The reference path runs work it discards: in x vector only mode both callers encode the
  clip through the Mimi tokenizer and drop the codes (request_builders.py:965, 984 and the
  wrapper, qwen-tts `qwen3_tts_model.py:426-432, 449`). That encode is the one allocation in
  the process with no input bound. It is an origin bug and is fixed here, section 4.5.
- No transient of the non LLM components is accounted anywhere. That matters only where the
  cap of 4.4 does not bind. It is deferred with a gate, section 4.8.

One more consequence of the order. The incremental codec graph capture refuses to run below
3 GiB free (`incremental_codec_cuda_graph_min_free_gb`, incremental_codec_cuda_graph.py:79,
219-243) and disables the runner with a warning. On main that reading is taken after the
engine has filled the card, so a full card can turn the codec graphs off silently. With the
vocoder built first the reading precedes the engine.

## 2. Every allocator in the process

| Allocation | Owner and creation point | Lifetime | In the reading after this plan |
| --- | --- | --- | --- |
| talker, code predictor, speaker encoder weights | ModelWorker (bootstrap.py:159) | resident | yes, today too |
| per row decode buffers, predictor K and V cache | talker init (sglang_model.py:895-948) | resident | yes, today too |
| predictor graphs | startup capture (sglang_model.py:1248-1297) | resident, one shared pool | yes, moved before the pool, 4.3 |
| speech tokenizer weights | one copy through the registry, 4.2 | resident | yes, moved before the pool |
| codec state arena, incremental codec graphs, legacy holders | vocoder factory, `warmup_now` (streaming_vocoder.py:1030) | resident | yes, the vocoder is built first, 4.2 |
| KV pool, req_to_token, allocator free list | alloc_memory_pool (bootstrap.py:180) | resident | sized by the rule of 4.1 |
| sglang decode and prefill graphs | init_sglang_cuda_graphs (engine_factory.py:223) | resident, private pools | sglang's own graph reserve, unchanged |
| LLM prefill and decode activations | sglang forward | transient | sglang's slack, unchanged |
| Mimi encode of a reference | the ref code batcher thread, batches of up to 8 clips (request_builders.py:870, 882) | transient | sglang's slack, or the room the cap leaves |
| speaker encoder forward | preprocessing workers, up to 8 (sglang_model.py:404-425) | transient | same |
| streaming and whole utterance decodes | vocoder workers | transient | same |
| cuDNN plans, graph executables, allocator cache | libraries and torch | resident, grows at first use | sglang's slack |

## 3. What already exists and is reused

| Mechanism | Where | Use |
| --- | --- | --- |
| sglang's sizing rule and slack | kv_cache_configurator.py:1764-1811, server_args.py:4956-4983 | untouched, fed a complete resident set |
| `max_total_tokens` as a minimum on the pool | kv_cache_configurator.py:1844-1859 | the cap, 4.4 |
| `mem_fraction_static`, `engine.kv_cache_bytes`, `gpu_memory_fraction` | server args, stage config | untouched deployment knobs |
| the reference hook and its batcher | request_builders.py:943-997, 762-905 | becomes the single encoder caller |
| the engine's input length refusal | omni_scheduler.py:1247-1258, sglang managers/utils.py:193-223 | final authority behind the ICL precheck |

## 4. Design

### 4.1 The rule

    pool = min(running × context, sglang's rule over the true resident set)

Every allocation that lives for the life of the process exists before sglang takes the
reading that sizes the pool. That is an ordering property, not a measurement, and it holds
on every card. The second term is sglang's, unchanged. The first term is the demand bound,
measured on Sep 6.

### 4.2 The vocoder before the engine, one tokenizer per process

Config order becomes preprocessing, vocoder, tts_engine (config.py:54-83). The entry stage
is the first of the list (schema.py:696) so preprocessing stays first. Routing follows the
named edges, `next` and `stream_to`, and does not change. Stages sharing a process keep
config order in the launch spec, "config order is load order inside one OS process"
(topology.py `_group_stages_by_process`), and construction is one factory at a time under
the GPU startup lock (stage_workers.py:493, 884).

`_load_qwen3_tts_tokenizer` (stages.py:46) becomes a process local registry keyed by
tokenizer path, device, dtype and attention implementation. The vocoder loads and registers,
the engine's hook of 4.3 gets the same object. A split layout where the vocoder runs in
another process misses the registry and loads its own copy, as today.

The engine uses only the encoder half, `encode` on the batcher thread (request_builders.py:870,
882) and in the wrapper. The vocoder holds `model.decoder` (streaming_vocoder.py:622) and may
fuse its activations in place under `fused_snake_activation` (640), which touches the decoder
only. The encoder already serves the batcher thread and up to 8 preprocessing workers on one
object today, so the sharing adds decoder use on disjoint submodules and nothing else.

Effect: the tokenizer weights, the codec state arena and every vocoder graph pool are in
free memory when sglang reads it.

### 4.3 Engine model setup and predictor capture before the pool

`create_sglang_infrastructure` (bootstrap.py) gains one optional keyword,
`before_memory_pool`, a callable invoked with the model worker between `ModelWorker`
construction and `alloc_memory_pool` (bootstrap.py:159-180). The engine factory always
passes a closure that calls a builder hook of the same name with the model worker, the
checkpoint directory, the device, the gpu id and the server args. The base hook does nothing.

Qwen3-TTS implements it with what `setup_model` and `setup_model_resources` do today: the
tokenizer attach, now a registry hit, the processor, the wrapper, the preprocessing context,
and the predictor graph capture. Those two hooks become empty for Qwen3-TTS. The predictor
capture reads only buffers the model allocated at init (sglang_model.py:895-948) and runs
SDPA (1816-1823). It does not touch the KV pool or the attention backends (1310-1350), so
nothing it needs is missing before the pool. sglang's own decode and prefill graphs stay
after the pool, where its graph reserve was written for them.

### 4.4 The cap

From the Sep 6 branch, measured there: `max_total_tokens = running × context` when no stage
byte budget is declared (engine_builder.py:149), and the builder's `mem_fraction_static`
default of 0.85 (engine_builder.py:93) removed so sglang derives its slack. The deployment
knob stays. At 16 running the pool holds 131072 tokens, about 14 GB from the archive's two
pool sizes and process footprints, which is every token admission can commit. The rest of
the card is room for the transients of section 2 without any accounting.

### 4.5 The reference path origin fix

Three changes inside the component that owns the work, request_builders.py.

1. The uploaded voice miss path (1137) calls the hook's `encode_one` instead of the wrapper's
   `create_voice_clone_prompt`. The hook is then the only caller of the Mimi encoder and of
   the speaker encoder in the process, and the batcher the only thread that runs the encoder.
2. The hook submits the Mimi encode only in ICL mode. In x vector only mode it runs the
   speaker encoder alone. The codes were never consumed in that mode: `build_voice_clone_inputs`
   takes the text route when `ref_code` is None (sglang_model.py:626-643), the cache artifact
   skips a None code (request_builders.py:697-717), and the vocoder trims nothing when
   `ref_code_len` is 0. No output changes.
3. The ICL precheck. Frames are the tokenizer's own rounding, ceiling of the samples at the
   feature extractor's rate over `encode_downsample_rate` (qwen-tts
   `modeling_qwen3_tts_tokenizer_v2.py:984`). A reference whose frames alone reach the engine's
   `context_length` is refused before the submit, since the engine refuses any input at or
   above `max_req_input_len`, which is below the context (omni_scheduler.py:319-322,
   1247-1258). Everything shorter goes to the engine as today, which keeps final authority.
   The standalone preprocessing process has no engine context and skips the precheck.

No input policy is added. An x vector reference of any length runs as today, without the
encode it never used.

### 4.6 The startup line

`post_scheduler_setup` (engine_builder.py) reports the pool in tokens and GiB from the pool's
own byte accounting, the admission bound, and `mem_fraction_static` as resolved. Free device
memory is already printed by sglang after the pool and after graph capture.

### 4.7 Memory map and flow, 80 GB H100

```
main 53239c285 at 16 running       this plan at 16 running        this plan where the cap does not bind
+---------------------------+      +---------------------------+  +---------------------------+
| free at the failure: 0.5  |      | free: about 40 GB          |  | free: sglang's slack       |
+---------------------------+      |   room for every transient |  +---------------------------+
| slack after residency: ~5 |      |                            |  | slack, 10 GB floor         |
+---------------------------+      +---------------------------+  +---------------------------+
| tokenizer x2, predictor,  |      | KV pool: 131072 tokens     |  | KV pool: rule − residency  |
| codec graphs, arena:      |      |   about 14 GB              |  +---------------------------+
| after sizing, from slack  |      +---------------------------+  | tokenizer x1, predictor,   |
+---------------------------+      | tokenizer x1, predictor,   |  | codec graphs, arena:       |
| KV pool: free − 10 GB     |      | codec graphs, arena:       |  | before sizing, in the read |
|   most of it never used   |      | before sizing, in the read |  +---------------------------+
+---------------------------+      +---------------------------+  | weights                    |
| weights                   |      | weights                    |  +---------------------------+
+---------------------------+      +---------------------------+
```

```
preprocessing factory                                          stages.py:108
vocoder factory: registry load, arena, codec graphs            stages.py:218-302     4.2
tts_engine factory -> build()
  adjust_overrides: max_total_tokens = running x context        engine_builder.py:149 4.4
  create_sglang_infrastructure
    ModelWorker: weights, both free readings                    bootstrap.py:159
    before_memory_pool: registry hit, processor, wrapper,
      preprocessing context, predictor graphs                   new                   4.3
    alloc_memory_pool: sglang's rule, min with the cap          bootstrap.py:180
  init_sglang_cuda_graphs: decode and prefill graphs            engine_factory.py:223
  post_scheduler_setup: the startup line                        engine_builder.py     4.6
serving: the hook prechecks ICL frames, skips the encode in
  x vector mode, the uploaded miss path goes through it         request_builders.py   4.5
```

### 4.8 Deferred, and the gate to bring it back

Revision 5's envelope, before pool measurement, allowances, `kv_cache_reserve_bytes`,
`headroom_bytes`, the byte aware batcher with its refusal above context frames, and the post
capture refusal. They account request time transients of the non LLM components. They matter
only where `running × context` no longer fits and sglang's slack is all the room there is,
about 72 running on an 80 GB card at 8192 context. Gate: a run at the smallest cap where
the bound stops binding that shows a transient of section 2 exceeding the slack. Until then
none of it ships.

sglang's post capture sizing (`SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`, off by default) is not
used. Omni runs it inside `init_cuda_graphs` (sglang_model_runner.py:477-478), before the
predictor capture and before the vocoder exists, and in that mode the derived fraction keeps
1.5 GB of slack instead of the 10 GB floor (server_args.py:4949-4953, kv_pool_runtime.py:56-58)
unless decode graphs do not cover the running cap. It would see one of the three consumers
and leave less room than today. With every consumer above the pool line it has nothing to do.

### 4.9 Alternatives read and not taken

- Telling sglang about the residency through a reserve. Needs the byte counts of graphs
  before they are captured, so a measurement or a constant. Ordering needs neither.
- A 30 s or context derived limit on references, revisions 3 and 4. An input policy where the
  fix is in the work (clarification).
- Chunking the speaker encoder or the Mimi encoder. Model changes with a numerical
  qualification, unnecessary once the discarded encode is gone.

## 5. Portability

The order is the same on every card and every layout. sglang reads the local device. The
plan promises that the resident set is complete at the reading, not equal pool sizes or
throughput across cards.

## 6. Validation

Remote unit checks, per commit: the stage list builds the vocoder before the engine with the
entry stage and the routing unchanged, the registry returns one object for an equal key and
loads for a different one, the bootstrap runs the callback between the worker and the pool,
the engine factory passes it and the Qwen3-TTS hook attaches the tokenizer and captures the
predictor graphs there and nowhere else, the cap and its refusal under a byte budget, the
uploaded miss path reaches the hook, x vector mode submits no encode and yields the same
prompt and artifact, the precheck refuses a reference at context frames before the submit
and passes one below it, the startup line carries every field.

Box, two frozen arms on `53239c285`: A main, B the branch. The cap alone was measured on
Sep 6 against the older main and holds.

| Gate | Setup | Evidence |
| --- | --- | --- |
| The bump failure | the sglang bump on both arms, default cap, CustomVoice with prefill graphs | B completes 16 of 16 where A fails, the startup line shows the pool at the bound |
| Resident set at sizing | default cap, cold boots | the free reading before the pool is below A's by the vocoder residency and the predictor pools, taken from the sglang pool logs and the codec graph footprint line |
| Codec graph gate | same | no `below headroom` warning from the incremental codec runner on B |
| The regime above the cap | the smallest cap where `running × context` exceeds the rule, and one above | B completes with the pool reduced by the residency, A's outcome recorded |
| Reference contract | ICL and x vector, ad hoc and uploaded misses, cache hits, a reference at context frames | codes and speaker embeddings identical to A for the same clips, no encode call in x vector mode, the refusal before device work |
| c1 and c16 corpus | two boots per arm, paired A B B A, the 1 s all GPU sample | c1 byte identical to A, c16 inside the archived range, throughput inside the paired spread, the kernel census |
| Startup time | same | B against A, the registry saves one tokenizer load |

## 7. The commit series inside the one PR

1. The vocoder before the engine, the keyed tokenizer registry, tests.
2. `before_memory_pool` in bootstrap and the engine factory, the Qwen3-TTS hook with the
   tokenizer attach and the predictor capture, tests.
3. The cap and the builder fraction default, the startup line, tests.
4. The reference hook as the single encoder caller, no encode in x vector mode, the ICL
   precheck, tests.

## 8. Decisions taken

1. Ordering, not measurement. No envelope, no probes, no reserve, no headroom field.
2. The cap stays as the demand term. It is not a memory saving for its own sake, it is the
   most the pool can ever hold live at that running count.
3. No input policy. The x vector encode is removed because its output was never used.
4. Transient accounting is deferred behind the gate of 4.8.
5. The post capture flag is left alone.

## 9. Validation tasks and open facts

1. The resident bytes of one tokenizer copy, the predictor graph pool, the codec state arena
   and the codec graph pools on `53239c285`, from the startup logs of the B arm.
2. The pool size in bytes at the cap from the startup line, against the 14 GB estimate.
3. The free reading before the pool on B against A, and how much of A's slack the residency
   consumed.
