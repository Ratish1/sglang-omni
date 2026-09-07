# Qwen3-TTS memory provisioning plan

Revised Sep 7 2026 after the kv pool archive (readout 11) and the vLLM comparison
(`research/vllm_omni_memory_sizing.md`, `research/model_memory_provisioning.md`). One PR
carries the whole thing: the admission cap already on
`perf/qwen3-tts-kv-pool-admission-bound`, one tokenizer per process, the vocoder built before
the engine, the probes, the reserve, and a budget aware whole utterance decode. The rope store
is its own branch and lands after. Every mechanism below was read at the cited line on that
branch merged with main `53e94dfa5`, or in sglang v0.5.18 at
`/Users/ratish/sglang-worktrees/v0.5.18`.

## 1. The requirement

The process on the card is one OS process holding three stages, built in config order
(stage_workers.py:493-500): preprocessing, tts_engine, vocoder. Memory is provisioned once,
inside the engine stage's factory, when sglang sizes the KV pool (bootstrap.py:180). Anything
resident at that moment is seen. Anything loaded or allocated after it lives off whatever the
pool left.

The correct rule, and the one vLLM implements for the second term:

    pool = min(what admission can commit, budget − measured need)

The first term is the running cap times the context length, 16 x 8192 = 131072 tokens,
14.0 GiB at 114688 bytes per token, exact and free. It is on the branch and measured: the
pool drops from 589142 tokens, ready memory from 75.2 GB to 25.0 GB, no allocator retry, c1
byte identical, throughput flat. The second term does not exist in this tree for any model,
and it is what keeps a deployment whose bound exceeds the card, 128 running, from filling it
again. This plan adds it.

## 2. What the process holds, and what sizes each part today

| Component | Where it is created | Before or after the pool | Sized by | Measured |
| --- | --- | --- | --- | --- |
| talker, predictor, speaker encoder weights | ModelWorker in create_sglang_infrastructure (bootstrap.py:159) | before | checkpoint | 4.6 GiB with the text embedding, from the snapshot's block sizes |
| KV pool | alloc_memory_pool (bootstrap.py:180) | the pool | this plan | 14.0 GiB at 16 running |
| sglang decode and prefill graphs | init_sglang_cuda_graphs (engine_factory.py:223) | after | sglang's graph reserve, formula (server_args.py:5078-5118) | inside the 6 GB at ready |
| predictor graphs | setup_model_resources (engine_builder.py:172-179) | after | bucket ladder | small at 16 running, 20 to 39 graphs at 128 |
| speech tokenizer, engine copy | setup_model (engine_builder.py:126-132) | after | checkpoint | not yet named, validation task |
| speech tokenizer, vocoder copy | create_vocoder_executor (stages.py:245) | after | checkpoint | same |
| vocoder decode graphs, 3 holders x 64 graphs | warmup_now in the vocoder factory (stages.py:279, streaming_vocoder.py:699-704) | after | shape table (streaming_vocoder.py:54-90) | 3 x 1576 MiB reserved, one stream per holder |
| reference encode transient | ref code thread and stream (request_builders.py:749-756, 834-843) | runtime | reference audio, 30 s for uploaded voices (speech_voices.py:33-34), 10 MiB for ad hoc (speech_limits.py:10) | 650 to 852 MiB reserved on that stream |
| LLM prefill and decode activations | sglang | runtime | sglang's activation reserve, formula (server_args.py:4956-4977) | inside sglang's slack |
| whole utterance decode transient | _vocode_payloads (streaming_vocoder.py:1865-1895) | runtime | batch up to 8 (stages.py:223) x up to 2048 generated frames (request_builders.py:43) plus reference frames | 400 to 760 MiB per call at about 52 frames, the largest allocation of both arms |
| non torch: CUDA context, cuDNN plans, graph executables | driver | runtime | nothing | whole device minus torch reserved, about 2.2 GB plus context at the c16 peak |

Three facts from the reads that the design rests on.

The preprocessing stage loads no tokenizer in the shipped layout. `load_frontend` is set only
when preprocessing runs in its own process (config.py:85-94, stages.py:117-124). So there are
two copies, the engine's and the vocoder's, and both load after the pool. The engine's copy
serves one call, `encode` of reference audio on the ref code thread (request_builders.py:840-843,
884-893). The vocoder's copy serves the decoder half only: `tokenizer.model.decoder` for the
graphs and windows, `tokenizer.decode` for whole utterances (streaming_vocoder.py:503,
1884-1886, 1927). Disjoint halves of one object.

Codec streaming to the vocoder happens only when the HTTP request streams. The engine emits
code chunks only if `stream_codec_output` and `params["stream"]` both hold
(request_builders.py:1546-1552), and the vocoder classifies a payload as streaming by
`params["stream"]` alone (scheduling/streaming_vocoder.py:170-177). Every non streaming request,
the benchmark included, arrives as one terminal payload with all its codes and is decoded in
one call, batch up to 8. Streaming requests decode windows of at most 24 frames, graph captured.
The unbounded transient is the non streaming path, and it is the production path for every
client that does not stream.

sglang's measurements are device wide `torch.cuda.mem_get_info` after `empty_cache`
(v0.5.18 utils/common.py:433-443), before the weights load (distributed/bootstrap.py:132-137)
and again before the pool (kv_cache_configurator.py:1768-1773). `empty_cache` releases the
allocator's cached blocks first, so a transient that already ran is not seen, only live
tensors and non torch allocations are. That is why a probe alone does not reserve anything.
Its peak has to be subtracted explicitly.

The failure the archives recorded: at 128 running the pool takes the card, the vocoder's
transients fill the allocator's cache, and cuDNN's own allocation for the speaker encoder's
convolution fails with `CUDNN_STATUS_INTERNAL_ERROR` at about 6 MiB free (revalidation
LIMITATIONS.md:9). Non torch allocations cannot be served from torch's cache, so the last few
MiB of the card decide whether cuDNN runs.

## 3. Design

Four changes, all omni owned, none in sglang. In the order they act at startup.

### 3.1 Build the vocoder before the engine, with one tokenizer

Move the vocoder ahead of the engine in the stage list (config.py:54-83), so config order
becomes preprocessing, vocoder, tts_engine. The vocoder factory loads the tokenizer, captures
its 192 graphs (stages.py:245-279) and publishes the tokenizer for the process. The engine's
`setup_model` takes the published object instead of loading a second one
(engine_builder.py:126-132) and attaches it to the talker for the ref code thread. Sharing is
safe by the reads above: the two users touch disjoint halves, the graph holders own their
static buffers per worker and not the module (streaming_vocoder.py:671-675), and the only in
place mutation of the decoder, the fused SnakeBeta swap, is off by default and would apply
to the one object either way.

Effect: the tokenizer weights and the 4.6 GiB of vocoder graph pools are resident when sglang
measures free memory, so its own rule charges them. One tokenizer copy fewer.

What must hold for the reorder, validation task 1: no construction order dependency between
the engine and the vocoder. `stream_to` targets are wired after all stages exist, readiness
is published after the last factory returns (stage_workers.py:456-465), and the vocoder
factory reads nothing from the engine. Preprocessing already builds before both.

### 3.2 Probe the components at their declared maximum before the pool

In the engine builder's `pre_infra_setup` (engine_factory.py:92, no weights loaded, CUDA
device set), run each non LLM component once at the largest shape a request can bring, and
record the torch peak of each run with `torch.cuda.max_memory_allocated` deltas:

- the tokenizer encoder on 30 s of audio, the uploaded voice bound
  (speech_voices.py:33-34), and the speaker encoder on the same audio
- the tokenizer decoder on one utterance of `max_new_tokens` plus the reference bound,
  2048 plus 360 frames at 12 Hz, batch 1
- the decoder on two frame counts, to measure bytes per frame

The runs build cuDNN's plans for those shapes, which then stay resident and are seen by
sglang's profile. The peaks are the measured need. The bytes per frame is the slope the
vocoder uses in 3.4. Nothing here is a constant: every shape comes from a declared bound
in the serving layer or the request defaults, and every byte count is measured on the card
at startup.

### 3.3 Subtract the measured need through the omni configurator

`_OmniKVCacheConfigurator` gains one field, `kv_cache_reserve_bytes`, carried the same way
`total_gpu_memory_fraction` is today: `infra_kwargs` (engine_factory.py:175) into
`ModelWorkerConfig` (model_worker.py:28-35) into the configurator
(sglang_model_runner.py:610-618). In `_profile_available_bytes`, when neither a byte budget
nor a stage fraction is declared, the upstream rule runs unchanged and the reserve is
subtracted from its result:

    bytes = upstream(free_after_load) − kv_cache_reserve_bytes
          = free − pre_load × (1 − f) − reserve

sglang's own slack, `pre_load × (1 − f)` with f derived (server_args.py:4979-4983), stays as
the LLM's activation and graph reserve. The omni reserve is the sum of the probe peaks from
3.2. The admission cap keeps applying as the minimum afterwards (kv_cache_configurator.py:
1844-1859), so the pool is

    tokens = min(running × context, (free − slack − reserve) // cell_size)

At 16 running on this card the second term is about 45 GiB against 14 GiB, the pool is the
one already measured, nothing changes. At 128 running the second term wins and the reserve is
what keeps the card from filling. The byte path and the fraction path are untouched: a
deployment that declares `engine.kv_cache_bytes` or `gpu_memory_fraction` keeps its own rule.

### 3.4 Make the whole utterance decode fit a budget

`_vocode_payloads` (streaming_vocoder.py:1865-1895) decodes its batch in one call today. It
becomes budget aware: with the measured bytes per frame from 3.2 and the reserve as its
budget, it splits a batch into calls whose total frames fit, and decodes a single utterance
that alone exceeds the budget through the windowed decoder that the streaming path already
uses (`chunked_decode`, streaming_vocoder.py:1192). A batch that fits is decoded exactly as
today, so c1 audio stays byte identical on this corpus, where every utterance is far below
the budget. c16 batches may split, which changes bytes on a path whose bytes already vary
per boot.

This is the piece that makes the reserve a guarantee: without it the reserve is a number the
vocoder can exceed on a batch of long utterances, with it the transient never exceeds what was
subtracted.

### 3.5 The startup line

`post_scheduler_setup` (engine_builder.py:206-218) reports pool tokens, the admission bound,
the reserve, and the free memory at pool end, so a deployment reads which term won.

### 3.6 Memory map, 80 GB H100

```
main, fraction 0.85            branch today, cap             this plan
+----------------------+ 81 GB +----------------------+ 81 GB +----------------------+ 81 GB
| free at peak: <1 GB  |       | free: ~48 GB         |       | free: ~48 GB at 16   |
|                      |       |                      |       |  reserve at 128      |
+----------------------+       |                      |       +----------------------+
| vocoder transients,  | ~7    |                      |       | vocoder transients   | <= reserve
| cache, cuDNN plans   |       +----------------------+       | (split to budget)    |
+----------------------+       | transients, graphs   | ~7    +----------------------+
| graphs, tokenizer x2 | ~6    +----------------------+       | graphs, tokenizer x1 | seen by
+----------------------+       | graphs, tokenizer x2 | ~6    | probes' cuDNN plans  | sglang
|                      |       +----------------------+       +----------------------+
|  KV pool 589142      | 62.9  | KV pool 131072       | 14.0  | KV pool = min(bound, | 14.0 at 16
|  (bound 131072)      |       |                      |       |  free−slack−reserve) | ~45 at 128
+----------------------+       +----------------------+       +----------------------+
| weights              | 4.6   | weights              | 4.6   | weights              | 4.6
+----------------------+       +----------------------+       +----------------------+
```

```
startup, this plan                                              file:line
preprocessing factory                                           stages.py:106
vocoder factory: tokenizer, publish, 192 graphs                 stages.py:216-279   (3.1)
tts_engine factory -> build()
  pre_infra_setup: probes, peaks, bytes per frame               engine_factory.py:92   (3.2)
  adjust_overrides: max_total_tokens = running x context        engine_builder.py:152-158
  infra_kwargs: kv_cache_reserve_bytes = sum of peaks           engine_factory.py:175  (3.3)
  create_sglang_infrastructure
    weights load, free measured after empty_cache               bootstrap.py:159, kvcc.py:1768
    _OmniKVCacheConfigurator: upstream rule − reserve           sglang_model_runner.py:68  (3.3)
    min with the cap, pool allocated                             kvcc.py:1844-1859, 1946-1968
  setup_model: take the published tokenizer                     engine_builder.py:126  (3.1)
  graphs, predictor graphs                                      engine_factory.py:223, 235
  post_scheduler_setup: pool, bound, reserve, free              engine_builder.py:206  (3.5)
serving: _vocode_payloads splits to the budget                  streaming_vocoder.py:1865 (3.4)
```

### 3.7 Alternatives read and not taken

- Route every non streaming request through the windowed decoder. Bounds the transient
  with no reserve, but changes the audio of every non streaming response and the c1 baseline
  with it. 3.4 keeps whole utterance decode where it fits and windows only what cannot.
- Load and probe inside the engine's `pre_infra_setup` without reordering the stages. Works
  for the tokenizer, but the vocoder graphs would still capture after the pool and need a
  second measurement to subtract. Reordering lets sglang's own rule see them.
- A stage `gpu_memory_fraction` for tts_engine through the existing process budget path. It
  is a constant no measurement pins for this model, and it still measures no activation peak.
- Sizing from `torch.cuda.set_per_process_memory_fraction` alone, the `total_reserve_bytes`
  mechanism (stage_workers.py:805-838). It fences torch but reserves nothing for cuDNN's own
  allocations, which is where the card failed.

## 4. What the freed memory unblocks

Unchanged from the first draft and now measured for the first two: 48 GB free at c16 on an
80 GB card, the larger graph ladders at 128 running, the rope store's c16 pair, a second
replica per card at about 32 GB each, and colocation of another stage process.

## 5. Validation

Unit, per commit: the cap tests already on the branch; the shared tokenizer is one object in
both stages with the encoder and decoder halves reachable; the reserve reaches the configurator
and is subtracted only on the upstream path; the split keeps every call within the budget and
windows an oversize utterance; the startup line carries all four numbers.

Box, paired protocol of plan 07, A is upstream main, B is the whole branch:

1. Startup: pool 131072, the reserve printed, ready memory below the 25.0 GB of the cap alone
   by one tokenizer copy, the probes' time in the ready line.
2. c1 full corpus: 1088 of 1088 byte identical to the archive, cached tokens 13576 of 74096.
3. c16 full corpus, two boots per arm: no allocator retry on B, quality inside the archived
   range of 114 to 135 errors and 71.12 to 71.34 similarity, throughput inside the paired
   spread.
4. 128 running with the request level subtalker top k 64: 64 of 64, no cuDNN error, and the
   new gate, whole device peak at least the reserve below the card.
5. 128 running with `max_new_tokens` 2048 on a long text set, the split exercised: every
   request completes, the log shows the splits, peak still the reserve below the card.
6. A streaming run at c16: unchanged path, unchanged audio contract.
7. Retraction at c16 as before, 192 of 192.

## 6. The commit series inside the one PR

Each commit reviewed by you before it lands on the branch.

1. On the branch already: the cap, the fraction removal, the startup line.
2. Stage order and the shared tokenizer, with the publish mechanism.
3. The probes in `pre_infra_setup` and the bytes per frame measurement.
4. `kv_cache_reserve_bytes` through `infra_kwargs`, `ModelWorkerConfig` and the configurator.
5. The budget aware `_vocode_payloads`.
6. The startup line extended.

## 7. Decisions

1. Reorder the stages, 3.1, against loading the tokenizer in the engine's hook. Recommended:
   reorder, it is one line in the config and it lets sglang see the graphs without a second
   measurement. Depends on validation task 1.
2. The reserve is the sum of the probe peaks, nothing added. Any margin would be a constant.
3. The vocoder's budget equals the reserve. Recommended, it is the number the pool sizing
   subtracted, so the guarantee is exact.
4. The reference bound used by the probe is the uploaded voice limit, 30 s. Ad hoc references
   are bounded in bytes, not seconds, so an ad hoc reference longer than 30 s decodes through
   the window path of 3.4 when it exceeds the budget.

## 8. Validation tasks and open facts

1. Read stage_workers.py:456-560 and the stream wiring for any dependency on the engine being
   built before the vocoder.
2. The size of one tokenizer copy, from the ready memory delta of commit 2's box run.
3. The bytes per frame of the decoder and the three probe peaks on this card, from the
   startup line of commit 3.
4. Whether `Qwen3TTSTokenizer.decode` pads a batch to its longest item, which decides whether
   a split changes bytes of the shorter items. Read from the installed qwen_tts on the box.
5. The 114688 bytes per token is arithmetic on the log, the unit test uses the resolved cell
   size.
