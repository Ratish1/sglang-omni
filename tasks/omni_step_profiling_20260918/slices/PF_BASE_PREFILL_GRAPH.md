# Slice PF: breakable prefill CUDA graphs for Base checkpoints

## 1. What is off and why

`Qwen3TtsEngineBuilder.generation_defaults` (`engine_builder.py:113-123` at upstream
main) sets `cuda_graph_backend_prefill: breakable` with the ladder
`QWEN3_TTS_PREFILL_CUDA_GRAPH_BS` (1, then 4 to 512) only when the checkpoint's
`tts_model_type` is `custom_voice`. #1581 made the Talker satisfy the breakable prefill
contract; #1900 enabled it by default for CustomVoice after an H100 measurement (TTFB p50
53.7 to 46.1 ms at 1 RPS, p95 audible TTFA 438.0 to 411.5 ms at 10 RPS, 2.8 s and 0.46 GB
of capture) and scoped it with "Base prefills additionally carry reference audio and
VoiceDesign is unmeasured; both keep the eager path". The Base 1.7B we serve therefore
prefills eagerly; the run01 server log says so ("Disable prefill CUDA graph because
cuda_graph_config resolved prefill.backend='disabled'").

The scoping is a missing measurement, not a defect:

- SGLang 0.5.19's breakable backend captures the Talker's layer stack per token-count
  bucket and replays it with the live `input_embeds`; attention runs eager at the graph
  breaks and the logits processor eagerly on top, so any mix of request lengths replays
  (`prefill_cuda_graph_runner.py:1197-1246`). A bucket more than twice the real token
  count falls back to eager (omni `_PREFILL_PADDING_FACTOR`, SGLang
  `_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR`).
- The reference audio is consumed in preprocessing (speaker encoder, reference codes);
  the Talker prefill receives `input_embeds` through the same
  `attach_omni_prefill_inputs` path for every checkpoint type
  (`model_runner.py:55-75`).
- What differs is only the length: Base ICL prompts are the reference code frames plus
  about 10 tokens. run01 serve.log: single prefills of 53 to 78 tokens, coalesced
  prefills of 222 to 920 tokens (3 to 14 requests).

## 1a. Audit of the implementation against SGLang 0.5.19 (read in full:
`runner/prefill_cuda_graph_runner.py`, `runner_backend/breakable_cuda_graph_backend.py`;
omni: `model_runner/prefill_inputs.py`, `model_runner/sglang_model_runner.py`,
`scheduling/bootstrap.py`, `scheduling/engine_factory.py`,
`utils/cuda_graph_batch_validator.py`, `models/qwen3_tts/model_runner.py`,
`models/qwen3_tts/sglang_model.py`; history #1581, #1900, #1907)

```
startup
  engine_factory.py:164-174   backend == breakable -> requires supports_breakable_prefill_cuda_graph
                              (Qwen3-TTS CAPABILITIES: True) -> enable_prefill_input_embeds
  bootstrap.py:46-70          model_config.is_multimodal = True only around init_cuda_graphs
  SGLang runner :343-386      is_multimodal -> PrefillInputBuffers / registry allocate the
                              static input_embeds slot (hidden_size wide)
  SGLang runner :560-571      layer_model = Talker.model (first module with .layers)
  capture :1259-1403,:710-757 dummy EXTEND batch; layer_model.forward(input_ids, positions, fb,
                              input_embeds=<static slot>) -> Qwen3TTSTalkerTextModel takes the
                              input_embeds branch (sglang_model.py:338-346), no codec lookup,
                              no feedback mask inside the graph
  cuda_graph_batch_validator.py:280-321  startup attests backend, buckets, and the
                              input_embeds slot; a missing slot is a hard error

each prefill
  Qwen3TTSModelRunner.before_prefill (model_runner.py:55-75)
    _ensure_mrope_positions   can_run_graph(fb) -> mirror positions into [3,T] mrope
                              (#1907: the captured graph binds the mrope slot; SGLang refreshes
                              it only when the live batch carries mrope positions)
    attach_omni_prefill_inputs  embeds on a private sidecar, NOT fb.input_embeds, because
                              can_replay_locally rejects fb.input_embeds (:1159)
  SGLModelRunner._extend_forward_kwargs (sglang_model_runner.py:311-342)
                              sidecar -> kwargs input_embeds (dtype checked), omni_prefill_rids
  SGLang execute (:1878-1921) load_batch pads to the bucket, refreshes static seq_lens,
                              positions, mrope, out_cache_loc; attention metadata eager
    _execute_body_capture (:1754-1811) layer_model.forward := replay closure;
      Talker.forward (eager) -> self.model(input_embeds=live) -> closure copies live embeds
      into the static slot[:rows] (:1777-1784) -> graph replay
      -> hidden[_extend_last_index] with live extend_seq_lens -> codec_head (eager tail)
```

Findings:

1. The two contracts a captured Talker prefill needs beyond SGLang's generic ones are the
   live embeddings and the live rope positions. Both are met: embeddings through the
   sidecar, kwargs, and the replay closure's slot copy; rope positions through the mrope
   mirror.
2. The `can_run_graph` verdict used by the mirror (before the sidecar is attached) equals
   the verdict at execute time: neither sees `fb.input_embeds`, and nothing the mirror
   changes is read by `can_replay_locally` (`:1132-1195`).
3. Padded rows: the output buffer keeps bucket rows; the Talker selects rows by the live
   `extend_seq_lens` cumsum, so padded rows never reach the codec head; the tail trims
   to the raw token count (`:1831-1859`).
4. Prefix-cache hits (serve.log shows `#cached-token: 6`) replay on CUDA; attention runs
   eager with the live prefix metadata; the sidecar embeddings cover only the extend
   window (`_projected_prefill_slice` with `prefix_len`).
5. Numerics: a replayed prefill rotates through the 2-D mrope kernel (`forward_triton`),
   an eager one through the 1-D kernel (`forward_native`) (#1907 docstring). Same values
   by construction (three equal rows), not necessarily the same bits; #1907 measured a
   fixed-seed request byte-identical on CustomVoice. To be re-measured on Base: a fixed-
   seed request, eager vs replayed, codes compared (run 07 `pf` arm plus a seeded pair).
6. Nothing on this path depends on the checkpoint type. The Base-only difference is
   prompt length, which the ladder and the 2x padding guard handle.

## 2. Why it matters on this card

run01 prefill bs 1: the step span is 25.9 ms for 10.45 ms of device time, about 470
launches, 4 stream syncs; the step is host-bound. A replayed layer stack removes the
per-kernel launches of 28 layers.

## 3. Change (branch `perf/qwen3-tts-base-prefill-graph`, `2f60fc1a8` on upstream `27b5b0d4f`, pushed)

The breakable backend and ladder become defaults for every checkpoint; the now-unused
`qwen3_tts_checkpoint_model_type` and the builder's `checkpoint_dir` default go; the
scoping test goes, the default test stays; the cookbook section is updated. 4 files,
+9 -80.

## 4. Measurements

- Run 07 arm `pf`: prefill bs 1 and 8 ledgers against base (span, host share, launches);
  decode unchanged; `serve.log` capture time and memory after capture (the 0.46 GB
  H100 figure is re-measured on the 4090, where 2.44 GB is free after decode capture at
  `mem_fraction_static` 0.85).
- Graph replay vs eager counts: SGLang logs `cuda graph: True/False` per prefill batch
  in serve.log; the fraction of prefills replayed at c1, c16 and c32 comes from those
  lines.
- Then the matrix TTFC cells and the census.

Run 07 (READOUT_07 section 3): prefill bs 1 step 26.06 to 13.73 ms, 91 of 93 prefills
replayed, capture 31 buckets in 3.58 s and 0.20 GB. Decode unchanged.

## 5. PR gate: run 09 (TESTING.md section 8)

One boot per arm and point, all six at once on cards 1 to 6: stream c16, seeded stream
c1, buffered c16; full English seed-tts, warmup 1; WER and similarity on each boot's
WAVs. Validation tasks the run answers (no claim until read):

1. Client deltas: req/s, audio s/s, RTF, TTFC mean/p50/p95/p99, inter-chunk, buffered
   latency; B's prefill replay share and A's eager share from `prefill_lines.txt`.
2. Finding 5: seeded c1 WAVs, A (eager) against B (replayed), byte identical or not. Not
   identical is not a failure; the census is the gate.
3. The ladder top (512) was sized for CustomVoice text prompts. Base coalesced prefills
   reach 920 tokens in run01; the share of prefills and of prefill tokens above 512 at
   c16 comes from B's prefill lines. A graph removes launch cost, which matters only
   while the step is host-bound; extending the ladder is considered only if eager
   extends above 512 are measured host-bound (ledger), with the capture memory stated.
4. VoiceDesign also takes the default with this change. Its prompt is text only like
   CustomVoice and nothing on the path depends on the checkpoint type (finding 6); the
   box holds only the Base checkpoint, so a VoiceDesign boot (capture, a few requests)
   needs the checkpoint downloaded first.
