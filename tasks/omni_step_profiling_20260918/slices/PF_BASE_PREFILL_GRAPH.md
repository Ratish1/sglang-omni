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

## 2. Why it matters on this card

run01 prefill bs 1: the step span is 25.9 ms for 10.45 ms of device time, about 470
launches, 4 stream syncs; the step is host-bound. A replayed layer stack removes the
per-kernel launches of 28 layers.

## 3. Change (branch `perf/qwen3-tts-base-prefill-graph`, not committed)

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
