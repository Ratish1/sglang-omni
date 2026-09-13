# Fun-CosyVoice3 Flow CUDA Graph and HiFT streaming mechanics

Date: 2026-09-13
Scope: read-only mechanics, no design proposals.

Sources read in full:

- omni worktree `/Users/ratish/sglang-omni/.worktrees/cosyvoice-stream-liveness`, head
  `691c18371` (merge of `upstream/main` into `perf/cosyvoice3-stream-scheduler-liveness`).
  - `sglang_omni/models/fun_cosyvoice3/stages.py` (1932 lines)
  - `sglang_omni/models/fun_cosyvoice3/config.py`
  - `sglang_omni/models/fun_cosyvoice3/streaming.py`
  - `sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py`
- CosyVoice model code. The `cosyvoice` package is NOT installed on this Mac
  (`import cosyvoice` fails, no site-packages copy anywhere on the filesystem). A partial
  vendored snapshot of the upstream sources exists in the repo at
  `/Users/ratish/sglang-omni/tasks/cosyvoice_utilization_20260912/inputs/external_sources/CosyVoice/`
  and that is what all `cosyvoice/...` anchors below point at:
  `cosyvoice/flow/flow.py`, `cosyvoice/flow/flow_matching.py`, `cosyvoice/flow/DiT/dit.py`,
  `cosyvoice/flow/DiT/modules.py`, `cosyvoice/utils/mask.py`, `cosyvoice/hifigan/generator.py`,
  `cosyvoice/hifigan/f0_predictor.py`, `cosyvoice/transformer/upsample_encoder.py`,
  `cosyvoice/transformer/convolution.py`.
  UNCONFIRMED: that this snapshot is byte-identical to the `cosyvoice` package installed in the
  H100 serving venv. It carries no provenance file (no commit hash, no tag).
  UNCONFIRMED: the checkpoint yaml `cosyvoice3.yaml` is not in the snapshot, so every value that
  comes from the yaml (DiT `static_chunk_size`, `inference_cfg_rate`, HiFT `upsample_rates`)
  is taken from class defaults or from the omni MLX config mirror and is flagged where used.

Constants used throughout:

- `TOKEN_MEL_RATIO = 2` (`streaming.py:14`), `PRE_LOOKAHEAD_LEN = 3` (`streaming.py:13`),
  `TOKEN_HOP_LEN = 25` (`streaming.py:12`), `TOKEN_MAX_HOP_LEN = 100` (`streaming.py:16`),
  `SAMPLE_RATE = 24000` (`streaming.py:17`).
- Mel frame rate 50 Hz, token rate 25 Hz, 480 audio samples per mel frame
  (`utils.py:146-149`: n_fft 1920, hop_size 480, 24 kHz).
- DiT `static_chunk_size = 50` mel frames = 1 second = one 25-token hop
  (`cosyvoice/flow/DiT/dit.py:119` default, mirrored at
  `sglang_omni/models/fun_cosyvoice3/mlx/vocoder/config.py:34`). UNCONFIRMED against the yaml.

---

## 1. FlowCudaGraphRunner

### What is captured

`FlowCudaGraphRunner.capture` (`stages.py:395-436`) captures exactly one function per shape:
`solve_flow_euler(self.flow.decoder, *static_inputs)` (`stages.py:427`), that is the whole
10-step Euler loop including the CFG doubling and all 10 DiT forwards. Nothing else is in the
graph: the token embedding, pre-lookahead layer, repeat_interleave, mask build and the
prompt scatter all stay outside it in `generate_flow` (`stages.py:530-628`).

Static inputs are built by `capture_inputs` (`stages.py:358-393`), six tensors in this order:

| tensor | shape | dtype | source |
| --- | --- | --- | --- |
| `noisy_mel` | (B, 80, T) | parameter dtype | `decoder.rand_noise[:, :, :T]` expanded and cloned, `stages.py:368-373` |
| `time_span` | (11,) | parameter dtype | `linspace(0, 1, 11)`, cosine warped if `t_scheduler == "cosine"`, `stages.py:374-376` |
| `token_condition` | (B, 80, T) | parameter dtype | `zeros_like(noisy_mel)`, `stages.py:377` |
| `mel_mask` | (B, 1, T) | parameter dtype | `ones`, `stages.py:378-380` |
| `speaker_embedding` | (B, 80) | autocast dtype (bf16) or parameter dtype | `stages.py:381-384`; width is `spk_embed_affine_layer.out_features` |
| `prompt_mel` | (B, 80, T) | parameter dtype | `zeros_like(noisy_mel)`, `stages.py:385` |

Parameter dtype is fp32 for the shipped config: the vocoder `dtype="bfloat16"`
(`config.py:133`) maps to `autocast_dtype=bfloat16` (`stages.py:1841`) and `fp16=False`
(`stages.py:1852`), so `CosyVoice3(checkpoint, fp16=False)` keeps fp32 weights
(`stages.py:817`). At run time the matching runtime tensors are also fp32, because
`PreLookaheadLayer` ends in `outputs + inputs` (`cosyvoice/transformer/upsample_encoder.py:102`)
where `inputs` is the fp32 `nn.Embedding` output, so the bf16 conv result is promoted back to
fp32. `speaker_embedding` is the output of `spk_embed_affine_layer` under autocast
(`stages.py:530-532`), hence bf16 in both capture and replay. That is why the dtype equality
check at replay passes at all.

Capture mechanics: one side stream (`stages.py:400-403`), one shared graph pool created once
(`self.pool = torch.cuda.graph_pool_handle()`, `stages.py:404`) and passed to every
`torch.cuda.graph(...)` (`stages.py:415-418`), one eager warmup solve before each capture
(`stages.py:407-412`), `capture_error_mode="thread_local"`, autocast wrapped around both the
warmup and the capture. After the loop the default stream waits on the capture stream and
`torch.cuda.empty_cache()` runs (`stages.py:433-434`).

### Shapes and buckets

`FLOW_CUDA_GRAPH_FRAME_BUCKET = 16` (`stages.py:70`). `verify_flow_cuda_graph_capture_shapes`
(`stages.py:318-334`) rejects any capture shape whose mel_frame is not a multiple of 16.

Default capture shapes, `FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES`
(`config.py:19-46`), 26 entries:

```
B=1: 304 320 336 352 368 384 400 416 432 448 464 480 496 512 528 544 560 576 592 608 640
B=2: 384 400 448 496 544
```

Note the B=1 ladder is every 16 frames from 304 to 608, then jumps to 640. Bucket 624 is not
captured, so a batch whose frame count falls in (608, 624] misses. B=1 coverage in seconds of
mel is 304/50 = 6.08 s to 640/50 = 12.8 s including the prompt region. No shape with B >= 3
is captured, so any Flow group of 3 or more rows always runs eager.

### Matching a request to a graph

`FlowCudaGraphRunner.run` (`stages.py:448-519`):

1. Reject if `noisy_mel.ndim != 3` (`stages.py:459-460`).
2. Reject if any of `noisy_mel, token_condition, mel_mask, prompt_mel` has a last dim that
   differs from `noisy_mel.shape[2]` (`stages.py:461-465`).
3. Key = `(batch_size, bucket)` with `bucket = ceil(T / 16) * 16` (`stages.py:471-476`).
4. Miss on the key means eager (`stages.py:477-478, 496-497`).
5. On a hit, `right_pad_mel_frames` (`stages.py:438-446`) zero-pads `noisy_mel`,
   `token_condition`, `mel_mask` and `prompt_mel` on the right to `bucket`;
   `time_span` and `speaker_embedding` pass through untouched (`stages.py:480-495`).
6. A final equality check on shape, dtype and device against every static input; any mismatch
   returns None and the caller falls back to eager (`stages.py:498-504`).
7. Replay: copy each padded input into its static buffer, `graph.replay()`, then return
   `static_output[..., :actual_mel_frame].clone()` (`stages.py:514-519`).

The padded frames therefore get a zero mel mask, which is what makes the padding harmless:
the non-streaming DiT mask path passes `masks` straight through, every query row is the
validity vector (`cosyvoice/flow/DiT/dit.py:166` repeats it to (B, L, L)), so valid queries
never attend to a padded column, and the post-attention output mask
`mask[:, 0, -1]` (`cosyvoice/flow/DiT/modules.py:400-405`) zeroes the padded positions.
The one conv over the time axis inside the DiT, `CausalConvPositionEmbedding`
(`cosyvoice/flow/DiT/modules.py:115-144`), pads only on the left
(`F.pad(x, (kernel_size - 1, 0, 0, 0))`), so it is strictly causal and a right pad cannot
reach an earlier frame. The padded output columns are then dropped by the `[:actual]` slice.

### When eager runs instead

`generate_flow` (`stages.py:630-662`):

```
if streaming or not finalize or flow.cuda_graph_runner is None:  # stages.py:630
    eager solve_flow_euler(..., streaming=streaming)             # stages.py:631-640
else:
    generated = runner.run(...)                                  # stages.py:642-649
    if generated is None:                                        # shape/dtype/key miss
        eager solve_flow_euler(..., streaming=False)             # stages.py:653-662
```

So every causal streaming hop is eager by construction, because `inference_causal` passes
`streaming=True, finalize=False` (`stages.py:730`). Only the buffered whole-utterance path
(`FunCosyVoice3Flow.inference`, `stages.py:711-721`, defaults `streaming=False,
finalize=True`) can hit a graph. The streaming leftover call does not go through
`generate_flow` at all: `token2wav_chunk` calls the native CosyVoice
`flow.inference` (`stages.py:1337-1351`), which reaches
`CausalConditionalCFM.forward` and upstream `solve_euler`
(`cosyvoice/flow/flow_matching.py:202-227, 71-124`). The same is true of every B=1
streaming hop (`streaming_vocoder.py:353-358`).

### Capture cost

Per graph the durable tensors are the six static inputs plus the static output:

```
bytes(B, T) = B * T * 4 * (80 noisy + 80 token_cond + 1 mask + 80 prompt + 80 output)
            = 1284 * B * T   (plus B*80*2 for the bf16 speaker embedding and 44 for time_span)
```

Summed over the 26 default shapes, sum(B*T) = 14304, so about 18.4 MB of static input and
output buffers in total. That is the small part. The activation memory for the 10 DiT steps
lives in the shared graph pool (`stages.py:404`), so it is allocated once and sized by the
largest capture rather than by their sum; its absolute size is UNCONFIRMED (nothing in this
code path logs allocator stats before and after capture).

Capture wall time: UNCONFIRMED. `capture()` logs nothing, and `create_vocoder_executor`
(`stages.py:1901-1911`) does not time it either.

### ASCII flow, buffered path with graphs

```
CosyVoice3Vocoder.decode_batch                       stages.py:1210
  |  make_flow_input per request                     stages.py:1387-1414
  |  PreparedFlowRequest.total_mel_frames = (P + G) * 2   stages.py:1221-1224
  v
adaptive_flow_requests_grouping                      stages.py:1075-1157
  |  (length sorted DP, gap <= 384, pad budget <= 25%)
  v
FunCosyVoice3Flow.inference(group)                   stages.py:711-721
  |
  +-> pack_flow_inputs                               stages.py:145-241
  |     token (B, maxTok) int32, token_mask, prompt_feat (B, maxPromptFrames, 80)
  |
  +-> generate_flow(streaming=False, finalize=True)  stages.py:522-662
  |     embed -> pre_lookahead -> repeat_interleave(2) -> (B, 80, T)
  |     mel_mask from total_mel_lengths              stages.py:602-609
  |     noisy_mel = rand_noise[:, :, :T].expand(B)   stages.py:615-620
  |     |
  |     +-> runner.run(...)                          stages.py:448-519
  |     |     bucket = ceil(T/16)*16; key (B, bucket)
  |     |     hit  -> pad, copy_, replay, slice [:T]
  |     |     miss -> None
  |     +-> solve_flow_euler eager fallback          stages.py:653-662
  |
  +-> split_generated_mels                           stages.py:665-684
        mel_i = generated[i, :, promptFrames_i : combinedTokens_i * 2]
  v
mel2wav_batch (HiFT, padded to longest, finalize=True)   stages.py:1423-1459
```

---

## 2. The streaming (causal) Flow path

### Call chain

```
FunCosyVoice3StreamingVocoderScheduler.run_step        streaming_vocoder.py:341-365
  |
  +-- B > 1 : _run_causal_hop_batch                    streaming_vocoder.py:367-407
  |     token_end = token_offset + hop + 3             streaming_vocoder.py:64-65, 373
  |     FlowBatchInput(token=tokens[:token_end], prompt_token, prompt_feat, embedding)
  |     CosyVoice3Vocoder.first_hop_batch              stages.py:1357-1370
  |       FunCosyVoice3Flow.inference_causal           stages.py:723-744
  |         pack_flow_inputs                           stages.py:145-241
  |         generate_flow(streaming=True, finalize=False)  stages.py:522-662  (always eager)
  |         split_generated_mels with lookahead stripped   stages.py:731-744
  |     mel[:, :, token_offset*2 :] -> hift_delta       streaming_vocoder.py:392-400
  |
  +-- B == 1 : _run_one_causal_hop                     streaming_vocoder.py:409-415
        _run_flow_hift -> token2wav_chunk              streaming_vocoder.py:450-472, stages.py:1307-1355
          native CosyVoice flow.inference(streaming=True, finalize=False)
          cosyvoice/flow/flow.py:369-414
```

### Shapes into the estimator

For one hop with generated-token offset O, hop H, lookahead L = 3 and prompt token length P
(already padded to a hop multiple, see below):

- target tokens in the window: `T_tok = O + H + L` (`streaming_vocoder.py:64-65`).
- combined tokens per row: `C = P + T_tok` (`stages.py:183-187`).
- non-finalize strips the lookahead: `pre_lookahead_layer(token_emb[:, :-3],
  context=token_emb[:, -3:])` (`stages.py:540-548` for equal lengths, `stages.py:553-570`
  for mixed lengths). Output length is `C - 3 = P + O + H` tokens; the upstream layer
  concatenates the context and convolves with kernel `L+1`, so the body length is preserved
  (`cosyvoice/transformer/upsample_encoder.py:82-103`).
- `repeat_interleave(token_mel_ratio=2)` then transpose (`stages.py:572-576`) gives the
  estimator condition `(B, 80, F)` with

```
F = 2 * (P + O + H)      mel frames per row
```

- `mel_mask` is `(B, 1, F)` built from `(C_i - 3) * 2` per row (`stages.py:587-609`).
- `prompt_mel` is `(B, 80, F)` with each row's own prompt feat scattered into
  `[:, :, :2P_i]` and zeros after (`stages.py:610-614`).
- `noisy_mel` is `rand_noise[:, :, :F]` expanded across the batch, so every row in the batch
  shares the same noise slice (`stages.py:615-620`,
  `cosyvoice/flow/flow_matching.py:200` where `rand_noise = randn([1, 80, 15000])`).
- inside `solve_flow_euler` every one of these is doubled on the batch axis for CFG, so the
  DiT sees `(2B, 80, F)` per step (`stages.py:255-271`).

Layout per row, frame axis:

```
0                      2P                         2(P+O)              2(P+O+H)
|---- prompt frames ----|---- already emitted -----|---- new frames ----|
|<---------------- recomputed from scratch every hop ----------------->|
```

`split_generated_mels` (`stages.py:665-684`) drops `[0, 2P)` and the scheduler then drops
`[2P, 2(P+O))` with `mel[:, :, token_offset * 2:]` (`streaming_vocoder.py:392, 395`), so only
`2H` new frames survive. The entire prefix is recomputed on every hop.

### Mixed prompt lengths in one packed call

`pack_flow_inputs` (`stages.py:145-241`) left-aligns `[prompt | generated]` per row into a
`(B, maxCombinedTokens)` int32 tensor, zero-fills the tail, and builds `token_mask` from each
row's own combined length (`stages.py:202-215`); the embedding is multiplied by that mask
(`stages.py:534`) so pad tokens contribute zero. `prompt_feat` is padded to the longest prompt
(`stages.py:217-224`).

When combined lengths differ, `generate_flow` cannot take the fast `[:, -3:]` lookahead slice
(it would read pad on shorter rows), so it runs `pre_lookahead_layer` once per row in a Python
loop and pads each result to `max_body` (`stages.py:549-570`). That is B separate small conv
pairs per hop, plus a concat. The scheduler groups by `(token_offset, hop_len)` only
(`streaming_vocoder.py:80-82, 319-334`), so rows with different reference-clip lengths do land
in the same batch and do take the loop branch.

### The causal chunk mask

`DiT.forward` (`cosyvoice/flow/DiT/dit.py:145-176`):

```
streaming=True : add_optional_chunk_mask(x, mask.bool(), False, False, 0, static_chunk_size=50, -1)
                 -> subsequent_chunk_mask(L, 50) & masks -> (B, L, L), unsqueeze(1)
streaming=False: add_optional_chunk_mask(x, mask.bool(), False, False, 0, 0, -1)
                 -> masks (B, 1, L) -> .repeat(1, L, 1) -> (B, L, L), unsqueeze(1)
```

`subsequent_chunk_mask` (`cosyvoice/utils/mask.py:127-158`) is block causal with full left
context: query `i` may attend to every `j < (floor(i / 50) + 1) * 50`. The `num_left_chunks`
argument is ignored by this implementation. So the visibility of query `i` depends only on
`i` and the chunk size, never on the sequence length.

The omni patch `_chunk_mask` (`stages.py:1871-1896`, installed onto
`cosyvoice_dit.add_optional_chunk_mask` at `stages.py:1896`, only when
`enable_flow_cuda_graph` is true) short-circuits exactly one case: `use_dynamic_chunk` false
AND `static_chunk_size == 0`, that is the `streaming=False` call. It returns `masks` after an
in-place `masked_fill_` of all-false rows, which replaces upstream's
`(chunk_masks.sum(dim=-1) == 0).sum().item()` device-to-host sync and `print`
(`cosyvoice/utils/mask.py:233-235`). That sync and that data dependent branch are what would
make a capture impossible, so the patch exists for the graph path.

Consequence worth recording: the `streaming=True` branch of `_chunk_mask` delegates straight
back to upstream `add_optional_chunk_mask` (`stages.py:1882-1891`), so every streaming DiT
forward still performs a `.item()` host sync and materialises a (B, L, L) bool mask. With 10
Euler steps and CFG that is 10 syncs and 10 masks of (2B, L, L) bools per hop.

### Is right padding the frame axis safe under the streaming mask?

Yes, on the mechanics above:

- Query visibility `j < (floor(i/50) + 1) * 50` does not depend on the padded length, so
  adding pad frames does not change any valid query's allowed column set.
- The `& masks` term keeps padded columns out of every valid query's attention.
- Only conv over time inside the DiT is `CausalConvPositionEmbedding`, left padded and
  therefore strictly causal (`cosyvoice/flow/DiT/modules.py:129-144`).
- The post-attention output mask uses `mask[:, 0, -1]` (`cosyvoice/flow/DiT/modules.py:404`),
  the last query row's visibility, which under the block-causal mask is always a superset of
  row validity (its chunk end is at or beyond the padded length), so it zeroes pad positions
  and nothing else.
- Rows whose mask is entirely false (a pad query row beyond its length) are forced to all-true
  and produce garbage that is dropped by the length slice.

Adding dummy rows is likewise safe: every op in the estimator is per row (attention masks are
per row, layernorm and convs are per row, the CFG split is per row), and `pack_flow_inputs`
plus `generate_flow` derive every length from the row's own entry. The only coupling is the
shared frame axis: a longer dummy row lengthens `F` for everybody and thus costs real compute.

Note that the same causality argument is what makes the recompute-from-zero design correct:
because chunk boundaries sit on a fixed 50-frame grid and the noise slice prefix is identical
across hops, the prefix of the solve is reproduced exactly on every hop, which is why slicing
`[2(P+O):]` is legitimate. It is also why the prompt must be hop aligned.

### Prompt hop alignment

`pad_flow_prompt_to_hop` (`streaming.py:110-154`), called from `_latch_prompts`
(`streaming_vocoder.py:219`), repeats the last prompt token `pad` times and the last prompt
mel frame `2 * pad` times, where `pad = ceil(P_raw / 25) * 25 - P_raw` (`streaming.py:39-46`).
After padding, `P` is a multiple of 25 tokens, so `2P` is a multiple of 50 mel frames and the
first generated frame lands exactly on a DiT chunk boundary for every row in a packed batch.

### Frame counts for the hop ladder

Hop growth: `next_stream_hop_len` doubles the hop up to `TOKEN_MAX_HOP_LEN = 100`
(`streaming.py:66-81`), called after each hop (`streaming_vocoder.py:156-161, 402, 414`),
with `disable_hop_growth=False` in the shipped config (`config.py:149`). The offset advances
by the hop actually used (`streaming_vocoder.py:401, 413`), so the ladder 25, 50, 100, 100
gives offsets 0, 25, 75, 175.

Per hop, per row:

- window tokens `T_tok = O + H + 3`
- body tokens `P + O + H`
- estimator frames `F = 2 * (P + O + H)`, and the DiT runs `2B` rows of that
- new mel frames emitted `2H`
- new audio samples `2H * 480`

| hop k | O | H | T_tok | F, P=25 | F, P=50 | F, P=75 | F, P=100 | new mel | new audio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 25 | 28 | 100 | 150 | 200 | 250 | 50 | 1.00 s |
| 2 | 25 | 50 | 78 | 200 | 250 | 300 | 350 | 100 | 2.00 s |
| 3 | 75 | 100 | 178 | 400 | 450 | 500 | 550 | 200 | 4.00 s |
| 4 | 175 | 100 | 278 | 600 | 650 | 700 | 750 | 200 | 4.00 s |
| 5 | 275 | 100 | 378 | 800 | 850 | 900 | 950 | 200 | 4.00 s |

(`P` is the padded prompt token count, so `2P` prompt frames: 50, 100, 150, 200 frames for the
four columns. A 4 s reference clip is 100 tokens, the P=100 column.)

The frames column is exactly `2P + 2(O + H)`, and `O + H` is the cumulative generated-token
count, so the per-hop Flow cost grows linearly in the utterance length and the total Flow work
over a stream is quadratic. None of these shapes can hit a captured graph anyway (streaming
hops are eager, `stages.py:630`), and the shapes with P=100 at hops 3 and 4 (550 and 750
frames) would not all be in the capture list even if they could.

---

## 3. The Euler solve

`solve_flow_euler` (`stages.py:244-315`), and the upstream twin
`ConditionalCFM.solve_euler` (`cosyvoice/flow/flow_matching.py:71-124`) used by the B=1 and
leftover paths.

- Steps: `time_span = linspace(0, 1, 11)` (`stages.py:621-623`), so `len(time_span) - 1 = 10`
  Euler steps; the loop is `for step in range(1, len(time_span))` (`stages.py:273`). Upstream
  hardcodes `n_timesteps=10` at the call site (`cosyvoice/flow/flow.py:409`), which also
  yields an 11-point span. With `t_scheduler == "cosine"` the span is warped by
  `1 - cos(t * pi / 2)` (`stages.py:625-626`, `cosyvoice/flow/flow_matching.py:224-226`).
- CFG doubling: the six per-step buffers are allocated at `2 * batch_size` rows
  (`stages.py:257-271`). Rows `[0, B)` carry the real condition, rows `[B, 2B)` keep
  `token_condition`, `speaker_embedding` and `prompt_mel` at zero, and only `noisy_mel` and
  `mel_mask` are copied into both halves (`stages.py:274-281`). The update is
  `x <- x + dt * ((1 + cfg) * cond - cfg * uncond)` (`stages.py:304-311`) with
  `cfg = decoder.inference_cfg_rate` from the checkpoint cfm_params (0.7 in the class default
  at `cosyvoice/flow/flow.py:39`; UNCONFIRMED for the pinned checkpoint yaml).
  So the estimator always runs at `2B` rows, 10 times per Flow call.
- `rand_noise`: a fixed `randn([1, 80, 50 * 300])` = 15000 frames created once at CFM
  construction under `set_all_random_seed(0)`
  (`cosyvoice/flow/flow_matching.py:199-200`). Both `generate_flow` (`stages.py:615-620`) and
  `capture_inputs` (`stages.py:368-373`) take `[:, :, :F]` and `expand(B, -1, -1).clone()`,
  so all rows share one noise prefix and the noise for a given frame index never changes
  between hops. `generate_flow` raises if `F > 15000` (`stages.py:581-585`).
- Per step the only values that actually change are `noisy_mel_cfg` (both halves, rewritten
  from the running `noisy_mel`) and `flow_time` (`stages.py:274-275, 279`). The other four
  copies re-write constant data every step. `dt` and `t` are Python-visible tensor scalars
  updated on device (`stages.py:272, 312-314`).
- For a captured streaming solve the static inputs would be the same six tensors that
  `capture_inputs` builds, plus the `streaming` flag baked in at capture time (it selects the
  mask branch in `DiT.forward`, `cosyvoice/flow/DiT/dit.py:163-166`), which means separate
  graphs per (B, T, streaming). The blocker on that branch today is that the streaming mask
  path still goes through upstream `add_optional_chunk_mask` with its `.item()` sync and
  data-dependent `print` (`cosyvoice/utils/mask.py:233-235`), which the omni `_chunk_mask`
  patch bypasses only for `static_chunk_size == 0` (`stages.py:1881-1894`).

---

## 4. HiFT

### What runs per call

`CosyVoice3Vocoder.hift_delta` (`stages.py:1372-1385`):

```
hift_delta(tts_mel_new, hift_mel=accumulated, speech_offset=samples_emitted, finalize)
  tts_mel = cat([hift_mel, tts_mel_new], dim=2)        stages.py:1380-1381
  tts_speech, _ = self.hift.inference(tts_mel, finalize)  stages.py:1382
  delta = tts_speech[:, speech_offset:].cpu()          stages.py:1383-1384
  returns (delta, tts_mel, tts_speech.shape[1])        stages.py:1385
```

The caller stores the returned accumulated mel and the new total sample count back into the
stream state (`streaming_vocoder.py:395-404` for the batch path, `streaming_vocoder.py:459-471`
for B=1). So `speech_offset` is purely a bookkeeping cursor into the freshly recomputed
waveform: it is the length of the waveform HiFT produced on the previous call, and the delta
is whatever the new call produced beyond it.

`CausalHiFTGenerator.inference` (`cosyvoice/hifigan/generator.py:713-726`) has no cache
arguments and no cache state. Grep over the class shows the only stateful attributes are
`stft_window` (a constant Hann window buffer, `generator.py:668`) and module weights; there is
no `cache`, no running STFT buffer, no source cache. (The non-causal `HiFTGenerator.inference`
at `generator.py:557-569` does take a `cache_source`, but CosyVoice3 uses the causal subclass,
which is the one whose signature `inference(speech_feat, finalize)` matches every omni call
site: `stages.py:1382`, `stages.py:1433`, `stages.py:1448`, and the MPS adapter at
`stages.py:96-115`.)

Therefore every call recomputes, over the full accumulated mel:

1. `f0_predictor` in float64 over all A frames (`generator.py:716-717`,
   `cosyvoice/hifigan/f0_predictor.py:95-102`).
2. `f0_upsamp` by 480 and the NSF source generator over all `A * 480` samples
   (`generator.py:719-721`).
3. `_stft` of the source signal (`generator.py:491-497`) over the whole accumulated signal.
4. `conv_pre`, the `CausalConv1dUpsample` stages with their resblocks, the source
   downsample and fusion branches, `conv_post` (`generator.py:672-707`).
5. `_istft` over the whole accumulated signal (`generator.py:499-505`).

So the cost of hop k is proportional to the accumulated mel length `A_k`, not to the new
frames. With the standard ladder, `A_k = 2 * (O_k + H_k)`: 50, 150, 350, 550, 750 frames for
hops 1..5, i.e. 1 s, 3 s, 7 s, 11 s, 15 s of audio re-vocoded each time. Total HiFT work over
an N-hop stream is the sum of those, quadratic in the utterance length, and it is paid on top
of the equally quadratic Flow recompute.

Non-finalize trims (`finalize=False` on every hop, `streaming_vocoder.py:400, 411`):

- the f0 branch holds back `condnet[0].causal_padding` = 3 mel frames
  (kernel 4, `causal_type='right'`, `f0_predictor.py:72-74`; formula at
  `cosyvoice/transformer/convolution.py:172` gives 3), and `inference` slices the mel by that
  amount before `decode` (`generator.py:725`);
- `decode` holds back `conv_pre_look_right` = 4 more mel frames and trims
  `prod(upsample_rates) * istft hop_len = 480` samples off the tail
  (`generator.py:672-710`, the `finalize is False` branches at `generator.py:677-679` and
  `generator.py:708-709`).

So a non-final call emits roughly `(A - 7) * 480 - 480` samples and the next call re-emits
them identically before the new tail; `speech_offset` is what cuts the repeat away. The
`finalize=True` leftover call (`streaming_vocoder.py:437-444`) is the only one that flushes
those held frames.

### samples per mel frame

`hift_samples_per_mel_frame` is derived lazily in `mel2wav_batch`
(`stages.py:1450-1454`):

```
stride = int(self.hift.istft_params["hop_len"])
for rate in self.hift.upsample_rates:
    stride *= int(rate)
```

That is `hop_len * prod(upsample_rates)`. For the 24 kHz CosyVoice3 HiFT that is
`4 * 8 * 5 * 3 = 480` (upsample rates from the omni MLX config mirror,
`mlx/vocoder/config.py:71-75`; the torch class defaults at `generator.py:586-588` are the
22.05 kHz `[8, 8]` with `4 * 64 = 256`, so the value really does come from the checkpoint
yaml, UNCONFIRMED directly). 480 samples per mel frame is independently consistent with
24000 Hz / 50 mel frames per second and with the mel hop_size 480 used to extract prompt
features (`utils.py:146-149`), and with the upstream self-test that slices `i * 480`
(`generator.py:741-745`).

Two mechanical notes: `hift_samples_per_mel_frame` is only ever computed on the multi-mel
branch of `mel2wav_batch` (the single-mel branch returns at `stages.py:1431-1434` before the
derivation), and it is not used by `hift_delta` at all, which relies on the returned waveform
length instead.

### ASCII flow, streaming hop

```
run_step                                   streaming_vocoder.py:341
  select_step_participants (same (offset, hop) key)   streaming_vocoder.py:319-334
  |
  +--B>1--> first_hop_batch -> inference_causal -> generate_flow(streaming=True, finalize=False)
  |            stages.py:1357-1370, 723-744, 522-662      [EAGER, 10 steps, 2B rows, F frames]
  |         mel_i (1, 80, 2(O+H))  -> slice [:, :, 2O:]   streaming_vocoder.py:392-395
  |
  +--B=1--> token2wav_chunk -> native flow.inference       stages.py:1307-1355
  |            cosyvoice/flow/flow.py:369-414 -> CausalConditionalCFM.forward
  |            tts_mel[:, :, 2O:]                          stages.py:1352
  v
hift_delta                                  stages.py:1372-1385
  cat(accumulated_mel, new_mel)  -> A frames
  hift.inference(A frames, finalize=False)   generator.py:713-726   [FULL RECOMPUTE]
  delta = wav[:, speech_offset:]
  state.hift_mel = A frames ; state.speech_offset = len(wav)   streaming_vocoder.py:395-404
```

---

## 5. The non-streaming batch path, for contrast

`decode_payloads` is the base-class entry (`sglang_omni/scheduling/vocoder_base.py:34-46`):
`prepare_item` per payload, one `decode_batch`, then `store_result` per payload.

Admission: `create_vocoder_executor` passes `request_cost_fn=vocoder.flow_scheduler_cost` and
`max_batch_cost=flow_batch_admission_frames` (`stages.py:1923-1931`, default 8000 frames from
`config.py:135` and `stages.py:54`). `flow_scheduler_cost` (`stages.py:1416-1421`) returns
`(prompt_tokens + generated_tokens) * 2`, that is the total mel frames including the prompt
region, and it pays a full `prepare_item` plus `make_flow_input` per queued message. The
scheduler's batch assembly (`sglang_omni/scheduling/simple_scheduler.py:106-142`) adds
messages while `batch_cost + msg_cost <= max_batch_cost` and `len(batch) < max_batch_size`
(16, `config.py:141`), pushing the first over-budget message back to the pending deque. The
first message is admitted unconditionally, so a request larger than 8000 frames runs as a
singleton.

Grouping: `decode_batch` (`stages.py:1210-1277`) runs `adaptive_flow_requests_grouping`
(`stages.py:1075-1157`), an lru_cached DP over the length-sorted requests. It tries
group counts 1, 2, 3, ... and returns the first partition whose padded workload
`sum over groups of (group size * longest in group)` exceeds the unpadded workload by no more
than `flow_merge_pad_budget_percent` (25.0, `config.py:137`), with every group's internal
length gap bounded by `flow_merge_max_gap_frames` (384, `config.py:136`).

Each group is then one `FunCosyVoice3Flow.inference` call under autocast
(`stages.py:1236-1243`), so the graph key is `(len(group), ceil(2 * maxCombinedTokens / 16) * 16)`.
Given the default capture list, only groups of exactly 1 or 2 rows can hit, and only when the
group's padded frame count lands on one of the captured buckets. Because the DP minimises
padding rather than steering toward captured buckets, a group size of 3 or more, or a bucket
outside 304..640 (or the missing 624), silently takes the eager fallback at
`stages.py:653-662`.

After Flow, HiFT is regrouped by a different rule: greedy over the mels sorted by length,
starting a new group whenever `longest * (n + 1) > hift_max_padding_waste * total`
with `hift_max_padding_waste = 1.5` (`stages.py:1244-1269`, default at `stages.py:1167`),
then `mel2wav_batch` pads each group to the longest mel and runs one `finalize=True`
HiFT call, slicing each row back to `length * 480` samples (`stages.py:1423-1459`).

```
messages -> simple_scheduler._collect_batch          simple_scheduler.py:106-142
              cost = (P + G) * 2 frames, budget 8000  stages.py:1416-1421
  v
decode_payloads -> decode_batch                      vocoder_base.py:34-46, stages.py:1210
  v
adaptive_flow_requests_grouping (DP, gap 384, pad 25%)   stages.py:1075-1157
  v   per group
FunCosyVoice3Flow.inference -> generate_flow(finalize=True)   stages.py:711-721, 522-662
  v   graph key (B, ceil(F/16)*16); B in {1, 2} only per config.py:19-46
runner.run or eager fallback                         stages.py:448-519, 653-662
  v
HiFT regroup by padding waste 1.5 -> mel2wav_batch   stages.py:1244-1269, 1423-1459
```

---

## UNCONFIRMED list

1. The `cosyvoice` package is not installed on this machine; all upstream anchors are to the
   partial vendored snapshot under
   `tasks/cosyvoice_utilization_20260912/inputs/external_sources/CosyVoice/`, which carries no
   commit or tag, so its identity with the serving venv copy is unverified.
2. `cosyvoice3.yaml` is absent from the snapshot. DiT `static_chunk_size = 50`,
   `inference_cfg_rate = 0.7`, HiFT `upsample_rates = [8, 5, 3]` and `istft hop_len = 4` come
   from class defaults and the omni MLX config mirror, not from the pinned checkpoint.
3. CUDA graph capture wall time and peak capture memory: nothing logs them
   (`stages.py:395-436`, `stages.py:1901-1911`). Only the static input plus output footprint
   is computable (about 18.4 MB across the 26 default shapes).
4. The exact emitted sample count of a non-final HiFT call is stated as the approximation
   `(A - 7) * 480 - 480`; the code never computes it in closed form, it uses the produced
   waveform length, and the reflection pad plus istft framing were not traced to the sample.
