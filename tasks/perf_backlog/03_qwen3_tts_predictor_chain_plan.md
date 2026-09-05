# 03. Qwen3-TTS predictor chain: fewer kernels in the replay, a shorter tail (T22)

Status: S1 validated on H200 and H100 against the #1947 head (section
12), branch `perf/qwen3-tts-predictor-chain` at `2c00eb688` with upstream
main `91e9c3095` merged, owed the rerun on that base before the PR.
`perf/qwen3-tts-profiling` at `7898e244e` carries the ledger, the same
series, the allocator snapshot hook and the merge. The slices after S1
are re planned in `04_qwen3_tts_decode_step_slices_plan.md` from the
verified mechanics, which supersedes sections 3 to 7 of this doc for S2
onward. Evidence: doc 02 sections 6 and 8 (the H100 census and timeline
of 2026-09-04), and the code at `perf/qwen3-tts-predictor-startup-capture`
head `fff86552c`.

## 1. Requirement

Cut the decode step of the Qwen3-TTS talker without changing the audio
of any request beyond the variation batching already produces, on H100,
in the default non streaming serving mode. The replay is one
`cudaGraphLaunch` of 1371 kernels whose time is the sum of kernel
durations plus a 0.45 us gap per kernel, and the host tail after it is
35 to 45 serial launches for 60 us of device work. Both are counted
items, not estimated ones.

## 2. The three targets and their sizes

Per step, at 1 row unless noted, from doc 02:

| Target | Today | Reachable | How |
| --- | ---: | ---: | --- |
| A. kernels in the predictor replay | 1371 kernels, 4.74 ms wall | about 860 kernels, about 3.7 ms | A1 to A4 below, omni code and one Triton kernel |
| B. host tail after the replay | 1.10 ms (1.40 at 16 rows) | about 0.7 ms | B1, B2 below, omni code |
| C. the GEMMs of the replay | 432 calls, 2.07 ms, 40% of bandwidth | sized after A | a GEMV path for M up to 16 |

The reachable column for A adds the rows of section 3. C is not sized
here: its floor is 0.75 ms of weight traffic per replay, and whether a
custom path beats cuBLAS at M 1 to 16 is a measurement.

## 3. Target A, the replay

Each removed kernel saves its duration plus the gap. Counts are per replay
of 16 sub-steps.

- A1 The sampler prologue. `_sample_subtalker_token_seeded` selects
  temperatures, top k, top p and seeds with four `index_select` on
  `row_indices`, which since the eager and graph paths were unified is
  always the identity `arange[:batch]` (sglang_model.py, the call in
  `_sample_subtalker_token`), clamps the temperatures, and computes the
  sub positions with three elementwise kernels per sub-step. Change: drop
  the `row_indices` argument, read the four parameters as slices of the
  staged tensors (views, no kernel), clamp the temperatures once where
  `prepare_decode_buffers` stages them, and compute the sub position
  table `[num_code_groups, batch]` once per predictor call and slice it
  per sub-step. Removes 8 kernels per sub-step, 128 per replay, about
  0.20 ms. Bit identical by construction: the same values reach the
  sampler kernel.
- A2 All rows sample. `_sample_subtalker_token` always computes the
  argmax and the `where` so that mixed batches get argmax rows. When every
  row samples, which the default checkpoint does, both are dead. Change:
  a fifth signature term, whether any row is argmax, so the graph of the
  default signature has no argmax and no `where`, and a mixed batch keeps
  today's graph under its own key. Removes 2 kernels per sub-step, 32 per
  replay, 0.09 ms at 1 row and 0.22 ms at 16 (the argmax is the kernel
  that grows with rows). Bit identical: the selected token is the sampled
  one in both cases.
- A3 One kernel for qk norm, rope and the two cache writes. Today per
  layer: `sglang::fused_qknorm_warp`, `sglang::fused_rope_kernel`, and
  two elementwise kernels writing k and v into the predictor's private
  cache. One Triton kernel that normalises q and k, applies rope, and
  writes k and v into the cache slot removes 3 kernels per layer, 240 per
  replay, about 0.30 ms. Numerics: the norm and rope arithmetic stays in
  fp32 as sglang's kernels do it, but the reduction order of the norm
  changes, so this is gated by G2.
- A4 The residual add into the next RMSNorm. The MLP output's residual
  add is a separate elementwise kernel before the next layer's input
  norm. sglang's `fused_add_rmsnorm` does both. Removes 1 kernel per
  layer, 80 per replay, about 0.10 ms. Same gate as A3.
- A5 The split K reduce of the down projection: 80 kernels, 0.13 ms. A
  different cuBLAS algorithm or a fused epilogue removes the reduce but
  may lengthen the main GEMM. Measured with the GEMV work of target C.
- A6 The sampler kernel itself, 20.6 us per call for a 2048 wide row, 15
  per replay, 0.30 ms. A second version of the kernel is its own item,
  sized by a micro benchmark against the reference.

## 4. Target B, the tail

- B1 `_write_feedback_buffers` (model_runner.py) runs a Python loop over
  rows that launches one kernel per row and then stacks. Change: take the
  next decode input embeds of all rows as one gather from the staged
  feedback queue into the embedding rows, one launch per step. Removes
  `rows` launches, 0.19 ms at 16 rows, more at 64. Bit identical: the
  same rows land in the same slots.
- B2 Withdrawn. `prepare_decode_buffers` already keys its staging by
  the ordered request ids and a per request epoch and returns early when
  the batch is unchanged (sglang_model.py, the rids check at the top of
  the function), so the six uploads do not run at steady state. The
  copies in the tail are sglang's forward batch build and graph input
  copies, not ours.
- B3 Measure the graph launch cost without the profiler: the ledger
  records the host duration of the backbone replay call and of
  `graph.replay` per step. If a 1371 node launch costs more than the
  0.1 ms a host can hide while the device drains the eager sampling, the
  node count of target A is also a host item, and the ledger shows the
  gain per slice.
- B4 Overlapping the next step's batch build with the replay is sglang's
  overlap schedule, which this model disables by default. It is not part
  of this plan: the Qwen3-Omni talker's attempt was held on a TTFA
  regression (doc 05 section 7.4), and B1 to B3 first show how much tail
  is left to hide.

## 5. Proof

- G1 Bit identity for S1: the existing predictor graph tests (every
  bucket, startup ladder against eager) plus the full corpus c1 A/B with
  1088 of 1088 WAVs equal to the base. The A2 term has a signature test
  for the mixed batch key.
- G2 Numerics band for the fusions: the reference band is the variation
  the current batching produces (doc 24 section 3, two boots of one
  revision at c16, and one reference encoded alone against in a batch).
  A fusion passes when its c1 output differs from the base by no more
  than that band on the same rows, and its WER and similarity sit inside
  the run to run band of doc 24 section 3.2, on the four server c16
  replication of doc 24.
- G3 The census diff: `perfkit.py diff` between the base census and the
  slice's census at 1 and 16 rows, kernels per replay down by the counts
  of section 3, busy down by at least the durations removed. The timeline
  at 16 rows for B1 and B2, the per row run of launches gone.
- G4 The A/B of doc 15, full corpus, c1 and c16, fresh servers, arms
  alternated, `--seed 1234`, the same protocol as the startup capture.

## 6. Slices

- S0 B3 on the profiling branch, one census run. Decides whether the
  launch cost enters the accounting. Box time only.
- S1 A1, A2, B1. Omni Python only, no new kernel, expected bit
  identical. About 0.3 ms off the replay and 0.2 ms off the tail at 16
  rows, 6% of the step. Its census diff also fixes the base for S2. The
  exact design is section 8.
- S2 A3, one Triton kernel in `predictor_kernels.py`, G2 gate.
- S3 A4 with `fused_add_rmsnorm`, G2 gate.
- S4 A5, A6 and target C, each behind its own micro benchmark.

Branch: `perf/qwen3-tts-predictor-chain` from the startup capture head,
rebased onto main when #1947 merges. Measurement runs on
`perf/qwen3-tts-profiling` with the chain branch merged in.

## 7. Open until measured

- B3, the unprofiled launch cost.
- The G2 band, produced by the S1 base run.
- Whether A5 pays, and the GEMV floor of target C.

## 8. Exact design of S1

Read against `perf/qwen3-tts-predictor-startup-capture` head `fff86552c`.

### 8.1 A1, the sampler prologue

Today, `_sample_subtalker_token` (sglang_model.py) builds
`row_indices = self._sub_identity_row_indices_tensor[:batch_size]` and
`_sample_subtalker_token_seeded(logits, layer_idx, row_indices=..., semantic_positions=...)`
runs, per sub-step:

```
temperatures = self._sub_temperature_tensor.index_select(0, row_indices).clamp_min(1e-5)
top_ks       = self._sub_top_k_tensor.index_select(0, row_indices)
top_ps       = self._sub_top_p_tensor.index_select(0, row_indices)
seeds        = self._sub_sampling_seed_tensor.index_select(0, row_indices)
sub_positions = semantic_positions * (num_code_groups - 1) + layer_idx + 1
```

Eight kernels whose only effect is copying the first `batch_size` rows and
adding two constants, because `row_indices` is always the identity since
the eager and graph paths were unified.

Target:

- `_sample_subtalker_token_seeded(self, logits, layer_idx, *, sub_positions)`.
  The four parameters become slices: `self._sub_temperature_tensor[:batch]`
  and so on, views of the staged tensors, no kernel. `row_indices`,
  `_sub_identity_row_indices_tensor` and its construction in `__init__`
  go away.
- The clamp moves to staging: `prepare_decode_buffers` stages
  `max(subtalker_temperature, 1e-5)` in the host list it already builds.
  Same value, no kernel.
- `_code_predictor_forward_incremental` computes the sub position table
  once per call, `positions_table = semantic_positions[None, :] * (num_code_groups - 1) + self._predictor_sub_offsets[:, None]`
  with `_predictor_sub_offsets = arange(1, num_code_groups)` allocated in
  `__init__`, and hands `positions_table[layer_idx]` to the sampler for
  sub-step `layer_idx`. One kernel per call instead of three per sub-step.
- `_sample_subtalker_token(logits, layer_idx, *, sub_positions)` passes it
  through. `_select_semantic_positions` keeps its checks, applied once on
  the table's source.

Removed per replay: 4 index_select, 1 clamp, 3 elementwise per sub-step,
128 kernels, minus the one table kernel. Tests: the sampling kernel tests
build the talker through `_build_sampling_talker` and call the seeded
sampler with `row_indices`, they switch to the `sub_positions` argument
with the same values, computed by the same rule in the test. The graph
bit identity tests are unchanged and prove the graph path.

### 8.2 A2, all rows sample

Today, `_sample_subtalker_token` returns
`torch.where(self._sub_do_sample_tensor[:batch], sampled, argmax)` whenever
any row samples, so a batch where every row samples still runs the
argmax reduce and the select on every sub-step.

Target:

- `prepare_decode_buffers` sets `self._sub_has_argmax_rows = len(sample_rows) < batch_size`
  next to `_sub_has_sampled_rows`.
- `_sample_subtalker_token`: no sampled rows, argmax as today. Sampled
  rows and no argmax rows, return the sampled tokens. Both, the `where`
  as today.
- `_predictor_graph_signature` appends the term, so the key of a batch
  where every row samples is `("sampled", max_top_k, has_top_p, has_unbounded, False)`
  and a mixed batch is captured under its own key with the `where`.
  `capture_predictor_graphs` derives `False` for the fifth term, the
  builder passes the same three values as today.

Removed per replay: the argmax and the `where`, 32 kernels, 0.09 ms at 1
row and 0.22 ms at 16. Bit identical for the default signature: with
every row sampling the `where` selected the sampled token. Tests: the
signature rule test gains the mixed case, `test_mixed_sampled_argmax_rows_use_graph_bit_identity`
and `test_mixed_padded_bucket_bit_identity_and_reuse` keep the mixed
path honest, and the graph key count for a mixed batch after an all
sampled batch is asserted.

### 8.3 B1, the feedback write

Today, `Qwen3TTSModelRunner._write_feedback_buffers` (model_runner.py)
loops over the rows, and for each calls the Omni talker helper
`_take_next_decode_input_embed`, which reads the row's pending feedback
(a row view of the previous step's `embeds_snap`), reads the next text
row from `pending_text_queue`, and returns `feedback + text`: one add
kernel per row, then one `torch.stack` into the feedback embedding
rows and one copy of the row ids. At 16 rows that is the run of 16
launches 12 us apart in the timeline, 0.19 ms, and it grows with the
batch.

Target, in `Qwen3TTSModelRunner` only, the Omni talker helpers untouched:

```
feedback_rows, text_rows, eager_rows = [], [], []
for row_idx, sched_req in enumerate(requests):
    data = sched_req.data
    feedback = peek(data.pending_feedback_queue)
    text = peek(data.pending_text_queue) or (data.tts_pad_embed if data.thinker_chunks_done else None)
    if feedback is None or text is None:
        eager_rows.append(row_idx)          # today's per row path for this row
        continue
    feedback_rows.append(feedback)
    text_rows.append(text)
    pop both queues
weight = decode_feedback_embedding.weight[:batch_size]
torch.stack(feedback_rows, out=weight[batched_rows])   # one cat kernel
weight[batched_rows].add_(torch.stack(text_rows))      # one cat, one add
history rows appended as views of one clone of weight[:batch_size]
```

Four kernels per step for the batched rows instead of one per row plus
the stack. The rows without feedback (the first decode step of a request)
or without text keep today's path, so a first step after prefill is
unchanged. `torch.stack` of up to 64 equal rows is one kernel.
`decode_input_embeds` is appended today and never read on the Qwen3-TTS
path (only cleared in the scheduler), so the history keeps views of the
step's clone. Bit identical: the same add on the same rows. Tests:
`test_pipeline.py` has the feedback buffer cases. They gain a case where
half the rows have no feedback yet and the rest are batched, asserting
the buffer rows equal the per row computation.

### 8.4 What is not in S1

B3, the ledger field for the host duration of `graph.replay` and of the
backbone replay, lands on the profiling branch first and is read from
the next census. A3, A4 and the GEMM path wait for S1's census diff as
their base.

## 9. Tracker and open work checked on 2026-09-04

- #1754, the Qwen3-TTS roadmap. T-PR18, "profile and optimize remaining
  GPU kernels after the architecture level work lands", lists the code
  predictor attention, QK norm plus RoPE fusion, RMSNorm and SwiGLU as
  candidates and asks for end to end profiling before any model specific
  kernel. Doc 02 is that profiling. Its first PR is #1794, the vocoder
  SnakeBeta fusion, merged. No PR exists for the predictor items.
- T-PR17, #1790, opt-in FP8 for the talker and predictor MLPs, touches
  the GEMM half of the replay. It is blocked: nine c16 runs produced 14
  max token runaways, isolated to the predictor graph path, and #1418
  lists it under evaluated no go. Target C stays BF16 and is sized after
  S1.
- #1418, the 1.7B single instance issue, attributes the gap to host side
  overhead and closed the admission and queue path (#1413, #1449, #1462,
  #1649) and the predictor sampling path (#1641, #1726). The per step
  tail of doc 02 section 8 is not covered by any of them.
- #1792, MOSS-TTS, captures the feedback embeddings inside the sampling
  graph. Same problem shape as B1 on a different model. B1 batches the
  eager path instead, because the Qwen3-TTS feedback needs the next text
  row that the host owns.
- #1779, starting the torch profiler with `with_stack` deadlocks the
  engine stage at c16. The census tool does not need stacks, which is why
  the tail is attributed from the code rather than from frames.
- #1936, fault 2 (illegal instruction under load with the predictor graph
  off) stays open and unrelated.

## 10. Implementation map of S1

Anchors are `perf/qwen3-tts-predictor-startup-capture` head `fff86552c`.
The package is `sglang_omni/models/qwen3_tts/`: `sglang_model.py` (1872
lines, the talker and the predictor), `model_runner.py` (370, the decode
hooks), `predictor_kernels.py` (143, the Triton embedding gather),
`sampling_kernels.py` (576, the Triton samplers), `engine_builder.py`. S1
touches the first two and their tests. No new file, no new class.

| File | Class, function, line | Change |
| --- | --- | --- |
| `sglang_model.py` | `Qwen3TTSTalker.__init__`, :949 | remove `_sub_identity_row_indices_tensor`. Add `_predictor_sub_offsets = torch.arange(1, num_code_groups, device, long)`. Add `_sub_has_argmax_rows = False` next to `_sub_has_sampled_rows` (:952). |
| `sglang_model.py` | `prepare_decode_buffers`, :976 | stage `max(subtalker_temperature, 1e-5)` in the host list (the row loop near :1020). Set `_sub_has_argmax_rows = len(sample_rows) < batch_size` next to `_sub_has_sampled_rows` (:1047). |
| `sglang_model.py` | `_predictor_graph_signature`, :1176 | the sampled tuple gains `bool(self._sub_has_argmax_rows)` as its fifth element. |
| `sglang_model.py` | `_predictor_graph_capture_state`, :1203 | save and restore `_sub_has_argmax_rows`, set it from `signature[4]` for the capture, as the other four fields are set from the signature at :1214. |
| `sglang_model.py` | `capture_predictor_graphs`, :1246 | the derived signature appends `False`. |
| `sglang_model.py` | `_code_predictor_forward_incremental`, :1426 | after the positions are validated once, `positions_table = semantic_positions[None, :] * (num_code_groups - 1) + self._predictor_sub_offsets[:, None]`. Each sub-step passes `positions_table[layer_idx]` where it passes `semantic_positions` today. |
| `sglang_model.py` | `_sample_subtalker_token`, :1546 | signature `(logits, layer_idx, *, sub_positions)`. Drop the `row_indices` slice and `_select_semantic_positions`. Return the sampled tokens when `not self._sub_has_argmax_rows`, keep the `where` otherwise. |
| `sglang_model.py` | `_select_semantic_positions`, :1581 | its checks move to the single validation of `semantic_positions` in `_code_predictor_forward_incremental`, the function goes. |
| `sglang_model.py` | `_sample_subtalker_token_seeded`, :1594 | signature `(logits, layer_idx, *, sub_positions)`. `temperatures = self._sub_temperature_tensor[:batch]`, same for `top_ks`, `top_ps`, `seeds`, no `index_select`, no `clamp_min`, no position arithmetic. The fused call and the ATen fallback take the slices as they took the copies. |
| `model_runner.py` | `Qwen3TTSModelRunner._write_feedback_buffers`, :288 | the batched path of section 8.3. Rows with a feedback row and a text row are stacked into `decode_feedback_embedding.weight[:batch]` and the text rows added in one call. Rows without one keep the per row helper. The history rows are views of one clone of the written rows, because the retract re-prefill reads that history (`test_retract_prefill.py`). |
| `engine_builder.py` | `setup_model_resources` | unchanged. |
| `tests/unit_test/qwen3_tts/test_sampling_kernels.py` | `_build_sampling_talker`, `_production_seeded_tokens`, `_reference_seeded_tokens`, `_fused_seeded_tokens` | the helpers compute `sub_positions` with the plan's rule from the same `semantic_positions` and pass it instead of `row_indices`. Every existing case keeps its expected tokens. |
| `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py` | `test_signature_rule_is_shared_by_batch_and_startup_paths` | the expected tuple has five terms, plus a mixed case (one argmax row) that expects `True`. |
| `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py` | `test_mixed_sampled_argmax_rows_use_graph_bit_identity`, `test_mixed_padded_bucket_bit_identity_and_reuse` | unchanged, they prove the `where` path. One assertion added: an all sampled batch and a mixed batch of the same bucket hold two keys. |
| `tests/unit_test/qwen3_tts/test_pipeline.py` | the feedback cases at :4904 to :4956 | one case with half the rows lacking a feedback row, asserting the written rows equal the per row computation and the queues are popped only for the batched rows. |
| `tests/unit_test/qwen3_tts/test_retract_prefill.py` | `test_write_feedback_buffers_records_decode_input_history` | unchanged, it is the guard for the history clone. |

Commits, in order, each compiling and tested on its own:

1. stage the clamped subtalker temperature
2. compute the predictor sub positions once per call
3. read the sampler parameters as slices of the staged rows
4. skip the argmax when every row samples
5. batch the feedback rows into the decode embedding

Expected census diff after 5, per replay at 1 row: kernels 1371 to about
1213, busy down by about 0.25 ms. Timeline at 16 rows: the run of 16
launches after the predictor replaced by 4.

S0 in parallel, on `perf/qwen3-tts-profiling` only: `StepLedger` gains
`host_launch_ms` from `time.perf_counter` around the backbone replay call
in `model_runner/base.py` `_prepare_and_forward` and around
`graph.replay` in `_predictor_forward_graphed`, reported per shape in the
summary like `forward_ms`.

## 11. S1 as committed and reviewed, and what to run

Commits on `perf/qwen3-tts-predictor-chain`, from `fff86552c`, head
`1d047d541`, pushed:

1. `1071895f3` stage the clamped subtalker temperature
2. `c77a3e740` compute the predictor sub positions once per call
3. `54c43ca9f` read the sampler parameters as slices of the staged rows
4. `d10706cea` skip the argmax when every row samples
5. `fd16b0e05` batch the feedback rows into the decode embedding
6. `8f83a1216` call the seeded sampler with its sub positions in the rank tests
7. `09b61f772` capture the mixed batch signature at startup
8. `2188e5776` extract the next decode input peek and pop from the talker runner
9. `901e7acef` index the batched feedback rows through the shared decode input policy
10. `60ad96b90` compact the decode input history into its own storage every 64 rows
11. `552937d6c` name the seed positions with the sampling state and tidy the capture state order
12. `5a0087b73` drop the decode input history compaction
13. `1d047d541` restore the predictor graph key budget

Commits 1 to 5 are the section 10 map. Commits 6 to 11 are the review of
that series, one commit per finding. Commits 12 and 13 remove the two
numbers that review had introduced without a derivation:

- Two rank tests in `test_predictor_cuda_graph.py` still called the seeded
  sampler with `row_indices`, which no longer exists. They pass the sub
  positions now (6).
- A batch with greedy rows carries the fifth signature term, so its graph
  would have been captured lazily inside serving on the first mixed batch.
  Startup captures both sampled variants. The second set shares the cuDNN
  plans of the first, its cost is the capture time the startup log prints
  (7). The key budget stays at its prior value of 32 (13). The mixed graphs
  count against it exactly as they would have when captured lazily, so the
  room for run time signatures is unchanged: at the default ladder of six
  buckets the startup set takes 12 keys, at 64 rows and twelve buckets it
  takes 24.
- The batched feedback write held its own copy of the rule that decides
  whether a row reads a staged feedback row plus a text row or embeds its
  token id. The rule lives in `QwenTalkerModelRunner._peek_next_decode_inputs`
  and `_pop_next_decode_inputs`, and the per row helper is written on them
  (8, 9).
- Each history row is a view of its step's batch wide clone, so a clone
  lives until the last request of that step's batch drops it. A compaction
  every 64 rows was added (10) and dropped again (12): the interval was a
  chosen number that no measurement pins, and the post decode already
  keeps the output codes the same way. The bound without compaction comes
  from the repo defaults, max_new_tokens 2048 and max_running_requests 16:
  one request that lives its full 2048 steps while the rest of the batch
  churns pins 2048 times batch rows of hidden times 2 bytes, 128 MB at
  16 rows and hidden 2048, 512 MB at 64 rows. Requests that finish at
  normal lengths pin only their own rows for long. The principled fix, if
  that bound ever binds, is to rebuild the history from the output codes at
  retract time, since the feedback row is a function of the codes and the
  text pad. That reworks the retract seam shared with Qwen3-Omni and is not
  S1.
- The seed offsets and positions carry the `_sub_` prefix of the sampling
  state they belong to, `sub_positions` is a required keyword of the
  sampler, the saved capture state is in signature order, and the dead
  test scaffolding is gone (11).

After 13 the series adds no chosen number. The two literals it touches
predate it: the 1e-5 temperature floor moved from a kernel on the device
to the host staging with the same value, and the greedy row staging of
temperature 1.0, top_p 1.0 and top_k 1 is unchanged.

Deviations from section 10 that stand: the sampler takes no `layer_idx`,
the sub position row carries it. The argmax signature is
`("argmax", 0, False, False, False)` so both tuples have five terms. The
test that required positions on the direct sampler call went with
`_select_semantic_positions`.

Three questions raised at review, answered from the code:

- Why remove kernels before fusing. Every kernel S1 removes was dead work:
  the four `index_select` copies per sub-step read parameters that a slice
  of the staged buffer already holds, the clamp repeated a bound the host
  staged, the argmax and the `where` ran for batches without a greedy row,
  and the per row feedback embedding ran one launch per row for one add
  per batch. Fusing dead work fuses nothing. The fused sampler of sglang
  is still called, the slices feed it. The fusions are S2 and S3, section
  8.4.
- Whether the slices are the right torch. A slice of a static buffer is a
  view, no kernel, and it is what sglang reads in its captured regions
  (`decode_cuda_graph_runner.py`, `cuda_graph_buffer_registry.py`, the
  eagle draft runner). The `index_select` copy was the wrong tool for a
  buffer that already holds the rows in batch order.
- Whether the feedback batching follows the flow. For Qwen3-TTS there is
  no thinker, so `thinker_chunks_done` is always set and a decode row is
  in one of two states: it has a staged feedback row (every step after the
  first) or it embeds its token id (the first decode after prefill, and
  after a retract re-prefill). The writer has one loop that sorts the rows
  into the two states, one batched stack and add, one clone, and one loop
  that appends the history. No loop nests in another.

On the box, in this order:

1. `pytest tests/unit_test/qwen3_tts -q` on `perf/qwen3-tts-predictor-chain`
   `1d047d541`.
2. The census on `perf/qwen3-tts-profiling` `93684aa0b` (the ledger
   commits plus the 13 above) with the doc 02 section 4 protocol, then
   `perfkit.py diff` against the 2026-09-04 census JSONs at 1 and 16 rows,
   and the two timelines. Expected: 1371 to about 1213 kernels per replay
   at 1 row, and the run of per row launches after the replay replaced by
   the batched write in the 16 row timeline.
3. The c1 A/B against `perf/qwen3-tts-predictor-startup-capture`
   `fff86552c` as arm A, 1088 of 1088 WAVs byte identical expected. The
   c16 A/B for the tail, which shows in the step wall time and not in the
   replay.
4. Peak allocator memory per arm of the c16 A/B, the check for the history
   bound above. The ledger does not report it yet, S0 adds
   `torch.cuda.max_memory_allocated` at profiler stop. Expected: arm B
   above arm A by no more than the live rows of the run, and a long
   request outliving its batch is the case that shows the bound.

## 12. S1 as measured, H200 and H100, 2026-09-04 and 2026-09-05

The shipping measurement is the new base run of 2026-09-05 on H100: A is
upstream main `91e9c3095`, B is `2c00eb688`, one fresh server per point,
profiling off for the full corpus, archive
`artifacts/qwen3-tts-chain-newbase-20260905-results.tar.gz`, PR body in
`pr_qwen3_tts_predictor_chain.md`. Its numbers: 1371 to 1222 kernels per
replay, replay wall 4.714 to 4.405 ms at 1 row and 5.258 to 4.706 at 16,
c1 byte identical 1088 of 1088, c1 median latency down 3.2% and QPS up
3.4%, c16 median down 4.0%, p95 down 6.3%, QPS up 3.8%, c16 errors 128
(A) and 119 (B) over 11943 words and similarity 71.32 and 71.20, both
inside the identical kernel spread of doc 24 section 3.2 (116 to 128
errors, 71.18 to 71.32), GPU peak at c16 81055 MiB (A) and 80735 MiB
(B), no lazy capture, fallback, retract or CUDA error in any log, 394
unit tests. The census names the layer's 14 kernels: FlashInfer
`RMSNormKernel`, the qkv GEMM, `fused_qknorm_warp`, `fused_rope_kernel`,
two `elementwise_kernel` cache writes (2.18 and 1.73 us at 1 row, 2.43
and 2.30 at 16), the cuDNN `sdpa_sm80_flash_fprop` kernel (flash for the
key length 1 sub-step), the o_proj addmm, the norm, gate_up, `act_and_mul`,
the split K down GEMM and its reduce, and the residual add
`vectorized_elementwise_kernel` (1.09 us at 1 row, 1.12 at 16).

The two earlier runs, both against the #1947 head `fff86552c` as arm A
and the series head as arm B with the doc 02 protocol for the census and
the doc 15 protocol for the A/B, agree with it:

- Census, H100: 1371 to 1222 kernels per replay at 1 row and at 16 rows,
  the 149 the section 10 map derives (60 index copies, 74 elementwise,
  15 argmax). Replay wall down 0.31 ms at 1 row and 0.55 ms at 16.
- A/B, H100: c1 byte identical, 1088 of 1088 WAVs. c1 median latency
  down 3.6% (QPS up 3.5%), c16 median down 2.1% (QPS up 1.7%). c16 p99
  up 0.9% and c16 WER up 0.025 points, both inside the doc 24 section 3
  band, and the filtered WER equal.
- Step composition at 16 rows on H100 in the profiled window: predictor
  replay 4.71 ms (53%), host gaps 2.75 ms (31%), backbone 2.00 ms,
  sampling and staging 0.13 ms.
- The tail. Section 4 attributed the per row feedback launches to most
  of the 1.1 to 1.4 ms tail. Measured, the batched write saved 0.03 to
  0.15 ms. The rest of the tail is the scheduler's host work between
  launches, the sglang forward batch build, the graph buffer fill, the
  result loop and the output streamer, none of which the series
  touches. Doc 04 section 4.3 and its E2 measurement take that over.
- Memory. Arm B's c16 peak was 724 MiB above arm A on H100 with one
  allocator retry in the first run and none in the second. The
  process wide allocator snapshot of 2026-09-05 (profiling branch hook,
  `perfkit.py snapshot`) attributes the largest allocation of both arms
  to the same cuDNN convolution of the vocoder's whole utterance decode
  at different batch sizes, 549 MiB and 765 MiB, with start and end
  allocated memory equal across arms. The series allocates only the 64
  KB history clone per step at 16 rows. The difference is which
  requests finish together, run to run, not the series. The exposure
  itself is doc 04 item M1.
