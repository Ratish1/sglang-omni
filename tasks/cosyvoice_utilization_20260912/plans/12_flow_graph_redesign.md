# Plan 2: Flow CUDA graphs with a derived shape key

Replaces roadmap rows 2.1 and 2.2, and moves row 3.3 in front of both. Supersedes
`../slices/22_flow_graph_shape_key.md`, which stands as the measurement record.

Trees read for this plan, in full where stated: pinned SGLang `v0.5.19`
(`0bcd8223`), upstream omni main `27a8293c` plus slice 1.1, the vendored
CosyVoice at `5eed0099` and the vendored x-transformers beside it,
`sgl-project/sglang-omni#1861` and `#2141`. Every claim below carries a
file:line; the measurements carry the run that produced them.

## 1. What is wrong, and what is not

The defect is the shape key and the capture granularity. It is **not** that the
graphs exist: on buffered c8 they are worth 1.67x (`../slices/22_flow_graph_shape_key.md`),
the largest win measured anywhere in this task, and the solver goes from 19,072
launches to 77 (stage 1 ledger, `../ROADMAP_20260915.md` row "buffered, graph").
Graphs stay on throughout.

Three measured facts frame everything:

| fact | where from |
|---|---|
| a captured shape costs **130 MiB**, linear in shape count, and torch sees under 5 MiB of it | `../MEMORY_TRACE_20260917.md`, sweep at 1/8/16/24/32/40/54 shapes |
| coverage is **35.7 percent** buffered and **0 percent** streaming, on the corpus the table was traced from | `stage2/flow_shapes.py` over instrumented runs |
| the value of a graph falls as the call grows: most of a 1-row call, **6.6 percent** at 16 rows x 512 frames | stage 1 ledger, busy/wall 0.41 against 0.92 |

Linearity is the important one: it proves the 130 MiB is driver side graph exec
memory, not capture pool growth, so SGLang's largest-first capture order
(`decode_cuda_graph_runner.py:1062-1064`, "Capture the large shapes first so that
the smaller shapes can reuse the memory pool") is irrelevant to it. The cost is
paid per instantiated graph, and our 54 graphs are topologically identical,
differing only in shapes.

## 2. The rules, taken from SGLang's source rather than its output

| rule | SGLang | ours today |
|---|---|---|
| one shape knob per phase | `ShapeKey.size`, "prefill: num_tokens, decode: bs" (`runner/shape_key.py:26-28`) | two, `(batch_size, mel_frames)` (`config.py:19-75`) |
| round up, never exact | `_pad_to_bucket` = `bisect_left` (`runner/base_cuda_graph_runner.py:136-150`); "No exact-shape check: load_batch bucket-pads" (`prefill_cuda_graph_runner.py:1190-1191`) | exact membership; #1861: "Batch size is never padded" |
| bound the padding waste, fall back to eager beyond it | "A replay executes every padded token in its capture bucket." `_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2` (`prefill_cuda_graph_runner.py:148-151`, guard at `:1192-1194`) | no such guard; a miss is silent eager |
| bake upper bounds into launch config, refresh contents per replay | "max_seq_len_q / max_seq_len_k are baked at capture as upper bounds ... the kernel reads real work extents from the cu_seqlens / cache_seqlens device buffers" (`flashattention_backend.py:613-615`, `:699-702`) | n/a, the padded layout has no ragged metadata |
| pack ragged work flat, express structure as prefix sums | no `[N, max_len]` layout anywhere on the prefill path; `extend_start_loc` / `cu_seqlens_q` are the packing (`prefill_cuda_graph_runner.py:1286-1288`) | `PackedRows` already does this, except in attention |
| a second knob becomes a small variant set, not a second key dimension | `_CHUNKED_PREFIX_VARIANTS = (1, 2, 4, 8, 16)` with `ShapeKey.variant_label` (`prefill_cuda_graph_runner.py:152-154`, `shape_key.py:29-35`) | n/a |

One rule we must **not** take from SGLang, because it has no evidence behind it:
`full_prefill_max_req` auto-derives as `max(chunked_prefill_size // 512, 1)`
(`model_runner_components/cuda_graph_setup.py:386`) and nothing in that tree
justifies the 512.

And one belief of mine that the source refuted: SGLang **does** capture
multi-step loops in one graph. The EAGLE draft runner captures `draft_forward`
(`speculative/eagle_draft_cuda_graph_runner.py:472`), which is
`for i in range(self.speculative_num_steps)` with a full model forward and draft
sampling per iteration (`speculative/eagle_worker_v2.py:633, 671`). So capturing
our ten step solver is not wrong by precedent, and the granularity question in
section 5 is a measurement, not an argument.

## 3. Why the graph belongs on the packed path, not the padded one

`generate_flow` reaches the runner only when the call is neither streaming nor a
chunk (`stages.py:656`), and hops and stream finals both leave through
`generate_flow_packed` (`stages.py:797, 814-816`). So the padded path the graph
sits on today serves buffered traffic only. Three further reasons, all read:

- **The packed path already hoists the step invariant work the padded path
  repeats.** `solve_flow_euler_packed` builds the attention object once before
  the loop (`packed_dit.py:207`), while the padded path rebuilds the whole
  `(B, 1, L, L)` chunk mask inside `DiT.forward` on every one of the ten steps
  (`dit.py:163-166`). At 32 CFG rows and 640 frames that mask is about 13 MB,
  allocated and `.repeat`-materialised ten times per call.
- **Its per token cost is already total frames**, because every per token module
  runs on the packed sequence (`packed_dit.py:143-159`). Only attention still
  scatters back to a padded `(rows, width)` layout (`packed_dit.py:103-109`).
- **It is already the flat, prefix sum layout SGLang's rule asks for**:
  `PackedRows` carries `starts_host`, `row_ids` and `positions`
  (`packed_dit.py:15-47`).

## 4. The design

### 4.1 Ragged attention, which is what makes the key one dimensional

Replace `RowAttention`'s scatter, SDPA under a `(rows, width, width)` mask,
gather (`packed_dit.py:97-109`) with one varlen FA3 call over the packed
sequence, the chunk causal semantics expressed as the segment layout
`hop_layout` already produces (`flow_hop_cache.py:73-98`). After this the cost of
a Flow call is total frames in every module, so the graph key is one number.

This is the same move that lets SGLang key decode on batch size alone: varlen
attention over a page table moved sequence length out of the key and into device
buffers. The kernel is already in this tree and already qualified: E6 measured it
at 53.3 to 54.0 dB against a float32 SDPA truth on real activations, and G0
measured the two kernels as equally accurate, differing only in per sample
rounding (`../stage2/README.md`).

### 4.2 The bucket ladder, derived from the model and bounded by waste

Three quantities, all already in the tree, none measured on any card:

- **quantum**: the attention chunk, `static_chunk_size` 50 frames. Every hop
  length is a multiple of it, because the prompt is padded to a 25 token
  multiple and hops are 25, 50 or 100 tokens (`streaming.py:12-16`). Measured
  hops: 100, 250, 250.
- **ceiling**: `flow_batch_admission_frames` (`config.py:153`), the scheduler's
  own admission budget, under the model's 15,000 frame noise ceiling
  (`flow_matching.py:199-200`, checked at `stages.py:589-593`).
- **waste bound**: the padding a round up may add. The repo already expresses
  exactly this notion as `flow_merge_pad_budget_percent` (`config.py:155`).

The ladder is then geometric from the quantum to the ceiling with ratio
`1 / (1 - waste)`, snapped to the quantum: 17 buckets at 25 percent, 28 at 12.5
(`stage2/flow_shapes.py`). The count is an output. **Stream finals are not chunk
multiples** (measured: 106, 316, 312, 352, because a final covers the whole
history and the AR stops where it stops), which is exactly why the key rounds up
rather than matching.

Paired with SGLang's guard, which we take and tighten: a call whose round up
exceeds the waste bound runs eager rather than paying the FLOPs. Padded frames
are real work, not skipped rows.

### 4.3 What is baked and what is refreshed

Following `flashattention_backend.py:613-615` exactly:

| baked at capture | refreshed per replay |
|---|---|
| every tensor address | every tensor's contents |
| the bucket's total frames as an upper bound on the launch config | the real per row extents, through `cu_seqlens_q` and the segment `cache_seqlens` |
| the row count slot capacity | which rows are real and which are zero length |
| the `streaming` variant | nothing; it selects the graph |

Two hazards carried over verbatim from SGLang's experience:

- **Padded frames execute.** They must be benign: zeroed inputs, and no segment
  in `cu_seqlens_q` covering them, so the varlen kernel neither reads nor writes
  them (`cuda_graph_buffer_registry.py:824-829`).
- **Staleness, not arithmetic, is the real risk.** Any buffer the graph reads
  past the real extent must be cleared or provably unread, and a buffer that is a
  **scatter destination must be cleared even when its length is zero**, because a
  zero length does not protect a write whose address comes from another buffer
  (`prefill_cuda_graph_runner.py:1720-1726`). That applies directly to us: the
  cached hop's pool slot indices are a scatter destination for `set_kv_buffer`
  (`flow_hop_cache.py:157-167`).

### 4.4 The second knob

`streaming` selects the chunk causal mask against the full mask
(`dit.py:163-166`). It is a boolean, so it is a `variant_label` in SGLang's
sense, not a key dimension: the table is buckets x variants, as
`_CHUNKED_PREFIX_VARIANTS` multiplies the prefill table. Whether both variants
are needed after 4.1 is an open question, not an assumption: with ragged
attention the mask becomes segment metadata, which may make the captured
topology identical for both. V3 in section 7.

## 5. The granularity question, settled by measurement

Each captured graph holds the whole ten step solver, about 19,072 kernel nodes,
which is why a shape costs 130 MiB where SGLang budgets 8 MB per captured shape
(`arg_groups/memory_hook.py:321`). Capturing one DiT forward instead and
replaying it ten times from the Python loop would divide the node count, and the
arithmetic says the launch cost of doing so is negligible: ten replays against
the 19,072 launches the eager path pays, at the ledger's 7.3 microseconds per
launch.

I am not deciding this by analogy, because the analogy runs the other way: EAGLE
captures its multi step loop deliberately (section 2). It is decided by V1, which
measures both granularities with the probe that produced the 130 MiB figure.

What makes either granularity safe is already established by the read of one
Euler step: the timestep enters as `flow_time`, a one element CUDA tensor
allocated once and written in place (`stages.py:262, 277`); `t` and `dt` are
0-dim device tensors (`stages.py:270, 310-312`); there is no host sync, no
`.item()`, no data dependent branch and no RNG anywhere in the DiT forward once
`patch_chunk_mask` is installed (`stages.py:935-979`, which removed an `.item()`,
a host branch and a `nonzero`-based index assignment). The only Python state per
step is the loop counter.

## 6. Slices

| slice | content | gate |
|---|---|---|
| 2.0 | the shape instrument and the aggregator, on `analysis/cosyvoice-flow-shapes` and `stage2/flow_shapes.py`; ran 2026-09-17 | none, it is a measurement |
| 2.1 | ragged attention in the packed path | the G0 harness run on this tree and on main: the shipped hop's SNR against the same float32 padded truth must not fall by more than the G0 margin. Not bit identity, which two attention kernels cannot reach (`../stage2/README.md`, the 4090 isolation) |
| 2.2 | the derived bucket table over total frames, round up, waste guard, one table for buffered, finals and hops | replay bit identical to eager per bucket; coverage and padding reported from the instrument; memory census |
| 2.3 | capture granularity, whichever V1 selects | same as 2.2, plus the per shape cost |

Deleted by 2.2: `FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES`
(`config.py:19-75`), `verify_flow_cuda_graph_capture_shapes` and the exact match
lookup in `FlowCudaGraphRunner.run` (`stages.py:465-496`).

## 7. Validation tasks

| # | unknown | how it is settled |
|---|---|---|
| V1 | cost per captured shape at both granularities, whole solver against one forward | `stage2/vocoder_memory.py` with the capture region switched; the whole solver figure is already 130 MiB |
| V2 | does a graphed replay still pay off once attention is ragged, and from what call size | the busy/wall curve says the gain falls with size; measure the crossover and let it set the ceiling of the ladder |
| V3 | are two `streaming` variants needed after 4.1 | compare the captured topology of the two; if identical, one table |
| V4 | the waste bound | the instrument reports padding per bucket ladder; SGLang's guard allows 2x, the first proposal here is 25 percent |
| V5 | does FA3 varlen capture and replay correctly in this repo's wheel | capture one bucket and compare replay against eager before anything else in 2.2 |

## 8. Found by the trace, independent of this plan

Three defects the read turned up that cost time or memory on their own. None
belongs in this plan's slices; each is its own change.

- **The rope is rebuilt on every Euler step** from step invariant inputs
  (`dit.py:158`, `x_transformers.py:732-744`), as is the whole attention mask in
  the padded path. `PackedDiT._rope` is ours and has the same shape
  (`packed_dit.py:165-170`).
- **Five of the seven per step buffer writes copy values that never change**
  (`stages.py:274-279`): `mel_mask_cfg` twice, and the conditional halves of
  `token_condition_cfg`, `speaker_embedding_cfg` and `prompt_mel_cfg`.
- **The Flow weights are float32 under a bfloat16 autocast**
  (`stages.py:2054`, 332.3M parameters, 1,267 MiB measured). The stage 1 ledger
  attributes **19 percent of a hop call's launches to autocast weight casts**, so
  this is not only 635 MiB, it is roughly 3,600 launches per call and the same
  fraction of every captured graph's node count. It is the one item here that
  changes a number, so it needs the G0 protocol.
