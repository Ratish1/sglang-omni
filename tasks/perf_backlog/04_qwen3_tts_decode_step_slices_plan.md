# 04. Qwen3-TTS decode step: the slices after S1 (T22 continued)

Status: plan, written 2026-09-05 from whole file reads and three verified
research reports. Base for every slice is upstream main `91e9c3095`, which
carries #1947 (the predictor startup capture) and the vocoder worker,
ramp, CustomVoice prefill graph and bootstrap frame changes that landed
after it. S1 sits on `perf/qwen3-tts-predictor-chain` at `2c00eb688`
(upstream main merged in, the same 13 series commits) and waits for its
H100 rerun on this base before its PR. Each later slice is its own branch
from upstream main, measured on its own, so a gain is attributable to one
change.

Evidence, all in this folder: `research/sglang_0518_decode_host_path.md`
(the sglang v0.5.18 decode iteration), `research/omni_qwen3_tts_step_mechanics_2c00eb688.md`
(the omni side of the same iteration), `research/sglang_0518_predictor_layer_kernels.md`
(the kernels behind one predictor layer and their numerical contracts).
Every claim in those files was checked by a second reader against the
pinned sources, and the corrections are appended to each file. Line
anchors below are at `2c00eb688` for omni and `v0.5.18` for sglang unless
named otherwise. Doc 02 holds the H100 census of 2026-09-04, doc 03 the
S1 design and its review.

## 1. Requirement

Cut the decode step of the Qwen3-TTS talker without regressing any
metric of the full corpus A/B: latency p50 and p99 at c1 and c16, QPS,
WER, speaker similarity, first audio latency when streaming, peak
allocated memory, startup time. A slice ships only when its A/B shows no
regression beyond the run to run band of doc 24 section 3 and its census
diff shows the kernels it claims to remove.

Rules that shape every slice:

- One change per PR, based on upstream main, validated by one server per
  arm on the same GPU with the model's benchmark on the full corpus at c1
  and c16. No stacking of unmerged slices.
- No constant that a measurement does not pin, and nothing tuned to one
  GPU. The code runs on local devices, H100, H200 and B200. Anything
  hardware or situation specific is env gated or lives in a kernel that
  exists for that hardware.
- Bit identity where it can be derived from the kernels. Where it cannot,
  a named experiment on the box decides before the slice is written, and
  the quality gate is the doc 24 band.
- Never patch sglang, sgl_kernel or torch. Use their kernels through
  their public entry points at omni owned seams.

## 2. Where the step goes today

From the H100 run of S1 on 2026-09-05 (profiled window, the profiler adds
about 1 ms of host time per step, doc 02 section 6):

| 16 rows, per step | ms | share |
| --- | ---: | ---: |
| predictor replay wall | 4.71 | 53% |
| host gaps, device idle | 2.75 | 31% |
| backbone replay | 2.00 | 22% |
| eager sampling and staging | 0.13 | 1% |

Kernels per predictor replay after S1: 1222 at 1 row and at 16 rows, from
1371, the exact 149 the S1 design removes. Replay wall down 0.31 ms at 1
row and 0.55 ms at 16. The c1 A/B was byte identical, 1088 of 1088 WAVs.

A middle sub-step after S1 has 76 kernels: 3 to enter (the code copy, the
fused codec embedding gather, the projection GEMM), 5 layers of 14
(RMSNorm, qkv GEMM, fused qk norm, fused rope, the k write, the v write,
the cuDNN attention, the o_proj GEMM with the residual in its epilogue,
RMSNorm, gate_up GEMM, act_and_mul, the down GEMM as split K and its
reduce, the residual add), and 3 to leave (final RMSNorm, head GEMM, the
fused seeded sampler). The census counted 16 per layer, the two unnamed
launches are inside the cuDNN attention call and are read from the new
census by kernel name.

The two targets of this plan are the replay, where each removed kernel
saves its duration plus a 0.45 us gap, and the host gaps, which are the
scheduler's own work between launches plus the one wait per step.

## 3. Mechanics established by the research

### 3.1 One predictor layer, today

`_predictor_forward_one_token` (sglang_model.py:1674-1710) runs the five
layers of the code predictor for one token per row. Per layer:

```
hidden                                  [B, 1, 1024] bf16
  input_layernorm(hidden)               RMSNorm, no residual     sgl_kernel rmsnorm, 1 kernel
  _predictor_cached_self_attention      :1768-1829
    qkv_proj GEMM                       :1782                    cuBLAS, 1
    split into q [B,2048] k [B,1024] v [B,1024]   views          0
    apply_qk_norm (in place)            :1786-1793               sglang::fused_qknorm_warp<128,...>, 1
    rotary_emb(positions, q, k)         :1794-1799               sglang::fused_rope_kernel<true,128,...>, 1
    reshape to [B,H,1,128] views        :1800-1802               0
    k cache write, v cache write        :1806-1807               2 strided copies
    scaled_dot_product_attention        :1812-1825               cuDNN, enable_gqa
  o_proj addmm into the residual        :1713-1766               cuBLAS epilogue, 1
  post_attention_layernorm(hidden)      RMSNorm, no residual     1
  mlp: gate_up GEMM, act_and_mul, down GEMM (split K + reduce)  4
  hidden = residual + mlp_out           :1706                    1 elementwise add
final: model.norm(hidden)               :1707-1709               1
```

The cache is `_predictor_k_cache[layers, max_batch, kv_heads, predictor_len, head_dim]`
(:905-914), so the write of one token is a strided copy into a slice of
the head major layout, one kernel for k and one for v. The attention
reads `cache[layer, :B, :, :L, :]`, a BHSD view whose batch stride
carries the full `predictor_len`.

The backbone layer (`Qwen3TTSTalkerDecoderLayer.forward`, :210-230)
does not add its residual by hand. It calls the norms in their residual
form, `input_layernorm(hidden, residual)` at :221 and
`post_attention_layernorm(hidden, residual)` at :228, and sglang's
`RMSNorm.forward_cuda` turns that into one `fused_add_rmsnorm` kernel
(layernorm.py:563-566). The predictor path is the only place in this
model that adds the residual as its own kernel.

### 3.2 The kernels and their numerical contracts

- RMSNorm (layernorm.py:471-567). With no residual, sgl_kernel `rmsnorm`
  (:567). With a residual, sgl_kernel `fused_add_rmsnorm` (:565), which
  mutates both tensors in place: the residual becomes the sum, the input
  becomes the normed output. Both wrappers hand the call to FlashInfer
  when it is importable, the dtype is fp16 or bf16 and Dynamo is not
  tracing (elementwise.py:115-120, :155-160), otherwise to the AOT op.
  The FlashInfer kernel bodies are fetched at build time and are not in
  the checkout, so whether the fused kernel normalizes the fp32 sum or
  the bf16 rounded sum is not derivable here. That is what E1 measures.
  Diversions from these two calls: batch invariant mode (deterministic
  inference) routes the residual form to `forward_native` and the plain
  form to a Triton kernel (:491-505), `variance_size_override` and the
  HF cast flag route elsewhere (:489, :520-558). None applies to the
  shipped configuration.
- QK norm. `fused_inplace_qknorm` (models/utils.py:495-502) for CUDA,
  head dim in {64, 128, 256, 512, 1024}, in place over the q and k views,
  fp32 sum of squares, butterfly warp reduce, `rsqrtf(sum / D + eps)`,
  one round to bf16 (qknorm.cuh:38, impl/norm.cuh:63-107).
- RoPE. The code predictor's config has `rope_scaling` null and
  `rope_theta` 1e6 (HF config.json of the 1.7B checkpoint), so `get_rope`
  builds the plain `RotaryEmbedding` with neox style. Its
  `forward_cuda` (rotary_embedding/base.py:363-385) takes an optional
  `fused_set_kv_buffer_arg` and passes it to
  `apply_rope_with_cos_sin_cache_inplace`. With the arg, the JIT kernel
  is `fused_rope_store_kernel` (rope.cuh:177) instead of
  `fused_rope_kernel` (:117). Both call the same `rope_rotate_head`
  (:56), fp32 rotation with one round to bf16 (:74-79), and the store
  variant writes the already rounded k vectors and the untouched v
  vectors into the cache rows (:81-87, :232-241). So the cache content
  of the store variant equals the in place k and v bit for bit. The
  fallback rope path for other head sizes asserts the arg is None
  (base.py:420-422).
- The store contract (rope.cuh:356-466, verified line by line): q
  `[N, Q, D]` and k, v `[N, K, D]` with head stride exactly D and unit
  last stride, `cos_sin_cache` fp32, positions and cache locations `[N]`
  of one shared integer dtype, k and v cache `[-1, R]` with the same row
  width and row stride, `R == num_kv_heads * D`. No scale, no bounds
  check on the locations. `FusedSetKVBufferArg` (rope.py:97-107) is a
  dataclass of four tensors and nothing else, so a private cache can be
  passed without any pool object. sglang's own gate for it
  (models/utils.py:284-305) is CUDA and a bf16 cache.
- Attention. torch's cuDNN dispatcher checks dtype, dense shapes, a unit
  last stride, GQA head counts and `check_cudnn_tensor_shapes`
  (sdp_utils.cpp:853-910, v2.13.0). It does not check the stride layout,
  `check_cudnn_layout` (:693-758) is only used by the experimental
  recompile avoidance switch. A slot major cache read through a
  transposed view is the BSHD layout that switch calls native.
- Activation, GEMM. `act_and_mul_kernel` (activation.cuh:65) computes
  silu times up in fp32 with one round. The GEMMs are `F.linear`
  (unquant.py:265) into cuBLASLt on sm90, and the split K choice on the
  down projection is cuBLAS's heuristic, not expressible from code.

### 3.3 The host path of one decode iteration

On the omni side (research file 2, execution order items 1 to 30) the
scheduler thread runs the synchronous loop `_event_loop_normal`. Per
iteration it makes 14 Python passes over the rows at steady state,
launches 5 device ops for the feedback write, calls the backbone replay
and the predictor replay, stages the sampled ids into pinned memory with
an event, and waits exactly once, `event.synchronize()` in
`_resolve_host_token_ids` (model_runner/base.py:185). Every other host
read of a device value is routed through that pinned tensor, including
sglang's `next_token_ids.tolist()` (batch_result_processor.py:934),
because `_make_batch_result` substitutes the host tensor
(omni_scheduler.py:1474-1476).

On the sglang side (research file 1) the same iteration has, with the
shipped configuration, one further class of work: the per row loops of
`filter_batch`, `prepare_for_decode`, `alloc_for_decode`,
`ForwardBatch.init_new`, the result loop with `update_finish_state`, and
the output streamer's five passes, plus the forward batch build, the
graph buffer fill (one `_foreach_copy_` per dtype pair), the attention
metadata refill, the sampler, the FutureMap scatter and one zmq send.
sglang's overlap scheduler is off for this model
(`disable_overlap_schedule: True`, engine_builder.py:89).

At finish, `apply_sglang_qwen3_tts_result` (request_builders.py:1491-1522)
stacks the request's codes and calls `.cpu()`, a blocking pageable copy
on the scheduler thread inside the decode loop (:1503-1506).

```
scheduler-tts_engine thread, one iteration, 16 rows
  recv, admin, admission bookkeeping (5 calls)          os.py:834-907, host only
  get_next_batch_to_run -> upstream prepare_for_decode   per row loops, no sync
  _emit_prefill_start_for_batch  [x16]                   os.py:1524-1541
  _build_sched_output            [x16]                   os.py:1412-1421
  ModelRunner.execute
    resolve_forward_inputs         device gather          overlap_utils.py:106-107
    before_decode
      prepare_decode_buffers       rid scan [x16], H2D only on a miss   sglang_model.py:977-1083
      _write_feedback_buffers      [x16 x3], 5 device ops              model_runner.py:288-348
    backbone replay                one graph launch
    sample                         sampler kernels
    post_decode -> code_predictor_forward   predictor replay, one graph launch
    _stage_token_ids               async D2H to pinned + event          base.py:121-134
    publish_next_tokens            FutureMap scatter
    _finalize
      event.synchronize()          THE WAIT                             base.py:185
      output processing  [x16], post_process_outputs [x16], 2 clones
  _emit_stream_output   [x16]     builder returns [] when not streaming  rb.py:1544-1550
  process_batch_result -> upstream result loop [x16], stream_output [x16]
    finished rows: stack + .cpu()  blocking pageable D2H                  rb.py:1503-1506
```

### 3.4 The vocoder and the memory exposure

In the default non streaming mode nothing reaches the vocoder per step.
The codes accumulate on the device in `data.output_codes` and go to the
vocoder once at finish. The vocoder stage collects up to 8 finished
requests within 2 ms (stages.py:223-224, streaming_simple_scheduler.py:63-64,
:249-308) and decodes the whole utterances in one batched call of the
speech tokenizer's decoder (streaming_vocoder.py:1865-1895), no CUDA
graph, no chunking. The activation therefore scales with the batch times
the longest utterance in it. The 2026-09-05 memory profile of both arms
showed the largest allocation as a cuDNN convolution buffer of that
decode, 549 MiB in one arm and 765 MiB in the other, with equal start and
end allocated memory and zero out of memory events. The shape of the
earlier retry, 5 rows by 512 channels by 71680 samples in fp32, is about
3 s of audio per row, so a 30 s non streaming request in a batch of 8
makes that one buffer about ten times larger. This is a property of the
non streaming decode, not of any slice here, and it is item M1 below.

The streaming path is bounded by the chunk window: the initial collector
takes up to 32 rows (2 ms), the follow up collectors up to 8 (1 ms), the
decode graphs cover batch 1, 2, 4 and 8, and any group above 8 rows runs
eagerly (streaming_vocoder.py:379-396, :1190-1192, :1623-1633).

## 4. Slices

| Slice | Change | Kernels per replay | Expected per step | Numerics |
| --- | --- | ---: | ---: | --- |
| S1 | sampler prologue, all rows sample, batched feedback | 1371 to 1222 | 0.31 ms at 1 row, 0.55 ms at 16 | bit identical, proven |
| S2 | rope writes k and v into a slot major cache | 1222 to 1062 | about 0.35 ms | bit identical by the kernel contract, E0 proves it |
| S3 | the residual add inside the next norm | 1062 to 982 | about 0.15 ms | E1 decides, else the doc 24 band |
| S4 | host tail: measure, then remove the dead passes | 0 | sized by E2 | bit identical |
| S5 | measured experiments: small M GEMMs, sampler kernel, attention layout | 0 to 160 | unknown | each behind a micro benchmark |

The expected numbers use the doc 02 census durations: the two cache
writes are 1.9 and 2.1 us plus two gaps per layer instance, 80 instances
per replay, and the residual add is one elementwise kernel of 1.7 us plus
a gap, 80 per replay. They are estimates of the busy time removed, the
census diff of section 6 is the measurement.

### 4.1 S2, rope writes the cache

Design. Change the private cache to slot major,
`[layers, max_batch, predictor_len, kv_heads, head_dim]`, and let
sglang's rope kernel store k and v into it through
`fused_set_kv_buffer_arg`. The attention reads the same memory through a
transposed view. No kernel of ours, no new numerics.

```
before                                              after
_predictor_k_cache [L, B, Hkv, P, D]                _predictor_k_cache [L, B, P, Hkv, D]
write: cache[l, :B, :, t:t+1, :].copy_(k)  2 kernels   rope(positions, q, k, fused=(v, k_rows[l], v_rows[l], slots[t, :B]))  0 extra
read:  cache[l, :B, :, :t+1, :]  BHSD view             cache[l, :B, :t+1].transpose(1, 2)  [B, Hkv, t+1, D], last stride 1

k_rows[l] = _predictor_k_cache[l].view(B_max * P, Hkv * D)       one 2D view per layer, built in __init__
slots     = arange(B_max)[None, :] * P + arange(P)[:, None]        int64 [P, B_max], built in __init__
slots[t, :B] is a contiguous row slice, no kernel, same dtype as _predictor_position_rows (int64)
```

Contract checks against section 3.2: q and k are the split views of the
qkv row, head stride D and unit last stride, as today. v is the third
split view, `[B, Hkv, D]` after the wrapper's `view_as(k)`. The cache
rows are `[B_max * P, Hkv * D]` with row stride `Hkv * D`, so
`R == num_kv_heads * D` holds and k and v share one shape. Positions are
`_predictor_position_rows[t, :B]`, int64, and the slots are int64, so
the shared dtype check holds. Every slot index is below `B_max * P` by
construction, which matters because the kernel does not check.

Gate, evaluated once per attention object at first use and cached on the
talker, not per call: the tensors are CUDA, the cache dtype is bf16,
`attn.compatible_with_fused_kv_buffer` (thinker_model.py:225-227, false
only for the MRoPE class, which the predictor does not use), and
`attn.rotary_emb.use_fallback_kernel` is false (base.py:121-124, the
condition under which sglang itself passes the arg, and the fallback path
asserts the arg is None). When the gate is false the write is the copy
path into the same slot major cache, `cache[l, :B, t].copy_(k.view(B, Hkv, D))`,
which is a contiguous row copy per row and still one kernel each for k
and v. CPU, MPS and the unit test talker take that path. This is a
capability gate on the pinned dependency, not a tuning constant.

What stays: the SDPA call, `enable_gqa`, the graph capture, the
signature, the startup capture, `_predictor_o_proj_add_residual`, the
sampler. The cuDNN plans are keyed by strides, so the startup capture
builds the plans for the new layout in its warmups exactly as it does for
the old one, and no serving step pays for it.

Files and functions:

| File | Function | Change |
| --- | --- | --- |
| `sglang_model.py` | `Qwen3TTSTalker.__init__` :895-914 | cache shape slot major, the per layer 2D views, the slot table, the cached gate value `None` |
| `sglang_model.py` | `_predictor_cached_self_attention` :1768-1829 | build the arg, pass it to `attn.rotary_emb`, drop the two copies on the fused path, read the cache through the transposed view |
| `vendor/sglang/models.py` | exports | add `FusedSetKVBufferArg` from `sglang.kernels.ops.attention.rope` next to the two helpers it already re exports |
| `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py` | `_build_talker` :102-105 | the fixture cache in the new layout, every bit identity test unchanged |
| `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py` | new accelerator test | a real `RotaryEmbedding` for head dim 128 and a random `[B, Hkv, D]` k and v: the stored rows equal the copy path bit for bit and the in place q and k equal the plain rope call, batch 1 and 16, positions 0 to 16 |
| `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py` | new test | the gate takes the copy path for the fixture's identity rotary and the fused path when a rotary reports `use_fallback_kernel` false on CUDA |

Overlap with open PRs: #1907 edits the prefill graph break at
sglang_model.py:61-63 and :195-199 and the prefill positions in
model_runner.py, none of which S2 touches.

Proof: E0 on the box before the commit, then G1 (c1 byte identity), G3
(census diff, 160 fewer kernels per replay at 1 and 16 rows, the
`fused_rope_store_kernel` name in place of `fused_rope_kernel`, attention
kernel time unchanged or better), G4.

### 4.2 S3, the residual add inside the next norm

Design. Make the predictor layer call the norms the way the backbone
layer does, in their residual form, so the standalone add disappears
into `fused_add_rmsnorm`:

```
hidden = token_embeds                                   safe to mutate, as today
residual = None
for layer in layers:
    if residual is None:
        residual = hidden
        normed = layer.input_layernorm(hidden)           rmsnorm
    else:
        normed, residual = layer.input_layernorm(mlp_out, residual)   fused add + norm, in place
    attn = _predictor_cached_self_attention(normed, ...)
    residual = _predictor_o_proj_add_residual(o_proj, attn, residual)  addmm epilogue, as today
    normed = layer.post_attention_layernorm(residual)    rmsnorm
    mlp_out = layer.mlp(normed)
normed, _ = model.norm(mlp_out, residual)                fused add + final norm
```

Five adds per sub-step become part of the five norms that follow them,
80 kernels per replay. The o_proj residual keeps the addmm epilogue that
the c1 identity of S1 already covers.

Numerics. The residual value is the same bits either way: torch's bf16
add and the fused kernel both add in fp32 and round once. The normed
output is the same bits only if the fused kernel normalizes the rounded
sum, which the FlashInfer source would tell and which is not in the
checkout. E1 answers it on the box in minutes. If E1 shows identity, S3
is bit identical and G1 applies. If not, S3 is the first slice under the
doc 24 band, and the plan records the measured difference.

Deterministic inference is not a target, but the residual form under
batch invariant mode routes to `forward_native` (layernorm.py:491-497),
the same path the backbone takes there, so it does not regress that mode
beyond what the backbone already does.

Files and functions: `_predictor_forward_one_token` (sglang_model.py:1674-1710)
only, plus the graph bit identity tests, which need no change because
graph and eager run the same code. One new test asserts the layer count
of norm calls with a residual, so a later refactor cannot bring the add
back silently, on the fake talker whose RMSNorm is the real sglang class.

Proof: E1, then G1 or G2, G3 (80 fewer kernels, the elementwise family
down by 80 per replay, the norm family unchanged in count), G4.

### 4.3 S4, the host tail

The tail after the predictor replay is 1.1 ms at 1 row and 1.4 ms at 16
(doc 02 section 8) and the host gaps are 31% of the step. S1 removed the
per row feedback launches from it and measured 0.03 to 0.15 ms, so the
rest is scheduler work whose split nobody has measured. S4 therefore
starts with E2 and removes what E2 names. The candidates from the
mechanics, each bit identical:

- The finish path copy: `torch.cat(...).cpu()` per finished request on
  the scheduler thread (request_builders.py:1503-1506). It waits for the
  device and then copies through pageable memory. The same pinned
  staging that `_stage_token_ids` uses (base.py:121-180) with the event
  waited on the vocoder side would take it off the critical path. Its
  share is one copy per finished request, so at c16 with short
  utterances it lands on about every second step.
- `_emit_prefill_start_for_batch` (omni_scheduler.py:1524-1541) and
  `_emit_stream_output` (:1423-1440) walk all rows every step, the first
  to find rows that already emitted, the second to call a builder that
  returns an empty list for every non streaming request. A per batch
  done count and a per request streaming flag on the data end both walks
  at steady state.
- `prepare_decode_buffers` builds a 16 tuple list every step to compare
  against the cached one (sglang_model.py:987-999). A batch epoch that
  the scheduler bumps when the running set changes replaces the scan.
- The `generation_steps` loop and the two comprehensions in `_finalize`
  (base.py:529-543) and the `RequestOutput` allocations
  (output_processor.py:53-61) are small and shared across models, so they
  move only if E2 shows them.
- The two clones in `post_process_outputs` (model_runner.py:253-254) are
  device work, not host time, and stay.

E2 is the measurement, section 5. S4 becomes one PR of the omni side
removals that E2 ranks above the noise floor, with the finish path copy
as its own commit because it changes a payload contract inside the
process.

Out of scope for S4, as its own plan: the overlap of result processing
with the next forward. sglang's overlap loop exists (research file 1,
section on the overlap scheduler) and omni has its own `enable_overlap`
flag, both off for this model. The Qwen3-Omni attempt was held on a
measured first audio regression (doc 05 section 7.4). After S4 the
remaining tail is the number that says whether that plan is worth
writing, and #1809 (block on the inbox when idle) is orthogonal, it
changes the idle path, not the busy one.

### 4.4 S5, measured experiments

Each of these is a measurement with a go or no go, not a design:

- The down projection's split K reduce, 80 kernels and 0.13 ms per
  replay. cuBLAS chooses it. A micro benchmark of the five predictor
  GEMM shapes at M 1 to 64 with `torch.mm`, `torch.addmm` and a Triton
  GEMV decides whether any path beats cuBLAS at these sizes. The floor is
  the weight traffic, 0.75 ms per replay against 2.07 ms today.
- The sampler kernel, 20.6 us per call for a 2048 wide row, 15 per
  replay. A second version is its own kernel project, measured against
  the reference bit for bit through the existing parity tests.
- The attention layout. S2's census diff shows the cuDNN kernel time on
  the BSHD view against today's BHSD view for free. If it moved, that is
  a finding for T29.
- The unprofiled graph launch cost, B3 of doc 03: the ledger's `host_ms`
  is the launch call wall of the backbone. E2 adds the predictor replay
  call wall. If a 1000 node launch costs the host more than the device
  hides, the node count is a host item too and the S2 and S3 gains
  include it.

### 4.5 M1, the non streaming vocoder decode memory

Not a slice of this series, recorded here because the memory profile of
S1 found it. The non streaming vocoder decodes whole utterances in
batches of up to 8 with no bound on the batch's total audio. On 80 GB
with the 0.85 fraction the transient headroom is about 8 GB. The
measurement that sizes it: the allocator snapshot route of the profiling
branch on a run with long inputs (30 s of text) at c16, read with
`perfkit.py snapshot`, which lists the live blocks by frame. The
candidate mechanisms, decided after that number: bound the batch by
total frames instead of rows, or route non streaming requests through
the chunked decoder the streaming path already owns. Both are their own
plan.

## 5. Experiments that gate the slices

Each runs on the box in minutes, before the slice's code is written, and
its result is recorded in this doc.

- E0, the rope store equals the copy path. On the H100, for head dim
  128, 16 query and 8 kv heads, batch 1 and 16, positions 0 to 16: build
  sglang's `RotaryEmbedding` through `get_rope`, random bf16 q, k, v,
  run the plain rope on clones and copy k and v into a slot major cache,
  run the rope with `FusedSetKVBufferArg` into a second cache, compare q,
  k and both caches with `torch.equal`. Expected: equal. This becomes
  the accelerator test of S2.
- E1, the fused add norm against the add and the norm. Random bf16 x and
  residual, hidden 1024, N in {1, 2, 4, 8, 16, 32, 64}, 1000 trials:
  compare `rmsnorm(residual + x)` (today's predictor path) with
  `RMSNorm.forward_cuda(x, residual)` (the fused path) bit for bit on
  both outputs. Also compare against the in repo JIT
  `fused_add_rmsnorm(..., cast_x_before_out_mul=False)`, which is known
  to normalize the rounded sum. Expected outcomes: the FlashInfer kernel
  is bit identical, then S3 is G1. Or only the JIT kernel is, then S3
  calls the JIT kernel directly at the omni seam (it takes plain tensors,
  research file 3 section B) and is G1. Or neither, then S3 is G2 with
  the measured max difference recorded.
- E2, the host phase breakdown. On the profiling branch, the ledger
  gains `perf_counter` marks at the omni hook boundaries: launch begin,
  before_decode end, backbone launch end (exists as `host_ms`), post
  decode predictor call begin and end, staging end, the wait (exists as
  `wait_ms`), output processing end, stream output end, process batch
  result end, and the finish path copy when it runs. One c16 window of
  200 steps without the torch profiler. Output: a per phase p50 and p90
  table at 1 and 16 rows. This ranks the S4 candidates and gives B3.

## 6. Proof, per slice

- G1 Bit identity: the predictor graph tests on the box, then the full
  corpus c1 A/B with every WAV byte identical to the base, 1088 of 1088.
- G2 Band: when G1 is not derivable, the c1 output differs from the base
  by no more than the doc 24 section 3 batch composition class, and WER
  and similarity at c16 sit inside the run to run band of doc 24
  section 3.2.
- G3 Census: `perfkit.py diff` of the base and the slice at 1 and 16
  rows, kernels per replay down by the slice's count, the removed family
  down by that count, no family up, replay busy down by at least the
  removed durations. The timeline for S4, the tail in ms.
- G4 The A/B: one server per arm on the same GPU, arms alternated, the
  seed-tts-eval corpus at c1 and c16 with `--seed 1234`, generate only,
  then transcribe and similarity, speed tables, and the peak memory
  sampled by `nvidia-smi` once a second per arm. Every metric of section
  1 reported, none worse beyond the band.
- Unit tests: contract and edge case tests only, the fakes model the real
  shapes, no mock and count.

## 7. Order and branches

1. S1 PR after its H100 rerun on the new base (doc 03 section 11 for
   what to run, the commands are in the 2026-09-05 protocol).
2. E0, E1, E2 in one box session on the profiling branch.
3. S2 on `perf/qwen3-tts-predictor-rope-store` from upstream main.
4. S3 on `perf/qwen3-tts-predictor-fused-residual` from upstream main.
   S2 and S3 touch different functions and can be written in parallel,
   they are measured separately.
5. S4 on `perf/qwen3-tts-decode-host-tail` from upstream main, after E2.
6. S5 items as their measurements come in. M1 as its own plan.

Measurement runs use `perf/qwen3-tts-profiling` with the slice merged
in, and the census against the slice's own base run on the same day.

## 8. Open until measured

- E0, E1, E2 results.
- The two unnamed kernels per layer inside the cuDNN attention call,
  from the new census's kernel names.
- The kernel name of the predictor's RMSNorm in the census. If it is
  `_rms_norm_kernel`, the engine ran under batch invariant mode and the
  census run must be repeated with the shipped configuration.
- The M1 number.
