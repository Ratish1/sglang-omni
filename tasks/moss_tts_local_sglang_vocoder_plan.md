# MOSS-TTS Local SGLang Vocoder Attention Plan

## Objective

Optimize MOSS-TTS Local non-streaming vocoder throughput by patching SGLang with
a reusable helper for the MOSS vocoder's small-batch, chunked, local-causal
attention pattern, then integrating that helper into `sglang-omni` without
changing waveform semantics.

The target is not another model-local function pointer swap. The target is to
move the repeated MOSS attention packing, cumulative-length creation, unpacking,
and cache update mechanics into a SGLang-owned workspace/helper that keeps
storage stable and removes hot-path Python/list/allocation overhead.

Success criteria:

- MOSS vocoder waveform parity stays exact: `max_abs_delta == 0` against the
  current processor decode path on the real model.
- Full SeedTTS 1088 generate-only run completes with `failed == 0`.
- Performance improves over the current remote-attention baseline, not just over
  the invalidated SGLang-kernel import experiment.
- The SGLang patch is generic enough to live in SGLang `jit_kernel` code, while
  MOSS-specific traversal and RoPE/session wiring stays in `sglang-omni`.

## Current Evidence

Decision-relevant files read:

- `sglang_omni/models/moss_tts_local/vocoder_decoder.py`
  - Owns the current MOSS decoder wrapper.
  - `_forward_streaming_flash()` still builds Python lists, cats Q/K/V,
    constructs fresh `torch.tensor(cu_q/cu_k)`, unpacks through Python lists,
    then delegates cache updates to the remote module.
- `sglang_omni/models/moss_tts_local/streaming_vocoder.py`
  - Owns non-streaming and streaming vocoder request execution.
  - Non-streaming with the owned decoder still calls
    `processor.decode_audio_codes(...)`, temporarily replacing `codec.decoder`.
  - The processor decode path internally drives chunked codec decode, so the
    "non-streaming" benchmark still exercises streaming-style codec state.
- MOSS remote tokenizer source:
  - `MossAudioTokenizerMultiheadAttention._forward_streaming_flash()` has the
    same list/pack/cat/cu-tensor shape as our wrapper.
  - `_update_streaming_cache()` preserves semantics by committing cache and
    offsets only under `exec_mask`.
- `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention.py`
  - Exposes `flash_attn_varlen_func` and `flash_attn_with_kvcache`.
  - `flash_attn_with_kvcache` can update KV in-place, but its fused append
    semantics do not directly match MOSS `exec_mask` commit semantics.
- `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention_v3.py`
  - SGLang FA3 path loads kernels-community or `sgl_kernel` kernels.
  - Varlen has fallback behavior; kvcache path is stricter and should remain
    guarded.
- `/Users/ratish/sglang/python/sglang/srt/layers/attention/vision.py`
  - Useful precedent for shape-cached `cu_seqlens` and max-seqlen resolution.
  - Not a drop-in MOSS implementation.
- `/Users/ratish/sglang/python/sglang/srt/layers/radix_attention.py`
  - Coupled to `ForwardBatch`, SGLang attention backend state, and KV pools.
  - Not the right first target for an audio-tokenizer decoder called inside
    `sglang-omni`.
- `/Users/ratish/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`
  and `triton_backend.py`
  - Their speed comes from planned request/token metadata, cache locations, and
    paged or sliding-window KV pools. The useful lesson is stable metadata and
    preallocated state, not forcing the vocoder into `ForwardBatch`.
- `vocoder.json`
  - Confirms six transformer decoder stages, 92 transformer layers total,
    LayerNorm, GELU, RoPE/sin positional logic, and local-causal attention
    contexts.

Measured state:

| case | completed | failed | qps | rtf mean | latency mean | output tok/s |
|---|---:|---:|---:|---:|---:|---:|
| processor baseline | 1088 | 0 | 4.925 | 0.3881 | 1.622 | 271.0 |
| owned + SGLang varlen import | 1088 | 0 | 4.864 | 0.3923 | 1.642 | 267.6 |
| owned + remote varlen | 1088 | 0 | 4.963 | 0.3845 | 1.609 | 273.1 |

Interpretation:

- The owned decoder is parity-clean.
- The SGLang varlen function import is not the optimization. It preserves the
  remote PyTorch control-flow bottleneck.
- A real SGLang patch must remove repeated Python packing/allocation and make
  cache/workspace storage stable.

## Boundary Map

```text
sglang-omni request path
  MossTTSLocalStreamingVocoderScheduler
    owns request batching, payloads, CPU audio output, session lifecycle
    unchanged except selecting optimized owned decoder

  MossTTSLocalVocoderDecoder
    owns MOSS module traversal, stage wrapping, RoPE call ordering, parity with
    the remote tokenizer implementation
    calls SGLang helper for local-causal chunked attention workspace execution

SGLang core patch
  sglang.jit_kernel.chunked_local_attention
    owns reusable workspace tensors, varlen metadata reuse, pack/unpack helpers,
    in-place cache update helpers, optional guarded kvcache path

not first target
  RadixAttention / FlashInferBackend / TritonBackend
    keep tied to SRT LLM serving, ForwardBatch, request pools, and token KV pools
```

## Mechanical Shape

The helper must preserve the known MOSS state layout:

```text
B = batch size
T = current chunk length
C = local context
H = num heads
D = head dim
E = H * D

x                  [B, T, E]
q, k_cur, v_cur    [B, H, T, D]
cached_keys        [B, H, C, D]
cached_values      [B, H, C, D]
cached_positions   [B, C] int64, -1 means invalid
offset             [B] int64
exec_mask          [B] bool
```

The workspace shape should be reusable across layers with the same
`(Bmax, Tmax, C, H, D, dtype, device)`:

```text
q_pack      [Bmax * Tmax, H, D]
k_pack      [Bmax * (C + Tmax), H, D]
v_pack      [Bmax * (C + Tmax), H, D]
out_pack    [Bmax * Tmax, H, D]
cu_q        [Bmax + 1] int32
cu_k        [Bmax + 1] int32
k_lens      [Bmax] int32
```

The first parity-preserving call remains:

```python
flash_attn_varlen_func(
    q_pack[:total_q],
    k_pack[:total_k],
    v_pack[:total_k],
    cu_q[:B + 1],
    cu_k[:B + 1],
    max_seqlen_q=T,
    max_seqlen_k=max_k,
    causal=True,
    window_size=(context, 0),
)
```

Important invariants:

- RoPE stays outside the helper in Phase 1. The helper receives already-rotated
  `q` and `k_cur`.
- Do not filter `exec_mask == False` rows out of attention in the parity path.
  Remote MOSS computes all rows and only gates cache/offset commit.
- Cache updates must write into stable storage with `copy_`, not rebind
  `state.cached_keys`, `state.cached_values`, or `state.cached_positions`.
- `window_size` must match the remote call: `(context, 0)`.
- `flash_attn_with_kvcache` is not the default Phase 1 path because fused append
  does not directly represent "compute every row, commit only active rows".

## Design

Multi-phase plan, because this crosses two repositories and one hot GPU path:

- SGLang core gets a reusable local-causal chunked attention workspace/helper.
- SGLang-Omni gets MOSS-specific integration and parity/benchmark gates.
- The kvcache and custom-kernel paths stay separate until the static varlen
  workspace proves correctness and measurable benefit.

### SGLang core API sketch

New SGLang file:

```text
/Users/ratish/sglang/python/sglang/jit_kernel/chunked_local_attention.py
```

Initial public surface:

```python
@dataclass
class LocalCausalVarlenWorkspace:
    q_pack: torch.Tensor
    k_pack: torch.Tensor
    v_pack: torch.Tensor
    out_pack: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    k_lens: torch.Tensor
    max_batch_size: int
    max_chunk_len: int
    context: int
    num_heads: int
    head_dim: int

    @classmethod
    def create(...)

def local_causal_varlen_attention_with_cache(
    q: torch.Tensor,              # [B, H, T, D]
    k: torch.Tensor,              # [B, H, T, D]
    v: torch.Tensor,              # [B, H, T, D]
    cache_k: torch.Tensor,        # [B, H, C, D]
    cache_v: torch.Tensor,        # [B, H, C, D]
    cache_pos: torch.Tensor,      # [B, C]
    offset: torch.Tensor,         # [B]
    exec_mask: torch.Tensor,      # [B]
    workspace: LocalCausalVarlenWorkspace,
    *,
    context: int,
    flash_attn_varlen_func: Callable | None = None,
) -> torch.Tensor:               # [B, H, T, D]
    ...
```

Export option:

```text
/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention.py
```

Either re-export the new symbols there, or keep direct import from
`sglang.jit_kernel.chunked_local_attention`.

### SGLang-Omni integration sketch

Changed file:

```text
sglang_omni/models/moss_tts_local/vocoder_decoder.py
```

Add:

- Lazy import for the SGLang workspace helper.
- One workspace per attention wrapper or per stage/layer shape. Prefer per
  attention wrapper first for simple lifetime ownership; optimize sharing after
  correctness.
- Environment/config switch:
  - `remote`: existing remote varlen path.
  - `sglang-varlen`: current kernel import path for comparison only.
  - `sglang-workspace`: new SGLang static workspace helper.
- Fallback to `remote` if the SGLang helper is unavailable.

Do not add a generic vocoder backend abstraction.

## Execution Plan

### Phase 1: stable cache update parity

Goal: remove cache rebinding before introducing new SGLang workspace execution.

Changes:

- In `MossTTSLocalAttention`, stop delegating streaming cache update to the
  remote `_update_streaming_cache()` if that method rebinds tensors.
- Add a local `_update_streaming_cache_in_place(...)` that mirrors remote logic
  but writes with `copy_`.
- Keep attention packing and kernel calls unchanged.

Exit gate:

- Unit parity: owned decoder still produces `max_abs_delta == 0` for all
  existing vocoder probes.
- State stability probe: `data_ptr()` for cached K/V/positions does not change
  across several chunk steps.
- Full 1088 generate-only result does not regress beyond normal run noise.

### Phase 2: SGLang static varlen workspace helper

Goal: land the real SGLang-level reusable helper.

Changes in `/Users/ratish/sglang`:

- Add `python/sglang/jit_kernel/chunked_local_attention.py`.
- Implement `LocalCausalVarlenWorkspace.create(...)`.
- Implement a first Python/Torch packing version that writes into preallocated
  tensors and reuses `cu_q/cu_k`.
- Keep FlashAttention math unchanged through `flash_attn_varlen_func`.
- Add SGLang tests with synthetic tensors:
  - compare against the exact Python list/pack reference.
  - cover contexts `125`, `250`, `400`.
  - cover `B in {1, 2, 4, 8, 16}`.
  - cover `T in {1, 5, 25, 100}`.
  - cover inactive `exec_mask` rows.
  - cover partially filled cache with `cache_pos == -1`.

Exit gate:

- SGLang helper output equals the Python reference for attention output and
  cache/offset state.
- No SRT `ForwardBatch`, paged KV pool, or `RadixAttention` dependency enters
  this helper.

### Phase 3: SGLang-Omni MOSS integration

Goal: route MOSS vocoder attention through the new SGLang helper.

Changes:

- Update `sglang_omni/models/moss_tts_local/vocoder_decoder.py`.
- Add a workspace owner to `MossTTSLocalAttention`.
- Use `sglang-workspace` path inside `_forward_streaming_flash()` after RoPE.
- Keep packed non-streaming path unchanged unless the helper is explicitly
  extended for it.
- Keep the default as `remote` until full benchmark proves improvement.

Exit gate:

- Real codec parity:
  - `processor ms`, `owned ms`, `owned max delta`, `session max delta`.
  - `owned max delta == 0` for `B=1, frames={25,100,300}` and
    `B=8, frames={100,300}`.
- Full 1088 generate-only:
  - `completed == 1088`.
  - `failed == 0`.
  - compare against the same-day `remote` baseline on the same GPU.

### Phase 4: move packing to device kernels

Goal: reduce the overhead still left in the workspace helper.

Changes in SGLang:

- Add Triton or SGL kernel helpers for:
  - packing current Q.
  - packing valid cached K/V plus current K/V.
  - unpacking attention output.
  - in-place tail cache update under `exec_mask`.
- Keep FlashAttention as the softmax/math kernel.

Exit gate:

- Same synthetic parity tests as Phase 2.
- Same real codec parity tests as Phase 3.
- Request profile shows reduced `ar_post_forward` or vocoder attention packing
  time versus Phase 3.

### Phase 5: guarded `flash_attn_with_kvcache` experiment

Goal: test whether SGLang kvcache fusion can help without breaking MOSS
semantics.

Restrictions for the first experiment:

- `context` finite.
- `exec_mask.all()` true.
- RoPE already applied externally.
- dense cache has enough capacity for `C + T`, or tail compaction is proven.
- fallback to static varlen workspace on any unsupported shape or inactive row.

Exit gate:

- Single-layer parity first.
- Real codec parity second.
- Only enable by explicit environment flag after full 1088 benchmark improves.

### Phase 6: optional full custom small-chunk attention kernel

Goal: highest-risk, highest-upside path after the varlen helper is proven.

Scope:

- One kernel computes local-causal chunked attention directly from cache and
  current K/V.
- Cache update may remain a separate helper unless fusion is proven safe.
- This path is opt-in until real codec parity is exact.

Exit gate:

- Exact parity or explicitly rejected if bit equality is impossible due to
  accumulation-order differences.

## Validation Commands

Local unit checks in `sglang-omni`:

```bash
python3 -m pytest tests/unit_test/moss_tts_local/test_vocoder_decoder.py -q
python3 -m pytest tests/unit_test/moss_tts_local/test_streaming_vocoder.py -q
python3 -m pytest tests/unit_test/moss_tts_local/test_vocoder_introspection.py -q
```

Remote parity/introspection on H100:

```bash
python -m benchmarks.eval.inspect_moss_tts_local_vocoder \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --device cuda:0 \
  --output-dir /data/moss_vocoder_sglang_workspace_parity \
  --probe batch=1,frames=25 \
  --probe batch=1,frames=100 \
  --probe batch=1,frames=300 \
  --probe batch=8,frames=100 \
  --probe batch=8,frames=300
```

Full generate-only benchmark:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --ref-format references \
  --token-count auto \
  --max-concurrency 8 \
  --max-samples 1088 \
  --output-dir /data/moss_vocoder_sglang_workspace_full_c8 \
  --lang en \
  --seed 42 \
  --warmup 1 \
  --allowed-local-media-path /tmp \
  --generate-only
```

If using an existing server, the server must include the same env and allowed
media path used in the benchmark. Do not compare runs where one server was
already warmed with a different branch or a different vocoder backend.

## Performance Readout

Every performance report must include:

- `completed`, `failed`
- `throughput_qps`
- `rtf_mean`, `rtf_p95`, `rtf_p99`
- `latency_mean`, `latency_p95`, `latency_p99`
- `output_throughput`
- request-profile stage breakdown when available
- which backend was active: `remote`, `sglang-varlen`, `sglang-workspace`,
  `sglang-kvcache-experimental`, or custom kernel

The relevant win signal is not only total RTF. The first internal win signal is
lower per-layer packing/cache overhead in the MOSS attention path.

## Risks And Guards

| risk | guard |
|---|---|
| helper computes inactive rows differently | do not filter inactive rows in parity path |
| `flash_attn_with_kvcache` mutates inactive cache rows | keep it experimental and require `exec_mask.all()` |
| RoPE offset mismatch | keep RoPE outside SGLang helper in Phase 1 |
| cache tensor address changes | assert `data_ptr()` stability in tests |
| SGLang patch cannot be tested from `sglang-omni` branch alone | develop SGLang core patch in `/Users/ratish/sglang`, then integrate via installed patched SGLang or upstream PR |
| full custom kernel is not bit-exact | do not make it default unless real codec parity is exact |
| benchmark noise hides small wins | require same-day branch/server/GPU comparison and request-profile breakdown |

## Non-Goals

- Do not rewrite the entire vocoder as an SGLang `ModelRunner` in this phase.
- Do not force MOSS vocoder into `RadixAttention` or paged KV pools before the
  small-batch helper is proven.
- Do not optimize tokenizer, quantizer, conv, or waveform projection code in
  this plan.
- Do not change streaming chunk emission behavior while optimizing
  non-streaming generation.
- Do not accept approximate waveform parity for the default path.

## Immediate Next Step

Start with Phase 1 in `sglang-omni`: make cache update storage-stable while
keeping the current attention math. Then implement Phase 2 in `/Users/ratish/sglang`
as the SGLang core patch. Only after the SGLang helper passes synthetic parity
should `vocoder_decoder.py` call it in the real MOSS path.
