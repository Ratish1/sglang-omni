# MOSS-TTS Local Vocoder Backend Plan

## Problem Summary

We need to accelerate MOSS-TTS Local non-streaming vocoding without changing
the MOSS vocoder semantics. The correct semantic owner is the Hugging Face MOSS
audio tokenizer decode path:

```text
MossTTSLocalProcessor.decode_audio_codes(...)
  -> processor.audio_tokenizer
    -> quantizer decode
    -> decoder stages
       -> projected transformers
       -> attention backend
    -> waveform
```

The work is still vocoder work. The relevant code is packaged as
`MOSS-Audio-Tokenizer-v2` because the model is an audio codec; the decode half
of that codec is the vocoder.

Success criteria:

- Default production backend preserves the processor waveform contract.
- Exact-preserving backends have `max_abs_delta == 0` against the golden
  processor path on fixed probes and mixed stress cases.
- Experimental approximate backends are opt-in and report objective drift.
- Long and mixed cases must not regress while short cases improve.
- The implementation avoids broad process-global patching as a production
  boundary.

Out of scope for the next phase:

- Rewriting the vocoder as an SGLang model runner.
- Forcing the vocoder through SRT `RadixAttention`, paged KV pools, or
  `ForwardBatch`.
- Making fused RoPE or the owned decoder default before exact parity is proven.
- Treating waveform drift as acceptable without an explicit audio-quality gate.

## Current Evidence

Decision-relevant local files:

- `sglang_omni/models/moss_tts_local/stages.py`
  - Loads the MOSS processor and passes
    `SGLANG_OMNI_MOSS_LOCAL_NONSTREAM_VOCODER_DECODER` into the vocoder
    scheduler.
- `sglang_omni/models/moss_tts_local/streaming_vocoder.py`
  - Owns vocoder request execution, batching, CPU waveform output, and
    non-streaming backend selection.
  - Current backend labels are `processor`, `owned_pytorch`, `sglang_patch`,
    and `session_offline`.
- `sglang_omni/models/moss_tts_local/vocoder_sglang_patch.py`
  - Current branch has a hardened experimental module-global patch with
    refcounting, adapter validation, and invocation counters.
  - H100 results prove the patch is active, but output drift and long/mixed
    regressions remain.
- `sglang_omni/models/moss_tts_local/vocoder_decoder.py`
  - Owns the current experimental owned decoder.
  - Contains custom packed flash, SGLang attention selection, single-unpadded
    metadata, cached RoPE, and detail profiling.
  - Stress results show this path is not exact and must remain experimental.
- `benchmarks/eval/inspect_moss_tts_local_vocoder.py`
  - Current development harness can compare processor, owned decoder,
    session-offline, and SGLang patch paths.
  - It needs backend controls, environment dumps, attention implementation
    dumps, and tensor-level capture/replay before more optimization work.
- `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention.py`
  - SGLang `flash_attn_varlen_func` routes to FA3 by default with `ver=3`,
    or FA4 with `ver=4`.
  - It is not an upstream `flash_attn.flash_attn_varlen_func` drop-in.
- `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention_v3.py`
  - On H100/CUDA 13, SGLang uses FA3 or `sgl_kernel.flash_attn`.
  - That changes the math backend relative to a processor path that does not
    have upstream `flash_attn_varlen_func` importable.

Empirical conclusions:

| path | result | decision |
|---|---|---|
| module-global SGLang patch | active, but nonzero waveform deltas and long/mixed slowdown | experimental only |
| owned decoder | fastest broad path seen, but stress max deltas up to large values | research/profiling only |
| fused RoPE | reduces RoPE time, but waveform drift and low SNR | reject for default |
| processor path | semantically golden | production default |

The latest active-patch stress result confirms this is no longer patch plumbing:

```text
attention_modules=92
active_invocation_count > 0 in every case
max_abs_delta remains nonzero
long and mixed cases are slower
```

## Boundary Map

```text
serving config boundary
  stages.py
    reads env/backend mode once at vocoder construction

runtime owner
  streaming_vocoder.py
    owns request batching, session lifecycle, backend selection, emitted metadata
    must keep processor as golden fallback

semantic owner
  HF processor/audio_tokenizer
    owns quantizer, decoder stage ordering, dtype policy, RoPE, masks, waveform

experimental local owners
  vocoder_sglang_patch.py
    owns module-global SGLang FA adapter experiments only

  vocoder_decoder.py
    owns owned decoder experiments only

measurement owner
  inspect_moss_tts_local_vocoder.py
    owns backend comparison, env dumps, stress cases, tensor capture/replay

not the next boundary
  SGLang SRT attention backends
    coupled to LLM serving metadata and KV pools, not MOSS codec decode
```

## System Mechanics

### Current Golden Flow

```text
codes_list: list[[T, n_vq] cpu int64]
  -> processor.decode_audio_codes(codes_list)
    -> remote processor pads and batches codes
    -> audio_tokenizer._decode_frame
      -> quantizer decode, dtype-sensitive
      -> decoder module list
      -> attention path chosen by remote attention_implementation resolver
      -> waveform tensors
  -> streaming_vocoder converts to CPU payload
```

Contract:

- Inputs: list of CPU `torch.int64` code tensors, shape `[frames, n_vq]`.
- Outputs: CPU waveform tensors with stable length and dtype.
- Ownership: HF processor and audio tokenizer own model semantics.
- Invariants: quantizer dtype, decoder stage ordering, RoPE positions, local
  masks, padding behavior, and waveform lengths must match golden output.
- Failure behavior: unsupported experimental backend must fail during worker
  initialization or fall back to processor, not silently drift.
- Cost model: request-time decode, heavy GPU kernels, per-stage transformer
  work, per-batch padding/packing, CPU output copy.

### Why The SGLang Patch Failed

The module-global patch replaces the remote module's flash-attention hook with
SGLang FA3/sgl-kernel on H100. Since upstream `flash_attn_varlen_func` is not
available in the H100 environment, this does not reproduce the processor
baseline. It changes the backend math and local-window behavior surface. The
nonzero waveform deltas are therefore expected and the mixed-case slowdown is
decisive.

### Why The Owned Decoder Is Not The Next Default

The owned decoder bypasses more of the remote implementation and includes
custom attention, packing, and RoPE logic. It gives a speed ceiling, but stress
parity failed. It should be used to identify hotspots and first-divergence
points, not as production behavior.

## Design Decision

Use a multi-phase plan because this affects runtime behavior, model output,
benchmark infrastructure, and optional SGLang kernel integration.

The next implementation should not add more SGLang kernels. It should establish
a reliable backend comparison harness and exact golden controls first:

1. Introduce explicit vocoder backend naming and experimental labels.
2. Add processor-owned SDPA/cuDNN backend selection and environment reporting.
3. Add tensor-level attention/stage capture and replay.
4. Use the replay result to decide whether a scoped SGLang FA3/FA4 adapter is
   mechanically valid.
5. Only after exact backend parity is understood, optimize the true top costs.

## Execution Plan

### Phase 1: Backend Contract And Harness

Goal: make every vocoder backend explicit, measurable, and non-ambiguous.

Changes:

- Add a small backend contract module:

```text
sglang_omni/models/moss_tts_local/vocoder_backend.py
```

Proposed enum:

```python
class MossTTSLocalVocoderBackend(str, Enum):
    PROCESSOR = "processor"
    PROCESSOR_SDPA = "processor-sdpa"
    PROCESSOR_FLASH2_UPSTREAM = "processor-flash2-upstream"
    SGLANG_PATCH_EXPERIMENTAL = "sglang-patch-experimental"
    OWNED_EXPERIMENTAL = "owned-experimental"
    SESSION_OFFLINE = "session-offline"
```

- Update `streaming_vocoder.py` to use these labels in metadata.
- Keep backward-compatible env values:
  - `owned`, `owned-pytorch` map to `owned-experimental`.
  - `sglang-patch`, `sglang_patch` map to `sglang-patch-experimental`.
  - `processor`, empty, and `default` map to `processor`.
- Log experimental backend usage at startup.
- Do not change default behavior.

Validation:

```bash
/Users/ratish/sglang-omni/.venv/bin/python -m pytest \
  tests/unit_test/moss_tts_local/test_streaming_vocoder.py \
  tests/unit_test/moss_tts_local/test_vocoder_sglang_patch.py -q
```

Exit criteria:

- Existing env names still work.
- Metadata clearly distinguishes production and experimental backends.
- No benchmark command can accidentally report `sglang_patch` as a production
  path.

### Phase 2: Golden Backend Introspection

Goal: know what the processor path actually uses on H100 before optimizing it.

Changes in `benchmarks/eval/inspect_moss_tts_local_vocoder.py`:

- Add `--backend` with values from `MossTTSLocalVocoderBackend`.
- Add `--dump-env`.
- Add `--dump-model-config`.
- Add `--dump-attention-impl`.
- Add `--sdpa-backend default|math|flash|efficient|cudnn` for processor-SDPA
  experiments using `torch.nn.attention.sdpa_kernel` where available.
- Record:
  - torch, CUDA, cuDNN, GPU capability.
  - SGLang import path and commit if importable.
  - flash-attn import status and version if importable.
  - remote module file and attention implementation counts.
  - decoder stage count and transformer layer count.
  - dtype for representative codec parameters.

Validation command:

```bash
CUDA_VISIBLE_DEVICES=6 /sgl-workspace/sglang-omni/.venv/bin/python \
  -m benchmarks.eval.inspect_moss_tts_local_vocoder \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --device cuda:0 \
  --output-dir /data/moss_vocoder_backend_golden \
  --backend processor \
  --dump-env \
  --dump-model-config \
  --dump-attention-impl \
  --stress-suite \
  --iterations 3
```

Exit criteria:

- Report says exactly which processor attention implementation is active.
- Report includes whether upstream flash-attn is actually importable.
- Report identifies whether cuDNN SDPA is enabled.

### Phase 3: Exact Processor-SDPA / cuDNN Experiments

Goal: find a lower-risk speed path that keeps the HF processor as semantic
owner.

Changes:

- Add a context manager that forces the remote MOSS attention implementation to
  `sdpa` for a processor decode run, restoring it afterward.
- Add optional SDPA backend selection around `processor.decode_audio_codes`.
- Keep quantizer and decoder stage code unchanged.
- Add negative-path behavior: if a requested SDPA backend is unavailable, record
  the error and skip or fail explicitly based on a CLI flag.

Commands:

```bash
for backend in default math flash efficient cudnn; do
  CUDA_VISIBLE_DEVICES=6 /sgl-workspace/sglang-omni/.venv/bin/python \
    -m benchmarks.eval.inspect_moss_tts_local_vocoder \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --device cuda:0 \
    --output-dir "/data/moss_vocoder_sdpa_${backend}" \
    --backend processor-sdpa \
    --sdpa-backend "${backend}" \
    --stress-suite \
    --iterations 3
done
```

Exit criteria:

- `max_abs_delta == 0` against processor default or clearly explained if the
  default itself used a different backend.
- Long and mixed cases do not regress.
- If one SDPA backend improves speed, test it in full 1088 generate-only.

### Phase 4: Attention Tensor Capture And Replay

Goal: localize drift before full waveform amplification.

Changes:

- Add an opt-in capture facility in the benchmark tool, not in serving hot path.
- Capture one or a bounded number of attention calls per stage/layer:
  - module path or stage/layer index.
  - input shape, dtype, stride, device.
  - q/k/v after projection.
  - q/k after RoPE.
  - position ids or mask metadata.
  - cu seqlens and max seqlens for packed paths.
  - attention output before and after output projection.
  - causal, context, window size.
- Add replay mode:
  - explicit fp32 reference attention.
  - PyTorch SDPA math.
  - PyTorch SDPA selected backend when available.
  - SGLang FA3 and FA4 adapters.

Proposed commands:

```bash
CUDA_VISIBLE_DEVICES=6 /sgl-workspace/sglang-omni/.venv/bin/python \
  -m benchmarks.eval.inspect_moss_tts_local_vocoder \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --device cuda:0 \
  --output-dir /data/moss_vocoder_attention_capture \
  --backend processor \
  --capture-attention /data/moss_vocoder_attention_capture/processor_attention.pt \
  --probe 1x100 \
  --iterations 1
```

```bash
CUDA_VISIBLE_DEVICES=6 /sgl-workspace/sglang-omni/.venv/bin/python \
  -m benchmarks.eval.inspect_moss_tts_local_vocoder \
  --replay-attention /data/moss_vocoder_attention_capture/processor_attention.pt \
  --attention-backends explicit,sdpa_math,sglang_fa3,sglang_fa4 \
  --output-dir /data/moss_vocoder_attention_replay
```

Exit criteria:

- We know whether SGLang FA3/FA4 differs at the first attention output.
- We know whether the mismatch is attention kernel, RoPE, mask/window, packing,
  or downstream amplification.
- No further SGLang backend work proceeds without this evidence.

### Phase 5: Local-Causal Mask Equivalence Test

Goal: prove `window_size=(context, 0)` and the SDPA dense mask cover identical
keys for MOSS shapes.

Add unit tests in:

```text
tests/unit_test/moss_tts_local/test_vocoder_attention_masks.py
```

Cases:

```text
T in {1, 2, 3, 4, 8, 17}
context in {1, 2, 3, 8, 125, 250, 400}
causal in {true, false}
equal q/k lengths and intentionally unequal q/k lengths for replay diagnostics
```

Exit criteria:

- Key sets match for the real non-streaming self-attention shape.
- If any mismatch appears, SGLang FA is rejected for exact backend work until
  the window mapping is corrected.

### Phase 6: Owned Decoder First-Divergence Tool

Goal: decide whether the fast owned decoder can be salvaged.

Changes:

- Add `--stage-parity` to the benchmark tool.
- Compare processor vs owned-experimental after:
  - quantizer decode.
  - each pretransform.
  - each projected transformer input projection.
  - RoPE.
  - attention before output projection.
  - attention after output projection.
  - FFN.
  - each decoder stage output.
  - final waveform.
- Dump the first divergent tensor with module path, max/mean delta, shape, dtype,
  and device.

Exit criteria:

- If divergence is in a small wrapper mistake, fix it and rerun stress.
- If divergence is systemic or expensive to repair, freeze owned decoder as
  profiling-only and remove it from production configs.

### Phase 7: Scoped SGLang Adapter Only If Replay Supports It

Goal: retain the SGLang backend possibility without global unsafe behavior.

Precondition:

- Attention replay shows SGLang FA3 or FA4 can match an accepted reference under
  the exact MOSS call contract, or product explicitly accepts approximate audio.

Changes:

- Keep `vocoder_sglang_patch.py` as experimental.
- Prefer object-scoped replacement of attention instance methods over module
  globals if remote module structure permits it.
- Keep the adapter upstream-compatible:
  - reject unsupported `dropout_p`, `alibi_slopes`, `deterministic`,
    `return_attn_probs`, and unknown kwargs.
  - pass `softmax_scale`, `causal`, `window_size`, `seqused_q`, `seqused_k`,
    and `softcap` explicitly.
  - make FA version explicit: `sglang-fa3-experimental` or
    `sglang-fa4-experimental`.

Exit criteria:

- Exact tensor replay parity or approximate-quality gate.
- Fresh-process stress suite.
- Full 1088 benchmark only after stress is safe.

## Validation Matrix

Every candidate backend must report:

```text
completed / failed
qps
rtf_mean, rtf_p95, rtf_p99
latency_mean, p95, p99
output_throughput
audio_throughput
waveform max_abs_delta, mean_abs_delta, SNR
attention implementation counts
active backend metadata
experimental flag status
```

Strict default gate:

```text
waveform lengths match
max_abs_delta == 0
no NaN/Inf
no long/mixed regression
no patch leakage after process shutdown
```

Approximate experimental gate:

```text
waveform lengths match
no NaN/Inf or clipping
SNR target documented before running
mel/STFT distance reported
ASR/WER non-regression on representative samples
human listening pass before production default
```

## Immediate Next Step

Implement Phase 1 and Phase 2 first.

Rationale:

- We already proved the SGLang module-global patch is active and not exact.
- We already proved the owned decoder is fast but not exact.
- We do not yet have a reliable golden-backend report or SDPA/cuDNN matrix.
- Without that, more SGLang kernel work is guessing.

First PR slice:

```text
sglang_omni/models/moss_tts_local/vocoder_backend.py
benchmarks/eval/inspect_moss_tts_local_vocoder.py
sglang_omni/models/moss_tts_local/streaming_vocoder.py
tests/unit_test/moss_tts_local/test_streaming_vocoder.py
tests/unit_test/moss_tts_local/test_vocoder_sglang_patch.py
```

First PR exit command:

```bash
/Users/ratish/sglang-omni/.venv/bin/python -m pytest \
  tests/unit_test/moss_tts_local/test_streaming_vocoder.py \
  tests/unit_test/moss_tts_local/test_vocoder_sglang_patch.py -q
```

First H100 exit command:

```bash
CUDA_VISIBLE_DEVICES=6 /sgl-workspace/sglang-omni/.venv/bin/python \
  -m benchmarks.eval.inspect_moss_tts_local_vocoder \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --device cuda:0 \
  --output-dir /data/moss_vocoder_backend_golden \
  --backend processor \
  --dump-env \
  --dump-model-config \
  --dump-attention-impl \
  --stress-suite \
  --iterations 3
```

Only after that report is clean should we decide whether the next code change is
SDPA/cuDNN backend control, owned-decoder first-divergence, or scoped SGLang FA
replay.
