# gpt-5.5 pro research prompt: moss-tts local vocoder sglang backend

We are optimizing MOSS-TTS Local non-streaming vocoder in SGLang-Omni. The strategic goal is to avoid a large custom vocoder reimplementation and instead patch/reuse SGLang attention or kernel internals cleanly inside the existing MOSS vocoder path. We need a production-quality design, not a brute-force or hacky implementation.

## repositories and branches

- SGLang-Omni repo: `/Users/ratish/sglang-omni`
- Current experimental worktree: `/Users/ratish/sglang-omni/.worktrees/moss-vocoder-monkey-patch`
- Branch: `perf/moss-vocoder-monkey-patch`
- Commit: `04573e8f Add SGLang vocoder attention patch mode`
- Local SGLang repo: `/Users/ratish/sglang`

## relevant files to read

Read these files completely, segmenting large files as needed:

- `sglang_omni/models/moss_tts_local/vocoder_sglang_patch.py`
- `sglang_omni/models/moss_tts_local/streaming_vocoder.py`
- `sglang_omni/models/moss_tts_local/vocoder_decoder.py`
- `benchmarks/eval/inspect_moss_tts_local_vocoder.py`
- `tests/unit_test/moss_tts_local/test_vocoder_sglang_patch.py`
- `tests/unit_test/moss_tts_local/test_streaming_vocoder.py`
- `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention.py`
- `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention_v3.py`
- Any relevant SGLang attention/backend files that explain varlen FlashAttention semantics, FA2 vs FA3 selection, window semantics, deterministic behavior, and kernel defaults

Also inspect the MOSS remote tokenizer implementation if available from Hugging Face cache or local introspection JSON artifacts. Important known snippets from the remote MOSS audio tokenizer:

```python
def forward(self, x, input_lengths, **kwargs):
    x = self.input_proj(x.transpose(1, 2))
    if not self.is_streaming and self.transformer.resolve_attention_implementation(x) == "flash_attention_2":
        batch_size, max_seqlen, _ = x.shape
        if max_seqlen > 0 and bool(input_lengths.any().item()):
            max_valid_seqlen = int(input_lengths.max().item())
            packed_x, valid_mask, cu_seqlens, position_ids = pack_padded_sequence(x, input_lengths)
            packed_x = self.transformer(
                packed_x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_valid_seqlen,
                position_ids=position_ids,
                input_lengths=input_lengths,
                **kwargs,
            )
            x = unpack_packed_sequence(packed_x, valid_mask, batch_size, max_seqlen)
        else:
            x = x.new_zeros(x.shape)
    else:
        x = self.transformer(x, input_lengths=input_lengths, **kwargs)
    x = self.output_proj(x).transpose(1, 2)
    return x, input_lengths
```

```python
def _forward_non_streaming_flash(self, x, cu_seqlens, max_seqlen, position_ids):
    q, k, v = self._project_qkv(x)
    q, k = self._apply_packed_rope(q, k, position_ids)
    out = self._run_flash_attention(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
    return out.reshape(x.shape[0], self.embed_dim)
```

```python
def _run_flash_attention(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    if flash_attn_varlen_func is None:
        raise RuntimeError("flash-attn is not installed.")
    window_size = (self.context, 0) if (self.context is not None and self.causal) else (-1, -1)
    return flash_attn_varlen_func(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=self.causal,
        window_size=window_size,
    )
```

## experiments already run

### owned decoder path

We built a repo-owned decoder replacement using SGLang-backed attention. It reached good full benchmark speed but failed strict stress parity against `processor.decode_audio_codes`.

Stress result:

| case | batch | total frames | max frames | processor ms | owned ms | owned max delta |
|---|---:|---:|---:|---:|---:|---:|
| single_short | 1 | 25 | 25 | 76.546 | 51.418 | 0.0732422 |
| single_typical | 1 | 100 | 100 | 83.922 | 50.836 | 0.123779 |
| single_long | 1 | 300 | 300 | 225.708 | 137.256 | 1.42578 |
| mixed_8 | 8 | 1025 | 300 | 754.969 | 331.375 | 0.941406 |
| mixed_16 | 16 | 2449 | 400 | 2005.005 | 766.758 | 1.53906 |

Conclusion: fast but not correctness-safe.

### fused rope experiment

An SGLang-style fused RoPE path was faster but not waveform-exact:

| probe | max_abs | mean_abs | snr |
|---|---:|---:|---:|
| bs=1, frames=25 | 0.06543 | 0.000906 | 31.80 dB |
| bs=1, frames=100 | 0.01807 | 0.000525 | 37.09 dB |
| bs=8, frames=100 | 0.03397 | 0.000543 | 34.78 dB |

Conclusion: do not use as correctness-safe default.

### monkey patch experiment

We then created a smaller patch that preserves the Hugging Face processor/vocoder implementation and only changes the remote module globals:

```python
module.flash_attn_varlen_func = sglang.jit_kernel.flash_attention.flash_attn_varlen_func
module.HAS_FLASH_ATTN = True
```

The patch installed successfully with `attention_modules=92`, but parity was still nonzero and mixed/long cases became slower:

| case | processor ms | sglang patch ms | max_abs_delta |
|---|---:|---:|---:|
| 1x25 | 173.240 | 104.982 | 0.0537109 |
| 1x100 | 164.564 | 100.018 | 0.0488281 |
| 1x300 | 225.327 | 278.221 | 0.224854 |
| 8x100 | 194.551 | 206.353 | 0.116211 |
| 8x300 | 547.350 | 607.493 | 0.573242 |
| single_short | 79.697 | 100.816 | 0.0732422 |
| single_typical | 85.374 | 102.283 | 0.123779 |
| single_long | 244.350 | 287.124 | 0.202881 |
| mixed_8 | 758.759 | 1064.330 | 0.513184 |
| mixed_16 | 1995.862 | 3716.965 | 0.674805 |

Important local observation: `/Users/ratish/sglang/python/sglang/jit_kernel/flash_attention.py` routes `flash_attn_varlen_func` through `flash_attention_v3.py` by default (`ver=3`). On H100, `flash_attention_v3.py` reports FA3 support and calls SGLang/sgl-kernel FA3 kernels, only falling back to upstream `flash_attn` FA2 when FA3 is unsupported. Therefore the monkey patch does not merely replace missing `flash_attn` with an equivalent function; it may force a different algorithm/kernel than the unpatched processor path.

## questions to answer

Please answer with a production-code, ML systems perspective:

1. Did the monkey patch target the correct HF hook in principle?
2. Are we missing additional HF globals/classes/methods required to make this exact, or is the observed drift expected from forcing SDPA/cuDNN or upstream behavior into SGLang FA3/FA2?
3. Could the introspection benchmark itself be wrong due to processor statefulness, streaming state, global patch restore contamination, random codes, CUDA nondeterminism, or calling order?
4. Compare SGLang `flash_attn_varlen_func` signature and defaults to upstream flash-attn. Are there semantic mismatches around:
   - FA2 vs FA3 selection
   - `deterministic`
   - `return_attn_probs` vs `return_softmax_lse`
   - `softcap`
   - `num_splits`
   - `window_size`
   - bottom-right causal alignment
   - dtype/accumulation order
   - dropout default
5. What is the next minimal experiment to isolate the cause:
   - patch to upstream `flash_attn.flash_attn_varlen_func` directly
   - patch to SGLang wrapper with `ver=4` or `ver=3`
   - force remote attention implementation without setting `HAS_FLASH_ATTN`
   - compare SDPA vs upstream FA2 vs SGLang FA3 using a single MOSS attention layer
   - measure exact tensor deltas at attention output before vocoder waveform
6. Strategically, should we pursue:
   - clean monkey patch only
   - upstream remote-code-compatible adapter
   - SGLang core kernel changes
   - optimize exact SDPA/cuDNN processor path
   - accept approximate waveform deltas under WER/listening gates
7. What would a top-tier production codebase do here, balancing correctness, speed, maintainability, and risk?

## desired output

Give a findings-first answer:

- correctness findings with severity
- whether the monkey patch is coded correctly
- whether the result invalidates the elegant strategy or just this backend choice
- exact next experiments and commands/code changes
- final recommendation for the branch direction
