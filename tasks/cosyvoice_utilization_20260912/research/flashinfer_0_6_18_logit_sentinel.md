# FlashInfer 0.6.18 ragged prefill: where the attention logits live, and the one magnitude limit the code has

Scope: flashinfer-python 0.6.18 (SGLang v0.5.19 pin), `BatchPrefillWithRaggedKVCacheWrapper`,
NHD, 16 heads, head_dim 64, bf16 q/k/v, sm_scale 1/8, H100 / sm90.

Sources read (all at tag `v0.6.18` unless stated):

- Python side, unpacked wheel: `/private/tmp/claude-501/-Users-ratish-sglang-omni/5d9d5e5e-868b-452c-b92f-ebdcfb84eacf/scratchpad/fi/src/flashinfer/{prefill.py,utils.py}`
- CUDA side, downloaded from `raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.6.18/...` into
  `/private/tmp/claude-501/-Users-ratish-sglang-omni/5d9d5e5e-868b-452c-b92f-ebdcfb84eacf/scratchpad/fisrc/`:
  `prefill.cuh`, `variants.cuh`, `variant_helper.cuh`, `mask.cuh`, `math.cuh`,
  `batch_prefill.cu`, `batch_prefill_sm90.cu`, `hopper/{attention_updater,mainloop,mainloop_mma,kernel_traits,prefill_sm90,variants,utils,epilogue}.cuh`
- The partial tree at
  `/Users/ratish/sglang-omni/.worktrees/cosyvoice-utilization-20260912/tasks/cosyvoice_utilization_20260912/inputs/external_sources/flashinfer`
  contains only `LICENSE`, `csrc/norm.cu`, `csrc/renorm.cu`, `include/flashinfer/norm.cuh`,
  `include/flashinfer/sampling.cuh` — **none** of the attention files. Everything below came from the fetched copies.

Line numbers are those of the v0.6.18 files.

---

## Headline

**Neither backend rounds S or the scaled logits to a 16-bit type. Every logit, row max, exp2
argument and running sum is fp32 in both FA2 and FA3.** The bf16-rounding hypothesis is not
supported by the code.

The one magnitude limit that *is* hard-coded is
`math.cuh:33  constexpr float inf = 5e4;` — FlashInfer's "negative infinity" is the finite value
**-50000.0f**, used both as the masked-logit fill and as the FA2 online-softmax running-max
initialiser, **in the raw (pre-`sm_scale`) logit domain**. That is a known, reported, fixed-on-main
bug (issues #4267 / #4450 / #4451 / #4452, PR #4401) and the fix is **not** in the 0.6.18 pin.

---

## Flow of one logit, file:line anchored

```
                      wrapper.run()                     prefill.py:4086-4092  sm_scale = 1/sqrt(64)   [python float]
                            |                           prefill.py:4314-4316  q touched ONLY for fp8
                            |                           prefill.py:4366       sm_scale -> run_args
                            v
                 ragged_run(...)  prefill.py:4380
                      /                    \
   custom mask? yes  /                      \  no          utils.py:457-458 / 587-600
                    v                        v
            ===== FA2 =====            ===== FA3 (sm90) =====
      csrc/batch_prefill.cu:145   csrc/batch_prefill_sm90.cu:102
                    |                        |
   q,k bf16 smem    |                        |  q,k bf16 smem
                    v                        v
   compute_qk           prefill.cuh:1250     gemm(TiledMmaQK)      hopper/mainloop_mma.cuh:77
   mma...f16f16k16_f32                       ss_op_selector<...,float>  hopper/kernel_traits.cuh:76
                    |                        |
      s_frag : float[8]  (RAW q.k)      tSrS : float  (RAW q.k)   hopper/mainloop_mma.cuh:73
      prefill.cuh:2200/2839/3617             |
                    |                        |
   logits_transform     prefill.cuh:1424     LogitsTransform       hopper/mainloop_mma.cuh:153
   identity, fp32 local 1461/1473            identity, fp32        hopper/variants.cuh:100
                    |                        |
   logits_mask  SELECT, fp32                 mask  SELECT, fp32    hopper/mainloop_mma.cuh:158-168
   prefill.cuh:1515-1516                     |
     else -> MaskFillValue                     else -> fill_value
     = DTypeQKAccum(-math::inf)                = -math::inf
     prefill.cuh:350-351                       hopper/attention_updater.cuh:168
              \                              /
               \____ math.cuh:33  inf = 5e4  ____/      <-- FINITE.  -5e4 in the RAW domain
                                                            (= -6250 in q.k/8 at sm_scale 1/8)
                    |                        |
   m : float, INIT = -5e4                    row_max : float, INIT FROM DATA
   prefill.cuh:949 / 2206                    hopper/attention_updater.cuh:169-170, 186
                    |                        |
   update_mdo_states                         OnlineSoftmax::update
   ptx_exp2(s*sm_scale_log2                  exp2f(s*scale - row_max*scale)
            - m*sm_scale_log2)               hopper/attention_updater.cuh:137
   prefill.cuh:1633-1641                     |
   ex2.approx.ftz.f32  math.cuh:45           |   sm_scale_log2 = sm_scale*log2e
   sm_scale_log2 = sm_scale*log2e            |   hopper/variants.cuh:91
   variants.cuh:54                           |
                    |                        |
      p : float  <-- FIRST and ONLY 16-bit conversion happens HERE, on p, not on s
                    |                        |
   DTypeQ s_frag_f16    prefill.cuh:1699     convert_type<DTypeKV>(tSrS)  hopper/mainloop_mma.cuh:175
                    |                        |
   PV mma -> o_frag : float                  PV wgmma -> acc_o : float  (ElementAccum=float, kernel_traits.cuh:78)
                    |                        |
   finalize_m  prefill.cuh:1810              finalize  hopper/attention_updater.cuh:221-238
     m *= sm_scale_log2  ONLY IF m != -5e4     inv_sum = pv_scale / sum      <-- no sum==0 guard in 0.6.18
                    |                        |
   OutputTransform                           epilogue  convert_type<DTypeO>  hopper/epilogue.cuh:162
   d_rcp = (m != -5e4) ? rcp(d) : 0.f
   variant_helper.cuh:86
                    |
     row whose RAW max never exceeded -5e4  ==>  d_rcp = 0  ==>  output exactly 0
```

---

## 0. Which kernel actually runs (CONFIRMED)

- `utils.py:427-457` `is_fa3_backend_supported(...)`, first clause:
  ```
  457:    if use_custom_mask:
  458:        return False
  ```
  CONFIRMED — a custom mask disqualifies FA3.
- `utils.py:473-478` `is_fa3_prefill_head_dim_supported`: equal head dims must be in `{64,128,256}`,
  so head_dim 64 is supported. CONFIRMED.
- `utils.py:587-600` `determine_attention_backend`: sm90a + supported ⇒ `"fa3"`, else `"fa2"`. CONFIRMED.
- `prefill.py:3804-3814` — the ragged wrapper's `plan()` resolves `backend == "auto"` through that
  function, passing `self._custom_mask_buf is not None` as `use_custom_mask`. CONFIRMED.
- `prefill.py:4318-4324` — `mask_mode = MaskMode.CUSTOM.value` whenever `_custom_mask_buf is not None`. CONFIRMED.
- Entry points: FA2 `csrc/batch_prefill.cu` (`ragged_run`, mask mode dispatched at line 145/272);
  FA3 `csrc/batch_prefill_sm90.cu` (line 102/198). CONFIRMED.
- `mask.cuh:21-26` — `MaskMode { kNone=0, kCausal=1, kCustom=2, kMultiItemScoring=3 }`. CONFIRMED.

So the observed resolution (FA3 without a custom mask, FA2 with one) matches the code exactly.

---

## 1. FA2 kernel (`include/flashinfer/attention/prefill.cuh`)

### 1.1 `DTypeQKAccum` and how it is chosen — CONFIRMED

Three identical definitions, one per launcher (single prefill, batch ragged, batch paged):

```
prefill.cuh:2554  using DTypeQKAccum =
prefill.cuh:2555      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
prefill.cuh:2556                                float>::type;
```
identically at `prefill.cuh:4164-4166` and `prefill.cuh:4364-4366`.

For bf16 inputs the second conjunct `std::is_same_v<DTypeQ, half>` is **false**, so
`DTypeQKAccum = float` *regardless* of `use_fp16_qk_reduction`; with
`use_fp16_qk_reduction=False` it is `float` for fp16 inputs too. CONFIRMED.

`KTraits` just forwards it: `prefill.cuh:291  using DTypeQKAccum = DTypeQKAccum_;`. CONFIRMED.

The `half` branch is additionally unreachable unless the build defines
`FP16_QK_REDUCTION_SUPPORTED` — `prefill.cuh:347-349` is a `static_assert(!std::is_same<DTypeQKAccum, __half>::value, ...)` otherwise. CONFIRMED.

### 1.2 What holds S = Q K^T after the mma — CONFIRMED

`compute_qk`, `prefill.cuh:1246-1257`:
```
1248:        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
1250:            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
1251:                s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
```
i.e. `m16n16k16` with **fp32 accumulate** (`f16f16f32`) into `s_frag`. The register array is declared
`DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][8]` at `prefill.cuh:2200`, `2839`, `3617` ⇒ `float[8]`. CONFIRMED.

### 1.3 Row max `m` and running sum `d` — CONFIRMED

- `prefill.cuh:2206 / 2844 / 3625`: `DTypeQKAccum m[NUM_MMA_Q][2];` ⇒ **float**.
- Same lines +1: `float d[NUM_MMA_Q][2];` ⇒ **float**.
- `init_states`, `prefill.cuh:949-950`: `m[mma_q][j] = DTypeQKAccum(-math::inf); d[mma_q][j] = 1.f;`.
- The warp reduction is `max(...)` over floats with `math::shfl_xor_sync` (`prefill.cuh:1618-1623`).

### 1.4 Where `sm_scale` enters, and the type at mask time and at exp2 — CONFIRMED

**`s_frag` holds the RAW `q·k` dot product, unscaled, all the way through masking.** `sm_scale` is
never applied to q, to k, or to S; it is folded into the exp2 argument only:

- `variants.cuh:54  sm_scale_log2 = params.sm_scale * math::log2e;`
- `update_mdo_states`, `prefill.cuh:1605  const float sm_scale = variant.sm_scale_log2;`
- `prefill.cuh:1622  float o_scale = math::ptx_exp2(m_prev * sm_scale - m[mma_q][j] * sm_scale);`
- `prefill.cuh:1633-1641`
  ```
  s_frag[mma_q][mma_kv][j * 2 + 0] = math::ptx_exp2(
      s_frag[mma_q][mma_kv][j * 2 + 0] * sm_scale - m[mma_q][j] * sm_scale);
  ```
  (four register positions, identical form)

`math::ptx_exp2` is `ex2.approx.ftz.f32` on an `f32` operand (`math.cuh:43-47`). So the scaled logit
exists only as an fp32 temporary inside that expression. The `half2` variant of `ptx_exp2`
(`math.cuh:178-184`) is reachable only through the `DTypeQKAccum == half` branch
(`prefill.cuh:1644-1684`), which bf16 inputs cannot take. CONFIRMED.

`LogitsTransform` for the DiT configuration (no alibi, no soft cap) is the identity
(`variants.cuh:67-76`). In `logits_transform` the value is copied to a `float logits` local
(`prefill.cuh:1424`, `1461`) and written straight back (`prefill.cuh:1473`/`1476`); the 16-bit
`fp16_ieee_to_fp32_value` / `fp16_ieee_from_fp32_value` round trip at `prefill.cuh:1455-1470` is
guarded on `std::is_same<DTypeQKAccum, __half>` and is dead for bf16. CONFIRMED.

### 1.5 Where S / scaled logits are converted to DTypeQ — CONFIRMED: only *after* exp2

The only 16-bit conversion on the score path is of the **probabilities**, for the PV mma:

- `compute_sfm_v`, `prefill.cuh:1699`: `typename KTraits::DTypeQ s_frag_f16[NUM_MMA_Q][NUM_MMA_KV][8];`
  filled from the already-exponentiated `s_frag` (`prefill.cuh:1704-1724`).
- VO-split path, `prefill.cuh:3292-3293`:
  `p_smem[...] = typename KTraits::DTypeQ(p0);` where `p0 = ptx_exp2(...)` on line 3286.

`o_frag` is `float` throughout (`prefill.cuh:1598`), `d` is `float`, and the only other conversion
is the final output write. **No conversion of S or of the scaled logits to bf16 exists in the FA2
path.** CONFIRMED.

---

## 2. FA3 / Hopper kernel

### 2.1 S accumulator in the wgmma mainloop — CONFIRMED: fp32

- `hopper/kernel_traits.cuh:52  using DTypeQKAccum = float;`
- `hopper/kernel_traits.cuh:76  cute::GMMA::ss_op_selector<DTypeQ, DTypeKV, DTypeQKAccum, TileShape_QKD>(), AtomLayoutQKD{}`
  — the QK wgmma's C type is that `float`.
- `hopper/kernel_traits.cuh:78  cute::GMMA::rs_op_selector<DTypeKV, DTypeKV, /*ElementAccum=*/float, TileShape_PDV, ...>`
  — the PV wgmma also accumulates in fp32.
- `hopper/mainloop_mma.cuh:73  Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0,1>(TileShape_QKD{}));`
  ⇒ fp32 register fragment. CONFIRMED.

Call chain verified: `hopper/prefill_sm90.cuh:276` calls `mma_f16<Ktraits, ...>`, defined in
`hopper/mainloop_mma.cuh` (included at `prefill_sm90.cuh:29`). CONFIRMED.

### 2.2 Softmax / rescale path — CONFIRMED: fp32 throughout

`hopper/attention_updater.cuh`:
- `169-170  TensorT row_max, row_sum, scores_scale;` where
  `169  using TensorT = decltype(make_tensor<float>(Shape<Int<NUM_ROWS_PER_THREAD>>{}));` ⇒ fp32.
- `125-140  scale_apply_exp2`: `tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - row_max * scale);`
  — fp32 `exp2f`, fp32 operands, `scale = sm_scale_log2`.
- `203  scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * sm_scale_log2);`
- `226-237  finalize`: `quad_allreduce_` on fp32, `inv_sum = pv_scale / sum`,
  `row_sum(mi) = row_max(mi) * sm_scale_log2 + math::ptx_log2(sum)`.
- `242-253  rescale_o`: fp32 multiply into `acc_o`.

`sm_scale_log2` comes from `hopper/variants.cuh:91  sm_scale_log2 = params.additional_params.sm_scale * math::log2e;`
(`StandardAttention`), passed to `OnlineSoftmax<..., WITH_SCALE=true>` at `hopper/variants.cuh:97`.
`StandardAttention::LogitsTransform` is the identity (`hopper/variants.cuh:100-101`). CONFIRMED.

### 2.3 Any 16-bit conversion of S before exp2 — CONFIRMED: none

`hopper/mainloop_mma.cuh:174-176`:
```
174:  attention_updater.update</*init=*/true>(tSrS);
175:  Tensor tOrP = make_tensor(convert_type<DTypeKV>(tSrS).data(),
176:                            convert_layout_acc_Aregs<typename Ktraits::TiledMmaPV>(tSrS.layout()));
```
The `convert_type<DTypeKV>` (bf16) happens **after** `update()` has already applied `exp2f`, i.e. it
converts P, not S. Same ordering at `mainloop_mma.cuh:225`, `271`, `308`. `convert_type` is a
`cutlass::NumericArrayConverter<..., round_to_nearest>` (`hopper/utils.cuh:144-153`). The only other
conversion is the output: `hopper/epilogue.cuh:162  Tensor tOrO_out = convert_type<DTypeO>(tOrO);`. CONFIRMED.

---

## 3. The custom-mask path: select in float, not a 16-bit additive mask — CONFIRMED

FA2 `logits_mask`, `prefill.cuh:1509-1519`:
```
1510:        const bool mask =
1511:            (!(MASK_MODE == MaskMode::kCausal || MASK_MODE == MaskMode::kMultiItemScoring
1512:                   ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= chunk_end))
1513:                   : kv_idx >= chunk_end)) &&
1514:            variant.LogitsMask(params, batch_idx, q_idx, kv_idx, qo_head_idx, kv_head_idx);
1515:        s_frag[mma_q][mma_kv][reg_id] =
1516:            (mask) ? s_frag[mma_q][mma_kv][reg_id] : (KTraits::MaskFillValue);
```
A **select** on the fp32 `s_frag`, with no addition and no 16-bit intermediate. The custom-mask bit
itself is read from a bit-packed `uint8_t` at `variants.cuh:84-85`:
```
84:        const uint64_t offset = static_cast<uint64_t>(qo_idx) * kv_len + kv_idx;
85:        mask &= ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
```
`MaskFillValue` is `DTypeQKAccum(-math::inf)` (`prefill.cuh:344-345` under
`FP16_QK_REDUCTION_SUPPORTED`, `prefill.cuh:350-351` otherwise). CONFIRMED.

With `MASK_MODE == kCustom` the mask is applied on **every** KV iteration, not just the diagonal tile
(`prefill.cuh:2379-2382`, `prefill.cuh:3946-3949`). CONFIRMED.

FA3 masks the same way, as a select to `AttentionUpdater::fill_value` on the fp32 `tSrS`
(`hopper/mainloop_mma.cuh:154-171`), e.g.
```
157:      } else if constexpr (!CAUSAL) {  // Just masking based on col
158:        if (kv_idx >= kv_len) {
159:          tSrS(i) = AttentionUpdater::fill_value;
```
CONFIRMED. (FA3 has no `kCustom` support at all — see §0.)

---

## 4. Does the wrapper rescale q before the kernel? — CONFIRMED: no

`BatchPrefillWithRaggedKVCacheWrapper.run`, `prefill.py:4086-4098`:
```
4086:        sm_scale = self._sm_scale
4091:        if sm_scale is None:
4092:            sm_scale = 1.0 / math.sqrt(q.size(-1))
```
(the only further adjustment is `sm_scale *= q_scale / k_scale` when `kv_cache_sf is not None`, i.e.
NVFP4 KV — not this case, `prefill.py:4093-4098`).

It is then passed as a plain Python float in `run_args` (`prefill.py:4366`) to
`self._cached_module.ragged_run(*run_args)` (`prefill.py:4380`). CONFIRMED.

- **`q` is never multiplied.** The only touch of `q` in this method is the fp8 fallback
  `prefill.py:4312-4316` (`q = q.to(torch.float16)` when `is_float8(q)`), and `q = q.contiguous()`
  at `prefill.py:4164` in the paged wrapper. A grep for `q *`, `q.mul`, `* sm_scale` over
  `prefill.py` returns no q-rescale on the fa2/fa3 path. CONFIRMED.
- **`log2e` is never applied on the host for fa2/fa3.** `utils.py:55` defines it; `prefill.py`
  multiplies by it at lines `269`, `5002`, `5630`, all three inside trtllm-gen entry points
  (`bmm1_scale = bmm1_scale * log2e` guarded by `isinstance(bmm1_scale, torch.Tensor)`), not the
  wrapper path. CONFIRMED. The device side does `sm_scale * math::log2e` itself
  (`variants.cuh:54`, `hopper/variants.cuh:91`).
- `plan()` stores no scaled tensor; it only resolves the backend and JIT module
  (`prefill.py:3801-3884`). CONFIRMED.

---

## 5. The one hard-coded magnitude limit, and the known issues

### 5.1 `math::inf` is a finite 5e4 — CONFIRMED

```
math.cuh:33  constexpr float inf = 5e4;
```

Every "negative infinity" in 0.6.18's attention is therefore **-50000.0f**:

| site | file:line | value |
|---|---|---|
| FA2 masked-logit fill | `prefill.cuh:350-351` `MaskFillValue = DTypeQKAccum(-math::inf)` | -5e4 |
| FA2 running-max init | `prefill.cuh:949` (`init_states`), also `2252`, `2897`, `3681` | -5e4 |
| FA2 "row was fully masked" test | `prefill.cuh:1810` `if (m[mma_q][j] != DTypeQKAccum(-math::inf))` | -5e4 |
| FA2 output guard | `variant_helper.cuh:86` `float d_rcp = (m != -math::inf) ? math::ptx_rcp(d) : 0.f;` | -5e4 |
| FA2 cross-warp merge init | `prefill.cuh:1903`, VO-split `3242`, `3260` | -5e4 |
| **FA3** masked-logit fill | `hopper/attention_updater.cuh:168` `constexpr static float fill_value = -math::inf;` | -5e4 |

CONFIRMED (each line read).

### 5.2 What that means arithmetically — DERIVED from the lines above, not measured

The sentinel lives in the **raw** `q·k` domain, because `s_frag` / `tSrS` are unscaled and `sm_scale`
is only applied inside `exp2` (§1.4, §2.2). With `sm_scale = 1/8`, a raw threshold of -5e4
corresponds to a *scaled* logit of **-6250**. The stated DiT range of ±1.3e5 in `q·k/8` is a raw
range of ±1.04e6, so the sentinel sits well inside the live logit range.

Two consequences follow mechanically from the quoted code:

1. **FA2, any mask mode.** `m` starts at -5e4 (`prefill.cuh:949`). A row whose raw maximum is below
   -5e4 never raises `m`, so `m` ends the loop still *bit-equal to the sentinel*. Two things then
   happen: every `s * sm_scale - m * sm_scale` is a large negative number and
   `ex2.approx.ftz.f32` (`math.cuh:45`) flushes it to 0 for any argument at or below -126; and
   `finalize_m` (`prefill.cuh:1810`) skips the `m *= sm_scale_log2` because `m == -math::inf`, so
   the output guard `variant_helper.cuh:86  float d_rcp = (m != -math::inf) ? math::ptx_rcp(d) : 0.f;`
   takes the `0.f` branch and **forces the whole row to exactly zero**, with the LSE left at the
   sentinel. A valid row is misclassified as "fully masked". This is exactly the behaviour reported
   in #4267 / #4450 / #4451 (#4451 quotes LSE `-49992.95703125` for the ragged-FA2 fixture).
2. **Both backends, masked or padded columns.** A masked entry is -5e4 raw. It is suppressed only
   if `(m_raw + 5e4) * sm_scale * log2e > 126`, i.e. for `sm_scale = 1/8` only if
   `m_raw > -5e4 + ~699`. Rows whose live maximum is near or below the sentinel therefore give the
   masked/padded columns non-negligible (or dominant) weight. In FA3 `row_max` is initialised from
   the data itself (`attention_updater.cuh:186 reduce_max</*init=*/true>`), so FA3 is exposed only
   through `fill_value`-filled columns (padding `kv_idx >= kv_len`, causal limit, sliding window),
   not through an initialiser.

I did not run either kernel, so the link from this hazard to the observed 0 dB SNR is UNVERIFIED;
the code facts and the arithmetic above are what the source supports.

### 5.3 Known issues and PRs — CONFIRMED (fetched via the GitHub API)

Searches over `repo:flashinfer-ai/flashinfer` for `"wrong result"`, `"large logits"`, `"5e4"`,
`overflow`, `"custom mask"`, `precision bfloat16 prefill`:

- **https://github.com/flashinfer-ai/flashinfer/issues/4267** — "[Bug] FA2 attention silently
  outputs zeros when every logit in a row is below -5e4 (math::inf sentinel is a finite value)",
  opened 2026-07-30, closed 2026-08-20, reported against 0.6.12.
- **https://github.com/flashinfer-ai/flashinfer/issues/4450** — "[Bug] Classic paged decode returns
  zeros for valid finite logits below the -5e4 sentinel", 2026-08-10, closed 2026-08-20. Fixture:
  every raw QK score `128*64*-64 = -524288`, every V element 1, correct output is exactly 1;
  v0.6.16.post3 and main@2fb785c both return zeros.
- **https://github.com/flashinfer-ai/flashinfer/issues/4451** — "[Bug] Ragged FA2 prefill returns
  zeros for a one-key finite row below the -5e4 sentinel", 2026-08-10, closed 2026-08-20. This is
  the `BatchPrefillWithRaggedKVCacheWrapper(..., backend="fa2")` surface. Reported LSE
  `-49992.95703125` (i.e. pinned at the sentinel) instead of `-66855.859375`.
- **https://github.com/flashinfer-ai/flashinfer/issues/4452** — "[Bug] Paged causal prefill returns
  finite LSE for fully masked rows when qo_len > kv_len", 2026-08-10, closed 2026-08-20.
- **https://github.com/flashinfer-ai/flashinfer/pull/4401** — "fix(attention): handle extreme
  negative logits in masked softmax", **merged 2026-08-20**, merge commit
  `d7f2c64647585b641590e309c75974519dea17db`. Its description states the diagnosis verbatim:
  > The FA2 attention kernels masked logits with a finite sentinel (`math::inf = 5e4`). Because the
  > online-softmax running maximum is tracked over the **raw** QK dot product, a valid logit below
  > `-5e4` could never raise the initialized maximum: every probability underflowed to zero and
  > valid rows returned zero output with a sentinel LSE (#4267, #4450).

  Files touched include `include/flashinfer/math.cuh` (+5/-1), `include/flashinfer/attention/prefill.cuh`
  (+29/-18), **`include/flashinfer/attention/hopper/attention_updater.cuh` (+10/-6)** — so the FA3
  path was affected too — plus `cascade.cuh`, `decode.cuh`, `mla.cuh`, `mla_hopper.cuh`, `state.cuh`.

  The fix replaces the sentinel with IEEE `-inf`
  (`math.cuh: constexpr float inf = cuda::std::numeric_limits<float>::infinity();`) and clamps every
  exponent subtrahend, e.g. in `hopper/attention_updater.cuh`:
  `float row_max_scaled = ::fmaxf(max(mi) * scale, -cuda::std::numeric_limits<float>::max());`
  and `float inv_sum = (sum > 0.f) ? pv_scale / sum : 0.f;`.

### 5.4 The fix is NOT in the 0.6.18 pin — CONFIRMED

- `v0.6.18` tag commit `69ff11fc49`, dated 2026-08-28; release published 2026-08-29.
- `GET /repos/flashinfer-ai/flashinfer/compare/d7f2c646...v0.6.18` returns
  `status=diverged ahead=47 behind=18` ⇒ the fix commit is **not** an ancestor of the tag (the
  release branch diverged from main before 2026-08-20).
- Content check, decisive: `math.cuh:33` is `constexpr float inf = 5e4;` at **both** `v0.6.18`
  **and** `v0.6.18.post1`, while `main` has
  `constexpr float inf = cuda::std::numeric_limits<float>::infinity();` with the comment
  "Masked-logit infinity. IEEE -inf (not a finite sentinel like the historical -5e4, which
  misclassified valid logits below it)".
- 0.6.18's `update_mdo_states` (`prefill.cuh:1622`, `1633-1641`) and
  `hopper/attention_updater.cuh:133-138` carry **none** of the clamps PR #4401 added.

### 5.5 What the search did *not* find — CONFIRMED (negative result)

No issue, PR or doc in `flashinfer-ai/flashinfer` documents a supported range or an upper bound on
attention logit magnitude. Searches for `overflow in:title`, `"custom mask" in:title`,
`precision bfloat16 prefill in:title`, `numerical accuracy attention in:title` returned nothing
relevant to bf16 prefill logits. The only precision-adjacent prefill items are
[#4297](https://github.com/flashinfer-ai/flashinfer/issues/4297) /
[#4298](https://github.com/flashinfer-ai/flashinfer/pull/4298) (the `FP16_QK_REDUCTION` path does not
compile and decodes QK logits with a value cast instead of a bit cast — the `half` `DTypeQKAccum`
branch, unreachable for bf16) and
[#4299](https://github.com/flashinfer-ai/flashinfer/issues/4299) (FA2 prefill never validates
`head_dim`; not a multiple of 16 silently returns wrong results — head_dim 64 is fine).

---

## 6. PyTorch SDPA, for comparison — CONFIRMED at tag `v2.13.0`

Downloaded to `/private/tmp/claude-501/-Users-ratish-sglang-omni/5d9d5e5e-868b-452c-b92f-ebdcfb84eacf/scratchpad/pt213/`.
Line numbers are v2.13.0's (they differ by a few lines from `main`).

### 6.1 MEM_EFFICIENT (CUTLASS fMHA)

`aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h` @ v2.13.0:

- `99   using accum_t = float;` and `105  using output_accum_t = accum_t;` — unconditional, for every
  input dtype including bf16. It is the `ElementAccumulator` of both GEMMs. CONFIRMED.
- `515-518  cutlass::Array<accum_t, kQueriesPerBlock> m_prime; s_prime; mi; out_rescale;` — row max,
  running sum and rescale are fp32. CONFIRMED.
- `1298  ... (accum_n < max_col) ? exp2f(frag[idx] - mi_row) : accum_t(0.0);` — the exponential runs
  directly on the fp32 MMA accumulator fragment. CONFIRMED.
- **Scale is applied to the fp32 logits after the GEMM, never folded into q:**
  `801  cutlass::multiplies<typename MM0::Mma::FragmentC>()(p.scale, accum);` (`p.scale` is
  `accum_t`, declared `155  accum_t scale = 0.0;`), and `1227  frag = cutlass::multiplies<Fragment>()(scaling * kLog2e, frag);`. CONFIRMED.
- **Masked-out sentinel is true `-inf`:** `642-643`, `865`, `896`, `1238`, `1324` all
  `-cutlass::platform::numeric_limits<accum_t>::infinity()`, with an explicit all-masked-row guard
  at `1178` (`mi == -inf ? 0 : mi`). CONFIRMED. No finite sentinel anywhere in the file.
- The only narrowing to `scalar_t` is of P **after** softmax, on the way into smem for the PV GEMM
  (`gemm/mma_from_smem.h`, `accumToSmem`). CONFIRMED (subagent read; consistent with the fp32 `frag`
  at `1298`).
- Caveat worth knowing if you feed SDPA an additive float mask: the bias pointer is `scalar_t`, so a
  float mask is rounded to bf16 before being added into the fp32 accumulator
  (`kernel_forward.h` bias path, `accum[idx] += bias_tensor_ref.at(...)`). Reported by subagent,
  UNVERIFIED at v2.13.0 line level.

### 6.2 FLASH_ATTENTION (FA2)

PyTorch no longer vendors the kernels: at v2.13.0
`aten/src/ATen/native/transformers/cuda/flash_attn/src/*` is **404**; `.gitmodules` declares
`third_party/flash-attention -> https://github.com/Dao-AILab/flash-attention.git`, pinned at
**`6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15`** (resolved through the git-tree API). All lines below
are from that pin's `csrc/flash_attn/src/`. CONFIRMED.

- `kernel_traits.h:26  using ElementAccum = float;` and
  `kernel_traits.h:33  MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>` — bf16 operands, fp32 accumulate.
  `acc_s` / `acc_o` are `partition_fragment_C`, i.e. fp32. CONFIRMED.
- `softmax.h:131-132  using TensorT = decltype(make_tensor<float>(...)); TensorT row_max, row_sum;` — fp32. CONFIRMED.
- `softmax.h:86  tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);` with
  `scale = params.scale_softmax_log2`. The scale enters the fp32 exp2 argument exactly as FlashInfer
  does; **q is not pre-scaled**. Host side: `flash_api.cpp:149-150  params.scale_softmax = softmax_scale; params.scale_softmax_log2 = softmax_scale * M_LOG2E;`. CONFIRMED.
- `softmax.h:157  float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);` — fp32 rescale. CONFIRMED.
- **The bf16 conversion is of P, after the exponential:** `flash_fwd_kernel.h:343-347`
  ```
  343:            ? softmax.template softmax_rescale_o</*Is_first=*/true, ...>(acc_s, acc_o, params.scale_softmax_log2)
  346:        // Convert acc_s from fp32 to fp16/bf16
  347:        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
  ```
  same ordering at `407-409`, `917-922`, `985-987`. CONFIRMED.
- **Masked-out sentinel is true `-INFINITY`:** 7 occurrences in `mask.h`, all writing `-INFINITY`
  into the fp32 `acc_s`; guards at `softmax.h:76`, `111`, `156` (`row_max == -INFINITY ? 0.f : ...`)
  and an all-masked LSE path at `softmax.h:180`. CONFIRMED. No finite sentinel.

### 6.3 Which backend torch picks

`aten/src/ATen/Context.h` default priority is flash → efficient → math (`Context.h:482` is the
`efficient_attention` entry of that list), and `can_use_flash_attention` passes for dense bf16,
head_dim 64, sm90, no mask; passing an `attn_mask` is what pushes you to MEM_EFFICIENT. CONFIRMED at
the list level. The subagent additionally reports that on recent builds
`check_prefer_cudnn_attention()` reinstalls `{cudnn, flash, efficient, math}` on sm90 with
cuDNN ≥ 9.15.1 unless `TORCH_CUDNN_SDPA_DEPRIORITIZED` is set — UNVERIFIED at v2.13.0 line level.

### 6.4 The contrast that matters

| | FlashInfer 0.6.18 FA2/FA3 | torch mem-eff | torch flash (FA2) |
|---|---|---|---|
| S = QK^T | fp32 | fp32 | fp32 |
| row max / running sum | fp32 | fp32 | fp32 |
| exp2 argument | fp32 | fp32 | fp32 |
| 16-bit rounding of S or scaled logits | **none** | none | none |
| 16-bit rounding of P (post-exp) | yes | yes | yes |
| sm_scale folded into q | no | no | no |
| **masked-out sentinel** | **finite `-5e4`** (`math.cuh:33`) | IEEE `-inf` | IEEE `-INFINITY` |
| running-max initialiser | **finite `-5e4`** (FA2, `prefill.cuh:949`) | `-inf` | from data / `-inf` |

The logit *type* is identical across all three. The only structural difference on the score path is
the sentinel: torch uses IEEE infinity, FlashInfer 0.6.18 uses a finite -50000 in the raw
(pre-`sm_scale`) domain, which is inside the live logit range for a DiT whose `q·k/8` reaches 1.3e5.
