# Slice P1: one two-token predictor pass instead of two one-token passes

## 1. Mechanism

Per decode position the predictor runs 16 passes over its 5 layers: the projected
talker hidden (cache slot 0), the projected layer-0 code embedding (slot 1), then 14
sub-steps (slots 2 to 15). The first two inputs are known before any pass runs, and
the only dependency between them is causal attention (slot 1 attends to slot 0). One
pass over both tokens reads every layer weight once instead of twice and launches the
layer kernels once.

The checkpoint's reference implementation computes it this way: qwen-tts 0.1.1 runs
the first predictor step as one two-token prefill,
`code_predictor.generate(inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1))`
(`qwen_tts/core/models/modeling_qwen3_tts.py:1671-1672`), then one token per sub-step.
omni's split into two one-token passes is the departure; P1 returns to the reference
structure.

Per replay: 17 input projections become 16, 16 layer passes become 15. Bytes: 2.65 GB
becomes 2.49 GB (157.3 MB of layers and 4.2 MB of projection less). Kernels: one pass
of 20 GEMMs plus its norm, rope, KV store, attention and residual kernels less.

Measured (bench 01, GEMMs only, CUDA graph): bs 16 5.252 to 4.699 ms, bs 1 3.626 to
3.367 ms. The saving holds at every batch size because it removes a pass, not a cost
that depends on M.

## 2. Code path today

```
Qwen3TTSModelRunner._collect_codes                    model_runner.py:176
  -> Qwen3TTSTalker.code_predictor_forward             sglang_model.py:1180
       -> _predictor_forward_graphed (replay)          :1446
       -> _code_predictor_forward_incremental          :1518   (also the captured body :1400)
            project_input(layer0_embed)                :1552
            project_input(talker_slice)                :1557   (+ clone if identity :1560)
            _predictor_forward_one_token(talker, 0)    :1568
            _predictor_forward_one_token(layer0, 1)    :1574  -> last_hidden
            15 x lm_head -> sample -> embed -> project -> one_token(cache 2..15)   :1586-1620
_predictor_forward_one_token                           :1756
  positions = _predictor_position_rows[cache_len, :B]  :1764
  per layer: norm -> _predictor_cached_self_attention :1865 -> o_proj + residual :1797 -> norm -> mlp
_predictor_cached_self_attention
  seq_len != 1 raises                                  :1876
  rope + fused KV store at _predictor_cache_slots[cache_len, :B]   :1891-1905
  SDPA over cache[:, :cache_len+1], is_causal=False    :1920 -> _predictor_gqa_attention :73
```

## 3. Code path after

```
_code_predictor_forward_incremental
  rows = stack((talker_slice, layer0_embed), dim=1).reshape(2B, H_talker)   one copy
  pair = project_input(rows)                                                one GEMM, M = 2B
  hidden = _predictor_forward_tokens(pair, batch_size=B, num_tokens=2, cache_len=0)
  last_hidden = hidden.view(B, 2, H)[:, 1:]                                 the slot 1 rows
  15 x ... _predictor_forward_tokens(new_embed, B, num_tokens=1, cache_len)  unchanged
_predictor_forward_tokens(num_tokens)
  positions = pair table [0,1]*maxB (num_tokens 2) or _predictor_position_rows row (1)
  rows = B * num_tokens; residual shape (rows, 1, H) so the fused o_proj check at :1834 holds
_predictor_cached_self_attention(num_tokens)
  cache_loc = pair slot table b*L + {0,1} (2) or _predictor_cache_slots row (1)
  q view (B, num_tokens, heads, d) -> (B, heads, num_tokens, d)
  keys cache[:, :cache_len + num_tokens]
  _predictor_gqa_attention(..., is_causal=num_tokens > 1)
```

Rows are ordered request by request (b0t0, b0t1, b1t0, ...) so each request's two
tokens are contiguous: the positions and cache slots of any batch prefix are a prefix of
two tables built once in `__init__` next to `_predictor_cache_slots` (`:967`), and the
query view is a free reshape. `is_causal=True` with a square 2 x 2 score matrix is the
standard lower triangle.

Paths that must keep working:

- Talker hidden identity projection (`project_input` returns its input when the sizes
  match, `:405`): the stack is a new tensor, so the clone at `:1560` goes.
- Rope without the fused KV store (`:1906-1912`): the copy writes two slots per request
  through the pair slot table view.
- Ascend (`_predictor_gqa_attention` `:82`): the fused NPU kernel takes no mask, so it
  stays on `num_tokens == 1`; the pair pass uses SDPA causal on every device. NPU
  performance of the pair pass is unmeasured.
- Batch invariant mode and the cutedsl GEMM backend: the o_proj fusion already steps
  aside for them (`:1811-1812`); nothing new.
- lm_head input: the slot 1 rows of the pair output are strided. sglang 0.5.19's SM90
  gemv and cutedsl bf16 backends call `x.view(-1, x.shape[-1])` on their input
  (`python/sglang/srt/layers/quantization/unquant.py:271,277`), which succeeds on the
  strided rows and hands the kernel a non-dense row stride. The rows are made dense with
  one B x 1024 copy per step.
- Qwen3-Omni's talker (`sglang_omni/models/qwen3_omni/components/talker.py:1606-1612`)
  runs the same two one-token passes. Same mechanism, separate PR after this one is
  measured.
- Predictor graphs (`:1372`): the captured body is `_code_predictor_forward_incremental`,
  so the pair pass is captured with no graph code change. Graph memory: the pair pass
  allocates 2B-row intermediates once per replay; pool size is read from the capture log.

## 4. Experiments

- P1-e1 numerics, one run (`scripts/predictor_pair_bench.py`). Record talker hiddens and
  layer-0 codes from the qwen-tts reference model's own decode on seed-tts voice clone
  requests (the predictor inputs it builds at `modeling_qwen3_tts.py:1672`), then batch
  them at bs 1, 2, 4, 8, 16. Run the chain three ways on the same inputs and
  seeds: current bf16, pair bf16, current in fp32 (weights upcast). Per sub-step: logits
  max abs and mean abs vs fp32, and code agreement vs fp32 argmax where greedy. Pass:
  pair's error to fp32 is within current's error to fp32 (per bs, per sub-step). Also
  under batch invariant mode: pair codes equal current codes.
- P1-e2 graph replay time and kernel count per bucket (1, 2, 4, 8, 12, 16), current vs
  pair, from the same script.

## 5. Gates and A/B

1. Unit tests (branch `perf/qwen3-tts-predictor-pair-pass`, TESTING.md run 04): the
   qwen3_tts suite with the helpers' new signatures, plus three contract tests: the pair
   pass equals two one-token passes (outputs and K/V cache), the fused rope KV store
   writes the pair where the copy path writes it, and the pair query attends causally.
2. P1-e1 pass.
3. Matrix A/B (base vs P1), predicted cells: decode steps shorter by the P1-e2 delta at
   each bs; TTFC shorter by one pass in the prefill step.
4. Census on B.
