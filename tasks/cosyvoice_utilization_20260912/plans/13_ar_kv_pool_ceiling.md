# Plan 13: bound the AR KV pool by the requests the engine admits

Roadmap row B1. Written 2026-09-17 from whole file reads of upstream main `8bd3930e2`, pinned
SGLang `v0.5.19`, and the Qwen3-TTS branch `perf/qwen3-tts-kv-pool-admission-bound` (never opened
as a PR; its mechanism is reused, its base is stale). Every claim carries a line. Gates run on the
RTX 4090 D; there is no H100. What the 4090 cannot settle is listed as an open validation task,
not assumed.

## 1. The defect, in the code

| fact | where |
|---|---|
| the CUDA branch of `generation_defaults` pins `mem_fraction_static: 0.85` and sets no token bound; the MLX and MPS branches of the same function set `max_total_tokens: self.context_length` | `models/fun_cosyvoice3/engine_builder.py:114-126`, `:93-94`, `:107-108` |
| the fraction never bounds the pool: the budget is free memory after weights minus a slack of the pre-load free memory times one minus the fraction, and the pool takes the rest | sglang `mem_cache/kv_cache_configurator.py:2007, 2021, 2043` |
| the engine cannot address more than `max_running_requests x context_length` tokens of live KV: 32 x 4096 = 131,072 | `engine_builder.py:29, :115`; scheduler bound `scheduling/omni_scheduler.py:332-336`, `allow_auto_truncate=False` at `:1288` |
| admission reserves prompt + min(max_new_tokens, 4096) + one page per candidate; 32 requests need 75,168 tokens | sglang `managers/schedule_policy.py:1195-1212`, `:845-862` |
| a user `max_total_tokens` is applied as `min(profiled, user)` after profiling; running requests become `min(requested, pool // 2)` | sglang `kv_cache_configurator.py:2079-2089`, `:2112-2115` |
| measured pools: 4090 854,232 tokens (9.78 GB) with peak occupancy 1,725 tokens at c8; H100 5,423,957 tokens (62.08 GB) | `../MEMORY_20260917.md:13-20`; `artifacts/run-de47ace7e/.../serve.log:50-51` |
| the pool is elastic in the wrong direction: Flow graphs off gave 16.3 GB of pool, not headroom | `../MEMORY_20260917.md:33-36` |
| on 24 GB the 2 GB slack the fraction leaves fails requests: 1 of 32 at c4, 9 of 32 at c8, cuBLAS handle in the ONNX tokenizer | `../slices/11_2_RESULT_20260917.md`, `request_builders.py:296`, `utils.py:92` |
| 4096 is omni's window, set by #1331, not the model's (32,768); the largest admitted request is about 3,000 tokens | `git log -L 29,29:engine_builder.py`; official `frontend.py:97`, `llm.py:497-498` |

## 2. The abstraction, as the codebase already has it

Omni's runner sizes the pool in three byte modes (`model_runner/sglang_model_runner.py:117-171`):
a declared `engine.kv_cache_bytes` taken as the pool, a stage `total_gpu_memory_fraction` for
colocated engines, and upstream's free minus slack. A token ceiling is not a fourth sizing mode:
it is SGLang's own post profiling cap, exposed to operators as `engine.max_total_tokens`
(`config/schema.py:156`) and declared mutually exclusive with the byte budget (`:190-196`). The
builder's job is to supply that ceiling from the two contract numbers it already owns when the
operator supplied neither. Precedence, all existing:

```text
engine.kv_cache_bytes (operator)      -> byte mode, no ceiling added      stage_kv_budget.py:35-88
engine.max_total_tokens (operator)    -> kept as given                    setdefault
neither                               -> max_running_requests x context_length
mem_fraction_static                   -> left to SGLang's reserve arithmetic   memory_hook.py:231-267
```

Leaving the fraction to SGLang is the second half. Its auto value reserves 512 MB plus 1.5 MB per
activation token plus the graph reserve, floored at 10 GB on cards over 60 GB (`memory_hook.py:255-267`);
that is the slack the pinned 0.85 was approximating by hand, and it scales with the card.

No helper, no class, no new file. Two hooks the base already declares and Qwen3-TTS already
overrides: `adjust_overrides` (`scheduling/engine_factory.py:112, :273`) and `post_scheduler_setup`
(`:236, :412`).

## 3. Code plan

`sglang_omni/models/fun_cosyvoice3/engine_builder.py`:

1. Module level imports: `from sglang.srt.runtime_context import get_model, get_schedule` and
   `from sglang_omni.scheduling.stage_kv_budget import peek_stage_kv_cache_bytes` (as the Qwen3-TTS
   builder on main imports them; check its import block and match).
2. `generation_defaults`, CUDA branch: delete `"mem_fraction_static": 0.85`. Nothing else moves.
   The MLX and MPS branches keep their `max_total_tokens: self.context_length` (running requests 1).
3. Add `adjust_overrides(self, overrides: dict[str, Any]) -> None`:
   ```python
   def adjust_overrides(self, overrides: dict[str, Any]) -> None:
       # note(ratish): the pool covers what admission can commit, the running
       # cap times the context length. A stage byte budget sizes it instead.
       if peek_stage_kv_cache_bytes() is None:
           overrides.setdefault(
               "max_total_tokens",
               int(overrides["max_running_requests"]) * self.context_length,
           )
   ```
   `overrides["max_running_requests"]` is always present: every branch of `generation_defaults`
   sets it and `build_generation_batch_overrides` merges defaults before user overrides
   (`engine_factory.py:108-111`). `setdefault` keeps an operator `engine.max_total_tokens` and the
   MLX and MPS values.
4. Add `post_scheduler_setup(self, scheduler: Any, model_runner: Any) -> None` mirroring the
   Qwen3-TTS one on main (`models/qwen3_tts/engine_builder.py:203-215`) with the model name
   changed, returning first when `use_mlx()` because the MLX worker sizes no SGLang pool
   (`scheduling/bootstrap.py:163-167`); `use_mlx` is imported function locally in this file's other
   hooks, follow that. Percent style log, the repo's stdlib logging rule.

Nothing else. `max_prefill_tokens: 4096` stays a literal (out of scope). The prefill graph
`max_bs` cap that `max_total_tokens` also applies (`scheduling/generation_batch_policy.py:86-88`,
sglang `memory_hook.py:206-210`) is inert at 131,072, above any chunk size.

`tests/unit_test/fun_cosyvoice3/test_engine_builder.py`, next to the existing MLX and MPS tests,
mirroring the Qwen3-TTS branch tests (`git -C .worktrees/qwen3-tts-kv-pool diff upstream/main...HEAD
-- tests/unit_test/qwen3_tts/test_pipeline.py`) and the log test on main
(`tests/unit_test/qwen3_tts/`, grep `post_scheduler_setup`):

- the CUDA defaults carry no `mem_fraction_static`;
- `adjust_overrides` sets `max_total_tokens == max_running_requests * context_length` for
  (32, 4096) and for an operator override of running requests;
- an existing `max_total_tokens` in the overrides is kept;
- no ceiling is added under an active `stage_kv_cache_budget`, and the budget is still consumable;
- the MLX and MPS defaults are unchanged (the existing tests already pin them; run them);
- the startup line reports pool tokens, GiB and the bound; the MLX path logs nothing.

Fakes model the real shapes: a scheduler namespace with `max_total_num_tokens` and a
`tp_worker.model_runner.token_to_kv_pool.get_kv_size_bytes()` returning two ints, `get_schedule`
and `get_model` patched at the module the builder imports them from. No comments in tests.

## 4. Gates, all on the 4090

| gate | what | pass |
|---|---|---|
| G-unit | `pytest tests/unit_test/fun_cosyvoice3 -q` on the box | all pass |
| G-boot | one boot, default launch; read `max_total_num_tokens=` in the scheduler line, the new startup line, "avail mem" after the pool | 131,072 tokens; free memory after the pool about 20 GB on the 24 GB card instead of 1.36 |
| G-c1 | seeded c1, 16 samples, main `8bd3930e2` against the branch, G1 identity gate (`stage2/g1_stream_identity.sh`) | byte identical on the samples the control reproduces; the pool size touches no arithmetic |
| G-c8 | unseeded c8, 32 samples, both arms | main 9 failures of 32 (measured), branch 0; from both logs: radix hit counts, "KV cache pool is full" lines, rejections at the window |
| G-c16 | unseeded c16, the whole English split, branch only | completes; this is the new B for slices 1.2 and plan 12 |

Falsifiers, read from the same runs: a nonzero retract line on the branch, a request the branch
rejects that main admits, a radix hit rate on main that is not near zero, a c1 delta outside 2 percent.

## 5. What the 4090 cannot settle

The H100 pair (c16 streaming and buffered, main against the branch, throughput within 2 percent,
near zero radix hits on both) is a validation task for whoever has the card; the reviewer of #2224
ran on H200s. The code read predicts identity: no admission, retraction, graph capture or
concurrency decision reads the pool beyond 131,072 for this model. Until it is measured, the PR
body states the prediction and the lines it rests on, not a measured H100 number.

## 6. PR

Branch `slice/cosyvoice-b1-kv-pool-ceiling` from upstream main. One commit: "bound the ar kv pool
by the requests the engine admits". Title "[Fun-CosyVoice3] Bound the AR KV pool by the requests
the engine admits". Body: mechanism (section 2 in three sentences), changes (file by file), the
4090 gates with numbers, the H100 prediction with its lines, and the operator knobs that override it.

## 7. Related, not in this PR

- The 2.1 census must be retaken at main `8bd3930e2` with per sample pairing (72 of 1,088 paired
  in the seeded run; three runaways on main against one), and the readout carries the paired table.
- Sizing the vocoder side budgets from free memory after the ceiling, the hop cache pool of 1.2
  and the graph table of plan 12, is what turns the freed memory into throughput on big cards.
