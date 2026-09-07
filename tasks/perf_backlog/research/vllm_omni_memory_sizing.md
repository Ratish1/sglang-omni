# How vLLM and vLLM-omni size the KV pool

Read on `/Users/ratish/vllm` at v0.26.1rc0-130-ge04a30a77 and `/Users/ratish/vllm-omni` at
v0.28.0-82-gb78c31eda, whole functions, on Sep 7 2026. The question was whether the admission
bound slice is how a serving engine should size memory, or a shortcut.

## 1. vLLM core

- `requested = total x gpu_memory_utilization`, refused at startup if free memory is below it
  (`vllm/v1/worker/utils.py:398-418`). The fraction is a budget for the process, not a guess
  about need.
- A profile run at the maximum batch shape, `max_num_batched_tokens` and `max_num_seqs`, with
  dummy multimodal inputs of the maximum feature size, measures the process's need: weights,
  the torch peak during the run, and the non-torch increase since the process started
  (`vllm/utils/mem_utils.py:234-330`, `gpu_worker.py:500-503`).
- `KV = requested - (weights + peak activation + non torch) - cudagraph estimate`
  (`gpu_worker.py:544-548`). `kv_cache_memory_bytes` replaces the whole computation
  (`gpu_worker.py:474-497`).
- The startup log prints the exact `--kv-cache-memory` that would fit the requested budget and
  the one that would fill the card (`gpu_worker.py:735-791`).
- `max_num_seqs x max_model_len` is not a sizing input. A pool smaller than what admitted
  requests need is handled by preemption.

## 2. vLLM-omni

- One OS process per stage replica (`vllm_omni/engine/stage_engine_core_proc_manager.py:121`).
- Every stage carries its own `gpu_memory_utilization` in the deploy yaml, and admission
  refuses a local stage without one (`vllm_omni/engine/stage_admission.py:207-264`).
- The same profile run accounting per stage, taken device wide under a device lock so each
  measurement is quiescent. NVML per process estimation was removed in favour of it
  (`vllm_omni/worker/base.py:91-137`).
- An admission ledger per device: the sum of `utilization x capacity` over the stages on it,
  plus 2 GiB per stage that captures graphs, plus an external reserve and a safety margin,
  must fit the capacity (`stage_admission.py:151-155, 181-185`). The 2 GiB is a deliberate
  constant, "intentionally over-reserves rather than measure per-model"
  (`stage_admission.py:41-44, 105-113`).
- Qwen3-TTS default: two stages on device 0, the talker at 0.3 with 64 sequences and
  `max_model_len` 4096, code2wav at 0.3 with 64 sequences (`vllm_omni/deploy/qwen3_tts.yaml`).
  The talker process holds the speaker encoder and the speech tokenizer encoder as parts of
  the model (`qwen3_tts_talker.py:388, 471-484`), so the profile run sees their weights and
  their activations. Higgs: 0.6 and 0.25 on one device. MOSS nano: 0.3.

## 3. What that says about the sglang-omni side

Three ingredients, and which ones exist here.

| Ingredient | vLLM-omni | sglang | sglang-omni |
| --- | --- | --- | --- |
| A per process budget on the card | `gpu_memory_utilization`, mandatory, summed at admission | `mem_fraction_static` is a slack coefficient over free memory, not a budget | `gpu_memory_fraction` is a budget on the process path only, `total_reserve_bytes` is a torch cap, placement sums fractions |
| A measured need before the pool | profile run at the max shape, every component in the process | none, a formula: 512 MB + 1.5 MB per activation token + graph reserve | NVML usage at pool time on the process path, which sees earlier stages' weights and no activation peak. Qwen3-TTS loads its tokenizer and vocoder after the pool |
| Pool = budget − need | yes | no, pool = free − slack | only under a byte budget |

The admission bound is not a sizing input anywhere in vLLM. It is not needed once the pool
is budget minus measured need. The slice uses it because that measured need does not exist
in this tree yet, it is the bound that costs nothing on the corpus, and a deployment can
replace it with `max_total_tokens` or `engine.kv_cache_bytes`.

The design that matches vLLM's, at the engine factory seam so every model inherits it: a per
process budget from the stage config, the non LLM components constructed or probed at their
maximum shape before the pool, the pool sized from the remainder, and a startup line that
prints the byte budget that fit. That is plan 05 decision 4 with the missing piece named, the
peak probe.
