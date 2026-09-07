# How every model in the tree sizes its SGLang KV pool

Question: do the other models fill the card the way Qwen3-TTS did, and is there a Higgs or Moss
pattern Qwen3-TTS should follow instead of the admission bound. Sources read on
`perf/qwen3-tts-kv-pool-admission-bound` at `8807833ca` on main `50db4a550`, sglang at tag
v0.5.18 in `/Users/ratish/sglang-worktrees/v0.5.18`. Four Opus reads, every cited line
re-read by hand before it entered this note. "Fills the card" below is the code's rule unless a
measurement is named, the only measured model is Qwen3-TTS.

## 1. The three sizing paths every model shares

```
stages.<name>.gpu_memory_fraction | total_reserve_bytes | engine.*      config/schema.py:331,342,106
   |  kv_cache_bytes excludes mem_fraction_static and max_total_tokens   schema.py:147-160
   |  total_reserve_bytes excludes gpu_memory_fraction                   schema.py:415-424
   v
process grouping by StageConfig.process, config order is load order     config/topology.py:132-143
   v
factory kwargs: gpu_memory_fraction -> total_gpu_memory_fraction        config/runtime.py:152-153
   |  injected only if the factory declares the parameter               runtime.py:179-181
   |  process_total_gpu_memory_fraction, cumulative in load order       pipeline/mp_runner.py:232-260
   |  total_reserve_bytes -> torch allocator cap for the whole process  pipeline/stage_workers.py:805-838
   |  kv_cache_bytes scoped around the one factory call                 stage_workers.py:874-878
   v
builder.build(): generation_defaults, deployment engine.* wins,
   adjust_overrides, fraction dropped under a byte budget               scheduling/engine_factory.py:123-145
   |  unset mem_fraction_static is popped, sglang derives it            sglang_backend/server_args_builder.py:84-85
   |  infra_kwargs() carries total_gpu_memory_fraction, base is {}      engine_factory.py:300-301
   v
_OmniKVCacheConfigurator._profile_available_bytes                       model_runner/sglang_model_runner.py:68
   |  kv_cache_bytes set   -> those bytes, after a free memory check     :93-120
   |  fraction set         -> total x fraction - this process's NVML use :127-147, utils/gpu_memory.py:239-240
   |  neither              -> upstream: free - pre_load_free x (1 - f)   v0.5.18 kv_cache_configurator.py:1764-1811
   v
tokens = bytes // cell size, then min with max_total_tokens              kv_cache_configurator.py:1844-1859
   v
max_num_reqs = min(max_running_requests, tokens // 2)                    kv_cache_configurator.py:1873-1886
```

The fraction sglang derives when none is set: reserve = 512 MB + 1.5 MB per activation token +
the graph reserve, floored at 10 GB above 60 GB, fraction = (card - reserve) / card
(v0.5.18 server_args.py:4956-4983). On an 80 GB card that is 0.84 to 0.87 depending on the
chunk size and the decode graph ladder.

Nothing outside the model directories derives `max_total_tokens` from anything. The grep hits
are the schema field, the prefill graph ladder cap that mirrors sglang's own
(generation_batch_policy.py:82-84), the weight sharing requirement that a follower declares
one (pipeline/weight_share.py:233-259), and the MLX pool size (mlx_model_worker.py:208-209).

## 2. Every model, one row

Processes are the default layout on the engine's GPU. "Reaches the configurator" says whether
the stage fraction becomes the process budget of the fraction path, which needs both a
`total_gpu_memory_fraction` factory parameter and an `infra_kwargs` that forwards it.

| Model | Processes on the engine's GPU | Engine fraction | Reaches the configurator | Running x context | Token cap | Default fills the card |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-TTS, main | 1, all three stages `process="pipeline"` (qwen3_tts/config.py:56-76) | builder 0.85 | no | 16 x 8192 = 131072 | none | yes, measured: 589142 tokens, 75.2 GB at ready |
| Qwen3-TTS, this slice | same | sglang derived | no | same | 131072 derived | no, measured: 25.0 GB at ready |
| Higgs TTS | 2, `tts_frontend` and `pipeline` with engine and vocoder, fractions 0.03, 0.85, 0.10 declared (higgs_tts/config.py:37-79) | builder 0.85, or the stage fraction (engine_builder.py:81-85) | no, the builder has no `infra_kwargs` | 64 x 4096 = 262144 (stages.py:468, engine_builder.py:29) | none | yes, by the rule |
| Fun CosyVoice3 | 1 | 0.85 (engine_builder.py:63) | no | 32 x 4096 = 131072 | none | yes, by the rule |
| Voxtral TTS | 1 | 0.85 (pipeline/engine_builder.py:40) | no | 16 x 8192 = 131072 | none | yes, by the rule |
| FishAudio S2 Pro | 2, preprocessing apart, vocoder with the engine | 0.85, 0.75 on NPU (engine_builder.py:91, 103) | no | 64 x 4096 = 262144 | none | yes, by the rule |
| Whisper ASR | 1 | 0.85 (whisper_asr/stages.py:16) | no | 64 x audio derived | none | yes, by the rule |
| MOSS Transcribe Diarize | 1 | 0.80 (stages.py:79) | no | 16 x checkpoint, 131072 fallback (stages.py:57-60) | none | yes, by the rule |
| Qwen3 ASR | 1 | unset, sglang derives (stages.py:17) | no | 64 x audio derived | none. The MPS branch pins 2048 by hand for 1 running x 2048 context (engine_builder.py:164-172) | yes, by the rule |
| ArkASR, Fun ASR | 1 | unset, sglang derives | no | 32 and 64 x audio derived | none | yes, by the rule |
| LLaDA2 Uni | 1, image encoder built before the engine | unset, sglang derives (stages.py:95-108) | no | 16 x 8192 | none | yes, by the rule |
| Ming Omni thinker | its own process, encoders in two more on the same GPU | unset, sglang derives (ming_omni/stages.py:284-320) | no, the factory has no fraction parameter | 16 x 8192 | none | yes, by the rule |
| dots.tts | 1 | 0.20 (engine_builder.py:61) | no | 16 x 2048 = 32768 | none | no, 80 percent slack by a constant |
| Zonos2 | 1 | 0.5 (engine_builder.py:131) | no | 16 x checkpoint | none | no, by a constant |
| MiniMax Music3 | 2, AR and DiT with DAV | 0.50 (engine_builder.py:60) | no | 32 x 10240 = 327680 | none | no, by a constant |
| MOSS TTS | 2, the vocoder apart, fractions 0.10, 0.72, 0.18 (moss_tts/config.py:20-22) | 0.72 | yes, the process sum 0.82 (stages.py:508-516, engine_builder.py:29-36) | 16 x checkpoint, 8192 fallback | none | no, 0.82 of the card minus the process's own use |
| MOSS TTS Local | 2, fractions 0.15, 0.67, 0.18 (moss_tts_local/config.py:24-26) | 0.67 | yes, 0.82, falls back to upstream when NVML cannot see the process (engine_builder.py:101-108) | 16 x 32768 = 524288 | none | no, 0.82 minus the process's use |
| Qwen3-Omni thinker | its own process, encoders and code2wav in three more (config.py:290-298) | unset, sglang derives, minus the 0.05 encoder reserve (stages.py:1007, 143-153) | with a config fraction only: 0.55 becomes 0.50 in the H100 fp8 config | 64 x 8192 = 524288 | none | yes by default, bounded by the fraction in the colocated configs |
| Qwen3-Omni talker | its own process, GPU 1 by default | unset, sglang derives | with a config fraction, 0.12 | 32 x 32768 = 1048576 | none | yes by default |
| Ming TTS | 1 by default (ming_tts/config.py:244-270), 3 with the example yaml | unset by default, the yaml pins 0.75 | yes with the yaml, 0.72 (engine_builder.py:147-152) | 8 x 8192 = 65536 | none | yes by default, 0.72 with the yaml |
| Audar TTS | 1 | no SGLang engine, llama.cpp with `n_ctx` 4096 (stages.py:243-250) | n/a | n/a | n/a | n/a |

Three facts fall out of the table.

1. No model derives its pool from admission. Eleven of them fill the card by the same rule
   Qwen3-TTS used, a fixed fraction of 0.80 to 0.85 or the fraction sglang derives. Three leave
   room by a smaller constant, 0.20, 0.5 and 0.50, that names no measurement. The rest bound the
   engine's process by a stage fraction, which is also a constant.
2. Every model with a vocoder, tokenizer or encoder in the engine's process pays for it out of
   the slack the fraction leaves, or by loading it before the engine so the profile sees it.
   Qwen3-TTS loads the tokenizer after the pool, in `setup_model`, and the vocoder after that,
   which is why the 0.85 rule ran the card to 80.6 GB.
3. The one existing hand written instance of the admission bound is qwen3_asr's MPS branch:
   `max_running_requests` 1, `context_length` 2048, `max_total_tokens` 2048.

## 3. The multi replica recipes pin a token cap by hand

Every `examples/mps_dp/configs/*.yaml` sets `max_total_tokens` next to a fraction: Higgs 100000
at 0.85 for three replicas on an H100 and 30000 for eight on an H200, MOSS delay 30000 at 0.30
under a stage fraction 0.37, MOSS local 30000 at 0.30 under 0.45, Qwen3 ASR 40000 at 0.20,
Whisper 40000, Fun ASR 30000, MOSS TD 40000. `launch.sh` refuses more than one replica without
a cap or a byte budget (launch.sh:405-407) and checks each replica's `#tokens:` line against
it (launch.sh:562-579). `examples/mps_dp/config.py:154-166` warns that without
`total_reserve_bytes` the replicas' total footprint is not validated, only the pool. Against
the admission bounds, the Higgs cap is 0.38 of 262144 and the MOSS local cap is 0.06 of 524288,
numbers the headers describe as validated on one card, not derived.

## 4. What this means for Qwen3-TTS

The Higgs pattern is the fraction constant plus a hand picked `max_total_tokens` per
deployment. Its per stage fractions satisfy the placement validator and do not reach the KV
sizing. The slice uses the same sglang mechanism, `max_total_tokens` applied as a minimum over
the profiled capacity, and replaces the hand picked number with the deployment's own admission
cap times the context length. That is the tree's mechanism with its input derived instead of
guessed.

The MOSS pattern is the multi process regime of plan 05: the engine's process is budgeted as a
fraction of the card and measured against its own NVML use. It needs two things Qwen3-TTS does
not have. A fraction constant, 0.72 or 0.82, that no measurement pins for this model. And a
process split that MOSS measured as a 70 percent gain for its Python heavy vocoder
(moss_tts/config.py:51-53), where Higgs measured the opposite for its layout
(higgs_tts/config.py:68-71) and Qwen3-TTS keeps one process for its vocoder CUDA graphs and
decode overlap. The fraction path is reachable without a split, a `gpu_memory_fraction` on
`tts_engine` plus an `infra_kwargs` forwarding it, but the constant would encode a guess about
what the tokenizer and vocoder need, which is the thing the snapshot measures directly.

What the slice does not close, and what no model in the tree closes: when running times
context exceeds the card, sglang clamps the cap to the profiled value and the process is back
in the fraction regime. At 128 running that left about 2 GB of headroom in the archive. The
fix that generalizes is sizing the pool after the process's other components are accounted,
plan 05 decision 4, or a declared budget per deployment.

A follow up candidate, not part of this slice: lift the derivation into `TtsEngineBuilder` so
every TTS builder gets the admission bound cap by default, Higgs at 262144, CosyVoice3 and
Voxtral at 131072, FishAudio at 262144, each measured on its own benchmark before it ships.
Whether the cap binds for a model depends on its cell size, unknown here for all but Qwen3-TTS.

## 5. Validation tasks

- The cell size per model, from each checkpoint, to turn the fraction rule into tokens and
  say whether the admission cap would bind for Higgs, CosyVoice3, Voxtral and FishAudio.
- Whether `adjust_mem_fraction_for_vlm` (v0.5.18 server_args.py:4990-4996) lowers the derived
  fraction for Qwen3-Omni and Ming, which depends on `is_multimodal` for those checkpoints.
- A serve log of Higgs and MOSS at default settings, to replace "by the rule" with a number.
- The `total_reserve_bytes` docstring names MPS preflight arithmetic, and nothing under
  `sglang_omni/mps/` reads it. Whether that sentence describes a removed path.
