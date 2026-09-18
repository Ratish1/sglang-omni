# Qwen3-TTS 1.7B Base on the 4090: every engine feature, resolved state and reason

Level 3 of the top-down order (model, architecture, framework features, PyTorch code,
kernels). Source for the resolved state: run04 `record_serve.log` (the shipped default
config at upstream 144bd6399) and the code cited. A feature that is off is listed with
the reason the code gives, and what we do about it.

| feature | resolved state | where decided | reason given | action |
| --- | --- | --- | --- | --- |
| Talker decode CUDA graph | on, full backend, bs 1, 2, 4, 8, 12, 16 (captured 1.90 s, 0.09 GB) | SGLang, `max_running_requests` 16 | | ledger per bs (run 07) |
| Talker prefill CUDA graph (breakable) | off: "Disable prefill CUDA graph because ... prefill.backend='disabled'" | `engine_builder.py:113-123` | scoped to CustomVoice, Base "until measured" (#1900) | slice PF: audited correct against SGLang 0.5.19, enabled for every checkpoint, run 07 arm `pf` |
| Overlap scheduling | off (`disable_overlap_schedule: True`) | `engine_builder.py:106` | omni runs its own loop; the Talker opts out of lookahead (`model_runner.py:120-124`: the lookahead hooks do not run the codec collect); talker lookahead PRs #1320 / #1204 held on a measured regression | read the host share of a step from the run 07 ledgers (host loop outside run_batch at bs 1 and 16) before reopening |
| torch.compile, Talker | off, and rejected if requested (`adjust_overrides`) | `engine_builder.py:185-187` | "Qwen3-TTS torch.compile is not supported" | none now |
| torch.compile, vocoder | only the steady width 8 (4 keys) | `streaming_vocoder.py:976,1021` | "every other width stays eager" | V2: all widths, dynamic compile (READOUT_05 section 3) |
| Attention backend | flashinfer (prefill and decode) | SGLang default for sm89 | FA3 needs sm90 | none |
| Sampling backend (layer-0 code) | pytorch | `engine_builder.py:105` | seeded sampling contract | none |
| Radix prefix cache | on (serve.log `#cached-token: 6` per request after the first) | SGLang default | | none; the breakable graph replays prefix hits on CUDA (PF audit finding 4) |
| Predictor CUDA graphs | on, 11 graphs for the two sampling signatures, 1.9 s | `sglang_model.py:1320` | | P1, P2 change the captured body |
| Reference encoder CUDA graphs | on, buckets 32 to 256 frames | `reference_encoder_cuda_graph.py` | | run 07 prefill ledgers show its time |
| Vocoder CUDA graphs | cold (widths 1, 2), window (1 to 64), warm x2 (1 to 8); batch buckets 1, 2, 4, 8 | `streaming_vocoder.py:914-1029` | | V1, V2 change the captured body |
| Fused SnakeBeta | off (`fused_snake_activation` default False, nothing enables it) | `stages.py:298` | none recorded; envelope limited to one checkpoint's channels, B <= 8 | superseded by compiling every width (inductor fuses the activation) |
| Prefill coalescing | off (`prefill_coalesce_requests` 0) | `engine_builder.py:62` | | none |
| Memory | `mem_fraction_static` 0.85: KV 114,384 tokens, 12.22 GiB; 2.44 GB free after decode capture | `engine_builder.py:103` | | sized from run 07 `mem.csv` peaks; PF adds prefill graph capture memory |

Checks still open at this level (answered by run 07): the decode step's host share at bs
1 (overlap), prefill replay vs eager counts with PF (`cuda graph: True/False` per
prefill batch in serve.log), and peak memory per boot.
