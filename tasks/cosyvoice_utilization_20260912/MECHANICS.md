# Fun-CosyVoice3 mechanics ledger

One line per mechanism: the code path, the origin of its cost, the evidence, the commit, the status.
Origins are named by the line that pays, never by the symptom. Updated with every commit.

## Serving loop, streaming vocoder (sglang_omni/scheduling, models/fun_cosyvoice3)

| mechanism | origin | evidence | commit | status |
|---|---|---|---|---|
| loop drains the inbox, then runs one step | streaming_simple_scheduler.py:138-145 | drain is non blocking, bounded by the AR token rate | 3b40bad7b | done, WER 6.82 to 1.38 |
| started streams ranked by playback slack, new by age | streaming_vocoder.py select_step_participants | replay test over the recorded c16 inbox | 4c4d54750 | done |
| finals run as steps through the packed non streaming call | 4a8d6f8c6, leftover_batch | 1,088 serial native finals were 354 s of a 530 s run | 4a8d6f8c6 | done |
| every hop in one packed causal call | 138c41980, hop_batch | native singleton call 280 ms host per 70 to 90 ms GPU | ad181263b | done |
| rows packed along the sequence, attention per row | packed_dit.py; the padded adapter paid rows times widest row | E2 at 16ade9019: packed vs padded 90 dB hops, 0.0 finals (fp32); model: runaway 2,048 token rows stalled 135 s of 359 s | cd2e42a9b | pending E2 rerun, c16 run |
| row attention is SDPA on the rows scattered to the padded layout, not flashinfer | flashinfer 0.6.18 uses a finite minus 5e4 as minus infinity in the raw logit domain (math.cuh:33, prefill.cuh:350 and 949, variant_helper.cuh:86); this DiT has no QK norm and whole query rows sit below it on hop blocks 2 and 4 and final blocks 5 and 20, so those rows return zeros | E2 forensics at cc1d2238a: 90 of 220 calls per batch under 40 dB, worst 0.6 and -3.2 dB, deterministic on fresh fa2, fa3 and auto wrappers; CPU replay of the sentinel semantics on the saved tensors: 0.6 dB hop (observed 0.6), 3.7 dB final (observed fresh fa2 3.7); 81 and 53 percent of query rows below the sentinel; fixed upstream in flashinfer PR #4401 after the SGLang pin | ee699a4a5 | done; packed forward bit identical to DiT.forward (0.0 in float64) |
| flashinfer stale mask buffer across plans | prefill.py plan keeps the last mask buffer, run uses it whenever set | E2 at 16ade9019 finals 16 dB after a masked hop plan | withdrawn with flashinfer | recorded for any future flashinfer use |
| chunk mask built without the host sync | stages.py _patch_chunk_mask | 10 of 12 .item() drains per Flow call | 33c8b2731 | done |
| empty row check with any, not a summed int64 copy | mask.py:233 upstream; our patch kept the sum | 2.29 GiB alloc in the OOM traceback; profiler: 8 bytes per element | af843811b | done, OOM 0 |
| HiFT constants resident (window, sine table, f0 float64) | generator.py:226,310 slice then copy per call | 4 of 83 syncs per HiFT call; costs 259 MB of pool | c03d7c226 | done, cost recorded |
| HiFT causal conv cache on the device | CausalConv1d cache built on CPU | 78 of 83 syncs per HiFT call | 9b407e5bc | done |
| vocoder before engine, weights and ONNX before the pool | kv_cache_configurator.py:2000 reads raw free memory | pool slack 11.1 GB either way; keeps 3.4 GB of vocoder out of it | 3e4d33862, 3ff6c4268 | done, separable |
| one hop and one final warmed before readiness | sibling vocoders warm in the factory | flashinfer kernel variants load at boot; boot gap unchanged, 14.6 s vs 18.3 s | 58d3c56a0 | done |

## Withdrawn, and why

| mechanism | why withdrawn |
|---|---|
| KV pool cap from the admission bound | stops preallocation, wastes HBM; the vocoder's working set was the real question |
| warmup at cap derived shapes | shapes defined by a cap the step no longer has; negative token count at small caps |
| step padding rules: 25 percent budget, boot measured floor and slope, 3,500 frame budget | each encodes a host to GPU ratio, pins to one GPU; the padded layout was the defect |

## Open, with the mechanism to check first

| item | origin to read | expected evidence |
|---|---|---|
| Flow CUDA graphs capture 55 (batch, frames) shapes for buffered calls only | config.py:19-75, stages.py:2054-2064 | G0 at 2eefbc476: 31.5 s boot, 142 MiB; ledger c16 buffered hits 37 of 204; hop and final calls never replay; replay over eager 1.9 to 2.6x at 1 row, 1.06 to 1.19x at 4 or more rows |
| eager Flow call launches 16.3k to 17.3k kernels plus 1.8k cuBLAS | solve_flow_euler_packed, 10 steps x 22 blocks | V0 and G0 at 2eefbc476; replay 47 to 65 launches |
| every Flow call issues 7 + 4 rows pageable host to device copies, each a stream sync | stages.py:198, 206, 209, 222, 226, 608; packed_dit.py:34, 38 (twice), 89; buffered stages.py:643 | V0 at 2eefbc476: 11 at 1 row, 39 at 8 rows; the upsample encoder origin of the earlier row was wrong |
| every hop recomputes the prompt and all earlier frames, then drops them | reference model.py:436, streaming.py hop math | E5 at 2eefbc476: prefix bit identical in float64 and bf16, cached hop exact in float64; ledger c16: cache runs 49.1 percent of hop frames, 67.9 percent with finals |
| row attention and conv position embed pay rows x widest | packed_dit.py:97-109, 161-163 | ledger c16: hops with rows x widest / total >= 3 take 1,348 ms p50 against 542 ms; E6: sgl_kernel FA3 varlen and page table at 53.3 dB min, production band 54.0, fastest |
| HiFT recomputes the full history per hop | hift_delta, reference does the same | hop window state; E1 exactness |
| preprocessing: campplus provider, thread pools, finalize lock | request_builders, utils | per request preprocessing time at c16 |
| runaway generations, 2,048 tokens for a 5 word text | benchmark client passes max_new_tokens=2048; the contract caps at 20x text | not ours; same rows on every arm |
| F4: ranking is started + unstarted, no aging for a new stream once runnable hops exceed the batch of 16; validated at c16 only | select_step_participants, commit "batch every runnable hop regardless of its token window" (PR 2) | c32 streaming A/B main vs stack head: first audio p95, C50; if it bites, one slack order with an unstarted stream's slack as minus its wait |
| F11: chunk collector pulls parked messages first, a batch is cut at a non chunk between two parked chunks; reaches dots.tts (batch 4) and MOSS-TTS Local (batch 8), Ming pinned to 1 | _collect_stream_chunk_batch, streaming_simple_scheduler.py:346 (PR 1) | MOSS-TTS Local streaming c16 main vs PR 1: req/s 6.93 to 12.87, TTFP p95 0.633 to 0.614 s, 200/200 both; dots.tts unmeasured |
