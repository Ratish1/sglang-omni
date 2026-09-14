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
| row attention is SDPA on the rows scattered to the padded layout, not flashinfer | this DiT has no QK norm; on hop blocks 2 and 4 and final blocks 5 and 20 the logits reach 1.3e5, where a 16 bit logit has a spacing of 1,024 | E2 forensics at cc1d2238a: 90 of 220 calls per batch under 40 dB with flashinfer, worst 0.6 and -3.2 dB, deterministic on fresh fa2, fa3 and auto wrappers, fa2 and fa3 disagreeing with each other; SDPA bf16 vs float32 58 to 70 dB on the same tensors; CPU replay: rounding the scaled logits to bf16 reproduces it (2.4 and 1.3 dB), rounding P, the exp argument or q does not | ee699a4a5 | done; packed forward now bit identical to DiT.forward (0.0 in float64) |
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
| Flow CUDA graphs capture 26 (batch, frames) shapes, 11 to 18 s of boot | config.py FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES, stages.py FlowCudaGraphRunner | SGLang buckets by batch and pads tokens; capture by total tokens after the packed layout |
| 280 ms host floor per Flow call, 18,100 launches | solve_flow_euler_packed, 10 steps x 22 blocks | graph the packed step by total token buckets |
| upsample encoder keeps 2 .item() syncs per call | upsample_encoder.py:286,299 via mask.py:233 | Nsight sync count per call |
| HiFT recomputes the full history per hop | hift_delta, reference does the same | hop window state; E1 exactness |
| preprocessing: campplus provider, thread pools, finalize lock | request_builders, utils | per request preprocessing time at c16 |
| runaway generations, 2,048 tokens for a 5 word text | benchmark client passes max_new_tokens=2048; the contract caps at 20x text | not ours; same rows on every arm |
