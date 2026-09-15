# Component profiling guide, Fun-CosyVoice3

How the stage 1 profile of 2026-09-15 was built, run and checked, so the next profile (a later
tree, or another TTS model) is built the same way. Scripts: `../stage1/`. First run: tree
`cc85ddaa9`, H100, archive `artifacts/stage1-20260915T151715Z.tar.gz`. Results and ranking:
`../ROADMAP_20260915.md`, section "Stage 1 component profile".

## 1. What it answers

For every call the serving pipeline makes: how long it takes at the shapes the workload forms,
whether that time is host dispatch, host blocked or device work, and which Python range and which
kernels own the device work. Concurrency runs answer none of this cleanly (calls interleave on one
GPU), so they are kept for the final A/B.

```text
request = reference encode + text prep              preprocessing
        + one prefill + one decode step per token    tts_engine (AR)
        + hops and a final (Flow, then HiFT)         vocoder
        + queueing in front of each
```

## 2. Cost model

```text
wall        = host dispatch (Python, kernel launches)
            + host blocked (stream, device and event syncs, blocking copies)
            + device tail after the last launch
device busy = union of the device intervals this call launched (never their sum)
```

| regime | reading | what moves it |
|---|---|---|
| launch bound | busy / wall well below 1, many launches | fewer launches: graph replay, fusion, dead work removed |
| sync bound | busy / wall below 1, blocked ms large | remove the transfer or the read |
| device bound | busy / wall near 1 | less device work: layout without padding, recompute removed, kernels, fusion |

A call's regime changes with batch: the Flow hop is launch bound at 1 row (0.41) and device bound at
16 rows (0.87 to 0.99). So every call is measured at batch 1 and at the batch the workload forms.

## 3. The harness

```text
profile_components.py / profile_ar.py
  build the component exactly as serving does
  for each operating point:
    trace_ledger.measure(call, prepare=...)
      warmup x2          prepare(); call(); synchronize
      timed x5           prepare(); synchronize; t0; call(); synchronize; t1   -> wall median, min, max
      warm_profiler()    one throwaway CUDA profiling session per process (CUPTI init)
      profiled x1        prepare(); ranges on; torch.profiler(CPU, CUDA):
                           record_function("call:<point>") around call()
                         synchronize (outside the range, inside the profile)
      export Chrome trace -> parse_trace -> ledger fields -> gzip trace
  write <component>.json (every field), <component>.md (tables), traces/<point>.trace.json.gz
```

- `prepare` runs before every run, outside the timed and profiled window. It resets state a call
  consumes (a prefill admits requests; the next repeat needs the same waiting set).
- Wall is always the uninstrumented median. The profiled run carries range and hook overhead (up to
  2.4 times wall on a 19,000 launch call), so its host time is never used as a timing.

Ranges, active only in the profiled run:

| label | set by | covers |
|---|---|---|
| `call:<point>` | measure | the whole call |
| `fn:<Owner>.<name>` | function_ranges, wraps a module global or class attribute and restores it | scheduler, runner, packing, attention wrappers, graph runners, HiFT STFT |
| `mod:<root>.<path with .* for indices>` | global module forward pre and post hooks | every module forward under the given roots |

## 4. Attribution rules

Read from the exported Chrome trace; nothing else of its schema is assumed.

1. The call is the `user_annotation` event named `call:<point>` (the CPU side; its
   `gpu_user_annotation` copy is on the device timeline and is not a host bound).
2. Runtime events are `cuda_runtime` and `cuda_driver` events on the call's thread inside it. cuBLAS
   and cuDNN launch through the driver API (`cuLaunchKernel`); reading only `cuda_runtime` drops their
   kernels (1,811 per Flow call here).
3. A device event (`kernel`, `gpu_memcpy`, `gpu_memset`) belongs to the runtime event with the same
   `args.correlation`, even when it runs after the host range ends (the device tail).
4. A runtime event belongs to the innermost range on its thread containing it (stack sweep).
5. Busy is the union of device intervals. Launches match `LaunchKernel` or `GraphLaunch`; syncs are
   `cudaStreamSynchronize`, `cudaDeviceSynchronize`, `cudaEventSynchronize`; blocking copies are
   `cudaMemcpy`.

Ledger fields: wall median / min / max, device busy ms and share of wall, launches, graph launches,
syncs, blocking copies, blocked ms, device events without a launch in the call, the trace category
inventory, per range (launches, self and inclusive device ms, pointwise kernels, syncs, blocked ms),
top kernels with owning ranges, and ranges whose instances launch an identical kernel sequence.

## 5. Operating points and why each

Points come from the stage 0 c16 call ledger (readout 03 section 6) and the 400 SeedTTS en samples.

| component | point | shape | covers |
|---|---|---|---|
| Flow hop | first, 1 and 16 rows | prompt 125 tokens (p50), window 28 (25 hop + 3 lookahead) | c1 first audio; c16 first hop cohort |
| Flow hop | late, 16 rows | window 378 (offset 275, hop 100) | long history, a third of hops are 100 |
| Flow hop | runaway, 16 rows | 15 rows window 78, 1 row window 2,051 | pad ratio 6.7, the p95 tail (ledger 5.7) |
| Flow final | 1 and 16 rows at 125 tokens, 16 rows at 500 | bidirectional packed call | mean and tail output length |
| Flow buffered | 16 rows, 125 tokens, graph (one captured key) and eager | padded call | graph table value at the formed batch |
| HiFT | delta at history 0, 1,000, 4,000 frames; final at 1,000; batch of 16 x 250 | per request streaming, buffered batch | history recompute growth |
| preprocessing | shortest (3.0 s), median (4.5 s), longest (8.8 s) reference; first call and repeat | load 16 k, load 24 k, CAM++, S3 tokenizer, prompt mel | cold and warm reference cost |
| AR prefill | 1 row at min, p50, max prompt (103, 151, 262 tokens) | eager prefill from embeddings | short, medium, long prompt |
| AR prefill | 16 rows near the median | 2,418 tokens | c16 arrival cohort |
| AR prefill | 32 waiting near the median | one step admits 28 rows, 4,214 tokens | the max_prefill_tokens ceiling |
| AR decode | 1, 16, 32 rows, graph and eager | one step at about 160 tokens | replay against eager, host per step |
| AR decode | 25 steps as one call, 1, 16, 32 rows | one hop, one stream chunk per request | stream emission cost |
| AR decode | 32 rows at 1,161 tokens | one step after 1,000 generated | long context |

Flow and HiFT use random speech tokens of the stated length: their kernels and times depend on
shapes only. Numerics are not measured here (stage 0 E2, E5, E6 did that).

## 6. Serving faithfulness rules

A profile that measures something serving does not do is worse than none. Every rule below was
checked against the call sites of the tree under test before the run.

| rule | how |
|---|---|
| build as serving builds | vocoder: `load_cosyvoice3_flow_hift` + `CosyVoice3Vocoder` (packed estimator); AR: `create_sglang_tts_engine_executor` with the stage factory arguments (bf16, 16 ONNX threads, hop 25), never started |
| device spec as placement passes it | device type and `gpu_id` separately; `cuda:0` is rejected by `resolve_device_spec` |
| one engine per process | `profile_ar.py` runs in its own process; SGLang sizes its KV pool from free memory |
| the AR step is the loop body | `get_next_batch_to_run`, `run_batch`, `process_batch_result`, `last_batch = batch`, as `_event_loop_normal` |
| real ingress | `preprocess_cosyvoice3_payload`, then `process_input_requests` with the stage request builder, stream on |
| grad mode as serving | no outer `inference_mode`: the scheduler thread has none, the model forward carries `no_grad`, the Flow and HiFT entry points carry their own |
| autocast only where serving has it | `hop_batch` and `leftover_batch` own theirs; the buffered point wraps `flow.inference` as `decode_batch` does |
| nothing but the call in the timed window | submits, aborts, drains, cache flushes and shape bookkeeping run in `prepare` or after measure |
| a prefill step finishes no request | requests stay live (min_new_tokens equal to max_new_tokens); the next repeat aborts them by id and steps them out in `prepare` |
| no prefix reuse across repeats | radix cache flushed in `prepare` (`flush_cache`, engine idle) |
| eager decode is the disabled graph path | `decode_cuda_graph_runner` cleared, the state `disable_cuda_graph` leaves; its only other readers are capturers not active here |
| the import path is the tree under test | `PYTHONPATH` holds the worktree, stage0, stage1, the CosyVoice clone and Matcha-TTS; each JSON records the loaded `sglang_omni` file |

## 7. Validity checklist

Run it on every archive before reading a number. The first run's results are in the right column.

| check | how | 2026-09-15 |
|---|---|---|
| tree and import path | `head.txt`, `import_path.txt`, provenance `sglang_omni` | cc85ddaa9, `/sgl-workspace/wt/cosy-main` |
| GPU alone, host load | `gpus_before.csv`, `host_load.txt` | GPU 0 empty; GPUs 2 and 3 at 100 percent, load 9.5 on 128 cores |
| every device event attributed | `device_events_without_launch_in_call` | 0 in all 29 ledgers |
| driver launches captured | `trace_categories` has `cuda_driver` | 1 to 1,811 per call |
| counts match an independent tool | stage 0 V0 and ledger | hop launches 19,072 against 17,259 + 1,811; syncs 11 and 71 against 7 + 4 rows; `pack_flow_inputs` 65 syncs (1 + 4 x 16) |
| one host call range per trace, no kernel before it | raw trace scan | yes, all 29; device tail up to 172 ms after the range, attributed |
| the formed batch is the intended one | `ar.md` step shapes | every point as designed; 32 waiting admitted 28 |
| no hidden work in the timed step | outbox counts; hop steps per call | 0 results, 1,359 stream chunks; 25 replays and 75 syncs per hop |
| repeat spread | wall max / min | Flow 1.00 to 1.02 (final 1 row 1.09); AR decode 1.02 to 1.12 |
| noisy points named | spread above 1.2 | HiFT history 1,000 (1.97), final 1,000 (1.84); AR prefill 16 rows (1.37); hop 25 at 32 rows (1.26); CAM++ longest repeat above first call |

A point that fails a check is excluded from ranking, not corrected by hand.

## 8. Reading beyond the ledger: kernel families

The markdown keeps the top kernels only. For shares, every device event of a trace is bucketed by
name, first match wins, as a share of the summed device time:

| family | pattern |
|---|---|
| attention | `sdpa`, `FlashAttn`, `flash::` |
| mask | `where_` |
| padding scatter | `index_put_kernel` |
| autocast weight cast | `bfloat16_copy_kernel` |
| conv | `xmma_fprop`, `implicit_convolve`, `cutlass__5x_cudnn`, `nchwToNhwc`, `nhwcToNchw`, `conv` |
| fp32 layer norm | `layer_norm_kernel<float` |
| fp32 pointwise | `BinaryFunctor<float`, `CUDAFunctor_add<float>`, `CUDAFunctor_mul<float` |
| copies | `direct_copy_kernel`, `Memcpy`, `CatArray` |
| fill | `FillFunctor` |
| linear | `nvjet`, `cublasLt`, `Kreduce` |
| activation pointwise | `sin_kernel`, `pow_tensor`, `reciprocal`, `act_and_mul`, `gelu`, `silu` |

Conv is matched before linear: cuDNN's `implicit_gemm` names contain `gemm`. Always print the largest
unmatched kernels; on the buffered Flow call they were 16 percent (bf16 add, mul and masked fill on
non contiguous tensors).

## 9. Pitfalls met, and the fix

| pitfall | effect | found | fix |
|---|---|---|---|
| call range matched on `gpu_user_annotation` | empty ledger (wrong thread, device timing) | review | match `user_annotation` only (927d157af) |
| `cuda_driver` launches ignored | cuBLAS and cuDNN kernels missing from busy and launches | review | read both runtime categories (927d157af) |
| first CUDA profiling session drops events | first point undercounted | review | one throwaway session per process (927d157af) |
| outer `inference_mode` around AR steps | host dispatch understated | review against the scheduler thread | removed (927d157af) |
| prefill requests with max_new_tokens 1 | prefill step also paid finish work | review | live requests, abort by id in prepare (841d02ae8) |
| teardown aborted `running_batch` only | a just prefilled request sits in `last_batch` | review | abort every submitted id (841d02ae8) |
| shape bookkeeping inside the timed call | small bias on 2 ms steps | review | attribute reads only; lengths after measure (841d02ae8) |
| `device="cuda:0"` to the stage factory | ValueError at build | box run | type and `gpu_id` separately (ce50caf73) |
| `matcha` not importable | failed at the first prompt mel, after engine boot | box run | CosyVoice and Matcha-TTS on `PYTHONPATH`, imported in the check line (841d02ae8) |
| `sys.path.insert`, function local imports | style, hidden import path | review | `PYTHONPATH` (927d157af) |

## 10. Limits

- Isolated floors. Serving adds GPU sharing between AR, Flow and HiFT and queueing; the census is the
  measure of those.
- Long context decode was measured at 32 rows only (plus 3 prefill lengths at 1 row); the 32 row step
  at 1,161 tokens is 11 percent above the step at 163.
- One reference prompt length (125 tokens) for Flow and HiFT points; Flow cost is linear in frames
  except attention, so other prompts are interpolation.
- Serving weights (share of step time per call type) come from the stage 0 ledger boots, whose times
  are relative only.
- HiFT history 1,000 points and CAM++ on the longest reference are noisy on a shared host; rerun them
  only if a plan depends on their exact cost.

## 11. Files

| file | role |
|---|---|
| `../stage1/trace_ledger.py` | measure, ranges, parse_trace, render_markdown |
| `../stage1/profile_components.py` | Flow, HiFT, preprocessing points |
| `../stage1/profile_ar.py` | AR engine, prefill and decode points |
| `../stage1/README.md` | box commands, return tar |
| `../stage0/common.py` | provenance, vocoder load, SeedTTS streams |
| `../readouts/03_stage0_2eefbc476_20260915.md` | the call ledger the points and weights come from |
