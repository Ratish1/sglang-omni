# [Qwen3-TTS] Size the KV pool from the admission bound

## Summary

On an 80 GB H100 the default Qwen3-TTS server allocates a KV pool of 589142 tokens, 62.9 GiB,
because the builder pins `mem_fraction_static` 0.85 and sglang hands the pool everything that
rule leaves free. The scheduler can commit at most `max_running_requests` times the context
length, 16 x 8192 = 131072 tokens, so 4.5 times the committable capacity sits in the pool. The
process also holds the speech tokenizer and the vocoder, whose whole utterance decode allocates
400 to 760 MiB per request, and at c16 the card runs to 80.6 GB with allocator retries in the
log. A 128 running deployment failed 2 of 64 requests in cuDNN at 6 MiB free.

This PR derives `max_total_tokens` from the admission bound in the Qwen3-TTS builder, leaves
`mem_fraction_static` to sglang's own derivation, and logs the pool against the bound at
startup. A deployment's own `max_total_tokens` or `engine.kv_cache_bytes` still wins.

## Changes

- `Qwen3TtsEngineBuilder.adjust_overrides` sets `max_total_tokens` to the merged running cap
  times the builder's context length when the deployment sets neither a token cap nor a stage
  byte budget. sglang applies the cap as a minimum against its profiled capacity, so a bound
  larger than the card leaves the fraction rule in charge, as on main.
- `generation_defaults` no longer carries `mem_fraction_static`. Unset, sglang derives the
  fraction from the card and the activation reserve.
- `post_scheduler_setup` logs `Qwen3-TTS KV pool holds N tokens against an admission bound of
  M (R running x C context)` once the scheduler exists.
- Tests: the cap for 16, 89 and 128 running, a deployment cap surviving, no cap under a stage
  byte budget, no fraction in the defaults, and the startup line at two pool sizes.

## Test results

A is upstream main 50db4a550, B is this branch. One H100, GPU 1, `Qwen/Qwen3-TTS-12Hz-1.7B-Base`,
seed-tts English split of 1088 samples, seed 1234, warmup 0, two fresh boots per arm and point in
A B B A order.

Startup, default config:

| Line | A | B |
| --- | ---: | ---: |
| KV pool tokens | 589142 | 131072 |
| K and V size each | 31.46 GB | 7.00 GB |
| Memory pool end, available | 11.79 GB | 60.70 GB |
| Whole device memory at ready | 75167 MiB | 25043 MiB |

Full corpus, both boots per arm:

| Point | Metric | A | B |
| --- | --- | ---: | ---: |
| c1 | qps | 2.230, 2.219 | 2.220, 2.220 |
| c1 | p99 latency s | 0.751, 0.751 | 0.753, 0.756 |
| c1 | WAV identity | reference | 4352 of 4352 byte identical to A |
| c1 | similarity | 71.30515 | 71.30515 |
| c1 | peak MiB | 76863 | 26773 |
| c16 | qps | 15.033, 14.985 | 14.863, 15.031 |
| c16 | p99 latency s | 1.827, 2.090 | 1.848, 1.839 |
| c16 | WER errors of 11943 words | 128, 114 | 135, 119 |
| c16 | similarity | 71.181, 71.335 | 71.124, 71.343 |
| c16 | peak MiB | 80631, 80567 | 31959, 31493 |
| c16 | allocator retries | 0, 1 | 0, 0 |

c16 output is stochastic per boot on both arms. Eighteen archived c16 boots of this model on
this corpus span 114 to 135 errors and 71.12 to 71.34 similarity, upstream main at both ends.

Kernel census from a profiler window at 1 and at 16 rows: 1371 kernels per talker token on both
arms, identical family counts, predictor wall p50 4.711 against 4.710 ms at c1 and 5.265 against
5.264 ms at c16.

Prefix cache: every c1 log sums to 13576 cached of 74096 prefill tokens on both arms.

128 running with `cuda_graph_max_bs` 128 and the request level subtalker top k 64, 64 requests
at 16 concurrency: 64 of 64 complete, no cuDNN error, no allocator retry. The bound of 1048576
tokens exceeds the card, sglang logs `max_total_tokens=1048576 is larger than the profiled value
579894. Use the profiled value instead.` and the pool is 579894 tokens, the fraction regime of
main.

Forced retraction at c16, `SGLANG_TEST_RETRACT=1` and interval 64, 192 requests: 192 of 192 on
both arms, 14 retractions each, every one re prefilled.

Allocator snapshot at c16 with the allocator history armed for the window: reserved 78014 MiB
on A against 30570 on B, live 69501 against 19376, the pool the only large block that changed.
The largest allocation of the window is the vocoder's whole utterance decode on both arms, 760
MiB on A and 679 MiB on B.

Suites on B: 412 Qwen3-TTS, 6123 CPU with CUDA hidden, 290 accelerator, all exit zero.

## Reproduction

```bash
python -m sglang_omni.cli serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml --host 127.0.0.1 --port 31001
```

Lines that decide it in the serve log:

```text
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 131072, K size: 7.00 GB, V size: 7.00 GB
Memory pool end. avail mem=60.70 GB
Qwen3-TTS KV pool holds 131072 tokens against an admission bound of 131072 (16 running x 8192 context)
```

Benchmark, one concurrency per fresh server:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts --generate-only --use-existing-server \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --meta zhaochenyang20/seed-tts-eval-arrow \
  --ref-format references --lang en --seed 1234 --warmup 0 --port 31001 \
  --max-concurrency 16 --output-dir out/c16
```

Peak memory is the one second `nvidia-smi --query-gpu=memory.used` sample over the run.
