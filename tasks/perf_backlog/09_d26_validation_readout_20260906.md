# Readout of the d26ac7a1e validation archive

Archive `qwen3-tts-d26-validation-20260906-results.tar.gz`. A is `7989a5ed2`, B is `d26ac7a1e`,
the pushed chain head. One H100, GPU 1, checked at 0 MiB before every boot. The whole host was
not idle: 3 to 7 of the other GPUs ran other tenants' jobs at every boot and the load average sat
between 9 and 26. The user relaxed the host gate to GPU 1 alone, so the speed numbers carry that
noise, and two boots per arm and point bound it.

## Verdict

The branch passes every gate the code owns. Suites, probes, byte identity, quality, the
retraction mechanics on three models, and the MOSS-TTS Local fix all hold. Two items stay open
and neither is a property of the branch: the allocator peak of a retraction window is still
unmeasured because the traces carry no allocator events, and the larger graph ladder waits on
the memory provisioning slice.

## Qwen3-TTS, full corpus, two boots per arm

| Point | Metric | A | B |
| --- | --- | ---: | ---: |
| c1 | qps | 2.196 (2.192, 2.200) | 2.266 (2.262, 2.269) |
| c1 | mean, p95, p99 latency s | 0.455, 0.657, 0.765 | 0.441, 0.637, 0.746 |
| c1 | WER, similarity | 1.00477, 71.30515 | 1.00477, 71.30515 |
| c1 | WAV identity | reference | 4352 of 4352 across the four boots and the earlier archive |
| c16 | qps | 14.265 (13.783, 14.747) | 15.092 (14.947, 15.237) |
| c16 | mean, p95, p99 latency s | 1.117, 1.636, 2.066 | 1.054, 1.504, 1.976 |
| c16 | WER | 1.005, 1.105 | 1.038, 1.038 |
| c16 | similarity | 71.281, 71.122 | 71.195, 71.264 |
| c16 | peak GPU MiB | 80343, 80825 | 80767, 80547 |

Every B c16 boot measured so far, 15.611, 15.259, 14.947 and 15.237, sits above every A boot,
15.038, 14.287, 14.747 and 13.783. The replay saving is 0.55 ms of a 9.15 ms step at 16 rows,
6 percent, which is the ceiling of a qps gain from this branch at c16, and the measured deltas
of 3.8, 6.8 and 5.8 percent sit under it. The c1 delta, 3.2 percent with byte identical output
and a 0.4 percent spread, is the number to quote.

The caching allocator's retry warning appears once or twice per c16 boot on both arms, every
request completing. It is the pool provisioning condition of the memory slice, not the branch.

## Qwen3-Omni, fp8 colocated, full corpus, two boots per arm

| Point | qps A | qps B | latency mean A, B | UTMOS A, B | WER percent A, B |
| --- | ---: | ---: | --- | --- | --- |
| c1 | 1.579 (1.565, 1.593) | 1.574 (1.570, 1.577) | 0.633, 0.636 | 4.4721, 4.4728 | 1.83, 2.12 |
| c16 | 7.570 (7.500, 7.640) | 7.521 (7.511, 7.531) | 2.103, 2.119 | 4.4715, 4.4670 | 1.72, 2.03 |
| c32 | 9.887 (9.790, 9.983) | 9.909 (9.835, 9.982) | 3.216, 3.209 | 4.4708, 4.4688 | 1.65, 3.01 |

Speed is equal at every point, which settles the single boot readings of the day before as
noise. No retraction, no exception in any of the twelve logs.

WER is where the archive holds the one observation that needs its own check. Three of B's
6528 outputs ran long: 20.7 s (c1 boot 2), 39.3 s (c16 boot 3) and 80.8 s (c32 boot 3, a
12 word prompt, 300 word errors on its own, which lifts that boot to 4.24 percent). None of
A's 6528 outputs exceeds 8.2 s. The same three prompts run at 3 to 7 s in every other boot of
both arms, so the outputs are stochastic long generations of the talker, well under its 4096
token cap, and the benchmark has no seed flag to replay them. The diff gives the talker no
new input: the helper refactor returns the same rows and pops the same queues (read in full,
`git diff 7989a5ed2..d26ac7a1e -- sglang_omni/models/qwen3_omni/talker_model_runner.py`), and
the compaction never ran because nothing retracted. Three events against zero is a 1 in 8
chance under equal rates. The check that resolves it: the three prompts, 50 requests each with
distinct request seeds, on both arms, counting outputs above 20 s. Recorded as a validation
task, not a finding against the branch.

## Retraction and the other models

| Run | Retractions | Complete | Errors |
| --- | ---: | --- | --- |
| Qwen3-Omni talker, bf16, test switch on the talker stage, A | 4 | 50 of 50 | none, every re prefill at N plus 1 tokens |
| same, B | 5 | 50 of 50 | none, every re prefill at N plus 1 tokens |
| Qwen3-TTS c16 profiled, B | 13 | 192 of 192 | none |
| MOSS-TTS Local c4, B | 3 of 7 hook firings | 16 of 16 | none, the case that raised before `d26ac7a1e` |

Suites on B: qwen3_tts 397 passed, CPU with CUDA hidden 6005 passed, accelerator 292 passed,
both parameterizations of `test_retracted_request_with_model_owned_data_is_requeued` passed.
Probes: 11, 23 and 39 keys with no missing mixed bucket, temperature bits PASS, survivor
amplification 16.

## Open items

- Allocator peak of a retraction window. The fresh trace holds zero `[memory]` events with the
  profiler memory flag exported by the wrapper. The next attempt reads the server process's
  environment (`/proc/<pid>/environ`) to confirm the flag reached the stage process before
  looking further.
- The long generation check above.
- The larger graph ladder and the rope store c16, after the memory provisioning slice.
