# Readout of the KV pool slice archive

Archive `qwen3-tts-kv-pool-20260906-results.tar.gz`. A is `50db4a550`, upstream main at the
sglang 0.5.18 pin. B is `8807833ca`, the three commits of `perf/qwen3-tts-kv-pool-admission-bound`
on that base. One H100, GPU 1, checked at 0 MiB before every boot, the one second sample of every
GPU kept for every run. The other tenants' GPUs ran between 4 and 99 percent summed utilization
during our windows and the load average sat between 3.6 and 6.2. Two boots per arm and point.

## Verdict

The slice does what it claims and regresses nothing the code owns. The pool drops from 589142
to 131072 tokens, the process at ready drops from 75167 MiB to 25043 MiB, no B boot logs an
allocator retry, c1 audio is byte identical to A and to the d26 archive, the kernel census is
identical, every request of every run completed, and all three suites pass on B.

Throughput is flat within the paired spread. The one B c16 boot that reads low ran during the
busiest host window of the four. Its quality point sits at the edge of the archived range, and
the archived range of upstream main's own boots already contains it.

Three expectations in runbook 10 were wrong, not the code: the graph count at 128 running on
this arm, the strict cached token equality at c16, and the c16 quality band. Runbook 10 carries
the corrections in its head note.

## Startup, default config

| Line | A | B |
| --- | ---: | ---: |
| KV pool tokens | 589142 | 131072 |
| K and V size each | 31.46 GB | 7.00 GB |
| Memory pool end, available | 11.79 GB | 60.70 GB |
| Whole device memory at ready | 75167 MiB | 25043 MiB |

B logs `Qwen3-TTS KV pool holds 131072 tokens against an admission bound of 131072 (16 running
x 8192 context)` after the six predictor graphs. The same lines repeat in every B boot of the
archive.

## Full corpus

| Point | Metric | A boot 1, boot 4 | B boot 2, boot 3 |
| --- | --- | ---: | ---: |
| c1 | qps | 2.230, 2.219 | 2.220, 2.220 |
| c1 | p50, p99 latency s | 0.439 0.751, 0.442 0.751 | 0.442 0.753, 0.441 0.756 |
| c1 | WER percent | 0.99640, 0.99640 | 1.00477, 1.00477 |
| c1 | similarity | 71.30515, 71.30515 | 71.30515, 71.30515 |
| c1 | peak MiB | 76863, 76863 | 26773, 26773 |
| c16 | qps | 15.033, 14.985 | 14.863, 15.031 |
| c16 | p50, p99 latency s | 1.038 1.827, 1.036 2.090 | 1.040 1.848, 1.030 1.839 |
| c16 | WER errors of 11943 words | 128, 114 | 135, 119 |
| c16 | similarity | 71.181, 71.335 | 71.124, 71.343 |
| c16 | output tokens | 56274, 56285 | 56196, 56178 |
| c16 | peak MiB | 80631, 80567 | 31959, 31493 |
| c16 | allocator retries | 0, 1 | 0, 0 |

Paired deltas by boot position, B minus A: c1 boots 1 and 2 minus 0.45 percent, boots 4 and 3
plus 0.05 percent. c16 boots 1 and 2 minus 1.13 percent, boots 4 and 3 plus 0.31 percent. The
sign flips between the pairs of each point. The A arm's own boots differ by 0.5 percent at c1
and 0.3 percent at c16.

Other GPUs' summed utilization during each benchmark window, from the one second sample:

| Boot | c1 | c16 |
| --- | ---: | ---: |
| 1 A | 98.7 | 50.0 |
| 2 B | 21.8 | 82.4 |
| 3 B | 25.9 | 70.1 |
| 4 A | 41.6 | 4.4 |

The slow c16 boot, B boot 2, ran while GPUs 0 and 6 both sat above 80 percent. GPU 1 itself
averaged 65 to 66 percent in every c16 boot of both arms.

### What "regressed compared to before" compares

The d26 archive's B arm was the predictor chain, not this branch. Its c16 mean of 15.09 was the
chain's gain over main on that day, and main itself read 14.27 that day and 15.01 today, a 5
percent swing on unchanged code between runs on this host. The comparison this slice owns is B
against A within one run, and that reads flat at both points.

### Where the c1 audio and WER disagree

All 4352 c1 WAVs match the d26 hashes, both arms. The c1 WER still differs by one substitution
on one sample, `common_voice_en_31959919-common_voice_en_31959927`, transcribed differently by
the ASR scorer on identical audio. Similarity is 71.30515434 on all four boots to the last digit.

### The c16 quality range

c16 output is stochastic per boot on both arms, the sampler's draws follow the batch
composition, so each boot is one sample of a distribution. Every archived c16 boot of Qwen3-TTS
on this corpus:

| Archive | Arm and errors of 11943 | Similarity |
| --- | --- | --- |
| chain ab, Sep 5 | A 114, B 117 | not scored |
| newbase, Sep 5 | A 128, B 119 | 71.323, 71.201 |
| s2, Sep 5 | A 135, B 118 | 71.178, 71.313 |
| revalidation, Sep 6 | A 132, B 119 | 71.183, 71.203 |
| d26, Sep 6 | A 120, 132 and B 124, 124 | 71.281, 71.122 and 71.195, 71.264 |
| kv pool, Sep 6 | A 128, 114 and B 135, 119 | 71.181, 71.335 and 71.124, 71.343 |

Eighteen boots span 114 to 135 errors and 71.12 to 71.34 similarity. Upstream main arms alone
already reached 135 errors on Sep 5 and 71.122 on Sep 6. The B boot 2 point of this run, 135 and
71.124, is inside that. The band of 116 to 128 in runbook 10 was written from a subset and is
withdrawn. In this run no sample has three or more errors in B boot 2 above every other boot,
the extra errors are single word differences spread over samples, 107 samples with any error
against 102, 98 and 94. The paired sample bootstrap of both c16 pairs spans zero on WER and on
similarity.

The gate that the pool size can fail is c1 byte identity plus the kernel census. c16 quality is
reported against the arm's archived range, not gated on a fixed band.

### Prefix cache

Every c1 log sums to 13576 cached and 60520 new tokens of 74096, both arms, so the cap keeps
every prefix hit the corpus has. At c16 the cached totals are 10009, 10059, 9896 and 9911 in
boot order. The two A boots differ from each other by 98 tokens, so the count depends on prefill
timing at c16, whether a sibling's prefix was inserted before the next prefill, and is not a
property of the pool size. The equality gate holds at c1 only.

## Kernel census

Both arms replay 1371 kernels per talker token at 1 and at 16 rows, the same nine families with
the same counts. Predictor wall p50 4.711 against 4.710 ms at c1 and 5.265 against 5.264 ms at
c16. Step wall p50 8.046 against 8.071 ms at c1 and 9.275 against 9.201 ms at c16. The pool size
touches no kernel.

## Allocator retries on A

A logged one recovered allocator failure in c16 boot 4, one in the census boot, one in the
snapshot boot and two in the retraction boot, each a vocoder or attention allocation of 90 to
734 MB requested with 84 to 438 MB free. The c16 boot 4 retry came six seconds after ready, at
`18:45:18`, and that boot carries the run's only p99 above 2 s, 2.090 against 1.827 on the other
A boot. The retry frees the allocator's cache and re-requests the block, which is synchronous.
The coincidence is recorded, the causal link is not measured. B logged none in any boot.

## 128 running with the request level subtalker top k 64

64 of 64 completed, six lazy captures for key `(bucket, 'sampled', 64, False, False)`, no cuDNN
error, no allocator retry. Startup captured 20 predictor graphs in 9.7 s, the single signature
startup set of main at `cuda_graph_max_bs` 128. The runbook's 39 was the chain branch's two
signature set and does not apply to this arm.

The cap did not bind. The bound is 128 times 8192, 1048576 tokens, sglang logged
`max_total_tokens=1048576 is larger than the profiled value 579894. Use the profiled value
instead.`, and the pool went to 579894 tokens under the fraction rule. Ready memory 74325 MiB,
peak 79459 MiB of 81559. B's own line reports it: `KV pool holds 579894 tokens against an
admission bound of 1048576`. At a running cap whose bound exceeds the card, this branch is the
fraction regime again and the process runs with about 2 GB of headroom. That regime is not
changed by this slice and stays open, see the last section.

## Retraction at c16, both arms

192 of 192 on both, 14 forced retractions each, no failure. On A every retraction is followed by
an isolated single sequence prefill with the retracted request's tokens plus one. On B, 12 are
isolated and two were re prefilled inside a coalesced batch with new arrivals, at `19:07:58` a
four sequence batch and at `19:08:04` a three sequence batch, which is prefill coalescing doing
its job.

The profiler memory flag reached the stage process, `stage_process_env.json` reads it from
`/proc/<pid>/environ`, and both traces still hold zero `[memory]` events. The torch profiler's
own note explains it: "profiler is thread local and is automatically propagated into the async
tasks" (`torch/autograd/profiler.py:121`, torch 2.12 locally, 2.13 on the box). The profile is
started on the stage's control thread, so allocations on the scheduler and vocoder threads are
never observed by it. The allocator history hook of the profiling branch is process wide and is
the tool that works, see the next section. Whole device peaks during the retraction window:
80939 MiB on A, 33377 MiB on B, against 32797 and 32109 in the two B windows without retraction.

## Allocator snapshot at c16

The profiling branch's hook alone was applied on both arms, `SGLANG_TORCH_PROFILER_MEMORY_SNAPSHOT=1`
confirmed in the process, one c16 window of 192 requests, snapshot at profiler stop.

| At profiler stop | A MiB | B MiB |
| --- | ---: | ---: |
| Reserved | 78014 | 30570 |
| Live | 69501 | 19376 |
| Cached, reserved minus live | 8513 | 11194 |
| Live blocks | 2754 | 2754 |
| Live with an allocation frame | 393 | 386 |
| Live without a frame, allocated before the window | 69108 | 18990 |
| Largest allocation in the window | 759.6 | 679.4 |
| Cumulative segment growth in the window | 7458 | 6854 |
| Out of memory events in the history | 0 | 0 |

History starts when the profile starts, after startup, so startup blocks carry no frames. Their
sizes still identify them, as arithmetic on the block sizes, not as call sites:

| Block group | A | B | Arithmetic |
| --- | --- | --- | --- |
| 56 blocks | 1150.67 MiB each, 64437 MiB | 256.00 MiB each, 14336 MiB | tokens times 2048 B, 28 layers times K and V |
| 1 block | 594 MiB | 594 MiB | 152064 times 2048 times 2 B, the text embedding |
| 28 and 28 blocks | 48 and 24 MiB | same | gate up and down projections, 28 layers |
| 30 and 59 blocks | 16 and 8 MiB | same | qkv and output projections |
| everything else | 1109 MiB | 1092 MiB | small weights and buffers |

So the non pool startup residue is 4671 MiB on A and 4654 MiB on B, and the pool is the only
large block that changed. Serving allocations with frames are under 400 MiB live at any moment
on both arms: 256 MiB under `talker.py:189 forward`, 64 MiB under `request_builders.py:840
_run`, 32 MiB each under the thinker prepare and the vocoder, a few MiB of decode history under
`omni_scheduler.py:1380 process_batch_result`.

The transient that drives the cache is the vocoder's whole utterance decode,
`streaming_vocoder.py:1884 _vocode_payloads`, the thirty largest allocations of both histories
are its, 390 to 760 MiB each, on stream 0. Correction after the Sep 7 review: the non streaming
loop runs one `_vocode_payloads` call at a time (streaming_simple_scheduler.py:383-430), so the
three allocations one millisecond apart are tensors of one call, not three calls. The three
streams at 1576 MiB are the vocoder's decode graph pools, proven by pool id from the pickle:
private pools (0, 3), (0, 4) and (0, 5), one per capture stream, 1544 MiB reserved and 12 MiB
live each on both arms, plus a 32 MiB default pool segment per stream for the static buffers.
The preprocessing thread's `_run` stream holds 650 MiB on A and 852 MiB on B. That is
the composition plan 05 section 1 asked for, at the resolution the window allows: pool, weights,
and a vocoder transient of about 2.3 GB across three streams, plus the allocator's cache of it.

Two things the snapshot cannot say. The blocks allocated before the window cannot be named by
call site, a history hook armed before model load would name them. And the difference between
whole device memory at ready, 25043 MiB on B, and the allocator's reserved bytes is outside the
allocator, CUDA context, graph executables and library workspaces, and was not measured.

## Suites on B

412 Qwen3-TTS, 6123 CPU with CUDA hidden and 6 skipped, 290 accelerator and 8 skipped, all three
exit zero. The eight parameter sets of the five new tests pass.

## Runbook 10 corrections

1. Section 4 expected 39 graphs at 128 running. That is the chain's two signature startup. On
   main plus this slice the single signature startup captures 20.
2. Section 3 gated c16 on equal cached token counts between arms. The count moves with prefill
   timing at c16, the two A boots differ from each other. The equality gate is c1 only.
3. Section 3 gated c16 quality on 116 to 128 errors and 71.18 to 71.32 similarity. Eighteen
   archived boots span 114 to 135 and 71.12 to 71.34, upstream main included. c16 quality is
   reported against that range, the gate is c1 identity plus the census.

## What stays open after this slice

- The large running cap regime. When `max_running_requests` times the context exceeds the card,
  the cap does not bind and the fraction rule fills the card again, with about 2 GB of headroom
  at 128 running. Sizing the pool from what the other components in the process need, the
  construction order accounting of plan 05 decision 4, or a deployment budget, closes that.
- Naming the startup blocks by call site needs the history armed before model load, a change to
  the profiling branch's hook.
- The rope store branch, rebased on the chain head plus this slice, for its c16 pair.
