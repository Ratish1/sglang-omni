# 26. E5 readout, control against early ids on one base, 2026-09-12

Doc 28 found the regime: this session's servers ran with
`SGLANG_FORCE_FUSED_OP_BACKEND=torch`. The pair is valid as a pair on that backend and
none of its numbers is a production number.

Archive `qwen3-tts-e5-22951f0ee-20260912.tar.gz`. Every number below was recomputed from
the raw files (speed_results.json, client_timestamps.jsonl, events_pass2/*.jsonl, serve.log,
the Nsight stats CSVs); the box's `readout.md` and `trace_readout.tsv` were read afterwards
and are corrected in section 4.

## 1. Provenance, verified

- Base `645b472cd` on both arms, control unmodified, early ids the same tree plus
  `early_ids.patch` (`status_before.txt` shows only model_runner.py modified, the diff in
  the archive is the patch). `import.txt` resolves each arm to its own worktree. SGLang
  0.5.19, torch 2.13.0+cu130, fa3 attention, pytorch sampling, the default single process
  layout, KV pool 572k tokens: the same configuration as the E4 session.
- GPUs 1 to 3 idle before all four step 1 boots; a tenant at 100 percent on GPU 1 before
  the control Nsight boot only.
- The whole session ran on physical GPU 2 and at half the speed of every earlier session:
  the scheduler's decode log lines (40 steps each) are 1.17 to 1.34 s apart, against 0.60
  to 0.63 s in the E4 session and pair 2, and 0.77 s in pair 1 (a GPU 1 tenant). Generation
  reads 480 to 510 tokens per second at 15 or more rows against 850 to 870. Throughput is
  9.2 to 9.9 req/s against 17.6, inter chunk 180 to 200 ms against 90. Both arms are equally
  affected, so the pair is like for like, but no E5 number compares with an E4 number, and
  the cause (the GPU, its clocks, or the host) is unknown and has to be found before the
  next session.
- Runaways: control pass 1 and both Nsight early passes carry a 163.84 s output. Step 1
  pass 2 on both arms is clean. The early Nsight window closed 34 s before its runaway
  started, so the window itself is unaffected.

## 2. Step 1, the clean pair (pass 2, no profiler, event recorder for 216 to 217 requests)

| client, pass 2 | control | early ids | delta |
| --- | ---: | ---: | ---: |
| req/s | 9.239 | 9.888 | +7.0% |
| TTFC mean / p50 / p99 | 128.9 / 118.9 / 358.6 ms | 144.8 / 133.8 / 334.9 ms | +15.9 ms mean, +14.9 ms p50 |
| inter chunk mean | 197.3 ms | 181.7 ms | -7.9% |
| latency mean / p99 | 1.723 / 2.839 s | 1.609 / 2.664 s | -6.6% / -6.2% |

Segments, p50 ms, admission to the coordinator's first audio chunk:

| segment | control | early ids | delta |
| --- | ---: | ---: | ---: |
| preprocessing dispatch to complete | 27.8 | 32.9 | +5.1 |
| engine input received to request build start | 9.9 | 14.1 | +4.2 |
| request build | 0.5 | 0.5 | |
| build end to queue enter | 23.6 | 19.3 | -4.3 |
| queue enter to prefill start | 1.2 | 10.4 | +9.2 |
| prefill start to first emit | 35.4 | 25.1 | -10.3 |
| vocoder first code received to first audio sent | 20.3 | 30.9 | +10.6 |
| admission to first audio | 127.0 | 144.2 | +17.2 |

Queue table (first audio segment by requests ahead in the initial worker):

| requests ahead | control n / p50 | early ids n / p50 |
| ---: | ---: | ---: |
| 0 | 165 / 19.2 ms | 152 / 23.4 ms |
| 1 | 40 / 26.3 ms | 41 / 45.1 ms |
| 2 | 5 / 44.8 ms | 14 / 52.6 ms |

Code chunks already received when first audio leaves: 1 on control, 2 on early ids.

Reading. `scheduler_prefill_start` is emitted at the top of `_run_batch`
(omni_scheduler.py:1448), before the model executes, and `scheduler_prefill_end` after
`execute` returns (1452). So the two talker segments trade exactly: on early ids a request
sits 9 ms longer in the waiting queue and its prefill returns 10 ms sooner (the prefill's
ids wait no longer covers its predictor), and the first frame is emitted at the same
absolute time on both arms (36.5 against 35.5 ms after queue enter). The longer queue wait
is where in the scheduler's loop the build thread's append lands: it lands when the
scheduler releases the interpreter lock, which on control is the long predictor wait near
the end of the iteration and on early ids the graph launch near its start.

So the absolute regression is in the three places outside the talker's step:

- preprocessing, +5.1 ms;
- the engine's request build executor picking the request up, +4.2 ms;
- the vocoder's first decode, +10.6 ms, of which the publication lead is not absolute: the
  chunk is published before its predictor finishes and the worker's `_wait_codes_ready`
  covers that, up to one predictor. On this regime the predictor is several ms, so a few
  ms of the +4.2 at depth zero are the lead. The rest is the initial worker's own decode
  being slower (section 3) and the deeper queue (median 2 chunks received before first
  audio, depth 1 at 45 against 26 ms).

## 3. Step 2, what the early ids Nsight window shows for its own threads

From `stats_cuda_kern_exec_sum.csv`, per thread, 20 s windows (A is the launch call's
host duration, Q the time from launch to device start, K the kernel's execution). Threads
were identified by their kernel mix: the scheduler launches index and sampling kernels,
the initial worker launches the codec decoder's transposed convolutions (dgrad kernels)
and Snake activations (sin, reciprocal, add), the reference encoder batcher launches the
quantizer's argmin reductions, fp32 GEMMs and fused attention, the preprocessing workers
launch reflection pads and the speaker encoder's bf16 convolutions, the two follow up
workers launch only copies (their warm decodes are graph replays, invisible at kernel
level).

| thread, early ids | launches | A avg | Q avg | K sum |
| --- | ---: | ---: | ---: | ---: |
| talker scheduler | 232k | 5.3 us | 1.25 ms | 628 ms |
| initial worker | 171k | 7.8 us | 1.11 ms | 1480 ms |
| follow up workers (2) | 5.3k each | 95 us | 4.1 ms | 11 ms |
| preprocessing workers (8) | 4 to 6k each | 7 to 17 us | 7 to 26 us | 14 to 21 ms |

The initial worker's kernels wait 1.1 ms on average before they start (medians of 4 to
500 us by kernel, so a heavy tail), and the follow up workers' launch calls take 95 us
each. The interpreter lock trace for the whole process: waiting 11.3 s and holding 7.9 s in
the early window (294 requests admitted) against 6.9 s and 6.9 s in the control window
(231), that is per request 38 against 30 ms waiting and 27 against 30 ms holding. Per
thread lock ownership needs the SQLite export, which the archive does not carry.

## 4. Step 2, why the control window cannot be the comparison

- The control thread the box's readout names as the initial worker (TID 1159764) launches
  argmin reductions (3004), fp32 GEMMs and fused attention kernels: it is the reference
  encoder batcher. That thread never calls a stream wait, which is why the readout found
  zero `cudaStreamWaitEvent` there against 346 on the early initial worker, and why its
  launch to start read 2.4 us against 1.1 ms. The central table of `trace_readout.tsv`
  compares two different threads.
- The control capture contains no transposed convolution kernel at all (dgrad count 0
  against 858) and no Snake decoder kernels outside the encoder thread, while its own event
  file records 227 first audio chunks in the window and the codec counters say every
  bootstrap ran eager. The decoder's kernels are absent from the trace, not from the run.
  The control Nsight session was interrupted during report generation (`protocol.md`) and
  its CUDA activity is incomplete. Symmetrically the early window has no encoder kernels
  (argmin 0 against 3004), so its capture may be incomplete on another stream.
- The control window also ran with a tenant at 100 percent on GPU 1.

So step 2 gave one usable half: the early ids threads' own dispatch and launch costs. It
did not give the control against early ids comparison, and the readout's decision (the
readiness dependency plus late device start) is not established by it. The dependency
exists (doc 25 section 2) and costs at most one predictor per plan.

## 5. Is the optimization misplaced

No. The staged copy sits where it has to (before the predictor, after the sample), the
consumers' ordering is unchanged, and the only new ordering effect a consumer sees is the
publication lead, which `_wait_codes_ready` and the first chunk's own event
(request_builders.py:1639, recorded after the reference cat and before the next step is
enqueued) cover correctly. What the change does to the process is what E4, the census
traces and E5 all show from different sides: the talker's stream stops draining every
step, and every thread that shares the device and the driver with it pays on each kernel:
a longer wait to start, a longer launch call, and a longer wait for the interpreter lock.
Each is small per kernel and the sibling paths run thousands of kernels per request. On the
E5 regime that sums to about 16 ms on the first chunk with a 7 percent throughput gain; on
the E4 regime, where the initial worker's queue was deeper, it was 30 to 100 ms.

## 6. Next

1. Explain the regime before anything else: on the box, `nvidia-smi -q -i 2 -d
   CLOCK,PERFORMANCE,POWER` during a c16 run (throttle reasons, SM clock), `nvidia-smi
   topo -m`, and node 0 load. If GPU 2 or its host side is the cause, the next session runs
   on an idle GPU 0 or 1 and step 1 is repeated there; the E5 pair stays valid as a pair.
2. Redo step 2 for both arms in one session: same command, `nsys start` after the same
   preconditioning, 20 s, then `nsys stats --report diagnostics` to confirm no dropped
   CUDA records, `nsys export --type sqlite`, and archive the `.sqlite` files. Identify the
   initial worker by its dgrad and Snake kernels, never by the readout's TID guess. Read per
   thread: Q and A as in section 3, lock waiting and holding from the NVTX GIL ranges by
   thread, and for the initial worker the Q of the first kernel after each
   `cudaStreamWaitEvent` (the readiness wait) against the rest (device dispatch).
3. Then the mitigation experiments of doc 25, starting with the bounded run ahead: an
   event after `code_predictor_forward` waited at the top of the next `before_decode`, so
   the stream drains once per step. Read against step 1's segments and the queue table on
   the same regime.
